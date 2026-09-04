"""Static ordering contracts for the privileged Linux installer."""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from eve import doctor

REPO_ROOT = Path(__file__).resolve().parent.parent
INSTALL = REPO_ROOT / "deploy" / "install.sh"
PROVISION = REPO_ROOT / "scripts" / "provision-whisper.sh"
FETCH = REPO_ROOT / "scripts" / "fetch-models.sh"


def _installer() -> str:
    return INSTALL.read_text()


def test_supported_platform_is_gated_before_environment_sync():
    body = _installer()
    sync = body.index('sync --locked --no-dev --python "$PYTHON_BIN"')
    for requirement in (
        '"$(uname -s)" != "Linux"',
        "x86_64|aarch64",
        "GNU_LIBC_VERSION",
        "GLIBC_MINOR",
        "CPython:3.11.*|CPython:3.12.*|CPython:3.13.*",
        '"$SYSTEMD_VERSION" -lt 243',
    ):
        assert body.index(requirement) < sync


def test_uv_may_provision_python_when_the_system_has_no_compatible_one():
    body = _installer()
    assert 'python find \'>=3.11,<3.14\'' in body
    assert "--no-python-downloads" not in body
    assert body.index("python find") < body.index("systemctl stop")


def test_systemd_paths_are_rejected_before_python_or_environment_mutation():
    body = _installer()
    validation = body.index('validate_systemd_path "config directory"')
    assert validation < body.index("python find")
    assert validation < body.index("sync --locked")
    assert validation < body.index('ensure_private_directory "$CONFIG_DIR"')


def test_installer_runtime_resolution_matches_a_fresh_service_environment(
    tmp_path: Path,
):
    file_mac = "02:00:00" + ":00:00:01"
    inherited_mac = "FF:FF:FF" + ":FF:FF:FF"
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "env").write_text(
        "EVE_MODELS_DIR=models-from-file\n"
        "KOKORO_DIR=kokoro-from-file\n"
        "WHISPER_DIR=whisper-from-file\n"
        f"VOICE_SPEAKER_MAC={file_mac}\n"
    )
    body = _installer()
    scrub = body.split(
        "# BEGIN TESTABLE RUNTIME ENVIRONMENT SCRUB\n", 1
    )[1].split("# END TESTABLE RUNTIME ENVIRONMENT SCRUB\n", 1)[0]
    probe = (
        "import json; from eve import config; "
        "print(json.dumps({"
        "'models': str(config.MODELS_DIR), "
        "'kokoro': str(config.kokoro_dir()), "
        "'whisper': str(config.whisper_dir()), "
        "'speaker': config.setting('VOICE_SPEAKER_MAC')"
        "}, sort_keys=True))"
    )
    clean_env = {
        key: value
        for key, value in os.environ.items()
        if not (
            key == "ANTHROPIC_API_KEY"
            or key.startswith(
                (
                    "VOICE_",
                    "KOKORO_",
                    "WHISPER_",
                    "EVE_",
                    "HEY_CLAUDE_",
                    "BARKEEP_",
                )
            )
        )
    }
    clean_env["EVE_CONFIG_DIR"] = str(config_dir)
    service = subprocess.run(
        [os.sys.executable, "-c", probe],
        cwd=REPO_ROOT,
        env=clean_env,
        check=True,
        capture_output=True,
        text=True,
    )

    dirty_env = {
        **clean_env,
        "EVE_MODELS_DIR": "/wrong/models",
        "KOKORO_DIR": "/wrong/kokoro",
        "WHISPER_DIR": "/wrong/whisper",
        "VOICE_SPEAKER_MAC": inherited_mac,
        "BARKEEP_URL": "https://wrong.invalid",
    }
    installer = subprocess.run(
        [
            "bash",
            "-c",
            scrub
            + "\nscrub_inherited_runtime_environment\n"
            + f'exec {os.sys.executable!s} -c "$PROBE"',
        ],
        cwd=REPO_ROOT,
        env={**dirty_env, "PROBE": probe},
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(installer.stdout) == json.loads(service.stdout)


def test_installer_pins_uv_to_the_checkout_environment():
    body = _installer()
    assert 'export UV_PROJECT="$REPO_DIR"' in body
    assert 'export UV_PROJECT_ENVIRONMENT="$REPO_DIR/.venv"' in body
    scrub = body.split(
        "# BEGIN TESTABLE RUNTIME ENVIRONMENT SCRUB\n", 1
    )[1].split("# END TESTABLE RUNTIME ENVIRONMENT SCRUB\n", 1)[0]
    for variable in (
        "UV_PROJECT", "UV_PROJECT_ENVIRONMENT", "UV_WORKING_DIR",
        "UV_CONFIG_FILE", "UV_ENV_FILE", "UV_NO_PROJECT",
    ):
        assert variable in scrub


def test_model_provisioners_pin_uv_to_the_same_checkout_environment():
    for script in (PROVISION, FETCH):
        body = script.read_text()
        assert 'export UV_PROJECT="$REPO_DIR"' in body
        assert 'export UV_PROJECT_ENVIRONMENT="$REPO_DIR/.venv"' in body


def test_an_active_service_is_stopped_before_venv_mutation_and_stays_closed_on_error():
    body = _installer()
    stop = body.index('sudo systemctl stop "$SERVICE_UNIT"')
    sync = body.index("sync --locked")
    assert stop < sync
    failure_handler = body[body.index("installer_exit()") : body.index("trap installer_exit")]
    assert "remains stopped" in failure_handler
    assert 'sudo systemctl start "$SERVICE_UNIT"' not in failure_handler


def test_exit_handler_stops_a_started_unit_before_claiming_fail_closed(
    tmp_path: Path,
):
    body = _installer()
    handler_start = body.index("installer_exit() {")
    handler_end = body.index("\n}\ntrap installer_exit EXIT", handler_start) + 3
    handler = body[handler_start:handler_end]
    trace = tmp_path / "sudo.trace"
    harness = (
        "say() { printf '%s\\n' \"$*\"; }\n"
        "cleanup_unit_tmp() { :; }\n"
        "sudo() { printf '%s\\n' \"$*\" >> \"$TRACE\"; }\n"
        "SERVICE_UNIT=eve@test\n"
        "SERVICE_STOPPED=0\n"
        "SERVICE_START_ATTEMPTED=1\n"
        "SERVICE_ENABLE_ATTEMPTED=0\n"
        "SERVICE_WAS_ENABLED=1\n"
        + handler
        + "\nfalse\ninstaller_exit\n"
    )
    result = subprocess.run(
        ["bash", "-c", harness],
        env={**os.environ, "TRACE": str(trace)},
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert trace.read_text() == "systemctl stop eve@test\n"
    assert "eve@test remains stopped" in result.stdout


def test_failed_fresh_start_is_removed_from_boot_startup(tmp_path: Path):
    body = _installer()
    handler_start = body.index("installer_exit() {")
    handler_end = body.index("\n}\ntrap installer_exit EXIT", handler_start) + 3
    handler = body[handler_start:handler_end]
    trace = tmp_path / "sudo.trace"
    harness = (
        "say() { printf '%s\\n' \"$*\"; }\n"
        "cleanup_unit_tmp() { :; }\n"
        "sudo() { printf '%s\\n' \"$*\" >> \"$TRACE\"; }\n"
        "SERVICE_UNIT=eve@test\n"
        "SERVICE_STOPPED=0\n"
        "SERVICE_START_ATTEMPTED=1\n"
        "SERVICE_ENABLE_ATTEMPTED=1\n"
        "SERVICE_WAS_ENABLED=0\n"
        + handler
        + "\nfalse\ninstaller_exit\n"
    )
    result = subprocess.run(
        ["bash", "-c", harness],
        env={**os.environ, "TRACE": str(trace)},
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert trace.read_text().splitlines() == [
        "systemctl stop eve@test",
        "systemctl disable eve@test",
    ]


def test_failed_refresh_keeps_an_existing_boot_enablement(tmp_path: Path):
    body = _installer()
    handler_start = body.index("installer_exit() {")
    handler_end = body.index("\n}\ntrap installer_exit EXIT", handler_start) + 3
    handler = body[handler_start:handler_end]
    trace = tmp_path / "sudo.trace"
    harness = (
        "say() { :; }\n"
        "cleanup_unit_tmp() { :; }\n"
        "sudo() { printf '%s\\n' \"$*\" >> \"$TRACE\"; }\n"
        "SERVICE_UNIT=eve@test\n"
        "SERVICE_STOPPED=1\n"
        "SERVICE_START_ATTEMPTED=1\n"
        "SERVICE_ENABLE_ATTEMPTED=0\n"
        "SERVICE_WAS_ENABLED=1\n"
        + handler
        + "\nfalse\ninstaller_exit\n"
    )
    result = subprocess.run(
        ["bash", "-c", harness],
        env={**os.environ, "TRACE": str(trace)},
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert trace.read_text().splitlines() == ["systemctl stop eve@test"]


def test_every_start_is_marked_unverified_until_the_settle_loop_finishes():
    body = _installer()
    attempted = body.rindex("SERVICE_START_ATTEMPTED=1")
    start = body.rindex('sudo systemctl start "$SERVICE_UNIT"')
    settle = body.index("for settle_second in 1 2 3 4 5")
    confirmed = body.rindex("SERVICE_START_ATTEMPTED=0")
    assert attempted < start < settle < confirmed


def test_voice_runtime_and_all_selected_units_pass_before_any_etc_write():
    body = _installer()
    runtime = body.index('run --no-sync eve doctor')
    main_verify = body.index('systemd-analyze verify "$UNIT_TMP"')
    speaker_verify = body.index(
        'systemd-analyze verify "$SPEAKER_TMP" "$SPEAKER_TIMER_TMP"'
    )
    first_etc_write = body.index("sudo install -m 0644")
    assert runtime < main_verify < speaker_verify < first_etc_write


def test_host_wide_journal_policy_is_explicit_and_defaults_off():
    body = _installer()
    prompt = 'ask_yes_no "Apply that host-wide limit? (y/n)" "n"'
    assert prompt in body
    assert "every service on this machine, not only Eve" in body
    journal_write = body.index("deploy/journald-retention.conf")
    speaker_verify = body.index(
        'systemd-analyze verify "$SPEAKER_TMP" "$SPEAKER_TIMER_TMP"'
    )
    assert speaker_verify < journal_write


def test_only_existing_device_groups_are_rendered_and_input_is_considered():
    body = _installer()
    assert "for group in audio video tty bluetooth input" in body
    assert 'group_exists "$group" && DEVICE_GROUPS+=' in body
    assert '--supplementary-groups "${DEVICE_GROUPS[@]}"' in body


def test_speaker_timer_requires_a_valid_mac_and_bluetoothctl():
    body = _installer()
    assert r'^([[:xdigit:]]{2}:){5}[[:xdigit:]]{2}$' in body
    guard = body.index('! command -v bluetoothctl >/dev/null')
    selection = body.index('if [ -n "$CONFIGURED_SPEAKER" ]')
    assert guard < selection


def test_startup_must_remain_active_during_a_bounded_settle_period():
    body = _installer()
    start = body.rindex('sudo systemctl start "$SERVICE_UNIT"')
    settle = body.index("for settle_second in 1 2 3 4 5")
    success = body.index('say "Running. Watch her')
    assert start < settle < success
    assert 'systemctl is-active --quiet "$SERVICE_UNIT"' in body[settle:success]


def test_microphone_discovery_survives_arecord_failure_and_keeps_device_number():
    body = _installer()
    assert "arecord -l 2>/dev/null || true" in body
    assert r"device \([0-9]*\)" in body
    assert r"DEV=\2" in body
    assert "| head -1" not in body


def test_provisioners_use_compatible_curl_flags_and_the_correct_build_dir():
    provision = PROVISION.read_text()
    fetch = FETCH.read_text()
    assert '--retry-all-errors' not in provision + fetch
    assert "--retry 3 --retry-delay 2" in provision
    assert "--retry 3 --retry-delay 2" in fetch
    assert 'cmake -S "$WHISPER_ROOT" -B "$WHISPER_ROOT/build"' in provision
    assert "--target whisper-cli" in provision


def test_every_runtime_pin_is_single_sourced_from_the_doctor_manifest():
    provision = PROVISION.read_text()
    fetch = FETCH.read_text()
    assert doctor.WHISPER_SOURCE_COMMIT \
        == "592feef04a1802b18cbeffd0fd0eb5d02570c2ec"
    for name in (
        "whisper-source-commit",
        "whisper-model-sha256",
        "kokoro-model-sha256",
        "kokoro-voices-sha256",
        "silero-model-sha256",
    ):
        assert name in provision + fetch


def test_framebuffer_hosts_receive_exact_chvt_visudo_guidance_only():
    body = _installer()
    framebuffer = body.index("if [ -e /dev/fb0 ]")
    rule = body.index("NOPASSWD: $CHVT_BIN 8")
    assert framebuffer < rule
    assert "sudo visudo -f /etc/sudoers.d/eve" in body
