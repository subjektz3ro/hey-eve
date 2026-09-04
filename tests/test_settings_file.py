"""The settings file, and the difference between a setting and a credential.

The file is parsed as data. Documented non-secret settings take effect without
sourcing it, while credentials remain available on demand without being
exported to child processes. These tests keep the installer, example file, and
runtime readers on that shared contract.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from eve import config

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def env_file(isolated_settings, monkeypatch):
    """Write an env file and point config at it, without touching os.environ."""

    def write(body: str) -> Path:
        path = isolated_settings / "env"
        path.write_text(body)
        monkeypatch.setattr(config, "VOICE_ENV", path)
        return path

    return write


class TestSettingsTakeEffect:
    def test_an_ordinary_setting_is_applied(self, env_file, monkeypatch):
        monkeypatch.delenv("VOICE_MIC", raising=False)
        env_file("VOICE_MIC=plughw:CARD=Other,DEV=0\n")
        assert "VOICE_MIC" in config.load_settings()
        assert os.environ["VOICE_MIC"] == "plughw:CARD=Other,DEV=0"

    def test_the_names_applied_are_returned(self, env_file, monkeypatch):
        for name in ("VOICE_MIC", "VOICE_PERSONA"):
            monkeypatch.delenv(name, raising=False)
        env_file("VOICE_MIC=hw:1\nVOICE_PERSONA=plain\n")
        assert set(config.load_settings()) == {"VOICE_MIC", "VOICE_PERSONA"}

    def test_a_real_environment_variable_still_wins(self, env_file, monkeypatch):
        # env.example promises this for direct runs, which is what makes
        # `VOICE_DEBUG=1 uv run python -m eve.main` work. The managed installer
        # separately scrubs inherited runtime settings before validation.
        monkeypatch.setenv("VOICE_MIC", "from-the-environment")
        env_file("VOICE_MIC=from-the-file\n")
        assert "VOICE_MIC" not in config.load_settings()
        assert os.environ["VOICE_MIC"] == "from-the-environment"

    def test_every_key_install_sh_writes_is_one_the_code_reads(self):
        # The drift this whole file exists to stop: the installer interviews
        # the user, writes the answers, and something silently ignores them.
        installer = (REPO_ROOT / "deploy" / "install.sh").read_text()
        written = {
            line.split()[1]
            for line in installer.splitlines()
            if line.strip().startswith("write_env_line ")
        }
        assert written, "the installer stopped writing settings; update this test"
        read = _names_the_code_reads()
        assert written <= read, f"install.sh writes keys nothing reads: {written - read}"

    def test_every_key_env_example_documents_is_one_the_code_reads(self):
        documented = {
            line.split("=")[0].lstrip("# ").strip()
            for line in (REPO_ROOT / "env.example").read_text().splitlines()
            if "=" in line and line.lstrip("# ")[:1].isupper()
        }
        documented = {name for name in documented if name.isupper() and name}
        unread = documented - _names_the_code_reads()
        assert not unread, f"env.example documents keys nothing reads: {unread}"


def _names_the_code_reads() -> set[str]:
    """Every setting name any module actually looks up."""
    import re

    names: set[str] = set()
    for path in sorted((REPO_ROOT / "eve").glob("*.py")):
        body = path.read_text()
        names |= set(re.findall(r'os\.environ(?:\.get)?\(\s*"([A-Z_]+)"', body))
        names |= set(re.findall(r'(?:secret|setting)\(\s*"([A-Z_]+)"', body))
        names |= set(re.findall(r',\s*"([A-Z_]+)"\s*\)', body))
    # Read by deploy/connect-speaker.sh rather than by Python.
    names.add("VOICE_SPEAKER_MAC")
    return names


class TestCredentialsAreNotSettings:
    @pytest.mark.parametrize("name", sorted(config.SECRETS))
    def test_a_credential_is_never_exported(self, name, env_file, monkeypatch):
        # The entire reason this file is parsed rather than sourced. An
        # exported key is inherited by the aplay and arecord processes she
        # spawns on almost every turn.
        monkeypatch.delenv(name, raising=False)
        env_file(f"{name}=sk-must-not-leak\n")
        assert name not in config.load_settings()
        assert name not in os.environ

    def test_but_it_is_still_readable_on_demand(self, env_file, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        env_file("ANTHROPIC_API_KEY=sk-readable\n")
        config.load_settings()
        assert config.secret("ANTHROPIC_API_KEY") == "sk-readable"

    def test_a_credential_beside_a_setting_does_not_block_it(self, env_file, monkeypatch):
        monkeypatch.delenv("VOICE_MIC", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        env_file("ANTHROPIC_API_KEY=sk-x\nVOICE_MIC=hw:2\n")
        assert config.load_settings() == ["VOICE_MIC"]

    def test_the_unit_has_no_environment_file_directive(self):
        # Adding one would fix the configuration bug by exporting the key,
        # which is the thing config.secret exists to prevent.
        unit = (REPO_ROOT / "deploy" / "eve@.service").read_text()
        assert "EnvironmentFile" not in unit


class TestParsing:
    def test_comments_and_blank_lines_are_skipped(self, env_file, monkeypatch):
        monkeypatch.delenv("VOICE_MIC", raising=False)
        env_file("# a comment\n\n   \nVOICE_MIC=hw:3\n")
        assert config.load_settings() == ["VOICE_MIC"]

    def test_a_value_containing_an_equals_sign_survives(self, env_file, monkeypatch):
        monkeypatch.delenv("VOICE_MIC", raising=False)
        env_file("VOICE_MIC=plughw:CARD=MV6,DEV=0\n")
        config.load_settings()
        assert os.environ["VOICE_MIC"] == "plughw:CARD=MV6,DEV=0"

    def test_quotes_are_stripped_the_way_secret_strips_them(self, env_file, monkeypatch):
        monkeypatch.delenv("VOICE_PERSONA", raising=False)
        env_file('VOICE_PERSONA="plain"\n')
        config.load_settings()
        assert os.environ["VOICE_PERSONA"] == "plain"

    def test_a_line_with_no_equals_sign_is_ignored(self, env_file):
        env_file("this is not a setting\n")
        assert config.load_settings() == []

    def test_a_missing_file_is_not_an_error(self, isolated_settings, monkeypatch):
        monkeypatch.setattr(config, "VOICE_ENV", isolated_settings / "absent")
        assert config.load_settings() == []

    def test_it_is_never_sourced(self, env_file, tmp_path, monkeypatch):
        # A key may contain any character at all precisely because this file
        # is data. If it were ever sourced, this would create the file.
        monkeypatch.delenv("VOICE_MIC", raising=False)
        marker = tmp_path / "pwned"
        env_file(f'VOICE_MIC=$(touch {marker})\n')
        config.load_settings()
        assert not marker.exists()


class TestTheSuiteStaysIsolated:
    """Importing eve now reads a file. That must never be somebody's real one.

    conftest points EVE_CONFIG_DIR at a path that cannot exist before it
    imports eve, so the loader is a no-op under test. Without that line, a run
    on any machine that has ever run this assistant would take that person's
    microphone, model and debug flag into the suite.
    """

    def test_the_config_directory_is_redirected_before_import(self):
        # Ordering, not values: the autouse fixture re-points EVE_CONFIG_DIR
        # at a real temporary directory once a test is running, so the only
        # place the guarantee is visible is conftest's own source. The line
        # has to come before the eve import or it is decoration.
        source = (REPO_ROOT / "tests" / "conftest.py").read_text()
        redirect = source.index('os.environ["EVE_CONFIG_DIR"]')
        imported = source.index("from eve import")
        assert redirect < imported, \
            "conftest must redirect the settings directory before importing eve"

    def test_a_setting_reaches_the_constant_that_uses_it(self, tmp_path):
        # The end the whole change exists for, and the half the first attempt
        # got wrong: config.py's own constants (VOICE_MIC, VOICE_PERSONA,
        # VOICE_MODEL) are computed while config.py is being imported, so a
        # loader called from anywhere else is too late for them. log.DEBUG is
        # here too because log.py does not import config at all — it only
        # works because eve/__init__ imports config before anything else.
        directory = tmp_path / "eve"
        directory.mkdir()
        (directory / "env").write_text(
            "ANTHROPIC_API_KEY=sk-must-not-leak\n"
            "VOICE_MIC=plughw:CARD=Elsewhere,DEV=0\n"
            "VOICE_PERSONA=plain\n"
            "VOICE_MODEL=claude-sonnet-5\n"
            "VOICE_HANG_S=1.25\n"
            "VOICE_DEBUG=1\n"
        )
        probe = (
            "import os, json\n"
            "from eve import config, speech, log\n"
            "print(json.dumps({\n"
            "  'mic': config.MIC_DEVICE, 'persona': config.PERSONA,\n"
            "  'model': config.MODEL, 'hang': speech._HANG_S,\n"
            "  'debug': log.DEBUG,\n"
            "  'key_exported': 'ANTHROPIC_API_KEY' in os.environ,\n"
            "  'key_readable': bool(config.secret('ANTHROPIC_API_KEY')),\n"
            "}))\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", probe],
            capture_output=True, text=True, timeout=120, cwd=REPO_ROOT,
            env={k: v for k, v in os.environ.items()
                 if not k.startswith("VOICE_")} | {
                "EVE_CONFIG_DIR": str(directory)},
        )
        assert result.returncode == 0, result.stderr
        got = json.loads(result.stdout)
        assert got["mic"] == "plughw:CARD=Elsewhere,DEV=0"
        assert got["persona"] == "plain"
        assert got["model"] == "claude-sonnet-5"
        assert got["hang"] == 1.25
        assert got["debug"] is True
        # And the invariant that must survive all of it.
        assert got["key_exported"] is False
        assert got["key_readable"] is True

    def test_importing_eve_reads_no_real_settings(self):
        # A subprocess, because the import has already happened in this one.
        # HOME is redirected too, so even a missing EVE_CONFIG_DIR could not
        # reach a real file.
        result = subprocess.run(
            [sys.executable, "-c",
             "import eve, json; print(json.dumps(eve.SETTINGS_APPLIED))"],
            capture_output=True, text=True, timeout=60, cwd=REPO_ROOT,
            env={**os.environ,
                 "EVE_CONFIG_DIR": "/nonexistent/eve-config-for-tests",
                 "HOME": "/nonexistent/home-for-tests"},
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "[]"


class TestRuntimePaths:
    def test_relative_paths_are_checkout_relative_not_caller_relative(
            self, tmp_path):
        elsewhere = tmp_path / "caller-cwd"
        elsewhere.mkdir()
        assert config.expanded_path("runtime/models") \
            == REPO_ROOT / "runtime" / "models"

        probe = (
            "import json\n"
            "from eve import config\n"
            "print(json.dumps({\n"
            "  'models': str(config.MODELS_DIR),\n"
            "  'kokoro': str(config.kokoro_dir()),\n"
            "  'silero': str(config.silero_model()),\n"
            "  'whisper': str(config.whisper_dir()),\n"
            "}))\n"
        )
        settings = tmp_path / "config"
        settings.mkdir()
        (settings / "env").write_text(
            "EVE_MODELS_DIR=runtime/models\n"
            "WHISPER_DIR=runtime/whisper.cpp\n"
        )
        environment = {
            key: value for key, value in os.environ.items()
            if not key.startswith(("EVE_", "HEY_CLAUDE_", "KOKORO_", "WHISPER_"))
        }
        environment.update({
            "EVE_CONFIG_DIR": str(settings),
            "PYTHONPATH": str(REPO_ROOT),
        })
        result = subprocess.run(
            [sys.executable, "-c", probe], cwd=elsewhere, env=environment,
            capture_output=True, text=True, timeout=60,
        )
        assert result.returncode == 0, result.stderr
        got = json.loads(result.stdout)
        assert got == {
            "models": str(REPO_ROOT / "runtime" / "models"),
            "kokoro": str(REPO_ROOT / "runtime" / "models" / "kokoro"),
            "silero": str(
                REPO_ROOT / "runtime" / "models" / "models" / "silero_vad.onnx"
            ),
            "whisper": str(REPO_ROOT / "runtime" / "whisper.cpp"),
        }

    def test_tilde_in_the_data_file_is_expanded(self, tmp_path):
        home = tmp_path / "home"
        settings = tmp_path / "config"
        home.mkdir()
        settings.mkdir()
        (settings / "env").write_text(
            "EVE_MODELS_DIR=~/weights\nKOKORO_DIR=~/voices\n"
        )
        probe = (
            "from eve import config\n"
            "print(config.MODELS_DIR)\n"
            "print(config.kokoro_dir())\n"
        )
        environment = {
            key: value for key, value in os.environ.items()
            if not key.startswith(("EVE_", "HEY_CLAUDE_", "KOKORO_", "WHISPER_"))
        }
        environment.update({
            "EVE_CONFIG_DIR": str(settings),
            "HOME": str(home),
            "PYTHONPATH": str(REPO_ROOT),
        })
        result = subprocess.run(
            [sys.executable, "-c", probe], cwd=tmp_path, env=environment,
            capture_output=True, text=True, timeout=60,
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.splitlines() == [
            str(home / "weights"), str(home / "voices")
        ]


class TestOnlySettingsAreSettings:
    """The file may set settings. It used to be able to set anything.

    load_settings puts what it finds into os.environ, and os.environ is
    inherited by every subprocess a turn spawns — arecord, aplay, amixer,
    whisper-cli, and the `sudo -n chvt` the face runs at startup. A line
    reading LD_PRELOAD= or PATH= was therefore code execution against all of
    them, wearing the costume of a configuration mistake.

    It is not a privilege boundary — the file is 0600 in a 0700 directory, so
    writing it already means being that account. What this buys is that a
    limited write primitive stays one, and that the file does what its name
    says.
    """

    @pytest.mark.parametrize("key", [
        "LD_PRELOAD", "LD_LIBRARY_PATH", "PATH", "PYTHONPATH", "PYTHONSTARTUP",
        "BASH_ENV", "IFS",
    ])
    def test_an_execution_lever_is_refused(self, key, env_file, monkeypatch):
        monkeypatch.delenv(key, raising=False)
        config.SETTINGS_IGNORED.clear()
        env_file(f"{key}=/tmp/whatever\n")
        config.load_settings()
        assert key not in os.environ
        assert key in config.SETTINGS_IGNORED

    def test_onnx_telemetry_cannot_be_reenabled_from_the_settings_file(
        self, env_file, monkeypatch
    ):
        # Package bootstrap owns this policy before config is imported. Keep
        # ORT_* outside the data-file allowlist so a later loader change cannot
        # turn a privacy boundary into an operator toggle.
        key = "ORT_DISABLE_TELEMETRY"
        monkeypatch.delenv(key, raising=False)
        config.SETTINGS_IGNORED.clear()
        env_file(f"{key}=0\n")
        assert config.load_settings() == []
        assert key not in os.environ
        assert key in config.SETTINGS_IGNORED
        assert not any(key.startswith(prefix) for prefix in config.SETTABLE)

    @pytest.mark.parametrize("key", [
        "VOICE_HANG_S", "KOKORO_DIR", "WHISPER_DIR", "EVE_CONFIG_DIR",
        "BARKEEP_URL", "HEY_CLAUDE_CONFIG",
    ])
    def test_a_real_setting_still_applies(self, key, env_file, monkeypatch):
        monkeypatch.delenv(key, raising=False)
        env_file(f"{key}=value\n")
        assert key in config.load_settings()
        assert os.environ[key] == "value"

    def test_every_key_env_example_documents_is_settable_or_secret(self):
        # The allowlist has to cover the documented surface, or a key would be
        # documented, written by install.sh, and silently dropped.
        documented = {
            line.split("=")[0].lstrip("# ").strip()
            for line in (REPO_ROOT / "env.example").read_text().splitlines()
            if "=" in line and line.lstrip("# ")[:1].isupper()
        }
        for key in {k for k in documented if k.isupper() and k}:
            assert key.startswith(config.SETTABLE) or key in config.SECRETS, key

    def test_what_was_refused_is_said_out_loud(self, env_file, monkeypatch):
        # A line that vanishes without comment is indistinguishable from one
        # that took effect, which is the failure this whole function exists
        # to fix — so refusing quietly would repeat it.
        monkeypatch.delenv("LD_PRELOAD", raising=False)
        config.SETTINGS_IGNORED.clear()
        env_file("LD_PRELOAD=/tmp/evil.so\n")
        config.load_settings()
        assert config.SETTINGS_IGNORED == ["LD_PRELOAD"]
