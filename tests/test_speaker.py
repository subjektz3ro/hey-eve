"""Keeping the speaker connected, and never being silent about it.

All three pieces here come from one incident on 2026-08-15. The Pi rebooted;
the SoundSticks came back Paired, Bonded and Trusted but not Connected;
/etc/asound.conf pins ALSA's `default` at that one address, so every aplay
died on open — and the journal showed two immaculate turns with timings,
token counts and cost. She reported success for replies nobody heard.

So: something has to reconnect the speaker (nothing on a Linux box does —
BlueZ reconnects after a *link loss* and a reboot is not one), and playback
failure has to say so.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "deploy"))
import render_service  # noqa: E402  (needs the path line above)

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "deploy" / "connect-speaker.sh"
UNIT = REPO_ROOT / "deploy" / "eve-speaker@.service"
TIMER = REPO_ROOT / "deploy" / "eve-speaker@.timer"

HOST = {
    "checkout": Path("/home/someone/eve"),
    "uv": Path("/home/someone/.local/bin/uv"),
    "config_dir": Path("/home/someone/.config/eve"),
    "uv_cache_dir": Path("/home/someone/.cache/uv"),
}

# Obviously synthetic, deliberately. A plausible-looking address in a
# fixture is a real address to anyone reading it later.
MAC = "AA:BB:CC:11:22:33"


def run_script(tmp_path, *, env_body=None, stub=None):
    """Run connect-speaker.sh with a fake bluetoothctl on PATH.

    The stub records its argv so a test can assert what was asked of it, which
    is the only externally visible thing the script does.
    """
    config_dir = tmp_path / "config"
    config_dir.mkdir(exist_ok=True)
    if env_body is not None:
        (config_dir / "env").write_text(env_body)

    bindir = tmp_path / "bin"
    bindir.mkdir(exist_ok=True)
    log = tmp_path / "calls.log"
    stub = stub if stub is not None else 'echo "Connected: no"'
    (bindir / "bluetoothctl").write_text(
        f'#!/bin/sh\necho "$@" >> "{log}"\n{stub}\n'
    )
    (bindir / "bluetoothctl").chmod(0o755)

    result = subprocess.run(
        ["bash", str(SCRIPT)],
        capture_output=True, text=True, timeout=30,
        env={**os.environ,
             "PATH": f"{bindir}:{os.environ['PATH']}",
             "EVE_CONFIG_DIR": str(config_dir),
             "HOME": str(tmp_path)},
    )
    calls = log.read_text().splitlines() if log.exists() else []
    return result, calls


class TestTheReconnectScript:
    def test_an_address_that_is_already_connected_is_left_alone(self, tmp_path):
        # The common case: the timer runs this every minute, so "already up"
        # must cost nothing and say nothing.
        result, calls = run_script(
            tmp_path,
            env_body=f"VOICE_SPEAKER_MAC={MAC}\n",
            stub='echo "Connected: yes"',
        )
        assert result.returncode == 0
        assert not any("connect" in call for call in calls), calls
        assert result.stdout.strip() == ""

    def test_a_disconnected_address_is_connected(self, tmp_path):
        result, calls = run_script(
            tmp_path,
            env_body=f"VOICE_SPEAKER_MAC={MAC}\n",
            stub='case "$1" in info) echo "Connected: no";; '
                 'connect) echo "Connection successful";; esac',
        )
        assert result.returncode == 0
        assert f"connect {MAC}" in calls
        assert "connected" in result.stdout

    def test_a_speaker_that_is_switched_off_is_not_a_failed_unit(self, tmp_path):
        # Exiting non-zero here would leave a red unit for a condition that
        # resolves itself the moment somebody presses the power button, and
        # the timer would keep re-reddening it every minute.
        result, _ = run_script(
            tmp_path,
            env_body=f"VOICE_SPEAKER_MAC={MAC}\n",
            stub='echo "Connected: no"',   # connect never succeeds
        )
        assert result.returncode == 0
        assert "will retry" in result.stdout

    def test_no_address_configured_is_a_wired_speaker_not_an_error(self, tmp_path):
        result, calls = run_script(tmp_path, env_body="ANTHROPIC_API_KEY=sk-x\n")
        assert result.returncode == 0
        assert calls == []
        assert "nothing to connect" in result.stdout

    def test_a_missing_env_file_is_survived(self, tmp_path):
        result, _ = run_script(tmp_path, env_body=None)
        assert result.returncode == 0

    def test_it_reads_the_env_file_without_sourcing_it(self, tmp_path):
        # The same file holds the API key. If this script ever sourced it, a
        # key containing a backtick would become code — which is the whole
        # reason config.secret() parses rather than sources.
        result, calls = run_script(
            tmp_path,
            env_body=f'ANTHROPIC_API_KEY=sk-$(touch {tmp_path}/pwned)\n'
                     f"VOICE_SPEAKER_MAC={MAC}\n",
            stub='echo "Connected: yes"',
        )
        assert result.returncode == 0
        assert not (tmp_path / "pwned").exists()
        assert calls  # it still found the address below the key

    def test_the_environment_beats_the_file(self, tmp_path):
        # Same precedence as config.secret(), so a one-off override works
        # without editing the file.
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        (config_dir / "env").write_text(f"VOICE_SPEAKER_MAC={MAC}\n")
        bindir = tmp_path / "bin"
        bindir.mkdir()
        log = tmp_path / "calls.log"
        (bindir / "bluetoothctl").write_text(
            f'#!/bin/sh\necho "$@" >> "{log}"\necho "Connected: no"\n')
        (bindir / "bluetoothctl").chmod(0o755)
        subprocess.run(
            ["bash", str(SCRIPT)], capture_output=True, text=True, timeout=30,
            env={**os.environ,
                 "PATH": f"{bindir}:{os.environ['PATH']}",
                 "EVE_CONFIG_DIR": str(config_dir),
                 "HOME": str(tmp_path),
                 "VOICE_SPEAKER_MAC": "AA:BB:CC:DD:EE:FF"},
        )
        assert any("AA:BB:CC:DD:EE:FF" in call for call in log.read_text().splitlines())


class TestTheUnitAndTimer:
    def test_the_unit_carries_no_absolute_home_path_of_its_own(self):
        # Same gate the main unit has, and the same reason: the repository
        # must not name one person's home directory.
        assert "/home/" not in UNIT.read_text()
        assert "/home/" not in TIMER.read_text()

    def test_it_renders_with_nothing_left_unresolved(self):
        import re
        rendered = render_service.render(UNIT.read_text(), **HOST)
        assert not re.findall(r"@[A-Z_]+@", rendered)
        assert "/home/someone/eve/deploy/connect-speaker.sh" in rendered
        assert 'Environment="EVE_CONFIG_DIR=/home/someone/.config/eve"' \
            in rendered

    def test_every_placeholder_it_uses_is_one_the_renderer_knows(self):
        import re
        used = set(re.findall(r"@[A-Z_]+@", UNIT.read_text()))
        assert used <= set(render_service.PLACEHOLDERS), used

    def test_it_runs_as_the_instance_account_not_root(self):
        # A2DP belongs to a user's Bluetooth session; root gets a different
        # one, which is the same reason eve@.service does this.
        assert "User=%i" in UNIT.read_text()

    def test_it_waits_for_the_stack_that_owns_the_endpoints(self):
        # Connecting before bluealsa is up produces a link with nowhere to go.
        assert "bluealsa.service" in UNIT.read_text()

    def test_the_timer_retries_rather_than_firing_once_at_boot(self):
        # A boot-only connect leaves two cases silent until the next reboot:
        # the speaker switched on after the Pi, and a link that drops later.
        body = TIMER.read_text()
        assert "OnBootSec=" in body
        assert "OnUnitActiveSec=" in body

    def test_the_timer_drives_the_matching_instance(self):
        assert "Unit=eve-speaker@%i.service" in TIMER.read_text()


class TestSayingSoWhenNothingIsHeard:
    def test_a_playback_failure_is_reported_rather_than_swallowed(self, monkeypatch, capsys):
        # The incident itself: aplay dies on open, speak() catches it, and the
        # turn looks perfect in the journal. One line is the whole fix.
        from eve import speech

        monkeypatch.setattr(speech.tts, "synth", lambda text: b"\0\1" * 1000)

        class DeadPlayer:
            """An aplay that dies on open, the way a missing sink does."""

            def __init__(self, *a, **k):
                self.stdin = self
                self.closed = False
                self.returncode = None
            def write(self, _):
                raise BrokenPipeError("aplay exited")
            def flush(self): pass
            def close(self): self.closed = True
            def poll(self): return self.returncode
            def wait(self, timeout=None):
                self.returncode = 1
                return 1
            def kill(self): self.returncode = -9

        monkeypatch.setattr(speech.subprocess, "Popen", DeadPlayer)
        monkeypatch.setattr(speech.time, "sleep", lambda _: None)
        speech.speak("A reply nobody will hear.")
        assert "nothing was heard" in capsys.readouterr().err

    def test_a_speaker_that_will_not_open_is_reported_as_absent(self, monkeypatch):
        from eve import speech

        def refuse(*a, **k):
            return subprocess.CompletedProcess(a[0] if a else [], 1)

        monkeypatch.setattr(speech.subprocess, "run", refuse)
        assert speech.speaker_available() is False

    def test_a_missing_aplay_is_absent_rather_than_a_traceback(self, monkeypatch):
        from eve import speech

        def missing(*a, **k):
            raise FileNotFoundError("aplay")

        monkeypatch.setattr(speech.subprocess, "run", missing)
        assert speech.speaker_available() is False

    def test_a_working_device_reports_available(self, monkeypatch):
        from eve import speech

        monkeypatch.setattr(
            speech.subprocess, "run",
            lambda *a, **k: subprocess.CompletedProcess(a[0] if a else [], 0))
        assert speech.speaker_available() is True

    def test_the_probe_actually_sends_audio_at_the_players_rate(self, monkeypatch):
        # aplay given an empty stream can exit 0 without ever opening the
        # device, which is the false negative this probe exists to avoid.
        from eve import config, speech

        seen = {}

        def capture(argv, **kwargs):
            seen["argv"], seen["input"] = argv, kwargs.get("input")
            return subprocess.CompletedProcess(argv, 0)

        monkeypatch.setattr(speech.subprocess, "run", capture)
        speech.speaker_available()
        assert seen["input"], "the probe must write real samples"
        assert str(config.TTS_RATE) in seen["argv"]
