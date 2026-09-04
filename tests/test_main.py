"""The entry point's surface: which face is fitted, and what the flags mean.

The listen loop itself needs a microphone and is not exercised here. What is
worth pinning is everything decided before it starts — because those are the
decisions that turn a working assistant into a silent one, and they are all
made from configuration rather than from code.
"""
from __future__ import annotations

import importlib
import subprocess
import sys

import pytest

from eve import config, main


class TestThereIsOneFace:
    """There were four, chosen by VOICE_FACE at startup.

    Three were never fitted to the machine and were a standing tax: every
    change to how she behaves had to be made, or deliberately not made, in
    four places. What is left is the seam that mattered — eve/head.py is
    everything she does, eve/void.py is the only thing that decides what any
    of it looks like — and nothing picks between them because there is
    nothing to pick.
    """

    def test_nothing_reads_a_face_setting_any_more(self):
        assert not hasattr(main, "FACE")
        assert config.secret("VOICE_FACE") == ""

    def test_the_presentation_still_chooses_her_voice(self, settings_file):
        settings_file(VOICE_PRESENT="male")
        module = importlib.reload(config)
        try:
            assert module.voice() == "am_michael"
        finally:
            importlib.reload(config)


class TestTheNoOpDisplay:
    def test_a_disabled_display_builds_nothing(self):
        # --no-display and --text both run without a panel, and every call
        # below has to be a no-op rather than an AttributeError.
        display = main._Display(enabled=False)
        assert display.orb is None

    def test_every_call_on_a_disabled_display_is_harmless(self):
        display = main._Display(enabled=False)
        display.state("thinking")
        display.state("listening", 0.5, "SIGNAL IN")
        display.wave([0, 1, 2])
        display.speech([])
        display.audio(1.0, 0.5)
        display.close()

    def test_a_panel_that_fails_to_open_does_not_stop_the_assistant(self):
        # There is no /dev/fb0 in CI, and there is none on a Pi whose panel
        # has come unseated either. Either way she should still answer.
        display = main._Display(enabled=True)
        assert display.orb is None or hasattr(display.orb, "set_state")
        display.close()


class TestTheAudioTemporaryDirectory:
    def test_captured_speech_prefers_ram(self):
        # It has to be a file because whisper.cpp reads one, but it should
        # not outlive the process that needed it, and /dev/shm never touches
        # the NVMe.
        assert main._AUDIO_TMP in ("/dev/shm", None)

    def test_a_host_without_tmpfs_falls_back_rather_than_failing(self):
        # macOS has no /dev/shm. The fallback is the normal temp directory,
        # which is less private but still works.
        module = importlib.reload(main)
        assert module._AUDIO_TMP is None or module._AUDIO_TMP == "/dev/shm"


class TestSignals:
    def test_sigterm_becomes_a_normal_unwind(self):
        # Python's default handler exits outright, so
        # `with tempfile.TemporaryDirectory()` never ran its cleanup: every
        # `systemctl restart` left a directory behind holding the last thing
        # anyone said. Two dozen accumulated in one afternoon of iterating.
        with pytest.raises(SystemExit):
            main._clean_exit(15, None)


class TestTheCommandLine:
    def _help(self):
        return subprocess.run(
            [sys.executable, "-m", "eve.main", "--help"],
            capture_output=True, text=True, timeout=60,
        )

    def test_help_lists_every_mode(self):
        out = self._help().stdout
        for flag in ("--say", "--text", "--push", "--quiet", "--no-display",
                     "--model"):
            assert flag in out

    def test_help_exits_cleanly(self):
        assert self._help().returncode == 0

    def test_the_modes_are_mutually_exclusive(self):
        result = subprocess.run(
            [sys.executable, "-m", "eve.main", "--text", "--push"],
            capture_output=True, text=True, timeout=60,
        )
        assert result.returncode != 0
        assert "not allowed with argument" in result.stderr

    def test_the_default_model_is_named_in_the_help(self):
        # So "which model is this actually using" is answerable without
        # reading config.py.
        assert config.MODEL in self._help().stdout


class TestTheFollowUpWindow:
    def test_it_is_measured_in_thinking_time_not_talking_time(self):
        # Measured from the end of the reply to the moment they START
        # speaking. Twenty-five seconds was not enough to compose a real
        # question.
        assert main.FOLLOW_UP_S >= 30

    def test_the_cutoff_warning_arrives_before_the_cutoff(self):
        assert 0 < main._CUTOFF_WARN_S


class TestLosingHerVoiceDoesNotLoseTheAssistant:
    """The regression that took her down in the field.

    The Kokoro weights went missing from ~/.local/share/eve/kokoro, so
    tts.synth raised inside speech.speak. `_turn` did not guard that call, so
    the exception unwound out of main() and the process exited — and because
    the unit restarts always, what a person actually saw was a question being
    thought about and then silence, every single time, with the reason buried
    in a traceback. She had answered correctly each time.
    """

    def _responder(self, reply="Oslo."):
        class Fake:
            last_usage = (10, 5)

            def respond(self, said, on_state=None):
                return reply

            def cost_usd(self):
                return 0.0

        return Fake()

    def test_a_synthesis_failure_does_not_end_the_turn(self, monkeypatch):
        def explode(*args, **kwargs):
            raise RuntimeError("Kokoro model not found in /nope")

        monkeypatch.setattr(main, "_speak", explode)
        display = main._Display(enabled=False)
        # The claim: this returns instead of propagating.
        main._turn(self._responder(), "capital of Norway?", display, quiet=False)

    def test_the_reason_is_logged_where_a_person_will_find_it(
            self, monkeypatch, capsys):
        def explode(*args, **kwargs):
            raise RuntimeError("Kokoro model not found in /nope")

        monkeypatch.setattr(main, "_speak", explode)
        main._turn(self._responder(), "capital?", main._Display(enabled=False),
                   quiet=False)
        err = capsys.readouterr().err
        assert "could not speak" in err
        assert "Kokoro model not found" in err


    def test_playback_failing_midway_is_survived_too(self, monkeypatch):
        # Not just a missing model: a busy ALSA device, an unplugged speaker,
        # a Bluetooth link that dropped. All of them used to be fatal.
        def explode(*args, **kwargs):
            raise OSError("aplay: device or resource busy")

        monkeypatch.setattr(main, "_speak", explode)
        main._turn(self._responder(), "capital?", main._Display(enabled=False),
                   quiet=False)

    def test_the_cost_line_is_still_reported(self, monkeypatch, capsys):
        # The turn genuinely happened and genuinely cost money, whether or not
        # it could be spoken aloud.
        monkeypatch.setattr(main, "_speak",
                            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x")))
        main._turn(self._responder(), "capital?", main._Display(enabled=False),
                   quiet=False)
        assert "in /" in capsys.readouterr().err

    def test_a_working_voice_is_still_used(self, monkeypatch):
        spoken = []
        monkeypatch.setattr(main, "_speak",
                            lambda reply, display: spoken.append(reply))
        main._turn(self._responder("Oslo."), "capital?",
                   main._Display(enabled=False), quiet=False)
        assert spoken == ["Oslo."]

    def test_quiet_mode_never_reaches_synthesis_at_all(self, monkeypatch):
        monkeypatch.setattr(main, "_speak",
                            lambda *a, **k: pytest.fail("should not speak"))
        main._turn(self._responder(), "capital?",
                   main._Display(enabled=False), quiet=True)


def test_an_unlisted_model_reports_unknown_cost_without_losing_the_turn(
        monkeypatch, capsys):
    class Responder:
        last_usage = (10, 5)

        def respond(self, said, on_state=None):
            return "Hello."

        def cost_usd(self):
            return None

    monkeypatch.setattr(main, "_speak", lambda *args, **kwargs: None)
    main._turn(
        Responder(), "hello", main._Display(enabled=False), quiet=True
    )

    assert "cost unknown" in capsys.readouterr().err


class TestSayingUpFrontWhetherSheHasAVoice:
    def test_missing_weights_are_reported_at_startup(self, tmp_path,
                                                     monkeypatch, capsys):
        # tts.paths() was always written to return None rather than raise, so
        # that a missing model would be "a diagnosable startup message, not a
        # traceback from inside the first spoken reply". Nothing checked it,
        # so what actually happened was the traceback.
        from eve import tts
        monkeypatch.setattr(tts, "KOKORO_DIR", tmp_path)
        assert tts.paths() is None

    def test_the_speaker_warning_is_reachable_with_working_weights(
            self, monkeypatch, capsys):
        monkeypatch.setattr(main.tts, "paths", lambda: (object(), object()))
        monkeypatch.setattr(main, "_warm_voice", lambda: None)
        monkeypatch.setattr(main.speech, "speaker_available", lambda: False)

        class ImmediateThread:
            def __init__(self, *, target, daemon):
                self.target = target
                assert daemon is True

            def start(self):
                self.target()

        monkeypatch.setattr(main.threading, "Thread", ImmediateThread)
        main._prepare_voice(quiet=False)
        assert "no speaker" in capsys.readouterr().err

    def test_missing_weights_do_not_hide_a_missing_speaker(
            self, monkeypatch, capsys):
        monkeypatch.setattr(main.tts, "paths", lambda: None)
        monkeypatch.setattr(main.speech, "speaker_available", lambda: False)
        main._prepare_voice(quiet=False)
        err = capsys.readouterr().err
        assert "no voice" in err
        assert "no speaker" in err

    def test_quiet_mode_probes_neither_voice_nor_speaker(self, monkeypatch):
        def reached():
            pytest.fail("quiet mode touched voice hardware")

        monkeypatch.setattr(main.tts, "paths", reached)
        monkeypatch.setattr(main.speech, "speaker_available", reached)
        main._prepare_voice(quiet=True)
