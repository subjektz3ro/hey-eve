"""What speak() and record_until_silence() leave behind.

The pure halves of both are covered elsewhere — sentence splitting, stall
prediction, throughput. This is the other half: threads, child processes and
the paths taken when the hardware is not there. It needs no microphone and no
ALSA device, which is the point. The bugs it pins were all live, and all of
them were inside the part the coverage register had written off as "I/O".
"""
from __future__ import annotations

import signal
import subprocess
import threading
import time

import pytest

from eve import config, speech


class FakePlayer:
    """An aplay that accepts everything and exits cleanly."""

    PIPE, DEVNULL = -1, -3
    SubprocessError = subprocess.SubprocessError

    def __init__(self, *args, **kwargs):
        self.written = bytearray()
        self.stdin = self
        self.closed = False
        self.killed = False
        self.returncode = None

    # stdin surface
    def write(self, block): self.written.extend(block)
    def flush(self): pass
    def close(self): self.closed = True

    # process surface
    def wait(self, timeout=None):
        self.returncode = 0
        return 0

    def poll(self): return self.returncode

    def kill(self):
        self.killed = True
        self.returncode = -9


@pytest.fixture
def player(monkeypatch):
    """Run the real speak() against a fake synthesiser and a fake player."""
    made = {}

    def fake_popen(*args, **kwargs):
        made["process"] = FakePlayer(*args, **kwargs)
        return made["process"]

    fake_subprocess = type("S", (), {
        "Popen": staticmethod(fake_popen),
        "PIPE": -1, "DEVNULL": -3,
        "SubprocessError": subprocess.SubprocessError,
    })
    monkeypatch.setattr(speech, "subprocess", fake_subprocess)
    monkeypatch.setattr(speech.tts, "synth",
                        lambda text: b"\0\1" * int(len(text) * 40 + 501))
    return made


def threads_now():
    return {t for t in threading.enumerate() if t is not threading.main_thread()}


class TestThePacerThreadTerminates:
    def test_speak_leaves_no_thread_running(self, player):
        # The bug: pace() only advanced `position` on a *full* step, but
        # exited on `position >= available`. Any reply whose byte count is not
        # an exact multiple of the step — about 1322 in 1323 — left the tail
        # unconsumed, so that test never became true and the thread span at
        # 200 wakeups a second for the life of the process. Two orphans were
        # still running on the Pi after two replies.
        before = threads_now()
        speech.speak("A reply of a length nobody chose deliberately.",
                     on_level=lambda level: None, on_wave=lambda s: None)
        leaked = threads_now() - before
        assert not leaked, f"speak() leaked {leaked}"

    def test_it_terminates_for_a_reply_of_any_length(self, player):
        # Sweep lengths so at least one is an exact multiple of the step and
        # the rest are not; before the fix only the exact multiple exited.
        before = threads_now()
        for words in range(1, 12):
            speech.speak("word " * words + "end.",
                         on_level=lambda level: None, on_wave=lambda s: None)
        assert not threads_now() - before

    def test_it_returns_promptly_rather_than_waiting_out_the_join(self, player):
        # The 2.0s pacer.join() timeout was paid on every single reply, before
        # on_done() and before the microphone reopened — so she was deaf for
        # 2.45s at exactly the moment main.py says people answer immediately.
        start = time.perf_counter()
        speech.speak("Short.", on_level=lambda level: None, on_wave=lambda s: None)
        elapsed = time.perf_counter() - start
        assert elapsed < speech._PLAYBACK_SETTLE_S + 1.0, \
            f"speak() took {elapsed:.2f}s; the pacer join is timing out again"

    def test_the_whole_reply_still_reaches_the_player(self, player):
        # Terminating early would be easy and wrong: everything synthesised
        # must still be written.
        speech.speak("One. Two. Three.",
                     on_level=lambda level: None, on_wave=lambda s: None)
        assert len(player["process"].written) > 0


class TestThePlayerIsAlwaysReaped:
    def test_an_exception_after_the_player_opens_does_not_leak_it(self, player):
        # speak()'s except tuple is (BrokenPipeError, OSError,
        # SubprocessError), and it held the only process.kill() in the
        # function. Anything else raised after open_player() walked straight
        # past it, leaving aplay blocked on a pipe nobody would write to
        # again, holding the ALSA device — and main._turn catches and carries
        # on, so they accumulated one per failed reply.
        #
        # The reachable trigger is a display callback, not tts.synth: the
        # buffering policy means synthesis finishes before the player opens,
        # but on_speech is invoked immediately after it does, and the face is
        # on the other end of it.
        def face_blows_up(mouths):
            raise ValueError("the display thread is having a day")

        with pytest.raises(ValueError):
            speech.speak("Say something.",
                         on_level=lambda level: None,
                         on_wave=lambda s: None,
                         on_speech=face_blows_up)
        assert player["process"].killed or player["process"].closed

    def test_the_player_is_reaped_even_then(self, player):
        def face_blows_up(mouths):
            raise ValueError("boom")

        before = threads_now()
        with pytest.raises(ValueError):
            speech.speak("Say something.",
                         on_level=lambda level: None,
                         on_wave=lambda s: None,
                         on_speech=face_blows_up)
        # and the pacer still goes with it
        assert not threads_now() - before

    def test_a_dead_player_is_reported_rather_than_swallowed(self, player, monkeypatch, capsys):
        def refuse(*args, **kwargs):
            dead = FakePlayer()
            dead.write = lambda block: (_ for _ in ()).throw(BrokenPipeError())
            player["process"] = dead
            return dead

        monkeypatch.setattr(speech.subprocess, "Popen", refuse)
        speech.speak("Nobody will hear this.",
                     on_level=lambda level: None, on_wave=lambda s: None)
        assert "nothing was heard" in capsys.readouterr().err


class FakeRecorder:
    """An arecord that dies immediately, the way a missing device does."""

    def __init__(self, returncode=1):
        self.returncode = returncode
        self.stdout = self
        self.closed = False

    def read(self, _n): return b""
    def close(self): self.closed = True
    def terminate(self): pass
    def wait(self, timeout=None): return self.returncode
    def kill(self): pass


class TestADeadRecorderIsNotSilence:
    @pytest.fixture(autouse=True)
    def _no_real_sleep(self, monkeypatch):
        # The backoff is the behaviour under test; waiting it out is not.
        self.slept = []
        monkeypatch.setattr(speech.time, "sleep", self.slept.append)

    def test_it_says_which_device_would_not_open(self, monkeypatch, tmp_path, capsys):
        # The whole failure: stderr is DEVNULL'd and the return code was never
        # read, so a dead microphone and a quiet room were the same answer.
        # She was permanently deaf while the panel still read READY.
        monkeypatch.setattr(speech.subprocess, "Popen",
                            lambda *a, **k: FakeRecorder(1))
        assert speech.record_until_silence(tmp_path / "heard.wav") is None
        assert "microphone did not open" in capsys.readouterr().err

    def test_it_pauses_before_letting_the_caller_try_again(self, monkeypatch, tmp_path):
        # main._listen_loop does `if captured is None: continue` with no sleep
        # of its own, so without this the failure is a fork bomb.
        monkeypatch.setattr(speech.subprocess, "Popen",
                            lambda *a, **k: FakeRecorder(1))
        speech.record_until_silence(tmp_path / "heard.wav")
        assert speech._CAPTURE_RETRY_S in self.slept

    def test_a_clean_stop_is_not_reported_as_a_failure(self, monkeypatch, tmp_path, capsys):
        # terminate() gives -SIGTERM, which is the normal end of every capture
        # and must never produce a warning.
        monkeypatch.setattr(speech.subprocess, "Popen",
                            lambda *a, **k: FakeRecorder(-signal.SIGTERM))
        speech.record_until_silence(tmp_path / "heard.wav")
        assert "did not open" not in capsys.readouterr().err
        assert not self.slept

    def test_the_pipe_is_closed_rather_than_left_to_the_collector(self, monkeypatch, tmp_path):
        recorder = FakeRecorder(-signal.SIGTERM)
        monkeypatch.setattr(speech.subprocess, "Popen", lambda *a, **k: recorder)
        speech.record_until_silence(tmp_path / "heard.wav")
        assert recorder.closed


class TestTheTranscriberIsBounded:
    def test_it_will_not_block_the_listen_loop_forever(self, monkeypatch, tmp_path):
        # While transcribe() runs she is deaf, so an unbounded decode takes
        # her off the air rather than costing one turn.
        seen = {}

        def capture(argv, **kwargs):
            seen.update(kwargs)
            return subprocess.CompletedProcess(argv, 0, stdout="hello", stderr="")

        monkeypatch.setattr(speech.subprocess, "run", capture)
        transcriber = speech.WhisperTranscriber(tmp_path / "bin", tmp_path / "model")
        assert transcriber.transcribe(tmp_path / "a.wav") == "hello"
        assert seen.get("timeout") == speech._TRANSCRIBE_TIMEOUT_S


class TestARunawaySourceIsBounded:
    def test_the_capture_ceiling_is_long_enough_for_a_person(self):
        # The runaway guard bounds *unbroken* speech, which a television is
        # not — it breathes between words exactly as a person does, so it
        # resets that counter. This is the backstop, and it must never fire
        # on somebody explaining something at length.
        assert speech._MAX_CAPTURE_S >= 180
        assert speech._MAX_CAPTURE_S > speech._HANG_S * 10

    def test_a_capture_that_never_ends_is_still_bounded(self, monkeypatch, tmp_path):
        # Silence forever after speech: the hang timer would normally end
        # this, so drive it with speech that never pauses long enough.
        monkeypatch.setattr(speech, "_MAX_CAPTURE_S", 0.5)
        monkeypatch.setattr(speech, "_HANG_S", 999.0)
        monkeypatch.setattr(speech, "_RUNAWAY_SPEECH_S", 0.0)
        monkeypatch.setattr(speech, "_DETECTOR", None)

        chunk_bytes = int(config.MIC_RATE * speech._CHUNK_S) * 2
        loud = (b"\x00\x40" * (chunk_bytes // 2))

        class Endless(FakeRecorder):
            def __init__(self):
                super().__init__(-signal.SIGTERM)
                self.sent = 0

            def read(self, _n):
                self.sent += 1
                # loud enough to be speech, then quiet enough not to be
                return loud if self.sent % 2 else b"\x00\x00" * (chunk_bytes // 2)

        monkeypatch.setattr(speech.subprocess, "Popen", lambda *a, **k: Endless())
        result = speech.record_until_silence(tmp_path / "heard.wav")
        assert result is not None      # it stopped, rather than recording forever


ARECORD_L = """\
**** List of CAPTURE Hardware Devices ****
card 0: Generic [HD-Audio Generic], device 0: ALC257 Analog [ALC257 Analog]
  Subdevices: 1/1
  Subdevice #0: subdevice #0
card 1: Snowball [Blue Snowball], device 0: USB Audio [USB Audio]
  Subdevices: 0/1
  Subdevice #0: subdevice #0
"""


class TestAMachineWithNoAlsa:
    """`arecord` absent entirely, which is not the same as a device absent.

    alsa-utils ships on Raspberry Pi OS, which is exactly why nothing noticed:
    on a minimal Debian or Fedora it is not installed. Unguarded, the Popen
    raised FileNotFoundError straight out of the listen loop, past a main()
    that catches only KeyboardInterrupt, and into a restart loop that reported
    a traceback rather than the one sentence worth saying.

    deploy/install.sh refuses to run without it now. This is the other half:
    somebody who installed by hand, or removed the package afterwards.
    """

    @pytest.fixture(autouse=True)
    def _no_arecord(self, monkeypatch):
        self.slept = []
        monkeypatch.setattr(speech.time, "sleep", self.slept.append)

        def absent(*args, **kwargs):
            raise FileNotFoundError(2, "No such file or directory", "arecord")

        monkeypatch.setattr(speech.subprocess, "Popen", absent)

    def test_it_does_not_take_the_assistant_down(self, tmp_path):
        assert speech.record_until_silence(tmp_path / "heard.wav") is None

    def test_it_names_the_package(self, tmp_path, capsys):
        # The whole point: a diagnosis rather than a traceback.
        speech.record_until_silence(tmp_path / "heard.wav")
        printed = capsys.readouterr().err
        assert "cannot record" in printed and "alsa-utils" in printed

    def test_it_pauses_rather_than_spinning(self, tmp_path):
        # main._listen_loop does `if captured is None: continue` with no sleep
        # of its own, so without this a missing package is a fork bomb.
        speech.record_until_silence(tmp_path / "heard.wav")
        assert speech._CAPTURE_RETRY_S in self.slept

    def test_she_keeps_trying(self, tmp_path):
        # Installing the package should fix her without a restart, which is
        # the same promise made for plugging the microphone back in.
        for _ in range(3):
            assert speech.record_until_silence(tmp_path / "heard.wav") is None


class TestFindingAMicrophone:
    """VOICE_MIC used to default to one particular Shure on one particular Pi.

    config.load_settings' own docstring names the trap that makes: on the
    machine it was written on, the installer's answer happened to match the
    code's default, "which is exactly why nobody noticed — it bites the second
    install, where a different microphone means total, silent deafness."

    `default` is not the answer either, and that is the whole reason this
    discovers rather than falling back. On a box with a Bluetooth speaker
    pinned as ALSA's default — which deploy/connect-speaker.sh exists to
    support — the *capture* side of `default` resolves to bluealsa and does
    not open at all.
    """

    def use(self, monkeypatch, listing, configured=""):
        monkeypatch.setattr(speech.config, "MIC_DEVICE", configured)
        monkeypatch.setattr(speech, "_mic_device", None)
        monkeypatch.setattr(
            speech.subprocess, "run",
            lambda *a, **k: subprocess.CompletedProcess(a[0], 0, listing, ""))
        return speech.mic_device()

    def test_a_configured_device_is_used_as_given(self, monkeypatch):
        assert self.use(monkeypatch, ARECORD_L, "hw:2,0") == "hw:2,0"

    def test_a_configured_device_is_not_second_guessed(self, monkeypatch):
        # Even when discovery would disagree. Somebody who names a device has
        # a reason, and it is usually that the obvious one is wrong.
        def explode(*a, **k):
            raise AssertionError("discovery ran despite a configured device")

        monkeypatch.setattr(speech.config, "MIC_DEVICE", "hw:9,0")
        monkeypatch.setattr(speech, "_mic_device", None)
        monkeypatch.setattr(speech.subprocess, "run", explode)
        assert speech.mic_device() == "hw:9,0"

    def test_without_one_it_takes_the_first_capture_card(self, monkeypatch):
        assert self.use(monkeypatch, ARECORD_L) == "plughw:CARD=Generic,DEV=0"

    def test_it_uses_alsas_reported_device_number(self, monkeypatch):
        listing = (
            "**** List of CAPTURE Hardware Devices ****\n"
            "card 4: Array [USB Array], device 7: USB Audio [USB Audio]\n"
        )
        assert self.use(monkeypatch, listing) == "plughw:CARD=Array,DEV=7"

    def test_it_asks_for_capture_hardware_rather_than_ALSA_defaults(self):
        # `arecord -l` lists real cards; `arecord -L` lists whatever the
        # config points at, including a Bluetooth sink that cannot record.
        import inspect
        assert '"arecord", "-l"' in inspect.getsource(speech._find_microphone)

    def test_it_wraps_the_card_in_plug(self, monkeypatch):
        # plughw: rather than hw:, so ALSA converts rather than refusing a
        # device that cannot do 16kHz natively.
        assert self.use(monkeypatch, ARECORD_L).startswith("plughw:")

    def test_a_machine_with_no_capture_hardware_falls_back(self, monkeypatch):
        empty = "**** List of CAPTURE Hardware Devices ****\n"
        assert self.use(monkeypatch, empty) == "default"

    def test_a_missing_arecord_falls_back_rather_than_raising(self, monkeypatch):
        monkeypatch.setattr(speech.config, "MIC_DEVICE", "")
        monkeypatch.setattr(speech, "_mic_device", None)

        def absent(*a, **k):
            raise FileNotFoundError("arecord")

        monkeypatch.setattr(speech.subprocess, "run", absent)
        assert speech.mic_device() == "default"

    def test_it_is_worked_out_once(self, monkeypatch):
        # The listen loop opens a recorder every capture; this must not be a
        # subprocess per capture.
        calls = []
        monkeypatch.setattr(speech.config, "MIC_DEVICE", "")
        monkeypatch.setattr(speech, "_mic_device", None)

        def counting(*a, **k):
            calls.append(a[0])
            return subprocess.CompletedProcess(a[0], 0, ARECORD_L, "")

        monkeypatch.setattr(speech.subprocess, "run", counting)
        for _ in range(5):
            speech.mic_device()
        assert len(calls) == 1

    def test_it_says_which_one_it_picked(self, monkeypatch, capsys):
        # The one line that turns "she cannot hear me" into a diagnosis.
        self.use(monkeypatch, ARECORD_L)
        assert "microphone: plughw:CARD=Generic,DEV=0" in capsys.readouterr().err

    def test_the_card_name_survives_the_round_trip(self, monkeypatch):
        # _mic_card reads the amixer control back off whatever this returned.
        self.use(monkeypatch, ARECORD_L)
        assert speech._mic_card() == "Generic"
