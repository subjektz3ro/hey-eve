"""Telling a muted microphone from a quiet room.

Muting the microphone is a normal way to use this assistant — it is how you
keep her out of a conversation you are having with somebody else — so she can
expect to spend hours in that state. Until now she could not perceive it at
all. `record_until_silence` with no lead-in waits forever, so a muted
microphone and an empty room produced identical behaviour: READY on the panel,
nothing in the journal, and an assistant that was completely deaf while
looking completely healthy.

That is the same visible symptom the suite already pins for a *dead* recorder
in test_speech_lifecycle.py. The difference is that this one is somebody
pressing a button on purpose, so the answer is not a diagnostic — it is to say
so on the panel, and eventually to go to sleep.

Two ways of telling, and the better one came from the hardware. The MV6 drives
an ALSA capture switch from its touch panel, so she can simply ask; that is
exact, and it separates the two cases the signal cannot — muted because
somebody pressed a button, and no signal because the microphone has failed.
The RMS reading below is the fallback for a device that will not say, and it
carries a measured surprise: this microphone does not mute to digital silence.

Sleep draws her asleep rather than blanking the panel. That was a real
correction, not a flourish: a dark screen cannot be told apart from a crash,
a dead backlight or a pulled plug, which is the failure this project keeps
finding. eve/void.py has the lids, the breathing and the z's; what is pinned
here is that sleeping never stops the renderer.
"""
from __future__ import annotations

import array
import signal

import pytest

from eve import config, main, speech

CHUNK_BYTES = int(config.MIC_RATE * speech._CHUNK_S) * 2


def chunk(magnitude: int) -> bytes:
    """One capture chunk whose RMS is exactly `magnitude`."""
    count = CHUNK_BYTES // 2
    return array.array("h", [magnitude, -magnitude] * (count // 2)).tobytes()


SILENT = chunk(0)     # what a hardware mute delivers
ROOM = chunk(13)      # room tone, as measured on the MV6 at its current gain


class Stream:
    """An arecord handing over a scripted sequence of chunks, then EOF."""

    def __init__(self, chunks):
        self.chunks = list(chunks)
        self.returncode = -signal.SIGTERM
        self.stdout = self

    def read(self, _n):
        return self.chunks.pop(0) if self.chunks else b""

    def close(self): pass
    def terminate(self): pass
    def wait(self, timeout=None): return self.returncode
    def kill(self): pass


class CountingDetector:
    """A Silero stand-in that reports what it was asked to do."""

    def __init__(self):
        self.calls = 0
        self.resets = 0

    def speech_probability(self, samples):
        self.calls += 1
        return 0.0

    def reset(self):
        self.resets += 1


def silence_runs(monkeypatch, tmp_path, chunks, detector=None):
    """Run one capture over `chunks`, returning the reported silence runs."""
    monkeypatch.setattr(speech.subprocess, "Popen", lambda *a, **k: Stream(chunks))
    monkeypatch.setattr(speech, "_DETECTOR", detector)
    seen: list[float] = []
    speech.record_until_silence(
        tmp_path / "heard.wav",
        lead_in=None,
        on_activity=lambda level, is_speech, remaining=None, silent_for=0.0:
            seen.append(silent_for),
    )
    return seen


def dithered_silence() -> bytes:
    """What the MV6 actually delivers while muted.

    Not zeroes. Measured across 93 consecutive windows with the mute engaged:
    every one of them non-zero, RMS between 0.4285 and 0.5358, peak sample
    value 1. It is a converter idling at one LSB, not a muted signal path.
    A quarter of the samples at +/-1 reproduces RMS 0.5.
    """
    count = CHUNK_BYTES // 2
    samples = [0] * count
    for index in range(0, count, 4):
        samples[index] = 1 if index % 8 == 0 else -1
    return array.array("h", samples).tobytes()


class TestTheMicrophoneThisWasMeasuredOn:
    """The trap the obvious implementation walks into.

    "A hardware mute emits digital silence" is what every datasheet implies
    and what this code was first written against. On the microphone actually
    attached to this machine it is false, so a test for exact zero would have
    shipped, passed, and detected a mute exactly never.
    """

    def test_a_muted_mv6_is_recognised_as_no_signal(self, monkeypatch, tmp_path):
        seen = silence_runs(monkeypatch, tmp_path, [dithered_silence()] * 10)
        assert seen[-1] == pytest.approx(10 * speech._CHUNK_S), \
            "the measured muted signal was read as a live room"

    def test_the_measured_mute_level_clears_the_threshold(self):
        # Muted maxes at 0.5358. Anything at or below the threshold counts as
        # dead input, so the threshold has to sit above that with room.
        assert speech._SILENT_RMS > 0.5358

    def test_an_exact_zero_test_would_not_have_worked(self):
        # Pinned as an assertion rather than a comment because the tempting
        # edit is `loudness == 0`, and it would look correct in review.
        import array as _array
        samples = _array.array("h", dithered_silence())
        assert any(sample != 0 for sample in samples)
        assert max(abs(sample) for sample in samples) == 1

    def test_it_still_sits_far_below_a_live_room(self):
        # 1.9x over the mute, 24x under room tone. Both margins matter: the
        # first stops a mute being missed, the second stops a working room
        # being called dead.
        assert speech._SILENT_RMS < 13 / 10


class TestWhatCountsAsNoSignal:
    def test_digital_silence_accumulates(self, monkeypatch, tmp_path):
        seen = silence_runs(monkeypatch, tmp_path, [SILENT] * 10)
        assert seen[-1] == pytest.approx(10 * speech._CHUNK_S)

    def test_a_quiet_room_never_registers_as_silence(self, monkeypatch, tmp_path):
        # The failure that would matter: MUTED on the panel of a working
        # assistant, and her face switched off, because nobody was talking.
        seen = silence_runs(monkeypatch, tmp_path, [ROOM] * 40)
        assert set(seen) == {0.0}

    def test_even_a_single_count_of_signal_breaks_the_run(self, monkeypatch, tmp_path):
        seen = silence_runs(monkeypatch, tmp_path, [chunk(2)] * 10)
        assert set(seen) == {0.0}

    def test_signal_returning_resets_the_run(self, monkeypatch, tmp_path):
        seen = silence_runs(monkeypatch, tmp_path,
                            [SILENT] * 5 + [ROOM] + [SILENT] * 2)
        assert seen[4] == pytest.approx(5 * speech._CHUNK_S)
        assert seen[5] == 0.0                       # the room-tone chunk
        assert seen[6] == pytest.approx(speech._CHUNK_S)

    def test_the_threshold_leaves_a_wide_margin_under_room_tone(self):
        # Deliberately far below what a room produces rather than just below
        # it. Being wrong low means nothing is ever shown, which is exactly
        # today's behaviour; being wrong high breaks a working assistant.
        assert speech._SILENT_RMS < 13 / 10

    def test_a_mute_produces_no_capture_to_transcribe(self, monkeypatch, tmp_path):
        monkeypatch.setattr(speech.subprocess, "Popen",
                            lambda *a, **k: Stream([SILENT] * 20))
        monkeypatch.setattr(speech, "_DETECTOR", None)
        assert speech.record_until_silence(tmp_path / "heard.wav",
                                           lead_in=None) is None


class TestTheDetectorIsNotRunOnZeroes:
    def test_it_is_skipped_once_the_silence_is_a_mute(self, monkeypatch, tmp_path):
        # Silero is the per-chunk cost of this loop. Asking it thirty times a
        # second whether a stream of zeroes contains speech is the work that
        # sleeping exists to stop, and it is pure waste for the hours a
        # microphone may sit muted.
        detector = CountingDetector()
        needed = int(speech.MUTE_AFTER_S / speech._CHUNK_S)
        silence_runs(monkeypatch, tmp_path, [SILENT] * (needed + 30), detector)
        assert detector.calls <= needed + 1, \
            "the detector kept running after the input was known to be dead"

    def test_it_still_runs_on_a_room_that_is_merely_quiet(self, monkeypatch, tmp_path):
        detector = CountingDetector()
        silence_runs(monkeypatch, tmp_path, [ROOM] * 20, detector)
        assert detector.calls == 20

    def test_it_is_reset_when_signal_returns(self, monkeypatch, tmp_path):
        # Silero carries RNN state between calls and has just been handed
        # however long the mute lasted. Starting it clean stops the silence
        # colouring its reading of the first words after it.
        detector = CountingDetector()
        needed = int(speech.MUTE_AFTER_S / speech._CHUNK_S)
        before = detector.resets
        silence_runs(monkeypatch, tmp_path,
                     [SILENT] * (needed + 5) + [ROOM] * 3, detector)
        assert detector.resets > before + 1, \
            "the detector was not restarted after the mute came off"


# Captured verbatim from the MV6 on the Pi, in both positions. The second
# boolean is why this is kept in full rather than trimmed to the line under
# test: 'USB Streaming IN Playback Switch' is the microphone's headphone
# output and it reads `on` whether or not the capture is muted. Anything that
# looks for "the boolean control" instead of the capture one reads it, and
# reports a muted microphone as live every single time.
MV6_MUTED = """\
numid=3,iface=MIXER,name='Microphone Capture Switch'
  ; type=BOOLEAN,access=rw------,values=1
  : values=off
numid=4,iface=MIXER,name='Microphone Capture Volume'
  ; type=INTEGER,access=rw---R--,values=1,min=0,max=72,step=0
  : values=60
  | dBminmax-min=0.00dB,max=36.00dB
numid=5,iface=MIXER,name='USB Streaming IN Playback Switch'
  ; type=BOOLEAN,access=rw------,values=1
  : values=on
numid=1,iface=PCM,name='Capture Channel Map'
  ; type=INTEGER,access=r--v-R--,values=1,min=0,max=36,step=0
  : values=3
"""
MV6_LIVE = MV6_MUTED.replace(
    "name='Microphone Capture Switch'\n  ; type=BOOLEAN,access=rw------,"
    "values=1\n  : values=off",
    "name='Microphone Capture Switch'\n  ; type=BOOLEAN,access=rw------,"
    "values=1\n  : values=on")


@pytest.fixture(autouse=True)
def _fresh_mute_cache(monkeypatch):
    """The poll cache is module state and would leak between tests."""
    monkeypatch.setattr(speech, "_mute_state", [0.0, None])


def amixer_returning(text, code=0):
    import subprocess as sp

    def run(argv, **kwargs):
        return sp.CompletedProcess(argv, code, stdout=text, stderr="")

    return run


class TestAskingTheMicrophoneItself:
    """The MV6 drives an ALSA capture switch from its touch panel.

    Verified by reading it in both positions on the real device. Worth
    preferring over the signal because the signal cannot tell a muted
    microphone from a broken one, and that difference is a fault report.
    """

    def test_a_muted_microphone_says_so(self, monkeypatch):
        monkeypatch.setattr(speech.subprocess, "run",
                            amixer_returning(MV6_MUTED))
        assert speech.mic_muted() is True

    def test_a_live_microphone_says_so(self, monkeypatch):
        monkeypatch.setattr(speech.subprocess, "run",
                            amixer_returning(MV6_LIVE))
        assert speech.mic_muted() is False

    def test_the_headphone_switch_is_not_mistaken_for_the_capture_switch(
            self, monkeypatch):
        # The trap. 'USB Streaming IN Playback Switch' reads `on` in both
        # dumps above, so a parser that takes the first BOOLEAN, or the last
        # one, or any one that is not matched by name, reports a muted
        # microphone as live and the whole feature never fires.
        monkeypatch.setattr(speech.subprocess, "run",
                            amixer_returning(MV6_MUTED))
        assert speech.mic_muted() is True, \
            "it read the headphone output instead of the capture switch"

    def test_a_device_with_no_capture_switch_has_no_opinion(self, monkeypatch):
        monkeypatch.setattr(speech.subprocess, "run", amixer_returning(
            "numid=1,iface=PCM,name='Capture Channel Map'\n  : values=3\n"))
        assert speech.mic_muted() is None

    def test_a_missing_amixer_has_no_opinion(self, monkeypatch):
        def explode(*a, **k):
            raise FileNotFoundError("amixer")

        monkeypatch.setattr(speech.subprocess, "run", explode)
        assert speech.mic_muted() is None

    def test_a_failing_amixer_has_no_opinion(self, monkeypatch):
        monkeypatch.setattr(speech.subprocess, "run",
                            amixer_returning("", code=1))
        assert speech.mic_muted() is None

    def test_a_wedged_amixer_has_no_opinion(self, monkeypatch):
        import subprocess as sp

        def hang(*a, **k):
            raise sp.TimeoutExpired("amixer", 3)

        monkeypatch.setattr(speech.subprocess, "run", hang)
        assert speech.mic_muted() is None

    @pytest.mark.parametrize("values, muted", [
        ("off", True), ("on", False), ("off,off", True),
        ("on,on", False), ("on,off", False),
    ])
    def test_every_channel_has_to_be_off_to_count(self, monkeypatch, values,
                                                  muted):
        # A half-muted stereo pair is still delivering something.
        monkeypatch.setattr(speech.subprocess, "run", amixer_returning(
            "numid=3,iface=MIXER,name='Microphone Capture Switch'\n"
            "  ; type=BOOLEAN,access=rw------,values=2\n"
            f"  : values={values}\n"))
        assert speech.mic_muted() is muted


class TestItIsNotAskedThirtyTimesASecond:
    def test_repeated_calls_poll_once(self, monkeypatch):
        # The caller is the capture loop. One subprocess a second is free;
        # thirty is a measurable fraction of a core on a Pi.
        calls = []

        def counting(argv, **kwargs):
            import subprocess as sp
            calls.append(argv)
            return sp.CompletedProcess(argv, 0, stdout=MV6_MUTED, stderr="")

        monkeypatch.setattr(speech.subprocess, "run", counting)
        for tick in range(30):
            speech.mic_muted(now=100.0 + tick * 0.03)
        assert len(calls) == 1

    def test_the_answer_does_not_go_stale(self, monkeypatch):
        monkeypatch.setattr(speech.subprocess, "run",
                            amixer_returning(MV6_MUTED))
        assert speech.mic_muted(now=100.0) is True
        monkeypatch.setattr(speech.subprocess, "run",
                            amixer_returning(MV6_LIVE))
        assert speech.mic_muted(now=100.0 + speech._MUTE_POLL_S + 0.01) is False


class TestFindingTheCard:
    def test_the_configured_device_name(self, monkeypatch):
        monkeypatch.setattr(speech.config, "MIC_DEVICE",
                            "plughw:CARD=MV6,DEV=0")
        assert speech._mic_card() == "MV6"

    def test_a_numeric_device(self, monkeypatch):
        monkeypatch.setattr(speech.config, "MIC_DEVICE", "hw:1,0")
        assert speech._mic_card() == "1"

    def test_a_plain_default(self, monkeypatch):
        monkeypatch.setattr(speech.config, "MIC_DEVICE", "default")
        assert speech._mic_card() == "0"


class TestADeadMicrophoneIsNotAMutedOne:
    """The distinction the signal alone cannot make.

    Both deliver nothing. Calling a broken microphone "muted" is the failure
    this project has already been bitten by twice — perfectly healthy panel,
    stone deaf — and going to sleep on it would make it permanent.
    """

    def test_a_live_switch_with_no_signal_is_a_fault(self):
        assert main._silence_state(60.0, muted=False) == main.DEAF

    def test_she_does_not_sleep_through_a_fault(self):
        # Sleeping would blank the panel and remove the only sign anything is
        # wrong, on a machine in a cupboard with nobody reading its journal.
        assert main._silence_state(main.SLEEP_AFTER_S * 10, muted=False) \
            == main.DEAF

    def test_a_muted_switch_is_a_choice_and_does_sleep(self):
        assert main._silence_state(main.SLEEP_AFTER_S, muted=True) == main.ASLEEP

    def test_a_brief_gap_is_still_just_a_gap(self):
        assert main._silence_state(0.5, muted=False) == main.LIVE

    def test_the_switch_beats_the_signal_when_they_disagree(self):
        # Muted, but the converter is dithering above the threshold, or the
        # room is loud enough to leak in. The device knows better.
        assert main._silence_state(0.0, muted=True) == main.MUTED

    def test_a_device_with_no_opinion_falls_back_to_the_signal(self):
        assert main._silence_state(0.0, muted=None) == main.LIVE
        assert main._silence_state(speech.MUTE_AFTER_S, muted=None) == main.MUTED
        assert main._silence_state(main.SLEEP_AFTER_S, muted=None) == main.ASLEEP


class TestWhatALongSilenceMeans:
    def test_a_live_room_is_live(self):
        assert main._silence_state(0.0) == main.LIVE

    def test_a_pause_between_words_is_not_a_mute(self):
        assert main._silence_state(speech.MUTE_AFTER_S * 0.5) == main.LIVE

    def test_a_sustained_dead_input_is_a_mute(self):
        assert main._silence_state(speech.MUTE_AFTER_S) == main.MUTED
        assert main._silence_state(speech.MUTE_AFTER_S * 10) == main.MUTED

    def test_a_very_long_one_is_sleep(self):
        assert main._silence_state(main.SLEEP_AFTER_S) == main.ASLEEP

    def test_muting_is_visible_long_before_she_sleeps(self):
        # Reaching for the mute button and looking up should show the answer
        # already there; the panel going dark is a separate, much later thing.
        assert speech.MUTE_AFTER_S < main.SLEEP_AFTER_S

    def test_sleep_cannot_be_reached_by_a_lull_in_a_conversation(self):
        # She must never blank her own face because somebody stopped to think.
        # The only thing that reaches this threshold is no signal at all.
        assert main.SLEEP_AFTER_S >= 300


class FakeOrb:
    def __init__(self):
        self.started = 0
        self.stopped = 0
        self.states: list[tuple] = []

    def start(self): self.started += 1
    def stop(self): self.stopped += 1
    def set_state(self, name, level=0.0, note=""): self.states.append((name, note))


@pytest.fixture
def panel(monkeypatch):
    """A display holding a fake face, rebuilt without importing a renderer."""
    display = main._Display(enabled=False)
    display.enabled = True
    display.orb = FakeOrb()

    return display


class TestSleepingAndWaking:
    """She keeps drawing while asleep, and that is the whole point.

    The first version stopped the renderer and blanked /dev/fb0, which was
    wrong for exactly the reason _silence_state is careful about a fault: a
    dark panel cannot be told apart from a crash, a dead backlight or a pulled
    plug. Blanking her to save a fifth of a core solved a problem nobody had
    at the cost of the one thing the panel is for.
    """

    def test_sleeping_does_not_stop_the_face(self, panel):
        panel.sleep()
        assert panel.orb.stopped == 0, "she went dark instead of going to sleep"
        assert panel.asleep is True

    def test_the_face_is_told_she_is_asleep(self, panel):
        panel.sleep()
        assert panel.orb.states[-1][0] == "asleep"

    def test_the_face_is_still_there_to_draw_it(self, panel):
        panel.sleep()
        assert panel.orb is not None

    def test_sleeping_twice_says_so_once(self, panel):
        panel.sleep()
        before = len(panel.orb.states)
        panel.sleep()
        assert len(panel.orb.states) == before

    def test_waking_clears_it(self, panel):
        panel.sleep()
        panel.wake()
        assert panel.asleep is False
        assert panel.orb is not None

    def test_waking_while_awake_does_nothing(self, panel):
        orb = panel.orb
        panel.wake()
        assert panel.orb is orb

    def test_nothing_has_to_be_rebuilt_to_come_back(self, panel):
        # The old wake() restarted the renderer from inside the capture loop,
        # which shells out to `sudo -n chvt` with a five second timeout and
        # would have stalled the microphone through the first words after the
        # mute came off. Not rebuilding is what removed that entirely.
        orb = panel.orb
        panel.sleep()
        panel.wake()
        assert panel.orb is orb, "she was rebuilt; the console-switch race is back"

    def test_a_display_that_was_never_fitted_survives_both(self):
        headless = main._Display(enabled=False)
        headless.sleep()
        headless.wake()
        headless.close()

    def test_closing_while_asleep_is_safe(self, panel):
        panel.sleep()
        panel.close()
