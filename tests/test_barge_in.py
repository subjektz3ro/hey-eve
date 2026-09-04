"""Being able to stop her.

The latency problem has two answers and only one of them is engineering.
Kokoro can be made faster; a reply can also be made escapable. Today a wrong
or over-long answer is a trap — it is going to finish, and the only way out is
to sit through it — so an assistant that is slow is also an assistant you
cannot correct. Saying "eve—" and having her stop turns that from a hostage
situation into a pause, and it is worth more than the seconds it saves,
because it gives back the two sentences that had not been synthesised yet.

The obstacle is that her microphone hears her own voice through the speakers,
and Silero calls that speech because it is speech. What makes it tractable is
that she knows exactly what she is playing, so the question asked of each
chunk is not "is there sound" but "is there more sound than I am making".

The unknown is the gain from her output, through the amplifier, across the
room, into the capsule — a property of a room nobody can measure from here.
So the single most important test in this file is the first one: that all of
this is off until somebody has calibrated it on real hardware.
"""
from __future__ import annotations

import subprocess
import threading
import time

import pytest

from eve import speech
from tests.test_speech_lifecycle import FakePlayer


class TestItIsOffUntilSomebodyMeasuresIt:
    """The regression guard, and the reason this can ship tonight.

    Turning it on untuned cuts her off mid-sentence at random, which is a
    worse assistant than a slow one. Everything below it is machinery waiting
    for a number that has to come from the room.
    """

    def test_it_is_disabled_by_default(self):
        assert speech._BARGE_IN is False
        assert speech._BARGE_CALIBRATE is False

    def test_no_second_microphone_is_opened_while_she_speaks(self, monkeypatch):
        # The observable cost of being wrong here: a capture running for the
        # length of every reply, on every turn, on a Pi.
        opened = []

        def fake_popen(argv, *args, **kwargs):
            opened.append(argv[0])
            return FakePlayer()

        fake = type("S", (), {
            "Popen": staticmethod(fake_popen), "PIPE": -1, "DEVNULL": -3,
            "SubprocessError": subprocess.SubprocessError,
        })
        monkeypatch.setattr(speech, "subprocess", fake)
        monkeypatch.setattr(speech.tts, "synth", lambda text: b"\0\1" * 600)
        speech.speak("One sentence.", on_level=lambda level: None)
        assert "arecord" not in opened

    def test_the_threshold_starts_conservative(self):
        # A false negative costs the feature. A false positive cuts her off
        # mid-answer, which is the thing being fixed.
        assert speech._BARGE_RATIO >= 2.0
        assert speech._BARGE_FLOOR >= 200


class TestTellingAPersonFromHerself:
    def test_her_own_voice_is_not_an_interruption(self):
        # What comes back through the speakers arrives at roughly her own
        # level. Whatever that level is, the ratio stays near one.
        for level in (200.0, 800.0, 3000.0, 12000.0):
            assert speech._is_interruption(level, level) is False

    def test_her_own_voice_a_little_louder_is_still_not(self):
        # Room gain is not exactly one, and the margin exists for that.
        assert speech._is_interruption(1500.0, 1000.0) is False

    def test_somebody_talking_over_a_quiet_passage_is(self):
        assert speech._is_interruption(2000.0, 400.0) is True

    def test_room_tone_between_her_words_is_not(self):
        """The failure the absolute floor exists to stop.

        Between her words `own` falls to almost nothing, so the ratio against
        ordinary room tone goes to infinity. Without a floor every pause in
        every sentence reads as an interruption and she can never finish a
        reply at all.
        """
        assert speech._is_interruption(13.0, 0.0) is False
        assert speech._is_interruption(13.0, 1.0) is False

    def test_silence_against_silence_is_not(self):
        assert speech._is_interruption(0.0, 0.0) is False

    def test_a_loud_room_over_her_silence_is(self):
        # She is between sentences and somebody speaks at the desk.
        assert speech._is_interruption(2000.0, 0.0) is True

    def test_the_threshold_is_exclusive(self):
        # Exactly at the ratio is not over it; a boundary that triggers makes
        # the tuning advice in the calibration line off by one step.
        monkey = speech._BARGE_FLOOR
        assert speech._is_interruption(monkey * 10, monkey * 10 / speech._BARGE_RATIO) \
            is False

    def test_both_conditions_are_required(self, monkeypatch):
        monkeypatch.setattr(speech, "_BARGE_FLOOR", 300.0)
        monkeypatch.setattr(speech, "_BARGE_RATIO", 2.0)
        assert speech._is_interruption(299.0, 1.0) is False    # loud enough? no
        assert speech._is_interruption(400.0, 300.0) is False  # louder than her? no
        assert speech._is_interruption(400.0, 100.0) is True   # both


class FakeMic:
    """An arecord handing over a scripted room, then room tone forever.

    Deliberately never EOFs. A real capture does not stop because a script ran
    out, and returning b"" would let the listener exit before `finished` is
    set — which would pass the tests below for the wrong reason and hide the
    teardown races they exist to catch. It stops when speak() stops it.
    """

    def __init__(self, chunks, quiet):
        self.chunks = list(chunks)
        self.stdout = self
        self.terminated = False
        self._quiet = quiet

    def read(self, _n):
        return self.chunks.pop(0) if self.chunks else self._quiet

    def close(self): pass
    def terminate(self): self.terminated = True
    def wait(self, timeout=None): return 0
    def kill(self): pass


def settle():
    """Let the listener thread notice `finished` and run its teardown."""
    time.sleep(0.2)


@pytest.fixture
def stage(monkeypatch):
    """A fake player and a fake microphone, with a scriptable room."""
    made: dict = {"room": [], "synth_calls": []}

    def fake_popen(argv, *args, **kwargs):
        if argv[0] == "arecord":
            made["mic"] = FakeMic(made["room"], room(13, 1)[0])
            return made["mic"]
        made["player"] = FakePlayer()
        return made["player"]

    fake = type("S", (), {
        "Popen": staticmethod(fake_popen), "PIPE": -1, "DEVNULL": -3,
        "SubprocessError": subprocess.SubprocessError,
    })
    monkeypatch.setattr(speech, "subprocess", fake)
    monkeypatch.setattr(speech, "_DETECTOR", None)
    monkeypatch.setattr(speech, "_PLAYBACK_SETTLE_S", 0.0)
    # Open the player on the first sentence. Left alone, the buffering rule
    # holds it back until the last one — correct, and covered in
    # test_speech_pacing.py — but the listener only starts once there is
    # something to interrupt, so a real lead would mean nothing to test.
    monkeypatch.setattr(speech, "_stalls", lambda *a, **k: False)

    def slow_synth(text):
        made["synth_calls"].append(text)
        time.sleep(0.08)      # let the listener get several chunks in
        return b"\0\1" * 600

    monkeypatch.setattr(speech.tts, "synth", slow_synth)
    return made


def room(magnitude: int, count: int) -> list[bytes]:
    """`count` capture chunks whose RMS is `magnitude`."""
    import array
    from eve import config
    per = int(config.MIC_RATE * speech._CHUNK_S)
    return [array.array("h", [magnitude, -magnitude] * (per // 2)).tobytes()
            for _ in range(count)]


# Every sentence over 25 characters, or _sentences() merges them into their
# neighbour and there is only ever one call to count.
REPLY = ("This is the first sentence of a long reply. "
         "This is the second sentence of that reply. "
         "This is the third sentence of that reply. "
         "This is the fourth sentence of that reply. "
         "This is the fifth sentence of that reply.")


class TestStoppingHer:
    def test_somebody_talking_over_her_stops_the_reply(self, stage, monkeypatch):
        monkeypatch.setattr(speech, "_BARGE_IN", True)
        stage["room"][:] = room(20000, 200)      # loud, continuous, and hers
        speech.speak(REPLY, on_level=lambda level: None)      # is nearly silent
        assert len(stage["synth_calls"]) < 5, \
            "she synthesised the whole reply after being interrupted"

    def test_the_player_is_killed_rather_than_drained(self, stage, monkeypatch):
        monkeypatch.setattr(speech, "_BARGE_IN", True)
        stage["room"][:] = room(20000, 200)
        speech.speak(REPLY, on_level=lambda level: None)
        assert stage["player"].killed

    def test_a_quiet_room_lets_her_finish(self, stage, monkeypatch):
        monkeypatch.setattr(speech, "_BARGE_IN", True)
        stage["room"][:] = room(13, 200)         # nobody is saying anything
        speech.speak(REPLY, on_level=lambda level: None)
        assert len(stage["synth_calls"]) == 5

    def test_a_brief_bang_does_not_stop_her(self, stage, monkeypatch):
        # A door, a dropped mug. Loud and short; a person talking over her is
        # loud and continuous.
        monkeypatch.setattr(speech, "_BARGE_IN", True)
        stage["room"][:] = (room(20000, speech._BARGE_CHUNKS - 1)
                            + room(13, 200))
        speech.speak(REPLY, on_level=lambda level: None)
        assert len(stage["synth_calls"]) == 5

    def test_the_microphone_is_released_afterwards(self, stage, monkeypatch):
        monkeypatch.setattr(speech, "_BARGE_IN", True)
        stage["room"][:] = room(20000, 200)
        speech.speak(REPLY, on_level=lambda level: None)
        settle()
        assert stage["mic"].terminated

    def test_it_leaves_no_thread_behind(self, stage, monkeypatch):
        # The pacer leaked one thread per reply for months because its exit
        # test could never become true; a second thread on the same path
        # deserves the same suspicion.
        monkeypatch.setattr(speech, "_BARGE_IN", True)
        stage["room"][:] = room(20000, 200)
        before = {t for t in threading.enumerate()}
        speech.speak(REPLY, on_level=lambda level: None)
        settle()
        leaked = {t for t in threading.enumerate()} - before
        assert not [t for t in leaked if t.is_alive()], f"leaked {leaked}"


class TestWhatTheJournalSays:
    def test_an_interruption_is_not_reported_as_a_broken_speaker(
            self, stage, monkeypatch, capsys):
        """Killing the player produces a BrokenPipeError either way.

        The existing handler exists for a real failure that was invisible for
        months — a reboot leaving the Bluetooth speaker unconnected, so replies
        were thought, billed and never heard. Printing that line every time
        somebody talks over her would bury the case it was written for.
        """
        monkeypatch.setattr(speech, "_BARGE_IN", True)
        stage["room"][:] = room(20000, 200)
        speech.speak(REPLY, on_level=lambda level: None)
        printed = capsys.readouterr().err
        assert "is the speaker connected?" not in printed

    def test_an_interruption_is_reported_as_itself(self, stage, monkeypatch, capsys):
        monkeypatch.setattr(speech, "_BARGE_IN", True)
        stage["room"][:] = room(20000, 200)
        speech.speak(REPLY, on_level=lambda level: None)
        assert "talked over her" in capsys.readouterr().err

    def test_synthesis_is_not_blamed_for_a_stop_somebody_asked_for(
            self, stage, monkeypatch, capsys):
        # A killed player leaves the pacer with nothing arriving, which the
        # gap counter reads as starvation.
        monkeypatch.setattr(speech, "_BARGE_IN", True)
        stage["room"][:] = room(20000, 200)
        speech.speak(REPLY, on_level=lambda level: None)
        assert "fell behind playback" not in capsys.readouterr().err


class TestCalibration:
    def test_it_runs_the_whole_path_and_never_interrupts(
            self, stage, monkeypatch):
        # The point of the mode: measure what her own voice reaches in this
        # room without her being cut off while you do it.
        monkeypatch.setattr(speech, "_BARGE_CALIBRATE", True)
        stage["room"][:] = room(20000, 200)
        speech.speak(REPLY, on_level=lambda level: None)
        assert len(stage["synth_calls"]) == 5
        assert not stage["player"].killed

    def test_it_reports_the_number_the_threshold_has_to_clear(
            self, stage, monkeypatch, capsys):
        monkeypatch.setattr(speech, "_BARGE_CALIBRATE", True)
        stage["room"][:] = room(20000, 200)
        speech.speak(REPLY, on_level=lambda level: None)
        settle()          # the line is written by the listener's teardown
        printed = capsys.readouterr().err
        assert "room/own ratio" in printed
        assert str(speech._BARGE_RATIO) in printed


class TestItCannotCostAReply:
    def test_a_microphone_that_will_not_open_is_survived(
            self, monkeypatch, capsys):
        # She simply cannot be interrupted, which is today's behaviour.
        monkeypatch.setattr(speech, "_BARGE_IN", True)

        def fake_popen(argv, *args, **kwargs):
            if argv[0] == "arecord":
                raise OSError("no such device")
            return FakePlayer()

        fake = type("S", (), {
            "Popen": staticmethod(fake_popen), "PIPE": -1, "DEVNULL": -3,
            "SubprocessError": subprocess.SubprocessError,
        })
        monkeypatch.setattr(speech, "subprocess", fake)
        monkeypatch.setattr(speech, "_PLAYBACK_SETTLE_S", 0.0)
        monkeypatch.setattr(speech.tts, "synth", lambda text: b"\0\1" * 600)
        speech.speak("One sentence here.", on_level=lambda level: None)
        assert "is the speaker connected?" not in capsys.readouterr().err
