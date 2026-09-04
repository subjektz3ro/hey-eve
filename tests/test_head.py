"""The behaviour half: what she does, with no drawing in it.

eve/head.py is everything she *is* — the state she is in, how she reacts to
being spoken to, when she blinks, where she looks, and the thread that turns
all of it into frames. eve/void.py is the only thing that decides what any of
it looks like. That split survived the three faces it was built for.

Most of what is pinned here is the **semantic layer**, and it is the part of
her face that is least obvious from the outside. A face moves because of
meaning, not loudness: viseme.py hands over each syllable already labelled —
a hedge, a number, a question, the last word before a comma — and this is
where those labels become an expression, *before* the phonetic layer scales
anything. Two sentences of identical volume should not produce identical eyes,
and this file is what says so.
"""
from __future__ import annotations

import numpy as np
import pytest

from eve import head, viseme


class Bare(head.Head):
    """The behaviour with a drawing that does nothing."""

    def __init__(self):
        super().__init__()
        self.frames = 0

    def _render(self, now: float, dt: float) -> None:
        self.frames += 1


def speaking(text: str, seconds: float = 2.0):
    """A face mid-reply, with `text` laid across `seconds` of audio."""
    face = Bare()
    face.set_speech(viseme.timeline(text, seconds))
    face.set_state("speaking", 0.4)
    return face


def eyes_at(face, when: float) -> dict:
    face.set_audio_position(when, 0.4)
    return face._target("speaking", 0.4, 0.0)


def cue_moment(face, cue: str) -> float | None:
    """When the first syllable carrying `cue` is sounding."""
    for mouth in face._mouths:
        if mouth.cue == cue:
            return (mouth.start + mouth.end) / 2
    return None


class TestWhatTheConversationTellsHer:
    def test_a_state_she_does_not_know_reads_as_idle(self):
        # main.py names states as strings, so a typo must not put the face
        # into an undrawable one.
        face = Bare()
        face.set_state("interpretive dance")
        assert face._state == "idle"

    def test_the_level_is_clamped_to_something_drawable(self):
        face = Bare()
        face.set_state("listening", 9.0)
        assert face._level == 1.0
        face.set_state("listening", -3.0)
        assert face._level == 0.0

    def test_leaving_speech_drops_the_timeline(self):
        # Otherwise the mouth keeps mouthing a reply that finished, against an
        # audio position that is no longer advancing.
        face = speaking("Hello there.")
        assert face._mouths
        face.set_state("idle")
        assert face._mouths == []
        assert face._audio_at == 0.0

    def test_synthesising_carries_its_progress(self):
        face = Bare()
        face.set_state("synthesizing", 0.63)
        assert face._progress == pytest.approx(0.63)

    def test_the_waveform_is_taken_without_the_caller_waiting(self):
        # speech.py hands this over from the pacer thread on a 30ms clock.
        face = Bare()
        face.set_wave(np.zeros(512, np.int16))
        face.set_audio_position(1.25, 0.5)
        assert face._audio_at == pytest.approx(1.25)


class TestSheReactsToMeaningAndNotOnlyVolume:
    """The semantic layer, which is the half of the face nobody can see.

    Each cue bends the brow and the eye *before* the phonetic layer scales
    them, so the same loudness produces different eyes depending on what is
    being said. Every assertion below is against head.REST, the resting pose,
    so what is being pinned is the direction of each expression rather than a
    number that means nothing on its own.
    """

    def test_a_dry_aside_narrows_her_and_lifts_one_corner(self):
        face = speaking("Ideal, obviously, assuming you enjoy that.")
        when = cue_moment(face, "dry")
        assert when is not None, "no dry cue in a sentence written to have one"
        eyes = eyes_at(face, when)
        assert eyes["flick"] > head.REST["flick"]
        assert eyes["asym"] > head.REST["asym"]

    def test_a_hedge_cocks_her_head(self):
        face = speaking("It works, assuming you agree.")
        when = cue_moment(face, "hedge")
        assert when is not None
        eyes = eyes_at(face, when)
        assert eyes["tilt"] != head.REST["tilt"]
        assert eyes["asym"] > head.REST["asym"]

    def test_a_negation_closes_her_down(self):
        face = speaking("No, that never works.")
        when = cue_moment(face, "negation")
        assert when is not None
        eyes = eyes_at(face, when)
        assert eyes["flick"] < head.REST["flick"]

    def test_an_intensifier_opens_her_up(self):
        # The one cue that goes the other way, which is what makes the rest
        # read as understatement.
        face = speaking("That is extremely bad.")
        when = cue_moment(face, "intensifier")
        assert when is not None
        assert eyes_at(face, when)["flick"] > head.REST["flick"]

    def test_a_number_gets_no_expression_at_all(self):
        # Facts are delivered flat. An assistant that emotes over a
        # temperature is doing something else.
        face = speaking("It is seventy five degrees.")
        when = cue_moment(face, "number")
        assert when is not None
        assert eyes_at(face, when)["asym"] == 0.0

    def test_a_question_widens_her(self):
        """And it colours the whole clause, not just the last word.

        Deliberately sampled away from the final syllable: that one also
        carries `stop`, which resets her to rest *after* the question has
        widened her, so the one place the question is invisible is the word
        the question mark is attached to.
        """
        face = speaking("Do you want the longer version?")
        asked = [m for m in face._mouths if m.question and not m.stop]
        assert asked, "the timeline did not mark the question"
        eyes = eyes_at(face, (asked[0].start + asked[0].end) / 2)
        assert eyes["flick"] > head.REST["flick"]

    def test_a_full_stop_returns_her_to_rest(self):
        """The sentence is over; whatever it was doing to her face is too.

        `flick` is excluded on purpose rather than by accident: `stress` adds
        to it *after* the reset, so a stressed sentence-final syllable lands a
        fraction off rest. That ordering is deliberate — the emphasis belongs
        to the word, and the expression belonged to the sentence that just
        ended.
        """
        face = speaking("Oslo. I would have assumed you knew that.")
        stops = [m for m in face._mouths if m.stop]
        assert stops, "no sentence end in a sentence with two of them"
        eyes = eyes_at(face, (stops[0].start + stops[0].end) / 2)
        for key in ("asym", "tilt"):
            assert eyes[key] == pytest.approx(head.REST[key])

    def test_the_same_words_at_the_same_volume_are_not_one_expression(self):
        # The whole claim of the semantic layer in one assertion.
        dry = speaking("Ideal, obviously, assuming you enjoy that.")
        flat = speaking("It is seventy five degrees.")
        a = eyes_at(dry, cue_moment(dry, "dry"))
        b = eyes_at(flat, cue_moment(flat, "number"))
        assert a != b


class TestThePhoneticLayerUnderneath:
    def test_a_vowel_squeezes_the_eyes(self):
        # There is no mouth, so the syllable has to read through them.
        face = speaking("Hello there.")
        heights = {eyes_at(face, m.start + 0.01)["h"] for m in face._mouths}
        assert len(heights) > 1, "every syllable produced the same eye"

    def test_nothing_sounds_once_the_reply_has_finished(self):
        # A syllable stays current for 120ms past its end, so the mouth does
        # not snap shut between words. Past that, nothing is sounding.
        face = speaking("Oslo.", seconds=4.0)
        assert viseme.beat(face._mouths, face._mouths[-1].end + 0.5) is None


class TestTheOtherStates:
    def test_attending_opens_her_and_lifts_her(self):
        face = Bare()
        eyes = face._target("listening", 0.0, 0.0)
        assert eyes["h"] > head.REST["h"]
        assert eyes["rise"] < head.REST["rise"]      # up is negative here

    def test_working_narrows_and_cocks_her(self):
        face = Bare()
        for state in ("thinking", "searching", "synthesizing", "transcribing"):
            eyes = face._target(state, 0.0, 0.0)
            assert eyes["h"] < head.REST["h"], state
            assert eyes["tilt"] != head.REST["tilt"], state

    def test_idle_drifts_rather_than_sitting_still(self):
        face = Bare()
        rises = {face._target("idle", 0.0, t)["rise"] for t in (0.0, 1.0, 2.0)}
        assert len(rises) == 3


class TestTheBlink:
    def test_she_blinks(self):
        # Sampled at 5ms. A blink is fully shut for only 40 of its 260
        # milliseconds, so a coarser sample catches the ramp and reports that
        # she never quite closed her eyes.
        face = Bare()
        assert max(face._blink(t / 200, 0.0) for t in range(4000)) \
            == pytest.approx(1.0)

    def test_she_does_not_blink_while_being_spoken_to_loudly(self):
        # A blink lands on the one moment somebody is watching for a reaction.
        face = Bare()
        assert all(face._blink(t / 20, 0.9) == 0.0 for t in range(400))

    def test_a_blink_is_faster_than_a_frame_budget_allows_to_be_missed(self):
        face = Bare()
        face._blink_at, face._blink_next = 0.0, 1e9
        assert face._blink(0.0, 0.0) == 0.0
        assert face._blink(0.075, 0.0) == 1.0        # shut
        assert face._blink(0.30, 0.0) == 0.0         # and open again


class TestTheFrameLoop:
    def test_the_base_class_draws_nothing_on_its_own(self):
        with pytest.raises(NotImplementedError):
            head.Head()._render(0.0, 0.02)

    def test_starting_twice_runs_one_thread(self, monkeypatch):
        monkeypatch.setattr(head.subprocess, "run", lambda *a, **k: None)
        face = Bare()
        try:
            face.start()
            first = face._thread
            face.start()
            assert face._thread is first
        finally:
            face.stop()

    def test_it_actually_draws(self, monkeypatch):
        import time
        monkeypatch.setattr(head.subprocess, "run", lambda *a, **k: None)
        face = Bare()
        try:
            face.start()
            time.sleep(0.15)
        finally:
            face.stop()
        assert face.frames > 1

    def test_stopping_blanks_the_panel(self, tmp_path, monkeypatch):
        # Otherwise the last frame she drew stays burned on the display for as
        # long as the machine is up, which reads as a crash rather than a stop.
        device = tmp_path / "fb0"
        device.write_bytes(b"\xff" * (head.FB_WIDTH * head.FB_HEIGHT * 2))
        monkeypatch.setattr(head, "FB", str(device))
        monkeypatch.setattr(head.subprocess, "run", lambda *a, **k: None)
        Bare().stop()
        assert set(device.read_bytes()) == {0}

    def test_a_vanished_panel_does_not_raise_on_the_way_out(self, monkeypatch):
        monkeypatch.setattr(head, "FB", "/nonexistent/fb0")
        Bare().stop()
