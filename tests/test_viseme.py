"""Laying a sentence's syllables across the audio that speaks it.

There is no phoneme alignment: Kokoro returns a sentence whose duration is
known exactly and the text that produced it, so the syllables are distributed
across that duration by weight. What is tested here is that the distribution
is sound — ordered, bounded, gapped — and that the two layers stay separate,
because the mouth follows the sound while the brows follow the meaning.
"""
from __future__ import annotations

import pytest

from eve import viseme


class TestSyllables:
    def test_a_word_yields_one_entry_per_vowel_group(self):
        assert len(viseme.syllables("hello")) == 2
        assert len(viseme.syllables("cat")) == 1
        assert len(viseme.syllables("laboratory")) == 5

    def test_a_silent_trailing_e_does_not_earn_its_own_syllable(self):
        assert len(viseme.syllables("time")) == 1
        assert len(viseme.syllables("make")) == 1

    def test_a_word_with_no_vowels_still_gets_a_shape(self):
        # "hmm", "shh". Returning nothing would freeze the mouth open on
        # whatever was there before.
        assert len(viseme.syllables("hmm")) == 1

    def test_punctuation_and_case_do_not_become_syllables(self):
        assert len(viseme.syllables("Yes!")) == 1
        assert viseme.syllables("...") == []

    @pytest.mark.parametrize("word, shape", [
        ("father", "A"), ("see", "E"), ("bit", "I"),
        ("go", "O"), ("put", "U"),
    ])
    def test_the_vowel_decides_the_shape(self, word, shape):
        assert viseme.syllables(word)[0]["shape"] == shape

    def test_a_word_opening_on_a_plosive_shuts_the_lips_first(self):
        # p, b, m. The closure is what makes "boot" read as "boot" rather
        # than as a mouth that was already open.
        assert viseme.syllables("boot")[0]["closes"] is True
        assert viseme.syllables("open")[0]["closes"] is False

    def test_only_the_first_syllable_of_a_plosive_word_closes(self):
        parts = viseme.syllables("banana")
        assert parts[0]["closes"] is True
        assert all(not part["closes"] for part in parts[1:])

    def test_the_first_syllable_of_a_long_word_takes_more_of_the_sentence(
            self):
        # Otherwise the mouth moves like a metronome, which reads as a hinge.
        parts = viseme.syllables("hello")
        assert parts[0]["weight"] > parts[1]["weight"]
        assert parts[0]["stress"] is True


class TestSemanticCues:
    def test_the_dry_register_she_actually_writes_in_is_marked(self):
        # The dry marker fires most often, which is the point of it.
        assert viseme.syllables("Ideal.")[0]["cue"] == "dry"
        assert viseme.syllables("naturally")[0]["cue"] == "dry"

    @pytest.mark.parametrize("word, cue", [
        ("assuming", "hedge"), ("never", "negation"),
        ("extremely", "intensifier"), ("seventy", "number"),
    ])
    def test_each_word_class_worth_reacting_to_is_recognised(self, word, cue):
        assert viseme.syllables(word)[0]["cue"] == cue

    def test_an_ordinary_word_carries_no_cue(self):
        assert viseme.syllables("weather")[0]["cue"] is None

    def test_a_comma_marks_the_syllable_before_it_as_a_clause_end(self):
        parts = viseme.syllables("Overcast, seventy five degrees.")
        assert parts[2]["clause"] is True

    def test_the_last_syllable_of_a_sentence_is_marked_as_a_stop(self):
        parts = viseme.syllables("Oslo.")
        assert parts[-1]["stop"] is True

    def test_a_question_colours_its_whole_clause_not_just_the_mark(self):
        # The brows lift across the phrase leading up to it. Marking only the
        # syllable the '?' lands on gives a twitch at the end instead. The
        # colouring stops at the previous sentence's full stop, so the
        # statement in front of the question keeps its own reading.
        parts = viseme.syllables("Oslo. Want the longer version?")
        assert parts[-1]["question"] is True
        assert sum(part["question"] for part in parts) > 1
        # "Oslo." is a separate sentence and is left alone.
        assert parts[0]["question"] is False
        assert parts[1]["question"] is False


class TestTheTimeline:
    def test_syllables_fill_the_audio_and_no_more(self):
        mouths = viseme.timeline("Overcast and seventy five degrees.", 3.0)
        assert mouths[0].start == 0.0
        assert mouths[-1].end <= 3.0 + 1e-9

    def test_an_offset_shifts_the_whole_sentence(self):
        # Sentences are synthesised one at a time and laid end to end, so
        # every sentence after the first arrives with an offset.
        mouths = viseme.timeline("Hello there.", 2.0, offset=5.0)
        assert mouths[0].start == pytest.approx(5.0)
        assert mouths[-1].end <= 7.0 + 1e-9

    def test_syllables_run_in_order_and_never_overlap(self):
        mouths = viseme.timeline("The capital of Norway is Oslo.", 4.0)
        # strict=False: the offset pairing is n-1 pairs by construction.
        for earlier, later in zip(mouths, mouths[1:], strict=False):
            assert earlier.end <= later.start + 1e-9

    def test_silence_is_left_between_syllables(self):
        # A mouth that never closes reads as a hinge, not a mouth. Nine
        # tenths sounding, a tenth closed.
        mouths = viseme.timeline("hello there friend", 3.0)
        gaps = [later.start - earlier.end
                for earlier, later in zip(mouths, mouths[1:], strict=False)]
        assert all(gap > 0 for gap in gaps)

    @pytest.mark.parametrize("text, seconds", [
        ("", 3.0), ("...", 3.0), ("hello", 0.0), ("hello", -1.0),
    ])
    def test_nothing_to_say_or_no_time_to_say_it_yields_no_mouths(
            self, text, seconds):
        assert viseme.timeline(text, seconds) == []


class TestLookingUpAMoment:
    def test_the_shape_sounding_now_is_the_one_returned(self):
        mouths = viseme.timeline("father", 1.0)
        shape, phase = viseme.at(mouths, 0.5)
        assert shape in viseme.SHAPES
        assert 0.0 <= phase <= 1.0

    def test_between_syllables_the_mouth_is_closed(self):
        # The whole reason this is a lookup rather than an index.
        mouths = viseme.timeline("hello there", 2.0)
        shape, _ = viseme.at(mouths, mouths[0].end + 1e-4)
        assert shape == "M"

    def test_after_the_last_syllable_the_mouth_is_closed(self):
        mouths = viseme.timeline("hello", 1.0)
        assert viseme.at(mouths, 99.0)[0] == "M"

    def test_a_plosive_shuts_the_lips_before_it_opens(self):
        # The lips are shut for the first third of a plosive-initial
        # syllable, then open on the vowel — "put" is closed, then U.
        mouths = viseme.timeline("put", 1.0)
        first = mouths[0]
        early = first.start + (first.end - first.start) * 0.1
        late = first.start + (first.end - first.start) * 0.8
        assert viseme.at(mouths, early)[0] == "M"
        assert viseme.at(mouths, late)[0] == "U"

    def test_every_shape_the_timeline_can_emit_has_a_size(self):
        mouths = viseme.timeline(
            "The father can see a bit and go to boot.", 4.0)
        for mouth in mouths:
            assert mouth.shape in viseme.SHAPES

    def test_the_beat_carries_the_semantic_layer_with_it(self):
        # `at` gives the mouth its shape; `beat` is what the brows read.
        mouths = viseme.timeline("Ideal.", 1.0)
        assert viseme.beat(mouths, 0.5).cue == "dry"

    def test_there_is_no_beat_before_the_first_syllable_starts(self):
        mouths = viseme.timeline("hello", 1.0, offset=2.0)
        assert viseme.beat(mouths, 0.0) is None
