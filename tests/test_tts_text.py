"""Rewriting text the way a narrator would read it, and levelling the result.

Both steps matter more than they look. Neural voices are trained on prose and
stumble over "5:48"; and the speaker this ends up on is quiet, so a polite
-12dBFS is inaudible across a room.

Nothing here loads Kokoro — these are the two pure transformations either
side of it.
"""
from __future__ import annotations

import array

import pytest

from eve import tts


class TestSayingNumbersOutLoud:
    @pytest.mark.parametrize("written, spoken", [
        ("5:48", "five forty eight"),
        ("12:00", "twelve o'clock"),
        ("9:05", "nine oh five"),
        ("1:30", "one thirty"),
    ])
    def test_a_clock_time_becomes_the_words_a_person_says(
            self, written, spoken):
        assert tts.speakable(written) == spoken

    @pytest.mark.parametrize("written, spoken", [
        ("0", "zero"), ("7", "seven"), ("19", "nineteen"),
        ("74", "seventy four"), ("100", "one hundred"),
        ("365", "three hundred sixty five"), ("-4", "minus four"),
    ])
    def test_a_bare_number_becomes_words(self, written, spoken):
        assert tts.speakable(written) == spoken

    def test_a_trailing_point_zero_is_dropped_as_float_noise(self):
        # "56.0" is what a float prints, not what a measurement sounds like.
        assert tts.speakable("56.0") == "fifty six"

    def test_a_real_decimal_keeps_its_fraction(self):
        assert tts.speakable("70.5") == "seventy point five"
        assert tts.speakable("0.25") == "zero point two five"

    def test_a_decimal_point_does_not_become_a_full_stop(self):
        # The bug this function exists for: `-?\d+` matched "56" and "0"
        # separately in "56.0" and left the point behind, so the engine heard
        # "fifty six." and started a new sentence at "zero percent".
        assert "." not in tts.speakable("56.0 percent")

    def test_a_number_inside_a_sentence_is_rewritten_in_place(self):
        assert tts.speakable("Overcast and 75 degrees.") == \
            "Overcast and seventy five degrees."

    def test_a_number_past_what_anyone_says_aloud_is_left_to_the_engine(self):
        assert tts.speakable("31415") == "31415"

    def test_an_em_dash_becomes_a_comma_because_it_is_read_as_silence(self):
        assert tts.speakable("Ideal — assuming you enjoy disappointment.") == \
            "Ideal, assuming you enjoy disappointment."
        assert tts.speakable("one–two") == "one, two"

    def test_ordinary_prose_is_left_completely_alone(self):
        text = "The capital of Norway is Oslo."
        assert tts.speakable(text) == text


class TestLevellingTheAudio:
    def _pcm(self, samples):
        return array.array("h", samples).tobytes()

    def test_a_quiet_recording_is_lifted_close_to_full_scale(self):
        # Near full-scale for s16 is the point, not a mastering choice.
        quiet = self._pcm([100, -100, 50] * 100)
        out = array.array("h", tts._normalize(quiet, tts.RATE, 1))
        assert max(abs(s) for s in out) == pytest.approx(tts.TARGET_PEAK, rel=0.02)

    def test_audio_already_loud_enough_is_not_touched(self):
        loud = self._pcm([30000, -30000] * 50)
        assert tts._normalize(loud, tts.RATE, 1) == loud

    def test_stereo_is_reduced_to_the_left_channel(self):
        # Interleaved L,R,L,R — the right channel is discarded rather than
        # summed, because summing a correlated pair just clips.
        stereo = self._pcm([1000, -1000] * 64)
        out = array.array("h", tts._normalize(stereo, tts.RATE, 2))
        assert len(out) == 64
        assert all(sample > 0 for sample in out)

    def test_any_input_rate_comes_out_at_the_players_rate(self):
        # The caller streams this straight into an already-open aplay, so
        # every engine's output has to arrive at one rate.
        source = self._pcm([5000, -5000] * 1000)
        out = tts._normalize(source, 22050, 1)
        expected = int(2000 * tts.RATE / 22050)
        assert len(array.array("h", out)) == pytest.approx(expected, abs=2)

    def test_digital_silence_does_not_divide_by_zero(self):
        # A sentence that synthesises to nothing must not take the reply down
        # with it — the peak is zero and the gain would be infinite.
        silence = self._pcm([0] * 128)
        out = array.array("h", tts._normalize(silence, tts.RATE, 1))
        assert set(out) == {0}

    def test_the_target_peak_stays_below_clipping(self):
        assert tts.TARGET_PEAK < 32767


class TestChoosingAVoice:
    def test_the_voice_pattern_accepts_the_bank_naming_scheme(self):
        assert tts._VOICE_RE.fullmatch("bf_emma")
        assert tts._VOICE_RE.fullmatch("am_michael")

    def test_anything_that_is_not_a_voice_name_is_rejected(self):
        # synth() falls back to the default rather than handing Kokoro a
        # path, which is what turns a typo into a wrong voice instead of a
        # crash mid-reply.
        for bad in ("../../etc/passwd", "emma", "BF_EMMA", ""):
            assert tts._VOICE_RE.fullmatch(bad) is None


class TestFindingTheModel:
    def test_missing_weights_report_absence_rather_than_raising(
            self, tmp_path, monkeypatch):
        # A missing model should be a diagnosable startup message, not a
        # traceback from inside the first spoken reply.
        monkeypatch.setattr(tts, "KOKORO_DIR", tmp_path)
        assert tts.paths() is None
        assert tts.voices() == []

    def test_both_files_must_be_present_not_just_one(self, tmp_path,
                                                     monkeypatch):
        monkeypatch.setattr(tts, "KOKORO_DIR", tmp_path)
        (tmp_path / "kokoro-v1.0.onnx").write_bytes(b"not really a model")
        assert tts.paths() is None
        (tmp_path / "voices-v1.0.bin").write_bytes(b"not really a bank")
        assert tts.paths() is not None


class TestTheNumbersSheGetsFromTheWeb:
    """The forms a search result actually contains.

    Every string here was verified against the real Kokoro G2P before the fix.
    The hyphen cases were regressions rather than gaps: raw "2024-2025"
    phonemises correctly on its own, and speakable() was actively making it
    worse by gluing "minus" onto the preceding word.
    """

    @pytest.mark.parametrize("written, spoken", [
        # The regression: a hyphen between two numbers is a range or a score.
        ("The score was 3-1.", "The score was three-one."),
        ("The 2024-2025 season.", "The 2024-2025 season."),
        ("Call 555-1234.", "Call five hundred fifty five-1234."),
        # A minus sign is still a minus sign when nothing runs into it.
        ("It is -5 degrees.", "It is minus five degrees."),
        ("-5 degrees.", "minus five degrees."),
    ])
    def test_a_hyphen_after_a_number_is_not_a_minus_sign(self, written, spoken):
        assert tts.speakable(written) == spoken

    @pytest.mark.parametrize("written, spoken", [
        ("It costs 1,000 dollars.", "It costs 1000 dollars."),
        ("About 1,250,000 people.", "About 1250000 people."),
        ("$1,250 a month.", "$1250 a month."),
    ])
    def test_a_grouped_number_stays_one_number(self, written, spoken):
        # She used to say "one, zero dollars": the separator split the number
        # and each fragment was spoken on its own. Stripping the separator
        # hands espeak a single integer, which it reads correctly.
        assert tts.speakable(written) == spoken

    def test_a_comma_that_is_punctuation_is_left_alone(self):
        # Only a comma sitting between a digit and exactly three digits is a
        # group separator; the rest are pauses and must survive.
        assert tts.speakable("Ready, 3, go.") == "Ready, three, go."

    def test_a_dotted_version_keeps_all_of_its_parts(self):
        # "one. two" both lost the middle component and started a new
        # sentence mid-phrase.
        assert tts.speakable("Version 1.0.2 shipped.") == "Version 1.0.2 shipped."

    def test_an_ordinary_decimal_is_still_spoken(self):
        # The version guard must not catch a plain decimal.
        assert tts.speakable("70.5 degrees") == "seventy point five degrees"

    def test_a_word_glued_to_a_number_is_not_a_negative(self):
        # The digit is still spoken — that is the whole job — but the hyphen
        # attached to a letter is not a minus sign.
        assert tts.speakable("model B-2") == "model B-two"
        assert "minus" not in tts.speakable("model B-2")
