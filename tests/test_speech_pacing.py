"""When to start playing, given that synthesis is slower than playback.

Kokoro on this Pi makes audio at about 0.6x real time. Handing each sentence
to the player as it finished therefore drained the buffer at every sentence
boundary — the reply gapped audibly and the waveform froze with it. So
playback waits until the finished audio is long enough to cover what the rest
still needs to be made.

This is the maths behind that decision, and it is worth testing precisely,
because being wrong by one sentence is inaudible in a unit test and very
audible in a room.
"""
from __future__ import annotations

import pytest

from eve import speech


class TestSplittingAReply:
    def test_a_reply_is_split_at_sentence_boundaries(self):
        assert speech._sentences(
            "It is the capital of Norway. I had assumed you knew that."
        ) == ["It is the capital of Norway.", "I had assumed you knew that."]

    def test_a_short_fragment_is_merged_backwards_into_what_precedes_it(self):
        # Synthesis has a fixed per-call cost, so "Yes." on its own spends
        # more time starting up than speaking. It joins the piece before it,
        # which is also what keeps the tag question attached to its answer.
        pieces = speech._sentences(
            "That is correct as far as it goes. Yes.")
        assert pieces == ["That is correct as far as it goes. Yes."]

    def test_a_short_opening_fragment_has_nothing_to_merge_into(self):
        # Merging is backwards only, so a reply that OPENS short keeps that
        # fragment as its own piece rather than swallowing the sentence
        # after it. Worth pinning: it is the difference between the first
        # word arriving in half a second and arriving in four.
        assert speech._sentences("Oslo. It is the capital of Norway.") == \
            ["Oslo.", "It is the capital of Norway."]

    def test_a_one_sentence_reply_stays_one_sentence(self):
        assert speech._sentences("Oslo, and I assumed you knew that.") == \
            ["Oslo, and I assumed you knew that."]

    def test_an_empty_reply_produces_nothing_to_say(self):
        assert speech._sentences("") == []
        assert speech._sentences("   \n  ") == []

    def test_a_leading_short_fragment_is_not_merged_backwards(self):
        # Nothing to merge into yet, so it stands alone rather than being
        # dropped — which is what an unguarded `pieces[-1]` would do.
        assert speech._sentences("Yes.") == ["Yes."]


class TestWhetherStartingNowWouldStall:
    def test_a_full_buffer_and_nothing_left_to_make_never_stalls(self):
        assert speech._stalls(10.0, 0.6, []) is False

    def test_an_empty_buffer_with_work_remaining_stalls(self):
        assert speech._stalls(0.0, 0.6, [200]) is True

    def test_enough_buffered_audio_covers_the_rest(self):
        # 60s of audio in hand, one short sentence left: the player cannot
        # possibly catch up with synthesis.
        assert speech._stalls(60.0, 0.6, [100]) is False

    def test_it_is_not_optimistic_by_the_final_sentence(self):
        # The bug this function was rewritten to fix. Computed in closed form
        # as (remaining synth - remaining audio), it credited the buffer with
        # the LAST sentence's duration — but the player arrives at that
        # sentence before it has been spoken. Optimistic by exactly one
        # sentence, worth 1.9s of silence.
        #
        # Walking it: with 3.0s buffered at 0.6x, a single 100-char sentence
        # needs 5.1s of compute to make 3.1s of audio, and the player runs
        # dry at 3.0s. The closed form would have called this safe.
        assert speech._stalls(3.0, 0.6, [100], safety=0.0) is True

    def test_a_faster_measured_ratio_lets_playback_start_sooner(self):
        remaining = [120, 120]
        assert speech._stalls(4.0, 0.4, remaining) is True
        assert speech._stalls(4.0, 2.5, remaining) is False

    def test_the_safety_margin_makes_the_decision_more_conservative(self):
        # Slack on top of the computed lead, for a synthesis that runs slower
        # than the sentences before it did.
        made, ratio, remaining = 6.0, 1.0, [60]
        assert speech._stalls(made, ratio, remaining, safety=0.0) is False
        assert speech._stalls(made, ratio, remaining, safety=10.0) is True


class TestMeasuringThroughput:
    def test_the_measurement_is_deliberately_pessimistic(self):
        # Measured while the CPU was idle, but every sentence after playback
        # starts competes with the pacer thread and with aplay. Synthesis
        # genuinely slows once audio is running, so the raw figure flatters
        # what comes next.
        assert speech._measured_ratio(6.0, 6.0) == pytest.approx(0.65)
        assert speech._measured_ratio(6.0, 10.0) < 6.0 / 10.0

    def test_an_absurdly_fast_measurement_is_clamped(self):
        # A near-zero synth time — a cached sentence, a scheduling artefact —
        # must not convince the pacer that it can start immediately.
        assert speech._measured_ratio(10.0, 0.001) == pytest.approx(3.0 * 0.65)

    def test_an_absurdly_slow_measurement_is_clamped_too(self):
        assert speech._measured_ratio(0.001, 10.0) == pytest.approx(0.2 * 0.65)

    def test_a_zero_synth_time_falls_back_to_the_measured_default(self):
        assert speech._measured_ratio(5.0, 0.0) == \
            pytest.approx(speech._SYNTH_RATIO * 0.65)


class TestPredictingWhenPlaybackBegins:
    def test_a_one_sentence_reply_waits_for_nothing(self):
        # The common case, since the model is told to be brief. There is
        # nothing left to make, so there is nothing to buffer against.
        assert speech._play_starts_after(["The capital of Norway is Oslo."]) == 0

    def test_a_long_reply_buffers_before_it_starts(self):
        sentences = ["A reasonably long sentence about the weather here."] * 6
        assert speech._play_starts_after(sentences) > 0

    def test_the_prediction_is_always_a_real_sentence_index(self):
        # It only feeds the progress bar's denominator, so being wrong costs
        # nothing — but being out of range would raise mid-reply.
        for count in range(1, 12):
            sentences = ["Something worth saying out loud here."] * count
            assert 0 <= speech._play_starts_after(sentences) < count


class TestTheConstantsAreMeasurements:
    def test_the_speaking_rate_is_the_measured_one_not_the_guess(self):
        # 14.0 was a guess and 40% low; real replies measured 18.8-20.2.
        assert 18.0 <= speech._CHARS_PER_SEC <= 21.0

    def test_synthesis_is_recorded_as_slower_than_real_time(self):
        # If this is ever >= 1.0 the whole buffering strategy is unnecessary,
        # and the code that implements it becomes dead weight.
        assert speech._SYNTH_RATIO < 1.0


class TestTheCountdownIsHonest:
    """"~8s" should mean about eight seconds.

    _measured_ratio discounts the measured throughput by 35% so that _stalls
    errs towards starting playback late — being wrong there is an audible gap
    mid-reply. That margin belongs to that decision. It was also being spent
    on the countdown, so every "~Ns" the panel showed was 1.54x longer than
    the truth: the one number on that screen a person checks against a clock.
    """

    def test_the_honest_ratio_is_what_synthesis_is_actually_doing(self):
        # Three seconds of audio from two seconds of compute is 1.5x.
        assert speech._honest_ratio(3.0, 2.0) == pytest.approx(1.5)

    def test_the_deciding_ratio_is_the_honest_one_discounted(self):
        assert speech._measured_ratio(3.0, 2.0) == pytest.approx(
            1.5 * speech._PESSIMISM)

    def test_the_countdown_would_have_been_half_again_too_long(self):
        # The size of the error, pinned so the fix cannot quietly revert.
        honest = speech._honest_ratio(6.0, 10.0)
        deciding = speech._measured_ratio(6.0, 10.0)
        assert honest / deciding == pytest.approx(1 / speech._PESSIMISM)
        assert honest / deciding > 1.5

    def test_a_first_estimate_uses_the_measured_starting_guess(self):
        # Before anything has been synthesised there is no measurement, and
        # the fallback was also discounted — so the very first number shown
        # was the most wrong one.
        assert speech._honest_ratio(0.0, 0.0) == pytest.approx(speech._SYNTH_RATIO)

    @pytest.mark.parametrize("audio, compute", [(6.0, 10.0), (3.0, 5.0), (1.0, 3.0)])
    def test_the_estimate_matches_the_arithmetic_a_person_would_do(
            self, audio, compute):
        # Seconds of audio still to make, divided by how fast it is being
        # made. Nothing else.
        ratio = speech._honest_ratio(audio, compute)
        chars_left = 100
        expected = chars_left / speech._CHARS_PER_SEC / ratio
        assert expected == pytest.approx(
            chars_left / speech._CHARS_PER_SEC / (audio / compute))

    def test_the_buffering_decision_keeps_its_margin(self):
        # The fix must not make _stalls optimistic — that trades a wrong
        # number for a gap you can hear.
        assert speech._PESSIMISM < 1.0
        assert speech._measured_ratio(6.0, 10.0) < speech._honest_ratio(6.0, 10.0)

    def test_both_are_still_clamped(self):
        assert speech._honest_ratio(1000.0, 0.001) <= 3.0
        assert speech._honest_ratio(0.001, 1000.0) >= 0.2
