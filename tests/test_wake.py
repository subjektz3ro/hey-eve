"""Was that said to her, and did it mean "we're done"?

These two questions are the whole front door. Getting the first one wrong
means she either ignores you or answers the television; getting the second
wrong means she keeps asking "anything else?" after you have already left the
room. Both have failed in ways worth pinning down.
"""
from __future__ import annotations

import pytest

from eve import wake


class TestBeingAddressed:
    @pytest.mark.parametrize("said, asked", [
        ("hey Eve what time is it", "what time is it"),
        ("Hey Eve, what time is it?", "what time is it?"),
        ("hello eve tell me about Estonia", "tell me about Estonia"),
        ("okay Eve. how hot is the pi", "how hot is the pi"),
    ])
    def test_the_wake_phrase_is_stripped_and_the_request_survives(
            self, said, asked):
        assert wake.address(said) == asked

    def test_the_name_alone_is_addressed_with_nothing_asked_yet(self):
        # Distinct from "not for us": the caller prompts for more instead of
        # dropping it, so saying just her name works like a raised eyebrow.
        assert wake.address("Eve") == ""
        assert wake.address("hey Eve") == ""

    def test_heave_is_addressed_because_that_is_what_hey_eve_becomes(self):
        # A name opening on a vowel makes the greeting run into it, and the
        # recogniser writes the result as one word. Matching it outright is
        # cheaper than trying to make people enunciate.
        assert wake.address("heave what is the weather") == "what is the weather"

    @pytest.mark.parametrize("mishearing", ["eave", "evie", "ava", "eva"])
    def test_the_recognisers_near_misses_still_count_as_her_name(
            self, mishearing):
        assert wake.address(f"hey {mishearing}, what time is it") \
            == "what time is it"

    @pytest.mark.parametrize("overheard", [
        "the eve of the launch was quiet",
        "I was reading about Eve yesterday",
        "what time is it",
        "",
    ])
    def test_speech_that_does_not_open_with_her_name_is_not_for_her(
            self, overheard):
        # The bare name has to OPEN the utterance. That anchor is what keeps
        # "the eve of the launch" out, and it does the same job the optional
        # greeting does — which matters, because the greeting is the part
        # most often lost to a speech detector that has not woken up yet.
        assert wake.address(overheard) is None


class TestNoise:
    @pytest.mark.parametrize("noise", [
        "[BLANK_AUDIO]", "(sighs)", "Thanks for watching!", "you",
        "Thank you.", "uh", "", "   ", ".",
    ])
    def test_whispers_placeholders_for_silence_are_not_requests(self, noise):
        assert wake.is_noise(noise) is True

    def test_a_real_question_is_not_noise(self):
        assert wake.is_noise("what is the capital of Norway") is False

    def test_anything_addressed_to_her_is_never_noise(self):
        # "okay" on its own is noise. "okay Eve" is a wake attempt, and
        # dropping a wake attempt is the failure a person actually notices.
        assert wake.is_noise("okay") is True
        assert wake.is_noise("okay Eve") is False


class TestDismissal:
    @pytest.mark.parametrize("closer", [
        "no", "nope", "nah", "no thanks", "nothing else", "that's all",
        "never mind", "I'm good", "we're done", "goodnight", "stop",
        "that's all, thank you", "okay that's it",
    ])
    def test_a_sign_off_and_nothing_more_ends_the_conversation(self, closer):
        assert wake.is_dismissal(closer) is True

    def test_a_bare_no_ends_it(self):
        # This is the regression the dismissal list was rewritten for. The
        # entries used to be split on whitespace as well as pipes, so the
        # first line became one entry — "no nope nah none negative" — and
        # plain "no", the commonest refusal there is, matched nothing.
        # Answering "no" to "want the longer version?" then spent a whole
        # turn and reopened the follow-up window for another minute.
        assert wake.is_dismissal("no") is True

    def test_a_refusal_in_front_of_a_closer_is_still_a_closer(self):
        # Refusals combine with closers freely, and enumerating the products
        # of two lists is how the set ends up holding "im good" but missing
        # "no im good".
        assert wake.is_dismissal("no I'm good") is True
        assert wake.is_dismissal("nope, that's all") is True

    @pytest.mark.parametrize("request_", [
        "no, tell me about Estonia",
        "no that's wrong",
        "stop the timer",
        "what's the weather",
    ])
    def test_a_refusal_with_a_request_attached_is_a_request(self, request_):
        # The first word is not the signal: "no" ends a conversation and
        # "no, tell me about Estonia" starts one. The difference is
        # everything after it, which is why this is matched in full rather
        # than searched for.
        assert wake.is_dismissal(request_) is False

    def test_anything_long_enough_to_be_a_sentence_is_not_a_sign_off(self):
        assert wake.is_dismissal("no I would rather hear about the weather") \
            is False

    def test_a_closer_with_a_courtesy_stuck_on_the_end_is_still_a_closer(self):
        assert wake.is_dismissal("that's it thanks") is True
        assert wake.is_dismissal("I'm all set, thank you") is True
