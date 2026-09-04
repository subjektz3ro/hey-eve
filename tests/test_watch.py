"""The first thing she can do without being asked.

Every other tool answers and is finished — you ask, she replies, the exchange
closes. That makes her a lookup table with a voice. A promise she can keep
later is the whole difference, and it is why "tell me when the rain starts"
was impossible before this file existed.

Three things are being pinned here, in rough order of how badly they go wrong.

The store has to be as defensive as memory.json, because next_due_in() is
called about thirty times a second on the listen loop and anything that raises
there takes her off the air rather than costing a feature.

She must never speak over anybody. The capture only yields early when nothing
has been said into it, so the worst a due reminder can cost is the length of
the sentence somebody was in the middle of.

And a promise made to a room she has been muted out of must not be spent. She
cannot tell from inside the loop whether anyone is there, so while she is
asleep the items are held rather than announced and marked delivered.
"""
from __future__ import annotations

import json

import pytest

from eve import main, tools, watch


class TestMakingAPromise:
    def test_it_is_stored_and_confirmed_out_loud(self):
        reply = watch.add(600, "the oven")
        assert "10 minutes" in reply
        assert [item["what"] for item in watch.pending()] == ["the oven"]

    def test_the_delay_is_said_the_way_a_person_says_it(self):
        assert "30 seconds" in watch.add(30, "a")
        assert "20 minutes" in watch.add(1200, "b")
        assert "3 hours" in watch.add(10800, "c")
        assert "2 days" in watch.add(172800, "d")

    def test_nothing_to_say_is_refused(self):
        assert "what to remind you" in watch.add(600, "   ")
        assert watch.pending() == []

    def test_a_delay_that_is_not_a_number_is_refused(self):
        # The model computes this, and a tool that raises on a bad argument
        # turns a slip into a failed turn rather than a sentence.
        assert "make sense" in watch.add("soon", "the oven")
        assert watch.pending() == []

    def test_something_already_due_is_refused(self):
        assert "too soon" in watch.add(0, "now")
        assert "too soon" in watch.add(-500, "the past")
        assert watch.pending() == []

    def test_a_promise_beyond_the_horizon_is_refused(self):
        # A bound on what an arithmetic slip can do, not a claim about what is
        # useful: a wrong number should be a refusal, not something she holds
        # until the next power cut.
        assert "further ahead" in watch.add(watch.MAX_HORIZON_S + 1, "someday")
        assert watch.pending() == []

    def test_the_queue_is_capped(self):
        for index in range(watch.MAX_PENDING):
            watch.add(600 + index, f"thing {index}")
        assert "will not take another" in watch.add(600, "one more")
        assert len(watch.pending()) == watch.MAX_PENDING

    def test_what_she_will_say_is_bounded(self):
        watch.add(600, "x" * 500)
        assert len(watch.pending()[0]["what"]) == watch.MAX_WHAT_CHARS

    def test_the_file_is_not_world_readable(self, isolated_settings):
        # It holds whatever was said in the room, same as memory.json.
        watch.add(600, "the oven")
        assert oct(watch.STORE.stat().st_mode)[-3:] == "600"


class TestComingDue:
    def test_nothing_comes_due_early(self):
        watch.add(600, "the oven")
        assert watch.due() == []
        assert len(watch.pending()) == 1

    def test_it_comes_due_and_is_handed_over(self):
        watch.add(600, "the oven")
        ready = watch.due(now=watch.time.time() + 601)
        assert [item["what"] for item in ready] == ["the oven"]

    def test_it_is_removed_as_it_is_handed_over(self):
        """Cleared before it is spoken, not after, and deliberately.

        The other order loses a reminder only if she dies between the two, and
        repeats one forever if anything downstream raises every time. An
        assistant stuck announcing the same sentence every few seconds is a
        far worse failure than one dropped reminder.
        """
        watch.add(600, "the oven")
        watch.due(now=watch.time.time() + 601)
        assert watch.pending() == []

    def test_several_come_due_oldest_first(self):
        watch.add(600, "second")
        watch.add(300, "first")
        ready = watch.due(now=watch.time.time() + 700)
        assert [item["what"] for item in ready] == ["first", "second"]

    def test_only_what_is_due_is_taken(self):
        watch.add(300, "soon")
        watch.add(9000, "later")
        assert [i["what"] for i in watch.due(now=watch.time.time() + 400)] == ["soon"]
        assert [i["what"] for i in watch.pending()] == ["later"]

    def test_a_store_it_cannot_clear_announces_nothing(self, monkeypatch):
        # Failing to clear and announcing anyway is the repeating-forever bug.
        watch.add(600, "the oven")

        def refuse(_items):
            raise OSError("read-only filesystem")

        monkeypatch.setattr(watch, "_save", refuse)
        assert watch.due(now=watch.time.time() + 601) == []


class TestHowLongUntilTheNextOne:
    def test_an_empty_store_is_never(self):
        assert watch.next_due_in() == float("inf")

    def test_it_counts_down(self):
        watch.add(600, "the oven")
        assert watch.next_due_in() == pytest.approx(600, abs=2)

    def test_it_goes_negative_once_due(self):
        watch.add(600, "the oven")
        assert watch.next_due_in(now=watch.time.time() + 700) < 0

    def test_it_reports_the_soonest(self):
        watch.add(9000, "later")
        watch.add(300, "sooner")
        assert watch.next_due_in() == pytest.approx(300, abs=2)

    def test_it_survives_a_store_that_cannot_be_read(self, isolated_settings):
        # Called ~30 times a second on the listen loop. Raising here is not a
        # lost feature, it is a deaf assistant.
        watch.STORE.write_text("{not json")
        assert watch.next_due_in() == float("inf")


class TestCancelling:
    def test_by_the_words_used_for_it(self):
        watch.add(600, "the oven timer")
        assert "Dropped 1" in watch.cancel("oven")
        assert watch.pending() == []

    def test_by_identifier(self):
        watch.add(600, "the oven")
        assert "Dropped 1" in watch.cancel(watch.pending()[0]["id"])

    def test_something_that_matches_nothing(self):
        watch.add(600, "the oven")
        assert "nothing outstanding" in watch.cancel("the car")
        assert len(watch.pending()) == 1


class TestSayingWhatIsOutstanding:
    def test_an_empty_list_is_a_sentence_not_a_blank(self):
        assert "not asked me" in watch.as_spoken_list()

    def test_each_one_is_read_with_its_remaining_time(self):
        watch.add(1200, "the oven")
        spoken = watch.as_spoken_list()
        assert "20 minutes" in spoken and "the oven" in spoken

    def test_it_is_prose_rather_than_a_table(self):
        watch.add(600, "the oven")
        watch.add(1200, "the bins")
        spoken = watch.as_spoken_list()
        assert "\n" not in spoken and "|" not in spoken


class TestAHandEditedStore:
    """The same defensiveness memory._load was given, for the same reason.

    This is read on the listen loop between every capture, so a file somebody
    emptied by typing `{}` into it must read as nothing rather than raise.
    """

    @pytest.mark.parametrize("body", ["{}", "null", "42", '"a string"', "true",
                                      "[1, 2, 3]", '["a"]', "{not json"])
    def test_it_reads_as_empty_rather_than_crashing(self, body, isolated_settings):
        watch.STORE.write_text(body)
        assert watch._load() == []
        assert watch.next_due_in() == float("inf")
        assert watch.due() == []

    def test_one_malformed_row_does_not_lose_the_others(self, isolated_settings):
        watch.STORE.write_text(json.dumps([
            {"id": "aa", "when": 1.0, "what": "good"},
            {"id": "bb"},                                # no when, no what
            {"when": "not a number", "what": "x", "id": "cc"},
            {"id": "dd", "when": 2.0, "what": "also good"},
        ]))
        assert [item["what"] for item in watch._load()] == ["good", "also good"]

    def test_a_missing_file_is_simply_empty(self, isolated_settings):
        assert not watch.STORE.exists()
        assert watch.pending() == []


class TestSheIsOfferedThemAsTools:
    def test_all_three_are_registered(self):
        names = {tool["name"] for tool in tools.TOOLS}
        assert {"set_reminder", "list_reminders", "cancel_reminder"} <= names

    def test_every_one_has_a_handler(self):
        for tool in watch.TOOLS:
            assert tool["name"] in tools.HANDLERS

    def test_the_dispatch_actually_runs_them(self):
        assert "10 minutes" in tools.run_tool(
            "set_reminder", {"seconds_from_now": 600, "what": "the oven"})
        assert "the oven" in tools.run_tool("list_reminders", {})
        assert "Dropped" in tools.run_tool("cancel_reminder", {"which": "oven"})

    def test_a_bad_call_is_a_sentence_rather_than_a_crash(self):
        # tools.run_tool catches, but the message still has to be speakable.
        assert "failed" in tools.run_tool("set_reminder", {"what": "no delay"})


class TestSheNeverSpeaksOverAnybody:
    def test_the_capture_only_yields_when_nothing_has_been_said(self):
        """`should_stop` is checked behind `not heard_speech`.

        Without that guard a reminder coming due mid-question would truncate
        the question, and the recording would be handed to whisper half
        finished — she would answer the first half of what you asked and
        announce the reminder over the rest.
        """
        import inspect

        from eve import speech

        source = inspect.getsource(speech.record_until_silence)
        line = next(ln for ln in source.splitlines() if "should_stop()" in ln)
        assert "not heard_speech" in line

    def test_a_due_item_ends_a_capture_that_nobody_is_speaking_into(
            self, monkeypatch, tmp_path):
        from tests.test_mute import ROOM, Stream

        from eve import speech

        monkeypatch.setattr(speech.subprocess, "Popen",
                            lambda *a, **k: Stream([ROOM] * 200))
        monkeypatch.setattr(speech, "_DETECTOR", None)
        calls = []

        def due_on_the_third_chunk():
            calls.append(1)
            return len(calls) >= 3

        assert speech.record_until_silence(
            tmp_path / "heard.wav", lead_in=None,
            should_stop=due_on_the_third_chunk) is None
        assert len(calls) == 3, "it kept recording after the promise came due"


class TestAPromiseIsNotSpentOnARoomSheCannotHear:
    def test_nothing_is_announced_while_she_is_asleep(self):
        """Muting is how you keep her out of a conversation.

        From inside the loop she cannot tell whether anybody is in the room, so
        announcing into a muted one marks the reminder delivered when nobody
        heard it. They keep until she has signal again.
        """
        import inspect

        source = inspect.getsource(main._listen_loop)
        held = source.index("if not display.asleep:")
        announced = source.index("watch.due()")
        assert held < announced

    def test_a_held_promise_does_not_busy_loop_the_capture(self):
        """The bug the guard on should_stop exists to prevent.

        While asleep nothing is announced, so a due item stays due. Without
        `not display.asleep` in the predicate it would end the capture, find
        that it is being held, and end the next one — thirty times a second,
        forever, around a promise she has already decided to keep.
        """
        import inspect

        source = inspect.getsource(main._listen_loop)
        predicate = source[source.index("should_stop=lambda:"):]
        predicate = predicate[:predicate.index("\n            )")]
        assert "not display.asleep" in predicate
