"""What she is allowed to remember, and what happens when writing it fails.

The store is small on purpose — every fact rides in the system prompt on
every turn — and it is the only thing in the project that persists what a
person said. Both properties are tested here: the bounds that keep it small,
and the atomicity that keeps a power cut from turning it into an empty file.
"""
from __future__ import annotations

import json

from eve import memory


class TestRemembering:
    def test_a_fact_survives_into_the_next_conversation(self):
        assert "Noted" in memory.remember("location", "lives in Fictionopolis")
        assert "fictionopolis" in memory.as_prompt().lower()

    def test_a_topic_is_corrected_rather_than_duplicated(self):
        memory.remember("location", "lives in Fictionopolis")
        result = memory.remember("location", "lives in Tallinn")
        # "Updated", not "Noted" — the caller says which happened out loud,
        # and keying by topic is what makes correction possible at all.
        assert "Updated" in result
        prompt = memory.as_prompt()
        assert "Tallinn" in prompt
        assert "Fictionopolis" not in prompt

    def test_topics_are_lowercased_so_case_is_not_a_second_fact(self):
        memory.remember("Coffee", "black, no sugar")
        memory.remember("COFFEE", "oat milk now")
        assert len(json.loads(memory.STORE.read_text())) == 1

    def test_a_fact_needs_both_halves(self):
        assert "I need both" in memory.remember("", "something")
        assert "I need both" in memory.remember("location", "   ")
        assert not memory.STORE.exists()

    def test_a_long_fact_is_truncated_rather_than_refused(self):
        memory.remember("essay", "x" * 500)
        stored = json.loads(memory.STORE.read_text())["essay"]
        assert len(stored) == memory.MAX_FACT_CHARS

    def test_she_stops_taking_new_facts_at_the_cap(self):
        # A spoken assistant cannot usefully recite a hundred facts, and the
        # cap is what keeps the prompt — and so the per-turn cost — bounded.
        for index in range(memory.MAX_FACTS):
            memory.remember(f"topic{index}", "a fact")
        refusal = memory.remember("one more", "a fact")
        assert str(memory.MAX_FACTS) in refusal
        assert "cannot take more" in refusal

    def test_the_cap_never_blocks_correcting_something_already_known(self):
        for index in range(memory.MAX_FACTS):
            memory.remember(f"topic{index}", "a fact")
        assert "Updated" in memory.remember("topic0", "a better fact")


class TestForgetting:
    def test_forgetting_drops_the_fact(self):
        memory.remember("location", "Fictionopolis")
        assert "Forgotten" in memory.forget("location")
        assert memory.as_prompt() == ""

    def test_forgetting_something_she_never_knew_says_so(self):
        assert "nothing stored" in memory.forget("astrology")


class TestTheFileItself:
    def test_the_store_is_owner_only_because_of_what_is_in_it(self):
        memory.remember("location", "Fictionopolis")
        assert memory.STORE.stat().st_mode & 0o777 == 0o600

    def test_writing_leaves_no_temporary_file_behind(self):
        # Written through mkstemp + os.replace, so a crash mid-write leaves
        # the previous file intact rather than a truncated one. A stale .tmp
        # in the config directory would mean the replace never happened.
        memory.remember("location", "Fictionopolis")
        leftovers = [p.name for p in memory.STORE.parent.iterdir()
                     if p.suffix == ".tmp"]
        assert leftovers == []

    def test_a_corrupt_store_reads_as_empty_rather_than_crashing(self):
        # This file is hand-editable by design. A stray comma in it must not
        # take the assistant down on the next question she is asked.
        memory.STORE.write_text("{not json at all")
        assert memory.as_prompt() == ""
        assert "Noted" in memory.remember("location", "Fictionopolis")

    def test_a_missing_store_is_simply_no_facts(self):
        assert not memory.STORE.exists()
        assert memory.as_prompt() == ""


class TestThePromptBlock:
    def test_no_facts_adds_nothing_to_the_prompt(self):
        assert memory.as_prompt() == ""

    def test_facts_arrive_sorted_so_the_prompt_is_stable_between_turns(self):
        memory.remember("zebra", "last")
        memory.remember("apple", "first")
        block = memory.as_prompt()
        assert block.index("apple") < block.index("zebra")

    def test_the_block_tells_the_model_to_use_them_not_recite_them(self):
        memory.remember("location", "Fictionopolis")
        block = memory.as_prompt()
        assert "Never mention having looked" in block


class TestTheToolSurface:
    def test_both_tools_are_offered_and_both_have_handlers(self):
        names = {tool["name"] for tool in memory.TOOLS}
        assert names == {"remember", "forget"}
        assert names == set(memory.HANDLERS)

    def test_every_tool_declares_its_required_arguments(self):
        for tool in memory.TOOLS:
            schema = tool["input_schema"]
            assert schema["required"]
            for name in schema["required"]:
                assert name in schema["properties"]

    def test_memory_examples_are_user_neutral(self):
        remember = next(
            tool for tool in memory.TOOLS if tool["name"] == "remember"
        )
        assert "where they live" in remember["description"]
        assert "where he lives" not in remember["description"]
