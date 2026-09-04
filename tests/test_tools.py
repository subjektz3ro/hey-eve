"""What she can look up, and what happens when looking it up goes wrong.

The tool surface is assembled at import time, and part of it is conditional:
the bar tool is offered only when a Barkeep token is configured. That makes
`import` itself a decision, which is why these tests reload the module inside
a controlled settings directory rather than trusting whatever was on the
machine when the suite started.
"""
from __future__ import annotations

import importlib
import shutil
import subprocess
from pathlib import Path

import pytest

from eve import bar, tools


@pytest.fixture
def reloaded_tools(settings_file):
    """Re-import tools against this test's settings, then put it back.

    Restoring matters: without it, the first test to configure a token would
    leave the bar tool attached for every test after it.
    """
    def load(**values: str):
        settings_file(**values)
        return importlib.reload(tools)

    yield load
    importlib.reload(tools)


class TestWhichToolsAreOffered:
    def test_web_search_runs_on_anthropics_side_and_has_no_handler(self):
        # It arrives already answered, so the tool loop must never try to
        # execute it. A handler here would mean a call that never returns.
        assert tools.WEB_SEARCH["type"].startswith("web_search")
        assert "web_search" not in tools.HANDLERS

    def test_web_search_is_capped_because_it_is_billed_per_search(self):
        assert tools.WEB_SEARCH["max_uses"] == tools.WEB_SEARCH_LIMIT > 0

    def test_the_bar_tool_is_not_offered_without_a_token(self, reloaded_tools):
        # Every call would otherwise come back as an authentication error the
        # model would then read aloud, which is worse than not having it.
        module = reloaded_tools(ANTHROPIC_API_KEY="sk-ant-notarealkey")
        names = {tool.get("name") for tool in module.TOOLS}
        assert "get_bar_status" not in names
        assert "get_bar_status" not in module.HANDLERS

    def test_the_bar_tool_appears_once_a_token_is_configured(
            self, reloaded_tools):
        module = reloaded_tools(
            ANTHROPIC_API_KEY="sk-ant-notarealkey",
            BARKEEP_TOKEN="a-real-looking-token",
        )
        names = {tool.get("name") for tool in module.TOOLS}
        assert "get_bar_status" in names
        assert "get_bar_status" in module.HANDLERS

    def test_memory_is_always_available_because_it_needs_no_credential(self):
        names = {tool.get("name") for tool in tools.TOOLS}
        assert {"remember", "forget"} <= names

    def test_every_client_side_tool_has_a_handler_to_run_it(self):
        # A tool offered without a handler is a call the model can make and
        # the loop cannot answer, which stalls the turn silently.
        offered = {tool["name"] for tool in tools.TOOLS
                   if "type" not in tool}
        assert offered == set(tools.HANDLERS)

    def test_every_tool_declares_an_input_schema(self):
        for tool in tools.TOOLS:
            if "type" in tool:      # server-side tools carry their own shape
                continue
            assert tool["input_schema"]["type"] == "object"


class TestRunningATool:
    def test_a_tool_that_does_not_exist_says_so_instead_of_raising(self):
        assert tools.run_tool("summon_a_demon", {}) == \
            "No such tool: summon_a_demon"

    def test_a_tool_that_crashes_does_not_end_the_conversation(
            self, monkeypatch):
        # The model reads this string and paraphrases it. A traceback here
        # would take down a turn that was otherwise recoverable.
        def explode() -> str:
            raise RuntimeError("the sensor is on fire")

        monkeypatch.setitem(tools.HANDLERS, "get_local_time", explode)
        result = tools.run_tool("get_local_time", {})
        assert "get_local_time tool failed" in result
        assert "the sensor is on fire" in result

    def test_wrong_arguments_are_reported_rather_than_raised(self):
        result = tools.run_tool("get_local_time", {"timezone": "UTC"})
        assert "failed" in result

    def test_a_real_tool_returns_prose_for_the_model_to_read(self):
        # Prose rather than JSON, because a sentence survives being
        # paraphrased aloud better than a structure it has to narrate.
        answer = tools.run_tool("get_local_time", {})
        assert answer.startswith("It is")
        assert "timezone" in answer

    def test_memory_is_reachable_through_the_same_dispatch(self):
        assert "Noted" in tools.run_tool(
            "remember", {"topic": "coffee", "fact": "black, no sugar"})


class TestReadingThisMachine:
    def test_the_tool_describes_every_supported_linux_host_not_only_a_pi(self):
        tool = next(
            item for item in tools.TOOLS
            if item.get("name") == "get_system_status"
        )
        assert "Linux host" in tool["description"]
        assert "Raspberry Pi" not in tool["description"]

    def test_it_returns_a_spoken_sentence_about_the_machine(self):
        status = tools.get_system_status()
        assert isinstance(status, str) and status

    def test_a_machine_that_answers_nothing_still_gets_an_answer(
            self, monkeypatch):
        # Every probe is individually optional — a Pi reports throttling, a
        # laptop has no thermal_zone0 — so all of them failing has to produce
        # a sentence rather than an empty string the model would read as
        # silence.
        def refuse(*args, **kwargs):
            raise OSError("no such file")

        monkeypatch.setattr(Path, "read_text", refuse)
        monkeypatch.setattr(shutil, "disk_usage", refuse)
        monkeypatch.setattr(subprocess, "run", refuse)
        assert tools.get_system_status() == \
            "Could not read this machine's status."


class TestTheBarSeam:
    def test_the_bar_module_exposes_the_same_dispatch_shape(self):
        # tools.py merges these in wholesale, so the shapes have to match.
        assert set(bar.HANDLERS) == {tool["name"] for tool in bar.TOOLS}
