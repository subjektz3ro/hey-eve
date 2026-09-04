"""The conversation loop, against a stand-in for the API.

`Responder` is the backend seam: today Claude over the network, later a local
model on a Hailo-10H, with main.py unable to tell the difference. So nothing
here talks to Anthropic — a fake client supplies the messages, and what is
tested is everything eve does *around* them: the bounded history, the bounded
tool loop, and the arithmetic that ends up in the log line.
"""
from __future__ import annotations

import pytest

from eve import assistant


class Block:
    """One content block, shaped like the SDK's."""

    def __init__(self, type_, **fields):
        self.type = type_
        for name, value in fields.items():
            setattr(self, name, value)


class Usage:
    def __init__(self, input_tokens=10, output_tokens=5):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class Message:
    def __init__(self, content, stop_reason="end_turn", usage=None):
        self.content = content
        self.stop_reason = stop_reason
        self.usage = usage or Usage()


class FakeStream:
    def __init__(self, message):
        self._message = message

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __iter__(self):
        for block in self._message.content:
            yield Block("content_block_start", content_block=block)

    def get_final_message(self):
        return self._message


class FakeClient:
    """Hands back a scripted sequence of messages, one per turn."""

    def __init__(self, messages):
        self._messages = list(messages)
        self.calls: list[dict] = []
        self.messages = self

    def stream(self, **kwargs):
        self.calls.append(kwargs)
        return FakeStream(self._messages.pop(0))


def responder_with(*messages, model="claude-haiku-4-5"):
    responder = assistant.ClaudeResponder.__new__(assistant.ClaudeResponder)
    responder.client = FakeClient(messages)
    responder.model = model
    responder.history = []
    responder.last_usage = (0, 0)
    responder.searches = 0
    return responder


def text(body: str) -> Block:
    return Block("text", text=body)


def tool_call(name: str, arguments: dict, call_id="call_1") -> Block:
    return Block("tool_use", name=name, input=arguments, id=call_id)


class TestTheKey:
    def test_a_missing_key_names_the_file_it_should_be_in(self):
        # The first question is where this fails, so the message has to be
        # actionable without reading the source.
        with pytest.raises(RuntimeError, match="no ANTHROPIC_API_KEY"):
            assistant._load_key()

    def test_the_error_points_at_the_settings_file(self, isolated_settings):
        with pytest.raises(RuntimeError) as raised:
            assistant._load_key()
        assert str(isolated_settings / "env") in str(raised.value)

    def test_a_configured_key_is_returned(self, settings_file):
        settings_file(ANTHROPIC_API_KEY="sk-ant-notarealkey")
        assert assistant._load_key() == "sk-ant-notarealkey"


class TestOneTurn:
    def test_the_reply_is_the_text_blocks_joined(self):
        responder = responder_with(Message([text("Oslo."), text("Obviously.")]))
        assert responder.respond("capital of Norway?") == "Oslo. Obviously."

    def test_the_exchange_is_kept_for_the_next_turn(self):
        responder = responder_with(Message([text("Oslo.")]))
        responder.respond("capital of Norway?")
        assert responder.history[0] == {
            "role": "user", "content": "capital of Norway?"}
        assert responder.history[1]["role"] == "assistant"

    def test_an_empty_reply_comes_back_as_an_empty_string(self):
        responder = responder_with(Message([]))
        assert responder.respond("hello") == ""

    def test_blank_text_blocks_do_not_become_stray_spaces(self):
        responder = responder_with(Message([text("Oslo."), text("   ")]))
        assert responder.respond("capital?") == "Oslo."

    def test_the_memory_block_rides_along_in_the_system_prompt(self):
        from eve import memory
        memory.remember("location", "lives in Tallinn")
        responder = responder_with(Message([text("Fine.")]))
        responder.respond("where am I?")
        assert "Tallinn" in responder.client.calls[0]["system"]

    def test_the_caller_is_told_what_is_happening_during_the_turn(self):
        # A web search can take seconds, and a panel that says "searching"
        # beats one that just sits on "processing".
        seen = []
        responder = responder_with(Message([
            Block("server_tool_use", name="web_search"),
            text("Overcast."),
        ]))
        responder.respond("weather?", on_state=seen.append)
        assert "searching" in seen
        assert "thinking" in seen

    def test_no_callback_is_fine(self):
        responder = responder_with(Message([text("Oslo.")]))
        assert responder.respond("capital?", on_state=None) == "Oslo."


class TestTheToolLoop:
    def test_a_tool_call_is_executed_and_answered(self):
        responder = responder_with(
            Message([tool_call("get_local_time", {})]),
            Message([text("It is late.")]),
        )
        assert responder.respond("what time is it?") == "It is late."
        answer = responder.history[2]
        assert answer["role"] == "user"
        assert answer["content"][0]["type"] == "tool_result"
        assert answer["content"][0]["tool_use_id"] == "call_1"

    def test_several_calls_in_one_message_are_all_answered(self):
        responder = responder_with(
            Message([tool_call("get_local_time", {}, "a"),
                     tool_call("get_system_status", {}, "b")]),
            Message([text("Both fine.")]),
        )
        responder.respond("status?")
        results = responder.history[2]["content"]
        assert [r["tool_use_id"] for r in results] == ["a", "b"]

    def test_the_loop_is_bounded_so_a_turn_cannot_run_forever(self):
        # A model that keeps calling tools would otherwise hold the
        # conversation open indefinitely while the person waits in silence.
        responder = responder_with(*[
            Message([tool_call("get_local_time", {})]) for _ in range(10)
        ])
        responder.respond("what time is it?")
        assert len(responder.client.calls) == 4

    def test_a_paused_server_side_turn_is_resumed(self):
        # A server-side tool loop that hits its own iteration limit stops
        # with `pause_turn`; the turn is unfinished, and re-sending it
        # resumes where it left off.
        responder = responder_with(
            Message([Block("server_tool_use", name="web_search")],
                    stop_reason="pause_turn"),
            Message([text("Overcast and seventy five.")]),
        )
        assert responder.respond("weather?") == "Overcast and seventy five."
        assert len(responder.client.calls) == 2

    def test_web_search_is_counted_but_never_executed_here(self):
        # It runs on Anthropic's servers and arrives already answered.
        responder = responder_with(Message([
            Block("server_tool_use", name="web_search"),
            Block("server_tool_use", name="web_search"),
            text("Overcast."),
        ]))
        responder.respond("weather?")
        assert responder.searches == 2
        # No tool_result was appended: nothing client-side was called.
        assert all(entry["role"] != "user" or isinstance(entry["content"], str)
                   for entry in responder.history)

    def test_other_server_tools_are_not_billed_as_web_searches(self):
        responder = responder_with(Message([
            Block("server_tool_use", name="some_future_server_tool"),
            text("Done."),
        ]))
        responder.respond("do something")
        assert responder.searches == 0

    def test_search_budget_is_shared_across_every_api_call_in_one_turn(self):
        responder = responder_with(
            Message([
                Block("server_tool_use", name="web_search"),
                Block("server_tool_use", name="web_search"),
            ], stop_reason="pause_turn"),
            Message([
                Block("server_tool_use", name="web_search"),
            ], stop_reason="pause_turn"),
            Message([text("Done.")]),
        )

        assert responder.respond("research this") == "Done."
        search_limits = [
            next(
                tool["max_uses"]
                for tool in call["tools"]
                if tool.get("name") == "web_search"
            )
            for call in responder.client.calls[:2]
        ]
        assert search_limits == [3, 1]
        assert all(
            tool.get("name") != "web_search"
            for tool in responder.client.calls[2]["tools"]
        )
        assert responder.searches == 3

    def test_the_search_count_resets_between_turns(self):
        responder = responder_with(
            Message([Block("server_tool_use", name="web_search"),
                     text("Overcast.")]),
            Message([text("Oslo.")]),
        )
        responder.respond("weather?")
        responder.respond("capital?")
        assert responder.searches == 0


class TestTheBoundedHistory:
    def test_old_exchanges_fall_off_the_end(self):
        # Every turn resends the whole history, so an unbounded list makes
        # each turn slower and more expensive than the last.
        from eve import config
        responder = responder_with(*[
            Message([text(f"reply {i}")]) for i in range(20)
        ])
        for i in range(20):
            responder.respond(f"question {i}")
        assert len(responder.history) <= config.HISTORY_TURNS * 2 + 2

    def test_the_window_never_opens_on_an_assistant_turn(self):
        # The API requires the first message to be a user message, so
        # trimming has to walk forward to one rather than cutting blind.
        responder = responder_with(*[
            Message([text(f"reply {i}")]) for i in range(20)
        ])
        for i in range(20):
            responder.respond(f"question {i}")
            assert responder.client.calls[-1]["messages"][0]["role"] == "user"

    def test_a_tool_result_is_never_the_first_message_either(self):
        # The subtle half, and a real bug this suite caught. A tool result is
        # *also* a user message, so a trim that stops at the first "user"
        # role happily starts the window on one — with its matching tool_use
        # already trimmed away. The API rejects that, so the turn fails, and
        # it takes a long tool-using conversation to reach.
        responder = responder_with(*(
            [Message([tool_call("get_local_time", {})]),
             Message([text("late")])] * 12
        ))
        for i in range(12):
            responder.respond(f"question {i}")
            first = responder.client.calls[-1]["messages"][0]
            assert first["role"] == "user"
            assert isinstance(first["content"], str)


class TestWhereAWindowMayOpen:
    def test_a_plain_question_may_open_one(self):
        assert _opens({"role": "user", "content": "what time is it?"}) is True

    def test_an_assistant_turn_may_not(self):
        assert _opens({"role": "assistant", "content": [text("Oslo.")]}) is False

    def test_a_tool_result_may_not(self):
        assert _opens({"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "a", "content": "late"},
        ]}) is False

    def test_a_mixed_user_message_carrying_a_tool_result_may_not(self):
        assert _opens({"role": "user", "content": [
            {"type": "text", "text": "and also"},
            {"type": "tool_result", "tool_use_id": "a", "content": "late"},
        ]}) is False

    def test_a_structured_user_message_with_no_tool_result_may(self):
        assert _opens({"role": "user", "content": [
            {"type": "text", "text": "what time is it?"},
        ]}) is True


def _opens(message):
    return assistant._opens_a_window(message)


class TestWhatATurnCost:
    def test_tokens_are_summed_across_every_call_in_the_turn(self):
        responder = responder_with(
            Message([tool_call("get_local_time", {})],
                    usage=Usage(100, 20)),
            Message([text("late")], usage=Usage(150, 30)),
        )
        responder.respond("what time is it?")
        assert responder.last_usage == (250, 50)

    def test_the_cost_uses_list_prices(self):
        responder = responder_with(Message([text("Oslo.")],
                                           usage=Usage(1_000_000, 1_000_000)))
        responder.respond("capital?")
        assert responder.cost_usd() == pytest.approx(6.00)

    def test_web_search_is_billed_on_top(self):
        responder = responder_with(Message([
            Block("server_tool_use", name="web_search"),
            text("Overcast."),
        ], usage=Usage(0, 0)))
        responder.respond("weather?")
        assert responder.cost_usd() == pytest.approx(0.01)

    def test_a_turn_that_did_nothing_cost_nothing(self):
        responder = responder_with(Message([], usage=Usage(0, 0)))
        responder.respond("...")
        assert responder.cost_usd() == 0.0


class TestTheSeam:
    def test_claude_satisfies_the_protocol_main_actually_uses(self):
        # The point of the Protocol: when the local model lands, it replaces
        # ClaudeResponder and nothing else in the project changes.
        assert isinstance(assistant.ClaudeResponder, type)
        assert hasattr(assistant.ClaudeResponder, "respond")
        assert hasattr(assistant.Responder, "respond")

    def test_the_model_can_be_overridden_without_touching_code(self):
        responder = responder_with(Message([text("hi")]),
                                   model="claude-sonnet-5")
        responder.respond("hello")
        assert responder.client.calls[0]["model"] == "claude-sonnet-5"

    def test_the_reply_is_capped_so_a_runaway_answer_cannot_be_spoken(self):
        from eve import config
        responder = responder_with(Message([text("hi")]))
        responder.respond("hello")
        assert responder.client.calls[0]["max_tokens"] == config.MAX_TOKENS
