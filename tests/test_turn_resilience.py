"""A turn that fails should cost a turn.

eve already refused to let a lost voice become a lost assistant — _speak has
been guarded, with a comment recording the crash loop that earned it. The
model call one line above it was not, and the model is the one component in
this pipeline that belongs to somebody else and is reached over a domestic
connection. It will fail; it should not take the process with it.

The rest of this file is the same principle applied to the other ways a turn
can end without a reply: a permanently unreadable memory file, an exhausted
tool loop, a missing key at startup.
"""
from __future__ import annotations

import json

import pytest

from eve import assistant, main, memory


class Recorder:
    """A display that remembers what it was asked to show."""

    def __init__(self):
        self.states = []

    def state(self, name, level=0.0, note=""):
        self.states.append((name, note))

    def close(self):
        pass

    @property
    def last(self):
        return self.states[-1] if self.states else (None, None)


def responder(reply="Oslo.", raises=None):
    class Fake:
        last_usage = (10, 5)

        def respond(self, said, on_state=None):
            if raises is not None:
                raise raises
            return reply

        def cost_usd(self):
            return 0.0

    return Fake()


class TestTheModelCallIsGuarded:
    @pytest.mark.parametrize("failure", [
        ConnectionError("connection reset by peer"),
        RuntimeError("overloaded_error"),
        TimeoutError("read timeout"),
        ValueError("401 authentication_error"),
    ])
    def test_an_api_failure_does_not_end_the_process(self, failure):
        # It used to: main() catches only KeyboardInterrupt, so anything from
        # respond() unwound all the way out and the unit restarted her
        # mid-conversation with no explanation on the panel.
        main._turn(responder(raises=failure), "what time is it?",
                   Recorder(), quiet=False)

    def test_the_reason_reaches_the_journal(self, capsys):
        main._turn(responder(raises=RuntimeError("overloaded_error")),
                   "hello", Recorder(), quiet=False)
        printed = capsys.readouterr().err
        assert "could not answer" in printed
        assert "overloaded_error" in printed

    def test_the_panel_stops_saying_it_is_thinking(self):
        display = Recorder()
        main._turn(responder(raises=RuntimeError("boom")), "hello",
                   display, quiet=False)
        assert display.last[0] != "thinking", \
            "she goes back to listening; the face has to say so"

    def test_a_working_turn_is_unaffected(self, monkeypatch):
        monkeypatch.setattr(main, "_speak", lambda reply, display: None)
        display = Recorder()
        main._turn(responder("Oslo."), "capital of Norway?",
                   display, quiet=False)
        assert "synthesizing" in [name for name, _ in display.states]


class TestAnEmptyReplyIsNotNothingHappening:
    def test_the_panel_does_not_stay_on_thinking(self, monkeypatch):
        # respond() returns "" when the four-iteration tool loop exhausts with
        # calls still pending, or when a turn comes back with only tool_use
        # blocks. Every state call after respond() sat inside `if reply:`, so
        # the face held THINKING while she was in fact listening.
        monkeypatch.setattr(main, "_speak", lambda reply, display: None)
        display = Recorder()
        main._turn(responder(""), "something unanswerable", display, quiet=False)
        assert display.last[0] == "engaged"
        assert "LISTENING" in display.last[1]

    def test_nothing_is_spoken(self, monkeypatch):
        spoken = []
        monkeypatch.setattr(main, "_speak",
                            lambda reply, display: spoken.append(reply))
        main._turn(responder(""), "hello", Recorder(), quiet=False)
        assert spoken == []


class TestAHandEditedMemoryFile:
    """Valid JSON that is not an object.

    The isinstance guard sat inside the comprehension, after the .items() call
    it was meant to protect, so it could never be False. AttributeError is not
    in _load's except clause and as_prompt() runs on every turn, so a file
    holding `[]` killed her every time anyone spoke — she booted, drew her
    face, showed READY, and died on the first question. Forever.
    """

    @pytest.mark.parametrize("body", ["[]", "[1, 2, 3]", "null", "42",
                                      '"a string"', "true",
                                      '[{"location": "Fictionopolis"}]'])
    def test_it_reads_as_empty_rather_than_crashing(self, body, isolated_settings):
        memory.STORE.write_text(body)
        assert memory._load() == {}
        assert memory.as_prompt() == ""

    def test_every_turn_would_have_hit_it(self, isolated_settings):
        # as_prompt is called from assistant.respond, so this is not an
        # obscure path — it is the path.
        memory.STORE.write_text("[]")
        assert memory.as_prompt() == ""

    def test_the_file_is_still_usable_afterwards(self, isolated_settings):
        memory.STORE.write_text("[]")
        assert memory.remember("location", "Fictionopolis").startswith("Noted")
        assert memory._load() == {"location": "Fictionopolis"}

    def test_a_real_store_is_untouched(self, isolated_settings):
        memory.STORE.write_text(json.dumps({"location": "Fictionopolis"}))
        assert memory._load() == {"location": "Fictionopolis"}

    def test_unparseable_json_still_reads_as_empty(self, isolated_settings):
        # The case that was already covered; it must not regress.
        memory.STORE.write_text("{not json at all")
        assert memory._load() == {}


class TestTheCostLineIsAboutTheModelInUse:
    def test_haiku_is_priced_as_haiku(self):
        r = assistant.ClaudeResponder.__new__(assistant.ClaudeResponder)
        r.model, r.last_usage, r.searches = "claude-haiku-4-5", (1_000_000, 0), 0
        assert r.cost_usd() == pytest.approx(1.00)

    def test_a_costlier_model_is_not_priced_as_haiku(self):
        # A configured model must use its own list price rather than Haiku's.
        r = assistant.ClaudeResponder.__new__(assistant.ClaudeResponder)
        r.model, r.last_usage, r.searches = "claude-sonnet-5", (1_000_000, 0), 0
        assert r.cost_usd() == pytest.approx(2.00)

    def test_output_tokens_cost_more_than_input(self):
        r = assistant.ClaudeResponder.__new__(assistant.ClaudeResponder)
        r.model, r.searches = "claude-haiku-4-5", 0
        r.last_usage = (0, 1_000_000)
        assert r.cost_usd() == pytest.approx(5.00)

    def test_an_unknown_model_does_not_invent_a_price_or_raise(self):
        r = assistant.ClaudeResponder.__new__(assistant.ClaudeResponder)
        r.model, r.last_usage, r.searches = "claude-something-new", (1_000, 0), 0
        assert r.cost_usd() is None

    def test_searches_are_still_counted(self):
        r = assistant.ClaudeResponder.__new__(assistant.ClaudeResponder)
        r.model, r.last_usage, r.searches = "claude-haiku-4-5", (0, 0), 3
        assert r.cost_usd() == pytest.approx(0.03)

    def test_every_price_covers_both_directions(self):
        for model, price in assistant.PRICES.items():
            assert len(price) == 2, model
            assert price[1] > price[0] > 0, model

    def test_the_default_model_is_priced(self):
        from eve import config
        assert config.MODEL in assistant.PRICES


class TestTheClientIsBounded:
    def test_a_wedged_connection_cannot_hold_her_for_ten_minutes(self):
        # The SDK default is a 600s *per-chunk* read timeout on a stream, so a
        # TCP connection that dies without a FIN parks the panel on THINKING
        # with the microphone shut and no way to interrupt.
        assert assistant.TIMEOUT_S <= 60

    def test_it_is_long_enough_for_a_real_turn(self):
        # A reply is capped at 300 tokens and measures 1.37s on Haiku.
        assert assistant.TIMEOUT_S >= 15

    def test_the_client_is_built_with_it(self, monkeypatch):
        seen = {}

        class FakeAnthropic:
            def __init__(self, **kwargs):
                seen.update(kwargs)

        monkeypatch.setattr(assistant, "_load_key", lambda: "sk-test")
        monkeypatch.setitem(
            __import__("sys").modules, "anthropic",
            type("m", (), {"Anthropic": FakeAnthropic}))
        assistant.ClaudeResponder()
        assert seen["timeout"] == assistant.TIMEOUT_S
