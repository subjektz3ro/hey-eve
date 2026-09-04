"""The conversation: text in, spoken-length text out, tools in between.

The Responder protocol is the second half of the backend seam. Today it is
Claude over the API; with an accelerator fitted it can be a local model,
and main.py will not know the difference.
"""
from __future__ import annotations

import os
from typing import Any, Protocol, cast

from eve import memory
from eve import tools
from eve import config


class Responder(Protocol):
    """One conversational turn: what was said -> what to say back."""

    def respond(self, user_text: str, on_state=None) -> str: ...


# US dollars per million tokens, input and output, at list price. VOICE_MODEL
# is a documented knob and config.py actively suggests claude-sonnet-5, so the
# table has to cover more than the default or the cost line lies.
PRICES: dict[str, tuple[float, float]] = {
    "claude-haiku-4-5":  (1.00, 5.00),
    "claude-sonnet-5":   (2.00, 10.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-opus-5":     (5.00, 25.00),
    "claude-opus-4-8":   (5.00, 25.00),
    "claude-fable-5":    (10.00, 50.00),
}

# Long enough that nothing real reaches it, short enough that a dead
# connection costs one turn rather than ten minutes. See ClaudeResponder.
TIMEOUT_S = float(os.environ.get("VOICE_API_TIMEOUT_S", "30"))

# Prompt caching is deliberately not used here, and the reason is worth
# recording so it is not "fixed" later: the whole prefix — tools plus system
# plus remembered facts — is about 1,400 tokens, and Haiku 4.5's minimum
# cacheable prefix is 4,096. A cache_control breakpoint below the minimum is
# not an error; it simply never caches, so the change would look like a
# saving and be nothing at all. Worth revisiting only if the prompt triples
# or the model changes to one with a lower floor.


def _opens_a_window(message: dict) -> bool:
    """Whether a trimmed history may legally begin at this message.

    Checking `role == "user"` is not enough, and that is the whole point of
    this function. A tool result is *also* a user message — it is how the
    loop hands a tool's output back — so a cut that stops at the first
    "user" role can land on a tool_result whose matching tool_use has just
    been trimmed off the front. The API rejects that outright, so the turn
    fails, and it only happens after enough tool-using exchanges to overflow
    the window: a bug that needs a long conversation to appear.
    """
    if message["role"] != "user":
        return False
    content = message["content"]
    if isinstance(content, str):
        return True
    return not any(
        isinstance(block, dict) and block.get("type") == "tool_result"
        for block in content
    )


def _load_key() -> str:
    """Read the API key from its 0600 env file, without exporting it."""
    key = config.secret("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError(
            f"no ANTHROPIC_API_KEY: set it in {config.VOICE_ENV} "
            "or in the environment"
        )
    return key


class ClaudeResponder:
    """Claude with the bar tools attached, holding a bounded history."""

    def __init__(self, model: str | None = None) -> None:
        import anthropic

        # The SDK's default read timeout is 600 seconds, which is sized for
        # batch generation rather than for someone standing in a room waiting
        # for an answer. On a streamed response httpx applies it per socket
        # read — the maximum gap between chunks — so a TCP connection that
        # dies mid-stream without a FIN parks her on THINKING for ten minutes
        # with the microphone shut and no way to interrupt her.
        #
        # A reply is capped at 300 tokens and measures 1.37s on Haiku. Thirty
        # seconds is a wedge detector, not a limit anything real approaches.
        self.client = anthropic.Anthropic(
            api_key=_load_key(), timeout=TIMEOUT_S, max_retries=3)
        self.model = model or config.MODEL
        self.history: list[dict] = []
        self.last_usage = (0, 0)
        self.searches = 0

    def _trim(self) -> None:
        """Keep the last N exchanges.

        Every turn resends the whole history, so an unbounded list makes each
        turn slower and more expensive than the last.
        """
        limit = config.HISTORY_TURNS * 2
        if len(self.history) > limit:
            # Never start the window on an assistant turn or a tool result —
            # see _opens_a_window for why the second half needs saying.
            cut = len(self.history) - limit
            while cut < len(self.history) \
                    and not _opens_a_window(self.history[cut]):
                cut += 1
            self.history = self.history[cut:]

    def respond(self, user_text: str, on_state=None) -> str:
        self.history.append({"role": "user", "content": user_text})
        self._trim()

        spoken: list[str] = []
        input_tokens = output_tokens = 0
        self.searches = 0

        # Tool loop. Bounded because a model that keeps calling tools would
        # otherwise hold the conversation open indefinitely while the person
        # waits in silence.
        for _ in range(4):
            # Streamed so the caller learns what is happening *during* the
            # turn: a web search can take seconds, and a panel that says
            # "searching" beats one that just sits on "processing".
            # Cast rather than annotate: both arguments are the SDK's
            # TypedDict shapes, but they are assembled here as plain dicts —
            # the tool list is built by concatenating optional integrations,
            # and history holds SDK content blocks alongside dicts we wrote.
            # The seam that matters is Responder above, and that one is typed.
            # Anthropic applies max_uses to one API request. A spoken turn can
            # span several requests while client tools or a paused server tool
            # are resolved, so lower (and eventually remove) the search tool
            # to enforce the advertised budget across the complete turn.
            remaining_searches = max(
                0, tools.WEB_SEARCH_LIMIT - self.searches
            )
            request_tools = [
                (
                    {**tool, "max_uses": remaining_searches}
                    if tool is tools.WEB_SEARCH
                    else tool
                )
                for tool in tools.TOOLS
                if tool is not tools.WEB_SEARCH or remaining_searches
            ]
            with self.client.messages.stream(
                model=self.model,
                max_tokens=config.MAX_TOKENS,
                system=config.system_prompt() + memory.as_prompt(),
                tools=cast(Any, request_tools),
                messages=cast(Any, self.history),
            ) as stream:
                for event in stream:
                    if on_state is None or event.type != "content_block_start":
                        continue
                    kind = event.content_block.type
                    if kind == "server_tool_use":
                        on_state("searching")
                    elif kind in ("text", "tool_use"):
                        on_state("thinking")
                message = stream.get_final_message()
            input_tokens += message.usage.input_tokens
            output_tokens += message.usage.output_tokens

            spoken += [b.text for b in message.content if b.type == "text"]
            self.history.append({"role": "assistant", "content": message.content})

            # Only client-side tools appear as `tool_use`. Web search runs on
            # Anthropic's servers and arrives already answered, so there is
            # nothing to execute for it here.
            calls = [b for b in message.content if b.type == "tool_use"]
            self.searches += sum(
                1
                for b in message.content
                if b.type == "server_tool_use"
                and getattr(b, "name", "") == "web_search"
            )
            if not calls:
                # A server-side tool loop that hits its own iteration limit
                # stops with `pause_turn`; the turn is unfinished, and
                # re-sending it (the assistant turn is already in history)
                # resumes where it left off.
                if message.stop_reason == "pause_turn":
                    continue
                break

            self.history.append({
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": call.id,
                        "content": tools.run_tool(call.name, dict(call.input)),
                    }
                    for call in calls
                ],
            })

        self.last_usage = (input_tokens, output_tokens)
        return " ".join(part.strip() for part in spoken if part.strip())

    def cost_usd(self) -> float | None:
        """Cost of the last turn at list prices, for the model actually used.

        Rates are selected for the configured model so the journal does not
        report the default model's cost for a different model. An unlisted
        model has no trustworthy local price, so its cost is left unknown.
        """
        prices = PRICES.get(self.model)
        if prices is None:
            return None
        input_tokens, output_tokens = self.last_usage
        per_in, per_out = prices
        return (
            input_tokens / 1e6 * per_in
            + output_tokens / 1e6 * per_out
            + self.searches * 0.01          # web search is $10 per thousand
        )
