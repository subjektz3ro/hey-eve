"""What the assistant is allowed to remember between conversations.

A small JSON file it writes to itself, keyed by topic so a fact can be
corrected rather than accumulating duplicates ("location" gets updated, not
appended). The contents are folded into the system prompt each turn, so
remembered things are simply known — there is no lookup call to forget to
make.

Deliberately not a general key-value store: it holds bounded, stable facts and
preferences that the model decides will be useful later, including anything
its user explicitly asks it to keep. The file remains readable and correctable
by hand.
"""
from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Callable
from pathlib import Path

from eve import config

STORE = config.CONFIG_DIR / "memory.json"

# A spoken assistant cannot usefully recite a hundred facts, and every one of
# them rides in the prompt on every turn.
MAX_FACTS = 40
MAX_FACT_CHARS = 200


def _load() -> dict[str, str]:
    try:
        data = json.loads(STORE.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    # The guard has to come before .items(), not inside the comprehension.
    # Where it used to sit it was evaluated per item — after the call it was
    # meant to protect — so it could never be False and never guarded
    # anything. A file holding `[]`, `null`, `42` or a bare string raised
    # AttributeError, which is not in the except above, out of as_prompt(),
    # which runs on every single turn. She booted fine, drew her face, and
    # then died the moment anyone spoke to her, forever.
    #
    # Every malformed *edit* — a stray comma, a missing brace, a truncated
    # write — is already JSONDecodeError. This is the narrower case of valid
    # JSON that is not an object, which is what someone typing `[]` to empty
    # the file by hand produces. The docstring above promises this file is
    # correctable by hand; this is what keeping that promise costs.
    if not isinstance(data, dict):
        return {}
    return {str(topic): str(fact) for topic, fact in data.items()}


def _save(facts: dict[str, str]) -> None:
    """Write atomically: a truncated memory file is worse than a stale one."""
    # 0700, not whatever the umask says. SECURITY.md promises "0600, inside a
    # 0700 directory", and install.sh does chmod it — but nothing had to have
    # run install.sh. Set the key in the environment instead and the first
    # thing she remembers creates this directory at 0755, quietly making the
    # documented guarantee false. The files inside are 0600 either way, so
    # what leaked was the listing rather than the contents; the claim should
    # still be true.
    #
    # chmod as well as mode=, because mode= is masked by the umask and this is
    # a guarantee rather than a preference. parents= is left at the default:
    # ~/.config is normally 0755 and is not ours to tighten.
    STORE.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    STORE.parent.chmod(0o700)
    handle, temporary = tempfile.mkstemp(dir=STORE.parent, suffix=".tmp")
    try:
        with os.fdopen(handle, "w") as file:
            json.dump(facts, file, indent=2, sort_keys=True)
        os.chmod(temporary, 0o600)  # it holds whatever it was told
        os.replace(temporary, STORE)
    except OSError:
        Path(temporary).unlink(missing_ok=True)
        raise


def remember(topic: str, fact: str) -> str:
    """Store or correct one fact under `topic`."""
    topic, fact = topic.strip().lower()[:60], fact.strip()[:MAX_FACT_CHARS]
    if not topic or not fact:
        return "I need both a topic and something to remember about it."

    facts = _load()
    updating = topic in facts
    if not updating and len(facts) >= MAX_FACTS:
        return (
            f"I am already holding {MAX_FACTS} things and cannot take more "
            "until something is forgotten."
        )
    facts[topic] = fact
    try:
        _save(facts)
    except OSError as exc:
        return f"I could not write that down: {exc}"
    return f"{'Updated' if updating else 'Noted'}: {topic}."


def forget(topic: str) -> str:
    """Drop one remembered fact."""
    facts = _load()
    topic = topic.strip().lower()[:60]
    if topic not in facts:
        return f"I have nothing stored under {topic}."
    facts.pop(topic)
    try:
        _save(facts)
    except OSError as exc:
        return f"I could not update my notes: {exc}"
    return f"Forgotten: {topic}."


def as_prompt() -> str:
    """The remembered facts, as a block to append to the system prompt."""
    facts = _load()
    if not facts:
        return ""
    lines = "\n".join(f"- {topic}: {fact}" for topic, fact in sorted(facts.items()))
    return (
        "\n\nWhat you already know about the person you are talking to. These "
        "are established facts, "
        "not background colour: before answering anything, check whether one "
        "of them settles a detail the question leaves out, and if it does, "
        "use it instead of the common case. Never mention having looked "
        "them up.\n"
        f"{lines}"
    )


TOOLS = [
    {
        "name": "remember",
        "description": (
            "Store a fact so it survives into future conversations. Call this "
            "whenever you are told to remember something, given a lasting "
            "preference, or gives you a detail you will clearly need again — "
            "where they live, how they take their coffee, what they are working on. "
            "Use a short stable topic like 'location' or 'coffee' so the fact "
            "can be corrected later rather than duplicated."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "Short stable key, e.g. 'location'.",
                },
                "fact": {
                    "type": "string",
                    "description": "The fact itself, in a sentence.",
                },
            },
            "required": ["topic", "fact"],
        },
    },
    {
        "name": "forget",
        "description": (
            "Delete something previously remembered. Call this when you are told "
            "to forget a topic or tells you a stored fact is wrong and should "
            "simply be dropped."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "topic": {"type": "string", "description": "The topic to drop."},
            },
            "required": ["topic"],
        },
    },
]

# Annotated for the same reason tools.HANDLERS is: remember() and forget()
# take different arguments, so the inferred value type collapses to something
# uncallable and the dispatch in tools.run_tool stops being checked.
HANDLERS: dict[str, Callable[..., str]] = {"remember": remember, "forget": forget}
