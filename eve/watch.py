"""Things she said she would come back about.

Every tool she has until now answers and is finished. That makes her a lookup
table with a voice: you ask, she replies, and the exchange is closed. A tool
that can reach you *later* is the difference between that and an assistant —
"tell me when the rain starts", "remind me in twenty minutes", "let me know
when that finishes" are all impossible without somewhere to put the promise.

This is that somewhere. A small JSON list beside memory.json, holding what she
owes and when it comes due, written atomically and read defensively for the
same reasons that file is — see memory._load for the version of this comment
that was paid for.

Two constraints shape the rest of it.

The first is that she must never speak over you. main._listen_loop only asks
for due items between captures, and the capture only yields early when nothing
has been said into it, so the worst case is that a reminder waits out the
sentence you were in the middle of.

The second is that a promise made to a room she has been shut out of is not
worth keeping on time. The microphone gets muted, and while it is muted she
cannot know whether anyone is there — so items are held rather than announced
into an empty room and marked delivered. They keep until she can hear again.
"""
from __future__ import annotations

import json
import os
import secrets
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

from eve import config

STORE = config.CONFIG_DIR / "reminders.json"

# She reads these aloud, one at a time, in a voice that takes about a second
# per five words. A queue longer than this is not a feature, it is an alarm
# clock going off in a language nobody asked for.
MAX_PENDING = 20
MAX_WHAT_CHARS = 200
# Four weeks. Not a policy about what is useful — a bound on what a wrong
# number can do, since the model computes this and an arithmetic slip should
# produce a refusal rather than a promise she keeps until the next power cut.
MAX_HORIZON_S = 30 * 86400
# Below this there is no point involving a file: by the time it is written the
# moment has passed, and "remind me in one second" is not a real request.
MIN_HORIZON_S = 5


def _load() -> list[dict]:
    """Everything outstanding, or nothing if the file is unusable.

    Defensive in the same two directions memory._load is, and for the same
    reason: this is read on the listen loop's hot path, between captures, so
    anything raising here takes her off the air rather than costing a feature.
    A file holding an object, a string, or a list with junk in it reads as
    empty rather than as an exception.
    """
    try:
        data = json.loads(STORE.read_text())
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(data, list):
        return []
    kept = []
    for item in data:
        if not isinstance(item, dict):
            continue
        try:
            kept.append({
                "id": str(item["id"]),
                "when": float(item["when"]),
                "what": str(item["what"]),
            })
        except (KeyError, TypeError, ValueError):
            continue          # one malformed row must not lose the others
    return kept


def _save(items: list[dict]) -> None:
    """Write atomically: a truncated list is worse than a stale one."""
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
            json.dump(items, file, indent=2)
        os.chmod(temporary, 0o600)   # it holds whatever was said in the room
        os.replace(temporary, STORE)
    except OSError:
        Path(temporary).unlink(missing_ok=True)
        raise


def add(seconds_from_now: float, what: str) -> str:
    """Promise to say `what` once `seconds_from_now` has passed."""
    what = what.strip()[:MAX_WHAT_CHARS]
    if not what:
        return "I need to know what to remind you about."
    try:
        delay = float(seconds_from_now)
    except (TypeError, ValueError):
        return "I could not make sense of when to do that."
    if delay < MIN_HORIZON_S:
        return "That is too soon to be worth writing down."
    if delay > MAX_HORIZON_S:
        return "That is further ahead than I will hold on to something."

    items = _load()
    if len(items) >= MAX_PENDING:
        return (f"I am already holding {MAX_PENDING} of these and will not "
                "take another until one comes due or is cancelled.")
    items.append({
        "id": secrets.token_hex(3),
        "when": time.time() + delay,
        "what": what,
    })
    try:
        _save(items)
    except OSError as exc:
        return f"I could not write that down: {exc}"
    return f"Noted. I will bring that up in {_spoken_delay(delay)}."


def _spoken_delay(seconds: float) -> str:
    """A delay in the words a person would use for it out loud."""
    if seconds < 90:
        return f"{round(seconds)} seconds"
    if seconds < 5400:
        return f"{round(seconds / 60)} minutes"
    if seconds < 172800:
        return f"{round(seconds / 3600)} hours"
    return f"{round(seconds / 86400)} days"


def pending(now: float | None = None) -> list[dict]:
    """Everything outstanding, soonest first."""
    now = time.time() if now is None else now
    return sorted(_load(), key=lambda item: item["when"])


def due(now: float | None = None) -> list[dict]:
    """Take everything that has come due, removing it from the store.

    Removed as it is handed over rather than after it is spoken, deliberately.
    The alternative loses a reminder only if she crashes between the two, and
    keeps one repeating forever if anything downstream raises every time —
    and an assistant stuck announcing the same thing every few seconds is a
    much worse failure than one dropped reminder.
    """
    now = time.time() if now is None else now
    items = _load()
    ready = [item for item in items if item["when"] <= now]
    if not ready:
        return []
    try:
        _save([item for item in items if item["when"] > now])
    except OSError:
        return []             # could not clear them; do not announce and repeat
    return sorted(ready, key=lambda item: item["when"])


def next_due_in(now: float | None = None) -> float:
    """Seconds until the next item comes due; inf when there is nothing.

    Called once per audio chunk, roughly thirty times a second, so it reads
    the file each time rather than holding a cache that a second process — or
    a hand edit — could make wrong. The file is a few hundred bytes and sits
    in the page cache; the loop it is on already runs a neural VAD.
    """
    now = time.time() if now is None else now
    items = _load()
    if not items:
        return float("inf")
    return min(item["when"] for item in items) - now


def cancel(which: str) -> str:
    """Drop one outstanding item, by id or by what it says."""
    which = which.strip().lower()
    items = _load()
    keep = [
        item for item in items
        if item["id"].lower() != which and which not in item["what"].lower()
    ]
    if len(keep) == len(items):
        return "I have nothing outstanding that matches that."
    try:
        _save(keep)
    except OSError as exc:
        return f"I could not update my notes: {exc}"
    return f"Dropped {len(items) - len(keep)} of them."


def as_spoken_list(now: float | None = None) -> str:
    """What is outstanding, as a sentence rather than a table."""
    items = pending(now)
    if not items:
        return "You have not asked me to come back about anything."
    now = time.time() if now is None else now
    return " ".join(
        f"In {_spoken_delay(max(0.0, item['when'] - now))}, {item['what']}."
        for item in items
    )


TOOLS = [
    {
        "name": "set_reminder",
        "description": (
            "Promise to say something out loud after a delay, and be held to "
            "it — this is the only way you can start a conversation rather "
            "than answer one. Call it whenever you are asked to remind, tell, "
            "or nudged about something later, or asks you to time something. "
            "Give the delay in seconds from now; for a request phrased as a "
            "time of day rather than a delay, call get_local_time first and "
            "work out the difference. Say what you will bring up in the words "
            "you will use when the moment comes, since that is read aloud "
            "verbatim."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "seconds_from_now": {
                    "type": "number",
                    "description": "How long to wait, in seconds.",
                },
                "what": {
                    "type": "string",
                    "description": "What to say when it comes due.",
                },
            },
            "required": ["seconds_from_now", "what"],
        },
    },
    {
        "name": "list_reminders",
        "description": (
            "Say what you have promised to come back about and how long is "
            "left on each. Call this when you are asked what is outstanding, what "
            "timers are running, or what you are still holding."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "cancel_reminder",
        "description": (
            "Drop something you promised to come back about. Call this when "
            "a reminder or timer is cancelled. Match on the words used for "
            "it; you do not need an identifier."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "which": {
                    "type": "string",
                    "description": "Words from the reminder, or its id.",
                },
            },
            "required": ["which"],
        },
    },
]

# Annotated for the reason tools.HANDLERS is: these take different arguments,
# so the inferred value type collapses to something uncallable and the
# dispatch in tools.run_tool stops being checked.
HANDLERS: dict[str, Callable[..., str]] = {
    "set_reminder": lambda seconds_from_now, what: add(seconds_from_now, what),
    "list_reminders": as_spoken_list,
    "cancel_reminder": cancel,
}
