"""What gets written down, and what does not.

This thing transcribes everything said near it, and most of that is not
addressed to it — half a conversation, the television, someone on the phone
in the next room. Printing those transcripts made wake-word misses easy to
debug, and it also meant a running record of the household went into the
systemd journal, on disk, for as long as journald felt like keeping it.

By default, transcript and reply content is not written to the process log.
Eve still records timing, token counts, cost, and operational errors. Stable
facts selected for memory are stored separately in ``memory.json`` as
documented in SECURITY.md.

Two ways to see content:

    VOICE_DEBUG=1                 for a debugging session; goes to the journal
    running it in a terminal      content prints to the terminal

Eve does not write interactive output to a file, but terminal scrollback or
session logging may retain it. Under systemd stdout is a pipe to journald —
not a terminal — so the service stays quiet without needing to be told.
"""
from __future__ import annotations

import os
import sys

_TRUE = ("1", "true", "yes", "on")

# Set VOICE_DEBUG=1 to record transcripts to the journal for a session.
DEBUG = os.environ.get("VOICE_DEBUG", "").strip().lower() in _TRUE

# Interactive runs show content; the terminal or session may retain it.
_INTERACTIVE = sys.stderr.isatty()

SHOW_CONTENT = DEBUG or _INTERACTIVE


def content(message: str) -> None:
    """Print something a person said or was told — suppressed by default."""
    if SHOW_CONTENT:
        print(message, file=sys.stderr)


def spoken(label: str, text: str) -> None:
    """A transcript, tagged with why it is being shown."""
    content(f"({label}: {text!r})")


def status(message: str) -> None:
    """Machine state: timings, costs, model names, errors.

    Always logged. Nothing here carries what was said, which is what makes it
    safe to keep — a turn that cost half a cent and took three seconds is
    diagnosable without knowing the question.
    """
    print(message, file=sys.stderr)


def banner() -> str:
    """One line for the log, so it is obvious which mode is running."""
    if DEBUG:
        return "transcript logging ON (VOICE_DEBUG) — content is going to disk"
    return "transcript logging off — remembered facts use memory.json"
