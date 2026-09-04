"""What the assistant can look up, as opposed to what it already knows.

Deliberately small and read-only. The assistant answers most questions from
the model alone; these exist for the things a general model cannot know —
the state of this particular machine and the device attached to it.

Everything here returns prose rather than JSON, because the model reads the
result and paraphrases it aloud, and a sentence survives that round trip
better than a structure it has to narrate.
"""
from __future__ import annotations

import shutil
import subprocess
import time
from collections.abc import Callable
from pathlib import Path

from eve import bar
from eve import memory
from eve import watch


def get_system_status() -> str:
    """Health of the Linux host this assistant runs on."""
    parts: list[str] = []

    try:
        temp_c = int(Path("/sys/class/thermal/thermal_zone0/temp").read_text()) / 1000
        parts.append(f"CPU temperature {temp_c:.0f} degrees Celsius.")
    except (OSError, ValueError):
        pass

    try:
        uptime_s = float(Path("/proc/uptime").read_text().split()[0])
        parts.append(f"Up {uptime_s / 86400:.1f} days.")
    except (OSError, ValueError):
        pass

    try:
        load1 = Path("/proc/loadavg").read_text().split()[0]
        parts.append(f"Load average {load1}.")
    except OSError:
        pass

    try:
        meminfo = {
            line.split(":")[0]: int(line.split()[1])
            for line in Path("/proc/meminfo").read_text().splitlines()[:5]
        }
        total_gb = meminfo["MemTotal"] / 1048576
        free_gb = meminfo["MemAvailable"] / 1048576
        parts.append(f"Memory {free_gb:.1f} of {total_gb:.1f} gigabytes free.")
    except (OSError, KeyError, ValueError, IndexError):
        pass

    try:
        usage = shutil.disk_usage("/")
        parts.append(
            f"Disk {usage.free / 1e9:.0f} of {usage.total / 1e9:.0f} gigabytes free."
        )
    except OSError:
        pass

    try:
        throttled = subprocess.run(
            ["vcgencmd", "get_throttled"], capture_output=True, text=True, timeout=3
        ).stdout.strip()
        if throttled and not throttled.endswith("=0x0"):
            parts.append("The system reports throttling.")
    except (OSError, subprocess.SubprocessError):
        pass

    return " ".join(parts) or "Could not read this machine's status."


def get_local_time() -> str:
    """The machine's own clock and timezone."""
    return time.strftime("It is %-I:%M %p on %A, %B %-d, %Y, timezone %Z.")


# Runs on Anthropic's servers, not here — so it has no handler below and the
# tool loop never executes it; results arrive in the same response. Haiku 4.5
# predates the dynamic-filtering variant, so it takes the basic version.
# Billed per search (about a cent each), hence the cap.
WEB_SEARCH_LIMIT = 3
WEB_SEARCH = {
    "type": "web_search_20250305",
    "name": "web_search",
    "max_uses": WEB_SEARCH_LIMIT,
}

TOOLS = [
    WEB_SEARCH,
    {
        "name": "get_system_status",
        "description": (
            "Read the health of the Linux host this assistant runs on: CPU "
            "temperature, uptime, load average, free memory and disk, and "
            "whether it is throttling. Call this when you are asked how the host "
            "is doing, whether it is hot, how long it has been up, or whether it "
            "is running out of space or memory."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_local_time",
        "description": (
            "Get the current date and time from this machine's own clock. Call "
            "this whenever the answer depends on what time or day it is now."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    # Offered only when a Barkeep token is configured: an integration, not
    # a dependency.
    *(bar.TOOLS if bar.available() else []),
    *memory.TOOLS,
    # The only tools here that do not answer and finish. Everything above
    # closes the exchange it was called in; these leave something behind that
    # makes her speak again later, unprompted — see eve/watch.py.
    *watch.TOOLS,
]

# Annotated because the merge below mixes zero-argument handlers with
# keyword ones, and the inferred union is `object` — which makes the call in
# run_tool untypeable and hides a genuine mistake behind a bare except.
HANDLERS: dict[str, Callable[..., str]] = {
    "get_system_status": get_system_status,
    "get_local_time": get_local_time,
    **(bar.HANDLERS if bar.available() else {}),
    **memory.HANDLERS,
    **watch.HANDLERS,
}


def run_tool(name: str, arguments: dict) -> str:
    """Execute one tool call, returning text for the model to read."""
    handler = HANDLERS.get(name)
    if handler is None:
        return f"No such tool: {name}"
    try:
        return handler(**arguments)
    except Exception as exc:  # a tool crash must not end the conversation
        return f"The {name} tool failed: {exc}"
