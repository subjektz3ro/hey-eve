#!/usr/bin/env python3
"""Work out which way the touchscreen is wired, and write down the answer.

Run it on the Pi, at a terminal you can see:

    cd ~/hey-eve && ~/.local/bin/uv run --no-sync python scripts/touch-probe.py

A resistive panel is bonded to the glass, not to the framebuffer, so whether
its X runs the same way as the picture's — and whether the two are swapped
outright — is a property of how the thing was assembled. It cannot be derived
and it must not be assumed; it has to be touched.

Touch three corners when asked. It prints the three settings and offers to
write them to ~/.config/eve/env for you.

    --check   touch a corner, have it name that corner. Confirms the
              answer the walk produced.
    --live    stream what the panel reports for twenty seconds, and say
              whether each axis is healthy. For when something is wrong
              rather than merely unknown.
"""
from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

from eve import config, touch

HOLD_S = 3.0            # how long a corner must be held to be measured
MIN_SAMPLES = 40        # ...and how many readings that has to produce

CORNERS = ("TOP-LEFT", "TOP-RIGHT", "BOTTOM-LEFT")
SETTINGS = ("VOICE_TOUCH_SWAP", "VOICE_TOUCH_FLIP_X", "VOICE_TOUCH_FLIP_Y")


def _write_settings(target: Path, lines: list[str]) -> None:
    """Replace the touch settings without exposing or truncating the env file."""
    existing = target.read_text().splitlines() if target.exists() else []
    kept = [line for line in existing
            if line.partition("=")[0].strip() not in SETTINGS]
    body = "\n".join(kept + lines) + "\n"

    target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    target.parent.chmod(0o700)
    handle, temporary = tempfile.mkstemp(
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary)
    try:
        os.fchmod(handle, 0o600)
        with os.fdopen(handle, "w") as file:
            file.write(body)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_path, target)
        target.chmod(0o600)
    except BaseException:
        try:
            os.close(handle)
        except OSError:
            pass
        temporary_path.unlink(missing_ok=True)
        raise


def open_panel():
    """The touchscreen, reading raw. None if there is not one."""
    device = touch.find_device()
    if device is None:
        print("No touchscreen found in /proc/bus/input/devices.",
              file=sys.stderr)
        return None
    # Raw, so the probe measures the panel rather than whatever the settings
    # currently claim about it — including a previous run's answer. --check is
    # the exception: its whole job is to try the answer on.
    if "--check" not in sys.argv:
        for name in ("SWAP_XY", "FLIP_X", "FLIP_Y"):
            setattr(touch, name, False)
    pad = touch.Touch(device)
    pad.start()
    if pad._thread is None:
        print(f"Could not read {device}. Is this user in the `input` group?",
              file=sys.stderr)
        return None
    print(f"Touchscreen: {device}")
    return pad


def measure(pad, label: str):
    """Wait for a held touch on one corner and return the median of it."""
    print(f"\n  Press and HOLD the {label} corner.", flush=True)
    while not pad.latest()[2]:
        time.sleep(0.02)

    xs, ys = [], []
    end = time.monotonic() + HOLD_S
    while time.monotonic() < end:
        x, y, down = pad.latest()
        if down:
            xs.append(x)
            ys.append(y)
        left = end - time.monotonic()
        print(f"\r    holding... {left:3.1f}s  ({len(xs)} readings) ",
              end="", flush=True)
        time.sleep(0.02)

    if len(xs) < MIN_SAMPLES:
        print(f"\r    lost it — only {len(xs)} readings. Hold it down for "
              f"the whole {HOLD_S:.0f} seconds.        ")
        return None
    xs.sort()
    ys.sort()
    spot = (xs[len(xs) // 2], ys[len(ys) // 2])
    print(f"\r    got it: x={spot[0]:+.3f} y={spot[1]:+.3f}"
          f"   ({len(xs)} readings)        ")
    print("    now lift your finger off.", flush=True)
    while pad.latest()[2]:
        time.sleep(0.02)
    return spot


def solve(seen: dict) -> tuple[bool, bool, bool] | None:
    """Which axis is which, and which way round, from three corners."""
    top_left, top_right, bottom_left = (seen[name] for name in CORNERS)
    across = (top_right[0] - top_left[0], top_right[1] - top_left[1])
    down = (bottom_left[0] - top_left[0], bottom_left[1] - top_left[1])

    print("\n" + "-" * 56)
    print(f"  going left to right, the panel moved  x{across[0]:+.3f}"
          f"  y{across[1]:+.3f}")
    print(f"  going top to bottom, the panel moved  x{down[0]:+.3f}"
          f"  y{down[1]:+.3f}")
    print("-" * 56)

    if max(abs(v) for v in across) < 0.25 or max(abs(v) for v in down) < 0.25:
        print("\n  Two of those corners read almost the same. Either the same")
        print("  spot got touched twice, or the presses were too light —")
        print("  it is resistive, so it wants a fingernail, not a fingertip.")
        return None

    # Whichever axis moved more going across is the one carrying the
    # picture's X; its sign says whether it runs the same way.
    swap = abs(across[1]) > abs(across[0])
    if swap:
        return swap, across[1] < 0, down[0] < 0
    return swap, across[0] < 0, down[1] < 0


def save(answer: tuple[bool, bool, bool]) -> None:
    """Offer to put the answer in the env file, and do it."""
    lines = [f"{name}={1 if value else 0}"
             for name, value in zip(SETTINGS, answer, strict=True)]
    print("\n  The answer:\n")
    for line in lines:
        print(f"      {line}")

    target = config.VOICE_ENV
    try:
        reply = input(f"\n  Write these to {target}? [Y/n] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return
    if reply not in ("", "y", "yes"):
        print("  Left alone. Paste them in yourself when you like.")
        return

    try:
        _write_settings(target, lines)
    except OSError as exc:
        print(f"  Could not write it: {exc}", file=sys.stderr)
        return
    print("  Written. Restart her to pick it up:\n"
          "      sudo systemctl restart eve@$(id -un)")


def check(pad) -> int:
    """Say where it thinks you are touching, with the settings applied.

    The corner walk solves the orientation from three presses; this is the
    other half of trusting it. Touch a corner, read back the name of that
    corner. Arithmetic that is wrong in a way three samples cannot see is
    obvious the moment it is asked to name where your finger is.
    """
    print(f"\n  Settings in use:  swap={touch.SWAP_XY}  "
          f"flip_x={touch.FLIP_X}  flip_y={touch.FLIP_Y}")
    print("  Touch any corner and it will name it. Twenty seconds.\n")
    end = time.monotonic() + 20.0
    said = None
    while time.monotonic() < end:
        x, y, down = pad.latest()
        if down:
            where = (("TOP" if y < -0.25 else "BOTTOM" if y > 0.25 else "MIDDLE")
                     + " " +
                     ("LEFT" if x < -0.25 else "RIGHT" if x > 0.25 else "CENTRE"))
            if where != said:
                said = where
                print(f"  x={x:+.2f} y={y:+.2f}   ->  {where}", flush=True)
        else:
            said = None
        time.sleep(0.03)
    print("\n  If those names matched the corners you touched, it is right.")
    return 0


def live(pad) -> int:
    """Stream what the panel reports, and judge each axis."""
    seconds = 20.0
    print(f"\n  Recording for {seconds:.0f} seconds. Drag a fingernail all")
    print("  over the panel — the whole surface, edge to edge, both ways.\n")
    lo, hi, samples = [9.0, 9.0], [-9.0, -9.0], 0
    start = time.monotonic()
    end = start + seconds
    while time.monotonic() < end:
        x, y, down = pad.latest()
        if down:
            samples += 1
            lo[0], hi[0] = min(lo[0], x), max(hi[0], x)
            lo[1], hi[1] = min(lo[1], y), max(hi[1], y)
        print(f"\r  {end - time.monotonic():4.0f}s left   "
              f"x {lo[0]:+.2f}..{hi[0]:+.2f}   y {lo[1]:+.2f}..{hi[1]:+.2f}   "
              f"{samples} samples ", end="", flush=True)
        time.sleep(0.02)
    print("\n")
    if not samples:
        print("  Nothing registered. Press harder — it is resistive, so a")
        print("  fingernail or a stylus rather than a fingertip.")
        return 1
    for axis, name in ((0, "touch X"), (1, "touch Y")):
        span = hi[axis] - lo[axis]
        verdict = ("dead — it never moved" if span < 0.15 else
                   "cramped — moves, but over a fraction of its range"
                   if span < 0.8 else "healthy")
        print(f"  {name}: {lo[axis]:+.3f} .. {hi[axis]:+.3f}"
              f"   span {span:.2f}   {verdict}")
    return 0


def main() -> int:
    pad = open_panel()
    if pad is None:
        return 1
    try:
        if "--check" in sys.argv:
            return check(pad)
        if "--live" in sys.argv:
            return live(pad)

        print("\n  Three corners, held for three seconds each.")
        print("  It is resistive: press with a fingernail, firmly.")
        print("  'Top-left' means as you look at her face on the panel.")

        seen = {}
        for label in CORNERS:
            spot = measure(pad, label)
            if spot is None:
                return 1
            seen[label] = spot

        answer = solve(seen)
        if answer is None:
            return 1
        save(answer)
        return 0
    finally:
        pad.stop()


if __name__ == "__main__":
    raise SystemExit(main())
