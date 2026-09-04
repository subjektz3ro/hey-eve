#!/usr/bin/env python3
"""Render the animations in README.md, headlessly.

    uv run python scripts/make-gifs.py

Her face draws into /dev/fb0 by name, so replacing `_write` hands back the
frame she would have sent to the panel and the whole thing runs on any
machine — no Pi, no display, no microphone. Everything below is the real
renderer at the real panel size; nothing here is a mock-up of it.

Kept as a script rather than done once by hand because the point of the
pictures is that they are current. When her face changes, run it again.
"""
from __future__ import annotations

import math
import random
from pathlib import Path

import numpy as np
from PIL import Image

from eve import head, idle, viseme, void

OUT = Path(__file__).resolve().parent.parent / "docs"
FPS = 15
COLOURS = 64          # a dark panel needs few; this halves the file size


def recorder(face):
    """Swap the framebuffer write for a frame collector."""
    frames: list[Image.Image] = []

    def keep(frame):
        rgb = np.clip(frame, 0, 255).astype(np.uint8)
        frames.append(Image.fromarray(rgb, "RGB").convert(
            "P", palette=Image.ADAPTIVE, colors=COLOURS))

    face._write = keep
    return frames


def save(name: str, frames: list[Image.Image]) -> None:
    path = OUT / f"{name}.gif"
    frames[0].save(path, save_all=True, append_images=frames[1:],
                   duration=int(1000 / FPS), loop=0, optimize=True)
    size = path.stat().st_size / 1024
    print(f"  {path.name:<12} {len(frames):>4} frames  {size:>6.0f} KB")


def fresh(seed: int, steady: bool = True):
    """A face, with the tube's glitch optionally held still.

    The scanline tear is the point of the look and the enemy of a small GIF:
    every torn frame differs from its neighbour everywhere at once, so the
    encoder cannot skip anything. It stays on for the turn, which is the one
    people will look at closest, and off for the longer clips.
    """
    random.seed(seed)
    face = void.Face()
    if steady:
        face._disturb = lambda frame, now, unstable: frame
    return face


def a_turn() -> None:
    """One exchange, end to end, as the panel shows it."""
    face = fresh(11, steady=False)
    frames = recorder(face)
    reply = ("Overcast and seventy five degrees. Ideal, assuming you enjoy "
             "disappointment.")
    mouths = viseme.timeline(reply, 5.0)

    def run(seconds, state, level=lambda t: 0.0, note="", audio=None):
        for step in range(int(seconds * FPS)):
            t = run.at + step / FPS
            face.set_state(state, level(step / FPS), note)
            if audio is not None:
                face.set_audio_position(step / FPS, level(step / FPS))
            face._render(t, 1.0 / FPS)
        run.at += seconds
    run.at = 0.0

    run(2.0, "idle", note="READY · SAY EVE")
    # Somebody speaks: the level is what she is hearing.
    run(3.0, "listening", lambda t: 0.35 + 0.3 * abs(math.sin(t * 4.5)))
    run(1.5, "transcribing")
    run(2.0, "thinking")
    face.set_speech(mouths)
    run(2.5, "synthesizing", lambda t: min(1.0, t / 2.5), "SPEAKING SOON")
    run(5.0, "speaking", lambda t: 0.4 + 0.35 * abs(math.sin(t * 7.0)),
        audio=True)
    run(2.0, "idle", note="READY · SAY EVE")
    save("turn", frames)


def asleep() -> None:
    """Muted long enough to drop off, then woken."""
    face = fresh(5)
    frames = recorder(face)
    face.set_state("asleep")
    for step in range(int(11.0 * FPS)):
        t = step / FPS
        if step == 1:
            face._idle.play(BY_NAME["stir"], t)
        elif step == int(3.5 * FPS):
            face._idle.play(BY_NAME["dream"], t)
        elif step == int(6.5 * FPS):
            face._idle.play(BY_NAME["jerk"], t)
        elif step == int(9.0 * FPS):
            face.set_state("idle", 0.0, "READY · SAY EVE")
        face._render(t, 1.0 / FPS)
    save("asleep", frames)


def the_bug() -> None:
    """The one idle that implies something else is in the room."""
    face = fresh(7)
    frames = recorder(face)
    face.set_state("idle", 0.0, "READY · SAY EVE")
    bug = next(b for b in idle.BEHAVIOURS if b.name == "bug")
    for step in range(int(bug.seconds * FPS)):
        t = step / FPS
        if step == 1:
            face._idle.play(bug, t)
            face._idle.bug = idle.Bug(bug.seconds)
        face._render(t, 1.0 / FPS)
    save("bug", frames)


def touched() -> None:
    """A finger on the glass, which is a bug you control."""
    face = fresh(3)
    frames = recorder(face)
    face.set_state("idle", 0.0, "READY · SAY EVE")
    finger = [(0.0, 0.0, False)]
    face._touch.latest = lambda: finger[0]
    for step in range(int(8.0 * FPS)):
        t = step / FPS
        if t < 1.0:
            finger[0] = (0.0, 0.0, False)
        else:
            # Dragged slowly across her, left to right.
            across = min(1.0, (t - 1.0) / 5.0)
            finger[0] = (-0.75 + 1.5 * across,
                         0.25 * math.sin(across * math.pi * 2), True)
        if t > 6.5:
            finger[0] = (finger[0][0], finger[0][1], False)
        face._render(t, 1.0 / FPS)
    save("touch", frames)


BY_NAME = {b.name: b for b in idle.SLEEP_BEHAVIOURS}


def main() -> int:
    OUT.mkdir(exist_ok=True)
    print(f"rendering into {OUT}/ at {FPS}fps, {head.FB_WIDTH}x{head.FB_HEIGHT}")
    for job in (a_turn, asleep, the_bug, touched):
        job()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
