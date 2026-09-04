"""Listen to candidate voices saying the same line, and pick one.

Each voice announces its own name before the line, so what you hear and what
you have to type are never separated by a neutral narrator you then have to
mentally discount.

    uv run python scripts/audition.py                    # the shortlist
    uv run python scripts/audition.py --all-english      # all 24
    uv run python scripts/audition.py bf_emma af_kore    # just these
    uv run python scripts/audition.py --say "Your text here."

Synthesis runs at roughly 0.6x real time on this Pi, so a nine-voice audition
takes a bit over a minute to prepare. It is all made up front and then played
back to back: comparing voices is much easier without pauses in between.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eve import tts

# The persona's own calibration line: flat delivery, a real answer, and a
# closing turn of the knife. If a voice cannot sell this one, it cannot sell
# the assistant.
LINE = ("Overcast, seventy-five degrees, and a seventy percent chance of "
        "rain. Ideal conditions, assuming you enjoy disappointment.")

# Cool and clinical, which is what the persona needs. The British voices are
# here because received pronunciation does condescension without effort; the
# American ones because GLaDOS herself is American and flat.
SHORTLIST = [
    "bf_emma", "bf_alice", "bf_isabella", "bf_lily",
    "af_kore", "af_nova", "af_sarah", "af_river", "af_alloy",
]

ENGLISH = ("af_", "am_", "bf_", "bm_")


def play(pcm: bytes) -> None:
    subprocess.run(
        ["aplay", "-q", "-f", "S16_LE", "-r", str(tts.RATE), "-c", "1", "-"],
        input=pcm, stderr=subprocess.DEVNULL, check=False,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("voices", nargs="*", help="voice ids; default is the shortlist")
    parser.add_argument("--all-english", action="store_true")
    parser.add_argument("--say", default=LINE, help="the line to audition")
    parser.add_argument("--save", type=Path, help="also write each take to this directory")
    args = parser.parse_args()

    available = set(tts.voices())
    if not available:
        print("No Kokoro model — run scripts/fetch-models.sh", file=sys.stderr)
        return 1

    if args.voices:
        chosen = args.voices
    elif args.all_english:
        chosen = sorted(v for v in available if v.startswith(ENGLISH))
    else:
        chosen = SHORTLIST

    unknown = [v for v in chosen if v not in available]
    if unknown:
        print(f"unknown voices: {', '.join(unknown)}", file=sys.stderr)
        return 1

    # Synthesised up front, so playback is back to back and comparable.
    takes = []
    for voice in chosen:
        name = voice.split("_", 1)[1].capitalize()
        print(f"  making {voice} …", flush=True)
        takes.append((voice, tts.synth(f"{name}. {args.say}", voice=voice)))

    if args.save:
        args.save.mkdir(parents=True, exist_ok=True)
        for voice, pcm in takes:
            (args.save / f"{voice}.raw").write_bytes(pcm)
        print(f"saved to {args.save}")

    print()
    for voice, pcm in takes:
        print(f"▶ {voice}", flush=True)
        play(pcm)

    print("\nTo keep one:  VOICE_TTS_VOICE=<id>  (or set it in the service unit)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
