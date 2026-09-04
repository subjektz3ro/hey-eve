"""Turning a sentence and its audio duration into mouth shapes.

There is no phoneme alignment here and there does not need to be. Kokoro
hands back a sentence of speech whose length is known exactly, and the text
that produced it is right there — so the syllables can be laid across that
duration in proportion to their weight. The result is not frame-accurate
lip sync, but it is the difference between a mouth that knows *which sound*
is being made and one that only knows how loud it is.

IC's mouth is a single dash, so a viseme can only be its width and height.
That turns out to be enough: "ah" is wide and tall, "ee" is wide and flat,
"oo" is narrow and tall, and a closed mouth is a thin line.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# Width and height of the dash, as a fraction of the screen half-extent.
SHAPES = {
    "A": (0.230, 0.165),   # father, cat
    "E": (0.270, 0.062),   # bed, see
    "I": (0.175, 0.055),   # bit, city
    "O": (0.120, 0.185),   # go, thought
    "U": (0.100, 0.145),   # boot, put
    "M": (0.130, 0.028),   # closed: p, b, m
}

_VOWELS = re.compile(r"[aeiouy]+")
_PLOSIVE = re.compile(r"^[pbmtdkg]")

# A face moves because of meaning, not loudness. These are the word classes
# worth reacting to, in the register this assistant actually writes in — the
# dry marker fires most often, which is the point of it.
CUES = (
    ("dry", re.compile(
        r"^(ideal|naturally|obviously|apparently|delightful|charming|wonderful"
        r"|clearly|presumably|evidently|course)$")),
    ("hedge", re.compile(r"^(assuming|supposedly|allegedly|arguably|somehow|reportedly)$")),
    ("negation", re.compile(r"^(not|no|never|nothing|none|cannot|can't|won't|don't)$")),
    ("intensifier", re.compile(
        r"^(very|extremely|genuinely|entirely|utterly|precisely|exactly|remarkably)$")),
    ("number", re.compile(
        r"^(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve"
        r"|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|percent)$")),
)


def _cue_of(word: str) -> str | None:
    for name, pattern in CUES:
        if pattern.match(word):
            return name
    return None


def _shape_of(group: str) -> str:
    """Which mouth shape a vowel group calls for."""
    if group.startswith(("oo", "ou", "ow", "oa", "o")):
        return "O"
    if group.startswith("u"):
        return "U"
    if group.startswith(("ee", "ea", "e")):
        return "E"
    if group.startswith(("i", "y")):
        return "I"
    return "A"


@dataclass(frozen=True)
class Mouth:
    """One syllable, in seconds from the start of the reply.

    Carries both layers: `shape` is phonetic — which sound is being made — and
    `cue`, `clause`, `stop` and `question` are semantic. The face needs both,
    because the mouth follows the sound and the brows follow the meaning.
    """

    start: float
    end: float
    shape: str
    closes: bool          # word begins with a plosive: shut the lips first
    cue: str | None = None
    stress: bool = False
    clause: bool = False  # last syllable before a comma
    stop: bool = False    # last syllable of a sentence
    question: bool = False


def syllables(text: str) -> list[dict]:
    """One entry per syllable, carrying its shape, weight and cues.

    Weight is a rough duration share. Stressed syllables — the first of a
    long word — get more of the sentence than the ones that trail after them,
    which is what stops the mouth moving like a metronome.
    """
    out: list[dict] = []
    for token in text.split():
        word = "".join(c for c in token.lower() if c.isalnum() or c == "'")
        if not word:
            continue
        groups = _VOWELS.findall(re.sub(r"e$", "", word)) or ["a"]
        closes = bool(_PLOSIVE.match(word))
        cue = _cue_of(word)
        for index, group in enumerate(groups):
            stressed = (index == 0 and len(groups) > 1) or len(word) > 7
            out.append({
                "shape": _shape_of(group), "closes": closes and index == 0,
                "weight": 1.35 if stressed else 1.0, "stress": stressed,
                "cue": cue, "clause": False, "stop": False, "question": False,
            })
        if out:
            if token.rstrip('"\'').endswith((",", ";", ":")):
                out[-1]["clause"] = True
            if token.rstrip('"\'').endswith((".", "!", "?")):
                out[-1]["stop"] = True
                if token.rstrip('"\'').endswith("?"):
                    # A question colours the whole clause leading up to it,
                    # not just the syllable the mark happens to land on.
                    for entry in reversed(out):
                        if entry is not out[-1] and entry["stop"]:
                            break
                        entry["question"] = True
    return out


def timeline(text: str, seconds: float, offset: float = 0.0) -> list[Mouth]:
    """Lay `text`'s syllables across `seconds` of audio, starting at `offset`.

    Silence is left between syllables rather than stretching each to fill its
    slot: a mouth that never closes reads as a hinge, not a mouth.
    """
    parts = syllables(text)
    if not parts or seconds <= 0:
        return []
    total = sum(p["weight"] for p in parts)
    mouths: list[Mouth] = []
    cursor = offset
    for part in parts:
        span = seconds * part["weight"] / total
        # Nine tenths sounding, a tenth closed, so consecutive syllables are
        # visibly separate rather than one continuous opening.
        mouths.append(Mouth(cursor, cursor + span * 0.90, part["shape"],
                            part["closes"], part["cue"], part["stress"],
                            part["clause"], part["stop"], part["question"]))
        cursor += span
    return mouths


def beat(mouths: list[Mouth], when: float) -> Mouth | None:
    """The syllable sounding at `when`, cues and all."""
    for mouth in mouths:
        if mouth.start - 0.05 <= when <= mouth.end + 0.12:
            return mouth
        if when < mouth.start:
            return None
    return None


def at(mouths: list[Mouth], when: float) -> tuple[str, float]:
    """The shape sounding at `when`, and how far through it we are.

    Returns the closed shape between syllables, which is the whole reason
    this is a lookup rather than a simple index.
    """
    for mouth in mouths:
        if mouth.start <= when <= mouth.end:
            phase = (when - mouth.start) / max(mouth.end - mouth.start, 1e-6)
            # The lips shut for the first third of a plosive-initial syllable.
            if mouth.closes and phase < 0.33:
                return "M", phase
            return mouth.shape, phase
        if when < mouth.start:
            break
    return "M", 0.0
