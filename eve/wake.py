"""Deciding whether something said in the room was addressed to the assistant.

There is no separate wake-word engine here. Whisper is already running, so the
wake word is matched against its transcript instead — which costs no new
dependency, needs no model training, and lets the whole request arrive in one
breath ("hey blackwall, what time is it") rather than forcing a beep in
between.

The cost of that choice: speech recognition runs on every utterance in the
room, not just the ones meant for us. Fine on a desk; the thing to replace if
this ever wants to be low-power always-on.
"""
from __future__ import annotations

import re

# The name is chosen as much for the recogniser as for the lore, because two
# earlier ones failed in ways worth recording.
#
# "Zero" was unusable. Whisper writes numbers as numerals, so it came back as
# the single character "0" — no list of word spellings could ever have matched
# that — and /z/ is a low-energy fricative the detector trips on late,
# clipping "hey" down to a bare "A". The real transcript was "A0.".
#
# Exotic in-universe names failed differently: the recogniser does not know
# them, so it substitutes something plausible. Delamain became "the",
# Sandevistan became "send", Pallas became "palace", Nyx became "nix".
#
# "Eve" transcribed clean in every test, with and without the greeting, in
# both voices. Its one real weakness is that it opens on a vowel, which is the
# quietest possible onset for a speech detector — so the greeting matters more
# here than it did, and "heave" is matched outright because that is what "hey
# Eve" becomes when the two run together.
_NAME = r"(?:eve|eave|eves|evie|ava|eva)"
_MERGED = r"(?:heave|heaves|hayve)"

_GREETING = r"(?:hey|hi|hello|ok|okay|yo|hay|a|and)"

# The greeting stays optional. It is the part most often lost — the quietest
# moment of the phrase, landing before the detector has decided anyone is
# speaking — and anchoring to the start of the utterance does the same job:
# "the eve of the launch" never begins with it.
_ADDRESS = re.compile(
    rf"""^\W*
    (?:
        {_MERGED}                     # "hey eve" run together
      | {_GREETING}[\s,.]*{_NAME}     # greeting, often lost or run together
      | {_NAME}                       # bare: must open the utterance
    )
    \b[\s,.!?-]*
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Whisper emits these for silence, breaths, and background noise. They are not
# speech and must never be treated as a question.
_NOISE = re.compile(
    r"^\W*(\[.*\]|\(.*\)|you|thanks for watching|thank you|bye|okay|so|uh|um|\.|,)?\W*$",
    re.IGNORECASE,
)


def is_noise(text: str) -> bool:
    """True when a transcript carries no actual request.

    Anything addressed to the assistant is never noise, however mangled the
    rest is — dropping a wake attempt is the failure a person actually
    notices.
    """
    if _ADDRESS.match(text):
        return False
    return bool(_NOISE.match(text.strip()))


# Ways of saying "we are done here". Matched against the *whole* remaining
# utterance rather than searched for, so "no, tell me about Estonia" is a
# request and "no thanks" is a goodbye — the difference is everything after
# the first word.
# Every entry is separated by a pipe. The first line used to separate its
# terms with spaces, which meant the whole line became the single entry
# "no nope nah none negative" — so a bare "no", the commonest refusal there
# is, never matched, and answering "no" to "want the longer version?" spent a
# turn and reopened the window for another minute.
_DISMISSALS = frozenset("""
    nothing | nothing else | nothing more | no more | not right now | not now
    thanks | thank you | thanks anyway | no need | never mind | nevermind
    thats all | thats it | thats everything | thats fine | thats okay
    thats great | thats perfect | thatll be all | that will be all
    forget it | maybe later | some other time
    im good | were good | all good | all set | im all set | were all set
    were done | im done | done | got it | understood
    goodbye | good bye | bye | goodnight | good night | see you
    stop | cancel | dismissed | that is all
""".replace("\n", "|").split("|")) - {""}
_DISMISSALS = frozenset(" ".join(p.split()) for p in _DISMISSALS if p.strip())

# Said on their own these end it; said in front of anything else they are
# just the start of a correction, so the remainder still has to be a closer.
_REFUSALS = ("no", "nope", "nah", "none", "negative", "not really", "nevermind")

_TIDY = re.compile(r"[^a-z ]")


def is_dismissal(text: str) -> bool:
    """True when the whole utterance is a sign-off and nothing more.

    Matched in full rather than searched, because the first word is not the
    signal: "no" ends a conversation and "no, tell me about Estonia" starts
    one. Anything with a request attached fails this and is answered normally.
    """
    cleaned = _TIDY.sub("", text.lower().replace("'", ""))
    cleaned = " ".join(cleaned.split())
    if not cleaned or len(cleaned.split()) > 5:
        return False
    for lead in ("okay ", "ok ", "alright ", "well ", "uh ", "um ", "yeah "):
        if cleaned.startswith(lead):
            cleaned = cleaned[len(lead):].strip()
    # A closer with a courtesy stuck on the end is still a closer.
    for tail in (" thank you very much", " thanks a lot", " thank you",
                 " thanks", " please"):
        if cleaned.endswith(tail) and cleaned != tail.strip():
            cleaned = cleaned[: -len(tail)].strip()
            break
    if cleaned in _DISMISSALS:
        return True
    # "no", and "no" in front of any other way of closing: refusals combine
    # with closers freely, and enumerating the products of two lists is how
    # the set ends up missing "no im good" while holding "im good".
    for word in _REFUSALS:
        if cleaned == word:
            return True
        if cleaned.startswith(word + " "):
            rest = cleaned[len(word) + 1:].strip()
            return rest in _DISMISSALS or rest in _REFUSALS
    return False


def address(text: str) -> str | None:
    """What was asked, if the assistant was addressed.

    Returns the request with the wake phrase stripped, an empty string when
    the wake phrase was said on its own (so the caller can prompt for more),
    or None when this speech was not for us.
    """
    match = _ADDRESS.match(text)
    if match is None:
        return None
    return text[match.end():].strip()
