"""Text to speech with Kokoro, on the CPU.

Kokoro-82M runs about half of real time on a Pi 5's four cores, which is slow
enough that the caller synthesises one sentence at a time and starts playing
the first while the rest is still being made.

Two things happen to the text on the way in, and both matter more than they
look. `speakable()` rewrites digits and clock times as words, because neural
voices are trained on prose and stumble over "5:48"; and `_normalize()` lifts
the result close to full scale, because the speaker this ends up on is quiet
and a polite -12dBFS is inaudible across a room.
"""
from __future__ import annotations

import re
import threading
from pathlib import Path

import numpy as np

from eve import config

# Where the two Kokoro files live. Not in the repo: together they are 354MB,
# which is not something to keep in git. scripts/fetch-models.sh puts them
# here, and KOKORO_DIR overrides for anyone who keeps models elsewhere.
#
# config owns path expansion and the relationship between EVE_MODELS_DIR and
# this subdirectory.  Keep this alias because it is part of the useful runtime
# diagnostics and several callers inspect it directly.
KOKORO_DIR = config.kokoro_dir()

def _default_voice() -> str:
    """The voice for the current presentation, unless one is named outright."""
    return config.setting("VOICE_TTS_VOICE") or config.voice()


DEFAULT_VOICE = _default_voice()

# The bar's speaker is quiet and this one is across the room; near full-scale
# for s16 is the point, not a mastering choice.
TARGET_PEAK = 29000

RATE = 44100          # what _normalize() always emits

# Which weights to open. A setting rather than a constant because the release
# fetch-models.sh already points at also publishes a half-precision build,
# measured 13% faster on aarch64 at 4 threads and 45% smaller to load — and
# synthesis is the longest wait in a turn, so 13% is worth having. It is not
# the default because fp16 is not bit-identical: the duration predictor
# shifts and a sentence comes out about 3% longer. That is a question for
# somebody's ears on the real speakers, not for a benchmark.
MODEL_FILE = config.setting("KOKORO_MODEL", default="kokoro-v1.0.onnx")

# How fast she talks. Compute scales with the audio produced, so raising this
# shortens the wait *and* the reply — but it changes how she sounds, so it
# ships at the value she has always had and moves only if someone decides
# they like 1.1 better. Kokoro accepts 0.5 to 2.0.
SPEED = float(config.setting("VOICE_TTS_SPEED", default="1.0"))

# Kokoro gets every core, deliberately.
#
# The obvious-looking change is to copy whisper's `-t 3` here — it leaves a
# core for the display thread, and vad.py pins Silero to one thread for the
# same reason. Measured on this graph, aarch64, onnxruntime 1.28:
#
#     1 thread 2.965s   2 threads 1.591s   3 threads 1.133s   4 threads 0.891s
#
# Three threads is 27% slower than four. Synthesis is the single longest wait
# between a person finishing a question and hearing an answer, and the face is
# decoration; paying two seconds of the first to smooth the second is exactly
# backwards. kokoro-onnx builds its session with no SessionOptions at all, so
# onnxruntime's default of 0 — every physical core — is already right; this
# only makes it deliberate, so nobody "fixes" the inconsistency later.
INTRA_OP_THREADS = int(config.setting("VOICE_TTS_THREADS", default="0")) or 0

_VOICE_RE = re.compile(r"[a-z]{2}_[a-z_]+")
_KOKORO = None        # loaded once; opening the model takes a few seconds
_LOADING = threading.Lock()   # warm_up() and the first synth() can race


# --- saying numbers out loud ---------------------------------------------

_ONES = ("zero one two three four five six seven eight nine ten eleven "
         "twelve thirteen fourteen fifteen sixteen seventeen eighteen "
         "nineteen").split()
_TENS = {2: "twenty", 3: "thirty", 4: "forty", 5: "fifty",
         6: "sixty", 7: "seventy", 8: "eighty", 9: "ninety"}


def _num_words(n: int) -> str:
    if n < 0:
        return "minus " + _num_words(-n)
    if n < 20:
        return _ONES[n]
    if n < 100:
        t, r = divmod(n, 10)
        return _TENS[t] + ("" if not r else " " + _ONES[r])
    if n < 1000:
        h, r = divmod(n, 100)
        return _ONES[h] + " hundred" + ("" if not r else " " + _num_words(r))
    return str(n)  # past our needs; let the engine cope


def _decimal_words(text: str) -> str:
    """Speak a decimal instead of letting its point become a full stop.

    ``-?\\d+`` matches "56" and "0" separately in "56.0", leaving the point
    behind: the engine hears "fifty six." and starts a new sentence at "zero
    percent". A trailing ".0" is dropped because it is noise from a float,
    not a measurement anyone says aloud.
    """
    whole, dot, frac = text.partition(".")
    words = _num_words(int(whole))
    frac = frac.rstrip("0")
    if not dot or not frac:
        return words
    return words + " point " + " ".join(_ONES[int(digit)] for digit in frac)


def _time_words(h: int, m: int) -> str:
    if m == 0:
        return _num_words(h) + " o'clock"
    if m < 10:
        return f"{_num_words(h)} oh {_num_words(m)}"
    return f"{_num_words(h)} {_num_words(m)}"


# A comma inside a number is a group separator, not a pause. Left in, the
# number regex below matches "1" and "000" separately and she says "one, zero
# dollars" — which she did, out loud, for every price the web search returned.
_GROUPED = re.compile(r"(?<=\d),(?=\d{3}(?:\D|$))")

# Two things this has to tell apart, and the second one used to be a
# regression rather than a gap.
#
# `version` is a dotted triple. Fed to the number branch, "1.0.2" matched
# "1.0" and then ".2", dropped the middle component and left a full stop
# mid-word: "version one. two". espeak reads a dotted version correctly on
# its own, so the right move is to not touch it.
#
# `number` no longer swallows a hyphen that follows an alphanumeric. It used
# to: "-?\d+" matched the "-1" of "3-1" and produced "threeminus one", and
# "555-1234" became "five hundred fifty fiveminus 1234". A hyphen between two
# numbers is a range or a score, and espeak already says "dash" for it — so
# the minus sign is only a minus when nothing runs into it from the left.
_NUMBER = re.compile(
    r"(?P<version>\d+(?:\.\d+){2,})"
    r"|(?<![A-Za-z0-9])(?P<number>-?\d+(?:\.\d+)?)"
)


def _spoken_number(match: re.Match) -> str:
    if match.group("version"):
        return match.group("version")
    return _decimal_words(match.group("number"))


def speakable(text: str) -> str:
    """Rewrite text the way a narrator would read it aloud.

    '5:48' -> 'five forty eight', '74' -> 'seventy four',
    '56.0' -> 'fifty six', em-dashes -> commas.

    This is the last thing that touches a reply before it becomes her voice,
    which is why its edge cases are worth this much regex: the system prompt
    steers her to look up weather, prices, sports and schedules, and those
    are exactly the answers that come back full of "3-1", "2024-2025" and
    "$1,250".
    """
    text = re.sub(r"\b(\d{1,2}):(\d{2})\b",
                  lambda m: _time_words(int(m.group(1)), int(m.group(2))),
                  text)
    text = re.sub(r"\s*[—–]\s*", ", ", text)
    text = _GROUPED.sub("", text)
    text = _NUMBER.sub(_spoken_number, text)
    return text


# --- synthesis ------------------------------------------------------------

def paths() -> tuple[Path, Path] | None:
    """The model and voice bank, or None if either is missing.

    Returning None rather than raising lets a caller fall back or warn: a
    missing model should be a diagnosable startup message, not a traceback
    from inside the first spoken reply.
    """
    model = KOKORO_DIR / MODEL_FILE
    bank = KOKORO_DIR / "voices-v1.0.bin"
    return (model, bank) if model.is_file() and bank.is_file() else None


def _engine():
    """The Kokoro session, built once, with its thread count made deliberate.

    Locked because warm_up() runs on a background thread at startup while the
    first spoken reply may already be calling synth(): two InferenceSessions
    over a 310MB graph at once is how an 8GB Pi discovers swap.
    """
    global _KOKORO
    if _KOKORO is not None:
        return _KOKORO
    with _LOADING:
        if _KOKORO is not None:      # somebody else built it while we waited
            return _KOKORO
        from kokoro_onnx import Kokoro

        found = paths()
        if found is None:
            raise RuntimeError(
                f"Kokoro model not found in {KOKORO_DIR}. "
                "Run scripts/fetch-models.sh, or set KOKORO_DIR."
            )
        model, bank = found
        if INTRA_OP_THREADS:
            import onnxruntime

            options = onnxruntime.SessionOptions()
            options.intra_op_num_threads = INTRA_OP_THREADS
            options.log_severity_level = 4
            _KOKORO = Kokoro.from_session(
                onnxruntime.InferenceSession(
                    str(model), options, providers=["CPUExecutionProvider"]),
                str(bank))
        else:
            # onnxruntime's own default is every physical core, which is what
            # this wants. See INTRA_OP_THREADS.
            _KOKORO = Kokoro(str(model), str(bank))
        return _KOKORO


def warm_up() -> None:
    """Open the model before anyone is waiting on it.

    A 310MB graph, a voice bank and espeak's first phonemize land inside the
    first tts.synth() otherwise — several seconds charged to the first person
    to ask a question after a restart, which is the worst possible moment to
    spend them. Worse, speech.speak() times that call as synthesis throughput,
    so the model load poisons the ratio for the rest of that reply.

    Raises what it raises; the caller decides whether a voiceless assistant is
    worth a line or a crash. It is a line — see main().
    """
    _engine()


def _to_s16(samples: np.ndarray) -> np.ndarray:
    """Kokoro's float32 in [-1, 1] -> the s16 the player wants.

    The float32 cast on the multiplier is not decoration. `int(s * 32767)`
    with s a np.float32 does the arithmetic in float32; promoting to float64
    rounds differently and moves thirteen samples in two hundred thousand.
    Bit-identical is the whole bargain here — see tests/test_tts_pcm.py.
    """
    return np.clip(np.trunc(samples * np.float32(32767)),
                   -32768, 32767).astype(np.int16)


def _normalize(frames: bytes, rate: int, channels: int) -> bytes:
    """Any engine's PCM -> mono 44100 s16 at a usable level.

    This used to be three per-sample Python loops — a resample, a peak scan
    and a gain pass — and all three run *before* the player is opened, so
    every millisecond is silence in front of the first word. On a Pi that was
    roughly 200ms per sentence spent doing what numpy does in one.
    """
    pcm = np.frombuffer(frames, dtype=np.int16)
    if channels == 2:  # take the left channel
        pcm = pcm[0::2]
    if rate != RATE and pcm.size:  # nearest-sample resample is fine for speech
        n_out = int(pcm.size * RATE / rate)
        # float64 then truncate, reproducing `int(i * rate / RATE)` exactly.
        # Integer floor division is a different function and does not match.
        index = np.minimum(
            (np.arange(n_out, dtype=np.float64) * rate / RATE).astype(np.int64),
            pcm.size - 1)
        pcm = pcm[index]
    if not pcm.size:
        return b""
    peak = max(1, int(pcm.max()), -int(pcm.min()))
    if peak < TARGET_PEAK:
        # float64, because the loop this replaces multiplied a Python int by
        # a Python float.
        pcm = np.clip(np.trunc(pcm.astype(np.float64) * (TARGET_PEAK / peak)),
                      -32768, 32767).astype(np.int16)
    return pcm.tobytes()


def synth(text: str, voice: str | None = None) -> bytes:
    """Render `text` to raw PCM: mono, 44100Hz, signed 16-bit little-endian.

    Raw rather than a WAV file because the caller streams it straight into an
    already-open `aplay`, and a header mid-stream would be heard as a click.
    """
    engine = _engine()
    requested = voice or DEFAULT_VOICE
    if not _VOICE_RE.fullmatch(requested):
        requested = DEFAULT_VOICE
    samples, rate = engine.create(speakable(text), voice=requested, speed=SPEED)
    # kokoro-onnx returns float32 in [-1, 1]; the player wants s16.
    return at_volume(_normalize(_to_s16(samples).tobytes(), int(rate), 1))


def at_volume(frames: bytes) -> bytes:
    """Turn her down to config.VOLUME.

    A separate pass rather than folding the factor into _normalize's target,
    for two reasons. _normalize is pinned byte-for-byte against the loops it
    replaced (tests/test_tts_pcm.py) and that contract is worth keeping
    literal; and normalising *to* a lower peak would not touch audio that was
    already loud enough, so a reply Kokoro happened to render hot would come
    out at full volume anyway. Scaling afterwards applies to everything.

    A no-op at 1.0, and it returns the same object rather than a copy, so an
    install that never sets VOICE_VOLUME pays nothing at all for this.
    """
    if config.VOLUME >= 1.0:
        return frames
    pcm = np.frombuffer(frames, dtype=np.int16)
    if not pcm.size:
        return frames
    # float64 then truncate, matching the rounding of the gain pass above it.
    return np.clip(np.trunc(pcm.astype(np.float64) * config.VOLUME),
                   -32768, 32767).astype(np.int16).tobytes()


def voices() -> list[str]:
    """Every voice in the bank, for auditioning."""
    if paths() is None:
        return []
    return sorted(_engine().get_voices())
