"""A sound that means "heard you", played before she can possibly answer.

The longest wait in a turn is the one before the first word, and most of what
makes it unpleasant is not its length — it is that silence and failure are
indistinguishable from the other side of the room. Twelve seconds of nothing
looks exactly like a microphone that did not open, a wake word that missed, or
a process that died, so the honest response is to say it again, which starts
the whole wait over.

This closes that gap without making anything faster. The instant the recorder
decides someone has stopped talking, a short tone goes out: capture worked,
the request is in flight, stop wondering. Everything after it is the same
speed it was.

Deliberately not synthesised. Kokoro is the thing being covered for; routing
the acknowledgement through it would put the acknowledgement behind the wait
it exists to explain. The tone is arithmetic — a few hundred samples of decaying
sine, built once — so the only cost between the microphone closing and the
sound leaving is spawning a player.

Three constraints this module keeps, all of them load-bearing:

  * It never raises. It runs one line before transcription on the ordinary
    path, and an assistant that dies because a courtesy noise failed is worse
    than one that is silent.
  * It never blocks. The caller is on its way to whisper.cpp; waiting for a
    Bluetooth player to open would spend the latency this exists to hide.
  * It does not touch speech._last_spoke_at. That clock owns the amplifier-wake
    logic, which works and is not mine to adjust from here. The visible cost is
    that a tone landing on a sleeping amplifier may go unheard — which is what
    happens today anyway, since today there is no tone at all.
"""
from __future__ import annotations

import os
import subprocess
import threading

from eve import config

# A single soft tone rather than a chime or a rising pair. It fires on every
# captured utterance, many times an hour, in a room someone lives in — so the
# thing it must not become is a notification sound. Short, low, and over.
FREQUENCY_HZ = float(os.environ.get("VOICE_EARCON_HZ", "660"))
DURATION_S = float(os.environ.get("VOICE_EARCON_S", "0.12"))
# A quarter of full scale. tts._normalize lifts speech to near-clipping on
# purpose, because the SoundSticks are quiet across a room; an acknowledgement
# held to that standard would be louder than the answer it introduces.
AMPLITUDE = float(os.environ.get("VOICE_EARCON_LEVEL", "0.25"))

# Set VOICE_EARCON=0 to silence it without touching the calling code.
ENABLED = os.environ.get("VOICE_EARCON", "1") not in ("0", "false", "no", "")

# How long to let the player run before giving up on it. The tone is 120ms;
# anything approaching this is a wedged device, and a wedged device must not
# accumulate one stuck process per utterance for the life of the service.
_PLAY_TIMEOUT_S = 5.0

_tone: bytes | None = None
_tone_lock = threading.Lock()


def _build() -> bytes:
    """The tone itself, as s16 mono PCM at the player's rate.

    An exponential decay rather than a plain gate: a rectangular envelope ends
    on whatever phase it happens to be at, and the discontinuity is audible as
    a click — which reads as a fault rather than as an acknowledgement. The
    5ms raised-cosine attack is the same argument at the other end.
    """
    import numpy as np

    count = max(1, int(config.TTS_RATE * DURATION_S))
    t = np.arange(count, dtype=np.float32) / np.float32(config.TTS_RATE)

    # Decay to roughly a fiftieth of the peak by the end, so the tail is
    # inaudible before the buffer runs out and nothing has to be gated off.
    envelope = np.exp(-t * np.float32(4.0 / max(DURATION_S, 1e-3)))

    attack = max(1, int(config.TTS_RATE * 0.005))
    if attack < count:
        ramp = np.linspace(0.0, np.pi, attack, dtype=np.float32)
        envelope[:attack] *= (1.0 - np.cos(ramp)).astype(np.float32) / np.float32(2.0)

    wave = np.sin(np.float32(2.0 * np.pi) * np.float32(FREQUENCY_HZ) * t)
    # Scaled by the same volume her voice is, and that is the whole reason
    # VOLUME lives in config rather than in tts: turning down only the speech
    # would leave a courtesy blip louder than the answer it introduces, which
    # is a worse noise than the one being fixed. AMPLITUDE stays what it means
    # — how loud this is *relative to her* — so the two move together.
    level = max(0.0, min(1.0, AMPLITUDE)) * config.VOLUME
    samples = wave * envelope * np.float32(level)
    # Same conversion the speech path uses, for the same reason: the multiply
    # stays in float32 and the clip is what stops a full-scale sample wrapping
    # to the opposite rail. See tts._to_s16 and tests/test_tts_pcm.py.
    return (
        np.clip(np.trunc(samples * np.float32(32767)), -32768, 32767)
        .astype(np.int16)
        .tobytes()
    )


def tone() -> bytes:
    """The PCM, built once and kept.

    Built lazily rather than at import: this module is imported by main before
    the panel exists, and a few hundred samples of sine is not worth spending
    there. The lock is because the listen loop and any future caller are
    different threads, and building it twice would be harmless but silly.
    """
    global _tone
    if _tone is None:
        with _tone_lock:
            if _tone is None:
                _tone = _build()
    return _tone


def _play(pcm: bytes) -> None:
    """Push the tone at the default device and reap the player.

    subprocess.run rather than a bare Popen: a fire-and-forget Popen leaves a
    zombie until something reaps it, and this fires on every single utterance.
    Running it inside the throwaway thread makes the wait free and the reaping
    automatic.
    """
    try:
        subprocess.run(
            ["aplay", "-q", "-f", "S16_LE",
             "-r", str(config.TTS_RATE), "-c", "1", "-"],
            input=pcm,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,   # bluealsa's plugin is chatty on open
            timeout=_PLAY_TIMEOUT_S,
        )
    except Exception:
        # Silence is the correct failure. There is nothing a person can do
        # about a courtesy tone that did not play, the real failure it might
        # indicate — no speaker — is already reported once at startup by
        # speech.speaker_available(), and logging here would put a line in the
        # journal for every utterance of a broken afternoon.
        pass


def acknowledge() -> None:
    """Say "heard you" now, and return immediately.

    Returns before the sound has played, on purpose: the caller's next move is
    a whisper.cpp decode measured in seconds, and the whole point is that the
    tone overlaps it rather than delaying it.
    """
    if not ENABLED:
        return
    try:
        pcm = tone()
    except Exception:
        return          # numpy missing, or a nonsense setting; never fatal
    threading.Thread(target=_play, args=(pcm,), daemon=True).start()
