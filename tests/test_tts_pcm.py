"""The PCM path, pinned byte-for-byte against the implementation it replaced.

Three per-sample Python loops used to sit between Kokoro finishing a sentence
and that sentence reaching aplay: float32 -> s16, a nearest-sample resample to
44.1kHz, and the gain pass. For one 3.5s sentence that is roughly half a
million interpreter iterations, and every one of them lands *before* the
player is opened — so on a Pi it is a fifth of a second of pure latency in
front of the first word, on a device whose worst quality is how long it takes
to start talking.

Rewriting them in numpy is a hundredfold, but only worth doing if the output
does not move, because "she sounds slightly different now" is not a trade
anyone agreed to. The reference implementations below are the originals,
kept verbatim. Every test here asserts the new code is bit-identical to them.

Two details make that harder than it looks, and both were got wrong first:

  * `int(s * 32767)` where s is a np.float32 does the multiply in float32.
    Doing it in float64 — which is what the obvious `samples * 32767` gives
    you — changes 13 samples in 200,000.
  * the resample index is `int(i * rate / RATE)`, a float64 division then a
    truncation. Integer `//` is not the same function.
"""
from __future__ import annotations

import array

import numpy as np
import pytest

from eve import tts


# --- the implementations this replaced, kept verbatim ---------------------

def reference_f32_to_s16(samples) -> bytes:
    return array.array(
        "h", (max(-32768, min(32767, int(s * 32767))) for s in samples)
    ).tobytes()


def reference_normalize(frames: bytes, rate: int, channels: int) -> bytes:
    if channels == 2:
        mono = array.array("h", frames)
        frames = mono[0::2].tobytes()
    if rate != tts.RATE:
        src = array.array("h", frames)
        n_out = int(len(src) * tts.RATE / rate)
        frames = array.array(
            "h", (src[min(len(src) - 1, int(i * rate / tts.RATE))]
                  for i in range(n_out))).tobytes()
    samples = array.array("h", frames)
    peak = max(1, max(samples, default=1), -min(samples, default=-1))
    if peak < tts.TARGET_PEAK:
        gain = tts.TARGET_PEAK / peak
        samples = array.array(
            "h", (max(-32768, min(32767, int(s * gain))) for s in samples))
    return samples.tobytes()


def kokoro_like(seconds: float, amplitude: float, seed: int = 0):
    """Float32 in [-1, 1] at Kokoro's native rate, shaped like speech."""
    rng = np.random.default_rng(seed)
    n = int(24000 * seconds)
    t = np.arange(n) / 24000.0
    voiced = (np.sin(2 * np.pi * 120 * t) + 0.5 * np.sin(2 * np.pi * 240 * t))
    envelope = np.abs(np.sin(2 * np.pi * 3.5 * t)) + 0.05
    noise = rng.standard_normal(n) * 0.02
    signal = (voiced * envelope + noise)
    signal = signal / np.abs(signal).max() * amplitude
    return signal.astype(np.float32)


AMPLITUDES = [0.02, 0.2, 0.5, 0.88, 1.0]


class TestTheFloatToPcmConversion:
    @pytest.mark.parametrize("amplitude", AMPLITUDES)
    def test_it_is_byte_identical_to_the_loop_it_replaced(self, amplitude):
        samples = kokoro_like(0.35, amplitude, seed=int(amplitude * 100))
        assert tts._to_s16(samples).tobytes() == reference_f32_to_s16(samples)

    def test_full_scale_input_does_not_wrap(self):
        # The clip is load-bearing: 1.0 * 32767 is fine but a float32 that
        # rounds just over it would wrap to -32768 without it.
        samples = np.array([1.0, -1.0, 0.999999, -0.999999], dtype=np.float32)
        assert tts._to_s16(samples).tobytes() == reference_f32_to_s16(samples)

    def test_the_multiply_happens_in_float32(self):
        # The specific bug: float64 rounds differently on samples that land
        # exactly between two s16 codes. If this ever fails, someone has
        # "simplified" the np.float32 cast away.
        samples = kokoro_like(1.0, 0.7, seed=11)
        wrong = np.clip(np.trunc(samples.astype(np.float64) * 32767),
                        -32768, 32767).astype(np.int16)
        right = tts._to_s16(samples)
        assert right.tobytes() == reference_f32_to_s16(samples)
        assert not np.array_equal(wrong, right), \
            "float64 used to differ here; if it no longer does, drop the note"


class TestNormalise:
    @pytest.mark.parametrize("amplitude", AMPLITUDES)
    def test_the_whole_path_is_byte_identical(self, amplitude):
        samples = kokoro_like(0.4, amplitude, seed=int(amplitude * 77))
        mine = tts._normalize(tts._to_s16(samples).tobytes(), 24000, 1)
        theirs = reference_normalize(reference_f32_to_s16(samples), 24000, 1)
        assert mine == theirs

    def test_a_stream_already_at_the_target_rate_is_identical(self):
        samples = kokoro_like(0.3, 0.4, seed=5)
        frames = tts._to_s16(samples).tobytes()
        assert tts._normalize(frames, tts.RATE, 1) == \
            reference_normalize(frames, tts.RATE, 1)

    def test_a_loud_stream_is_left_alone_exactly_as_before(self):
        loud = array.array("h", [30000, -30000] * 512).tobytes()
        assert tts._normalize(loud, tts.RATE, 1) == \
            reference_normalize(loud, tts.RATE, 1) == loud

    def test_the_stereo_fold_is_identical(self):
        stereo = array.array("h", [1000, -1000] * 512).tobytes()
        assert tts._normalize(stereo, tts.RATE, 2) == \
            reference_normalize(stereo, tts.RATE, 2)

    def test_silence_does_not_divide_by_zero(self):
        silence = array.array("h", [0] * 1024).tobytes()
        assert tts._normalize(silence, tts.RATE, 1) == \
            reference_normalize(silence, tts.RATE, 1)

    def test_an_empty_stream_is_survived(self):
        assert tts._normalize(b"", tts.RATE, 1) == reference_normalize(b"", tts.RATE, 1)

    def test_one_sample_is_survived(self):
        one = array.array("h", [7]).tobytes()
        assert tts._normalize(one, 24000, 1) == reference_normalize(one, 24000, 1)

    @pytest.mark.parametrize("rate", [8000, 16000, 22050, 24000, 48000])
    def test_every_resample_ratio_lands_on_the_same_samples(self, rate):
        # Upsampling and downsampling both, since the index arithmetic is the
        # half that is easiest to get subtly wrong.
        source = array.array(
            "h", [int(10000 * np.sin(i / 9)) for i in range(rate // 8)]
        ).tobytes()
        assert tts._normalize(source, rate, 1) == reference_normalize(source, rate, 1)


class TestItIsActuallyFaster:
    def test_the_rewrite_is_not_slower_than_what_it_replaced(self):
        # Not a benchmark — a guard. The entire reason for this change is
        # speed, so a future edit that reintroduces a per-sample loop should
        # fail here rather than quietly costing a fifth of a second per
        # sentence on the Pi. The margin is deliberately loose.
        import time

        samples = kokoro_like(1.0, 0.4, seed=3)
        frames = reference_f32_to_s16(samples)

        start = time.perf_counter()
        reference_normalize(frames, 24000, 1)
        slow = time.perf_counter() - start

        start = time.perf_counter()
        tts._normalize(tts._to_s16(samples).tobytes(), 24000, 1)
        fast = time.perf_counter() - start

        assert fast < slow / 5, f"expected a large win, got {slow / max(fast, 1e-9):.1f}x"
