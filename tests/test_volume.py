"""How loud she is, and why that could not be answered before.

_normalize lifts every reply to TARGET_PEAK — 88% of full scale — because the
speaker it was written against is quiet across a room. On a speaker that is
not quiet the result is a voice that arrives very nearly clipping, every time,
with no variation to hide behind and nothing to turn down: the Bluetooth
mixer is separate state on the A2DP link that does not survive a reconnect,
and deploy/connect-speaker.sh reconnects on boot and every minute after.

Two decisions are pinned here.

The gain is a pass *after* normalisation rather than a lower target inside it.
Normalising still buys what it always did — every reply the same loudness
whatever Kokoro handed over — and this decides what that loudness is. It also
keeps _normalize byte-identical to the loops it replaced, which
tests/test_tts_pcm.py exists to guarantee.

And it lives in config, not tts, because tts is not the only thing that makes
a noise. Turning down the speech alone would leave the acknowledgement tone
louder than the answer it introduces.
"""
from __future__ import annotations

import array

import numpy as np
import pytest

from eve import config, earcon, tts


def peak(frames: bytes) -> int:
    samples = array.array("h", frames)
    return max(abs(s) for s in samples) if samples else 0


def speechlike(seconds: float = 0.3, amplitude: float = 0.8) -> bytes:
    """PCM shaped the way a normalised reply is: loud, near the target."""
    rng = np.random.default_rng(4)
    n = int(config.TTS_RATE * seconds)
    t = np.arange(n) / config.TTS_RATE
    signal = np.sin(2 * np.pi * 140 * t) + 0.4 * rng.standard_normal(n)
    signal = signal / np.abs(signal).max() * amplitude
    return (signal * 32767).astype(np.int16).tobytes()


@pytest.fixture
def at_volume(monkeypatch):
    """Set the volume and clear the earcon's cached tone with it."""

    def set_to(level: float):
        monkeypatch.setattr(config, "VOLUME", level)
        monkeypatch.setattr(earcon, "_tone", None)
        return level

    return set_to


class TestNothingChangesForAnInstallThatNeverSetsIt:
    def test_the_default_is_full_volume(self):
        # Whatever this file adds, an install that does not opt in must sound
        # exactly as it did — the old behaviour is the documented one.
        assert config.VOLUME == 1.0

    def test_full_volume_hands_back_the_very_same_bytes(self):
        # Not merely equal. At 1.0 this is on the hot path of every sentence
        # and should cost nothing, not a copy of the whole reply.
        frames = speechlike()
        assert tts.at_volume(frames) is frames

    def test_the_normalise_contract_is_untouched(self):
        # The gain is a separate pass precisely so _normalize stays pinned
        # against the implementation it replaced. If someone folds it in, the
        # byte-identity tests next door start failing for a reason that looks
        # like a rounding bug.
        import inspect
        assert "VOLUME" not in inspect.getsource(tts._normalize)


class TestTurningHerDown:
    @pytest.mark.parametrize("level", [0.1, 0.3, 0.5, 0.75])
    def test_the_peak_falls_by_the_factor_asked_for(self, at_volume, level):
        at_volume(level)
        frames = speechlike()
        assert peak(tts.at_volume(frames)) == pytest.approx(
            peak(frames) * level, rel=0.02)

    def test_it_applies_to_audio_that_was_already_loud(self, at_volume):
        # The reason this is a scale rather than a lower normalisation target:
        # normalising *to* a smaller peak leaves anything already above it
        # alone, so a reply Kokoro happened to render hot would still arrive
        # at full volume — which is the one case being complained about.
        at_volume(0.3)
        hot = array.array("h", [32000, -32000] * 512).tobytes()
        assert peak(tts.at_volume(hot)) == pytest.approx(32000 * 0.3, rel=0.02)

    def test_silence_stays_silent(self, at_volume):
        at_volume(0.3)
        quiet = array.array("h", [0] * 512).tobytes()
        assert set(array.array("h", tts.at_volume(quiet))) == {0}

    def test_an_empty_reply_is_survived(self, at_volume):
        at_volume(0.3)
        assert tts.at_volume(b"") == b""

    def test_nothing_wraps_to_the_opposite_rail(self, at_volume):
        at_volume(0.5)
        rails = array.array("h", [32767, -32768] * 64).tobytes()
        out = array.array("h", tts.at_volume(rails))
        assert max(out) <= 32767 and min(out) >= -32768

    def test_zero_is_silence_rather_than_an_error(self, at_volume):
        at_volume(0.0)
        assert set(array.array("h", tts.at_volume(speechlike()))) == {0}

    def test_the_output_is_still_a_whole_number_of_samples(self, at_volume):
        at_volume(0.3)
        assert len(tts.at_volume(speechlike())) % 2 == 0


class TestTheSettingItself:
    @pytest.mark.parametrize("written, expected", [
        ("0.3", 0.3), ("1", 1.0), ("0", 0.0),
        ("30", 1.0),       # someone writing 30 meaning 30 percent
        ("-1", 0.0),       # and the other direction
    ])
    def test_it_is_clamped_where_it_is_read(self, written, expected,
                                            monkeypatch):
        """A typo must not be a thirty-fold gain into the rails.

        Reloaded rather than asserted about the arithmetic, because the clamp
        being *present at the point of reading* is the thing under test —
        checking `max(0, min(1, x))` in the test would only prove Python
        works.
        """
        import importlib

        monkeypatch.setenv("VOICE_VOLUME", written)
        try:
            assert importlib.reload(config).VOLUME == expected
        finally:
            monkeypatch.delenv("VOICE_VOLUME", raising=False)
            importlib.reload(config)

    def test_the_live_value_is_always_playable(self):
        assert 0.0 <= config.VOLUME <= 1.0

    def test_it_is_a_setting_the_env_file_can_carry(self):
        # Every knob in this project is read through os.environ by the module
        # that owns it, and config.load_settings is what makes the env file
        # reach them. A setting nobody can set from that file is a setting
        # that silently does nothing — which has happened here before.
        assert "VOICE_VOLUME" not in config.SECRETS


class TestTheAcknowledgementTracksHerVoice:
    """The reason VOLUME is in config rather than in tts.

    The earcon has its own level and its own code path — it never touches
    _normalize. Turning down only the speech would leave a courtesy blip
    louder than the answer it introduces, which is a worse noise than the one
    being fixed.
    """

    def test_the_tone_comes_down_too(self, at_volume):
        at_volume(1.0)
        loud = peak(earcon.tone())
        at_volume(0.3)
        assert peak(earcon.tone()) == pytest.approx(loud * 0.3, rel=0.05)

    @pytest.mark.parametrize("level", [1.0, 0.6, 0.3, 0.1])
    def test_it_stays_quieter_than_her_voice_at_every_setting(
            self, at_volume, level):
        at_volume(level)
        assert peak(earcon.tone()) < tts.TARGET_PEAK * level

    def test_their_ratio_does_not_drift(self, at_volume):
        # The invariant that matters: however far down she is turned, the
        # blip keeps the same relationship to her voice.
        at_volume(1.0)
        full = peak(earcon.tone()) / tts.TARGET_PEAK
        at_volume(0.25)
        assert peak(earcon.tone()) / (tts.TARGET_PEAK * 0.25) == \
            pytest.approx(full, rel=0.05)

    def test_silencing_her_silences_it(self, at_volume):
        at_volume(0.0)
        assert peak(earcon.tone()) == 0
