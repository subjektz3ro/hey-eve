"""The sound that means "heard you".

It exists because silence and failure look identical from across a room. Twelve
seconds of nothing after a question is indistinguishable from a wake word that
missed, and the reasonable response to that ambiguity — say it again — starts
the entire wait over.

Almost everything here is about what the tone must NOT do. It fires many times
an hour in a room someone lives in, one line ahead of the model call, on the
ordinary path. So it must not click, must not be loud, must not block the
caller, must not raise, and must not touch the amplifier-wake clock it sits
next to. Those are the tests; the sine wave is the easy part.
"""
from __future__ import annotations

import array
import subprocess
import threading

import pytest

from eve import config, earcon


class TestTheToneItself:
    def test_it_is_well_formed_pcm_at_the_players_rate(self):
        pcm = earcon.tone()
        assert len(pcm) % 2 == 0, "s16 mono cannot have an odd byte count"
        expected = int(config.TTS_RATE * earcon.DURATION_S)
        assert len(pcm) // 2 == pytest.approx(expected, abs=1)

    def test_it_is_built_once_and_kept(self):
        # It fires on every addressed utterance. Rebuilding a few thousand
        # samples each time would not be expensive, but it would be pointless,
        # and the cache is what lets acknowledge() be honestly non-blocking.
        assert earcon.tone() is earcon.tone()

    def test_it_never_reaches_the_rails(self):
        # A courtesy noise that clips is worse than no courtesy noise. The
        # clip in _build is a guard, not a shaping tool — nothing should ever
        # be getting near it at the default level.
        samples = array.array("h", earcon.tone())
        assert max(abs(s) for s in samples) < 32767

    def test_it_is_quieter_than_speech(self):
        # tts._normalize lifts speech to TARGET_PEAK on purpose, because the
        # SoundSticks are quiet across a room. An acknowledgement mixed at
        # that level would be louder than the answer it introduces.
        from eve import tts
        samples = array.array("h", earcon.tone())
        assert max(abs(s) for s in samples) < tts.TARGET_PEAK

    def test_it_starts_from_silence(self):
        # The 5ms raised-cosine attack. A tone that begins at full amplitude
        # starts with a step discontinuity, and a step is a click — which
        # reads as a fault rather than as an acknowledgement.
        samples = array.array("h", earcon.tone())
        assert abs(samples[0]) <= 1

    def test_it_decays_to_nothing_rather_than_being_cut_off(self):
        # Same argument at the other end: a rectangular envelope ends on
        # whatever phase it happens to be at, and clicks.
        samples = array.array("h", earcon.tone())
        tail = samples[-int(len(samples) * 0.05):]
        peak = max(abs(s) for s in samples)
        assert max(abs(s) for s in tail) < peak * 0.1

    def test_it_is_loudest_early(self):
        # An acknowledgement should land and be gone, not swell.
        samples = array.array("h", earcon.tone())
        half = len(samples) // 2
        assert max(abs(s) for s in samples[:half]) > \
            max(abs(s) for s in samples[half:])

    def test_it_is_short_enough_to_not_be_a_notification_sound(self):
        # Fires many times an hour, forever. The failure mode this guards is
        # someone disabling the whole feature because it became irritating.
        assert earcon.DURATION_S <= 0.3


class TestItNeverCostsTheCaller:
    def test_acknowledge_returns_before_the_sound_has_played(self, monkeypatch):
        # The caller's next move is a model call. Waiting on a Bluetooth
        # player to open would spend exactly the latency this exists to hide.
        started = threading.Event()
        release = threading.Event()

        def slow_run(*args, **kwargs):
            started.set()
            release.wait(timeout=5)

        monkeypatch.setattr(subprocess, "run", slow_run)
        earcon.acknowledge()
        assert started.wait(timeout=2), "the player never started"
        release.set()          # acknowledge() already returned, above

    def test_a_missing_player_is_survived(self, monkeypatch):
        # aplay absent, or the ALSA device gone with the Bluetooth link. This
        # runs one line before the model call on the ordinary path; an
        # assistant that dies because a courtesy noise failed is worse than a
        # silent one.
        def explode(*args, **kwargs):
            raise FileNotFoundError("aplay")

        monkeypatch.setattr(subprocess, "run", explode)
        earcon._play(b"\0\0")          # must not raise

    def test_a_wedged_player_is_survived(self, monkeypatch):
        def hang(*args, **kwargs):
            raise subprocess.TimeoutExpired("aplay", 5)

        monkeypatch.setattr(subprocess, "run", hang)
        earcon._play(b"\0\0")

    def test_the_player_is_bounded(self, monkeypatch):
        # Without a timeout, one stuck device accumulates a stuck process per
        # utterance for the life of the service.
        seen = {}

        def capture(*args, **kwargs):
            seen.update(kwargs)

        monkeypatch.setattr(subprocess, "run", capture)
        earcon._play(b"\0\0")
        assert 0 < seen["timeout"] <= 10

    def test_the_player_thread_cannot_hold_shutdown_open(self, monkeypatch):
        threads = []
        real = threading.Thread          # captured, or the lambda recurses
        monkeypatch.setattr(
            earcon.threading, "Thread",
            lambda **kw: threads.append(kw) or real(**kw))
        monkeypatch.setattr(subprocess, "run", lambda *a, **k: None)
        earcon.acknowledge()
        assert threads and threads[0]["daemon"] is True

    def test_a_broken_tone_does_not_reach_the_caller(self, monkeypatch):
        # numpy missing, or someone setting VOICE_EARCON_S to nonsense.
        monkeypatch.setattr(earcon, "tone", lambda: (_ for _ in ()).throw(
            RuntimeError("no numpy")))
        earcon.acknowledge()           # must not raise


class TestItCanBeTurnedOff:
    def test_disabling_it_plays_nothing(self, monkeypatch):
        played = []
        monkeypatch.setattr(earcon, "ENABLED", False)
        monkeypatch.setattr(subprocess, "run",
                            lambda *a, **k: played.append(a))
        earcon.acknowledge()
        assert played == []


class TestItLeavesTheAmplifierLogicAlone:
    def test_it_does_not_touch_the_last_spoke_clock(self, monkeypatch):
        """The one constraint that came from outside the code.

        speech._last_spoke_at drives the amplifier-wake padding: after
        _SPEAKER_IDLE_S of quiet, the next reply gets silence in front of it so
        the SoundSticks eat that instead of the first sentence. That logic
        works and was explicitly out of scope.

        Updating the clock here would suppress the padding on the reply that
        follows — plausibly an improvement, plausibly the return of a bug that
        cost people the first sentence of every answer. It is not a change to
        make silently as a side effect of adding a noise, so this pins it.
        """
        from eve import speech

        monkeypatch.setattr(subprocess, "run", lambda *a, **k: None)
        before = speech._last_spoke_at
        earcon.acknowledge()
        assert speech._last_spoke_at == before


class TestWhereItFiresFrom:
    def test_it_is_not_reached_before_the_wake_word_matches(self):
        """It must never chirp at a conversation that was not addressed to her.

        The microphone gets muted specifically to keep her out of ordinary
        conversation, so a tone on every captured utterance would be the exact
        opposite of what this room wants. main._listen_loop therefore calls it
        after wake.address() has returned a request, not where the capture
        closes — which costs the whisper decode and buys a sound that cannot
        fire at anyone else.
        """
        import inspect

        from eve import main

        source = inspect.getsource(main._listen_loop)
        acknowledged = source.index("earcon.acknowledge()")
        addressed = source.index("request = wake.address(transcript)")
        assert addressed < acknowledged, \
            "the tone must not fire before the utterance is known to be hers"
