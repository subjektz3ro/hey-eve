"""Speech in and out.

Both directions sit behind small protocols so the inference backends can be
swapped without touching the rest of the app — which is the whole point of
the seam: an accelerator replaces the Whisper implementation here and nothing
else in the project changes.
"""
from __future__ import annotations

import os
import re
import signal
import subprocess
import threading
import time
import wave
from pathlib import Path
from typing import cast
from typing import Protocol

from eve import config, log, tts, vad, viseme

# Bluetooth buffers audio: aplay exits before the speakers have finished
# sounding. Opening the microphone straight away captures the tail of our own
# voice, which lands in Whisper as garbage mixed into whatever was actually
# said. Wait out the buffer before listening again.
_PLAYBACK_SETTLE_S = 0.45
# Roughly how far the speakers lag the pipe over Bluetooth.
_PLAYBACK_LATENCY_S = 0.20

# The SoundSticks power their amplifier down when nothing has played for a
# while. The Bluetooth link stays up throughout — bluetoothctl reports the
# device as connected the whole time — so there is nothing to query: the amp
# is asleep and whatever arrives during the second it takes to come back is
# simply gone. That was the first sentence of the reply.
#
# The signal we do have is our own clock. After this long without playing,
# assume the amp has dropped out and send silence first, so it eats that
# instead of the answer.
_SPEAKER_IDLE_S = float(os.environ.get("VOICE_SPEAKER_IDLE_S", "45"))
_SPEAKER_WAKE_S = float(os.environ.get("VOICE_SPEAKER_WAKE_S", "1.5"))

_last_spoke_at = 0.0


class Transcriber(Protocol):
    """Audio file on disk -> what was said."""

    def transcribe(self, wav_path: Path) -> str: ...


# --- output --------------------------------------------------------------

_SENTENCE = re.compile(r"(?<=[.!?])\s+")


def _sentences(text: str) -> list[str]:
    """Split a reply into speakable pieces, keeping them worth synthesizing.

    Very short fragments are merged into their neighbour: synthesis has a
    fixed per-call cost, so "Yes." on its own would spend more time starting
    up than speaking.
    """
    pieces: list[str] = []
    for part in _SENTENCE.split(text.strip()):
        part = part.strip()
        if not part:
            continue
        if pieces and len(part) < 25:
            pieces[-1] = f"{pieces[-1]} {part}"
        else:
            pieces.append(part)
    return pieces


# Kokoro's throughput on this Pi, measured: seconds of audio produced per
# second of compute. Only a starting guess — the real ratio for this reply
# replaces it as soon as the first sentence is done.
_SYNTH_RATIO = 0.6
# Spoken characters per second, to estimate audio that does not exist yet.
# Measured across real replies at 18.8-20.2; 14.0 was a guess and 40% low.
_CHARS_PER_SEC = 19.5
# Slack on top of the computed lead, for a synthesis that runs slower than
# the sentences before it did.
_LEAD_SAFETY_S = 1.5


def _stalls(made_audio_s: float, ratio: float, remaining: list[int],
            safety: float = _LEAD_SAFETY_S) -> bool:
    """Whether starting now would leave the player waiting mid-reply.

    Once playback starts it consumes a second of audio per second while
    synthesis supplies about 0.6, so the buffer has to already hold the
    shortfall. An earlier attempt computed that in closed form as (remaining
    synth - remaining audio), which is wrong: it credits the buffer with the
    final sentence's duration, but the player arrives at that sentence
    *before* it has been spoken. Optimistic by exactly one sentence, worth
    1.9s of silence.

    So walk it. Each sentence becomes ready at a predictable moment, the
    player runs dry at another, and if the first ever comes after the second
    there is a gap. Cheap, and it cannot be off by a term.
    """
    dry_at = made_audio_s        # when the player runs out, if nothing arrives
    ready_at = 0.0               # when the next sentence finishes synthesising
    for chars in remaining:
        audio = chars / _CHARS_PER_SEC
        ready_at += audio / ratio
        if ready_at > dry_at - safety:
            return True
        dry_at += audio
    return False


# How much to discount the measured throughput when *deciding* whether to
# start playing. Every sentence after playback begins competes with the pacer
# thread and with aplay, so a figure measured while the CPU was idle flatters
# what comes next, and being wrong here means an audible gap mid-reply.
_PESSIMISM = 0.65


def _honest_ratio(made_audio_s: float, synth_s: float) -> float:
    """Audio seconds produced per second of compute, as measured.

    What synthesis is actually doing, with no safety margin — which is what
    anything *reporting* to a person wants. Telling someone eight seconds and
    finishing in five is not caution, it is being wrong.
    """
    ratio = made_audio_s / synth_s if synth_s > 0 else _SYNTH_RATIO
    return min(3.0, max(0.2, ratio))


def _measured_ratio(made_audio_s: float, synth_s: float) -> float:
    """The same figure, discounted, for deciding when playback may start.

    The discount belongs to that decision and nothing else. It used to leak
    into the countdown as well, which made every "~Ns" on the panel 1.54x
    longer than the truth — the one number on that screen a person checks
    against a clock.
    """
    return _honest_ratio(made_audio_s, synth_s) * _PESSIMISM


def _play_starts_after(sentences: list[str]) -> int:
    """Which sentence playback is expected to begin after.

    Predicted up front purely so the progress bar has an honest denominator.
    Being wrong costs nothing — the real decision is still made per sentence
    against real measurements — but it means the bar fills as the first word
    arrives rather than stopping at some arbitrary fraction.
    """
    made = 0.0
    ratio = _SYNTH_RATIO * _PESSIMISM
    for index, sentence in enumerate(sentences):
        made += len(sentence) / _CHARS_PER_SEC
        if not _stalls(made, ratio, [len(rest) for rest in sentences[index + 1:]]):
            return index
    return len(sentences) - 1


# --- being interrupted ----------------------------------------------------
# The latency problem has two solutions and only one of them is engineering.
# Kokoro can be made faster; a reply can also be made *escapable*. Today a
# wrong or over-long answer is a trap — it is going to finish, and the only
# way out is to wait it out — so every second of synthesis is a second nobody
# can spend. Being able to say "eve—" and have her stop turns that from a
# hostage situation into a pause.
#
# The obstacle is acoustic. Her microphone hears her own voice through the
# SoundSticks, and Silero calls that speech because it is speech, so a naive
# detector triggers on herself instantly and permanently. What makes it
# tractable is that she knows precisely what she is playing: the pacer already
# walks `rendered` on a real-time clock and computes the loudness of the block
# currently being heard. The room is compared against that rather than against
# a fixed threshold, so what is being asked is not "is there sound" but "is
# there more sound than I am making" — which is the actual question.
#
# It is poor-man's echo suppression and it has one unknown: the gain from her
# digital output, through the amplifier, across the room, into the capsule.
# That is a property of this room, this volume and where the speakers are
# sitting, and there is no way to derive it from here.
#
# So this ships off. VOICE_BARGE_CALIBRATE=1 runs the whole path but never
# interrupts, and logs the highest room-to-own ratio it saw across each reply;
# a handful of replies with nobody talking gives the ceiling her own voice
# reaches, and VOICE_BARGE_RATIO wants to sit above it. Then VOICE_BARGE_IN=1.
# Turning it on untuned would cut her off mid-sentence at random, which is a
# worse assistant than a slow one.
_BARGE_IN = os.environ.get("VOICE_BARGE_IN", "0") not in ("0", "false", "no", "")
_BARGE_CALIBRATE = os.environ.get(
    "VOICE_BARGE_CALIBRATE", "0") not in ("0", "false", "no", "")
# How much louder than her own voice the room has to be. Starts deliberately
# high: a false negative costs the feature, a false positive cuts her off.
_BARGE_RATIO = float(os.environ.get("VOICE_BARGE_RATIO", "2.5"))
# And an absolute floor, so that the quiet moments *between* her words — where
# her own level is near zero and the ratio is therefore enormous — cannot
# trigger on room tone. Room tone measures ~13; speech at the desk ~380.
_BARGE_FLOOR = float(os.environ.get("VOICE_BARGE_FLOOR", "300"))
# Consecutive 32ms chunks required. A door closing is loud and brief; a person
# talking over her is loud and continuous. 5 is 160ms.
_BARGE_CHUNKS = int(os.environ.get("VOICE_BARGE_CHUNKS", "5"))


def _is_interruption(room: float, own: float) -> bool:
    """Whether this chunk is somebody talking over her rather than her.

    `own` is the loudness of the audio being heard at this instant, which the
    pacer knows exactly. Comparing against it rather than against a fixed
    threshold is the entire trick: a fixed threshold cannot separate her voice
    from a person's, because at the microphone they are the same kind of
    signal at similar levels.

    The floor is the other half, and it is not redundant. Between her words —
    and in the gaps inside them — `own` falls to nearly nothing, so the ratio
    against ordinary room tone goes to infinity and every quiet moment reads
    as an interruption. Requiring an absolute level as well means the room has
    to be genuinely loud *and* louder than her.
    """
    if room < _BARGE_FLOOR:
        return False
    return room / max(own, 1.0) > _BARGE_RATIO


def speak(text: str, on_level: "callable | None" = None,
          on_wave: "callable | None" = None,
          on_done: "callable | None" = None,
          on_progress: "callable | None" = None,
          on_speech: "callable | None" = None,
          on_audio: "callable | None" = None) -> None:
    """Say `text` through the default ALSA playback device.

    Kokoro can make audio more slowly than playback on the reference host.
    Handing each sentence to the player the moment it is finished can therefore
    drain the buffer at sentence boundaries: the audio gaps and the trace
    freezes with it. Multi-sentence replies cannot be smooth that way.

    So playback is held until the finished audio is long enough to cover the
    time the remaining sentences still need. For a one-sentence reply — the
    common case, since the model is told to be brief — the remainder is empty
    and there is no wait at all.

    The display is driven by a separate pacer thread walking the audio on a
    real-time clock. Reporting from the write loop instead looks choppy: the
    player accepts a large buffer instantly, so every callback fires in a
    burst and then nothing moves until the buffer drains.
    """
    import numpy as np

    sentences = _sentences(text)
    if not sentences:
        return

    process: "subprocess.Popen | None" = None
    pacer: "threading.Thread | None" = None
    # How long the player spent with nothing to play. Any of this is an
    # audible gap mid-sentence, so it is measured rather than assumed.
    starved = [0.0]
    # True while the player has run dry mid-reply and is waiting on the next
    # sentence. The panel needs this: a pause it does not explain reads as a
    # crash, and the trace sitting frozen on SIGNAL OUT is exactly what that
    # looks like from the other side of the room.
    stalling = [False]
    rendered = bytearray()          # everything handed to the player so far
    rendered_lock = threading.Lock()
    finished = threading.Event()
    all_written = threading.Event()  # every sentence is in the player
    # Her own loudness at the moment currently being heard, published by the
    # pacer so the interruption listener has something to compare the room
    # against. See the note above _BARGE_IN.
    playing_level = [0.0]
    interrupted = threading.Event()

    def pace() -> None:
        """Report the envelope at the speed it is actually heard."""
        step = int(config.TTS_RATE * 0.03) * 2       # 30ms of audio per update
        position = 0
        # The clock starts when audio first exists, not when this thread does:
        # synthesis is slower than real time, so starting it here would race
        # through the backlog and then freeze.
        started = None
        while True:
            with rendered_lock:
                available = len(rendered)
                block = bytes(rendered[position:position + step]) if position + step <= available else b""
            if not block:
                if finished.is_set():
                    # `finished` means the writer is done, so `rendered` can
                    # never grow again — and a tail shorter than one step can
                    # therefore never become a full block. The test used to
                    # be `position >= available`, which that tail makes
                    # permanently false: position stops at the last whole
                    # step and this loop spun forever.
                    #
                    # tts.synth returns an arbitrary even number of bytes, so
                    # a reply whose length happened to be an exact multiple
                    # of 2646 exited cleanly and every other one — about
                    # 1322 in 1323 — leaked a thread that woke 200 times a
                    # second for the life of the process, and burned the full
                    # two seconds of pacer.join() before the microphone
                    # reopened. Two orphans were still spinning on the Pi
                    # after two replies when this was found.
                    break
                if started is not None and not all_written.is_set():
                    stalling[0] = True
                    # Playback has begun, more sentences are still coming, and
                    # the next block does not exist: that is the gap, and it
                    # is the only thing that is. Two things that are not:
                    # falling behind the pacer's own clock (ordinary jitter on
                    # a loaded Pi) and the spin at the end while aplay drains
                    # its Bluetooth buffer. Counting those reported seconds of
                    # "gap" nobody heard, twice.
                    starved[0] += 0.005
                time.sleep(0.005)
                continue
            stalling[0] = False
            if started is None:
                # Bluetooth delays what is audible, and any amplifier-wake
                # padding sits in front of the speech; offset by both so
                # picture and sound agree.
                started = time.monotonic() + _PLAYBACK_LATENCY_S + warmup
            samples = np.frombuffer(block, dtype=np.int16)
            if samples.size:
                loudness = float(np.sqrt((samples.astype(float) ** 2).mean()))
                # What she is making right now, for the listener to measure the
                # room against. Published unconditionally: it is one float and
                # the alternative is a second RMS pass over the same samples.
                playing_level[0] = loudness
                if on_level is not None:
                    on_level(min(1.0, loudness / 9000.0))
                if on_wave is not None:
                    on_wave(samples)
                if on_audio is not None:
                    # Where playback has reached, in seconds of reply. The
                    # mouth needs this: the shape it should be making is a
                    # function of position in the text, not of loudness.
                    on_audio((position / 2) / config.TTS_RATE,
                             min(1.0, loudness / 9000.0))
            position += step
            # Sleep until this block's moment arrives, rather than a fixed
            # interval, so the trace cannot drift out of step with the audio.
            due = started + (position / 2) / config.TTS_RATE
            behind = time.monotonic() - due
            if behind > 0.5:
                # Synthesis stalled and the player drained; re-anchor rather
                # than trying to catch up in one burst.
                started += behind
            time.sleep(max(0.0, due - time.monotonic()))

    def watch_for_interruption() -> None:
        """Stop her talking when somebody talks over her.

        Runs a second capture alongside playback and asks, of each 32ms chunk,
        whether the room is louder than she is by more than _BARGE_RATIO. Both
        halves of that matter: the ratio is what discriminates a person from
        her own voice coming back through the speakers, and the floor is what
        stops the near-silence *between* her words — where her level is nearly
        zero and any ratio is therefore enormous — reading as an interruption.

        Silero is asked last rather than first. It is the expensive question
        and it agrees with almost anything voice-shaped, including her; the
        cheap comparisons above eliminate nearly every chunk before it is
        consulted. It stays in because the two failure modes it does catch —
        a door, a dropped mug — are loud, brief and not speech.

        Kills the player directly rather than asking the writer to stop. The
        writer is usually blocked in stdin.write against a full ALSA buffer,
        so there is no loop for a flag to be noticed in; killing it turns that
        into a BrokenPipeError, which the caller already handles, and which
        `interrupted` then tells apart from a genuinely dropped link.
        """
        chunk_bytes = int(config.MIC_RATE * _CHUNK_S) * 2
        try:
            capture = subprocess.Popen(
                ["arecord", "-q", "-D", mic_device(), "-f", "S16_LE",
                 "-r", str(config.MIC_RATE), "-c", "1", "-t", "raw"],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
        except (OSError, subprocess.SubprocessError):
            return          # no second microphone available; she just cannot
                            # be interrupted, which is today's behaviour
        run = 0
        loudest = 0.0       # the highest ratio seen, for calibration runs
        try:
            while not finished.is_set():
                chunk = capture.stdout.read(chunk_bytes) if capture.stdout else b""
                if not chunk:
                    break
                samples = np.frombuffer(chunk, dtype=np.int16)
                if not samples.size:
                    continue
                room = float(np.sqrt((samples.astype(float) ** 2).mean()))
                loudest = max(loudest, room / max(playing_level[0], 1.0))
                if not _is_interruption(room, playing_level[0]):
                    run = 0
                    continue
                detector = _detector()
                if detector is not None and detector.speech_probability(
                        samples) < _SPEECH_PROBABILITY:
                    run = 0
                    continue
                run += 1
                if run < _BARGE_CHUNKS or _BARGE_CALIBRATE:
                    continue
                interrupted.set()
                if process is not None and process.poll() is None:
                    process.kill()
                break
        except (OSError, ValueError):
            pass            # a courtesy feature must never end a reply
        finally:
            try:
                capture.terminate()
                capture.wait(timeout=2)
            except (OSError, subprocess.SubprocessError):
                capture.kill()
            if capture.stdout is not None:
                capture.stdout.close()
            if _BARGE_CALIBRATE:
                # The number the threshold has to clear. Run a few replies
                # with nobody talking: the highest of these is how loud she
                # gets in her own microphone, and VOICE_BARGE_RATIO belongs
                # above it. Then run one talking over her to see the headroom.
                log.status(f"     [barge-in: loudest room/own ratio this "
                           f"reply {loudest:.2f}, threshold {_BARGE_RATIO}]")

    def open_player() -> None:
        """Start the player, and the pacer that reports what it is playing."""
        nonlocal process, pacer, warmup
        process = subprocess.Popen(
            ["aplay", "-q", "-f", "S16_LE",
             "-r", str(config.TTS_RATE), "-c", "1", "-"],
            stdin=subprocess.PIPE,
            stderr=subprocess.DEVNULL,  # bluealsa's plugin is chatty on open
        )
        if time.monotonic() - _last_spoke_at > _SPEAKER_IDLE_S:
            # Silence, not speech, for the moment the amplifier is waking.
            # Deliberately not added to `rendered`: the trace should follow
            # the words, not the padding — but the pacer is told about it, or
            # the picture would run ahead of the sound by this much.
            warmup = _SPEAKER_WAKE_S
            process.stdin.write(b"\0" * (int(config.TTS_RATE * warmup) * 2))
        if on_level is not None or on_wave is not None:
            pacer = threading.Thread(target=pace, daemon=True)
            pacer.start()
        # Only once there is something to interrupt, and only when asked for:
        # this opens a second capture for the length of the reply, which is
        # not something to do on every turn on the strength of a threshold
        # nobody has calibrated yet. See the note above _BARGE_IN.
        if _BARGE_IN or _BARGE_CALIBRATE:
            threading.Thread(target=watch_for_interruption, daemon=True).start()

    warmup = 0.0                # silence queued ahead of the speech, if any
    pending = bytearray()       # made, but deliberately not played yet
    mouths: list = []           # viseme timeline, grown a sentence at a time
    queued: list = []           # sentences made but not yet handed over
    spoken_at = [0.0]           # seconds of reply already timelined
    made_chars = 0
    made_audio_s = 0.0
    synth_s = 0.0

    # Everything the progress bar needs. The denominator is the work before
    # the first word is heard, not the whole reply: a bar that fills to 70%
    # and then vanishes because playback started is worse than no bar.
    all_chars = sum(len(sentence) for sentence in sentences) or 1
    target_chars = sum(
        len(sentence) for sentence in sentences[:_play_starts_after(sentences) + 1]
    ) or 1
    sentence_started = [0.0]      # when the in-flight sentence began
    sentence_cost = [0.0]         # how long it is expected to take
    sentence_chars = [0]          # and how much of the reply it accounts for

    def report() -> None:
        """Tick the bar, before the first word and during any stall after it.

        Reporting only on sentence completion would step 0, 50, 100 and sit
        still for eight seconds between steps, which is what made the old
        animation unreadable — motion with no meaning. Interpolating inside
        the current sentence gives a bar that moves continuously and still
        means something.

        It keeps running for the whole reply rather than stopping once audio
        starts, because the pause that matters most is the one *mid*-reply:
        synthesis is slower than playback, so a long answer can drain the
        buffer and go quiet. That silence is unavoidable. Being unexplained
        is not.
        """
        while not finished.is_set():
            if process is not None and not stalling[0]:
                time.sleep(0.05)          # audio is flowing; the trace has it
                continue
            buffering = process is not None
            whole = all_chars if buffering else target_chars
            done = made_chars
            if sentence_cost[0] > 0:
                # Interpolate across the sentence in flight, not across
                # everything still outstanding: the latter restarts the bar
                # from zero at every sentence boundary.
                elapsed = time.monotonic() - sentence_started[0]
                share = min(1.0, elapsed / sentence_cost[0])
                done += share * sentence_chars[0]
            fraction = min(0.99, done / whole)
            # The honest ratio, not the discounted one. _measured_ratio's
            # margin exists so playback does not start too early; spending it
            # again here just told people a number that was half again too
            # big, every single time.
            ratio = _honest_ratio(made_audio_s, synth_s)
            left = max(0.0, (whole - done) / _CHARS_PER_SEC / ratio)
            on_progress(fraction, left, buffering)
            time.sleep(0.1)

    ticker = None
    if on_progress is not None:
        ticker = threading.Thread(target=report, daemon=True)
        ticker.start()

    try:
        for index, sentence in enumerate(sentences):
            if interrupted.is_set():
                # Abandon the rest of the reply rather than synthesising into
                # a player that has been killed. This is where interruption
                # actually pays: synthesis is the expensive half of a turn, so
                # stopping her three sentences in gives back the two that had
                # not been made yet, not just the ones already spoken.
                break
            clock = time.monotonic()
            sentence_started[0] = clock
            # Display-only, so the honest ratio again: an over-long estimate
            # here makes the bar saturate early and then sit still, which is
            # the specific way a progress bar stops meaning anything.
            ratio_now = _honest_ratio(made_audio_s, synth_s)
            sentence_cost[0] = len(sentence) / _CHARS_PER_SEC / ratio_now
            sentence_chars[0] = len(sentence)
            pcm = tts.synth(sentence)
            synth_s += time.monotonic() - clock
            sentence_seconds = len(pcm) / 2 / config.TTS_RATE
            made_audio_s += sentence_seconds
            made_chars += len(sentence)
            queued.append((sentence, sentence_seconds))
            sentence_cost[0] = 0.0        # nothing in flight until the next

            pending.extend(pcm)

            if process is None and _stalls(
                made_audio_s, _measured_ratio(made_audio_s, synth_s),
                [len(rest) for rest in sentences[index + 1:]],
            ):
                continue                      # keep building the lead
            if process is None:
                if on_progress is not None:
                    on_progress(1.0, 0.0, False)  # bar completes as sound starts
                open_player()

            block = bytes(pending)
            pending.clear()
            if on_speech is not None:
                # A sentence's text and the exact length of the audio it
                # produced are both known here, which is all the alignment
                # a mouth of one dash can use.
                for spoken, seconds in queued:
                    mouths.extend(viseme.timeline(spoken, seconds, spoken_at[0]))
                    spoken_at[0] += seconds
                queued.clear()
                on_speech(list(mouths))
            with rendered_lock:
                rendered.extend(block)
            process.stdin.write(block)
            process.stdin.flush()

        all_written.set()
        process.stdin.close()
        process.wait(timeout=120)
    except (BrokenPipeError, OSError, subprocess.SubprocessError) as exc:
        # A dropped Bluetooth link must not kill the assistant — but it must
        # not be *silent* either, and for a long time it was. aplay's stderr
        # goes to DEVNULL because bluealsa's plugin is genuinely chatty on
        # every open, so this line is the only evidence that will ever exist
        # that a reply was thought, billed, and never heard.
        #
        # Found the hard way. After a reboot the speaker came back paired,
        # bonded and trusted but not connected; /etc/asound.conf pins
        # `default` at that one address, so every aplay died on open. The
        # journal showed two immaculate turns — timings, token counts, cost —
        # and she reported success for replies nobody heard.
        #
        # Unless she was stopped on purpose. Interrupting her kills the player
        # from another thread, so the writer lands here with a BrokenPipeError
        # that is indistinguishable from a dropped link by type alone — and
        # reporting "is the speaker connected?" every time somebody talks over
        # her would make the journal useless for the case it was written for.
        if not interrupted.is_set():
            log.status(f"     (nothing was heard: playback failed, "
                       f"{exc.__class__.__name__} — is the speaker connected?)")
    finally:
        global _last_spoke_at
        _last_spoke_at = time.monotonic()
        finished.set()          # first, so the pacer can start unwinding now
        # Reaping the player belongs here, not in the except. tts.synth raises
        # RuntimeError when the weights are missing, and that is not in the
        # tuple above — so an exception mid-reply used to walk straight past
        # the only process.kill() in this function, leaving aplay blocked on a
        # pipe nobody would ever write to again, holding the ALSA device, one
        # per failed reply. main._turn catches and carries on, so they stack.
        if process is not None:
            try:
                if process.stdin is not None and not process.stdin.closed:
                    process.stdin.close()
                if process.poll() is None:
                    process.kill()
                process.wait(timeout=2)
            except (OSError, subprocess.SubprocessError):
                pass          # best-effort teardown; never take the assistant down
        if pacer is not None:
            pacer.join(timeout=2)
        if on_level is not None:
            on_level(0.0)
        # The picture hands back the instant the audio stops. Only the
        # microphone has to wait out the Bluetooth buffer, and letting the
        # panel serve that sentence too made it sit on SIGNAL OUT for half a
        # second after the room had gone quiet — which reads as still busy.
        if on_done is not None:
            on_done()
        if interrupted.is_set():
            log.status("     [stopped: something talked over her]")
        elif starved[0] > 0.1:
            # Only when she was allowed to finish. An interrupted reply has a
            # killed player and a pacer that spent its last moments with
            # nothing arriving, which the gap counter reads as starvation —
            # so reporting both would blame synthesis for a stop somebody
            # asked for.
            log.status(f"     [speech gapped {starved[0]:.2f}s — synthesis "
                       f"fell behind playback]")
        time.sleep(_PLAYBACK_SETTLE_S)   # let the speakers actually go quiet


def speaker_available() -> bool:
    """Whether the default ALSA device can actually be opened right now.

    The same reasoning as tts.paths() returning None rather than raising: a
    missing output device should be a diagnosable startup message, not a
    reply that is thought, billed and then swallowed. It is checked the same
    way the model is, at startup, in main().

    Ten milliseconds of silence rather than an empty stream, because aplay
    given nothing to play can exit successfully without ever opening the
    device — which is precisely the false negative this exists to avoid.
    """
    silence = b"\0" * (int(config.TTS_RATE * 0.01) * 2)
    try:
        probe = subprocess.run(
            ["aplay", "-q", "-f", "S16_LE",
             "-r", str(config.TTS_RATE), "-c", "1", "-"],
            input=silence,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,   # bluealsa's plugin is chatty on open
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return probe.returncode == 0


# --- input ---------------------------------------------------------------

# Endpointing thresholds. Measured against the MV6 at its current gain, where
# room tone sits near RMS 13 and speech at a normal desk distance near 380.
# Silero judges exactly 512 samples at a time, so the capture chunk matches
# its window: 32ms, which also gives the display ~31 updates a second.
_CHUNK_S = vad.WINDOW / vad.RATE
_SPEECH_PROBABILITY = 0.5   # Silero's own recommended threshold
_SPEECH_RMS = 200        # room tone measures ~13, normal speech ~380;
                         # 120 was low enough that clicks and music tripped it
# How long a silence has to run before the turn is considered over. Better
# speech detection does not shorten this: a person gathering their next
# clause is genuinely silent, and 0.45s cut people off mid-thought. Tunable
# without a code change, because the right value is a matter of how someone
# talks rather than something to derive.
_HANG_S = float(os.environ.get("VOICE_HANG_S", "2.0"))
_LEAD_IN_S = 4.0         # give up if nothing is said at all
# There is no limit on how long a person may talk, and there should not be:
# the wall-clock cap that used to live here counted the pauses inside a
# sentence and cut people off mid-thought while they were still speaking.
#
# What remains guards against a *source that never stops* — a television, a
# podcast, music with continuous vocals. Silero calls that speech forever, so
# the hang timer never fires, and the loop would sit recording it and stay
# deaf to the wake word indefinitely.
#
# The distinguishing feature is not length, it is breathing. A person leaves
# 32ms gaps between words even in a three-minute explanation, and any single
# one of them resets this counter. Only genuinely unbroken speech accumulates,
# which is why two minutes of it is a safe thing to call a malfunction.
# Set VOICE_RUNAWAY_S=0 to remove even this.
_RUNAWAY_SPEECH_S = float(os.environ.get("VOICE_RUNAWAY_S", "120"))
_PREROLL_S = 0.4         # audio kept from before the threshold trips
_MIN_SPEECH_S = 0.35     # shorter than this is a cough, not a request

# --- a muted microphone is not a quiet room ------------------------------
# To this loop they were the same thing, and so they were the same thing to
# her: with lead_in=None the recorder waits forever, so a muted microphone
# looked exactly like a healthy idle — READY on the panel, nothing in the
# journal, and completely deaf. That is the same visible symptom the comment
# further down records for a *dead* recorder, except muting is a normal thing
# to do on purpose, so it deserves to be shown rather than diagnosed.
#
# They are distinguishable in the signal, but NOT in the obvious way, and the
# obvious way is a trap worth recording. Measured on the MV6, muted, 93
# consecutive 32ms windows:
#
#     exactly zero      0 of 93
#     min / max RMS     0.4285 / 0.5358
#     peak sample       1
#
# It does not mute to digital silence. It dithers at one LSB, so every window
# is non-zero and a test for `rms == 0` — which is what "a hardware mute emits
# silence" leads you to write — would never have fired once. The feature would
# have shipped, passed its tests, and done nothing at all.
#
# Room tone on the same microphone measures ~13 (see the endpointing note
# above), so one sits with 1.9x headroom over the mute and 24x under a live
# room. That gap is wide because a listening microphone always hears
# something and a muted one only hears its own converter.
#
# Wrong low, the run never starts, nothing is shown, and the behaviour is
# exactly what it was before any of this. Wrong high, a working assistant in a
# quiet room says MUTED and switches off its own face, which is a fault where
# there was none. Move it only against numbers from the microphone in hand.
_SILENT_RMS = float(os.environ.get("VOICE_SILENT_RMS", "1.0"))
# How long that has to hold before it is a mute rather than a gap between
# words. Long enough not to flicker, short enough that reaching for the mute
# button and looking up at the panel shows the answer already there.
MUTE_AFTER_S = float(os.environ.get("VOICE_MUTE_AFTER_S", "3.0"))

# --- and the microphone's own account of itself ---------------------------
# The above infers a mute from the signal, which is the best that can be done
# for a device that will not say. This one will: the MV6 exposes a boolean
# 'Microphone Capture Switch' on its control interface that follows the touch
# panel exactly — verified by reading it in both positions.
#
# Worth preferring wherever it exists, and not only because it is exact. The
# signal cannot tell a muted microphone from a broken one; both deliver
# nothing. The switch can, and that distinction is the difference between a
# panel that says MUTED because you pressed a button and a panel that says
# something is wrong — which is the failure this project has already been
# bitten by twice, both times looking perfectly healthy while stone deaf.
#
# Read through amixer rather than by binding libasound: the control interface
# is a separate device from the PCM, so this works while she holds the
# microphone open, and it measures 4ms. Polled at 1Hz and only while the
# signal is already silent, so an ordinary conversation spawns nothing.
_MUTE_POLL_S = 1.0
_CARD_RE = re.compile(r"CARD=([^,\s]+)")
_mute_state: list = [0.0, None]      # when it was last read, what it said


def _mic_card() -> str:
    """The ALSA card of the configured capture device, for amixer -c."""
    device = mic_device()
    if match := _CARD_RE.search(device):
        return match.group(1)
    # hw:1,0 and plughw:1,0 — the field after the colon is the card index.
    return device.partition(":")[2].partition(",")[0] or "0"


def _read_mute() -> bool | None:
    """Ask the card whether its capture switch is off. None if it will not say.

    Matches on the name ending in 'Capture Switch' rather than on this
    microphone's numid, which is assigned per device and would silently read
    the wrong control on anything else. That suffix is the ALSA convention.
    """
    try:
        result = subprocess.run(
            ["amixer", "-c", _mic_card(), "contents"],
            capture_output=True, text=True, timeout=3, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    switch = False
    for line in result.stdout.splitlines():
        line = line.strip()
        if line.startswith("numid="):
            switch = line.endswith("Capture Switch'")
        elif switch and line.startswith(": values="):
            values = [v.strip() for v in line.partition("=")[2].split(",")]
            if values and all(value in ("on", "off") for value in values):
                # Muted only when every channel is; a half-muted stereo pair
                # is still delivering something.
                return all(value == "off" for value in values)
    return None


def mic_muted(now: float | None = None) -> bool | None:
    """Whether the microphone says it is muted. None when it has no opinion.

    Cached, because the caller is the capture loop and asks about thirty times
    a second. One subprocess per second is free; thirty is not.
    """
    now = time.monotonic() if now is None else now
    if now - _mute_state[0] < _MUTE_POLL_S:
        return _mute_state[1]
    _mute_state[0] = now
    _mute_state[1] = _read_mute()
    return _mute_state[1]
# How long to wait before opening the microphone again after the recorder
# itself failed. Long enough that a missing device is not a busy loop, short
# enough that plugging it back in feels immediate. She keeps retrying forever
# on purpose: recovering without a restart is the whole point.
_CAPTURE_RETRY_S = 2.0
# The longest single capture, as a backstop rather than a policy. The runaway
# guard above bounds *unbroken* speech, and a television is not unbroken — it
# has the same inter-word gaps a person does, so it resets that counter just
# as a person would. The only remaining exit is _HANG_S of continuous quiet,
# which continuous programme audio may not offer for a long time. This bounds
# what whisper is then handed. A person does not reach it: five minutes of
# genuinely uninterrupted speech is not a request, it is a radio.
_MAX_CAPTURE_S = float(os.environ.get("VOICE_MAX_CAPTURE_S", "300"))
# Ceiling on one transcription. Generous against 3.9x real time on the longest
# capture above; it exists to bound a wedge, not to shorten a decode.
_TRANSCRIBE_TIMEOUT_S = float(os.environ.get("VOICE_TRANSCRIBE_TIMEOUT_S", "180"))


_UNLOADED_DETECTOR = object()
_DETECTOR: vad.SileroVAD | None | object = _UNLOADED_DETECTOR


def _detector() -> vad.SileroVAD | None:
    """Load Silero on first audio use, after installer integrity checks run."""
    global _DETECTOR
    if _DETECTOR is _UNLOADED_DETECTOR:
        _DETECTOR = vad.load()
    return cast("vad.SileroVAD | None", _DETECTOR)

# The microphone, worked out once if nobody named one.
_mic_device: str | None = None


def mic_device() -> str:
    """The capture device to open, discovering one if none was configured.

    `arecord -l` lists capture *hardware*, so the first card it names is a real
    microphone rather than whatever ALSA's `default` currently points at — and
    on a box with a Bluetooth speaker pinned as the default, the capture side
    of `default` resolves to bluealsa and does not open at all.

    plughw: rather than hw:, so ALSA does the rate conversion instead of
    refusing a device that cannot do 16kHz natively.

    Worked out once and remembered. It runs a subprocess, and the caller is
    the listen loop, which opens a recorder every capture.
    """
    global _mic_device
    if config.MIC_DEVICE:
        return config.MIC_DEVICE
    if _mic_device is not None:
        return _mic_device
    _mic_device = _find_microphone() or "default"
    log.status(f"     (microphone: {_mic_device}"
               f"{' — none found, trying the default' if _mic_device == 'default' else ''})")
    return _mic_device


def _find_microphone() -> str:
    """The first capture card ALSA reports, as a plughw device name."""
    try:
        listing = subprocess.run(["arecord", "-l"], capture_output=True,
                                 text=True, timeout=5, check=False).stdout
    except (OSError, subprocess.SubprocessError):
        return ""
    match = re.search(r"^card \d+: (\S+).*?, device (\d+):", listing, re.M)
    return (f"plughw:CARD={match.group(1)},DEV={match.group(2)}"
            if match else "")


def record_until_silence(
    path: Path,
    lead_in: float | None = _LEAD_IN_S,
    on_level: "callable | None" = None,
    on_wave: "callable | None" = None,
    on_activity: "callable | None" = None,
    should_stop: "callable | None" = None,
) -> Path | None:
    """Record one utterance, stopping when the speaker stops.

    `lead_in` is how long to wait for speech to begin; None waits forever,
    which is what always-listening mode wants. `on_level` receives each
    chunk's loudness scaled to 0..1, so a display can follow the room.

    `should_stop` is how anything else ever gets a turn. With lead_in=None
    this function owns the process for as long as nobody speaks, so a caller
    that wants to do something on its own initiative — say a thing it promised
    earlier, which is the only reason this exists — has no moment to do it in.
    Polled once per chunk, and honoured only while nothing has been said yet:
    the answer to "may I interrupt" is always no once somebody has started
    talking, so the worst it can cost is the length of the sentence in
    progress. It returns None when it fires, which is the same answer silence
    gives, so no caller needs a third case.

    Returns None when nothing was said, so the caller can loop again rather
    than paying to transcribe silence.
    """
    import numpy as np

    chunk_bytes = int(config.MIC_RATE * _CHUNK_S) * 2  # s16 mono
    try:
        capture = subprocess.Popen(
            [
                "arecord", "-q",
                "-D", mic_device(),
                "-f", "S16_LE",
                "-r", str(config.MIC_RATE),
                "-c", "1",
                "-t", "raw",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except OSError as exc:
        # Almost always alsa-utils missing outright. It ships on Raspberry Pi
        # OS, which is exactly why this went unnoticed for so long — the same
        # reason the microphone default was one particular Shure. On a minimal
        # Debian or Fedora it is not installed, and unguarded this raised
        # FileNotFoundError straight out of the listen loop, past a main()
        # that catches only KeyboardInterrupt, into a restart loop reporting a
        # traceback instead of the one sentence worth saying.
        #
        # Handled like every other dead recorder below: say what is wrong,
        # pause so a permanent fault is not a busy loop, and keep trying,
        # because installing the package should fix her without a restart.
        log.status(f"     (cannot record: {exc}. Is alsa-utils installed?)")
        time.sleep(_CAPTURE_RETRY_S)
        return None

    # Keep the audio from just *before* speech was detected. Without it the
    # attack of the first word is lost — inaudible in a long sentence, fatal
    # in a two-word wake phrase, where "hey claude" reached Whisper as "ey
    # claude" and came back as "Yeah."
    import collections

    preroll: collections.deque[bytes] = collections.deque(
        maxlen=max(1, int(_PREROLL_S / _CHUNK_S))
    )

    detector = _detector()
    if detector is not None:
        detector.reset()

    audio, heard_speech, quiet_for, waited, spoken_for = bytearray(), False, 0.0, 0.0, 0.0
    unbroken = 0.0      # speech with no gap at all; see _RUNAWAY_SPEECH_S
    silent_for = 0.0    # digital silence, which is not quiet; see _SILENT_RMS
    try:
        while True:
            chunk = capture.stdout.read(chunk_bytes)
            if not chunk:
                break
            samples = np.frombuffer(chunk, dtype=np.int16)
            loudness = (
                float(np.sqrt((samples.astype(float) ** 2).mean()))
                if samples.size else 0.0
            )
            if on_level is not None:
                on_level(min(1.0, loudness / 2500.0))
            if on_wave is not None:
                on_wave(samples)

            if loudness <= _SILENT_RMS:
                silent_for += _CHUNK_S
            elif silent_for:
                # Signal is back. Silero carries RNN state between calls, and
                # it has just been handed however long the mute lasted; start
                # it clean rather than letting the silence colour its reading
                # of the first words after it.
                silent_for = 0.0
                if detector is not None:
                    detector.reset()

            # Speech, as judged by the model — falling back to loudness only
            # if the model could not be loaded. Skipped outright once the
            # input has been silent long enough to be a mute rather than a
            # pause: Silero is the per-chunk cost of this loop, and asking it
            # thirty times a second whether a stream of zeros contains speech
            # is precisely the work there is no point doing.
            if silent_for >= MUTE_AFTER_S:
                is_speech = False
            elif detector is not None:
                is_speech = detector.speech_probability(samples) >= _SPEECH_PROBABILITY
            else:
                is_speech = loudness > _SPEECH_RMS

            if on_activity is not None:
                # Only meaningful when a runaway is actually building; for a
                # person this stays far away and nothing is ever shown.
                on_activity(
                    min(1.0, loudness / 2500.0), is_speech,
                    _RUNAWAY_SPEECH_S - unbroken
                    if (heard_speech and _RUNAWAY_SPEECH_S) else None,
                    silent_for,
                )

            if not heard_speech and should_stop is not None and should_stop():
                # Something the caller owes has come due and nobody is
                # mid-sentence. `heard_speech` is still False, so this falls
                # out through the ordinary "nothing was said" path below and
                # returns None — the caller loops, notices what it was waiting
                # to do, and reopens the microphone afterwards.
                break

            if is_speech:
                if not heard_speech:
                    # Recover the attack of the first word before it was loud
                    # enough to trip the threshold.
                    audio += b"".join(preroll)
                    preroll.clear()
                heard_speech, quiet_for = True, 0.0
                audio += chunk
                spoken_for += _CHUNK_S
                unbroken += _CHUNK_S
                if _RUNAWAY_SPEECH_S and unbroken >= _RUNAWAY_SPEECH_S:
                    break
            elif heard_speech:
                # Keep the trailing silence: Whisper transcribes the end of a
                # sentence better with a little room after it.
                quiet_for += _CHUNK_S
                spoken_for += _CHUNK_S
                unbroken = 0.0
                audio += chunk
                if quiet_for >= _HANG_S:
                    break
                if _MAX_CAPTURE_S and spoken_for >= _MAX_CAPTURE_S:
                    log.status(f"     (stopped listening after "
                               f"{spoken_for:.0f}s — is something playing?)")
                    break
            else:
                # Still waiting for anyone to say anything. Hold the most
                # recent chunks so the first word's attack survives.
                preroll.append(chunk)
                waited += _CHUNK_S
                unbroken = 0.0
                if lead_in is not None and waited >= lead_in:
                    break
    finally:
        capture.terminate()
        try:
            capture.wait(timeout=2)
        except subprocess.SubprocessError:
            capture.kill()          # a wedged recorder must not wedge the loop
        if capture.stdout is not None:
            capture.stdout.close()

    # A recorder that died is not the same thing as a room that was quiet, and
    # for a long time this function could not tell you which had happened.
    #
    # After a reboot with the microphone's card renumbered, arecord exits
    # immediately, the read returns b"", and this returned None — exactly what
    # silence returns. The caller's `if captured is None: continue` then span
    # with no sleep and no log line, because stderr is DEVNULL'd. She was
    # permanently deaf, at a hundred spawns a second, still showing READY.
    #
    # Both halves matter: the message is the only diagnostic that will ever
    # exist, and the pause is what stops a dead device becoming a busy loop.
    # Retrying forever is deliberate — plugging the microphone back in should
    # fix her without a restart.
    usable = heard_speech and (spoken_for - quiet_for) >= _MIN_SPEECH_S
    if capture.returncode not in (0, -signal.SIGTERM, -signal.SIGKILL):
        log.status(f"     (the microphone did not open: arecord exited "
                   f"{capture.returncode} on {mic_device()})")
        if not usable:
            # Only pause when there is nothing to show for it. A recorder that
            # died *after* catching a real sentence should not also cost the
            # person the answer.
            time.sleep(_CAPTURE_RETRY_S)
            return None

    # `spoken_for` includes the trailing silence, so subtract it to get the
    # length of actual speech.
    if not usable:
        return None

    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(config.MIC_RATE)
        handle.writeframes(bytes(audio))
    return path


def whisper_paths() -> tuple[Path, Path]:
    """The one binary/model pair covered by the runtime contract."""
    return (
        config.WHISPER_DIR / "build" / "bin" / "whisper-cli",
        config.WHISPER_DIR / "models" / "ggml-base.en.bin",
    )


def whisper_transcriber() -> "WhisperTranscriber":
    """Open the pinned whisper.cpp pair, with an actionable missing-file error."""
    binary, model = whisper_paths()
    if not binary.is_file() or not os.access(binary, os.X_OK) or not model.is_file():
        raise RuntimeError(
            f"pinned whisper.cpp runtime not found under {config.WHISPER_DIR}: "
            f"binary={'ok' if binary.is_file() and os.access(binary, os.X_OK) else 'missing'} "
            f"model={'ok' if model.is_file() else 'missing'}. "
            "Run scripts/provision-whisper.sh."
        )
    return WhisperTranscriber(binary, model)


class WhisperTranscriber:
    """Whisper on the Pi's CPU, via whisper.cpp.

    Deliberately shells out rather than binding the library: the process
    boundary is what lets an accelerated implementation drop in later without
    dragging this one's dependencies along.
    """

    def __init__(self, binary: Path, model: Path) -> None:
        self.binary = binary
        self.model = model

    def transcribe(self, wav_path: Path) -> str:
        result = subprocess.run(
            [
                str(self.binary),
                "-m", str(self.model),
                "-f", str(wav_path),
                "--no-timestamps",
                "--no-prints",
                "--language", "en",
                # Greedy decode over beam search: on short commands the
                # accuracy difference is negligible and it is markedly faster.
                "-bs", "1",
                "-bo", "1",
                # Leave a core free: with all four busy the display thread
                # is starved and the animation visibly stutters.
                "-t", "3",
            ],
            capture_output=True,
            text=True,
            check=False,
            # Unbounded, this is the one call that can take the assistant off
            # the air indefinitely: it blocks the listen loop, so while it
            # runs she is deaf. base.en decodes at about 3.9x real time here,
            # so even the longest capture the recorder will now hand over is
            # well inside this. Reaching it means whisper is wedged, and a
            # wedged transcription should cost one turn, not the afternoon.
            timeout=_TRANSCRIBE_TIMEOUT_S,
        )
        if result.returncode != 0:
            raise RuntimeError(f"whisper failed: {result.stderr.strip()[:200]}")
        return result.stdout.strip()
