"""Run the assistant.

    uv run eve                 # always listening for "hey Eve"
    uv run eve --text          # typed input, spoken replies
    uv run eve --push          # press enter to talk, no wake word
    uv run eve --say "hello"   # speech check only
"""
from __future__ import annotations

import argparse
import os
import signal
import tempfile
import threading
import time
from pathlib import Path

import eve
from eve import assistant
from eve import config
from eve import earcon
from eve import log
from eve import speech
from eve import tts
from eve import wake
from eve import watch

# After answering, keep listening this long for a follow-up that needs no
# wake word — "and what about tomorrow?" should just work. Measured from the
# end of the reply to the moment they start speaking, so this is thinking
# time, not talking time. Twenty-five seconds was not enough to compose a
# real question.
FOLLOW_UP_S = float(os.environ.get("VOICE_FOLLOW_UP_S", "60"))
# How long before the capture ceiling to start showing a countdown.
_CUTOFF_WARN_S = 10.0

# How long the microphone has to deliver nothing at all before the face enters
# its visible sleep state. Muting it is a normal way to use her — it is how you
# keep her out of a conversation you are having with someone else — so being
# in this state for hours is ordinary rather than a fault.
#
# Worth doing rather than cosmetic: the face is the most expensive thing
# running when nothing else is (astro's renderer measures 4.24ms a frame,
# about a fifth of a core, indefinitely), and while the microphone is muted
# there is nobody it is being drawn for. Long enough that a lull in a real
# conversation never reaches it; see speech.MUTE_AFTER_S for the shorter
# threshold that only changes what the panel says.
SLEEP_AFTER_S = float(os.environ.get("VOICE_SLEEP_AFTER_S", "600"))

# Captured speech is written to a file because whisper.cpp reads one, but it
# is written to RAM. /dev/shm never touches the NVMe and is gone at reboot,
# so a recording of whoever was in the room does not outlive the process that
# needed it. Falls back to the normal temp dir if tmpfs is missing.
_AUDIO_TMP = "/dev/shm" if Path("/dev/shm").is_dir() else None


def _clean_exit(signum, frame):
    """Turn SIGTERM into a normal unwind.

    Python's default handler exits the interpreter outright, so `with
    tempfile.TemporaryDirectory()` never ran its cleanup: every `systemctl
    restart` left a directory behind holding the last thing anyone said. Two
    dozen of them accumulated in a single afternoon of iterating.
    """
    raise SystemExit(0)


# What a run of dead input means for the panel. Four states, because they
# answer genuinely different questions: MUTED is the honest replacement for
# READY · SAY EVE, which would be a lie; ASLEEP keeps drawing a recognizable
# sleep state; and DEAF is a fault rather than a choice.
LIVE, MUTED, ASLEEP, DEAF = "live", "muted", "asleep", "deaf"


def _silence_state(silent_for: float, muted: bool | None = None) -> str:
    """What it means that nothing at all has arrived for this long.

    `muted` is the microphone's own account of itself — see speech.mic_muted,
    which reads the capture switch the MV6's touch panel drives. None when the
    device has no opinion, which is every microphone that does not expose one.

    The reason it is worth asking rather than inferring is the DEAF case. The
    signal alone cannot tell a muted microphone from a broken one, because
    both deliver nothing, so a purely signal-based reading has to guess — and
    guessing "muted" on a dead microphone is exactly the failure this project
    has already been bitten by twice: stone deaf, looking perfectly healthy.
    A device that says it is listening while nothing arrives is a fault, and
    it must neither be dressed up as a choice nor quietly slept through.

    Split out of the capture callback so all of this can be tested without a
    microphone, a framebuffer, or a loop that never returns.
    """
    if muted is True:
        return ASLEEP if silent_for >= SLEEP_AFTER_S else MUTED
    if silent_for < speech.MUTE_AFTER_S:
        return LIVE
    if muted is False:
        return DEAF
    # No opinion from the device: fall back to what the signal alone implies,
    # which is the best that can be done and is what every microphone without
    # a capture switch will get.
    return ASLEEP if silent_for >= SLEEP_AFTER_S else MUTED


class _Display:
    """The face, or a no-op stand-in when the panel isn't in use."""

    def __init__(self, enabled: bool) -> None:
        self.orb = None
        self.enabled = enabled
        self.asleep = False
        if enabled:
            self._build(announce=True)

    def _build(self, announce: bool = False) -> None:
        """Fit the face and start it drawing.

        Imported here rather than at the top of the module so that a machine
        with no panel, and a test run with no framebuffer, never pay for numpy
        and PIL and a rasterised font they will not use.
        """
        try:
            from eve import void

            self.orb = void.Face()
            self.orb.start()
            if announce:
                log.status(f"presenting: {config.PRESENT} "
                           f"· voice {config.voice()}")
        except Exception as exc:  # a missing panel must not stop the assistant
            self.orb = None
            log.status(f"(display off: {exc})")

    def sleep(self) -> None:
        """Tell the face she has dropped off. She keeps drawing.

        This used to stop the renderer and blank /dev/fb0, and that was wrong
        for precisely the reason _silence_state is careful about a fault: a
        dark panel cannot be told apart from a crash, a dead backlight or a
        pulled plug, and looking fine while being broken is this project's
        recurring failure. Blanking her to save a fifth of a core was solving
        a problem nobody had at the cost of the one thing the panel is for.

        She is not off, she is asleep, and that is a thing she can show —
        lids down, breathing, the odd twitch, and z's floating away. It costs
        a state and eve/void.py draws it. Tearing the face down and rebuilding
        it went with the old approach, along with the console-switch race that
        came free with restarting a renderer from inside the capture loop.
        """
        if self.asleep:
            return
        self.asleep = True
        log.status("     (asleep — the microphone has gone quiet; muted?)")
        self.state("asleep")

    def wake(self) -> None:
        """She is being spoken to again. Called on every chunk while awake."""
        if not self.asleep:
            return
        self.asleep = False
        log.status("     (awake — the microphone is delivering signal again)")

    def state(self, name: str, level: float = 0.0, note: str = "") -> None:
        if self.orb is not None:
            self.orb.set_state(name, level, note)

    def wave(self, samples) -> None:
        if self.orb is not None and hasattr(self.orb, "set_wave"):
            self.orb.set_wave(samples)

    def speech(self, mouths) -> None:
        """Hand over the mouth shapes for the reply about to be spoken."""
        if self.orb is not None and hasattr(self.orb, "set_speech"):
            self.orb.set_speech(mouths)

    def audio(self, seconds: float, level: float) -> None:
        """Where playback has reached, so the mouth can follow the text."""
        if self.orb is not None and hasattr(self.orb, "set_audio_position"):
            self.orb.set_audio_position(seconds, level)
        elif self.orb is not None:
            self.orb.set_state("speaking", level)

    def close(self) -> None:
        if self.orb is not None:
            self.orb.stop()


def _turn(
    responder: assistant.ClaudeResponder,
    said: str,
    display: _Display,
    *,
    quiet: bool,
) -> None:
    """One exchange: think, print, speak, report what it cost."""
    display.state("thinking")
    started = time.monotonic()
    # The other half of the guard around _speak below. Losing her voice must
    # not lose the assistant — and neither must losing the network. This was
    # bare, so an expired key, a 400, or an outage outlasting the client's
    # retries unwound out of main() and killed the process; the unit restarts
    # always, so what a person saw was the panel going dark mid-question and
    # coming back with no explanation.
    #
    # The model is the one component in this pipeline that is somebody else's
    # and reachable only over a domestic connection. It will fail. It should
    # cost the turn, not the day.
    try:
        reply = responder.respond(said, on_state=display.state)
    except Exception as exc:
        log.status(f"     (could not answer: {exc.__class__.__name__}: {exc})")
        display.state("engaged", 0.0, "NO ANSWER · TRY AGAIN")
        return
    thought_for = time.monotonic() - started

    log.content(f"bar> {reply}")
    tokens_in, tokens_out = responder.last_usage
    turn_cost = responder.cost_usd()
    cost_text = f"${turn_cost:.5f}" if turn_cost is not None else "cost unknown"
    log.status(
        f"     [{thought_for:.1f}s  {tokens_in} in / {tokens_out} out  "
        f"{cost_text}]"
    )
    if not quiet and reply:
        # Synthesis of the first sentence happens before any audio exists;
        # say so rather than showing a silent "speaking". The word is the
        # one in LOOK: the note always overrode the label, so the table said
        # SPEAKING SOON and the panel said SYNTH, which is nobody's idea of
        # what is happening.
        display.state("synthesizing", 0.0, "SPEAKING SOON")
        # Losing her voice must not lose the assistant. This used to be an
        # unguarded call, so anything raising inside synthesis or playback
        # unwound all the way out of main() and the process exited — and
        # because the unit restarts always, what a person saw was a reply
        # being thought about and then nothing, over and over, with the
        # reason buried in a traceback in the journal.
        #
        # It happened for real: the Kokoro weights went missing from
        # ~/.local/share/eve/kokoro, tts.synth raised, and every single turn
        # killed her a few seconds after answering correctly.
        #
        # She can still hear, think, answer and remember without a voice, so
        # the honest behaviour is to say so once and stay up.
        try:
            _speak(reply, display)
        except Exception as exc:
            log.status(f"     (could not speak: {exc})")
            display.state("engaged", 0.0, f"LISTENING {FOLLOW_UP_S:.0f}s")
    else:
        # An empty reply is not nothing happening. The tool loop can exhaust
        # its four iterations with calls still pending, or a turn can come
        # back with only tool_use blocks and no text — and every state call
        # above sits inside the `if`, so the panel simply stayed on THINKING
        # while she went back to listening. She was waiting for you; she just
        # looked like she was still working.
        display.state("engaged", 0.0, f"LISTENING {FOLLOW_UP_S:.0f}s")


def _speak(reply: str, display: _Display) -> None:
    """Say the reply out loud, driving the panel from the audio."""
    speech.speak(
        reply,
        # A real bar, with the seconds remaining beside it: synthesis is
        # the longest wait in a turn and the only one whose length is
        # knowable in advance.
        on_progress=lambda fraction, left, buffering: display.state(
            "synthesizing", fraction,
            # "BUFFERING" rather than "SPEAKING SOON" mid-reply: the
            # first tells you it is still coming, the second reads as
            # starting over.
            ("BUFFERING" if buffering else "SPEAKING SOON")
            + f" {fraction * 100:.0f}%"
            + (f"  ~{left:.0f}s" if left >= 1 else "")),
        on_level=lambda level: display.state("speaking", level),
        on_wave=display.wave,
        on_speech=display.speech,
        on_audio=display.audio,
        # Fires the moment the audio ends, before the microphone's
        # settle wait. Sitting on SPEAKING after the room has gone quiet
        # reads as "it is busy", when in fact it is waiting for you.
        on_done=lambda: display.state(
            "engaged", 0.0, f"LISTENING {FOLLOW_UP_S:.0f}s"),
    )


def _warm_voice() -> None:
    """Build the Kokoro session ahead of the first spoken reply."""
    started = time.monotonic()
    try:
        tts.warm_up()
    except Exception as exc:      # she can still hear, think and answer
        log.status(f"     (voice did not warm: {exc})")
        return
    log.status(f"     (voice ready in {time.monotonic() - started:.1f}s)")


def _prepare_voice(*, quiet: bool) -> None:
    """Warm synthesis and report output readiness without blocking listening."""
    if quiet:
        return
    if tts.paths() is None:
        log.status(f"no voice: Kokoro weights missing from {tts.KOKORO_DIR}. "
                   "She will hear, think and answer, but not speak. "
                   "Fix with scripts/fetch-models.sh")
    else:
        # Open the 325MB graph now, while nobody is waiting on it.  A daemon
        # thread lets a slow disk delay her voice without delaying her ears.
        threading.Thread(target=_warm_voice, daemon=True).start()

    # Model readiness and an ALSA sink are independent.  This used to be an
    # ``elif`` after the catch-all non-quiet branch above, making the warning
    # unreachable on every spoken run.
    if not speech.speaker_available():
        log.status("no speaker: the default ALSA device would not open. She "
                   "will hear, think, answer and synthesise, but nothing will "
                   "be heard. If it is the Bluetooth speaker, "
                   "deploy/connect-speaker.sh brings it back.")


def _listen_loop(
    responder: assistant.ClaudeResponder,
    display: _Display,
    *,
    quiet: bool,
    require_wake: bool,
) -> int:
    """Always-listening loop: transcribe speech, answer what was addressed."""
    transcriber = speech.whisper_transcriber()
    log.status('listening — say "hey Eve" (ctrl-c to quit)')
    log.status(log.banner())

    with tempfile.TemporaryDirectory(dir=_AUDIO_TMP) as tmp:
        heard = Path(tmp) / "heard.wav"
        follow_up_until = 0.0
        # Outside the loop below, because `activity` is rebuilt on every pass
        # and this has to remember whether the fault has already been reported
        # across all of them.
        reported_deaf = [False]

        while True:
            # Anything she promised to come back about, said here rather than
            # from inside the capture — so she is never speaking into an open
            # microphone, and never cuts across somebody mid-sentence.
            #
            # Held while she is asleep. The microphone is muted, which is how
            # you keep her out of a conversation, and from in here she cannot
            # tell whether anybody is even in the room. Announcing into that
            # spends the reminder rather than delivering it, so they keep.
            if not display.asleep:
                for item in watch.due():
                    log.content(f"eve> {item['what']}")
                    if not quiet:
                        display.state("synthesizing", 0.0, "SPEAKING SOON")
                        try:
                            _speak(item["what"], display)
                        except Exception as exc:
                            log.status(f"     (could not speak: {exc})")
                    # She opened this exchange, so leave the door open the way
                    # answering a question does: "snooze that" and "say that
                    # again" both have to land without the wake word.
                    follow_up_until = time.monotonic() + FOLLOW_UP_S

            # When this capture first heard anyone speak. A list because the
            # closure below only appends to it.
            began: list[float] = []

            def activity(level: float, is_speech: bool,
                         remaining: float | None = None,
                         silent_for: float = 0.0) -> None:
                """Say plainly whether it is hearing speech, whether a
                follow-up still needs the wake word, and — the case this loop
                could not previously tell apart from an empty room — whether
                it is being given any signal at all."""
                # Only asked once the signal has already gone quiet, so an
                # ordinary conversation never spawns the amixer poll at all.
                dead = _silence_state(
                    silent_for, speech.mic_muted() if silent_for else None)
                if dead == ASLEEP:
                    display.sleep()
                    return
                # Cheap and idempotent; the first chunk carrying signal after
                # a mute is what brings the panel back, before anything has
                # been said into it.
                display.wake()
                if dead in (MUTED, DEAF):
                    if dead == DEAF and not reported_deaf[0]:
                        # Once per fault, not thirty times a second. This is
                        # the only diagnostic that will ever exist for it:
                        # arecord is alive and returning zeroes, so nothing
                        # else in this loop has anything to complain about.
                        reported_deaf[0] = True
                        log.status("     (no signal: the microphone reports "
                                   f"itself live but {speech.mic_device()} is "
                                   "delivering silence)")
                    # Reusing "idle" rather than inventing a state: each of the
                    # four renderers knows a fixed set of them, and the note is
                    # what carries the meaning here anyway. READY · SAY EVE
                    # would be a lie — she cannot hear it.
                    display.state("idle", 0.0,
                                  "MUTED" if dead == MUTED else "NO SIGNAL")
                    return
                reported_deaf[0] = False
                if is_speech:
                    if not began:
                        began.append(time.monotonic())
                    # Count down before the hard cut. Being truncated is bad;
                    # being truncated with no warning is what made it feel
                    # like the thing had simply stopped listening.
                    if remaining is not None and remaining <= _CUTOFF_WARN_S:
                        display.state("listening", level,
                                      f"SIGNAL IN · {max(0.0, remaining):.0f}s")
                    else:
                        display.state("listening", level)
                    return
                if began:
                    # Mid-utterance, in one of the pauses inside a sentence.
                    # The follow-up clock stopped mattering the moment they
                    # started talking, and showing it running down here told
                    # people it had given up on them while they were speaking.
                    display.state("engaged", level, "LISTENING")
                    return
                remaining = follow_up_until - time.monotonic()
                if remaining > 0:
                    display.state("engaged", level, f"LISTENING {remaining:.0f}s")
                else:
                    display.state("idle", level, "READY · SAY EVE")

            activity(0.0, False)
            captured = speech.record_until_silence(
                heard,
                lead_in=None,  # wait indefinitely for someone to speak
                on_wave=display.wave,
                on_activity=activity,
                # The only way anything gets a turn while nobody is speaking.
                # Never while asleep: nothing is announced in that state, so a
                # due item would end the capture, find that it is being held,
                # and end the next one — a busy loop around a promise she has
                # already decided to keep until she can hear again.
                should_stop=lambda: not display.asleep
                and watch.next_due_in() <= 0,
            )
            if captured is None:
                continue

            display.state("transcribing")
            transcript = transcriber.transcribe(captured)
            log.spoken("heard", transcript)
            if wake.is_noise(transcript):
                continue

            request = wake.address(transcript)
            # Judged from when they STARTED talking, not from now. Checking it
            # here meant a follow-up spent its own window being spoken and
            # then decoded: a twenty-second question plus five seconds of
            # Whisper outlived a twenty-five second window every time, and the
            # answer was discarded for arriving late to a conversation it had
            # been part of from the start.
            in_follow_up = (began[0] if began else time.monotonic()) < follow_up_until

            if request is not None and in_follow_up and wake.is_dismissal(request):
                # "no thanks" is not a question. Close the follow-up window and
                # go back to waiting for the wake word rather than spending a
                # turn, a synthesis and forty seconds of her time on it.
                log.spoken("done", transcript)
                follow_up_until = 0.0
                continue

            if request is None:
                if not (in_follow_up and not require_wake):
                    # Not addressed to us. Print it so wake-word misses are
                    # debuggable rather than mysterious silence.
                    log.spoken("ignored", transcript)
                    continue
                request = transcript.strip()
                if wake.is_dismissal(request):
                    log.spoken("done", transcript)
                    follow_up_until = 0.0
                    continue

            if not request:
                # Wake word alone. Acknowledge on the panel rather than out
                # loud: speaking here means waiting out the playback buffer
                # before the microphone opens, and people answer immediately,
                # so the first half-second of the reply was being lost.
                display.state("listening")
                captured = speech.record_until_silence(
                    heard,
                    lead_in=8.0,
                    on_level=lambda level: display.state("listening", level),
                    on_wave=display.wave,
                )
                if captured is None:
                    continue
                display.state("thinking")
                request = transcriber.transcribe(captured).strip()
                if not request or wake.is_noise(request):
                    continue

            # Heard you, and heard that it was meant for her. This is the
            # first moment either of those is known, and it is about eleven
            # seconds ahead of the first spoken word — the model takes 1.4s
            # and Kokoro the rest, and all of it is silence today.
            #
            # Deliberately here rather than where the capture closes. Firing
            # on every captured utterance would chirp at every conversation in
            # the room, which is precisely what the microphone gets muted to
            # avoid; waiting until the wake word has matched costs the whisper
            # decode and buys a sound that never fires at anyone else. It also
            # makes the tone mean the more useful of the two things: not
            # "audio arrived" but "that was for me" — a missed wake word being
            # the failure people actually hit.
            earcon.acknowledge()
            log.content(f"you> {request}")
            _turn(responder, request, display, quiet=quiet)
            follow_up_until = time.monotonic() + FOLLOW_UP_S


def main() -> int:
    # Before anything opens a file: systemd stops this with SIGTERM.
    signal.signal(signal.SIGTERM, _clean_exit)

    parser = argparse.ArgumentParser(
        description="Voice assistant with local speech processing"
    )
    parser.add_argument(
        "command", nargs="?", choices=("doctor",),
        help="doctor: verify the local model runtime without using hardware",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--say", metavar="TEXT", help="speak TEXT and exit (speech check)")
    mode.add_argument("--text", action="store_true", help="typed input, spoken replies")
    mode.add_argument("--push", action="store_true", help="press enter to talk")
    parser.add_argument("--quiet", action="store_true", help="print replies, do not speak")
    parser.add_argument("--no-display", action="store_true", help="leave the panel alone")
    parser.add_argument("--model", help=f"override the model (default {config.MODEL})")
    args = parser.parse_args()

    if args.command == "doctor":
        if args.say or args.text or args.push or args.quiet or args.no_display \
                or args.model:
            parser.error("doctor cannot be combined with assistant mode options")
        from eve import doctor

        return doctor.main([])

    if args.say:
        speech.speak(args.say)
        return 0

    # The panel first, deliberately. ClaudeResponder raises on a missing API
    # key, and building it first meant that failure happened before there was
    # anything to show it on — a black screen and a traceback in a journal
    # nobody was reading, for the one error env.example says she reports.
    display = _Display(enabled=not args.no_display and not args.text)
    try:
        responder = assistant.ClaudeResponder(model=args.model)
    except Exception as exc:
        display.state("idle", 0.0, "NO API KEY")
        log.status(f"cannot start: {exc}")
        display.close()
        return 1
    log.status(f"model: {responder.model}")
    # Say which settings the env file actually supplied. Most of them did
    # nothing at all until recently, so the first run after this change can
    # legitimately alter her microphone, model or persona — and an operator
    # deserves to see that in the journal rather than discover it by ear.
    if eve.SETTINGS_APPLIED:
        log.status("from " + str(config.VOICE_ENV) + ": "
                   + ", ".join(sorted(eve.SETTINGS_APPLIED)))
    if config.SETTINGS_IGNORED:
        # Not settings, so not applied. Said out loud because a line that is
        # silently ignored is indistinguishable from one that took effect —
        # which is the whole failure load_settings was written to fix.
        log.status("ignored in " + str(config.VOICE_ENV) + " (not a setting): "
                   + ", ".join(sorted(config.SETTINGS_IGNORED)))
    _prepare_voice(quiet=args.quiet)

    try:
        if args.text:
            while True:
                try:
                    said = input("you> ").strip()
                except (EOFError, KeyboardInterrupt):
                    print()
                    return 0
                if not said:
                    continue
                if said in {"quit", "exit"}:
                    return 0
                _turn(responder, said, display, quiet=args.quiet)

        if args.push:
            transcriber = speech.whisper_transcriber()
            with tempfile.TemporaryDirectory(dir=_AUDIO_TMP) as tmp:
                wav = Path(tmp) / "turn.wav"
                while True:
                    try:
                        input("\n[enter to talk] ")
                    except (EOFError, KeyboardInterrupt):
                        print()
                        return 0
                    display.state("listening")
                    captured = speech.record_until_silence(
                        wav, on_level=lambda level: display.state("listening", level)
                    )
                    if captured is None:
                        log.status("(heard nothing)")
                        continue
                    display.state("thinking")
                    said = transcriber.transcribe(captured)
                    log.content(f"you> {said}")
                    if said and not wake.is_noise(said):
                        _turn(responder, said, display, quiet=args.quiet)

        return _listen_loop(
            responder, display, quiet=args.quiet, require_wake=False
        )
    except KeyboardInterrupt:
        print()
        return 0
    finally:
        display.close()


if __name__ == "__main__":
    raise SystemExit(main())
