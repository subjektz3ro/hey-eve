"""Her behaviour, apart from the drawing.

Everything she *does* lives here: the state she is in, the two attention
springs, the blink, the gaze, the viseme and semantic layers that move her eyes
while she talks, the idle scheduler, and the thread that turns all of it into
frames. What none of it says is what any of that looks like — that is
eve/void.py, which subclasses this and supplies `_render`.

The split earned itself. There were four faces once: an astrobot drawn as a
point cloud, an IC head in box characters, an oscilloscope, and the void. Only
one was ever fitted to the machine, and the other three were a standing tax —
every change to how she behaves had to be made, or deliberately not made, in
four places. They are gone. What survives is the half of the astrobot that was
never about the astrobot at all.

Panel geometry lives here too, because it belongs to the hardware rather than
to any drawing: a 480x320 SPI display at /dev/fb0, on VT 8.
"""
from __future__ import annotations

import math
import random
import subprocess
import threading
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from eve import idle, viseme

FB = "/dev/fb0"
FB_WIDTH, FB_HEIGHT = 480, 320
FPS = 50
# Frames per second while she is synthesising, as opposed to FPS the rest of
# the time. Kokoro is the longest wait between a person finishing a question
# and hearing an answer, and during it this thread is competing for the same
# four cores — so the face yields, which is the one moment it costs nothing to.
#
# Nothing is lost visually. The panel is a 480x320 SPI display: 307KB a frame,
# so 50fps is 123Mbit/s against a link that clocks 16-62.5MHz, and most of
# those frames are overwritten before they can be scanned out. In this state
# the head is in pieces and the only thing carrying information is the
# progress bar, which speech.py's ticker updates at 10Hz anyway.
#
# Measure before believing it: time tts.synth on one fixed sentence with the
# face at FPS, at SYNTH_FPS, and with --no-display. If the difference is under
# five percent, say so and take this out.
SYNTH_FPS = 12
VT = 8

FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
READOUT_SIZE = 20

# Per state: (eye colour, in pieces, readout)
LOOK = {
    "idle":         ((92, 232, 160), False, "READY"),
    "engaged":      ((92, 232, 160), False, "LISTENING"),
    "listening":    ((127, 227, 255), False, "LISTENING"),
    "transcribing": ((127, 227, 255), True,  "HEARD"),
    "thinking":     ((201, 166, 255), True,  "THINKING"),
    "searching":    ((201, 166, 255), True,  "SEARCHING"),
    "synthesizing": ((201, 166, 255), True,  "SPEAKING SOON"),
    "speaking":     ((255, 210, 120), False, ""),
    # Not off — asleep. The microphone has been muted long enough that there
    # is nobody to be attentive at, so she drops off rather than blanking the
    # panel: a dark screen cannot be told apart from a crash or a dead
    # backlight, and this project's failure mode is looking fine while being
    # broken. eve/void.py draws it. A dim cool blue because every other state
    # here is a working colour and this one is the absence of work.
    "asleep":       ((104, 140, 205), False, "ASLEEP"),
}
ATTENTIVE = ("listening", "engaged")

SPRING_K, SPRING_DAMP = 140.0, 15.0
_TAU = 2.0 * math.pi        # animation phases wrap here, before float32
# How many rasterised status lines to keep. See Head._label.
_LABEL_CACHE = 64

# Her eyes at rest, and how fast each part of them eases toward a target.
#
# Measured rather than guessed. A human eye is about a fifth of head width and
# character design goes larger; mine were a fifth and a half — smaller than a
# real person's — and sat *above* the midline, which reads older and severe.
# They now sit just below centre.
#
# `flick` is one shape doing the work of two: it reads as a lash, a lifted
# outer corner and a brow at once, and unlike a separate mark it can never be
# mistaken for a second pair of eyes, which is what kept looking like a spider.
# It also holds its character when the eye narrows, because it does not resize
# with the aperture.
REST = {"sep": 0.50, "rise": 0.03, "w": 0.25, "h": 0.225, "rake": 0.10,
        "flick": 0.85, "asym": 0.10, "tilt": 0.0, "arc": 0.0, "gaze": 0.0}
EASE = {"sep": 0.18, "rise": 0.055, "w": 0.07, "h": 0.07, "rake": 0.18,
        "flick": 0.09, "asym": 0.095, "tilt": 0.22, "arc": 0.10, "gaze": 0.05}


class Head:
    """Someone who pays attention, thinks in pieces, and has no mouth.

    Wear it by subclassing and supplying `_render`. Everything else — what
    state she is in, how she reacts to being spoken to, when she blinks, where
    she looks, what she does when nothing is happening, and the thread that
    runs it — is here and is the same whatever she is drawn as.
    """

    def __init__(self) -> None:
        self._state = "idle"
        self._level = 0.0
        self._note = ""
        self._progress = 0.0
        self._assembly, self._velocity = 1.0, 0.0
        self._scale, self._scale_v = 0.78, 0.0
        self._centre = 120.0
        self._hud = 1.0
        self._eyes = dict(REST)
        self._mouths: list[viseme.Mouth] = []
        self._audio_at = 0.0
        self._was = ""
        self._heard = 0.0               # smoothed voice level while attending
        self._blink_at, self._blink_next, self._blink_double = -9.0, 2.0, False
        self._gaze_to, self._gaze_next = 0.0, 1.2
        self._idle = idle.Scheduler()
        self._pose = dict(idle.NEUTRAL)
        self._labels: dict[str, np.ndarray] = {}
        self._font = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # --- what the conversation tells it ---------------------------------

    def set_state(self, state: str, level: float = 0.0, note: str = "") -> None:
        with self._lock:
            self._state = state if state in LOOK else "idle"
            self._level = max(0.0, min(1.0, level))
            self._note = note
            if state == "synthesizing":
                self._progress = max(0.0, min(1.0, level))
            if state not in ("speaking", "synthesizing"):
                self._mouths, self._audio_at = [], 0.0

    def set_speech(self, mouths: list[viseme.Mouth]) -> None:
        with self._lock:
            self._mouths = mouths

    def set_audio_position(self, seconds: float, level: float) -> None:
        with self._lock:
            self._audio_at = seconds
            self._level = max(0.0, min(1.0, level))

    def set_wave(self, samples) -> None:
        return

    # --- the eyes -------------------------------------------------------

    def _target(self, state: str, level: float, now: float) -> dict:
        eyes = dict(REST)
        if state in ATTENTIVE:
            # Perking up: opened, lifted, brows raised — and then it keeps
            # responding. Entering the state is only half of paying attention;
            # the other half is reacting when you actually start talking, which
            # is the moment a person would look up.
            eyes["h"] = 0.29 + self._heard * 0.055
            eyes["w"] = 0.36 + self._heard * 0.020
            eyes["rise"] = -0.01 - self._heard * 0.030
            eyes["flick"] = 1.05 + self._heard * 0.22
        elif state in ("thinking", "searching", "synthesizing", "transcribing"):
            eyes["h"], eyes["tilt"] = 0.10, 0.20       # narrowed, cocked
            eyes["flick"] = 0.62
        elif state == "speaking":
            beat = viseme.beat(self._mouths, self._audio_at)
            # The semantic layer. A face moves because of meaning, not
            # loudness — so each cue bends the brow and the eye before the
            # phonetic layer scales them.
            if beat is not None:
                if beat.cue == "dry":
                    eyes["flick"] += 0.42
                    eyes["asym"] = 0.30
                    eyes["h"] *= 0.72
                elif beat.cue == "hedge":
                    eyes["tilt"] = 0.16
                    eyes["flick"] += 0.22
                    eyes["rake"] += 0.12
                    eyes["asym"] = 0.26
                elif beat.cue == "negation":
                    eyes["h"] *= 0.64
                    eyes["flick"] -= 0.34
                    eyes["rake"] -= 0.08
                    eyes["tilt"] = -0.10
                elif beat.cue == "intensifier":
                    eyes["h"] *= 1.45
                    eyes["flick"] += 0.20
                elif beat.cue == "number":
                    eyes["asym"] = 0.0        # facts get no expression
                if beat.question:
                    eyes["flick"] += 0.38
                    eyes["h"] *= 1.30
                    eyes["asym"] = 0.02
                if beat.clause:
                    for key in ("h", "flick", "asym", "tilt"):
                        eyes[key] += (REST[key] - eyes[key]) * 0.66
                if beat.stop:
                    eyes.update({k: REST[k] for k in REST})
                if beat.stress:
                    eyes["flick"] += 0.05
            # The phonetic layer, underneath: the vowel squeezes the eyes and
            # every syllable bobs them, since there is no mouth to do it.
            shape, phase = viseme.at(self._mouths, self._audio_at)
            wide, tall = viseme.SHAPES[shape]
            eyes["h"] *= (1.0 - level * 0.32) * (0.80 + tall * 1.3)
            eyes["w"] *= (0.94 + wide * 0.35)
            eyes["rise"] += math.sin(phase * math.pi) * level * 0.045
        else:
            eyes["rise"] = -0.06 + math.sin(now * 0.7) * 0.008
        return eyes

    def _blink(self, now: float, level: float) -> float:
        if now > self._blink_next and level < 0.12:
            self._blink_at = now
            if self._blink_double:
                self._blink_next, self._blink_double = now + 0.30, False
            else:
                self._blink_next = now + 2.2 + random.random() * 3.6
                self._blink_double = random.random() > 0.72
        since = now - self._blink_at
        if since < 0 or since > 0.26:
            return 0.0
        if since < 0.055:
            return since / 0.055
        if since < 0.095:
            return 1.0
        return max(0.0, 1.0 - (since - 0.095) / 0.125)
    def _label(self, text: str, colour) -> np.ndarray | None:
        """Rasterise a status line once and keep it.

        Text is the only thing here that cannot be done with array maths, so
        it is drawn once per distinct label rather than once per frame.
        """
        if not text:
            return None
        cached = self._labels.get(text)
        if cached is not None:
            return cached
        if self._font is None:
            try:
                self._font = ImageFont.truetype(FONT_PATH, READOUT_SIZE)
            except OSError:
                self._font = ImageFont.load_default()
        image = Image.new("L", (FB_WIDTH, READOUT_SIZE + 8), 0)
        ImageDraw.Draw(image).text((0, 0), text, font=self._font, fill=255)
        # Bounded, oldest first. The cache is worth keeping — rasterising this
        # line costs about a third of a millisecond, on the thread with a 20ms
        # budget — but it was keyed on the label *text*, and main.py feeds it
        # a percentage and a countdown: "SPEAKING SOON 63%  ~4s". Every
        # combination is a distinct key holding a 53KB float32 array, so it
        # grew all day, every day, on a machine with no MemoryMax and an
        # intended uptime of months.
        #
        # Sixty-four is far more than the design needs: one entry per state
        # plus whatever the countdown is currently showing, and the evicted
        # ones are seconds nobody will see again.
        if len(self._labels) >= _LABEL_CACHE:
            del self._labels[next(iter(self._labels))]
        self._labels[text] = np.asarray(image, np.float32) / 255.0
        return self._labels[text]
    # --- the frame loop --------------------------------------------------

    def _render(self, now: float, dt: float) -> None:
        """Draw one frame. Supplied by whatever is wearing this."""
        raise NotImplementedError

    def _frame_period(self) -> float:
        """How long this frame is allowed to take, given what she is doing."""
        # Read without the lock deliberately: a str attribute cannot tear, and
        # the worst a stale read costs is one frame at the other rate.
        return 1.0 / (SYNTH_FPS if self._state == "synthesizing" else FPS)

    def _run(self) -> None:
        started = last = time.monotonic()
        while not self._stop.is_set():
            now = time.monotonic()
            self._render(now - started, min(0.05, now - last))
            last = now
            time.sleep(max(0.0, self._frame_period() - (time.monotonic() - now)))

    def start(self) -> None:
        if self._thread is not None:
            return
        # Move the console off the panel before drawing to /dev/fb0, or a
        # getty's cursor blinks over her face. check=False because a headless
        # box has no VT to switch to and that is not a reason to refuse to
        # start. This needs the unit to leave NoNewPrivileges unset — see the
        # note in deploy/eve@.service about why it does.
        subprocess.run(["sudo", "-n", "chvt", str(VT)], timeout=5, check=False)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        try:
            with open(FB, "r+b") as fb:
                fb.write(np.zeros(FB_WIDTH * FB_HEIGHT, np.uint16).tobytes())
        except OSError:
            pass
