"""Her face: two almond eyes drawn in box characters, and nothing else.

She has no shell. There were three other faces once — an astrobot drawn as a
point cloud, an IC head, an oscilloscope — and this is the one that stayed,
because it is the part that was doing the work anyway: the eyes, the lash and
the readout, drawn as glyphs on a 6.02 x 12 character grid.

Three decisions carry the whole look:

  * The eye is a *lid profile*, not a box. A bounding rectangle in character
    cells is the shape a terminal reaches for and it reads as a form field.
    Sampling an upper and a lower lid curve per column, with the upper fuller
    inboard and the lower dipping outboard, gives an almond instead. The
    centreline lifts toward the temple, which is the entire feminine read.

  * Glyphs sit at the curve's exact height in *pixels*, not on the cell
    lattice. Snapping to 6 x 12 quantises her sway into twelve-pixel lurches —
    she moves less than a pixel most frames, so the stutter is not that she
    moves often, it is that when she finally does she jumps a whole row.
    Free placement drops the worst step from 12 px to 1 px. The grid is a
    look, not a lattice.

  * The disturbance is the tube's, not the renderer's. Scanlines are always
    on; every few seconds bands shear sideways, the vertical hold lets go and
    snaps back, and the guns land apart. Phosphor gain is multiplicative,
    because adding a constant on a black panel produces a grey rectangle
    instead of a displaced trace.

Everything she *does* is inherited rather than reimplemented. This subclasses
head.Head and supplies only the drawing, so the easing constants, both
attention springs, the blink, the viseme and semantic layers, the chase and
the idle scheduler all live one file over. That split is what survived the
other three faces: how she behaves and what she looks like were never the same
question, and only one of them was ever face-specific.
"""
from __future__ import annotations

import math
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from eve import head, idle, touch, viseme
from eve.head import (
    ATTENTIVE, EASE, FB, FB_HEIGHT, FB_WIDTH, FONT_PATH, LOOK, SPRING_DAMP,
    SPRING_K, _TAU,
)

# The panel's real character cell: DejaVu Sans Mono at 10 px advances 6.02,
# and 12 px of leading fits 26 rows.
CELL_W, CELL_H = 6.02, 12.0
GLYPH_SIZE = 10
TILE = 7                       # pixels of ink a cell can hold, with slack

# The head these eyes are cut out of. It is never drawn — nothing here renders
# a shell — but the plate it describes is what her eyes are positioned and
# scaled against, and what her sweep arcs through.
BODY_W, BODY_H, BODY_D = 152.0, 116.0, 46.0
PLATE_W, PLATE_H = 0.908, 0.879
_PW, _PH = BODY_W * PLATE_W, BODY_H * PLATE_H
_ASPECT = _PW / _PH

# Losing the shell frees most of the panel. Drawn at the scale the plate above
# implies, the eyes occupy about a third of the width and the rest is the head
# they were cut out of — which is not drawn, so she reads as small and far
# away. This opens her out to fill the panel, leaving margin for the lash and
# for a band shearing sideways.
VOID_SCALE = 1.52             # the wedge lash is shorter, so there is room
VOID_CENTRE = 134.0            # she hangs a little above middle, over the readout
VOID_SPEAK_DROP = 0.35         # ...and only partly follows the speaking shift

# The almond. `s` runs -1 at the inner corner to +1 at the outer one.
#
# Measured against the panel, not guessed. The aperture wants about 1.6:1 —
# below 1.5 it reads as an owl, above 2.4 as a slit. The first pass was 1.62,
# already the right proportion, and it still looked like a cat glaring: the
# fault was CANTHUS_LIFT at 0.40, which is a glare rather than a lift. Chasing
# the ratio instead only made her round.
LID_POWER = 0.58               # >0.5 tapers the corners to points
LID_BOT_DEPTH = 0.86           # measured: gives a 1.58:1 aperture, an almond
LID_BIAS = 0.16                # upper fuller inboard, lower fuller outboard
CANTHUS_LIFT = 0.24            # 0.40 read as a glare rather than a lift
VOID_HEIGHT = 1.00             # 1.2+ measured owl-round; her own h is right
# The lash leaves the corner with no gap at all. The gap was the whole
# problem: a stroke that starts a cell clear of a hairline outline reads as a
# tail, because what makes a lash a lash is mass at its base. It now departs
# on the lid's own slope and is doubled for its first stretch.
LASH_RISE = 2.6                # cells out per cell up, once it has swept up
LASH_CELLS = 3.4               # shorter than the detached one, and it needs to be
LASH_WEDGE = 0.45              # the fraction of it carrying the heavy stroke

# The lid itself. The crease is what makes an eye read as an eye rather than
# as a hole, and it is what the lash now grows out of. It runs fullest over
# the outer third and stops before both corners — one that goes corner to
# corner reads as a wrinkle.
CREASE_GAP = 0.62              # how far above the lash line it sits, in apertures
CREASE_REACH = 0.86            # how far toward the corners it survives
CREASE_POWER = 0.34            # flatter than the lid it sits over
LID_FILL = True                # dither the lid skin between crease and lash line

# The pupil. Eyes do not drift, they jump: a ballistic crossing, then a
# fixation, and never perfectly still even during one. What she had was a
# sine — measured at 0.11 of the aperture across, 0.00 down, and not one
# frame moving fast enough to count as a saccade.
SACCADE_CROSS = (0.035, 0.080)  # seconds to cross
FIXATION = (0.22, 1.80)         # ...and to hold afterwards
PUPIL_REACH = 0.62             # of the half-aperture, horizontally
PUPIL_RISE = 0.50              # vertically, as a fraction of that
MICRO = 0.018                  # microsaccades: never actually still
# How often a re-aim goes to the speck rather than wherever she would have
# looked anyway. High on purpose: a thing crawling toward your face holds your
# eyes, and at 0.8 one glance in five went somewhere else, which across a
# whole encounter read as her losing track of it.
BUG_PULL = 0.96
# How far she actually travels when something is chasing her. idle.py runs the
# chase in panel-halves; these are what a half is worth in pixels here. Sized
# so that at idle.FLEE_REACH_X she has cleared the middle by most of her own
# width and is still wholly on the panel — going off the edge reads as broken
# rather than as frightened, which is the opposite of the point.
FLEE_X_PX = 150.0
FLEE_Y_PX = 132.0

# --- being touched --------------------------------------------------------
# The panel is a touchscreen, and a finger is a bug you control — so it drives
# the vocabulary the speck already built: she looks at it, her eyes go wide as
# it closes on her, and she gets out of the way. Reusing that is most of why
# this is short.
#
# What is different is the arrival. The speck wanders in over two seconds and
# leaves the same way, and everything about it fades with presence. A finger
# is simply there, between one frame and the next, which is why first contact
# gets a startle: there is no approach to be curious about, so the squint has
# nothing to do and the whole reaction is the second half of the bug's.
TOUCH_PUSH = 6.5           # harder than the speck; it is on the glass, not near
TOUCH_HOME = 3.4           # and she settles faster, because it simply stops
TOUCH_REACH = 0.60         # inside this she treats it as being on her
TOUCH_FALLOFF = 0.55
TOUCH_WIDE = 0.55          # how far the lids pull back at full alarm
TOUCH_SHAKE = 2.4
# Dilation is the cheapest expression in the face: one number, no geometry.
DILATION = {"listening": 1.5, "engaged": 1.4, "transcribing": 1.25,
            "thinking": 0.6, "searching": 0.6, "synthesizing": 0.75,
            "speaking": 1.15}

# Where she looks, by state. Until now the target was random.uniform(-1, 1)
# in every state she has, so she wandered the room while being spoken to and
# stared blankly while working. These are the three that were wrong.
CONTACT = ("listening", "engaged")     # attending to someone means looking at them
AVERTING = ("thinking", "searching")   # and thinking means breaking that off
CONTACT_SPREAD = 0.22          # small saccades, eye to mouth, not room-scanning
CONTACT_RISE = 0.29
CONTACT_HOLD = (0.55, 2.05)    # and holds three times longer than idle
# Speaking is contact too. She was wandering across two thirds of the
# aperture while answering, which is the same failure as listening that way —
# so the baseline while she talks sits near centre and only a cue sends her
# off it.
SPEAK_SPREAD = 0.30
SPEAK_HOLD = (0.30, 0.90)
AVERT_HOLD = (0.50, 1.40)
AVERT_GLANCE = 0.22            # ...with an occasional look back, which is what
                               #    makes the return to contact mean something

# The lid gives the two waiting states opposite directions of travel, which
# is what stops them being the same picture: she closes down to work, and
# opens back up as she gets ready to speak.
THINK_LID = 0.30
SYNTH_LID = 0.34
# The aperture can also fill from the bottom as synthesis runs. It is off:
# the lid opening is the cue that was asked for, and the fill on top of it
# reads as heavy at this size. Left here because it is two lines to want back.
SYNTH_FILL = False

# --- asleep ---------------------------------------------------------------
# She is not off, she is asleep, and that difference has to carry across a
# room. The first version of this blanked the panel, which was wrong for
# exactly the reason main._silence_state is careful about a fault: a dark
# screen is indistinguishable from a crash, a dead backlight or a pulled
# plug, and this project's recurring failure is looking fine while being
# broken. Being asleep is a thing she can show, so she shows it.
#
# Nothing here is tuned for cost. An earlier draft dropped her to five frames
# a second while asleep, which confuses slow *motion* with a slow *frame
# rate*: a four-second breath at fifty frames is smooth and at five is a
# slideshow, and the z's stutter visibly. She sleeps at full rate.
SLEEP_LID = 1.0                # fully shut, so the two lash curves meet
SLEEP_BREATH_S = 4.5           # a sleeping adult's rate; faster reads as panting
SLEEP_BREATH_PX = 5.5          # how far she rises and falls on it
SLEEP_DROOP = 8.0              # and she has sunk below where she hangs awake
SLEEP_TILT = 0.10              # with her head gone over to one side
# What she does while she is under is idle.SLEEP_BEHAVIOURS, scheduled by the
# same machinery as her waking idles. This used to be a single hand-rolled
# twitch here that lifted the lids 40% and moved nothing else, which on the
# real panel read as a jump scare: an eye snapping open with the rest of her
# perfectly still is a horror beat, not a sleeper. See the note above
# idle._sleep_stir for what replaced it and why the body moves instead.
#
# Waking. She was going from fully shut to fully alert between two frames,
# which is the other half of what made sleep read as a light switch. Her eyes
# come open over most of a second and then she yawns — a behaviour she has
# always had and, until now, never got to use where it obviously belongs.
WAKE_LID_S = 0.9
WAKE_YAWN_AT = 0.55            # into the eyes opening, not after it

# --- the snooze -----------------------------------------------------------
# Little z's floating off her, which is the one unambiguous way a picture has
# ever said "asleep". They are not in _GLYPHS because they are the only thing
# on this panel drawn at more than one size: a z that keeps its size reads as
# a sprite sliding sideways, and a z that grows reads as something drifting
# away from her. Growing is the whole trick, and it costs an atlas.
#
# Three in the air at once is what makes it a snooze. One is a typo and two
# is a coincidence; three staggered up the diagonal is snoring.
Z_SIZES = (12, 17, 23, 30, 38)
Z_LIFE = 4.2                   # seconds from leaving her to gone
Z_EVERY = 1.25                 # LIFE / EVERY ~ 3.4, so three or four are alive
Z_RISE = 92.0                  # pixels climbed over a lifetime
Z_DRIFT = 38.0                 # and carried to the right
Z_WOBBLE = 9.0                 # a slow sine across the climb, so none fly straight
# Offset from her centre: up and well clear of the eye. Started closer and
# smaller, which read as something stuck to her temple rather than something
# leaving her — the gap is what makes them float rather than sit.
Z_FROM = (76.0, -36.0)

# Gaze from meaning. viseme.beat() already reports whether the syllable she
# is on is a hedge, a number, a question — she has been spending it entirely
# on her brow. (dx, dy, hold, dilation); dx is signed away from centre.
CUE_GAZE = {
    "stop":        (0.00,  0.00, 0.80, 1.10),   # dead centre: the next thing is for you
    "question":    (0.10, -0.22, 0.70, 1.35),
    "number":      (0.35,  0.35, 0.90, 0.80),   # scaled toward where she was, then still
    "hedge":       (0.52, -0.42, 0.40, 0.95),
    "dry":         (0.48, -0.34, 0.45, 0.90),
    "negation":    (0.42,  0.05, 0.22, 1.00),
    "intensifier": (0.08, -0.05, 0.50, 1.70),
}

# Something in the empty panel while she works. Sparse, slow and dim on
# purpose: denser or faster and it stops being weather and becomes a
# screensaver.
RAIN_COLUMNS = 16
# Everything below is calibrated for the panel rather than for a screenshot.
# It is a cheap IPS with a lit backlight: its black is grey, so anything under
# roughly half brightness sinks into the backlight and is simply not there.
# These were all set by eye against a PNG on a good display, which is the
# wrong reference and made the columns invisible in a dark room.
RAIN_LIT = 0.74                # the tail
RAIN_HEAD = 1.05               # ...and the character leading it, always lit
RAIN_TURN = 7.0                # glyph changes a second, so a column shimmers
CREASE_LIT = 0.78              # the fold above the lash line
SKIN_LIT = 0.55                # the dithered lid between crease and lash line
LOWER_LIT = 0.82               # the lower lid, still lighter than the upper
FILL_LIT = 0.60                # the aperture fill, when it is switched on

SAMPLES = 30                   # points per lid curve
DIAGONAL = 0.52                # rows per cell before a rule becomes a stroke
UPRIGHT = 1.7                  # ...and before it becomes a bar
MET = 0.62                     # cells apart below which the lids count as met

# The tube. Displacements run large on purpose: the trace is a
# repeating pattern, so shifting a band three pixels sideways produces an
# image identical to the one before it.
SCANLINE = 0.90
GLITCH_CALM = (5.0, 14.0)
GLITCH_UNSTABLE = (2.5, 7.0)   # while she is in pieces she comes apart more
GLITCH_LENGTH = (0.10, 0.28)
GLITCH_BAND = 22
GLITCH_GAIN = 1.25
GLITCH_SNAP = 0.04             # the hold catches again this long before the end
CHROMA = 2

READOUT_TOP = 264
RAIL_Y, RAIL_H = 302, 3

_GLYPHS = "─│╱╲█·░▒━" + "01<>[]{}/|=+*:\\"
G_H, G_V, G_UP, G_DN, G_PUP, G_DOT, G_LIGHT, G_MID, G_HEAVY = range(9)
# The rain glyphs: deliberately mixed symbols
# rather than letters, because words in the background pull the eye off the
# face. Fifteen of them, so a column turning over never obviously repeats.
RAIN_FIRST, RAIN_GLYPHS = 9, len(_GLYPHS) - 9


def _masks() -> list[np.ndarray]:
    """Each glyph as a float alpha tile, rasterised once at import.

    Text is the only thing here that cannot be done with array maths, so it
    happens once rather than once per frame; everything after this is a
    maximum into a slice.
    """
    try:
        font = ImageFont.truetype(FONT_PATH, GLYPH_SIZE)
    except OSError:                                  # a box without DejaVu
        font = ImageFont.load_default()
    out = []
    for char in _GLYPHS:
        image = Image.new("L", (TILE, int(CELL_H)), 0)
        ImageDraw.Draw(image).text((0, 1), char, font=font, fill=255)
        out.append(np.asarray(image, np.float32) / 255.0)
    return out


MASKS = _masks()


def _z_masks() -> list[np.ndarray]:
    """A lowercase z at each size it will be seen at, rasterised once.

    Its own atlas rather than an entry in _GLYPHS, because every other glyph
    on this panel is drawn at one size on one grid and these are the
    exception — see the note above Z_SIZES for why growing is the point.
    """
    out = []
    for size in Z_SIZES:
        try:
            font = ImageFont.truetype(FONT_PATH, size)
        except OSError:                              # a box without DejaVu
            font = ImageFont.load_default()
        box = int(size * 1.7) + 2
        image = Image.new("L", (box, box), 0)
        ImageDraw.Draw(image).text((1, 0), "z", font=font, fill=255)
        out.append(np.asarray(image, np.float32) / 255.0)
    return out


ZS = _z_masks()


def _smooth(t: float) -> float:
    """Smoothstep, clamped. idle has its own; this is the only use here."""
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def _lid_top(s: float) -> float:
    return max(0.0, 1.0 - s * s) ** LID_POWER * (1.0 - LID_BIAS * s)


def _lid_bot(s: float) -> float:
    return (LID_BOT_DEPTH * max(0.0, 1.0 - s * s) ** LID_POWER
            * (1.0 + LID_BIAS * 1.1 * s))


def _centreline(s: float) -> float:
    """Where the eye's midline sits: higher at the temple than at the nose."""
    return -CANTHUS_LIFT * (0.55 * s + 0.45 * s * abs(s))


def _crease(s: float) -> float:
    """How far the crease clears the lash line, as a fraction of the aperture."""
    return CREASE_GAP * max(0.0, 1.0 - s * s) ** CREASE_POWER


class Void(head.Head):
    """Her behaviour, drawn as characters and nothing else."""

    def __init__(self) -> None:
        super().__init__()
        self._scan = np.full((FB_HEIGHT, 1, 1), 256, dtype=np.uint16)
        self._scan[1::2] = int(256 * SCANLINE)
        self._tears: list[tuple[int, int, int]] = []
        self._drops: list[int] = []
        self._slip = self._jitter = 0
        self._glitch_until = -1.0
        self._glitch_next = random.uniform(*GLITCH_CALM)
        self._pup = {"x": 0.0, "y": 0.0, "fx": 0.0, "fy": 0.0,
                     "tx": 0.0, "ty": 0.0, "t0": -9.0, "dur": 0.06,
                     "next": 0.0, "side": 1.0}
        # Asleep: the z's in the air, and when she next jumps in her sleep.
        # `_z_next` is set on the frame she falls asleep rather than here, so
        # the first z leaves her a beat after her eyes close instead of being
        # already airborne the moment the state changes.
        self._zs: list[tuple[float, float]] = []
        self._z_next = 0.0
        # When she started waking, while her eyes are still on their way open,
        # and whether the yawn that goes with it is still owed.
        self._woke_at: float | None = None
        self._wake_yawn = False
        # The glass, and where she has got to away from whatever is on it.
        # Kept separate from the pose's dx/dy rather than folded into it: the
        # speck and a finger can both be happening, and the two displacements
        # simply add.
        self._touch = touch.Touch()
        self._touched = False
        self._flee = (0.0, 0.0)
        self._flee_v = (0.0, 0.0)
        self._rain = [
            {"col": (i * 11 + 3) % int(FB_WIDTH / CELL_W),
             "phase": random.uniform(0.0, 40.0),
             "speed": random.uniform(5.0, 13.0),
             "length": random.randint(5, 13)}
            for i in range(RAIN_COLUMNS)
        ]

    # --- geometry -------------------------------------------------------

    def _project(self, u: float, v: float, cam: dict) -> tuple[float, float]:
        """One point on the visor, in panel pixels."""
        x, y = u * _PW, v * _PH
        z = -BODY_D * math.sqrt(max(0.0, 1.0 - 0.55 * min(1.0, u * u + v * v))) + 3.0
        y += cam["bob"]
        y2 = y * cam["ca"] - z * cam["sa"]
        z2 = y * cam["sa"] + z * cam["ca"]
        x2 = x * cam["cb"] + z2 * cam["sb"]
        z2 = -x * cam["sb"] + z2 * cam["cb"]
        k = 560.0 / (560.0 + z2) * cam["scale"] * VOID_SCALE
        return (FB_WIDTH / 2 + x2 * k * 0.92 + cam["jx"],
                cam["centre"] + y2 * k * 0.92 + cam["jy"])

    def _outline(self, side: int, descent: float, cam: dict
                 ) -> tuple[list, list, list]:
        """The lash line, the lower lid and the crease, in panel pixels.

        `descent` is the axis the lid buys: how far the upper lid has come
        down, entirely separate from how open the aperture is. Before this
        the two were one parameter, so she could squint but never look
        heavy-lidded — and a blink squeezed the eye shut from above and below
        at once, which is not what an eye does.
        """
        eyes = self._eyes
        tall = eyes["h"] * VOID_HEIGHT
        h_top = max(0.012, tall)
        h_bot = max(0.012, tall * 0.94)
        cu = side * eyes["sep"] + eyes["gaze"] * 0.30
        cv = eyes["rise"]
        top, bot, crease = [], [], []
        for i in range(SAMPLES):
            s = -1.0 + 2.0 * i / (SAMPLES - 1)
            shear = -eyes["rake"] * s * eyes["w"] * _ASPECT
            du = side * s * eyes["w"]
            mid = _centreline(s) * eyes["h"] + shear
            v_top = cv + mid - _lid_top(s) * h_top
            v_bot = cv + mid + _lid_bot(s) * h_bot
            v_crease = v_top - _crease(s) * h_top
            # The lid descends onto a stationary lower lid, and the crease
            # compresses ahead of it rather than riding down untouched.
            v_top += (v_bot - v_top) * descent
            v_crease += (v_bot - v_crease) * descent * 0.86
            top.append(self._project(cu + du, v_top, cam))
            bot.append(self._project(cu + du, v_bot, cam))
            crease.append(self._project(cu + du, v_crease, cam)
                          if abs(s) < CREASE_REACH else None)
        # s runs inner to outer, which for the left eye is right to left
        # across the panel. Everything downstream interpolates on rising x.
        if top[0][0] > top[-1][0]:
            top.reverse()
            bot.reverse()
            crease.reverse()
        return top, bot, crease

    @staticmethod
    def _height_at(curve: list, x: float) -> float:
        if x <= curve[0][0]:
            return curve[0][1]
        if x >= curve[-1][0]:
            return curve[-1][1]
        for i in range(1, len(curve)):
            if x <= curve[i][0]:
                a, b = curve[i - 1], curve[i]
                span = b[0] - a[0]
                f = (x - a[0]) / span if span > 1e-6 else 0.0
                return a[1] + (b[1] - a[1]) * f
        return curve[-1][1]

    # --- drawing --------------------------------------------------------

    def _put(self, frame: np.ndarray, glyph: int, x: float, y: float,
             colour: np.ndarray) -> None:
        self._blit(frame, MASKS[glyph], x, y, colour)

    def _blit(self, frame: np.ndarray, mask: np.ndarray, x: float, y: float,
              colour: np.ndarray) -> None:
        """One alpha tile, maximum'd into the frame at a pixel position.

        Split out of _put so the snooze can hand over a tile from its own
        atlas: everything else on this panel comes from MASKS at one size,
        and the z's are the exception rather than a reason to widen MASKS.
        """
        x0, y0 = int(round(x)), int(round(y))
        h, w = mask.shape
        sx0, sy0 = max(0, -x0), max(0, -y0)
        x0, y0 = max(0, x0), max(0, y0)
        w = min(w - sx0, FB_WIDTH - x0)
        h = min(h - sy0, FB_HEIGHT - y0)
        if w <= 0 or h <= 0:
            return
        band = frame[y0:y0 + h, x0:x0 + w]
        tile = (mask[sy0:sy0 + h, sx0:sx0 + w, None] * colour).astype(np.uint16)
        np.maximum(band, tile, out=band)

    def _glyph(self, curve: list, x: float, y: float) -> int:
        """Let the local slope pick the character.

        Placement is already at the curve's exact height, so a glyph only
        turns diagonal where it genuinely climbs faster than half a row per
        cell. Below that a rule sitting a few pixels lower reads as a curve;
        above it, a rule reads as a broken line.
        """
        rise = self._height_at(curve, x + CELL_W) - y
        if abs(rise) > CELL_H * UPRIGHT:
            return G_V
        if rise < -CELL_H * DIAGONAL:
            return G_UP
        if rise > CELL_H * DIAGONAL:
            return G_DN
        return G_H

    # --- being touched --------------------------------------------------

    def start(self) -> None:
        super().start()
        self._touch.start()

    def stop(self) -> None:
        self._touch.stop()
        super().stop()

    def _feel(self, now: float, dt: float) -> tuple[float, float, float]:
        """React to a finger on the glass.

        Returns how alarmed she is and which way the finger lies from her, and
        moves her away from it as a side effect — the same integrator the bug
        runs on, because they are the same problem.

        The touch reader reports -1..1 across the panel; her displacement is
        in the units idle.py runs the chase in. The conversion is the panel's
        half-size over the pixels-per-unit, and getting it wrong is exactly
        the bug that had the speck drawn on her face while every number said
        it was clear, so it is spelled out rather than approximated.
        """
        finger_x, finger_y, down = self._touch.latest()
        if down and not self._touched:
            # She jumps at first contact, before working out what it is. The
            # same reaction a bang in the room already gets, and the reason
            # there is no curiosity stage here: a finger does not approach.
            self._idle.startle(now)
        self._touched = down

        finger = None
        push = near = 0.0
        if down:
            finger = (finger_x * (FB_WIDTH / 2) / FLEE_X_PX,
                      finger_y * (FB_HEIGHT / 2) / FLEE_Y_PX)
            gap = math.hypot(self._flee[0] - finger[0],
                             self._flee[1] - finger[1])
            beyond = max(0.0, gap - TOUCH_REACH)
            near = math.exp(-(beyond * beyond) / TOUCH_FALLOFF)
            push = TOUCH_PUSH * near

        self._flee, self._flee_v = idle.flee_step(
            self._flee, self._flee_v, finger, push, dt, TOUCH_HOME)

        if not down or finger is None:
            return 0.0, 0.0, 0.0
        return (
            _smooth((near - 0.30) / 0.40),
            max(-1.0, min(1.0, finger[0] - self._flee[0])),
            max(-1.0, min(1.0, finger[1] - self._flee[1])),
        )

    def _bug(self, frame: np.ndarray, x: float, y: float,
             colour: np.ndarray, now: float) -> None:
        """The speck, drawn as something with legs.

        It used to be one G_DOT — a full stop in a seven pixel cell — and from
        a few feet away that is a stuck pixel rather than a creature. Two body
        cells and four legs on her own character grid make it about thirty
        pixels across, and swapping the leg glyphs a dozen times a second is
        what actually sells it: the motion does more work here than the size,
        because a thing that scuttles is alive and a thing that slides is a
        cursor.
        """
        step = int(now * 13) % 2
        legs = colour * 0.7
        self._put(frame, G_PUP, x - CELL_W * 0.55, y, colour)
        self._put(frame, G_PUP, x + CELL_W * 0.55, y, colour)
        for side in (-1.0, 1.0):
            lead = (step == 0) == (side < 0)
            self._put(frame, G_UP if lead else G_DN,
                      x + side * CELL_W * 2.0, y - CELL_H * 0.62, legs)
            self._put(frame, G_DN if lead else G_UP,
                      x + side * CELL_W * 2.0, y + CELL_H * 0.62, legs)

    # --- asleep ---------------------------------------------------------

    def _snooze(self, frame: np.ndarray, now: float, centre: float,
                colour: np.ndarray, emit: bool = True) -> None:
        """The z's, floating off her and dissolving.

        Emitted on a stagger rather than a metronome so the line of them up
        the diagonal never looks like a ruler, and each carries a phase so no
        two wobble in step. They grow with age and fade at both ends: in fast
        enough not to pop, out slowly enough to dissolve rather than stop.

        `emit` goes false while she is waking, so the last few finish their
        climb and dissolve on their own rather than vanishing between two
        frames the instant her eyes start to open.
        """
        while self._zs and now - self._zs[0][0] > Z_LIFE:
            self._zs.pop(0)
        if emit and now >= self._z_next:
            self._z_next = now + Z_EVERY * random.uniform(0.82, 1.18)
            self._zs.append((now, random.uniform(0.0, _TAU)))

        ox = FB_WIDTH / 2 + Z_FROM[0]
        oy = centre + Z_FROM[1]
        for born, phase in self._zs:
            age = (now - born) / Z_LIFE
            if not 0.0 <= age <= 1.0:
                continue
            alpha = min(1.0, age * 9.0) * max(0.0, 1.0 - age) ** 0.85
            if alpha <= 0.01:
                continue
            mask = ZS[min(len(ZS) - 1, int(age * len(ZS)))]
            self._blit(
                frame, mask,
                ox + Z_DRIFT * age + math.sin(age * 3.4 + phase) * Z_WOBBLE,
                oy - Z_RISE * age,
                colour * alpha,
            )

    def _lids(self, frame: np.ndarray, top: list, bot: list, crease: list,
              colour: np.ndarray, fill: float = 0.0) -> None:
        """Walk both lid curves together, a column at a time.

        Together rather than separately because at the corners they converge,
        and two strokes a few pixels apart land as a visible X — the one
        artefact here that reads as a mistake rather than as texture. Where
        the lids have already met, one glyph is drawn instead of two, which
        also gives the blink its arc for nothing.
        """
        x = top[0][0]
        end = top[-1][0]
        while x <= end + 0.1:
            high = self._height_at(top, x)
            low = self._height_at(bot, x)
            if low - high < CELL_H * MET:
                self._put(frame, self._glyph(top, x, high),
                          x, (high + low) * 0.5 - CELL_H * 0.5, colour)
            else:
                self._put(frame, self._glyph(top, x, high),
                          x, high - CELL_H * 0.5, colour)
                # The upper line carries more weight than the lower. On its
                # own that is most of what says "lashes" at this resolution.
                self._put(frame, self._glyph(bot, x, low),
                          x, low - CELL_H * 0.5, colour * LOWER_LIT)
                # ...and while she is getting ready to speak, the aperture
                # fills from the bottom. On its own it barely registers,
                # because a working eye is narrowed and there is nothing to
                # fill — it only reads underneath the lid opening.
                if fill > 0.0:
                    floor = low - CELL_H * 0.30
                    line = floor - (floor - (high + CELL_H * 0.40)) * fill
                    level = floor
                    while level > line:
                        self._put(frame, G_MID, x, level - CELL_H * 0.5,
                                  colour * FILL_LIT)
                        level -= CELL_H
            x += CELL_W

        seen = [point for point in crease if point is not None]
        if len(seen) < 3:
            return
        x, end = seen[0][0], seen[-1][0]
        while x <= end + 0.1:
            fold = self._height_at(seen, x)
            lash_line = self._height_at(top, x)
            if lash_line - fold < CELL_H * 0.5:
                x += CELL_W                      # the crease has met the lid
                continue
            self._put(frame, self._glyph(seen, x, fold),
                      x, fold - CELL_H * 0.5, colour * CREASE_LIT)
            if LID_FILL:
                skin = fold + CELL_H
                while skin < lash_line - CELL_H * 0.35:
                    self._put(frame,
                              G_LIGHT if int(skin / CELL_H) % 2 else G_MID,
                              x, skin - CELL_H * 0.5, colour * SKIN_LIT)
                    skin += CELL_H
            x += CELL_W

    def _advance_pupil(self, now: float, state: str) -> None:
        """Saccades, fixations, and a speck worth watching.

        The crossing is ballistic and short; almost all of the time is spent
        holding, with a tremor on top because an eye that is genuinely still
        looks painted on.
        """
        pup = self._pup
        if now > pup["next"]:
            pup["fx"], pup["fy"], pup["t0"] = pup["x"], pup["y"], now
            pup["dur"] = random.uniform(*SACCADE_CROSS)
            bug = self._idle.bug_at(now)
            beat = (viseme.beat(self._mouths, self._audio_at)
                    if state == "speaking" else None)
            cue = None
            if beat is not None:
                cue = ("stop" if beat.stop else
                       "question" if beat.question else beat.cue)
            if bug is not None and random.random() < BUG_PULL:
                bx, by, presence = bug
                # Where it is *relative to her*, not where it is on the panel.
                # Absolute was right while she never moved; now that she runs,
                # it aimed her eyes at a fixed point and she watched the place
                # the thing used to be while it circled her.
                dx = bx - self._pose["dx"]
                dy = by - self._pose["dy"]
                pup["tx"] = max(-1.0, min(1.0, dx * 1.5)) * presence
                pup["ty"] = max(-1.0, min(1.0, dy * 1.7)) * presence
                # And re-aimed constantly rather than at the next saccade. A
                # fixation of up to half a second is right for glancing at
                # something; for keeping your eyes on a thing that is moving
                # it is most of the encounter spent looking somewhere else.
                pup["next"] = now + pup["dur"] + random.uniform(0.05, 0.14)
            elif state in CONTACT:
                # Small saccades between eye and mouth level — the triangle
                # people actually scan on a face — instead of the room.
                pup["tx"] = random.uniform(-1.0, 1.0) * CONTACT_SPREAD
                pup["ty"] = -CONTACT_RISE if random.random() < 0.5 else CONTACT_RISE
                pup["next"] = now + pup["dur"] + random.uniform(*CONTACT_HOLD)
            elif state in AVERTING:
                if random.random() < AVERT_GLANCE:
                    pup["tx"] = random.uniform(-0.15, 0.15)
                    pup["ty"] = 0.0
                    pup["next"] = now + pup["dur"] + 0.35
                else:
                    pup["tx"] = random.uniform(0.55, 1.0) * pup["side"]
                    pup["ty"] = -random.uniform(0.55, 1.0)
                    if random.random() < 0.25:
                        pup["side"] = -pup["side"]
                    pup["next"] = now + pup["dur"] + random.uniform(*AVERT_HOLD)
            elif cue in CUE_GAZE:
                dx, dy, hold, _ = CUE_GAZE[cue]
                if cue == "number":              # settle where she is, and hold
                    pup["tx"], pup["ty"] = pup["x"] * dx, pup["y"] * dy
                else:
                    pup["tx"] = dx * (-1.0 if pup["x"] > 0 else 1.0) \
                        if cue == "negation" else dx * random.choice((-1.0, 1.0))
                    pup["ty"] = dy
                pup["next"] = now + pup["dur"] + hold
            elif state == "speaking":
                pup["tx"] = random.uniform(-1.0, 1.0) * SPEAK_SPREAD
                pup["ty"] = random.uniform(-1.0, 1.0) * SPEAK_SPREAD * 0.6
                pup["next"] = now + pup["dur"] + random.uniform(*SPEAK_HOLD)
            else:
                pup["tx"] = random.uniform(-1.0, 1.0)
                pup["ty"] = random.uniform(-1.0, 1.0) * PUPIL_RISE
                pup["next"] = now + pup["dur"] + random.uniform(*FIXATION)
        travel = min(1.0, (now - pup["t0"]) / max(1e-6, pup["dur"]))
        eased = 1.0 - (1.0 - travel) ** 3           # fast away, settling in
        pup["x"] = pup["fx"] + (pup["tx"] - pup["fx"]) * eased
        pup["y"] = pup["fy"] + (pup["ty"] - pup["fy"]) * eased
        pup["x"] += math.sin(now * 31.7) * MICRO + math.sin(now * 11.3) * MICRO * 0.6
        pup["y"] += math.sin(now * 27.1) * MICRO * 0.7

    def _pupil(self, frame: np.ndarray, top: list, bot: list,
               state: str, colour: np.ndarray) -> None:
        """A block that goes where she is looking, sized by how she feels."""
        x0, x1 = top[0][0], top[-1][0]
        half = (x1 - x0) * 0.5
        cx = (x0 + x1) * 0.5 + self._pup["x"] * half * PUPIL_REACH
        open_at = max(0.0, self._height_at(bot, cx) - self._height_at(top, cx))
        cy = ((self._height_at(top, cx) + self._height_at(bot, cx)) * 0.5
              + self._pup["y"] * open_at * 0.30)
        wide = DILATION.get(state, 1.0)
        if state == "speaking":
            beat = viseme.beat(self._mouths, self._audio_at)
            if beat is not None:
                cue = ("stop" if beat.stop else
                       "question" if beat.question else beat.cue)
                if cue in CUE_GAZE:
                    wide = CUE_GAZE[cue][3]
        elif state == "synthesizing":
            wide = 0.75 + self._progress * 0.75      # widening as she gets ready
        cols = max(1, int(round(1.5 * wide)))
        for dc in range(-cols, cols + 1):
            for dr in (-0.5, 0.5):
                px = cx + dc * CELL_W * 0.62
                py = cy + dr * CELL_H
                lo = self._height_at(top, px) + CELL_H * 0.30
                hi = self._height_at(bot, px) - CELL_H * 0.30
                if hi - lo > CELL_H * 0.4 and lo <= py <= hi:
                    self._put(frame, G_PUP, px, py - CELL_H * 0.5, colour)

    def _eye(self, frame: np.ndarray, side: int, descent: float, cam: dict,
             colour: np.ndarray, state: str, fill: float = 0.0) -> None:
        top, bot, crease = self._outline(side, descent, cam)
        self._lids(frame, top, bot, crease, colour, fill)
        self._pupil(frame, top, bot, state, colour)

        # The lash, leaving the corner with no gap and on the lid's own
        # slope, doubled for its first stretch and tapering after. The gap
        # and the constant width were what made the old one read as a tail:
        # what makes a lash a lash is mass where it meets the lid.
        tip = top[0] if side < 0 else top[-1]
        behind = top[1] if side < 0 else top[-2]
        run = max(1e-6, abs(tip[0] - behind[0]))
        slope = (tip[1] - behind[1]) / run * side * CELL_W
        reach = max(0.10, self._eyes["flick"]
                    + (self._eyes["asym"] if side < 0 else 0.0))
        count = max(2, int(round(reach * LASH_CELLS)))
        for i in range(1, count + 1):
            along = i / count
            climb = slope * (1.0 - along) * 0.55 + (CELL_H / LASH_RISE) * along
            lx = tip[0] + side * i * CELL_W
            ly = tip[1] - CELL_H * 0.5 - i * climb
            glyph = (G_DN if side < 0 else G_UP) if i == count else G_H
            self._put(frame, glyph, lx, ly, colour * (1.0 - along * 0.35))
            if along <= LASH_WEDGE:
                self._put(frame, G_HEAVY, lx, ly, colour * 0.5 * (1.0 - along))

    def _columns(self, frame: np.ndarray, now: float,
                 colour: np.ndarray) -> None:
        """Something in the empty panel while she works.

        Drawn before her, so it stays behind. Twelve columns rather than the
        twenty-six the effect wants: denser or faster and it stops reading as
        weather and starts reading as a screensaver.
        """
        rows = int(FB_HEIGHT / CELL_H)
        for column in self._rain:
            span = rows + column["length"] + 6
            head = (now * column["speed"] + column["phase"]) % span - column["length"]
            for step in range(column["length"]):
                row = int(head - step)
                if not 0 <= row < rows - 4:
                    continue
                # The leading character is always lit and the tail falls away
                # behind it. Shade blocks were the mistake before this: they
                # are mostly empty pixels, so they sank into the backlight
                # however high the multiplier went.
                fade = (RAIN_HEAD if step == 0
                        else RAIN_LIT * (1.0 - step / column["length"]) ** 0.7)
                # Which glyph turns over as the column falls, so it shimmers
                # rather than scrolling a fixed string down the panel.
                pick = int(row * 7 + column["col"] * 13
                           + now * RAIN_TURN) % RAIN_GLYPHS
                self._put(frame, RAIN_FIRST + pick,
                          column["col"] * CELL_W, row * CELL_H, colour * fade)

    # --- the tube -------------------------------------------------------

    def _disturb(self, frame: np.ndarray, now: float, unstable: bool
                 ) -> np.ndarray:
        """Scanlines always, and every few seconds the hold lets go.

        Bands shear sideways and their phosphor brightens multiplicatively;
        adding a constant was the giveaway in the first version, because this
        panel's background is pure black and a band gained a visible grey
        rectangle instead of a displaced trace.
        """
        frame = (frame * self._scan) >> 8
        if now > self._glitch_until:
            if now < self._glitch_next:
                return frame                        # the quiet, usual case
            self._glitch_until = now + random.uniform(*GLITCH_LENGTH)
            calm = GLITCH_UNSTABLE if unstable else GLITCH_CALM
            self._glitch_next = self._glitch_until + random.uniform(*calm)
            self._slip = random.choice((-1, 1)) * random.randint(3, 14)
            self._jitter = random.choice((-1, 1)) * random.randint(4, 12)
            self._tears = [
                (random.randrange(0, FB_HEIGHT - GLITCH_BAND),
                 random.randint(4, GLITCH_BAND),
                 random.choice((-1, 1)) * random.choice((13, 21, 34, 55, 89)))
                for _ in range(random.randint(2, 4))
            ]
            self._drops = [random.randrange(0, FB_HEIGHT - 2)
                           for _ in range(random.randint(0, 2))]

        frame = np.roll(frame, self._jitter + random.randint(-2, 2), axis=1)
        if now < self._glitch_until - GLITCH_SNAP:
            frame = np.roll(frame, self._slip, axis=0)
        for top, height, shift in self._tears:
            band = frame[top:top + height]
            frame[top:top + height] = np.minimum(
                np.roll(band, shift, axis=1).astype(np.uint32) * GLITCH_GAIN,
                255).astype(np.uint16)
        for row in self._drops:
            frame[row:row + 2] = 0
        # The three guns land in slightly different places.
        frame[..., 0] = np.roll(frame[..., 0], CHROMA, axis=1)
        frame[..., 2] = np.roll(frame[..., 2], -CHROMA, axis=1)
        return frame

    # --- the frame ------------------------------------------------------

    def _render(self, now: float, dt: float) -> None:
        with self._lock:
            state, level = self._state, self._level
            note, progress = self._note, self._progress
        colour_rgb, in_pieces, label = LOOK[state]
        speaking = state == "speaking"
        attentive = state in ATTENTIVE
        asleep = state == "asleep"
        if asleep and self._was != "asleep":
            # She is dropping off on this frame. Nothing in the air yet: the
            # first z should leave her a beat after her eyes close rather
            # than being already airborne the instant the state changed.
            self._zs.clear()
            self._z_next = now + 1.1
            self._woke_at, self._wake_yawn = None, False
        elif self._was == "asleep" and not asleep:
            # And waking on this one. Both halves are queued rather than done
            # here: the lid ramp is applied after descent exists, and the yawn
            # has to be handed to the scheduler *after* its own update call,
            # which drops whatever it holds on the frame the mode changes.
            self._woke_at, self._wake_yawn = now, True

        # Assembly, and the two attention springs. A snap reads as attention
        # and a fade reads as a zoom, and the impulses are what make the
        # difference.
        want = 0.0 if in_pieces else 1.0
        self._velocity += ((want - self._assembly) * SPRING_K
                           - self._velocity * SPRING_DAMP) * dt
        self._assembly += self._velocity * dt

        if attentive and self._was not in ATTENTIVE:
            self._scale_v += 1.9
        if attentive:
            rising = level - self._heard
            self._heard += (level - self._heard) * (1 - math.exp(-dt / (
                0.06 if level > self._heard else 0.55)))
            if rising > 0.22 and self._heard < 0.30:
                self._scale_v += 1.1
        else:
            if state == "idle" and level > 0.55 and self._heard < 0.25:
                self._idle.startle(now)
            self._heard += (level - self._heard) * (1 - math.exp(-dt / 0.30))
        self._was = state

        scale_to = (1.06 if speaking
                    else (0.84 + self._heard * 0.06) if attentive else 0.78)
        self._scale_v += ((scale_to - self._scale) * 150.0
                          - self._scale_v * 14.0) * dt
        self._scale += self._scale_v * dt
        self._centre += ((150.0 if speaking else 116.0) - self._centre) * (
            1 - math.exp(-dt / 0.22))
        self._hud += ((0.0 if speaking else 1.0) - self._hud) * (
            1 - math.exp(-dt / 0.18))

        self._pose = self._idle.update(now, state in ("idle", "engaged"),
                                       asleep=asleep)
        if self._wake_yawn:
            # Queued a moment into the eyes opening rather than at the same
            # instant, which is what stops it reading as a grimace. play()
            # takes a time in the future and every pose function clamps, so
            # it simply holds neutral until it arrives.
            self._wake_yawn = False
            self._idle.play(idle.YAWN, now + WAKE_LID_S * WAKE_YAWN_AT)

        wanted = self._target(state, level, now)
        for key, rate in EASE.items():
            if key == "gaze":
                continue
            self._eyes[key] += (wanted[key] - self._eyes[key]) * (
                1 - math.exp(-dt / rate))
        if state == "synthesizing":
            self._eyes["h"] = max(self._eyes["h"], 0.10 + 0.17 * progress)
        if now > self._gaze_next:
            self._gaze_to = (random.random() * 2 - 1) * 0.05
            self._gaze_next = now + 0.7 + random.random() * 2.6
        self._eyes["gaze"] += (self._gaze_to - self._eyes["gaze"]) * (
            1 - math.exp(-dt / EASE["gaze"]))
        self._eyes["gaze"] += self._pose["gaze"]
        self._eyes["rise"] += self._pose["rise"]
        self._eyes["w"] *= self._pose["wide"]
        self._eyes["flick"] += self._pose["flick"]

        # And whatever is on the glass. Read every frame whether or not
        # anything is touching, because the integrator has to keep walking her
        # home after a finger lifts.
        felt, toward_x, toward_y = self._feel(now, dt)
        if felt > 0.0:
            self._eyes["gaze"] += toward_x * 0.09
            self._eyes["rise"] += toward_y * 0.03
            self._eyes["flick"] += 0.55 * felt

        shut = self._blink(now, level)
        self._advance_pupil(now, state)
        # The idle poses speak in lid *openness*, where 1.0 is neutral, under
        # 1.0 is a lid coming down and over 1.0 is one pulled back in
        # surprise. Descent is the inverse of that with the blink added on
        # top, so a wink is one lid descending and a startle is both
        # retracting — which is what those poses always meant and could not
        # previously say.
        # The two waiting states travel in opposite directions: she closes
        # down to work and opens back up as she gets ready to speak. That is
        # what stops thinking and synthesizing being the same picture, which
        # is what they were — one branch of _target served both.
        working = 0.0
        fill = 0.0
        if state in AVERTING:
            working = THINK_LID
        elif state == "synthesizing":
            working = SYNTH_LID * (1.0 - progress)
            fill = progress if SYNTH_FILL else 0.0
        descent = (
            max(-0.28, min(1.0,
                (1.0 - self._pose["lid_l"]) + shut + working)),
            max(-0.28, min(1.0,
                (1.0 - self._pose["lid_r"]) + shut + working)),
        )
        lean = 1.0 if attentive else 0.0
        spin = (math.sin(now * 0.20) * 0.26 * (1 - lean * 0.7)
                + self._pose["roll"])
        tilt = ((0.06 + math.sin(now * 0.13) * 0.05) * (1 - lean * 0.7)
                + self._pose["tilt"] + self._eyes["tilt"] * 0.35)
        if felt > 0.0:
            # Wide, not narrowed. Being touched skips the curious half of the
            # speck's arc entirely — there is no approach to squint at.
            descent = (descent[0] - TOUCH_WIDE * felt,
                       descent[1] - TOUCH_WIDE * felt)
        shake = max(self._pose["shake"], TOUCH_SHAKE * felt)
        bob = math.sin(now * 1.05) * 4.0 - lean * 8.0
        scale = self._scale / 0.86 * self._pose["scale"]

        if asleep:
            # Everything computed above describes somebody awake. Overridden
            # here rather than branched around up there, so the states she
            # can be in stay one code path and sleep reads as what it is: a
            # pose laid over the top of her ordinary self.
            #
            # The lids compose with the pose rather than replacing it. Poses
            # speak in openness — 1.0 neutral, over 1.0 a lid pulled back —
            # so a sleep behaviour cracks her by going above one, and the
            # most any of them asks for is six hundredths. The version this
            # replaced pinned descent flat and lifted it four tenths, which
            # is an eye snapping open with nothing else moving.
            descent = (SLEEP_LID - max(0.0, self._pose["lid_l"] - 1.0),
                       SLEEP_LID - max(0.0, self._pose["lid_r"] - 1.0))
            breath = math.sin(now * _TAU / SLEEP_BREATH_S)
            bob = SLEEP_DROOP + breath * SLEEP_BREATH_PX
            scale *= 1.0 + breath * 0.010
            # Her own sway all but stops; what moves her now is whatever the
            # sleep behaviour is doing, which arrives through the pose.
            spin = math.sin(now * 0.07) * 0.09 + self._pose["roll"]
            tilt = (SLEEP_TILT + math.sin(now * 0.05) * 0.03
                    + self._pose["tilt"])
        elif self._woke_at is not None:
            since = now - self._woke_at
            if since >= WAKE_LID_S:
                self._woke_at = None
            else:
                # Her eyes come open rather than cutting. She was asleep a
                # moment ago, and going from shut to fully alert between two
                # frames is what made waking read as nothing having happened.
                held = SLEEP_LID * (1.0 - _smooth(since / WAKE_LID_S))
                descent = (max(descent[0], held), max(descent[1], held))

        cam = {
            "ca": math.cos(tilt), "sa": math.sin(tilt),
            "cb": math.cos(spin), "sb": math.sin(spin),
            "bob": bob,
            "scale": scale,
            "centre": VOID_CENTRE + (self._centre - 116.0) * VOID_SPEAK_DROP,
            # Jitter, plus wherever she has run to. _project adds both to the
            # projected position of every point, so this moves all of her
            # rather than any part of her — which is what nothing had ever
            # used it for until the bug learned to give chase.
            "jx": (random.uniform(-shake, shake) if shake else 0.0)
                  + (self._pose["dx"] + self._flee[0]) * FLEE_X_PX,
            "jy": (random.uniform(-shake, shake) if shake else 0.0)
                  + (self._pose["dy"] + self._flee[1]) * FLEE_Y_PX,
        }

        frame = np.zeros((FB_HEIGHT, FB_WIDTH, 3), np.uint16)
        colour = np.array(colour_rgb, np.float32) * self._pose["glow"]
        lit = colour * (0.86 + 0.14 * self._assembly)
        if state in AVERTING:
            self._columns(frame, now, colour)
        for side in (-1, 1):
            self._eye(frame, side, descent[0 if side < 0 else 1],
                      cam, lit, state, fill)

        if asleep or self._woke_at is not None:
            self._snooze(frame, now, cam["centre"], colour, emit=asleep)

        # The thing that is bothering her.
        bug = self._idle.bug_at(now)
        if bug is not None:
            bx, by, presence = bug
            # The same pixels-per-unit her own displacement uses, which it did
            # not before: she moved at FLEE_X_PX and the speck was drawn at
            # FB_WIDTH * 0.42, so the gap the simulation was carefully holding
            # open bore no relation to the gap on the panel. It came out
            # sitting on her face while every number said it was clear.
            self._bug(frame,
                      FB_WIDTH / 2 + bx * FLEE_X_PX,
                      cam["centre"] + by * FLEE_Y_PX,
                      colour * presence, now)

        # The readout: one line of type and a hairline that fills. Speaking
        # hands her the whole panel, so it fades out entirely.
        if self._hud > 0.02:
            mask = self._label(note or label, colour_rgb)
            if mask is not None:
                band = frame[READOUT_TOP:READOUT_TOP + mask.shape[0], 0:FB_WIDTH]
                tint = (mask[:, :, None] * np.array(colour_rgb, np.float32)
                        * self._hud).astype(np.uint16)
                np.maximum(band, tint[:band.shape[0]], out=band)
            fraction = progress if state == "synthesizing" else (
                min(1.0, level * 1.2) if speaking else 0.0)
            rail = (np.array(colour_rgb, np.float32) * 0.18
                    * self._hud).astype(np.uint16)
            strip = frame[RAIL_Y:RAIL_Y + RAIL_H, 10:FB_WIDTH - 10]
            np.maximum(strip, rail, out=strip)
            filled = int((FB_WIDTH - 20) * max(0.0, min(1.0, fraction)))
            if filled > 0:
                bright = (np.array(colour_rgb, np.float32)
                          * self._hud).astype(np.uint16)
                lit_strip = frame[RAIL_Y:RAIL_Y + RAIL_H, 10:10 + filled]
                np.maximum(lit_strip, bright, out=lit_strip)

        frame = self._disturb(frame, now, in_pieces)
        self._write(frame)

    def _write(self, frame: np.ndarray) -> None:
        packed = (((frame[..., 0] >> 3) << 11)
                  | ((frame[..., 1] >> 2) << 5)
                  | (frame[..., 2] >> 3))
        try:
            with open(FB, "r+b") as fb:
                fb.write(packed.astype("<u2").tobytes())
        except OSError:
            pass


Face = Void
