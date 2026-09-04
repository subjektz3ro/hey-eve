"""What she does when nothing is happening.

A face that only sways is furniture. These are the moves that make an object
on a desk feel like it is in the room with you — and the whole design problem
is rarity. An idle you see every eight seconds stops being behaviour and
becomes a screensaver, so the scheduler here spends most of its effort
deciding *not* to do something.

Three rules keep it from feeling canned:

  * Weighted choice, heavily skewed. A glance is twenty times likelier than a
    yawn, so the rare ones stay surprising.
  * No repeats within the last three. Randomness alone will happily play the
    same trick twice in a row, and one repeat is all it takes to notice the
    loop.
  * A per-behaviour cooldown on top. Even if the dice favour it, she will not
    roll her eyes twice inside two minutes.

The bug is the odd one out: a single point that wanders in, pesters her, and
leaves. It is the only idle that implies something else exists in the room,
which is worth more for presence than anything she does with her own face.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass

# What an idle may change. Everything is a delta on the resting pose, so a
# behaviour only has to describe the part it moves.
NEUTRAL = {
    "gaze": 0.0, "rise": 0.0, "lid_l": 1.0, "lid_r": 1.0, "wide": 1.0,
    "roll": 0.0, "tilt": 0.0, "scale": 1.0, "shake": 0.0, "glow": 1.0,
    "flick": 0.0,
    # Where she has got to on the panel, in panel-halves from centre. Every
    # other key above moves part of her face; these two move all of it, and
    # until the bug learned to chase her nothing ever set them. A renderer
    # that does not read them simply stays put — see void._render, which maps
    # them onto the jx/jy its projection already had.
    "dx": 0.0, "dy": 0.0,
}


def _ease(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def _bump(t: float) -> float:
    """Nought to one and back, smoothly."""
    return math.sin(max(0.0, min(1.0, t)) * math.pi)


# --- the behaviours ------------------------------------------------------
# Each takes progress 0..1 through its own duration and returns deltas.

def _glance(p):
    to = 1.0 if p < 0.5 else -1.0
    hold = _ease(p / 0.18) * (1.0 - _ease((p - 0.72) / 0.28))
    return {"gaze": to * 0.055 * hold}


def _look_around(p):
    stops = (0.6, -0.5, 0.25, 0.0)
    index = min(int(p * len(stops)), len(stops) - 1)
    return {"gaze": stops[index] * 0.06, "roll": stops[index] * 0.07}


def _track(p):
    # Follows something crossing the room, then loses interest in it.
    swing = (p / 0.65) * 2 - 1 if p < 0.65 else 1 - _ease((p - 0.65) / 0.35) * 2
    return {"gaze": swing * 0.075, "roll": swing * 0.11}


def _double_take(p):
    if p < 0.34:
        return {}
    if p < 0.40:
        k = _ease((p - 0.34) / 0.06)
        return {"gaze": -0.07 * k, "roll": -0.13 * k}
    if p < 0.44:
        return {"gaze": -0.07, "roll": -0.13}
    if p < 0.50:
        k = _ease((p - 0.44) / 0.06)
        return {"gaze": -0.07 * (1 - k), "roll": -0.13 * (1 - k),
                "lid_l": 1 + 0.32 * k, "lid_r": 1 + 0.32 * k}
    if p < 0.72:
        return {"lid_l": 1.32, "lid_r": 1.32, "flick": 0.30}
    k = 1 - _ease((p - 0.72) / 0.28)
    return {"lid_l": 1 + 0.32 * k, "lid_r": 1 + 0.32 * k, "flick": 0.30 * k}


def _startle(p):
    k = _bump(p) ** 0.5
    return {"lid_l": 1 + k, "lid_r": 1 + k, "scale": 1 + 0.055 * k,
            "shake": k * 2.6 if p < 0.4 else 0.0, "flick": 0.35 * k}


def _look_away(p):
    if p < 0.28:
        k = _ease(p / 0.28)
        return {"roll": -0.36 * k, "gaze": -0.05 * k, "flick": -0.12 * k}
    if p < 0.62:
        return {"roll": -0.36, "gaze": -0.05, "flick": -0.12}
    k = 1 - _ease((p - 0.62) / 0.38)
    return {"roll": -0.36 * k, "gaze": -0.05 * k, "flick": -0.12 * k}


def _eye_roll(p):
    k = _bump(p)
    return {"rise": -0.055 * k, "gaze": 0.05 * math.sin(p * math.pi * 2),
            "lid_l": 1 - 0.28 * k, "lid_r": 1 - 0.28 * k, "flick": 0.25 * k}


def _yawn(p):
    k = _bump(p) ** 0.6
    return {"lid_l": 1 - 0.90 * k, "lid_r": 1 - 0.90 * k, "tilt": -0.17 * k,
            "scale": 1 + 0.055 * k, "rise": 0.02 * k}


def _wink(p):
    s = _ease(p / 0.14) * (1.0 - _ease((p - 0.42) / 0.36))
    return {"lid_r": 1 - s * 0.95, "lid_l": 1 + s * 0.14, "roll": s * 0.07,
            "flick": s * 0.30}


def _shiver(p):
    k = _bump(p)
    return {"shake": k * 3.2, "roll": math.sin(p * 34) * 0.05 * k,
            "lid_l": 1 - 0.22 * k, "lid_r": 1 - 0.22 * k}


def _stretch(p):
    k = _bump(p)
    return {"scale": 1 + 0.10 * k, "rise": -0.03 * k, "tilt": -0.09 * k,
            "lid_l": 1 - 0.5 * k, "lid_r": 1 - 0.5 * k, "flick": 0.22 * k}


def _nod(p):
    return {"tilt": 0.10 * math.sin(p * math.pi * 2), "rise": 0.012 * math.sin(p * math.pi * 2)}


def _huff(p):
    """A short exhale: eyes narrow, everything sinks a fraction, recovers."""
    k = _bump(p)
    return {"lid_l": 1 - 0.35 * k, "lid_r": 1 - 0.35 * k, "rise": 0.022 * k,
            "scale": 1 - 0.022 * k, "flick": -0.18 * k}


# --- and what she does while she is asleep -------------------------------
# The first version of sleep had exactly one move: the lids lifted 40% and
# came back, every half-minute or so. Watched on the real panel it read as a
# jump scare — an eye snapping open with the whole rest of her perfectly
# still is a horror-film beat, not somebody sleeping.
#
# What was missing is that a sleeping body moves and a sleeping face mostly
# does not. A hypnic jerk is the torso going while the eyes stay shut. So
# these are body moves, and the most any of them opens her is a crack.
#
# They also have to be far more frequent than the waking idles. Awake, rarity
# is the whole design — an idle every eight seconds is a screensaver. Asleep
# it inverts: stillness is the default state, so without something every ten
# seconds or so the panel reads as a frozen frame rather than a sleeper.

def _sleep_stir(p):
    """A small resettle. The commonest thing a sleeper does."""
    k = _bump(p)
    return {"roll": 0.055 * k, "tilt": 0.030 * k, "rise": 0.010 * k}


def _sleep_jerk(p):
    """A hypnic jerk: the body goes and the eyes stay shut.

    Fast in and slow out, because a jolt is a snap followed by a long settle
    and a symmetrical curve reads as a nod. The lids move barely at all — six
    hundredths against the four tenths of the version this replaced, which is
    the entire difference between a twitch and a jump scare.
    """
    k = _ease(p / 0.04) * math.exp(-p * 5.5)
    return {"shake": 2.6 * k, "roll": -0.075 * k, "tilt": 0.055 * k,
            "scale": 1 + 0.028 * k, "rise": -0.014 * k,
            "lid_l": 1 + 0.05 * k, "lid_r": 1 + 0.06 * k}


def _sleep_dream(p):
    """Rapid eye movement. The lids flutter; nothing else much happens.

    The asymmetry is deliberate — the two lids run on opposite phases, which
    is what stops it reading as a slow blink.
    """
    k = _bump(p)
    flutter = math.sin(p * math.pi * 11.0)
    return {"lid_l": 1 + 0.055 * k * (0.55 + 0.45 * flutter),
            "lid_r": 1 + 0.050 * k * (0.55 - 0.45 * flutter),
            "gaze": 0.030 * k * flutter,
            "tilt": 0.014 * k * math.sin(p * math.pi * 3.0)}


def _sleep_sigh(p):
    """A breath deeper than the ones underneath, and a slow settle out."""
    k = _bump(p)
    return {"scale": 1 + 0.032 * k, "rise": -0.018 * k, "roll": 0.022 * k}


def _sleep_turn(p):
    """Turning over: a long slow drift away and back.

    Out and back rather than to a new resting place, because a behaviour that
    ends somewhere other than where it started snaps home when it finishes.
    Over six seconds the return is too slow to read as one.
    """
    k = _bump(p)
    return {"roll": 0.105 * k, "tilt": -0.045 * k, "rise": 0.009 * k,
            "lid_l": 1 + 0.03 * k, "lid_r": 1 + 0.02 * k}


@dataclass
class Behaviour:
    name: str
    seconds: float
    weight: float
    cooldown: float
    pose: object
    last_at: float = -999.0


# Named, because waking her up plays it deliberately rather than waiting for
# the dice — the one moment a yawn is not a surprise but the obvious thing to
# do. See Scheduler.play.
YAWN = Behaviour("yawn", 2.6, 2, 260.0, _yawn)

BEHAVIOURS = [
    Behaviour("glance",      1.6,  26, 4.0,   _glance),
    Behaviour("look_around", 4.2,  10, 25.0,  _look_around),
    Behaviour("nod",         2.2,   8, 30.0,  _nod),
    Behaviour("track",       6.0,   6, 50.0,  _track),
    Behaviour("look_away",   5.0,   6, 60.0,  _look_away),
    Behaviour("huff",        1.6,   6, 45.0,  _huff),
    Behaviour("double_take", 3.6,   5, 95.0,  _double_take),
    Behaviour("shiver",      0.9,   3, 150.0, _shiver),
    Behaviour("stretch",     2.4,   3, 150.0, _stretch),
    Behaviour("eye_roll",    1.6,   3, 180.0, _eye_roll),
    Behaviour("wink",        0.9,   3, 240.0, _wink),
    YAWN,
    Behaviour("bug",        16.0,   2, 320.0, None),      # drawn, not posed
]

# Asleep. Weighted the other way round from the waking set: a plain resettle
# is what she mostly does, and the jerk is the rare one you are pleased to
# catch. The gap between them is short because stillness is already the
# default here — see the note above _sleep_stir.
SLEEP_BEHAVIOURS = [
    Behaviour("stir",   2.2, 20,   7.0, _sleep_stir),
    Behaviour("dream",  2.8, 14,  20.0, _sleep_dream),
    Behaviour("sigh",   2.6, 10,  34.0, _sleep_sigh),
    Behaviour("turn",   6.0,  7,  50.0, _sleep_turn),
    Behaviour("jerk",   1.8,  4,  75.0, _sleep_jerk),
]
SLEEP_GAP = (4.5, 12.0)

# Not scheduled — this one fires on an actual noise in the room, which is the
# only idle here that is a genuine reaction rather than a performance.
STARTLE = Behaviour("startle", 1.5, 0, 12.0, _startle)


# --- the chase -----------------------------------------------------------
# How much of the panel she may use while running, in panel-halves from
# centre. She never leaves it: a face that goes off the edge reads as broken
# rather than frightened, and being unable to get any further is the point of
# the walls — cornered is a state worth being able to reach.
FLEE_REACH_X, FLEE_REACH_Y = 0.58, 0.34
# Where the soft walls stop being a suggestion. See _pen.
FLEE_LIMIT_X, FLEE_LIMIT_Y = 0.70, 0.44
FLEE_PUSH = 5.0         # how hard the bug repels her
FLEE_DAMP = 3.2         # and how fast she stops once it backs off
FLEE_HOME = 2.0         # the weak pull back to centre with nothing chasing her
FLEE_WALL = 26.0        # soft edges
FLEE_MAX = 3.4          # top speed, panel-halves per second
FLEE_SWERVE = 2.0       # how much of the push goes sideways rather than away
BUG_SPEED = 1.15        # panel-halves per second; hers is three times this...
BUG_PURSUIT = 0.78      # ...and this is why she never gets to settle
BUG_WANDER = 0.16       # it is an insect, not a guided missile
# How far off her it hovers. Her face is most of a panel-half wide, so a
# standoff under about 0.6 puts the thing *on* her rather than beside her —
# which is what 0.34 did, and it read as a mole rather than as a pest. This
# is the one number here with a hard floor under it: it has to clear her.
BUG_STANDOFF = 0.90
# How the fear reads that distance. Anchored on the standoff rather than on
# zero, so "as close as it ever gets" is exactly 1.0 whatever the standoff is
# retuned to — otherwise every threshold below has to move with it.
NEAR_FALLOFF = 0.55

# The two lids do opposite jobs at different distances, and that is the whole
# read. You narrow your eyes to resolve something small and far away — that is
# curiosity, not fear — and they snap open when you work out what you are
# looking at. Squint first, wide second; the changeover is the beat the whole
# idle turns on, so alarm rises fast once it starts.
FOCUS_AT = 0.34         # how near it has to be before she peers at it
FOCUS_WIDTH = 0.19
SQUINT = 0.36           # how far the lids come down while she is focusing
ALARM_FROM = 0.46       # where recognition takes over from curiosity
ALARM_WIDE = 0.58       # and how far they pull back once it has


class Bug:
    """A speck that wanders in, chases her round the panel, and leaves.

    The only idle that implies something else is in the room, and the only one
    that is a simulation rather than a curve. She is not playing an animation
    of being frightened: she is being pushed away from a thing that follows
    her, so the chase falls out of the two of them rather than being scripted,
    and no two runs of it are the same.

    Both bodies live here. The bug keeps a parametric crossing — in from an
    edge, out the far side, so it always arrives and always leaves — with a
    pursuit term laid over the middle that steers it at wherever she has got
    to. Her flight is a spring: repelled by the bug, damped, walled in by the
    panel and pulled weakly home. She is quicker than it, so she always
    escapes; it is more persistent than she is, so she never gets to settle.

    What it replaced tracked it with her eyes and squinted when it came close,
    which reads as mildly irritated by something on the far side of a window.
    She never moved, and the speck was a single full stop seven pixels wide.
    """

    def __init__(self, seconds: float) -> None:
        self.seconds = seconds
        self.side = random.choice((-1.0, 1.0))
        self.seed = random.random() * 100.0
        self.x, self.y = self.side * 1.40, -0.12     # where the bug is
        self.hx, self.hy = 0.0, 0.0                  # where she has fled to
        self.vx, self.vy = 0.0, 0.0
        self._last_p = 0.0

    # --- the simulation -------------------------------------------------

    def advance(self, p: float) -> None:
        """Move both of them on by however much time this frame was.

        Driven from progress rather than a clock because that is all the
        scheduler has to give, and clamped because a stalled renderer must
        produce a slow frame rather than a teleport.
        """
        dt = max(0.0, min(0.08, (p - self._last_p) * self.seconds))
        self._last_p = p
        if dt <= 0.0:
            return

        # The bug's own errand, and how much it currently cares about her.
        travel = min(1.0, p / 0.94)
        drift_x = self.side * (1.40 - 2.80 * travel)
        drift_y = -0.12 + math.sin(travel * 6.0 + self.seed) * 0.44
        hunting = (_ease((p - 0.16) / 0.20)
                   * (1.0 - _ease((p - 0.72) / 0.18)) * BUG_PURSUIT)
        want_x = drift_x + (self.hx - drift_x) * hunting
        want_y = drift_y + (self.hy - drift_y) * hunting
        # Wander, so it is not a guided missile. A pursuer that corrects
        # perfectly every frame does not read as an insect, and worse, it
        # always wins: measured without this it closed to a gap of 0.01 and
        # sat on her face for the rest of the visit.
        want_x += math.sin(p * 17.0 + self.seed) * BUG_WANDER
        want_y += math.cos(p * 13.0 + self.seed * 1.7) * BUG_WANDER * 0.8
        # It hovers near her rather than landing on her. Without this it aims
        # at exactly where she is, and once she has run out of room it simply
        # arrives — measured, it closed to a gap of 0.02 and sat on her face
        # for the rest of the visit, which is a different and much worse idle
        # than being chased. Standing off also caps how hard she is pushed,
        # which is what stops her being welded to the edge of the panel.
        off_x, off_y = want_x - self.hx, want_y - self.hy
        off = math.hypot(off_x, off_y)
        if off < BUG_STANDOFF:
            if off < 1e-6:
                off_x, off_y, off = 1.0, 0.0, 1.0
            want_x = self.hx + off_x / off * BUG_STANDOFF
            want_y = self.hy + off_y / off * BUG_STANDOFF
        # And it travels at a fixed speed rather than closing a fraction of
        # the gap each frame. Exponential convergence has no speed limit, so
        # it caught her whatever she did; a slow pursuer against a quick
        # evader is what makes the chase a chase. FLEE_MAX is three times this.
        step_x, step_y = want_x - self.x, want_y - self.y
        reach = math.hypot(step_x, step_y) or 1e-6
        travel_now = min(BUG_SPEED * dt, reach)
        self.x += step_x / reach * travel_now
        self.y += step_y / reach * travel_now

        # And her, running from it.
        #
        # The homing term ramps hard at the end. The scheduler drops this
        # behaviour the moment p passes 1 and her displacement simply stops
        # being applied, so being anywhere but the middle when that happens is
        # a teleport — she has to be walked home before the music stops rather
        # than left wherever the chase finished.
        homing = FLEE_HOME * (1.0 + 7.0 * _ease((p - 0.82) / 0.18))
        # How hard she is pushed is how near it feels, which is the same
        # question near() already answers — and answering it twice is how this
        # broke. The falloff here was written against a standoff of 0.34; once
        # the renderer and the simulation were made to agree about distance
        # and the speck was moved clear of her face, the same curve delivered
        # a tenth of the force and she stopped running.
        # Faded by presence as well. Without it a speck that has finished
        # dissolving goes on shoving her from off-panel, so she spends the end
        # of the visit running from something invisible and is still displaced
        # when the behaviour is dropped — which is the teleport home this is
        # all supposed to avoid.
        push = FLEE_PUSH * self.near() * _presence(p)
        (self.hx, self.hy), (self.vx, self.vy) = flee_step(
            (self.hx, self.hy), (self.vx, self.vy),
            (self.x, self.y), push, dt, homing)

    def near(self) -> float:
        """How close it has got to her, 0 to 1.

        Measured from the standoff, not from her centre, so the closest it is
        ever allowed to get reads as exactly 1.0. Anchored the other way, every
        threshold that keys off this would have to be retuned whenever the
        standoff moved — and the standoff had to move a long way once the
        renderer and the simulation were made to agree about distance.
        """
        dx, dy = self.hx - self.x, self.hy - self.y
        beyond = max(0.0, math.hypot(dx, dy) - BUG_STANDOFF)
        return math.exp(-(beyond * beyond) / NEAR_FALLOFF)

    def cornered(self) -> float:
        """How far she has run out of room to run, 0 to 1.

        Measured against the hard limit rather than where the soft walls
        start. Any real push carries her past FLEE_REACH — that is what the
        soft walls are for — so scaling by it reported her as fully cornered
        for 97% of every chase, which made the word mean nothing.
        """
        return min(1.0, max(abs(self.hx) / FLEE_LIMIT_X,
                            abs(self.hy) / FLEE_LIMIT_Y))

    # --- what each of them looks like ------------------------------------

    def at(self, p: float) -> tuple[float, float, float]:
        """Where to draw the bug, and how present it is."""
        return self.x, self.y, _presence(p)

    def pose(self, p: float) -> dict:
        self.advance(p)
        presence = self.at(p)[2]
        near = self.near() * presence

        # Curiosity, then alarm. Focus is a band at middle distance rather
        # than a slope, so it fades out again as recognition arrives — she
        # stops peering at it the moment she knows what it is.
        focus = math.exp(-((near - FOCUS_AT) / FOCUS_WIDTH) ** 2)
        alarm = _ease((near - ALARM_FROM) / 0.22)
        # Everything she does about it fades with presence, the lids included.
        # Left ungated they sit a whisker narrowed even with nothing on the
        # panel, because a focus band centred at middle distance is not quite
        # zero at zero — a fortieth of a squint, permanently, for no reason.
        lid = 1.0 + (ALARM_WIDE * alarm
                     - SQUINT * focus * (1.0 - alarm)) * presence

        toward_x = max(-1.0, min(1.0, self.x - self.hx))
        toward_y = max(-1.0, min(1.0, self.y - self.hy))
        return {
            "gaze": toward_x * 0.085 * presence,
            # She stops cocking her head once she is running: a tilt while
            # fleeing reads as curiosity, which is the wrong half of this.
            "roll": toward_x * 0.11 * presence * (1.0 - alarm * 0.75),
            "rise": toward_y * 0.030 * presence,
            "lid_l": lid, "lid_r": lid,
            "flick": (0.30 * focus + 0.60 * alarm) * presence,
            "scale": 1.0 - 0.035 * alarm * presence,
            # Trembling hardest with her back to the wall, which is the one
            # thing the walls above buy that a bounded range would not.
            "shake": 3.0 * alarm * (0.35 + 0.65 * self.cornered()) * presence,
            # Not gated on presence, deliberately. She is wherever she ran to,
            # and the bug fading out must not teleport her home — the spring
            # walks her back on its own once nothing is pushing her.
            "dx": self.hx, "dy": self.hy,
        }


def flee_step(pos: tuple[float, float], vel: tuple[float, float],
              threat: tuple[float, float] | None, push: float, dt: float,
              homing: float = FLEE_HOME):
    """One step of her running away from something.

    Shared by the bug and by a finger on the glass, because they are the same
    problem wearing different hats: a thing at a position, a push away from
    it, damping, soft walls and a weak pull home. What differs is only how the
    thing moves — the bug does its own pursuing, a finger is pursued by you —
    and that stays with each caller.

    Two details are load-bearing and both were paid for.

    The push carries a sideways component as well as a straight one. Repulsion
    alone drives an evader into whichever corner it reaches first and holds it
    there: measured, she was pinned for 97% of every chase, which is hiding
    rather than running. And of the two ways round the thing, this takes the
    one heading back toward open panel — fixed to a side, it was as likely to
    drive her deeper into the corner she was already in.

    The hard stop sits behind the soft walls. The walls give the feel; _pen
    guarantees the promise, so retuning any constant here can never put half
    her face off the edge of the panel.
    """
    hx, hy = pos
    vx, vy = vel
    ax = ay = 0.0
    if push > 0.0 and threat is not None:
        dx, dy = hx - threat[0], hy - threat[1]
        gap = math.hypot(dx, dy)
        if gap < 1e-3:
            # Dead centre. The direction away from a thing you are standing on
            # is undefined, and normalising it gives zero — so she takes the
            # full force of it and does not move at all. Any way out will do;
            # what matters is that there is one. A speck never quite manages
            # this because it stands off, but a finger lands wherever it likes.
            dx, dy, gap = 1.0, 0.0, 1.0
        ax = dx / gap * push
        ay = dy / gap * push
        swerve_x, swerve_y = -dy / gap, dx / gap
        if swerve_x * hx + swerve_y * hy > 0.0:
            swerve_x, swerve_y = -swerve_x, -swerve_y
        ax += swerve_x * push * FLEE_SWERVE
        ay += swerve_y * push * FLEE_SWERVE
    ax += -vx * FLEE_DAMP - hx * homing
    ay += -vy * FLEE_DAMP - hy * homing
    ax -= FLEE_WALL * max(0.0, abs(hx) - FLEE_REACH_X) * _sign(hx)
    ay -= FLEE_WALL * max(0.0, abs(hy) - FLEE_REACH_Y) * _sign(hy)
    vx = max(-FLEE_MAX, min(FLEE_MAX, vx + ax * dt))
    vy = max(-FLEE_MAX, min(FLEE_MAX, vy + ay * dt))
    hx, vx = _pen(hx + vx * dt, vx, FLEE_LIMIT_X)
    hy, vy = _pen(hy + vy * dt, vy, FLEE_LIMIT_Y)
    return (hx, hy), (vx, vy)


def _presence(p: float) -> float:
    """How much of the speck there is: it fades in, and it fades out."""
    return max(0.0, min(1.0, _ease(p / 0.08), 1.0 - _ease((p - 0.90) / 0.10)))


def _sign(value: float) -> float:
    return 1.0 if value >= 0.0 else -1.0


def _pen(position: float, velocity: float, limit: float) -> tuple[float, float]:
    """Keep her inside the panel, and stop her pressing against the edge.

    Killing the velocity as well as the position matters: a clamp on its own
    leaves the integrator holding a speed it can never spend, so she sticks to
    the wall for as long as the push lasts and then leaps off it.
    """
    if position > limit:
        return limit, min(0.0, velocity)
    if position < -limit:
        return -limit, max(0.0, velocity)
    return position, velocity


class Scheduler:
    """Decides what she does, and — mostly — that she does nothing."""

    def __init__(self) -> None:
        self.current: Behaviour | None = None
        self.started = 0.0
        self.next_at = 5.0
        self.recent: list[str] = []
        self.bug: Bug | None = None
        self.sleeping = False

    def update(self, now: float, idle: bool, asleep: bool = False) -> dict:
        """Advance and return the pose deltas for this frame.

        `asleep` swaps the catalogue rather than adding a branch everywhere:
        a sleeper still stirs, sighs and jerks, they are simply different
        moves on the same pose vocabulary at a much shorter interval. It
        defaults to False so every existing caller behaves identically.
        """
        if not (idle or asleep):
            self.current, self.bug = None, None
            self.next_at = now + random.uniform(4.0, 9.0)
            self.sleeping = False
            return dict(NEUTRAL)

        if asleep != self.sleeping:
            # Dropping off, or waking. Abandon whatever was mid-flight rather
            # than letting a glance play out after her eyes have shut, and
            # leave a beat before the first sleep move so falling asleep is
            # not immediately followed by twitching.
            self.sleeping = asleep
            self.current, self.bug = None, None
            self.next_at = now + (random.uniform(2.0, 5.0) if asleep else 0.0)

        pool = SLEEP_BEHAVIOURS if asleep else BEHAVIOURS
        gap = SLEEP_GAP if asleep else (4.5, 12.0)

        if self.current is not None:
            p = (now - self.started) / self.current.seconds
            if p >= 1.0:
                self.current.last_at = now
                self.current, self.bug = None, None
                self.next_at = now + random.uniform(*gap)
            else:
                pose = (self.bug.pose(p) if self.bug is not None
                        else self.current.pose(p))
                return {**NEUTRAL, **pose}

        if now >= self.next_at:
            pick = self._choose(now, pool)
            if pick is not None:
                self.current, self.started = pick, now
                self.recent = (self.recent + [pick.name])[-3:]
                self.bug = Bug(pick.seconds) if pick.name == "bug" else None
            else:
                self.next_at = now + 2.0
        return dict(NEUTRAL)

    def play(self, behaviour: Behaviour, now: float) -> None:
        """Start one behaviour deliberately, whatever else was running.

        `now` may be in the future, which is how waking gets a yawn a beat
        after her eyes open rather than at the same instant: progress comes
        out negative until it arrives and every pose function clamps.
        """
        behaviour.last_at = now
        self.current, self.started, self.bug = behaviour, now, None
        self.recent = (self.recent + [behaviour.name])[-3:]

    def startle(self, now: float) -> None:
        """Something banged. Interrupt whatever she was doing and jump."""
        if now - STARTLE.last_at < STARTLE.cooldown:
            return
        self.play(STARTLE, now)

    def _choose(self, now: float, pool: list) -> Behaviour | None:
        pool = [b for b in pool
                if b.name not in self.recent and now - b.last_at >= b.cooldown]
        if not pool:
            return None
        total = sum(b.weight for b in pool)
        roll = random.random() * total
        for behaviour in pool:
            roll -= behaviour.weight
            if roll <= 0:
                return behaviour
        return pool[-1]

    def bug_at(self, now: float) -> tuple[float, float, float] | None:
        """Where to draw the speck, if there is one."""
        if self.bug is None or self.current is None:
            return None
        p = (now - self.started) / self.current.seconds
        x, y, presence = self.bug.at(p)
        return (x, y, presence) if presence > 0 else None
