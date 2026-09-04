"""Her asleep: lids down, breathing, and little z's floating off.

The first version of sleep blanked the panel. That was wrong for exactly the
reason main._silence_state is careful about a fault — a dark screen cannot be
told apart from a crash, a dead backlight or a pulled plug, and looking fine
while being broken is this project's recurring failure. Blanking her to save a
fifth of a core solved a problem nobody had at the cost of the one thing the
panel is for.

So she is drawn asleep instead, and the frame rate does not drop while she is:
that confuses slow *motion* with a slow *frame rate*. A four-second breath at
fifty frames is smooth and at five is a slideshow, and the z's stutter.

Everything here runs headless. `_write` is the only line in the renderer that
touches hardware, so replacing it hands back the frame she would have drawn
and the rest is ordinary array arithmetic.
"""
from __future__ import annotations

import numpy as np
import pytest

from eve import head, idle, void


def frame_at(face, now: float, dt: float = 0.02) -> np.ndarray:
    """Render one frame and return it rather than writing it to /dev/fb0."""
    caught = {}
    face._write = lambda drawn: caught.__setitem__("frame", drawn)
    face._render(now, dt)
    return caught["frame"]


@pytest.fixture
def sleeper(monkeypatch):
    """A void face, asleep, with the tube's glitch held still.

    _disturb shears bands sideways and drops the vertical hold, which is the
    point of the look and the enemy of asserting where any particular ink
    landed. It is exercised by the frames every other test here draws.
    """
    face = void.Face()
    monkeypatch.setattr(type(face), "_disturb",
                        lambda self, frame, now, unstable: frame)
    face.set_state("asleep")
    return face


class TestSheCanBeToldSheIsAsleep:
    def test_asleep_is_a_state_she_accepts(self):
        # set_state coerces anything not in LOOK to "idle", so without the
        # table entry she would simply stay awake and nothing would say why.
        face = void.Face()
        face.set_state("asleep")
        assert face._state == "asleep"

    def test_it_carries_a_label_and_a_colour_of_its_own(self):
        colour, in_pieces, label = head.LOOK["asleep"]
        assert label == "ASLEEP"
        assert in_pieces is False, "she is asleep, not disassembled"
        assert len(colour) == 3

    def test_it_does_not_read_as_one_of_the_working_states(self):
        assert "asleep" not in head.ATTENTIVE
        assert "asleep" not in void.AVERTING
        assert "asleep" not in void.CONTACT


class TestTheZAtlas:
    def test_a_z_is_rasterised_at_several_sizes(self):
        assert len(void.ZS) == len(void.Z_SIZES) >= 4

    def test_they_grow(self):
        # The whole trick. A z that keeps its size reads as a sprite sliding
        # sideways; one that grows reads as something drifting away from her.
        sizes = [mask.shape[0] for mask in void.ZS]
        assert sizes == sorted(sizes)
        assert sizes[-1] > sizes[0] * 2

    def test_each_one_actually_has_a_letter_in_it(self):
        # A missing font would rasterise to an empty tile and the snooze
        # would silently become nothing at all.
        for mask in void.ZS:
            assert mask.max() > 0.5

    def test_they_are_alpha_tiles_like_every_other_glyph(self):
        for mask in void.ZS:
            assert mask.dtype == np.float32
            assert 0.0 <= mask.min() and mask.max() <= 1.0


class TestTheSnooze:
    def test_the_first_one_waits_a_beat_after_her_eyes_close(self, sleeper):
        frame_at(sleeper, 0.0)
        assert sleeper._zs == [], "a z was already airborne as she dropped off"

    def test_then_they_start(self, sleeper):
        frame_at(sleeper, 0.0)
        frame_at(sleeper, 2.0)
        assert len(sleeper._zs) == 1

    def test_about_three_are_in_the_air_at_once(self, sleeper):
        # One is a typo and two is a coincidence; three staggered up the
        # diagonal is snoring. Z_LIFE / Z_EVERY is what sets it.
        frame_at(sleeper, 0.0)
        for tick in range(1, 900):
            frame_at(sleeper, tick * 0.05)
        assert 2 <= len(sleeper._zs) <= 5

    def test_they_are_culled_once_they_have_faded(self, sleeper):
        frame_at(sleeper, 0.0)
        for tick in range(1, 400):
            frame_at(sleeper, tick * 0.05)
        oldest = min(born for born, _ in sleeper._zs)
        assert 20.0 - oldest <= void.Z_LIFE + void.Z_EVERY

    def test_no_two_wobble_in_step(self, sleeper):
        frame_at(sleeper, 0.0)
        for tick in range(1, 400):
            frame_at(sleeper, tick * 0.05)
        phases = [phase for _, phase in sleeper._zs]
        assert len(set(phases)) == len(phases)

    def test_falling_asleep_again_starts_from_nothing(self, sleeper):
        frame_at(sleeper, 0.0)
        for tick in range(1, 200):
            frame_at(sleeper, tick * 0.05)
        assert sleeper._zs
        sleeper.set_state("idle")
        frame_at(sleeper, 10.1)
        sleeper.set_state("asleep")
        frame_at(sleeper, 10.2)
        assert sleeper._zs == [], "she woke and went back to sleep mid-snore"


SLEEP_POSES = [b for b in idle.SLEEP_BEHAVIOURS if b.pose is not None]


def sampled(behaviour, steps=200):
    """Every pose a behaviour passes through, start to finish."""
    return [{**idle.NEUTRAL, **behaviour.pose(i / steps)}
            for i in range(steps + 1)]


class TestWhatSheDoesInHerSleep:
    """The correction: a sleeping body moves and a sleeping face does not.

    The first version had one move — the lids lifted 40% and came back, every
    half-minute or so, with the rest of her perfectly still. On the panel that
    is a horror-film beat, not somebody asleep. What was missing is that a
    hypnic jerk is the torso going while the eyes stay shut.
    """

    def test_there_is_a_catalogue_rather_than_a_single_twitch(self):
        assert len(SLEEP_POSES) >= 4

    def test_it_is_not_the_waking_catalogue(self):
        waking = {b.name for b in idle.BEHAVIOURS}
        assert not {b.name for b in idle.SLEEP_BEHAVIOURS} & waking

    def test_none_of_them_opens_her_eyes(self):
        """The jump-scare guard, and the whole reason this file changed.

        Poses speak in lid openness, 1.0 neutral and above it a lid pulled
        back. Anything much over a crack while she is asleep reads as her
        eyes snapping open, which is what shipped and what got reported.
        """
        for behaviour in SLEEP_POSES:
            for pose in sampled(behaviour):
                for side in ("lid_l", "lid_r"):
                    assert pose[side] <= 1.10, (
                        f"{behaviour.name} opens {side} to {pose[side]:.2f}; "
                        "asleep she may crack, not stare")

    def test_the_jerk_is_a_body_move(self):
        # If the only thing it changes is the face it is a jump scare again,
        # whatever the amplitude.
        poses = sampled(next(b for b in SLEEP_POSES if b.name == "jerk"))
        assert max(p["shake"] for p in poses) > 1.0
        assert max(abs(p["roll"]) for p in poses) > 0.02
        assert max(abs(p["scale"] - 1.0) for p in poses) > 0.01

    def test_the_jerk_snaps_in_and_settles_out(self):
        # A symmetrical curve reads as a nod. A jolt is fast in, slow out.
        poses = sampled(next(b for b in SLEEP_POSES if b.name == "jerk"))
        peak = max(range(len(poses)), key=lambda i: poses[i]["shake"])
        assert peak < len(poses) * 0.2, "it swelled instead of jolting"

    def test_the_dream_flutters_both_lids_out_of_step(self):
        # In step is a slow blink; out of step is rapid eye movement.
        poses = sampled(next(b for b in SLEEP_POSES if b.name == "dream"))
        mid = poses[len(poses) // 2 - 10: len(poses) // 2 + 10]
        assert any(p["lid_l"] != p["lid_r"] for p in mid)

    def test_every_one_ends_where_it_started(self):
        # A behaviour finishing anywhere else snaps home when the scheduler
        # drops it, and a snap is the one thing sleep cannot have.
        for behaviour in SLEEP_POSES:
            for edge in (behaviour.pose(0.0), behaviour.pose(1.0)):
                for key, value in edge.items():
                    assert abs(value - idle.NEUTRAL[key]) < 0.05, \
                        f"{behaviour.name} leaves {key} at {value}"

    def test_something_happens_far_more_often_than_when_awake(self):
        # Awake, rarity is the design — an idle every eight seconds is a
        # screensaver. Asleep it inverts: stillness is already the default,
        # so without something every ten seconds she reads as a frozen frame.
        assert max(idle.SLEEP_GAP) <= 12.0
        assert max(b.cooldown for b in idle.SLEEP_BEHAVIOURS) < 120.0


class TestTheSchedulerServesTheRightSet:
    def test_asleep_it_plays_sleep_moves(self):
        sched, seen = idle.Scheduler(), set()
        for tick in range(4000):
            sched.update(tick * 0.05, idle=False, asleep=True)
            if sched.current is not None:
                seen.add(sched.current.name)
        assert seen and seen <= {b.name for b in idle.SLEEP_BEHAVIOURS}

    def test_awake_it_still_plays_the_old_ones(self):
        sched, seen = idle.Scheduler(), set()
        for tick in range(4000):
            sched.update(tick * 0.05, idle=True)
            if sched.current is not None:
                seen.add(sched.current.name)
        assert seen and not seen & {b.name for b in idle.SLEEP_BEHAVIOURS}

    def test_neither_is_still_neutral(self):
        sched = idle.Scheduler()
        assert sched.update(10.0, idle=False) == idle.NEUTRAL

    def test_dropping_off_abandons_whatever_was_running(self):
        # A glance finishing after her eyes have shut is the seam this
        # prevents; the scheduler blends nothing, so it would simply appear.
        sched = idle.Scheduler()
        sched.play(idle.YAWN, 1.0)
        assert sched.current is not None
        sched.update(1.2, idle=False, asleep=True)
        assert sched.current is None

    def test_a_forced_behaviour_may_be_scheduled_ahead(self):
        # How waking gets a yawn a beat after the eyes open rather than at
        # the same instant: progress is negative until it arrives.
        sched = idle.Scheduler()
        sched.play(idle.YAWN, 10.0)
        assert sched.update(9.5, idle=True) == idle.NEUTRAL
        assert sched.current is idle.YAWN


class TestWakingUp:
    """She was going from fully shut to fully alert between two frames.

    That is the other half of what made sleep read as a light switch: it was
    reported as "no waking up animation", and it was not an oversight in the
    drawing — nothing was ever asked to happen.
    """

    def wake(self, face, at=6.0):
        for tick in range(int(at / 0.05)):
            frame_at(face, tick * 0.05)
        face.set_state("idle")
        frame_at(face, at)
        return at

    def test_her_eyes_do_not_snap_open(self, sleeper):
        woke = self.wake(sleeper)
        assert sleeper._woke_at == pytest.approx(woke)

    def test_the_ramp_is_long_enough_to_see(self):
        assert void.WAKE_LID_S >= 0.5

    def test_it_finishes(self, sleeper):
        woke = self.wake(sleeper)
        frame_at(sleeper, woke + void.WAKE_LID_S + 0.1)
        assert sleeper._woke_at is None, "she never finished waking up"

    def test_she_yawns(self, sleeper):
        # The behaviour existed the whole time and never got to run here,
        # which is the single most obvious thing a face does on waking.
        self.wake(sleeper)
        assert sleeper._idle.current is idle.YAWN

    def test_the_yawn_comes_after_the_eyes_start_opening(self, sleeper):
        woke = self.wake(sleeper)
        assert sleeper._idle.started > woke, "she yawned with her eyes shut"

    def test_she_keeps_moving_all_the_way_through_it(self, sleeper):
        woke = self.wake(sleeper)
        frames = [frame_at(sleeper, woke + step * 0.1) for step in range(1, 8)]
        assert not any(np.array_equal(frames[i], frames[i + 1])
                       for i in range(len(frames) - 1))

    def test_the_last_z_s_drift_off_rather_than_vanishing(self, sleeper):
        woke = self.wake(sleeper)
        assert sleeper._zs, "there were none in the air to begin with"
        before = len(sleeper._zs)
        frame_at(sleeper, woke + 0.1)
        assert len(sleeper._zs) == before, "they blinked out as she woke"

    def test_no_new_ones_are_emitted_while_she_wakes(self, sleeper):
        woke = self.wake(sleeper)
        before = len(sleeper._zs)
        for step in range(1, 20):
            frame_at(sleeper, woke + step * 0.05)
        assert len(sleeper._zs) <= before, "she was still snoring, awake"

    def test_waking_again_does_not_double_up(self, sleeper):
        self.wake(sleeper)
        assert sleeper._wake_yawn is False


class TestSheIsDrawnRatherThanBlanked:
    def test_a_frame_can_be_drawn_with_no_panel_at_all(self, sleeper):
        drawn = frame_at(sleeper, 3.0)
        assert drawn.shape == (head.FB_HEIGHT, head.FB_WIDTH, 3)

    def test_the_panel_is_not_dark(self, sleeper):
        # The entire point. A blank frame here is the bug this replaced.
        drawn = frame_at(sleeper, 3.0)
        assert drawn.max() > 0, "she went dark instead of going to sleep"
        assert int((drawn.max(axis=2) > 0).sum()) > 200

    def test_the_z_s_reach_the_frame(self, sleeper):
        """Ink above and to the right of her, where only z's ever go.

        Her eyes sit around VOID_CENTRE; the readout is far below. The band
        checked here is above both, so anything in it came from the snooze.
        """
        frame_at(sleeper, 0.0)
        for tick in range(1, 120):
            frame_at(sleeper, tick * 0.05)
        drawn = frame_at(sleeper, 6.0)
        above = drawn[0:70, 260:400]
        assert above.max() > 0, "nothing floated off her"

    def test_nothing_floats_off_her_while_she_is_awake(self, sleeper, monkeypatch):
        # The bug idle is silenced rather than the assertion weakened: it
        # wanders to centre - 108, which lands in this band, and it is the
        # only other thing that draws up here.
        monkeypatch.setattr(sleeper._idle, "bug_at", lambda now: None)
        sleeper.set_state("idle")
        for tick in range(120):
            drawn = frame_at(sleeper, tick * 0.05)
            assert drawn[0:70, 260:400].max() == 0, \
                "she is snoring with her eyes open"
        assert sleeper._zs == []

    def test_she_keeps_breathing_between_frames(self, sleeper):
        """The frame has to keep changing or it reads as a hung process.

        This is what the breath buys, and it is why the frame rate does not
        drop while she sleeps: the motion is slow, the sampling is not.
        """
        first = frame_at(sleeper, 3.0)
        later = frame_at(sleeper, 3.0 + void.SLEEP_BREATH_S / 2)
        assert not np.array_equal(first, later)

    def test_the_breath_is_a_sleeping_persons_rate(self):
        # Faster than this reads as panting, slower as something wrong.
        assert 3.0 <= void.SLEEP_BREATH_S <= 6.0
