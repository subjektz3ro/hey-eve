"""What she does when nobody is talking to her.

Idles are the difference between a face and a screensaver, and they fail in
one specific way: doing something too often, or two of the same thing in a
row, which reads as a loop rather than as a person. The scheduler's real job
is deciding that she does *nothing*, so that is most of what is tested here.
"""
from __future__ import annotations

import math

import pytest

from eve import idle


@pytest.fixture(autouse=True)
def rested_behaviours():
    """Clear the module-level cooldown clocks between tests.

    `Behaviour.last_at` is mutable state on module-level instances, so
    without this a test that fires `glance` leaves it on cooldown for every
    test that runs after it — and the suite starts depending on its order.
    """
    saved = [(behaviour, behaviour.last_at) for behaviour in idle.BEHAVIOURS]
    saved.append((idle.STARTLE, idle.STARTLE.last_at))
    for behaviour, _ in saved:
        behaviour.last_at = -999.0
    yield
    for behaviour, was in saved:
        behaviour.last_at = was


class TestEasing:
    @pytest.mark.parametrize("t", [-5.0, 0.0, 0.5, 1.0, 5.0])
    def test_easing_stays_inside_the_unit_interval(self, t):
        # Progress can overshoot by a frame at 50fps; an unclamped ease
        # would drive a pose past its own extreme and snap back.
        assert 0.0 <= idle._ease(t) <= 1.0

    def test_easing_is_smooth_at_both_ends(self):
        assert idle._ease(0.0) == 0.0
        assert idle._ease(1.0) == 1.0
        assert idle._ease(0.5) == pytest.approx(0.5)

    def test_a_bump_goes_up_and_comes_back(self):
        # Nought to one and back. A behaviour that ended anywhere other than
        # where it started would ratchet the pose over an evening.
        assert idle._bump(0.0) == pytest.approx(0.0)
        assert idle._bump(0.5) == pytest.approx(1.0)
        assert idle._bump(1.0) == pytest.approx(0.0, abs=1e-9)


class TestTheBehaviourTable:
    def test_every_behaviour_has_a_pose_except_the_one_that_is_drawn(self):
        for behaviour in idle.BEHAVIOURS:
            if behaviour.name == "bug":
                assert behaviour.pose is None    # drawn, not posed
            else:
                assert callable(behaviour.pose)

    def test_every_posed_behaviour_returns_deltas_the_face_understands(self):
        # A key the renderer does not know is silently ignored, so a typo
        # here is a behaviour that runs and does nothing visible.
        for behaviour in idle.BEHAVIOURS:
            if behaviour.pose is None:
                continue
            for progress in (0.0, 0.25, 0.5, 0.75, 1.0):
                for key in behaviour.pose(progress):
                    assert key in idle.NEUTRAL, \
                        f"{behaviour.name} moves unknown key {key!r}"

    def test_a_pose_never_asks_for_more_than_the_face_can_give(self):
        # These are targets the face eases toward, not frame-accurate
        # positions — `_look_around` deliberately steps straight to its first
        # stop and lets the spring do the moving. So the invariant is not
        # "starts at neutral" but "stays in range": a gaze of 5.0 would put
        # her eyes somewhere off the panel.
        limits = {"gaze": 0.5, "rise": 0.5, "roll": 0.5, "tilt": 0.5,
                  "flick": 2.0, "shake": 4.0, "lid_l": 2.0, "lid_r": 2.0,
                  "wide": 2.0, "scale": 2.0, "glow": 3.0}
        for behaviour in idle.BEHAVIOURS:
            if behaviour.pose is None:
                continue
            for step in range(0, 101):
                for key, value in behaviour.pose(step / 100).items():
                    assert abs(value) <= limits[key], \
                        f"{behaviour.name} drives {key} to {value}"

    def test_a_finished_behaviour_hands_the_face_back_at_rest(self):
        # This is what stops a behaviour's last pose from becoming the new
        # resting face: the scheduler returns NEUTRAL itself once the
        # behaviour's own duration is spent, rather than trusting the
        # behaviour to end where it started. `_track` in particular ends at
        # one extreme of its swing on purpose.
        scheduler = idle.Scheduler()
        scheduler.update(10.0, idle=True)
        behaviour = scheduler.current
        assert behaviour is not None
        assert scheduler.update(10.0 + behaviour.seconds + 0.01, idle=True) \
            == idle.NEUTRAL

    def test_rarer_behaviours_carry_longer_cooldowns(self):
        # A glance every few seconds reads as alive; a yawn every few seconds
        # reads as broken.
        glance = next(b for b in idle.BEHAVIOURS if b.name == "glance")
        yawn = next(b for b in idle.BEHAVIOURS if b.name == "yawn")
        assert glance.weight > yawn.weight
        assert glance.cooldown < yawn.cooldown

    def test_startle_is_not_in_the_scheduled_pool(self):
        # It fires on an actual noise in the room — the only idle here that
        # is a reaction rather than a performance.
        assert idle.STARTLE not in idle.BEHAVIOURS
        assert idle.STARTLE.weight == 0


class TestTheScheduler:
    def test_she_is_at_rest_the_moment_someone_speaks_to_her(self):
        scheduler = idle.Scheduler()
        scheduler.update(0.0, idle=True)
        assert scheduler.update(1.0, idle=False) == idle.NEUTRAL
        assert scheduler.current is None

    def test_being_addressed_mid_behaviour_cancels_it(self):
        scheduler = idle.Scheduler()
        scheduler.update(10.0, idle=True)
        assert scheduler.current is not None
        scheduler.update(10.5, idle=False)
        assert scheduler.current is None
        assert scheduler.bug is None

    def test_nothing_happens_before_the_first_scheduled_moment(self):
        scheduler = idle.Scheduler()
        assert scheduler.update(0.0, idle=True) == idle.NEUTRAL
        assert scheduler.current is None

    def test_a_behaviour_starts_once_its_moment_arrives(self):
        scheduler = idle.Scheduler()
        scheduler.update(scheduler.next_at, idle=True)
        assert scheduler.current is not None

    def test_a_behaviour_runs_to_its_own_length_and_then_stops(self):
        scheduler = idle.Scheduler()
        scheduler.update(10.0, idle=True)
        behaviour = scheduler.current
        assert behaviour is not None
        scheduler.update(10.0 + behaviour.seconds * 0.5, idle=True)
        assert scheduler.current is behaviour
        scheduler.update(10.0 + behaviour.seconds + 0.01, idle=True)
        assert scheduler.current is None

    def test_the_same_behaviour_does_not_repeat_immediately(self):
        # Three-deep history plus per-behaviour cooldowns. Two glances in a
        # row is the single thing that most makes it read as a loop.
        scheduler = idle.Scheduler()
        now = 10.0
        seen = []
        for _ in range(4):
            scheduler.update(now, idle=True)
            if scheduler.current is None:
                now += 2.0
                continue
            seen.append(scheduler.current.name)
            now += scheduler.current.seconds + 0.01
            scheduler.update(now, idle=True)
            now += 15.0
        for earlier, later in zip(seen, seen[1:], strict=False):
            assert earlier != later

    def test_a_bang_interrupts_whatever_she_was_doing(self):
        scheduler = idle.Scheduler()
        scheduler.update(10.0, idle=True)
        scheduler.startle(10.5)
        assert scheduler.current is idle.STARTLE
        assert scheduler.bug is None

    def test_a_second_bang_inside_the_cooldown_is_ignored(self):
        # Otherwise a rattling fan makes her flinch fifty times a second.
        scheduler = idle.Scheduler()
        scheduler.startle(10.0)
        scheduler.current = None
        scheduler.startle(10.1)
        assert scheduler.current is None

    def test_every_frame_returns_a_complete_pose(self):
        # The renderer indexes these keys directly, so a partial dict is a
        # KeyError inside the draw loop — a dark panel, at 3am.
        scheduler = idle.Scheduler()
        now = 0.0
        for _ in range(200):
            pose = scheduler.update(now, idle=True)
            assert set(pose) == set(idle.NEUTRAL)
            now += 0.4


def run_bug(seconds=16.0, steps=400):
    """Drive one whole visit, returning the speck and everything it posed.

    It is a simulation now rather than a curve, so `at` reports where the
    thing has actually got to and only means anything once the clock has been
    turned. Nothing here may sample it at a bare `p`.
    """
    speck = idle.Bug(seconds)
    poses = [speck.pose(step / steps) for step in range(steps + 1)]
    return speck, poses


class TestTheBug:
    def test_it_comes_in_from_one_side_and_leaves_by_the_other(self):
        speck = idle.Bug(16.0)
        entry = speck.at(0.0)[0]
        for step in range(401):
            speck.pose(step / 400)
        assert entry * speck.at(1.0)[0] < 0      # opposite signs: it crossed

    def test_it_fades_in_and_out_rather_than_appearing(self):
        speck = idle.Bug(16.0)
        assert speck.at(0.0)[2] == pytest.approx(0.0, abs=1e-6)
        assert speck.at(0.5)[2] == pytest.approx(1.0)
        assert speck.at(1.0)[2] == pytest.approx(0.0, abs=1e-6)

    def test_presence_is_never_negative(self):
        speck = idle.Bug(16.0)
        for step in range(101):
            assert speck.at(step / 100)[2] >= 0.0

    def test_she_squints_when_it_gets_close_to_her_face(self):
        # Reacting to the thing is what sells it; the speck alone is a dot.
        speck = idle.Bug(16.0)
        lids = [speck.pose(step / 200).get("lid_l", 1.0)
                for step in range(1, 200)]
        assert min(lids) < 0.95

    def test_an_absent_speck_moves_nothing(self):
        # It no longer poses literally nothing — dx and dy report where she
        # is standing, which is the middle before anything has happened and
        # has to keep being reported afterwards or she teleports home as the
        # speck fades. Everything else must be neutral.
        speck = idle.Bug(16.0)
        pose = speck.pose(0.0)
        for key, value in pose.items():
            assert value == pytest.approx(idle.NEUTRAL[key], abs=1e-9), key

    def test_the_scheduler_reports_no_speck_when_there_is_none(self):
        scheduler = idle.Scheduler()
        assert scheduler.bug_at(0.0) is None


class TestSheRunsFromIt:
    """The chase, which is a simulation rather than an animation.

    She is not playing a scripted dodge: she is being pushed away from a thing
    that follows her, inside walls she cannot cross. Everything below is a
    property of that, which is why no two visits look the same.
    """

    def trace(self, seconds=16.0, steps=400):
        """One whole visit, with both bodies recorded per frame."""
        speck = idle.Bug(seconds)
        rows = []
        for step in range(steps + 1):
            pose = speck.pose(step / steps)
            rows.append((speck.x, speck.y, speck.hx, speck.hy,
                         speck.near(), speck.cornered(), pose,
                         speck.vx, speck.vy))
        return speck, rows

    def test_she_actually_leaves_the_middle(self):
        # The whole complaint: she used to watch it go past without moving.
        _, rows = self.trace()
        assert max(abs(r[2]) for r in rows) > 0.25

    def test_she_uses_the_panel_rather_than_flinching(self):
        _, rows = self.trace()
        travelled = max(abs(r[2]) for r in rows) / idle.FLEE_REACH_X
        assert travelled > 0.5, "she barely moved; a lean is not a chase"

    def test_her_running_is_what_makes_the_gap(self):
        """The one statement that is true of evasion and of nothing else.

        Two readings that seemed obvious are both wrong. "It ends up on the
        far side of her" is false exactly when the pursuit works, because a
        pursuer follows. "She travels along the line from it to her" is false
        too, now that she swerves round it rather than backing away in a
        straight line — most of that motion is sideways by design.

        What survives: the closest it ever gets to *her* must be further than
        the closest it ever gets to the middle of the panel. If she had stood
        still, the second number is the gap she would have had.
        """
        _, rows = self.trace()
        to_centre = min(math.hypot(bx, by) for bx, by, *_ in rows)
        to_her = min(math.hypot(hx - bx, hy - by)
                     for bx, by, hx, hy, *_ in rows)
        assert to_her > to_centre + 0.05, \
            "it got as close as it would have to a face that never moved"

    def test_it_never_actually_catches_her(self):
        _, rows = self.trace()
        assert min(math.hypot(hx - bx, hy - by)
                   for bx, by, hx, hy, *_ in rows) > 0.2

    def test_she_keeps_moving_rather_than_hiding(self):
        """Measured as distance travelled, not as where she happens to be.

        "Is she near an edge" turned out to be the wrong question. While
        something is actively pushing her the equilibrium between that push
        and the walls genuinely sits near the limit, so she is *supposed* to
        spend a chase out there — that is what using the panel looks like.
        Hiding is not being near an edge, it is not moving.
        """
        _, rows = self.trace()
        travelled = sum(
            math.hypot(b[2] - a[2], b[3] - a[3])
            for a, b in zip(rows, rows[1:], strict=False))
        assert travelled > 3.0, "she barely moved; a flinch is not a chase"

    def test_she_does_not_spend_it_all_on_one_side(self):
        _, rows = self.trace()
        assert max(r[2] for r in rows) > 0.15
        assert min(r[2] for r in rows) < -0.15

    def test_she_never_leaves_the_panel(self):
        # A face half off the edge reads as broken rather than frightened.
        # The soft walls give the feel; _pen guarantees the promise.
        _, rows = self.trace()
        assert max(abs(r[2]) for r in rows) <= idle.FLEE_LIMIT_X + 1e-9
        assert max(abs(r[3]) for r in rows) <= idle.FLEE_LIMIT_Y + 1e-9

    def test_she_is_back_in_the_middle_before_it_ends(self):
        """Otherwise the last frame of the chase is a teleport.

        The scheduler drops the behaviour the moment progress passes one and
        her displacement simply stops being applied. Being anywhere but the
        middle at that point puts her back in the centre between two frames,
        which is the one thing a chase must not end with.
        """
        _, rows = self.trace()
        assert abs(rows[-1][2]) < 0.08 and abs(rows[-1][3]) < 0.08

    def test_it_hunts_her_rather_than_crossing_past(self):
        # Without the pursuit term she is a bystander and it is a screensaver.
        _, rows = self.trace()
        assert max(r[4] for r in rows) > 0.35, "it never got near her"

    def test_being_cornered_is_reachable(self):
        # The one thing the walls buy that a clamped range would not.
        _, rows = self.trace()
        assert max(r[5] for r in rows) > 0.6

    def test_she_shakes_hardest_with_her_back_to_the_wall(self):
        _, rows = self.trace()
        worst = max(rows, key=lambda r: r[6]["shake"])
        assert worst[5] > 0.4, "the tremble peaked in open space"


class TestCuriosityThenAlarm:
    """Squint first, wide second, and in that order.

    You narrow your eyes to resolve something small and far away — that is
    curiosity, not fear — and they snap open when you work out what you are
    looking at. The changeover is the beat the whole idle turns on, and
    getting it backwards is what made the old version read as mild irritation
    at something on the far side of a window.
    """

    def lids(self, steps=400):
        speck = idle.Bug(16.0)
        return [speck.pose(step / steps)["lid_l"] for step in range(steps + 1)]

    def test_she_peers_at_it_first(self):
        assert min(self.lids()) < 0.93, "she never focused on it"

    def test_and_her_eyes_go_wide_when_it_arrives(self):
        assert max(self.lids()) > 1.15, "she was never alarmed by it"

    def test_the_squint_comes_before_the_stare(self):
        """The *first* squint before the *first* stare, not the deepest.

        A chase is not one approach. It closes, she bolts, it comes back —
        and every retreat drags the distance back through the focus band, so
        she peers at it again on the way out. The deepest squint of a whole
        visit is usually one of those later passes, which says nothing about
        the order the two readings arrive in.
        """
        lids = self.lids()
        squinted = next(i for i, lid in enumerate(lids) if lid < 0.93)
        stared = next(i for i, lid in enumerate(lids) if lid > 1.15)
        assert squinted < stared, "she panicked and then got curious about it"

    def test_she_stops_peering_once_she_knows_what_it_is(self):
        """Focus is a band at middle distance, not a slope.

        It has to fade back out as recognition arrives, or the two readings
        fight and the lids sit somewhere meaningless in between.

        Measured against near() *times presence*, which is what the pose
        actually reacts to. near() alone stays high while the speck dissolves,
        so a test using it catches her peering at something that is on its way
        out — correct behaviour, read as a failure, and only on the seeds
        where the fade happened to line up.
        """
        speck = idle.Bug(16.0)
        worst = 0.0
        for step in range(401):
            p = step / 400
            pose = speck.pose(p)
            if speck.near() * speck.at(p)[2] > 0.75:
                worst = min(worst, pose["lid_l"] - 1.0)
        assert worst > -0.05, "she was still squinting with it on her face"

    def test_nothing_on_the_panel_means_nothing_on_her_face(self):
        speck = idle.Bug(16.0)
        assert speck.pose(0.0)["lid_l"] == pytest.approx(1.0)
