"""Reading the panel's touchscreen.

The hardware has had this sense the whole time and nothing ever asked it a
question. What is pinned here is the reading, not the reacting: where a finger
is and whether it is down. What she *does* about it belongs with the rest of
her behaviour.

Two of these tests exist because of scars elsewhere in this project rather
than because of anything touch-specific. The device is found by name, because
event nodes renumber and the microphone's card doing exactly that after a
reboot left her permanently deaf while the panel still read READY. And the
axis ranges are asked of the driver rather than assumed, because 0-4095 is
right often enough to look correct and wrong often enough to be a bug.
"""
from __future__ import annotations

import math
import struct
from pathlib import Path

import pytest

from eve import touch

# Verbatim from the Pi, including the two devices that must not be chosen.
PROC_DEVICES = """\
I: Bus=0019 Vendor=0001 Product=0001 Version=0100
N: Name="pwr_button"
P: Phys=gpio-keys/input0
H: Handlers=kbd event0

I: Bus=0000 Vendor=0000 Product=0000 Version=0000
N: Name="ADS7846 Touchscreen"
P: Phys=spi0.1/input0
H: Handlers=mouse0 event1

I: Bus=0019 Vendor=0001 Product=0001 Version=0100
N: Name="gpio_ir_recv"
P: Phys=gpio_ir_recv/input0
H: Handlers=kbd event2
"""


def write_devices(tmp_path, monkeypatch, body=PROC_DEVICES):
    path = tmp_path / "devices"
    path.write_text(body)
    monkeypatch.setattr(touch, "_DEVICES", path)
    return path


class TestFindingIt:
    def test_it_is_found_by_name(self, tmp_path, monkeypatch):
        write_devices(tmp_path, monkeypatch)
        assert touch.find_device().name == "event1"

    def test_the_other_devices_are_not_mistaken_for_it(self, tmp_path,
                                                       monkeypatch):
        # A power button and an infra-red receiver sit either side of it in
        # the real file, and both have event handlers.
        write_devices(tmp_path, monkeypatch)
        assert touch.find_device().name not in ("event0", "event2")

    def test_a_machine_without_one_says_so_rather_than_raising(
            self, tmp_path, monkeypatch):
        # Every developer's laptop, and this module has to import there.
        write_devices(tmp_path, monkeypatch, "N: Name=\"nothing\"\nH: Handlers=kbd event0\n")
        assert touch.find_device() is None

    def test_a_missing_file_is_survived(self, tmp_path, monkeypatch):
        monkeypatch.setattr(touch, "_DEVICES", tmp_path / "absent")
        assert touch.find_device() is None

    def test_a_differently_named_touchscreen_is_still_found(self, tmp_path,
                                                            monkeypatch):
        # The controller name is this panel's; the generic word is the one
        # that has to keep working when something else is fitted.
        write_devices(tmp_path, monkeypatch,
                      'N: Name="Generic Capacitive TouchScreen"\n'
                      'H: Handlers=event5\n')
        assert touch.find_device().name == "event5"


class TestTheProtocol:
    def test_an_event_is_twenty_four_bytes(self):
        # If this ever changes the reader silently desynchronises rather than
        # failing, which is why the module asserts it at import too.
        assert touch._EVENT.size == 24

    def test_the_ioctl_number_is_the_one_the_kernel_wants(self):
        # EVIOCGABS(ABS_X) = _IOR('E', 0x40, struct input_absinfo).
        assert touch._eviocgabs(touch.ABS_X) == 0x80184540
        assert touch._eviocgabs(touch.ABS_Y) == 0x80184541

    def test_absinfo_is_six_ints(self):
        assert touch._ABSINFO.size == 24


class TestScalingRawCounts:
    @pytest.mark.parametrize("raw, want", [
        (0, -1.0), (2048, 0.0), (4096, 1.0),
    ])
    def test_the_range_maps_onto_the_panel(self, raw, want):
        assert touch._scale(raw, 0, 4096) == pytest.approx(want)

    def test_anything_outside_is_clamped(self):
        # A resistive panel reports nonsense on the way up and down.
        assert touch._scale(-500, 0, 4096) == -1.0
        assert touch._scale(9000, 0, 4096) == 1.0

    def test_a_degenerate_range_does_not_divide_by_zero(self):
        assert touch._scale(100, 500, 500) == 0.0

    def test_a_driver_range_that_is_not_zero_based(self):
        # ADS7846 calibration often leaves a dead band at both ends.
        assert touch._scale(200, 200, 3900) == pytest.approx(-1.0)
        assert touch._scale(3900, 200, 3900) == pytest.approx(1.0)


def events(*records) -> bytes:
    return b"".join(touch._EVENT.pack(0, 0, kind, code, value)
                    for kind, code, value in records)


class FakeStream:
    """An evdev node handing over a scripted byte stream."""

    def __init__(self, blob: bytes):
        self.blob = blob
        self.at = 0
        self.closed = False

    def read(self, n):
        chunk = self.blob[self.at:self.at + n]
        self.at += len(chunk)
        return chunk

    def close(self):
        self.closed = True

    def fileno(self):
        raise OSError("no real fd")      # so _axis_range takes its fallback


def drive(monkeypatch, blob, **settings):
    """Run the reader over a scripted stream and return what it settled on."""
    for name in ("SWAP_XY", "FLIP_X", "FLIP_Y"):
        monkeypatch.setattr(touch, name, settings.get(name.lower(), False))
    pad = touch.Touch(device=Path("/dev/input/event9"))
    monkeypatch.setattr(pad, "_open", lambda: FakeStream(blob))
    pad._run()
    return pad.latest()


class TestReadingAFinger:
    def test_a_touch_is_reported_where_it_happened(self, monkeypatch):
        x, y, down = drive(monkeypatch, events(
            (touch.EV_KEY, touch.BTN_TOUCH, 1),
            (touch.EV_ABS, touch.ABS_X, 4095),
            (touch.EV_ABS, touch.ABS_Y, 0),
            (touch.EV_SYN, 0, 0),
        ))
        assert down is True
        assert x == pytest.approx(1.0, abs=0.01)
        assert y == pytest.approx(-1.0, abs=0.01)

    def test_letting_go_is_reported(self, monkeypatch):
        _, _, down = drive(monkeypatch, events(
            (touch.EV_KEY, touch.BTN_TOUCH, 1),
            (touch.EV_SYN, 0, 0),
            (touch.EV_KEY, touch.BTN_TOUCH, 0),
            (touch.EV_SYN, 0, 0),
        ))
        assert down is False

    def test_nothing_is_committed_before_the_sync(self, monkeypatch):
        """A move arrives as an X, a Y, and then a report.

        Acting on the X alone puts the finger at the new column and the old
        row for one frame — a diagonal flick every time you drag, on a panel
        whose whole job is to be reacted to.
        """
        x, y, _ = drive(monkeypatch, events(
            (touch.EV_KEY, touch.BTN_TOUCH, 1),
            (touch.EV_ABS, touch.ABS_X, 2048),
            (touch.EV_ABS, touch.ABS_Y, 2048),
            (touch.EV_SYN, 0, 0),
            (touch.EV_ABS, touch.ABS_X, 4095),      # no sync after this
        ))
        assert x == pytest.approx(0.0, abs=0.01)
        assert y == pytest.approx(0.0, abs=0.01)

    def test_pressure_alone_does_not_move_her(self, monkeypatch):
        x, y, _ = drive(monkeypatch, events(
            (touch.EV_ABS, touch.ABS_PRESSURE, 900),
            (touch.EV_SYN, 0, 0),
        ))
        assert (x, y) == (pytest.approx(0.0, abs=0.01),
                          pytest.approx(0.0, abs=0.01))

    def test_a_truncated_record_ends_the_read_rather_than_desyncing(
            self, monkeypatch):
        blob = events((touch.EV_KEY, touch.BTN_TOUCH, 1),
                      (touch.EV_SYN, 0, 0)) + b"\x00\x03"
        assert drive(monkeypatch, blob)[2] is True

    def test_garbage_is_survived(self, monkeypatch):
        drive(monkeypatch, struct.pack("24s", b"not an input event"))


class TestTheAxesCanBeCorrected:
    """A resistive panel is wired to the glass, not to the framebuffer.

    Which way its X runs relative to the picture — and whether the two are
    swapped outright — is a property of how the thing was assembled. It
    cannot be derived, so scripts/touch-probe.py measures it and these three
    settings carry the answer.
    """

    TOP_RIGHT = events(
        (touch.EV_KEY, touch.BTN_TOUCH, 1),
        (touch.EV_ABS, touch.ABS_X, 4095),
        (touch.EV_ABS, touch.ABS_Y, 0),
        (touch.EV_SYN, 0, 0),
    )

    def test_untouched_it_reports_what_the_driver_said(self, monkeypatch):
        x, y, _ = drive(monkeypatch, self.TOP_RIGHT)
        assert (round(x), round(y)) == (1, -1)

    def test_swapping_exchanges_the_axes(self, monkeypatch):
        x, y, _ = drive(monkeypatch, self.TOP_RIGHT, swap_xy=True)
        assert (round(x), round(y)) == (-1, 1)

    def test_flipping_x_mirrors_it(self, monkeypatch):
        x, y, _ = drive(monkeypatch, self.TOP_RIGHT, flip_x=True)
        assert (round(x), round(y)) == (-1, -1)

    def test_flipping_y_mirrors_it(self, monkeypatch):
        x, y, _ = drive(monkeypatch, self.TOP_RIGHT, flip_y=True)
        assert (round(x), round(y)) == (1, 1)

    def test_swap_happens_before_the_flips(self, monkeypatch):
        # Order matters and only one of the two is right. Swapping first is
        # what makes the probe's three answers independent of each other.
        x, y, _ = drive(monkeypatch, self.TOP_RIGHT, swap_xy=True, flip_x=True)
        assert (round(x), round(y)) == (1, 1)


class TestItCannotTakeHerDown:
    def test_a_device_that_will_not_open_is_reported_once(self, monkeypatch,
                                                          capsys):
        pad = touch.Touch(device=Path("/dev/input/event9"))

        def refuse():
            raise PermissionError("not in the input group")

        monkeypatch.setattr(pad, "_open", refuse)
        pad._run()                       # must not raise
        assert "no touch" in capsys.readouterr().err

    def test_starting_without_a_device_does_nothing(self):
        pad = touch.Touch(device=None)
        assert pad.available is False
        pad.start()
        assert pad._thread is None

    def test_stopping_one_that_never_started_is_safe(self):
        touch.Touch(device=None).stop()

    def test_it_can_be_switched_off_entirely(self, monkeypatch, tmp_path):
        node = tmp_path / "event9"
        node.write_bytes(b"")
        monkeypatch.setattr(touch, "ENABLED", False)
        pad = touch.Touch(device=node)
        pad.start()
        assert pad._thread is None

    def test_the_reader_thread_cannot_hold_shutdown_open(self, monkeypatch,
                                                         tmp_path):
        node = tmp_path / "event9"
        node.write_bytes(b"")
        monkeypatch.setattr(touch, "ENABLED", True)
        pad = touch.Touch(device=node)
        monkeypatch.setattr(pad, "_open", lambda: FakeStream(b""))
        pad.start()
        assert pad._thread is not None and pad._thread.daemon
        pad.stop()

    def test_nothing_is_reported_before_anything_is_read(self):
        assert touch.Touch(device=None).latest() == (0.0, 0.0, False)


class TestSheReactsToIt:
    """A finger drives the vocabulary the speck already built.

    She looks at it, her eyes go wide as it closes, and she gets out of the
    way — the same integrator, because being chased by a bug and being poked
    are the same problem wearing different hats. What is missing is the
    curious half: a finger does not approach, so there is nothing to squint
    at and first contact gets a startle instead.
    """

    @pytest.fixture(autouse=True)
    def _fresh_startle(self, monkeypatch):
        # Behaviour instances are module-level and carry last_at between
        # tests, so a cooldown set by somebody else silently refuses the jump.
        from eve import idle
        monkeypatch.setattr(idle.STARTLE, "last_at", -999.0)

    def face(self, monkeypatch, at=None):
        from eve import void
        made = void.Face()
        monkeypatch.setattr(type(made), "_disturb",
                            lambda self, frame, now, unstable: frame)
        monkeypatch.setattr(made, "_write", lambda frame: None)
        reading = (at[0], at[1], True) if at else (0.0, 0.0, False)
        monkeypatch.setattr(made._touch, "latest", lambda: reading)
        made.set_state("idle")
        return made

    def run(self, made, seconds=2.0, fps=50):
        for step in range(int(seconds * fps)):
            made._render(step / fps, 1.0 / fps)

    def test_a_finger_on_her_moves_her(self, monkeypatch):
        made = self.face(monkeypatch, at=(0.0, 0.0))
        self.run(made)
        assert math.hypot(*made._flee) > 0.15, "she stood there and took it"

    def test_she_goes_the_other_way(self, monkeypatch):
        # Touched on the left, she should end up right of centre.
        made = self.face(monkeypatch, at=(-0.6, 0.0))
        self.run(made)
        assert made._flee[0] > 0.05

    def test_an_untouched_panel_leaves_her_where_she_is(self, monkeypatch):
        made = self.face(monkeypatch)
        self.run(made)
        assert math.hypot(*made._flee) < 0.01

    def test_she_comes_back_when_you_lift(self, monkeypatch):
        made = self.face(monkeypatch, at=(0.0, 0.0))
        self.run(made, seconds=1.5)
        assert math.hypot(*made._flee) > 0.1
        monkeypatch.setattr(made._touch, "latest", lambda: (0.0, 0.0, False))
        for step in range(200):
            made._render(2.0 + step / 50, 1.0 / 50)
        assert math.hypot(*made._flee) < 0.05, "she stayed where she was pushed"

    def test_first_contact_makes_her_jump(self, monkeypatch):
        from eve import idle
        made = self.face(monkeypatch, at=(0.0, 0.0))
        made._render(0.0, 0.02)
        assert made._idle.current is idle.STARTLE

    def test_it_only_jumps_once_per_press(self, monkeypatch):
        # Counted at the call, not read off STARTLE.last_at: the scheduler
        # also stamps last_at when a behaviour *finishes*, so that number
        # answers a different question than the one being asked.
        made = self.face(monkeypatch, at=(0.0, 0.0))
        jumps = []
        real = made._idle.startle
        monkeypatch.setattr(made._idle, "startle",
                            lambda now: jumps.append(now) or real(now))
        self.run(made, seconds=3.0)
        assert made._touched is True
        assert len(jumps) == 1, "holding a finger down is one event, not many"

    def test_she_never_leaves_the_panel(self, monkeypatch):
        made = self.face(monkeypatch, at=(0.0, 0.0))
        seen = 0.0
        for step in range(400):
            made._render(step / 50, 1.0 / 50)
            seen = max(seen, abs(made._flee[0]))
        from eve import idle
        assert seen <= idle.FLEE_LIMIT_X + 1e-9

    def test_a_panel_with_no_touchscreen_changes_nothing(self, monkeypatch):
        made = self.face(monkeypatch)
        assert made._touch.available is False
        self.run(made, seconds=0.5)          # must not raise
