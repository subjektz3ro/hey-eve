"""The panel is a touchscreen, and until now nothing read it.

She is an object on a desk with a face on it, and the one thing a person
does with an object on a desk is put a finger on it. Everything she reacts
to so far arrives through the microphone; this is the other sense the
hardware already has and the software never asked about.

Read straight off the evdev character device rather than through a library.
An input event is twenty-four bytes — a timeval, a type, a code and a value —
and that is the whole protocol, so a dependency here would be all packaging
and no substance. It also keeps the reader in the same shape as everything
else that touches hardware in this project: a subprocess or a device node, a
thread, and a rule that it may never take the assistant down with it.

Three things are deliberate.

The device is found by *name*, not by /dev/input/event1. Event nodes renumber
— the microphone's card did exactly that after a reboot and left her
permanently deaf while showing READY, which is the most expensive bug this
project has had. A touchscreen that quietly stops working is cheaper than
that, but not free, and the fix is the same fix.

The ranges come from the driver over ioctl rather than being assumed. This is
an ADS7846, a resistive controller reporting raw ADC counts, and what those
counts run between is a property of the panel it is wired to. Guessing 0-4095
is right often enough to look correct and wrong often enough to be a bug.

And nothing here decides what a touch *means*. It reports where the finger is
in panel coordinates and whether it is down. What she does about it belongs
with the rest of her behaviour, in eve/idle.py, next to the thing that already
knows how to react to something approaching her face.
"""
from __future__ import annotations

import fcntl
import os
import struct
import threading
from pathlib import Path

from eve import log

# One evdev record: struct input_event. Two longs of timeval, then type, code
# and a signed value. Little-endian 64-bit, which is every machine this runs
# on; the size assert below is what catches it if that ever stops being true.
_EVENT = struct.Struct("llHHi")
assert _EVENT.size == 24

EV_SYN, EV_KEY, EV_ABS = 0x00, 0x01, 0x03
ABS_X, ABS_Y, ABS_PRESSURE = 0x00, 0x01, 0x18
BTN_TOUCH = 0x14A

# EVIOCGABS(code): _IOR('E', 0x40 + code, struct input_absinfo). The struct is
# six int32s — value, min, max, fuzz, flat, resolution — hence 24 in the size
# field. Spelled out rather than hidden behind a helper because getting an
# ioctl number wrong produces a confusing errno rather than an obvious one.
_ABSINFO = struct.Struct("6i")


def _eviocgabs(code: int) -> int:
    return (2 << 30) | (_ABSINFO.size << 16) | (ord("E") << 8) | (0x40 + code)


# Where the kernel lists what it has. Parsed rather than globbed because the
# name is the only stable handle on a device; see the module docstring.
_DEVICES = Path("/proc/bus/input/devices")
# Substrings that mark a device as the panel's touchscreen. ADS7846 is the
# controller on this one; the generic word covers anything else fitted later.
_WANTED = ("touchscreen", "ads7846")

# Set VOICE_TOUCH=0 to leave the panel alone entirely.
ENABLED = os.environ.get("VOICE_TOUCH", "1") not in ("0", "false", "no", "")
# Swap or invert the axes. A resistive panel is wired to the glass, not to the
# framebuffer, so which way its X runs relative to the picture is a property
# of the assembly and cannot be derived — scripts/touch-probe.py reports what
# this panel actually does, and these three carry the answer.
SWAP_XY = os.environ.get("VOICE_TOUCH_SWAP", "0") not in ("0", "false", "no", "")
FLIP_X = os.environ.get("VOICE_TOUCH_FLIP_X", "0") not in ("0", "false", "no", "")
FLIP_Y = os.environ.get("VOICE_TOUCH_FLIP_Y", "0") not in ("0", "false", "no", "")


def find_device() -> Path | None:
    """The touchscreen's event node, located by name.

    Returns None rather than raising when there is no touchscreen, which is
    the ordinary case on a machine with no panel and on every developer's
    laptop — this module has to import cleanly there.
    """
    try:
        blocks = _DEVICES.read_text().split("\n\n")
    except OSError:
        return None
    for block in blocks:
        lowered = block.lower()
        if not any(want in lowered for want in _WANTED):
            continue
        for line in block.splitlines():
            if not line.startswith("H: Handlers="):
                continue
            for handler in line.partition("=")[2].split():
                if handler.startswith("event"):
                    return Path("/dev/input") / handler
    return None


def _axis_range(stream, code: int, fallback: tuple[int, int]) -> tuple[int, int]:
    """What this driver says the axis runs between."""
    try:
        buffer = bytearray(_ABSINFO.size)
        fcntl.ioctl(stream, _eviocgabs(code), buffer)
        _, low, high, *_ = _ABSINFO.unpack(bytes(buffer))
        if high > low:
            return low, high
    except OSError:
        pass
    return fallback


class Touch:
    """Where the finger is, if there is one.

    One thread, one file, and a tuple anyone may read. Deliberately not a
    queue of events: nothing downstream cares about the history, only about
    where the finger is now, and a queue nobody drains is a leak.
    """

    def __init__(self, device: Path | None = None) -> None:
        self.device = device or find_device()
        self._x = 0.0            # -1 (left) .. 1 (right)
        self._y = 0.0            # -1 (top)  .. 1 (bottom)
        self._down = False
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._stream = None

    @property
    def available(self) -> bool:
        return self.device is not None and self.device.exists()

    def latest(self) -> tuple[float, float, bool]:
        """Panel coordinates in -1..1, and whether it is being touched."""
        with self._lock:
            return self._x, self._y, self._down

    def start(self) -> None:
        if self._thread is not None or not ENABLED or not self.available:
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        stream = self._stream
        if stream is not None:
            try:
                stream.close()          # unblocks the read
            except OSError:
                pass
        if self._thread is not None:
            self._thread.join(timeout=2)

    def _open(self):
        """The event stream. Its own method so a test can hand over a fake."""
        return open(self.device, "rb", buffering=0)

    def _run(self) -> None:
        try:
            self._stream = self._open()
        except OSError as exc:
            # Almost always the `input` group. Said once, at startup, because
            # a touchscreen that cannot be opened is a fact about the install
            # rather than an error in a loop.
            log.status(f"     (no touch: {self.device} would not open: {exc})")
            return
        stream = self._stream
        x_lo, x_hi = _axis_range(stream, ABS_X, (0, 4095))
        y_lo, y_hi = _axis_range(stream, ABS_Y, (0, 4095))
        log.status(f"     (touch: {self.device.name}, "
                   f"x {x_lo}-{x_hi}, y {y_lo}-{y_hi})")
        raw_x, raw_y, down = (x_lo + x_hi) // 2, (y_lo + y_hi) // 2, False
        try:
            while not self._stop.is_set():
                record = stream.read(_EVENT.size)
                if not record or len(record) < _EVENT.size:
                    break
                _, _, kind, code, value = _EVENT.unpack(record)
                if kind == EV_ABS:
                    if code == ABS_X:
                        raw_x = value
                    elif code == ABS_Y:
                        raw_y = value
                elif kind == EV_KEY and code == BTN_TOUCH:
                    down = bool(value)
                elif kind == EV_SYN:
                    # Commit on the report, not per axis. A move arrives as an
                    # X, a Y and a sync, and acting on the X alone puts the
                    # finger somewhere it never was for one frame.
                    self._commit(raw_x, raw_y, down, x_lo, x_hi, y_lo, y_hi)
        except (OSError, ValueError):
            pass                 # the device went away; she simply stops feeling
        finally:
            try:
                stream.close()
            except OSError:
                pass

    def _commit(self, raw_x: int, raw_y: int, down: bool,
                x_lo: int, x_hi: int, y_lo: int, y_hi: int) -> None:
        x = _scale(raw_x, x_lo, x_hi)
        y = _scale(raw_y, y_lo, y_hi)
        if SWAP_XY:
            x, y = y, x
        if FLIP_X:
            x = -x
        if FLIP_Y:
            y = -y
        with self._lock:
            self._x, self._y, self._down = x, y, down


def _scale(value: int, low: int, high: int) -> float:
    """Raw ADC counts to -1..1, clamped."""
    if high <= low:
        return 0.0
    return max(-1.0, min(1.0, (value - low) / (high - low) * 2.0 - 1.0))
