"""What her face actually draws, and what it costs to draw it.

The coverage register wrote the renderer off as needing a panel. It needs a
*file* — /dev/fb0 is opened by name, so pointing that name at a temporary one
turns "needs hardware" into a fixture, and every optimisation below is then
provable rather than hoped-at.

Two live bugs were sitting inside that exemption when it was lifted: an
animation phase that froze after eleven days of uptime, and a readout cache
that grew for as long as the process ran. Both are pinned here.
"""
from __future__ import annotations

import pytest

from eve import head, void

STATES = ["idle", "listening", "transcribing", "thinking", "searching",
          "synthesizing", "speaking", "engaged", "asleep"]


@pytest.fixture
def panel(tmp_path, monkeypatch):
    """A framebuffer that is a file, so the drawing path runs for real."""
    device = tmp_path / "fb0"
    device.write_bytes(b"\0" * (head.FB_WIDTH * head.FB_HEIGHT * 2))
    monkeypatch.setattr(void, "FB", str(device))
    monkeypatch.setattr(head, "FB", str(device))
    return device


class TestSheDraws:
    def test_something_reaches_the_panel(self, panel):
        face = void.Face()
        face.set_state("idle", 0.0, "READY")
        face._render(1.0, 0.02)
        assert panel.read_bytes() != b"\0" * len(panel.read_bytes())

    def test_the_same_inputs_give_the_same_frame(self, panel):
        # Not a hash of the picture — a guarantee that the renderer is a
        # function of its inputs, so every measurement here means something.
        import random
        frames = []
        for _ in range(2):
            random.seed(4)
            face = void.Face()
            face.set_state("listening", 0.4, "LISTENING")
            face._render(2.0, 0.02)
            frames.append(panel.read_bytes())
        assert frames[0] == frames[1]

    def test_a_vanished_panel_does_not_take_her_down(self, panel, monkeypatch):
        # The SPI display can be unbound at runtime. A face that raises here
        # takes the whole assistant with it, for a decoration.
        face = void.Face()
        monkeypatch.setattr(void, "FB", "/nonexistent/fb0")
        face._render(1.0, 0.02)

    @pytest.mark.parametrize("state", STATES)
    def test_every_state_renders(self, state, panel):
        face = void.Face()
        face.set_state(state, 0.5, state.upper())
        face._render(1.0, 0.02)
        face._render(1.02, 0.02)


class TestSheStillMovesAfterAMonth:
    """Animation phases used to be float32, and float32 runs out.

    `now` is seconds of uptime, and under NEP 50 a float32 array in the sum
    made the whole expression float32 — 24 bits of mantissa. Measured, the
    per-frame change reached exactly zero by day six for one phase and day
    eleven for another: the picture stopped moving, and a restart "fixed" it.
    """

    @pytest.mark.parametrize("days", [0, 1, 7, 12, 30, 365])
    def test_the_frame_still_changes_between_ticks(self, days, panel):
        uptime = days * 86400.0
        face = void.Face()
        face.set_state("idle", 0.0, "READY")
        face._render(uptime, 0.02)
        first = panel.read_bytes()
        face._render(uptime + 0.02, 0.02)
        assert panel.read_bytes() != first, \
            f"nothing moved at {days} days of uptime"

    def test_the_wrap_is_invisible_at_ordinary_uptimes(self):
        # sin is periodic, so the fix must change nothing for every uptime
        # the old code handled correctly.
        import math
        for seconds in (0.0, 0.5, 12.34, 600.0):
            assert math.isclose(math.sin(seconds % head._TAU),
                                math.sin(seconds), abs_tol=1e-9)


class TestTheReadoutCacheIsBounded:
    """It was keyed on the label text, and the label carries a countdown.

    "SPEAKING SOON 63%  ~4s" is a distinct key holding a 53KB float32 array,
    and speech.py feeds a new one several times a second. It grew all day,
    every day, on a machine with no MemoryMax and an intended uptime of months.
    """

    def test_it_stops_growing(self, panel):
        face = void.Face()
        for tick in range(head._LABEL_CACHE * 3):
            face._label(f"SPEAKING SOON {tick}%  ~{tick}s", (255, 255, 255))
        assert len(face._labels) <= head._LABEL_CACHE

    def test_the_bound_is_generous_enough_to_still_be_a_cache(self):
        # One entry per state plus whatever the countdown is showing.
        assert head._LABEL_CACHE >= 16

    def test_a_repeated_label_is_not_rasterised_twice(self, panel):
        face = void.Face()
        first = face._label("READY", (255, 255, 255))
        assert face._label("READY", (255, 255, 255)) is first

    def test_an_empty_label_is_not_cached(self, panel):
        face = void.Face()
        assert face._label("", (255, 255, 255)) is None
        assert not face._labels

    def test_the_oldest_goes_first(self, panel):
        face = void.Face()
        for tick in range(head._LABEL_CACHE + 1):
            face._label(f"LINE {tick}", (255, 255, 255))
        assert "LINE 0" not in face._labels
        assert f"LINE {head._LABEL_CACHE}" in face._labels


class TestSheYieldsWhileSynthesising:
    """Kokoro is the longest wait in a turn, and the face competes for cores.

    It yields during exactly that one state, which is the moment it costs
    nothing to: the only thing carrying information then is the progress bar,
    and speech.py updates that at 10Hz regardless.
    """

    def test_it_renders_less_often_while_synthesising(self):
        face = void.Face()
        face.set_state("thinking")
        busy = face._frame_period()
        face.set_state("synthesizing")
        assert face._frame_period() > busy

    @pytest.mark.parametrize("state", [s for s in STATES if s != "synthesizing"])
    def test_every_other_state_runs_at_full_rate(self, state):
        face = void.Face()
        face.set_state(state)
        assert face._frame_period() == pytest.approx(1.0 / head.FPS)

    def test_it_speeds_back_up_the_moment_she_speaks(self):
        face = void.Face()
        face.set_state("synthesizing")
        slow = face._frame_period()
        face.set_state("speaking")
        assert face._frame_period() < slow

    def test_the_reduced_rate_still_beats_the_progress_ticker(self):
        # speech.py's report() sleeps 0.1s, so anything at or above 10fps
        # draws every update it will ever be given.
        assert head.SYNTH_FPS >= 10


class TestThePixelFormat:
    def test_a_frame_is_the_panel_exactly(self, panel):
        face = void.Face()
        face.set_state("idle")
        face._render(1.0, 0.02)
        # RGB565: two bytes a pixel, and the driver takes the whole buffer.
        assert len(panel.read_bytes()) == head.FB_WIDTH * head.FB_HEIGHT * 2

    def test_nothing_overflows_a_channel(self, panel):
        # The frame is uint16 while it is being drawn, so a value over 255
        # would wrap into the next channel on the way down to RGB565.
        face = void.Face()
        face.set_state("speaking", 1.0)
        caught = {}
        face._write = lambda frame: caught.__setitem__("f", frame)
        face._render(1.0, 0.02)
        assert int(caught["f"].max()) <= 255
