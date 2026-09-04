"""Her face, and the head it is drawn on.

There were four faces once — an astrobot drawn as a point cloud, an IC head in
box characters, an oscilloscope, and the void. Only the void was ever fitted to
the machine, and the other three were a standing tax: every change to how she
behaves had to be made, or deliberately not made, in four places. They are
gone. eve/head.py is the half of the astrobot that was never about the
astrobot — everything she *does* — and eve/void.py is the only thing that
decides what any of it looks like.

That split is worth keeping even with one face on the other end of it, and the
first test here is what says so: behaviour has no drawing in it.
"""
from __future__ import annotations

import importlib

import pytest

from eve import head, void


class TestSheLoadsWithoutHardware:
    def test_the_module_imports_on_a_machine_with_no_framebuffer(self):
        # main.py imports this at the moment it fits the face, so a face that
        # only fails on import is a face that fails on the Pi, at 3am, with no
        # panel to show the traceback on.
        module = importlib.import_module("eve.void")
        assert hasattr(module, "Face")

    def test_she_rasterises_into_the_framebuffer_rather_than_the_console(self):
        # The console is root-only and this service is not root.
        assert head.FB == "/dev/fb0"
        assert void.FB == head.FB

    def test_the_face_is_a_subclass_and_not_a_second_implementation(self):
        assert issubclass(void.Face, head.Head)
        # Exported under a plain name, which is what let main.py construct it
        # identically back when there was a choice of them.
        assert void.Face is void.Void


class TestTheSplitIsReal:
    """Behaviour on one side, drawing on the other.

    With one face left this could quietly rot into a single module, and the
    reason not to is that the two halves answer different questions and change
    for different reasons — the chase and the blink are about what she is
    like, the lash and the glyph grid are about what she looks like.
    """

    def test_the_behaviour_half_draws_nothing(self):
        # It has no _render of its own: the subclass supplies it, and asking
        # the base for one is a programming error rather than a blank frame.
        with pytest.raises(NotImplementedError):
            head.Head()._render(0.0, 0.02)

    def test_the_behaviour_half_knows_nothing_about_glyphs(self):
        source = importlib.import_module("eve.head").__doc__ or ""
        assert "glyph" not in source.lower()
        for gone in ("CELL_W", "MASKS", "ZS", "LASH_RISE", "_GLYPHS"):
            assert not hasattr(head, gone), f"{gone} belongs to the drawing"

    def test_the_drawing_half_owns_no_state_machine(self):
        # set_state, the springs and the idle scheduler live once.
        for shared in ("set_state", "_blink", "_target", "_label", "_run"):
            assert getattr(void.Face, shared) is getattr(head.Head, shared), \
                f"{shared} was reimplemented on the drawing side"


class TestTheConsoleSwitch:
    """She moves the console off the panel before drawing on it.

    This needs real privileges, so it goes through `sudo -n chvt` and the unit
    deliberately does not set NoNewPrivileges — see the note in
    deploy/eve@.service. An earlier unit did set it, which blocked this
    outright and put two sudo errors in the journal on every start.
    """

    def test_starting_her_switches_the_console_first(self, monkeypatch):
        calls = []
        monkeypatch.setattr(head.subprocess, "run",
                            lambda *a, **k: calls.append((a[0], k)))
        face = void.Face()
        try:
            face.start()
        finally:
            face.stop()
        assert calls, "the console was never switched"
        argv, kwargs = calls[0]
        assert argv == ["sudo", "-n", "chvt", str(head.VT)]
        # check=False: a headless box has no VT to switch to, and that is not
        # a reason to refuse to start.
        assert kwargs.get("check") is False
