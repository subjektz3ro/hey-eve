"""The calibration helper updates the secret-bearing env file safely."""

from __future__ import annotations

import importlib.util
import os
import stat
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "touch-probe.py"
SPEC = importlib.util.spec_from_file_location("eve_touch_probe", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
touch_probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(touch_probe)


def test_settings_replace_is_atomic_if_publish_fails(tmp_path, monkeypatch):
    target = tmp_path / "config" / "env"
    target.parent.mkdir()
    secret_key = "ANTHROPIC_API_" + "KEY"
    original = f"{secret_key}=private\nVOICE_TOUCH_SWAP=0\n"
    target.write_text(original)
    target.chmod(0o600)

    def refuse_replace(_source, _target):
        raise OSError("simulated interruption")

    monkeypatch.setattr(touch_probe.os, "replace", refuse_replace)
    with pytest.raises(OSError, match="simulated interruption"):
        touch_probe._write_settings(target, ["VOICE_TOUCH_SWAP=1"])

    assert target.read_text() == original
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    assert list(target.parent.glob(".env.*.tmp")) == []


def test_new_settings_file_is_never_published_group_readable(tmp_path):
    target = tmp_path / "config" / "env"
    old_umask = os.umask(0o022)
    try:
        touch_probe._write_settings(
            target,
            ["VOICE_TOUCH_SWAP=1", "VOICE_TOUCH_FLIP_X=0"],
        )
    finally:
        os.umask(old_umask)

    assert stat.S_IMODE(target.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    assert target.read_text() == (
        "VOICE_TOUCH_SWAP=1\nVOICE_TOUCH_FLIP_X=0\n"
    )
