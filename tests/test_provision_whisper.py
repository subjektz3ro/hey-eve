"""Safety boundaries around the separately owned whisper.cpp checkout."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "provision-whisper.sh"


def _git(*args: str, cwd: Path) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


def _fake_runtime_manifest(tmp_path: Path, checkout: Path) -> tuple[Path, Path]:
    fake_uv = tmp_path / "uv"
    values = {
        "whisper-dir": str(checkout),
        "whisper-binary": str(checkout / "build/bin/whisper-cli"),
        "whisper-model": str(checkout / "models/ggml-base.en.bin"),
        "whisper-source-url": "https://github.com/ggml-org/whisper.cpp.git",
        "whisper-source-commit": "592feef04a1802b18cbeffd0fd0eb5d02570c2ec",
        "whisper-model-url": "https://example.invalid/model",
        "whisper-model-sha256": "0" * 64,
    }
    cases = "\n".join(
        f"  {key}) printf '%s\\n' '{value}' ;;" for key, value in values.items()
    )
    fake_uv.write_text(
        "#!/usr/bin/env bash\n"
        "last=${!#}\n"
        "case \"$last\" in\n"
        f"{cases}\n"
        "  *) exit 2 ;;\n"
        "esac\n"
    )
    fake_uv.chmod(0o755)
    tools = tmp_path / "tools"
    tools.mkdir()
    for name in ("cmake", "c++"):
        tool = tools / name
        tool.write_text("#!/bin/sh\nexit 99\n")
        tool.chmod(0o755)
    return fake_uv, tools


def _run_provision(tmp_path: Path, checkout: Path) -> subprocess.CompletedProcess[str]:
    fake_uv, tools = _fake_runtime_manifest(tmp_path, checkout)
    return subprocess.run(
        ["bash", str(SCRIPT)],
        cwd=REPO_ROOT,
        env={
            **os.environ,
            "UV_BIN": str(fake_uv),
            "HOME": str(tmp_path),
            "PATH": f"{tools}:{os.environ['PATH']}",
        },
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_a_clean_unrelated_git_checkout_is_never_repurposed(tmp_path):
    wrong = tmp_path / "valuable-project"
    wrong.mkdir()
    _git("init", "-q", cwd=wrong)
    _git("config", "user.email", "test" + "@" + "example.invalid", cwd=wrong)
    _git("config", "user.name", "Test", cwd=wrong)
    tracked = wrong / "keep.txt"
    tracked.write_text("do not replace me\n")
    _git("add", "keep.txt", cwd=wrong)
    _git("commit", "-qm", "fixture", cwd=wrong)
    original_head = _git("rev-parse", "HEAD", cwd=wrong)

    result = _run_provision(tmp_path, wrong)
    assert result.returncode != 0
    assert "not the official whisper.cpp checkout" in result.stderr
    assert "no checkout was changed" in result.stderr.lower()
    assert _git("rev-parse", "HEAD", cwd=wrong) == original_head
    assert tracked.read_text() == "do not replace me\n"


@pytest.mark.parametrize(
    "origin",
    (
        "https://mirror.invalid/ggml-org/whisper.cpp.git",
        "git" + "@" + "mirror.invalid:ggml-org/whisper.cpp.git",
    ),
)
@pytest.mark.parametrize("has_marker", (False, True))
def test_only_the_official_upstream_origin_is_reused(
    tmp_path: Path, origin: str, has_marker: bool
):
    checkout = tmp_path / "whisper.cpp"
    checkout.mkdir()
    _git("init", "-q", cwd=checkout)
    _git("remote", "add", "origin", origin, cwd=checkout)
    if has_marker:
        (checkout / ".eve-source-commit").write_text(
            "592feef04a1802b18cbeffd0fd0eb5d02570c2ec\n"
        )

    result = _run_provision(tmp_path, checkout)

    assert result.returncode != 0
    assert "not the official whisper.cpp checkout" in result.stderr
    assert origin in result.stderr
