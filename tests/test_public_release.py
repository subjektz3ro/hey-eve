"""Release metadata and CI promises that should not drift silently."""
from __future__ import annotations

import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text())
LOCK = tomllib.loads((ROOT / "uv.lock").read_text())
CI = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
DEPENDABOT = (ROOT / ".github" / "dependabot.yml").read_text()


def test_distribution_name_does_not_take_the_eve_pypi_project():
    assert PYPROJECT["project"]["name"] == "hey-eve"
    assert PYPROJECT["project"]["scripts"] == {"eve": "eve.main:main"}
    assert all("github.com/subjektz3ro/hey-eve" in url
               for url in PYPROJECT["project"]["urls"].values())


def test_python_range_matches_the_kokoro_runtime_contract():
    assert PYPROJECT["project"]["requires-python"] == ">=3.11,<3.14"
    assert LOCK["requires-python"] == ">=3.11, <3.14"


def test_build_backend_is_reproducible():
    assert PYPROJECT["build-system"]["requires"] == ["hatchling==1.32.0"]


def test_lock_names_the_installable_distribution():
    editable = [
        package for package in LOCK["package"]
        if package.get("source") == {"editable": "."}
    ]
    assert [package["name"] for package in editable] == ["hey-eve"]


def test_onnx_runtime_stays_below_the_not_yet_canaried_release():
    dependency = next(
        value for value in PYPROJECT["project"]["dependencies"]
        if value.startswith("onnxruntime")
    )
    assert dependency == "onnxruntime>=1.17,<1.29"

    project = next(
        package for package in LOCK["package"]
        if package.get("source") == {"editable": "."}
    )
    locked_requirement = next(
        requirement for requirement in project["metadata"]["requires-dist"]
        if requirement["name"] == "onnxruntime"
    )
    assert locked_requirement["specifier"] == ">=1.17,<1.29"

    runtime = next(
        package for package in LOCK["package"]
        if package["name"] == "onnxruntime"
    )
    major, minor, *_ = (int(part) for part in runtime["version"].split("."))
    assert (major, minor) < (1, 29)


def test_dependabot_holds_only_the_not_yet_canaried_onnx_range():
    uv_updates = DEPENDABOT.split("package-ecosystem: uv", 1)[1].split(
        "package-ecosystem: github-actions", 1
    )[0]
    assert "dependency-name: onnxruntime" in uv_updates
    assert 'versions: [">=1.29"]' in uv_updates


def test_eve_forces_the_telemetry_guard_before_onnx_runtime_imports():
    # A fresh interpreter makes the ordering observable. The import hook sees
    # the exact environment present at the first onnxruntime import, while a
    # pre-existing zero value proves bootstrap assigns rather than setdefaults.
    probe = r'''
import builtins
import os

real_import = builtins.__import__

def checked_import(name, *args, **kwargs):
    if name == "onnxruntime" or name.startswith("onnxruntime."):
        if os.environ.get("ORT_DISABLE_TELEMETRY") != "1":
            raise RuntimeError("ONNX Runtime imported before telemetry opt-out")
    return real_import(name, *args, **kwargs)

builtins.__import__ = checked_import
import eve.vad
assert os.environ["ORT_DISABLE_TELEMETRY"] == "1"
'''
    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=ROOT,
        env={
            **os.environ,
            "EVE_CONFIG_DIR": "/nonexistent/eve-onnx-privacy-test",
            "ORT_DISABLE_TELEMETRY": "0",
        },
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr


def test_actions_are_immutable_and_checkout_drops_its_credential():
    actions = re.findall(r"^\s*- uses: ([^#\s]+)(?:\s+#.*)?$", CI,
                         re.MULTILINE)
    assert actions
    assert all(re.search(r"@[0-9a-f]{40}$", action) for action in actions)
    checkouts = [action for action in actions if action.startswith("actions/checkout@")]
    assert CI.count("persist-credentials: false") == len(checkouts)


def test_ci_uses_the_supported_boundaries_and_pinned_tools():
    assert "ubuntu-latest" not in CI
    assert 'python: ["3.11", "3.13"]' in CI
    setup_uv_uses = CI.count("uses: astral-sh/setup-uv@")
    assert setup_uv_uses > 0
    assert CI.count('version: "0.12.5"') == setup_uv_uses
    test_job = CI.split("\n  test:\n", 1)[1].split("\n  lint:\n", 1)[0]
    assert 'uv run python -c "import kokoro_onnx, onnxruntime"' in test_job
    assert "uv export --locked --python ${{ matrix.python }}" in CI
    assert "uvx --python ${{ matrix.python }}" in CI
    assert "--from pip-audit==2.10.1 pip-audit" in CI


def test_release_safety_does_not_depend_on_installing_packages():
    job = CI.split("\n  release-safety:\n", 1)[1].split("\n  test:\n", 1)[0]
    assert "needs:" not in job
    assert "setup-uv" not in job
    assert "uv sync" not in job
    assert "python3 scripts/check_release_hygiene.py" in job


def test_managed_service_docs_share_the_systemd_floor():
    for relative in (
        "README.md",
        "deploy/README.md",
        "docs/hardware.md",
        "SUPPORT.md",
    ):
        assert "systemd 243 or newer" in (ROOT / relative).read_text()
