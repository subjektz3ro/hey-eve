"""The coverage gate, checked against itself.

Worth testing for the same reason the hygiene scanner is: a gate that passes
because it never looks at anything is indistinguishable from a gate that
passes because the code is covered.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import check_coverage  # noqa: E402  (needs the path line above)

REPO_ROOT = Path(__file__).resolve().parent.parent


def report(files: dict[str, float], statements: int = 100) -> dict:
    return {
        "files": {
            name: {"summary": {"num_statements": statements,
                               "percent_covered": covered}}
            for name, covered in files.items()
        },
        "totals": {"num_statements": statements * max(len(files), 1),
                   "percent_covered": (sum(files.values()) / len(files)
                                       if files else 0.0)},
    }


@pytest.fixture
def run(tmp_path, monkeypatch):
    def go(files, *, default=85.0, exceptions=None, statements=100):
        monkeypatch.setattr(
            check_coverage, "load_floors",
            lambda: (default, exceptions or {}))
        path = tmp_path / "coverage.json"
        path.write_text(json.dumps(report(files, statements)))
        return check_coverage.main(["check_coverage.py", str(path)])
    return go


class TestTheFloor:
    def test_a_file_above_the_default_passes(self, run):
        assert run({"eve/wake.py": 99.0}) == 0

    def test_a_file_below_the_default_fails(self, run):
        assert run({"eve/wake.py": 40.0}) == 1

    def test_a_file_exactly_on_its_floor_passes(self, run):
        # An epsilon, because percent_covered is a float and 85.0 computed
        # two ways is not always the same 85.0.
        assert run({"eve/wake.py": 85.0}) == 0

    def test_the_average_cannot_hide_a_bare_file(self, run):
        # This is the whole argument for per-file floors. The repo-wide
        # number here is a comfortable 90%, and one file has no tests at all.
        assert run({"eve/wake.py": 99.0, "eve/memory.py": 99.0,
                    "eve/tools.py": 99.0, "eve/main.py": 63.0}) == 1

    def test_a_file_with_no_statements_is_skipped(self, run):
        # `__init__.py` is one line of docstring; a 0% reading on it is
        # arithmetic, not a gap.
        assert run({"eve/__init__.py": 0.0}, statements=0) == 0


class TestTheDebtRegister:
    def test_an_exempted_file_is_held_to_its_own_number(self, run):
        assert run({"eve/scope.py": 20.0},
                   exceptions={"eve/scope.py": 17.0}) == 0

    def test_drifting_below_your_own_floor_still_fails(self, run):
        # The register records what a file achieves, not permission to rot.
        assert run({"eve/scope.py": 12.0},
                   exceptions={"eve/scope.py": 17.0}) == 1

    def test_a_comfortably_beaten_exception_is_flagged_but_does_not_fail(
            self, run, capsys):
        # Stale exceptions keep a file's real standard artificially low for
        # whoever changes it next.
        assert run({"eve/scope.py": 60.0},
                   exceptions={"eve/scope.py": 17.0}) == 0
        assert "raise it" in capsys.readouterr().out

    def test_an_exception_for_a_file_that_no_longer_exists_is_reported(
            self, run, capsys):
        assert run({"eve/wake.py": 99.0},
                   exceptions={"eve/deleted.py": 10.0}) == 0
        assert "not measured" in capsys.readouterr().out

    def test_the_summary_says_how_many_files_are_exempted(self, run, capsys):
        run({"eve/wake.py": 99.0}, exceptions={"eve/scope.py": 17.0})
        assert "1 files exempted" in capsys.readouterr().out


class TestItsOwnConfiguration:
    def test_the_floors_come_out_of_pyproject(self):
        default, exceptions = check_coverage.load_floors()
        assert default > 0
        assert exceptions

    def test_every_exempted_path_is_a_file_that_exists(self):
        # An exception naming a deleted file is dead configuration that
        # quietly stops covering anything.
        _, exceptions = check_coverage.load_floors()
        for name in exceptions:
            assert (REPO_ROOT / name).is_file(), f"{name} is not in the tree"

    def test_no_exception_is_set_above_the_default(self):
        # That would be a floor the file already has to meet anyway, written
        # in a place that implies it is exempt.
        default, exceptions = check_coverage.load_floors()
        for name, floor in exceptions.items():
            assert floor <= default, f"{name} does not need an exception"


class TestUsage:
    def test_it_refuses_to_run_without_a_report(self):
        assert check_coverage.main(["check_coverage.py"]) == 2

    def test_it_refuses_extra_arguments(self):
        assert check_coverage.main(["check_coverage.py", "a.json", "b.json"]) \
            == 2
