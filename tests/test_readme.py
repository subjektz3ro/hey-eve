"""The README's checkable claims, checked.

A README is the one file nobody runs, so its numbers rot quietly and the rot
is invisible until somebody arrives, follows it, and finds it wrong on the
first command. This project already pins the same kind of claim in
tests/test_settings_file.py, where env.example is checked against the settings
the code actually reads.

Only claims a machine can settle. The prose about what she is like is not
testable and is not the point; a test count and a panel size are, and those
are exactly the ones that drift.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from eve import head

README = Path(__file__).resolve().parent.parent / "README.md"


@pytest.fixture(scope="module")
def readme() -> str:
    return README.read_text()


class TestTheNumbersItQuotes:
    def test_the_test_command_is_present_without_a_stale_count(self, readme):
        assert "uv run pytest -q" in readme
        assert re.search(r"uv run pytest[^\n]*#\s*[\d,]+ tests", readme) is None

    def test_the_panel_size_is_the_one_she_draws(self, readme):
        # Quoted twice, once as a fact and once as a constraint on porting.
        stated = set(re.findall(r"(\d{3})×(\d{3})", readme))
        assert stated, "the README no longer states a panel size"
        for width, height in stated:
            assert (int(width), int(height)) == (head.FB_WIDTH, head.FB_HEIGHT)

    def test_the_framebuffer_it_names_is_the_one_she_opens(self, readme):
        assert head.FB in readme


class TestTheCommandsItGives:
    """Every command in a fenced block should be one that exists.

    Not run — several want an API key, a microphone or 356MB of weights. What
    is checked is that the *file* each one invokes is really in the repository,
    which is the failure that actually happens: a script gets renamed and the
    README keeps pointing at where it used to be.
    """

    def scripts_mentioned(self, readme: str) -> set[str]:
        found = set()
        for block in re.findall(r"```bash\n(.*?)```", readme, re.S):
            found |= set(re.findall(r"(?:\./)?((?:deploy|scripts)/[\w.-]+)", block))
        return found

    def test_every_script_it_tells_you_to_run_exists(self, readme):
        root = README.parent
        for script in self.scripts_mentioned(readme):
            assert (root / script).is_file(), f"README runs a missing {script}"

    def test_it_does_tell_you_to_run_some(self, readme):
        # Guards the check above against silently matching nothing.
        assert len(self.scripts_mentioned(readme)) >= 2

    def test_every_module_it_names_is_importable(self, readme):
        import importlib
        for module in set(re.findall(r"`(eve/[a-z_]+)\.py`", readme)):
            importlib.import_module(module.replace("/", "."))


class TestTheFilesItLinksTo:
    def test_every_relative_link_resolves(self, readme):
        root = README.parent
        for target in re.findall(r"\]\((?!https?:)([^)#]+)\)", readme):
            assert (root / target).exists(), f"README links to a missing {target}"

    def test_the_animations_are_there(self, readme):
        root = README.parent
        shown = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", readme)
        assert shown, "the README shows no animations"
        for image in shown:
            assert (root / image).is_file(), f"README shows a missing {image}"
