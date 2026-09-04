"""Fixtures, and one hard guarantee: no test touches the real settings.

`~/.config/eve/env` holds an Anthropic API key on any machine that has ever
run this assistant, and `memory.json` next to it holds whatever its owner told
it to remember. A test that read either would be reading someone's private
data, and a test that *wrote* either would corrupt it.

So the redirect is autouse rather than opt-in. Every test in this suite runs
with EVE_CONFIG_DIR pointing somewhere disposable, and the modules that
captured the real path at import time are re-pointed at the same place.
"""
from __future__ import annotations

import os

# Before `eve` is imported, not after, and this line is load-bearing.
# Importing the package applies the settings in ~/.config/eve/env so that the
# module constants computed at import — speech._HANG_S, tts.KOKORO_DIR,
# log.DEBUG — reflect them. Left alone, a test run on a machine that has ever
# run this assistant would read that person's real file and quietly take their
# microphone, their model and their debug flag into the suite. Pointing the
# directory at a path that cannot exist makes the loader a no-op, which is the
# same thing CI does for the same reason.
os.environ["EVE_CONFIG_DIR"] = "/nonexistent/eve-config-for-tests"

import pytest  # noqa: E402  (must follow the line above)

from eve import config, memory, speech, watch  # noqa: E402  (ditto)


@pytest.fixture(autouse=True)
def isolated_settings(tmp_path, monkeypatch):
    """A throwaway ~/.config/eve for the duration of one test.

    Both halves are needed. The environment variable covers anything that
    resolves the directory lazily; the setattrs cover config.VOICE_ENV,
    memory.STORE and watch.STORE, which are module-level constants computed at
    import and so would otherwise still name the real files.

    Every new module that keeps state beside the env file belongs on that list.
    watch.STORE holds things somebody asked to be reminded about, which is the
    same category of private as the other two.
    """
    runtime_prefixes = (
        "VOICE_", "KOKORO_", "WHISPER_", "EVE_", "HEY_CLAUDE_", "BARKEEP_",
    )
    runtime_exact = {"ANTHROPIC_API_KEY"}
    original_runtime = {
        key: value for key, value in os.environ.items()
        if key in runtime_exact or key.startswith(runtime_prefixes)
    }

    directory = tmp_path / "eve-config"
    directory.mkdir()
    monkeypatch.setenv("EVE_CONFIG_DIR", str(directory))
    monkeypatch.delenv("HEY_CLAUDE_CONFIG", raising=False)
    monkeypatch.setattr(config, "CONFIG_DIR", directory)
    monkeypatch.setattr(config, "VOICE_ENV", directory / "env")
    monkeypatch.setattr(memory, "STORE", directory / "memory.json")
    monkeypatch.setattr(watch, "STORE", directory / "reminders.json")
    # A microphone that is not this machine's. VOICE_MIC now defaults to empty
    # and speech.mic_device() discovers one by running `arecord -l`, which a
    # test must never do: it would make the suite depend on the hardware of
    # whoever is running it, and it fires inside the very calls the audio
    # tests replace subprocess for.
    monkeypatch.setattr(config, "MIC_DEVICE", "plughw:CARD=TESTMIC,DEV=0")
    monkeypatch.setattr(speech, "_mic_device", None)
    yield directory

    # config.load_settings() writes directly to os.environ, outside
    # monkeypatch's bookkeeping. Restore those additions explicitly so a
    # setting exercised late in one test cannot alter a subprocess in the
    # next test; the suite must be independent of collection order.
    for key in list(os.environ):
        if key in runtime_exact or key.startswith(runtime_prefixes):
            os.environ.pop(key)
    os.environ.update(original_runtime)


@pytest.fixture
def settings_file(isolated_settings):
    """Write the 0600 env file the way install.sh does."""

    def write(**values: str) -> None:
        body = "".join(f"{key}={value}\n" for key, value in values.items())
        path = isolated_settings / "env"
        path.write_text(body)
        path.chmod(0o600)

    return write
