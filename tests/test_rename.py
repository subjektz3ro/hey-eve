"""Surviving the rename from `hey-claude` to eve.

The rename changed every default path without moving the data that lived at
them. Nothing caught it, because the weights are only opened on the first
spoken reply: startup looked clean, and an hour later the first thing anyone
asked her to say put her into a crash loop.

The tests below are that outage, written down. `test_the_outage` is the whole
scenario end to end; the rest pin the two mechanisms that make it survivable.
"""
from __future__ import annotations

import importlib


from eve import config


class TestASettingKeepsItsOldName:
    def test_the_current_name_is_used(self, monkeypatch):
        monkeypatch.setenv("EVE_CONFIG_DIR", "/new")
        assert config.setting("EVE_CONFIG_DIR", "HEY_CLAUDE_CONFIG") == "/new"

    def test_the_former_name_still_answers(self, monkeypatch):
        # Nobody's env file has to be edited for a rename. This is the whole
        # reason the rename is safe to make at all.
        monkeypatch.delenv("EVE_CONFIG_DIR", raising=False)
        monkeypatch.setenv("HEY_CLAUDE_CONFIG", "/old")
        assert config.setting("EVE_CONFIG_DIR", "HEY_CLAUDE_CONFIG") == "/old"

    def test_the_current_name_wins_when_both_are_set(self, monkeypatch):
        monkeypatch.setenv("EVE_CONFIG_DIR", "/new")
        monkeypatch.setenv("HEY_CLAUDE_CONFIG", "/old")
        assert config.setting("EVE_CONFIG_DIR", "HEY_CLAUDE_CONFIG") == "/new"

    def test_an_empty_value_is_not_a_value(self, monkeypatch):
        # An env file line left blank by the installer must fall through to
        # the default rather than resolving paths under "".
        monkeypatch.setenv("EVE_CONFIG_DIR", "")
        monkeypatch.delenv("HEY_CLAUDE_CONFIG", raising=False)
        assert config.setting("EVE_CONFIG_DIR", "HEY_CLAUDE_CONFIG", "fallback") \
            == "fallback"

    def test_neither_set_gives_the_default(self, monkeypatch):
        monkeypatch.delenv("EVE_CONFIG_DIR", raising=False)
        monkeypatch.delenv("HEY_CLAUDE_CONFIG", raising=False)
        assert config.setting("EVE_CONFIG_DIR", "HEY_CLAUDE_CONFIG", "d") == "d"

    def test_a_setting_with_no_former_name_still_works(self):
        assert config.setting("NOT_SET_ANYWHERE", default="d") == "d"


class TestFindingDataUnderTheOldName:
    def test_a_fresh_install_resolves_to_the_current_name(self, tmp_path):
        # Nothing exists yet, so the answer is where things should be written.
        assert config.data_dir(tmp_path, "kokoro/kokoro-v1.0.onnx") \
            == tmp_path / "eve"

    def test_the_current_name_wins_when_it_has_the_data(self, tmp_path):
        for name in ("eve", "hey-claude"):
            weights = tmp_path / name / "kokoro"
            weights.mkdir(parents=True)
            (weights / "kokoro-v1.0.onnx").write_bytes(b"weights")
        assert config.data_dir(tmp_path, "kokoro/kokoro-v1.0.onnx") \
            == tmp_path / "eve"

    def test_a_pre_rename_install_is_found_where_it_actually_is(self,
                                                                tmp_path):
        # THE case. The new directory does not exist; the old one holds
        # 325MB. Pointing at the new one is what cost her voice.
        weights = tmp_path / "hey-claude" / "kokoro"
        weights.mkdir(parents=True)
        (weights / "kokoro-v1.0.onnx").write_bytes(b"weights")
        assert config.data_dir(tmp_path, "kokoro/kokoro-v1.0.onnx") \
            == tmp_path / "hey-claude"

    def test_an_empty_legacy_directory_is_not_mistaken_for_data(self,
                                                                tmp_path):
        # A leftover directory with nothing in it must not divert writes away
        # from the current name.
        (tmp_path / "hey-claude" / "kokoro").mkdir(parents=True)
        assert config.data_dir(tmp_path, "kokoro/kokoro-v1.0.onnx") \
            == tmp_path / "eve"

    def test_each_kind_of_data_is_resolved_independently(self, tmp_path):
        # Silero and Kokoro are fetched separately and can genuinely end up
        # in different places, so one being migrated must not drag the other.
        (tmp_path / "hey-claude" / "models").mkdir(parents=True)
        (tmp_path / "hey-claude" / "models" / "silero_vad.onnx").write_bytes(b"v")
        (tmp_path / "eve" / "kokoro").mkdir(parents=True)
        (tmp_path / "eve" / "kokoro" / "kokoro-v1.0.onnx").write_bytes(b"k")
        assert config.data_dir(tmp_path, "models/silero_vad.onnx") \
            == tmp_path / "hey-claude"
        assert config.data_dir(tmp_path, "kokoro/kokoro-v1.0.onnx") \
            == tmp_path / "eve"


class TestTheOutage:
    def test_a_pre_rename_install_keeps_its_voice(self, tmp_path, monkeypatch):
        """The whole failure, reproduced and then survived.

        Aug 14 15:37  weights fetched to ~/.local/share/hey-claude/
        Aug 15 00:03  the rename ships; the default path becomes .../eve
        Aug 15 01:08  first attempt to speak -> RuntimeError, crash loop

        Nothing was deleted and nothing was misconfigured. The data simply
        stopped being where the code had started looking.
        """
        share = tmp_path / "share"
        legacy = share / "hey-claude" / "kokoro"
        legacy.mkdir(parents=True)
        (legacy / "kokoro-v1.0.onnx").write_bytes(b"325MB, pretend")
        (legacy / "voices-v1.0.bin").write_bytes(b"28MB, pretend")
        assert not (share / "eve").exists()

        monkeypatch.setattr(config, "DATA_DIR", share)
        monkeypatch.delenv("KOKORO_DIR", raising=False)
        from eve import tts
        tts = importlib.reload(tts)
        try:
            assert tts.paths() is not None, \
                "she went mute on an install that still has all its weights"
            assert tts.KOKORO_DIR == legacy
        finally:
            importlib.reload(tts)

    def test_the_silero_detector_survives_it_too(self, tmp_path, monkeypatch):
        # Same rename, same stranding. A missing VAD is quieter — she falls
        # back to a loudness threshold and starts waking on door slams.
        share = tmp_path / "share"
        legacy = share / "hey-claude" / "models"
        legacy.mkdir(parents=True)
        (legacy / "silero_vad.onnx").write_bytes(b"2.3MB, pretend")

        monkeypatch.setattr(config, "DATA_DIR", share)
        monkeypatch.delenv("EVE_MODELS_DIR", raising=False)
        monkeypatch.delenv("HEY_CLAUDE_MODELS", raising=False)
        from eve import vad
        vad = importlib.reload(vad)
        try:
            assert vad.MODEL.exists()
        finally:
            importlib.reload(vad)


class TestNoStaleNamesRemain:
    def test_the_old_name_survives_only_where_it_is_meant_to(self):
        # Every remaining mention should be a deliberate compatibility path
        # or an explanation of one — never a live default.
        import subprocess
        from pathlib import Path

        root = Path(__file__).resolve().parent.parent
        hits = subprocess.run(
            ["git", "-C", str(root), "grep", "-lIE", "HEY_CLAUDE_"],
            capture_output=True, text=True,
        ).stdout.split()
        allowed = {
            "eve/config.py",          # the compatibility helper itself
            "eve/vad.py",             # passes the former name to it
            "deploy/install.sh",      # ditto
            "deploy/connect-speaker.sh",   # finds the env file the same way
            "scripts/fetch-models.sh",
            "env.example",            # documents that it still answers
            "tests/test_rename.py",   # this file
            "tests/conftest.py",      # clears it, so isolation is not fooled
            "tests/test_settings_file.py",  # checks the prefix is still settable
            "tests/test_install_contract.py",  # verifies legacy env is scrubbed
            "CHANGELOG.md",
        }
        assert set(hits) <= allowed, f"stale name leaked into {set(hits) - allowed}"

    def test_no_module_reads_the_former_name_directly(self):
        # It must go through config.setting, or the fallback is not uniform
        # and the next rename repeats the outage.
        import subprocess
        from pathlib import Path

        root = Path(__file__).resolve().parent.parent
        direct = subprocess.run(
            ["git", "-C", str(root), "grep", "-nE",
             r'environ(\.get\(|\[)"HEY_CLAUDE', "--", "eve"],
            capture_output=True, text=True,
        ).stdout.strip()
        assert direct == "", f"reads the former name outside config.setting:\n{direct}"
