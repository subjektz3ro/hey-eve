"""Settings, and the one property that makes it safe to keep them in git.

Nothing in this repository holds a credential. A key read from the supported
0600 file is deliberately never exported — which matters because this process
spawns `aplay` and `arecord` on almost every turn, and an exported key would be
sitting in the environment of both.
"""
from __future__ import annotations

import os

from eve import config


def test_fresh_barkeep_url_matches_its_loopback_http_default():
    assert config.BARKEEP_URL == "http://127.0.0.1:8080"


class TestReadingSecrets:
    def test_a_key_comes_out_of_the_settings_file(self, settings_file):
        settings_file(ANTHROPIC_API_KEY="sk-ant-notarealkey")
        assert config.secret("ANTHROPIC_API_KEY") == "sk-ant-notarealkey"

    def test_the_environment_beats_the_file(self, settings_file, monkeypatch):
        # This is what makes `VOICE_DEBUG=1 uv run python -m eve.main` work
        # without editing anything, and what lets a test override a value the
        # owner has set for real.
        settings_file(VOICE_MODEL="claude-haiku-4-5")
        monkeypatch.setenv("VOICE_MODEL", "claude-sonnet-5")
        assert config.secret("VOICE_MODEL") == "claude-sonnet-5"

    def test_reading_a_secret_does_not_put_it_in_the_environment(
            self, settings_file):
        # The whole point of parsing on demand. If this ever regresses, the
        # key reaches every subprocess the assistant starts.
        settings_file(ANTHROPIC_API_KEY="sk-ant-notarealkey")
        before = dict(os.environ)
        config.secret("ANTHROPIC_API_KEY")
        assert os.environ["PATH"] == before["PATH"]
        assert "ANTHROPIC_API_KEY" not in os.environ

    def test_quotes_around_a_value_are_stripped(self, settings_file):
        # People copy these out of shell scripts, where the quotes are
        # syntax. Here the file is data, so they are not.
        settings_file(BARKEEP_TOKEN='"quoted-token"')
        assert config.secret("BARKEEP_TOKEN") == "quoted-token"

    def test_surrounding_whitespace_is_ignored(self, settings_file):
        settings_file(VOICE_FACE="  void  ")
        assert config.secret("VOICE_FACE") == "void"

    def test_a_missing_file_is_an_empty_answer_not_an_error(self):
        # A fresh clone has no settings file. Reading one must not raise, or
        # the assistant cannot get far enough to say what is wrong.
        assert config.secret("ANTHROPIC_API_KEY") == ""

    def test_an_unknown_key_is_empty(self, settings_file):
        settings_file(ANTHROPIC_API_KEY="sk-ant-notarealkey")
        assert config.secret("NOT_A_SETTING") == ""

    def test_a_comment_line_is_not_mistaken_for_a_key(self, isolated_settings):
        (isolated_settings / "env").write_text(
            "# ANTHROPIC_API_KEY=sk-ant-commented-out\n"
            "VOICE_PERSONA=plain\n"
        )
        assert config.secret("ANTHROPIC_API_KEY") == ""
        assert config.secret("VOICE_PERSONA") == "plain"

    def test_a_value_containing_an_equals_sign_survives_intact(
            self, isolated_settings):
        # Parsed with partition, not split, so a base64 token keeps its
        # padding instead of being cut at the first '='.
        (isolated_settings / "env").write_text("BARKEEP_TOKEN=abc==def=\n")
        assert config.secret("BARKEEP_TOKEN") == "abc==def="


class TestPresentation:
    """It used to choose a face and a voice together.

    Picking one without the other is how you end up with a masculine head
    speaking in Emma's voice, so they were one setting. There is one face now,
    so it chooses the voice — and the mismatch it guarded against cannot
    happen any more.
    """

    def test_each_presentation_names_a_voice_from_the_bank(self):
        assert config.VOICES["female"] == "bf_emma"
        assert config.VOICES["male"] == "am_michael"

    def test_an_unknown_presentation_falls_back_rather_than_failing(
            self, monkeypatch):
        monkeypatch.setattr(config, "PRESENT", "nonsense")
        assert config.voice() == config.VOICES["female"]

    def test_the_male_presentation_selects_its_voice(self, monkeypatch):
        monkeypatch.setattr(config, "PRESENT", "male")
        assert config.voice() == "am_michael"


class TestTheSystemPrompt:
    def test_brevity_is_stated_as_a_rule_not_a_preference(self):
        # It is a latency budget: Kokoro makes audio more slowly than the
        # audio plays, so length is the single thing that decides whether
        # this feels like talking to someone.
        assert "one or two sentences" in config.SYSTEM_PROMPT

    def test_it_forbids_markdown_because_it_is_read_aloud(self):
        assert "Never use markdown" in config.SYSTEM_PROMPT

    def test_the_persona_rides_on_top_of_the_base_instructions(
            self, monkeypatch):
        monkeypatch.setattr(config, "PERSONA", "glados")
        prompt = config.system_prompt()
        assert prompt.startswith(config.SYSTEM_PROMPT)
        assert "GLaDOS" in prompt

    def test_plain_adds_nothing_at_all(self, monkeypatch):
        monkeypatch.setattr(config, "PERSONA", "plain")
        assert config.system_prompt() == config.SYSTEM_PROMPT

    def test_an_unknown_persona_degrades_to_plain(self, monkeypatch):
        monkeypatch.setattr(config, "PERSONA", "shakespearean")
        assert config.system_prompt() == config.SYSTEM_PROMPT

    def test_it_does_not_invent_a_bar_without_the_integration(
            self, monkeypatch):
        monkeypatch.setattr(config, "PERSONA", "plain")
        monkeypatch.setattr(config, "secret", lambda name: "")
        assert "BUSY Bar" not in config.system_prompt()

    def test_it_names_the_bar_only_when_its_tool_is_available(
            self, monkeypatch):
        monkeypatch.setattr(config, "PERSONA", "plain")
        monkeypatch.setattr(
            config,
            "secret",
            lambda name: "token" if name == "BARKEEP_TOKEN" else "",
        )
        prompt = config.system_prompt()
        assert "BUSY Bar" in prompt
        assert "read-only" in prompt

    def test_the_persona_may_never_replace_a_correct_answer(self):
        # An assistant that is unhelpful in character is just unhelpful.
        assert "never replaces" in config.PERSONA_GLADOS
        assert "Drop the persona immediately" in config.PERSONA_GLADOS


class TestCostDials:
    def test_history_is_bounded_because_every_turn_resends_all_of_it(self):
        # Cost grows quadratically within a session otherwise.
        assert config.HISTORY_TURNS > 0

    def test_a_reply_is_capped_well_below_what_the_model_could_produce(self):
        # A backstop for when prompting fails to keep it brief. At roughly
        # 19.5 spoken characters a second, 300 tokens is already a minute.
        assert config.MAX_TOKENS <= 500
