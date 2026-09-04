"""Transcript logging is opt-in and distinct from documented memory."""
from __future__ import annotations

from eve import log


class TestContentIsSuppressedByDefault:
    def test_a_transcript_is_not_printed_when_content_is_off(
            self, monkeypatch, capsys):
        monkeypatch.setattr(log, "SHOW_CONTENT", False)
        log.content("what time is my flight to Tallinn")
        assert capsys.readouterr().err == ""

    def test_a_tagged_transcript_is_suppressed_too(self, monkeypatch, capsys):
        monkeypatch.setattr(log, "SHOW_CONTENT", False)
        log.spoken("heard", "my card number is 4111 1111 1111 1111")
        captured = capsys.readouterr().err
        assert captured == ""

    def test_timings_and_costs_are_always_logged(self, monkeypatch, capsys):
        # These describe the machine rather than the people in front of it,
        # which is what makes them safe to keep: a turn that cost half a cent
        # and took three seconds is diagnosable without knowing the question.
        monkeypatch.setattr(log, "SHOW_CONTENT", False)
        log.status("turn: 1.37s, 412 in / 88 out, $0.0006")
        assert "1.37s" in capsys.readouterr().err


class TestContentWhenItIsAskedFor:
    def test_content_prints_once_it_is_enabled(self, monkeypatch, capsys):
        monkeypatch.setattr(log, "SHOW_CONTENT", True)
        log.content("what time is it")
        assert "what time is it" in capsys.readouterr().err

    def test_a_tagged_transcript_says_why_it_is_being_shown(
            self, monkeypatch, capsys):
        monkeypatch.setattr(log, "SHOW_CONTENT", True)
        log.spoken("heard", "hello")
        captured = capsys.readouterr().err
        assert "heard" in captured and "hello" in captured


class TestTheBanner:
    def test_the_log_says_plainly_which_mode_is_running(self, monkeypatch):
        # One line at startup, so "is it recording me?" is answerable by
        # reading the journal rather than the source.
        monkeypatch.setattr(log, "DEBUG", True)
        assert "going to disk" in log.banner()
        monkeypatch.setattr(log, "DEBUG", False)
        banner = log.banner()
        assert "transcript logging off" in banner
        assert "memory.json" in banner


class TestHowTheDefaultIsDecided:
    def test_debug_is_off_unless_the_variable_is_deliberately_truthy(self):
        # The parse is a fixed set, not bool(str): VOICE_DEBUG=0 and
        # VOICE_DEBUG=false must both mean off, and both used to mean on.
        assert log._TRUE == ("1", "true", "yes", "on")
        for value in ("0", "false", "no", "off", ""):
            assert value.strip().lower() not in log._TRUE
