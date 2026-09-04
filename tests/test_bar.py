"""Asking Barkeep what the bar is doing.

The assistant never draws to the device itself — the bar's draw model is
last-writer-wins, so a second writer would fight whatever app is running.
It asks the supervisor instead, and turns the answer into a sentence.

No socket is opened anywhere in this file. The HTTP boundary is `bar._get`,
and every test replaces it.
"""
from __future__ import annotations

import ssl

import json
import urllib.error

import pytest

from eve import bar, tools


@pytest.fixture
def barkeep(monkeypatch):
    """Stand in for the control plane, without a bar or a network."""
    def answer(state):
        def _get(path: str) -> dict:
            assert path == "/api/state"
            if isinstance(state, Exception):
                raise state
            return state
        monkeypatch.setattr(bar, "_get", _get)
    return answer


class TestWhetherTheToolIsOffered:
    def test_no_token_means_the_tool_is_not_offered(self):
        assert bar.available() is False

    def test_a_token_makes_it_available(self, settings_file):
        settings_file(BARKEEP_TOKEN="a-real-looking-token")
        assert bar.available() is True

    def test_a_blank_token_line_is_the_same_as_no_token(self, settings_file):
        # install.sh writes the key unconditionally and leaves it empty when
        # the operator skips it, so "present but empty" is the common case.
        settings_file(BARKEEP_TOKEN="")
        assert bar.available() is False


class TestReportingTheBarsState:
    def test_it_says_what_is_on_the_display(self, barkeep):
        barkeep({"foreground": "skystrip", "apps": []})
        assert "The bar is showing skystrip." in bar.get_bar_status()

    def test_an_idle_bar_reads_as_showing_nothing(self, barkeep):
        barkeep({"foreground": None, "apps": []})
        assert "showing nothing" in bar.get_bar_status()

    def test_the_running_apps_description_and_uptime_are_included(
            self, barkeep):
        barkeep({
            "foreground": "skystrip",
            "apps": [{
                "name": "skystrip",
                "description": "the sky above a place",
                "uptime_s": 7200,
                "status": "running",
            }],
        })
        answer = bar.get_bar_status()
        assert "the sky above a place" in answer
        assert "2.0 hours" in answer

    def test_a_crash_looping_app_is_called_out(self, barkeep):
        barkeep({
            "foreground": "dsn",
            "apps": [{"name": "dsn", "status": "running",
                      "crash_looping": True}],
        })
        assert "crash looping" in bar.get_bar_status()

    def test_stopped_apps_are_listed_separately(self, barkeep):
        barkeep({
            "foreground": "skystrip",
            "apps": [
                {"name": "skystrip", "status": "running"},
                {"name": "dsn", "status": "stopped"},
                {"name": "hello", "status": "stopped"},
            ],
        })
        answer = bar.get_bar_status()
        assert "installed but stopped: dsn, hello" in answer

    def test_only_the_foreground_apps_details_are_narrated(self, barkeep):
        # Reading every app's description aloud would be a paragraph. The
        # question is "what is it showing", not "what is installed".
        barkeep({
            "foreground": "skystrip",
            "apps": [
                {"name": "skystrip", "status": "running",
                 "description": "the sky"},
                {"name": "dsn", "status": "running",
                 "description": "deep space network"},
            ],
        })
        answer = bar.get_bar_status()
        assert "the sky" in answer
        assert "deep space network" not in answer

    def test_the_answer_is_prose_and_never_json(self, barkeep):
        # The model reads this and paraphrases it aloud; a structure it has
        # to narrate survives that round trip much worse than a sentence.
        barkeep({"foreground": "skystrip", "apps": []})
        answer = bar.get_bar_status()
        assert "{" not in answer and "}" not in answer
        assert answer.endswith(".")


class TestWhenBarkeepIsNotThere:
    @pytest.mark.parametrize("failure", [
        urllib.error.URLError("connection refused"),
        OSError("host unreachable"),
        json.JSONDecodeError("not json", "", 0),
    ])
    def test_an_unreachable_control_plane_is_reported_not_raised(
            self, barkeep, failure):
        # This runs inside a tool call. Raising here would fail the whole
        # turn over a bar that is merely switched off.
        barkeep(failure)
        answer = bar.get_bar_status()
        assert "unreachable" in answer
        assert "state is unknown" in answer

    def test_the_failure_class_is_named_but_not_the_whole_traceback(
            self, barkeep):
        barkeep(urllib.error.URLError("connection refused"))
        assert "URLError" in bar.get_bar_status()


class TestTheDispatchWrapper:
    """Dispatch goes through tools.run_tool, and only ever did.

    bar.py used to carry its own byte-for-byte copy of that function. Nothing
    in the assistant called it — the two tests below were its only callers, so
    the test suite was the sole reason it stayed alive. They now describe the
    dispatcher that actually runs, with bar's handler as the subject.
    """

    def test_an_unknown_tool_name_says_so(self):
        assert tools.run_tool("polish_the_leds", {}) == \
            "No such tool: polish_the_leds"

    def test_a_crash_inside_a_handler_is_returned_as_text(self, monkeypatch):
        def explode() -> str:
            raise RuntimeError("bus error")

        monkeypatch.setitem(tools.HANDLERS, "get_bar_status", explode)
        assert "get_bar_status tool failed" in \
            tools.run_tool("get_bar_status", {})

    def test_bar_no_longer_carries_a_duplicate(self):
        assert not hasattr(bar, "run_tool"), \
            "the duplicate is back; dispatch belongs in tools.run_tool"


class TestTheTokenIsNotHandedToStrangers:
    """Verification is off for loopback and on for everywhere else.

    Loopback permits plain HTTP or a self-signed certificate. Anywhere else
    requires HTTPS with normal certificate and hostname verification.
    """

    @pytest.mark.parametrize("url", [
        "https://127.0.0.1:8080", "https://localhost:8080", "https://[::1]:8080",
    ])
    def test_loopback_skips_verification(self, url, monkeypatch):
        monkeypatch.setattr(bar.config, "BARKEEP_URL", url)
        assert bar._tls_context().verify_mode is ssl.CERT_NONE

    @pytest.mark.parametrize("url", [
        "https://bar.example.com:8080",     # someone else's machine
        "https://192.0.2.50:8080",          # TEST-NET is still not loopback
        "https://127.0.0.1.example.com/",   # a name that merely starts like it
    ])
    def test_anywhere_else_is_verified(self, url, monkeypatch):
        monkeypatch.setattr(bar.config, "BARKEEP_URL", url)
        context = bar._tls_context()
        assert context.verify_mode is ssl.CERT_REQUIRED
        assert context.check_hostname is True

    def test_a_url_it_cannot_parse_is_rejected(self, monkeypatch):
        monkeypatch.setattr(bar.config, "BARKEEP_URL", "not a url at all")
        with pytest.raises(ValueError, match="valid URL|must use"):
            bar._tls_context()

    @pytest.mark.parametrize("url", [
        "http://bar.example.com:8080",
        "http://192.0.2.50:8080",
        "ftp://127.0.0.1:8080",
    ])
    def test_plaintext_or_unsupported_remote_endpoints_are_rejected_before_io(
        self, url, monkeypatch
    ):
        called = False

        def capture(*args, **kwargs):
            nonlocal called
            called = True
            raise AssertionError("network access must not be attempted")

        monkeypatch.setattr(bar.config, "BARKEEP_URL", url)
        monkeypatch.setattr(bar, "_open", capture)
        assert "unreachable" in bar.get_bar_status()
        assert called is False

    @pytest.mark.parametrize("url", [
        "http://127.0.0.1:8080",
        "http://localhost:8080",
        "http://[::1]:8080",
    ])
    def test_plain_http_is_allowed_on_loopback(self, url, monkeypatch):
        seen = {}

        def capture(request):
            seen["url"] = request.full_url
            raise urllib.error.URLError("stop here")

        monkeypatch.setattr(bar.config, "BARKEEP_URL", url)
        monkeypatch.setattr(bar, "_open", capture)
        assert "unreachable" in bar.get_bar_status()
        assert seen["url"].endswith("/api/state")

    @pytest.mark.parametrize("url", [
        "https://user" + "@" + "bar.example.com:8080",
        "https://bar.example.com:8080?token=wrong-place",
        "https://bar.example.com:8080/#fragment",
    ])
    def test_ambiguous_or_credential_bearing_base_urls_are_rejected(
        self, url, monkeypatch
    ):
        monkeypatch.setattr(bar.config, "BARKEEP_URL", url)
        with pytest.raises(ValueError):
            bar._base_url()

    def test_the_token_travels_in_a_header_not_a_url(self, monkeypatch):
        # A token in a query string ends up in logs, history and referrers.
        seen = {}

        def capture(request):
            seen["url"] = request.full_url
            seen["auth"] = request.get_header("Authorization")
            raise urllib.error.URLError("stop here")

        monkeypatch.setattr(bar.config, "BARKEEP_URL", "https://127.0.0.1:8080")
        monkeypatch.setattr(bar, "_token", lambda: "secret-token-value")
        monkeypatch.setattr(bar, "_open", capture)
        bar.get_bar_status()
        assert "secret-token-value" not in seen["url"]
        assert seen["auth"] == "Bearer secret-token-value"

    def test_inherited_proxy_settings_cannot_route_barkeep_tokens(
        self, monkeypatch
    ):
        seen = {}

        class Opener:
            def open(self, request, timeout):
                seen["request"] = request
                seen["timeout"] = timeout
                return object()

        def build_opener(*handlers):
            seen["handlers"] = handlers
            return Opener()

        monkeypatch.setattr(bar.config, "BARKEEP_URL", "http://127.0.0.1:8080")
        monkeypatch.setattr(bar.urllib.request, "build_opener", build_opener)
        request = bar.urllib.request.Request(
            "http://127.0.0.1:8080/api/state",
            headers={"Authorization": "Bearer private-value"},
        )

        bar._open(request)

        proxy_handlers = [
            handler for handler in seen["handlers"]
            if isinstance(handler, bar.urllib.request.ProxyHandler)
        ]
        assert len(proxy_handlers) == 1
        assert proxy_handlers[0].proxies == {}
        assert seen["request"] is request
        assert seen["timeout"] == 5

    def test_an_unreachable_bar_does_not_echo_the_token(self, monkeypatch):
        # The result is read aloud, so whatever it says leaves the machine as
        # sound. It reports the exception's class, never its message.
        def explode(*a, **k):
            raise urllib.error.URLError("connection refused to secret-token-value")

        monkeypatch.setattr(bar, "_token", lambda: "secret-token-value")
        monkeypatch.setattr(bar, "_open", explode)
        spoken = bar.get_bar_status()
        assert "secret-token-value" not in spoken
        assert "unreachable" in spoken

    @pytest.mark.parametrize("code", [301, 302, 303, 307, 308])
    def test_redirects_cannot_forward_the_bearer_token(self, code):
        request = bar.urllib.request.Request(
            "https://bar.example.com/api/state",
            headers={"Authorization": "Bearer private-value"},
        )
        handler = bar._RejectRedirects()
        error_handler = getattr(handler, f"http_error_{code}")
        with pytest.raises(urllib.error.HTTPError, match="redirects are refused"):
            error_handler(request, None, code, "redirect", {
                "Location": "https://other.example.com/collect",
            })
