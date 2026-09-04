"""The bar, as the model sees it: tools backed by Barkeep's HTTP API.

Barkeep supervises the apps that draw to the panel, so the assistant asks it
rather than drawing to the device directly — the bar's draw model is
last-writer-wins, and a second writer would fight whatever app is running.
"""
from __future__ import annotations

import json
import ipaddress
import ssl
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from typing import Any

from eve import config

# Fresh Barkeep installs use loopback HTTP. A loopback instance may also use
# Barkeep's generated self-signed certificate, for which certificate
# verification is disabled because the connection never leaves the host.
# Outside loopback, bearer-token traffic requires verified HTTPS.


def _is_loopback(host: str) -> bool:
    if host.lower() == "localhost":
        return True
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False
    return address.is_loopback


def _base_url() -> urllib.parse.SplitResult:
    """Validate the configured endpoint before constructing a token request."""
    try:
        parsed = urllib.parse.urlsplit(config.BARKEEP_URL)
        host = parsed.hostname or ""
    except ValueError as exc:
        raise ValueError("BARKEEP_URL is not a valid URL") from exc
    if parsed.scheme not in {"http", "https"} or not host:
        raise ValueError("BARKEEP_URL must use http or https and include a host")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("BARKEEP_URL must not contain credentials")
    if parsed.query or parsed.fragment:
        raise ValueError("BARKEEP_URL must not contain a query or fragment")
    if parsed.scheme == "http" and not _is_loopback(host):
        raise ValueError("plain HTTP is allowed only for loopback Barkeep URLs")
    return parsed


def _tls_context() -> ssl.SSLContext:
    """Verification off for loopback, on for anywhere else."""
    parsed = _base_url()
    if not _is_loopback(parsed.hostname or ""):
        return ssl.create_default_context()
    context = ssl.create_default_context()
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    return context


class _RejectRedirects(urllib.request.HTTPRedirectHandler):
    """Barkeep API calls never forward their bearer token to another URL."""

    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> None:
        return None

    def http_error_302(
        self,
        req: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
    ) -> None:
        raise urllib.error.HTTPError(
            req.full_url, code, "Barkeep API redirects are refused", headers, fp
        )

    http_error_301 = http_error_302
    http_error_303 = http_error_302
    http_error_307 = http_error_302
    http_error_308 = http_error_302


def _open(request: urllib.request.Request) -> Any:
    opener = urllib.request.build_opener(
        # Barkeep is an explicitly configured local/LAN endpoint. Inherited
        # proxy variables must not route its bearer token through another
        # process or host, especially when the URL itself is loopback.
        urllib.request.ProxyHandler({}),
        urllib.request.HTTPSHandler(context=_tls_context()),
        _RejectRedirects(),
    )
    return opener.open(request, timeout=5)


def _token() -> str:
    """Barkeep's API token, from this app's own env file."""
    return config.secret("BARKEEP_TOKEN")


def available() -> bool:
    """Whether to offer the bar tool at all.

    Without a token every call would come back as an authentication error the
    model would then read aloud, which is worse than not having the tool.
    """
    return bool(_token())


def _get(path: str) -> dict:
    _base_url()
    request = urllib.request.Request(
        f"{config.BARKEEP_URL.rstrip('/')}{path}",
        headers={"Authorization": f"Bearer {_token()}"},
    )
    with _open(request) as response:
        return json.load(response)


def get_bar_status() -> str:
    """A spoken-language summary of what the bar is doing right now.

    Returns prose rather than JSON: the model reads this and paraphrases it
    aloud, and a compact sentence survives that round trip better than a
    structure it has to describe.
    """
    try:
        state = _get("/api/state")
    except (urllib.error.URLError, OSError, json.JSONDecodeError, ValueError) as exc:
        return f"Barkeep is unreachable ({exc.__class__.__name__}); the bar's state is unknown."

    foreground = state.get("foreground") or "nothing"
    parts = [f"The bar is showing {foreground}."]

    for app in state.get("apps", []):
        if app.get("name") != foreground:
            continue
        if description := app.get("description"):
            parts.append(f"That app is: {description}.")
        if (uptime := app.get("uptime_s")) is not None:
            parts.append(f"It has been running for {uptime / 3600:.1f} hours.")
        if app.get("crash_looping"):
            parts.append("It is crash looping.")

    stopped = [
        app["name"] for app in state.get("apps", [])
        if app.get("status") != "running"
    ]
    if stopped:
        parts.append(f"Also installed but stopped: {', '.join(stopped)}.")

    return " ".join(parts)


# The tool surface handed to the model. Descriptions are prescriptive about
# *when* to call, not just what the tool does — that is what drives correct
# triggering.
TOOLS = [
    {
        "name": "get_bar_status",
        "description": (
            "Get what the BUSY Bar is currently displaying and which of its "
            "apps are running or stopped. Call this whenever you are asked what "
            "the bar is doing, what is on the display, whether an app is "
            "running, or how long something has been up."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
]

# Annotated to match tools.HANDLERS, which merges this in: an unannotated
# single-entry dict still widens on merge and breaks the dispatch check.
HANDLERS: dict[str, Callable[..., str]] = {"get_bar_status": get_bar_status}
