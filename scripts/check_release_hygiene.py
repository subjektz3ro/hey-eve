#!/usr/bin/env python3
"""Reject private data and local artifacts from the snapshot Git will publish.

The index is the release candidate. Reading the working tree instead lets a
staged secret hide behind a clean unstaged edit, while reading ``HEAD`` misses
the exact changes about to be committed. Every indexed regular file and
symlink is therefore read by object id, with replace refs disabled. Ignored
owner data is never opened. A nonignored untracked path is reported by name
only, because an accidental file beside the snapshot is itself unfinished
release work and its contents may be private.

This remains a snapshot gate, not a history scanner. A repository that has
ever held private data must be exported into a fresh Git root before it is
made public.

    uv run python scripts/check_release_hygiene.py
"""

from __future__ import annotations

import ipaddress
import re
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True, order=True)
class Finding:
    rule: str
    path: str


@dataclass(frozen=True)
class IndexEntry:
    mode: str
    oid: str
    stage: int
    path: str


MatchFilter = Callable[[re.Match[bytes]], bool]


def _always(_match: re.Match[bytes]) -> bool:
    return True


@dataclass(frozen=True)
class ContentRule:
    name: str
    pattern: re.Pattern[bytes]
    is_finding: MatchFilter = _always


def _private_ipv4(match: re.Match[bytes]) -> bool:
    try:
        address = ipaddress.IPv4Address(match.group().decode("ascii"))
    except ipaddress.AddressValueError:
        return False
    return any(address in network for network in _PRIVATE_IPV4_NETWORKS)


def _private_ipv6(match: re.Match[bytes]) -> bool:
    candidate = match.group().decode("ascii").strip("[]")
    candidate = candidate.split("%", 1)[0]
    try:
        address = ipaddress.IPv6Address(candidate)
    except ipaddress.AddressValueError:
        return False
    return address.is_link_local or address in _ULA_NETWORK


def _noncomment_project_token(match: re.Match[bytes]) -> bool:
    line_start = match.string.rfind(b"\n", 0, match.start()) + 1
    prefix = match.string[line_start:match.start()].rstrip()
    if prefix.endswith(b"#"):
        return False
    value = match.group().split(b"=", 1)[1].lstrip(b"\"'")
    placeholders = {
        b"sk-ant-" + b"...",
        b"sk-ant-" + b"xxx",
        b"sk-ant-api03-" + b"...",
    }
    return value not in placeholders


_PRIVATE_IPV4_NETWORKS = tuple(ipaddress.IPv4Network(network) for network in (
    "10." + "0.0.0/8",
    "172." + "16.0.0/12",
    "192." + "168.0.0/16",
))
_ULA_NETWORK = ipaddress.IPv6Network("fc00" + "::/7")


CONTENT_RULES = (
    ContentRule(
        "absolute-home-path.posix",
        re.compile(
            rb"(?<![A-Za-z0-9_.-])/(?:Users|home)/[A-Za-z0-9._-]+"
            rb"(?=$|[/\x00\t\r\n '\"`:;])"
        ),
    ),
    ContentRule(
        "absolute-home-path.root",
        re.compile(rb"(?<![A-Za-z0-9_.-])/root(?=$|[/\x00\t\r\n '\"`:;])"),
    ),
    ContentRule(
        "personal.email",
        re.compile(
            rb"(?i)(?<![A-Z0-9._%+-])[A-Z0-9._%+-]+@"
            rb"[A-Z0-9.-]+\.[A-Z]{2,}(?![A-Z0-9.-])"
        ),
    ),
    ContentRule(
        "network.private-ipv4",
        re.compile(rb"(?<![0-9])(?:[0-9]{1,3}\.){3}[0-9]{1,3}(?![0-9])"),
        _private_ipv4,
    ),
    ContentRule(
        "network.private-ipv6",
        re.compile(
            rb"(?i)(?<![0-9A-F:])(?:\[[0-9A-F:]{2,}(?:%[A-Z0-9_.-]+)?\]"
            rb"|[0-9A-F]*:[0-9A-F:]+(?:%[A-Z0-9_.-]+)?)(?![0-9A-F:])"
        ),
        _private_ipv6,
    ),
    # A literal scp-style host attached to a personal checkout is the shape
    # used by old deploy remotes. Variables and documentation URLs do not
    # match; an operator machine name does.
    ContentRule(
        "network.operator-hostname",
        re.compile(
            rb"(?i)(?<![A-Z0-9_.-])(?:[A-Z0-9._-]+@)?"
            rb"[A-Z][A-Z0-9.-]{1,252}:(?=/(?:Users|home)/)"
        ),
    ),
    ContentRule(
        "network.operator-hostname",
        re.compile(
            rb"(?im)^(?:EVE_DEPLOY_HOST|HOSTNAME)[ \t]*=[ \t]*"
            rb"[\"']?[A-Z][A-Z0-9.-]{1,252}[\"']?[ \t]*$"
        ),
    ),
    ContentRule(
        "hardware.mac-address",
        re.compile(
            rb"(?i)(?<![0-9A-F])(?:[0-9A-F]{2}:){5}[0-9A-F]{2}"
            rb"(?![0-9A-F])"
        ),
    ),
    # Geometry and timing code contains thousands of decimal pairs. Only a
    # labelled geographic value is private-location evidence.
    ContentRule(
        "personal.coordinates",
        re.compile(
            rb"(?i)(?<![A-Z0-9_])(?:lat(?:itude)?|lon(?:gitude)?|"
            rb"coordinates?|gps)[\"']?[ \t]*[:=][ \t]*[\[(]?"
            rb"[ \t]*[-+]?\d{1,3}\.\d{3,}(?:[ \t]*,[ \t]*"
            rb"[-+]?\d{1,3}\.\d{3,})?"
        ),
    ),
    ContentRule(
        "credential.anthropic-key",
        re.compile(rb"(?<![A-Za-z0-9_-])sk-ant-[A-Za-z0-9_-]{20,}"),
    ),
    ContentRule(
        "credential.openai-key",
        re.compile(rb"(?<![A-Za-z0-9_-])sk-(?!ant-)(?:proj-)?[A-Za-z0-9_-]{32,}"),
    ),
    ContentRule(
        "credential.private-key",
        re.compile(
            rb"-{5}BEGIN[ ](?:RSA[ ]|EC[ ]|DSA[ ]|OPENSSH[ ]|ENCRYPTED[ ])?"
            rb"PRIVATE[ ]KEY-{5}"
        ),
    ),
    ContentRule(
        "credential.github-token",
        re.compile(
            rb"(?<![A-Za-z0-9_])(?:gh[pousr]_[A-Za-z0-9]{30,255}"
            rb"|github_pat_[A-Za-z0-9_]{30,255})(?![A-Za-z0-9_])"
        ),
    ),
    ContentRule(
        "credential.aws-access-key",
        re.compile(rb"(?<![A-Z0-9])(?:AKIA|ASIA)[A-Z0-9]{16}(?![A-Z0-9])"),
    ),
    # The two names in eve.config.SECRETS are project secrets even when a
    # short or unfamiliar value does not resemble a provider's token format.
    ContentRule(
        "credential.project-token",
        re.compile(
            rb"(?<![A-Za-z0-9_])(?:ANTHROPIC_API_KEY|BARKEEP_TOKEN)"
            rb"[ \t]*=[ \t]*[\"']?[^\s\"'\\#]+"
        ),
        _noncomment_project_token,
    ),
)

# Exceptions are exact matched bytes at one path, never a pass for a whole
# file or rule. Constructing match-shaped values keeps this scanner from
# exempting its own source.
_TEST_EMAIL = b"test" + b"@" + b"example.invalid"
_GITHUB_GIT_USER = b"git" + b"@" + b"github.com"
_FICTIONAL_HOME = b"/home/" + b"someone"
_SYNTHETIC_MAC_1 = b"AA:BB:CC" + b":11:22:33"
_SYNTHETIC_MAC_2 = b"AA:BB:CC" + b":DD:EE:FF"


def _assignment(key: bytes, value: bytes, quote: bytes = b"") -> bytes:
    return key + b"=" + quote + value


_ANTHROPIC = b"ANTHROPIC_API_" + b"KEY"
_BARKEEP = b"BARKEEP_" + b"TOKEN"
_SHORT_ANTHROPIC = b"sk-ant-" + b"notarealkey"
_SHORT_BARKEEP = b"a-real-" + b"looking-token"

CONTENT_MATCH_EXCEPTIONS: dict[tuple[str, str, bytes], str] = {
    ("personal.email", "scripts/provision-whisper.sh", _GITHUB_GIT_USER):
        "standard GitHub SSH transport account in the official origin allowlist",
    ("personal.email", "tests/test_deploy.py", _TEST_EMAIL):
        "IANA-reserved address used only to configure throwaway repositories",
    ("absolute-home-path.posix", "tests/test_service_unit.py", _FICTIONAL_HOME):
        "fictional account required to assert rendered service paths",
    ("absolute-home-path.posix", "tests/test_speaker.py", _FICTIONAL_HOME):
        "fictional account required to assert rendered speaker service paths",
    ("hardware.mac-address", "tests/test_speaker.py", _SYNTHETIC_MAC_1):
        "locally administered synthetic address used by the speaker fixture",
    ("hardware.mac-address", "tests/test_speaker.py", _SYNTHETIC_MAC_2):
        "locally administered synthetic address used by the speaker fixture",
    (
        "credential.project-token", "tests/test_assistant.py",
        _assignment(_ANTHROPIC, _SHORT_ANTHROPIC, b'"'),
    ): "deliberately invalid short value used by the responder fixture",
    (
        "credential.project-token", "tests/test_bar.py",
        _assignment(_BARKEEP, _SHORT_BARKEEP, b'"'),
    ): "deliberately invalid value used by the local Barkeep fixture",
    (
        "credential.project-token", "tests/test_config.py",
        _assignment(_ANTHROPIC, _SHORT_ANTHROPIC, b'"'),
    ): "deliberately invalid short value used by settings parser tests",
    (
        "credential.project-token", "tests/test_config.py",
        _assignment(_BARKEEP, b"abc==def="),
    ): "deliberately invalid quoted value used by settings parser tests",
    (
        "credential.project-token", "tests/test_settings_file.py",
        _assignment(_ANTHROPIC, b"sk-readable"),
    ): "deliberately invalid short value used by settings isolation tests",
    (
        "credential.project-token", "tests/test_settings_file.py",
        _assignment(_ANTHROPIC, b"sk-x"),
    ): "deliberately invalid short value used by settings isolation tests",
    (
        "credential.project-token", "tests/test_settings_file.py",
        _assignment(_ANTHROPIC, b"sk-must-not-leak"),
    ): "sentinel whose non-disclosure is the assertion under test",
    (
        "credential.project-token", "tests/test_speaker.py",
        _assignment(_ANTHROPIC, b"sk-x"),
    ): "deliberately invalid short value used by the shell-data fixture",
    (
        "credential.project-token", "tests/test_speaker.py",
        _assignment(_ANTHROPIC, b"sk-$(touch"),
    ): "truncated shell-injection sentinel used to prove data is not sourced",
    (
        "credential.project-token", "tests/test_tools.py",
        _assignment(_ANTHROPIC, _SHORT_ANTHROPIC, b'"'),
    ): "deliberately invalid short value used by tool availability fixtures",
    (
        "credential.project-token", "tests/test_tools.py",
        _assignment(_BARKEEP, _SHORT_BARKEEP, b'"'),
    ): "deliberately invalid value used by tool availability fixtures",
}

# These are configuration-template fields, not values that may be omitted.
# The public template must contain each exactly once and with an empty value.
ENV_EXAMPLE_SECRET_KEYS = (_ANTHROPIC, _BARKEEP)

GENERATED_DIRS = frozenset({
    ".cache",
    ".eggs",
    ".hypothesis",
    ".mypy_cache",
    ".nox",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "htmlcov",
    "models",
    "node_modules",
})

GENERATED_FILES = frozenset({
    ".coverage",
    ".ds_store",
    "coverage.json",
    "coverage.sqlite",
    "coverage.xml",
    "thumbs.db",
})

PRIVATE_FILES = frozenset({
    ".env",
    ".netrc",
    ".npmrc",
    ".pypirc",
    "auth.json",
    "credentials",
    "credentials.json",
    "env",
    "heard.wav",
    "memory.json",
    "reminders.json",
    "secrets.json",
})

PRIVATE_SUFFIXES = (
    ".bak",
    ".db",
    ".key",
    ".log",
    ".orig",
    ".p12",
    ".pem",
    ".pfx",
    ".sqlite",
    ".sqlite3",
    ".swp",
)

WEIGHT_OR_AUDIO_SUFFIXES = (
    ".aac",
    ".aiff",
    ".bin",
    ".flac",
    ".m4a",
    ".mp3",
    ".ogg",
    ".onnx",
    ".opus",
    ".wav",
)


def _git(*args: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["git", "--no-replace-objects", "-C", str(REPO_ROOT), *args],
        capture_output=True,
        check=True,
    )


def index_entries() -> list[IndexEntry]:
    """Return every index stage without consulting worktree file content."""
    entries: list[IndexEntry] = []
    for record in _git("ls-files", "--stage", "-z").stdout.split(b"\0"):
        if not record:
            continue
        metadata, raw_path = record.split(b"\t", 1)
        mode, oid, stage = metadata.decode("ascii").split(" ")
        entries.append(IndexEntry(
            mode=mode,
            oid=oid,
            stage=int(stage),
            path=raw_path.decode("utf-8", "surrogateescape"),
        ))
    return entries


def tracked_files() -> list[str]:
    """Every unique path in the index, including conflicted paths."""
    return sorted({entry.path for entry in index_entries()})


def untracked_files() -> list[str]:
    """Nonignored names only; their possibly private bytes are never read."""
    output = _git("ls-files", "--others", "--exclude-standard", "-z").stdout
    return [
        name.decode("utf-8", "surrogateescape")
        for name in output.split(b"\0") if name
    ]


def indexed_blob(entry: IndexEntry) -> bytes:
    """Read the literal indexed object, bypassing refs/replace/* overlays."""
    return _git("cat-file", "blob", entry.oid).stdout


def path_findings(name: str) -> list[Finding]:
    parts = tuple(part.lower() for part in PurePosixPath(name).parts)
    basename = parts[-1]
    found: list[Finding] = []

    if any(part in GENERATED_DIRS or part.endswith(".egg-info")
           for part in parts[:-1]):
        found.append(Finding("generated-directory", name))
    if (basename in GENERATED_FILES or basename.endswith((".pyc", ".pyo"))
            or basename.startswith(".coverage.")):
        found.append(Finding("generated-file", name))

    private_name = (
        basename in PRIVATE_FILES
        or basename.startswith(".env.")
        or (len(parts) == 1 and basename.startswith("env.")
            and basename != "env.example")
        or basename.startswith("memory.json.")
        or basename.startswith("reminders.json.")
        or (len(parts) == 1 and basename.startswith("heard."))
        or basename.endswith(PRIVATE_SUFFIXES)
    )
    if private_name:
        found.append(Finding("private-data-file", name))
    if basename.endswith(WEIGHT_OR_AUDIO_SUFFIXES):
        found.append(Finding("model-weight-or-audio", name))
    return found


def content_findings(name: str, blob: bytes) -> list[Finding]:
    found: list[Finding] = []
    for rule in CONTENT_RULES:
        for match in rule.pattern.finditer(blob):
            if not rule.is_finding(match):
                continue
            if (rule.name, name, match.group()) in CONTENT_MATCH_EXCEPTIONS:
                continue
            found.append(Finding(rule.name, name))
            break
    return found


def env_example_findings(blob: bytes | None) -> list[Finding]:
    path = "env.example"
    if blob is None:
        return [Finding("env-example.missing", path)]

    found: list[Finding] = []
    lines = blob.splitlines()
    for key in ENV_EXAMPLE_SECRET_KEYS:
        values = [line[len(key) + 1:] for line in lines
                  if line.startswith(key + b"=")]
        if len(values) != 1:
            found.append(Finding("env-example.secret-key-count", path))
        elif values[0] != b"":
            found.append(Finding("env-example.secret-not-blank", path))
    return found


def scan() -> list[Finding]:
    entries = index_entries()
    found: list[Finding] = []
    env_blob: bytes | None = None

    if not entries:
        found.append(Finding("empty-index", "[index]"))

    for name in untracked_files():
        found.append(Finding("untracked-path", name))

    for entry in entries:
        found += path_findings(entry.path)
        if entry.stage != 0:
            found.append(Finding("unmerged-index-entry", entry.path))
        if entry.mode not in {"100644", "100755", "120000"}:
            found.append(Finding("unscannable-index-entry", entry.path))
            continue
        try:
            blob = indexed_blob(entry)
        except subprocess.CalledProcessError:
            found.append(Finding("unreadable-index-object", entry.path))
            continue
        found += content_findings(entry.path, blob)
        if entry.path == "env.example" and entry.stage == 0:
            env_blob = blob

    found += env_example_findings(env_blob)
    return sorted(set(found))


def _display_path(path: str) -> str:
    """Keep adversarial filenames from injecting control bytes into CI logs."""
    return path.encode("unicode_escape", "backslashreplace").decode("ascii")


def main() -> int:
    findings = scan()
    count = len(tracked_files())
    if not findings:
        print(f"release hygiene: {count} indexed files, nothing to report")
        return 0
    print(f"release hygiene: {len(findings)} finding(s) across {count} "
          f"indexed files\n", file=sys.stderr)
    for finding in findings:
        print(f"  {finding.rule}: {_display_path(finding.path)}", file=sys.stderr)
    print("\nMatched content is deliberately not shown.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
