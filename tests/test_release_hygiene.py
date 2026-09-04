"""The release gate must inspect what Git will publish, not what is visible."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import check_release_hygiene as hygiene  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
BLANK_ENV = "ANTHROPIC_API_KEY=\nBARKEEP_TOKEN=\n"


def git(repo: Path, *args: str, stdin: bytes | None = None) -> bytes:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        input=stdin,
        capture_output=True,
        check=True,
    ).stdout


@pytest.fixture
def repository(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-b", "main")
    (repo / "env.example").write_text(BLANK_ENV)
    git(repo, "add", "env.example")
    monkeypatch.setattr(hygiene, "REPO_ROOT", repo)
    return repo


@pytest.fixture
def scanned(repository):
    """Stage exactly the planted body and scan the resulting index."""
    def scan(name: str, body: str | bytes):
        target = repository / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(body, bytes):
            target.write_bytes(body)
        else:
            target.write_text(body)
        git(repository, "add", "-f", "--", name)
        return hygiene.scan()

    return scan


def rules(findings) -> set[str]:
    return {finding.rule for finding in findings}


class TestTheRealRepository:
    def test_the_indexed_snapshot_is_clean_right_now(self):
        assert hygiene.scan() == []

    def test_it_is_actually_looking_at_index_entries(self):
        assert len(hygiene.index_entries()) > 10

    def test_every_exception_is_exact_live_and_explained(self):
        by_path = {entry.path: hygiene.indexed_blob(entry)
                   for entry in hygiene.index_entries() if entry.stage == 0}
        rules_by_name = {
            name: [rule for rule in hygiene.CONTENT_RULES if rule.name == name]
            for name in {rule.name for rule in hygiene.CONTENT_RULES}
        }
        for (rule, path, matched), reason in \
                hygiene.CONTENT_MATCH_EXCEPTIONS.items():
            assert rule in rules_by_name
            assert path in by_path
            assert matched in by_path[path]
            assert any(
                match.group() == matched and content_rule.is_finding(match)
                for content_rule in rules_by_name[rule]
                for match in content_rule.pattern.finditer(by_path[path])
            )
            assert len(reason) > 20

    def test_the_template_declares_exactly_the_known_secret_keys(self):
        entries = {entry.path: hygiene.indexed_blob(entry)
                   for entry in hygiene.index_entries() if entry.stage == 0}
        assert hygiene.env_example_findings(entries["env.example"]) == []


class TestTheIndexBoundary:
    def test_a_staged_secret_cannot_hide_behind_a_clean_unstaged_edit(
            self, repository):
        secret = "sk-ant-" + "api03-" + "A" * 28
        target = repository / "config.py"
        target.write_text(secret)
        git(repository, "add", "config.py")
        target.write_text("nothing private here\n")

        assert "credential.anthropic-key" in rules(hygiene.scan())

    def test_an_unstaged_secret_is_not_mistaken_for_the_snapshot(
            self, repository):
        target = repository / "config.py"
        target.write_text("nothing private here\n")
        git(repository, "add", "config.py")
        target.write_text("sk-ant-" + "api03-" + "B" * 28)

        assert hygiene.scan() == []

    def test_a_replace_ref_cannot_hide_an_indexed_secret(self, repository):
        secret = "sk-ant-" + "api03-" + "C" * 28
        target = repository / "config.py"
        target.write_text(secret)
        git(repository, "add", "config.py")
        secret_oid = git(repository, "rev-parse", ":config.py").strip().decode()
        clean_oid = git(
            repository, "hash-object", "-w", "--stdin",
            stdin=b"nothing private here\n",
        ).strip().decode()
        git(repository, "replace", secret_oid, clean_oid)
        assert git(repository, "cat-file", "blob", secret_oid) == \
            b"nothing private here\n"

        assert "credential.anthropic-key" in rules(hygiene.scan())

    def test_a_symlink_target_is_scanned_as_an_index_blob(self, repository):
        private_target = b"/home/" + b"operator/eve"
        oid = git(
            repository, "hash-object", "-w", "--stdin", stdin=private_target,
        ).strip().decode()
        git(repository, "update-index", "--add", "--cacheinfo", "120000",
            oid, "shortcut")

        assert "absolute-home-path.posix" in rules(hygiene.scan())

    def test_an_empty_index_fails_even_when_the_directory_is_empty(
            self, tmp_path, monkeypatch):
        repo = tmp_path / "empty"
        repo.mkdir()
        git(repo, "init", "-b", "main")
        monkeypatch.setattr(hygiene, "REPO_ROOT", repo)

        assert {"empty-index", "env-example.missing"} <= rules(hygiene.scan())

    def test_a_nonignored_untracked_path_fails_without_reading_it(
            self, repository):
        private = repository / "notes.txt"
        private.write_text("sk-ant-" + "api03-" + "D" * 28)
        private.chmod(0)
        try:
            findings = hygiene.scan()
        finally:
            private.chmod(0o600)

        assert hygiene.Finding("untracked-path", "notes.txt") in findings
        assert "credential.anthropic-key" not in rules(findings)

    def test_an_ignored_owner_file_is_neither_opened_nor_reported(
            self, repository):
        (repository / ".gitignore").write_text("owner-state\n")
        git(repository, "add", ".gitignore")
        private = repository / "owner-state"
        private.write_text("private\n")
        private.chmod(0)
        try:
            assert hygiene.scan() == []
        finally:
            private.chmod(0o600)


class TestCredentialRules:
    @pytest.mark.parametrize(("planted", "expected"), [
        (
            "ANTHROPIC_API_KEY=" + "sk-ant-" + "api03-" + "E" * 28,
            "credential.anthropic-key",
        ),
        (
            "key = " + "sk-" + "proj-" + "F" * 32,
            "credential.openai-key",
        ),
        (
            "token = '" + "ghp_" + "G" * 36 + "'",
            "credential.github-token",
        ),
        ("aws = " + "AKIA" + "H" * 16, "credential.aws-access-key"),
        (
            "-----BEGIN " + "OPENSSH PRIVATE KEY-----",
            "credential.private-key",
        ),
    ])
    def test_provider_credentials_are_caught(self, scanned, planted, expected):
        assert expected in rules(scanned("config.py", planted))

    def test_the_documentation_placeholder_is_allowed(self, scanned):
        placeholder = "ANTHROPIC_API_KEY=" + "sk-ant-" + "...\n"
        assert scanned("README.md", placeholder) == []

    def test_a_real_key_is_caught_inside_a_comment(self, scanned):
        secret = "sk-ant-" + "api03-" + "I" * 28
        assert "credential.anthropic-key" in rules(
            scanned("notes.md", "# an old key was " + secret + "\n"))

    @pytest.mark.parametrize("key", ["ANTHROPIC_API_KEY", "BARKEEP_TOKEN"])
    def test_a_short_project_secret_is_caught_by_its_setting_name(
            self, scanned, key):
        assignment = key + "=" + "opaque-private-value\n"
        assert "credential.project-token" in rules(
            scanned("settings.txt", assignment))

    def test_a_comment_does_not_become_a_project_secret(self, scanned):
        assignment = "# " + "BARKEEP_TOKEN" + "=documented-placeholder\n"
        assert scanned("notes.md", assignment) == []

    def test_an_exact_fixture_exception_does_not_exempt_its_file(
            self, scanned):
        token = "a-real-" + "looking-token-but-different"
        assignment = 'BARKEEP_TOKEN="' + token + '"\n'
        assert "credential.project-token" in rules(
            scanned("tests/test_bar.py", assignment))


class TestPersonalDataRules:
    @pytest.mark.parametrize("planted", [
        "WorkingDirectory=" + "/home/" + "operator/eve",
        "ExecStart=" + "/Users/" + "operator/.local/bin/uv",
        "ReadWritePaths=" + "/" + "root/.config",
    ])
    def test_absolute_home_paths_are_caught(self, scanned, planted):
        assert any(rule.startswith("absolute-home-path")
                   for rule in rules(scanned("deploy/unit.service", planted)))

    def test_an_email_address_is_caught(self, scanned):
        address = "operator" + "@" + "personal.example"
        assert "personal.email" in rules(scanned("notes.md", address))

    def test_the_exact_reserved_deploy_fixture_is_allowed(self, scanned):
        address = "test" + "@" + "example.invalid"
        body = f'git("config", "user.email", "{address}")\n'
        assert scanned("tests/test_deploy.py", body) == []

    def test_official_git_ssh_account_is_allowed_only_in_provisioner(
            self, scanned):
        account = "git" + "@" + "github.com"
        origin = account + ":ggml-org/whisper.cpp.git\n"
        assert scanned("scripts/provision-whisper.sh", origin) == []
        assert "personal.email" in rules(scanned("notes.md", origin))

    @pytest.mark.parametrize("address", [
        "10." + "23.4.5",
        "172." + "23.4.5",
        "192." + "168.4.5",
    ])
    def test_rfc1918_addresses_are_caught(self, scanned, address):
        assert "network.private-ipv4" in rules(scanned("config.txt", address))

    @pytest.mark.parametrize("address", [
        "127." + "0.0.1",
        "192." + "0.2.50",
        "198." + "51.100.50",
        "203." + "0.113.50",
    ])
    def test_loopback_and_test_net_ipv4_are_allowed(self, scanned, address):
        assert scanned("config.txt", address) == []

    @pytest.mark.parametrize("address", [
        "fd12" + ":3456::1",
        "fe80" + "::1%eth0",
    ])
    def test_ula_and_link_local_ipv6_are_caught(self, scanned, address):
        assert "network.private-ipv6" in rules(scanned("config.txt", address))

    @pytest.mark.parametrize("address", ["::1", "2001" + ":db8::1"])
    def test_loopback_and_documentation_ipv6_are_allowed(self, scanned, address):
        assert scanned("config.txt", address) == []

    def test_a_literal_deployment_hostname_is_caught(self, scanned):
        host = "desk" + "pi7"
        remote = "git remote add pi " + host + ":/home/" + "operator/eve\n"
        assert "network.operator-hostname" in rules(
            scanned("deploy/README.md", remote))

    def test_a_literal_deploy_host_assignment_is_caught(self, scanned):
        assignment = "EVE_DEPLOY_HOST" + "=deskpi7\n"
        assert "network.operator-hostname" in rules(
            scanned("env.example", BLANK_ENV + assignment))

    def test_a_mac_address_is_caught(self, scanned):
        address = "00:11:22" + ":33:44:55"
        assert "hardware.mac-address" in rules(
            scanned("settings.txt", "VOICE_SPEAKER_MAC=" + address))

    def test_a_labelled_coordinate_is_caught(self, scanned):
        coordinate = "latitude" + "=30.2672"
        assert "personal.coordinates" in rules(
            scanned("settings.txt", coordinate))

    def test_unlabelled_renderer_geometry_is_allowed(self, scanned):
        assert scanned("renderer.py", "point = (30.2672, 97.7431)\n") == []


class TestTemplateContract:
    def test_env_example_is_required(self, repository):
        git(repository, "rm", "-f", "env.example")
        assert "env-example.missing" in rules(hygiene.scan())

    @pytest.mark.parametrize("key", ["ANTHROPIC_API_KEY", "BARKEEP_TOKEN"])
    def test_each_secret_key_is_required_exactly_once(self, scanned, key):
        other = "BARKEEP_TOKEN" if key == "ANTHROPIC_API_KEY" \
            else "ANTHROPIC_API_KEY"
        body = f"{key}=\n{key}=\n{other}=\n"
        assert "env-example.secret-key-count" in rules(
            scanned("env.example", body))

    @pytest.mark.parametrize("key", ["ANTHROPIC_API_KEY", "BARKEEP_TOKEN"])
    def test_each_documented_secret_must_be_blank(self, scanned, key):
        other = "BARKEEP_TOKEN" if key == "ANTHROPIC_API_KEY" \
            else "ANTHROPIC_API_KEY"
        body = f"{key}=not-a-secret\n{other}=\n"
        assert "env-example.secret-not-blank" in rules(
            scanned("env.example", body))


class TestPrivateAndGeneratedPaths:
    @pytest.mark.parametrize(("name", "expected"), [
        ("env", "private-data-file"),
        (".env.production", "private-data-file"),
        ("memory.json", "private-data-file"),
        ("memory.json.bak", "private-data-file"),
        ("reminders.json", "private-data-file"),
        ("state/reminders.json.tmp", "private-data-file"),
        ("capture.flac", "model-weight-or-audio"),
        ("private.pem", "private-data-file"),
        ("state.sqlite3", "private-data-file"),
        ("coverage.sqlite", "generated-file"),
        (".tox/state.json", "generated-directory"),
        ("package.egg-info/PKG-INFO", "generated-directory"),
    ])
    def test_private_and_generated_paths_are_refused(
            self, scanned, name, expected):
        assert expected in rules(scanned(name, b"fixture"))

    def test_the_documented_template_path_is_allowed(self, scanned):
        assert scanned("env.example", BLANK_ENV) == []


class TestReporting:
    def test_matched_content_is_never_printed(self, scanned, capsys):
        secret = "sk-ant-" + "api03-" + "J" * 28
        scanned("config.py", secret)
        assert hygiene.main() == 1
        captured = capsys.readouterr()
        assert secret not in captured.out + captured.err

    def test_a_finding_names_its_rule_and_path(self, scanned, capsys):
        secret = "sk-ant-" + "api03-" + "K" * 28
        scanned("config.py", secret)
        assert hygiene.main() == 1
        assert "credential.anthropic-key: config.py" in capsys.readouterr().err

    def test_a_clean_index_exits_zero(self, scanned, capsys):
        scanned("README.md", "Nothing to see here.\n")
        assert hygiene.main() == 0
        assert "nothing to report" in capsys.readouterr().out

    def test_control_characters_in_paths_are_escaped(self):
        assert hygiene._display_path("bad\nname") == r"bad\nname"


class TestRunningItAsCIDoes:
    def test_the_script_exits_zero_against_this_index(self):
        result = subprocess.run(
            [sys.executable, str(REPO_ROOT / "scripts"
                                 / "check_release_hygiene.py")],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
        assert result.returncode == 0, result.stderr
