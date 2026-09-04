"""Rendering the systemd unit, and the contract that keeps it honest.

The unit used to be a literal file naming one user and one home directory.
Now it is a template, and two things have to hold: rendering must leave
nothing unresolved, and the digest ship.sh checks must actually change when
either half of the contract does. A digest that does not move is worse than
no digest, because it reports safety it is not measuring.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "deploy"))
import render_service  # noqa: E402  (needs the path line above)

REPO_ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = REPO_ROOT / "deploy" / "eve@.service"

HOST = {
    "checkout": Path("/home/someone/eve"),
    "uv": Path("/home/someone/.local/bin/uv"),
    "config_dir": Path("/home/someone/.config/eve"),
    "uv_cache_dir": Path("/home/someone/.cache/uv"),
}


def _section(unit: str, name: str) -> str:
    """The directives under one [Section] header.

    A substring split is not good enough here: the unit's own comments name
    both sections while explaining why each directive lives where it does.
    Only a line that *is* the header starts a section.
    """
    out, inside = [], False
    for line in unit.splitlines():
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            inside = stripped == f"[{name}]"
            continue
        if inside and not stripped.startswith("#"):
            out.append(line)
    return "\n".join(out)


@pytest.fixture
def rendered():
    return render_service.render(TEMPLATE.read_text(), **HOST)


class TestTheTemplateItself:
    def test_it_carries_no_absolute_home_path_of_its_own(self):
        # The failure this whole mechanism exists to prevent, and the reason
        # scripts/check_release_hygiene.py gates on it in CI.
        assert re.search(r"/(?:home|Users|root)/", TEMPLATE.read_text()) is None

    def test_the_instance_name_supplies_the_account(self):
        # `eve@someone`. Runs as that user rather than root because it needs
        # that user's Bluetooth session for the speakers.
        assert "User=%i" in TEMPLATE.read_text()

    def test_every_placeholder_it_uses_is_one_the_renderer_knows(self):
        used = set(re.findall(r"@[A-Z_]+@", TEMPLATE.read_text()))
        assert used <= set(render_service.PLACEHOLDERS)
        assert used, "a template with no placeholders is not a template"


class TestRendering:
    def test_nothing_is_left_unresolved(self, rendered):
        # A unit with a literal @UV_EXECUTABLE@ in ExecStart starts and then
        # dies on every restart, which is worse than not starting at all.
        assert re.findall(r"@[A-Z_]+@", rendered) == []

    def test_the_host_paths_land_where_they_belong(self, rendered):
        assert "WorkingDirectory=/home/someone/eve" in rendered
        assert "ExecStart=/home/someone/.local/bin/uv run --no-sync" in rendered
        assert 'Environment="EVE_CONFIG_DIR=/home/someone/.config/eve"' in rendered
        assert 'Environment="UV_PROJECT=/home/someone/eve"' in rendered
        assert 'Environment="UV_PROJECT_ENVIRONMENT=/home/someone/eve/.venv"' \
            in rendered
        assert 'Environment="UV_CACHE_DIR=/home/someone/.cache/uv"' in rendered

    def test_only_groups_the_installer_found_are_rendered(self):
        unit = render_service.render(
            TEMPLATE.read_text(), **HOST,
            supplementary_groups=("audio", "input"),
        )
        assert "SupplementaryGroups=audio input" in unit

    @staticmethod
    def _directives(rendered):
        """Only real directives — the file explains itself in comments too."""
        return [line.split("=", 1)[0].lstrip("+-")
                for line in rendered.splitlines()
                if "=" in line and not line.lstrip().startswith("#")]

    def test_no_directive_here_can_block_the_console_switch(self, rendered):
        # The faces run `sudo -n chvt` to move the console off the panel.
        # NoNewPrivileges=yes blocks that outright — it was set once, and the
        # cost was two sudo errors per start and a getty's cursor on her face.
        assert "NoNewPrivileges" not in self._directives(rendered)

    def test_the_filesystem_sandbox_is_deliberately_absent(self, rendered):
        # Borrowed wholesale from barkeep, a web control plane, and applied to
        # a process that drives a framebuffer, ALSA, Bluetooth and a VT, and
        # loads a 325MB ONNX graph through libraries that cache under $HOME.
        # The template keeps barkeep's *structure*; its threat model is not
        # this service's. See the long note in deploy/eve@.service.
        present = set(self._directives(rendered))
        for directive in ("ProtectSystem", "ProtectHome", "PrivateTmp",
                          "ReadWritePaths"):
            assert directive not in present, \
                f"{directive} came back without being proven on hardware"

    def test_the_note_explaining_why_survives_with_it(self, rendered):
        # The absence is a decision, not an oversight. If someone deletes the
        # reasoning, the next person re-adds the block and rediscovers all of
        # it the hard way.
        assert "not this service's threat model" in rendered

    def test_it_still_runs_as_the_instance_account_not_root(self, rendered):
        # Dropping the sandbox is not the same as dropping least privilege.
        assert "User=%i" in rendered

    def test_it_never_syncs_at_startup(self, rendered):
        # install.sh and ship.sh both finish a locked sync before restarting,
        # so service startup must not race a deploy-time resolver.
        assert "run --no-sync" in rendered


class TestRefusingBadInput:
    def test_a_relative_path_is_refused(self):
        with pytest.raises(SystemExit, match="absolute path"):
            render_service.render(TEMPLATE.read_text(),
                                  **{**HOST, "checkout": Path("eve")})

    def test_a_newline_in_a_path_cannot_invent_a_directive(self):
        # A unit file is line-oriented, so an unchecked newline is arbitrary
        # systemd configuration.
        attack = Path("/home/someone/eve\nExecStartPre=/bin/rm -rf /")
        with pytest.raises(SystemExit, match="single line"):
            render_service.render(TEMPLATE.read_text(),
                                  **{**HOST, "checkout": attack})

    @pytest.mark.parametrize("suffix", [" with-space", "%i", '"quote', "\\slash"])
    def test_systemd_special_characters_in_paths_are_refused(self, suffix):
        with pytest.raises(SystemExit, match="cannot be rendered safely"):
            render_service.render(
                TEMPLATE.read_text(),
                **{**HOST, "checkout": Path("/home/someone/eve" + suffix)},
            )

    def test_an_invalid_group_name_cannot_invent_a_directive(self):
        with pytest.raises(SystemExit, match="group names"):
            render_service.render(
                TEMPLATE.read_text(), **HOST,
                supplementary_groups=("audio", "input\nExecStart=/bin/false"),
            )

    def test_a_placeholder_the_renderer_has_never_heard_of_is_refused(self):
        # The forward-compatible half. Adding @GPU_DEVICE@ to the template
        # and forgetting to teach the renderer about it would otherwise ship
        # a unit that installs cleanly and fails on every start.
        with pytest.raises(SystemExit, match=r"@GPU_DEVICE@"):
            render_service.render(
                "ExecStart=@UV_EXECUTABLE@ --device @GPU_DEVICE@\n", **HOST)

    def test_lowercase_at_signs_are_not_mistaken_for_placeholders(self):
        # The template's own comments talk about the `eve@someone` instance.
        assert "eve@someone" in render_service.render(
            "# the eve@someone instance\nExecStart=@UV_EXECUTABLE@\n", **HOST)


class TestTheContractDigest:
    def test_the_same_inputs_give_the_same_digest(self):
        first = render_service.contract_digest(b"template", b"renderer")
        second = render_service.contract_digest(b"template", b"renderer")
        assert first == second

    def test_editing_the_template_moves_it(self):
        # ship.sh compares this against the installed unit's first line. If
        # it did not move, a unit-changing release would deploy silently
        # against a stale unit.
        before = render_service.contract_digest(b"template", b"renderer")
        after = render_service.contract_digest(b"template!", b"renderer")
        assert before != after

    def test_editing_the_renderer_moves_it_too(self):
        # A renderer that starts emitting a different ReadWritePaths= changes
        # the installed unit just as surely as editing the template does.
        before = render_service.contract_digest(b"template", b"renderer")
        after = render_service.contract_digest(b"template", b"renderer!")
        assert before != after

    def test_editing_a_related_unit_moves_it_too(self):
        before = render_service.contract_digest(
            b"template", b"renderer", (("speaker", b"one"),)
        )
        after = render_service.contract_digest(
            b"template", b"renderer", (("speaker", b"two"),)
        )
        assert before != after

    def test_the_two_halves_cannot_be_slid_past_each_other(self):
        # NUL separators, so ("ab", "c") and ("a", "bc") are different
        # contracts rather than the same concatenation.
        assert render_service.contract_digest(b"ab", b"c") \
            != render_service.contract_digest(b"a", b"bc")


class TestTheWrittenFile:
    def test_the_first_line_is_the_contract_ship_reads(self, tmp_path):
        # ship.sh reads it on the host with a single `read -r`, so it has to
        # be the first line and nothing else may precede it.
        output = tmp_path / "eve@.service"
        subprocess.run(
            [sys.executable, str(REPO_ROOT / "deploy" / "render_service.py"),
             "--template", str(TEMPLATE),
             "--checkout", str(HOST["checkout"]),
             "--uv", str(HOST["uv"]),
             "--config-dir", str(HOST["config_dir"]),
             "--uv-cache-dir", str(HOST["uv_cache_dir"]),
             "--output", str(output)],
            check=True, capture_output=True,
        )
        first = output.read_text().splitlines()[0]
        assert first.startswith("# eve-unit-contract-sha256=")
        assert len(first.split("=", 1)[1]) == 64
        assert output.read_text().splitlines()[1] \
            == "# eve-config-dir=/home/someone/.config/eve"
        assert output.read_text().splitlines()[2] \
            == "# eve-checkout-dir=/home/someone/eve"

    def test_the_digest_written_matches_the_one_ship_recomputes(self,
                                                                tmp_path):
        # ship.sh rebuilds it on the host from `git show` of both files. If
        # these two ever disagree, every deploy stops with a false alarm.
        output = tmp_path / "eve@.service"
        subprocess.run(
            [sys.executable, str(REPO_ROOT / "deploy" / "render_service.py"),
             "--template", str(TEMPLATE),
             "--checkout", str(HOST["checkout"]),
             "--uv", str(HOST["uv"]),
             "--config-dir", str(HOST["config_dir"]),
             "--uv-cache-dir", str(HOST["uv_cache_dir"]),
             "--output", str(output)],
            check=True, capture_output=True,
        )
        written = output.read_text().splitlines()[0].split("=", 1)[1]
        expected = render_service.contract_digest(
            TEMPLATE.read_bytes(),
            (REPO_ROOT / "deploy" / "render_service.py").read_bytes(),
            tuple(
                (name, (REPO_ROOT / "deploy" / name).read_bytes())
                for name in render_service.RELATED_CONTRACT_FILES
            ),
        )
        assert written == expected


class TestTheResourceCeiling:
    """A crash loop and a leak should both stop, rather than continue.

    The sandbox block was withdrawn deliberately and these are not it — they
    bound resources rather than restricting access, so none of the reasoning
    in the unit's own note about ProtectHome and chvt applies to them.
    """

    def test_the_start_limit_is_in_the_section_systemd_reads(self, rendered):
        # It was briefly in [Service], where systemd ignores it — and
        # `systemd-analyze verify` only *warns*, so install.sh's exit-code
        # check passed and the unit installed with no limit at all. The
        # warning is easy to miss in a wall of install output; this is not.
        unit_section = _section(rendered, "Unit")
        for directive in ("StartLimitIntervalSec", "StartLimitBurst"):
            assert directive in unit_section, \
                f"{directive} must be in [Unit]; systemd ignores it in [Service]"

    def test_the_ceiling_is_in_the_section_systemd_reads(self, rendered):
        # The mirror image: these two are [Service] keys and are ignored in
        # [Unit], so the same mistake in reverse is just as silent.
        service_section = _section(rendered, "Service")
        for directive in ("MemoryMax", "OOMPolicy"):
            assert directive in service_section, \
                f"{directive} must be in [Service]"

    def test_the_start_limit_is_actually_reachable(self, rendered):
        # systemd's default is five starts in ten seconds, and RestartSec=3
        # can never reach it: five restarts take fifteen. So a permanently
        # broken install looped forever at a steady three seconds and the unit
        # stayed `active (auto-restart)` rather than going `failed`.
        interval = int(re.search(r"StartLimitIntervalSec=(\d+)", rendered).group(1))
        burst = int(re.search(r"StartLimitBurst=(\d+)", rendered).group(1))
        restart_sec = int(re.search(r"RestartSec=(\d+)", rendered).group(1))
        assert burst * restart_sec < interval, \
            "the burst cannot be reached before the window resets"

    def test_a_transient_crash_still_recovers_on_its_own(self, rendered):
        assert "Restart=always" in rendered
        burst = int(re.search(r"StartLimitBurst=(\d+)", rendered).group(1))
        assert burst >= 3, "one bad restart should not need an operator"

    def test_there_is_a_memory_ceiling(self, rendered):
        # 8GB, no meaningful swap, a 310MB ONNX graph and renderers that
        # cache. An unbounded leak takes the machine, not just the service.
        assert re.search(r"MemoryMax=\d+[MG]", rendered)

    def test_being_killed_by_the_oom_reaper_counts_as_failure(self, rendered):
        # The default is `continue`, which restarts into the same wall.
        assert "OOMPolicy=stop" in rendered

    def test_core_dumps_cannot_persist_live_credentials_history_or_audio(
            self, rendered):
        assert "LimitCORE=0" in _section(rendered, "Service")
        speaker = render_service.render(
            (REPO_ROOT / "deploy" / "eve-speaker@.service").read_text(),
            **HOST,
        )
        assert "LimitCORE=0" in _section(speaker, "Service")

    def test_the_withdrawn_sandbox_has_not_crept_back(self, rendered):
        # Directives only — the unit's own comment block names all four while
        # arguing at length for their removal, and that argument is the thing
        # most worth keeping.
        directives = [line.split("=")[0].strip()
                      for line in rendered.splitlines()
                      if "=" in line and not line.lstrip().startswith("#")]
        for name in ("NoNewPrivileges", "ProtectSystem",
                     "ProtectHome", "PrivateTmp"):
            assert name not in directives, f"{name} is back"

    def test_the_note_explaining_the_withdrawal_survives_with_it(self, rendered):
        assert "NoNewPrivileges" in rendered, \
            "the reasoning went with the directives; it should not have"

    def test_security_md_does_not_claim_hardening_the_unit_lacks(self):
        # It did, for some time after the withdrawal — which is worse than
        # never having claimed it.
        security = (REPO_ROOT / "SECURITY.md").read_text()
        claimed = re.findall(
            r"^- `(NoNewPrivileges|ProtectSystem|ProtectHome|PrivateTmp)=",
            security, re.MULTILINE)
        assert not claimed, f"SECURITY.md still lists {claimed} as set"
