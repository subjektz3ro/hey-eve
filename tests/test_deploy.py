"""ship.sh's guards, checked against real git repositories.

No Pi, no panel, no microphone, no network. Each test builds a throwaway
origin and a throwaway clone in a temp directory and runs the actual script
with `--dry-run`, so the guards are exercised as written rather than as
described.

`--dry-run` never contacts the host and never fetches, which is what makes
this possible at all — the argument checking, the ref resolution and THE
guard (is this commit actually on origin?) all happen locally.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

# ship.sh's first act is `cd "$(dirname "$0")/.."`, so it always operates on
# the checkout it is *part of*, never on the caller's working directory. That
# is correct for a deploy script and it is the whole reason these tests copy
# it into the throwaway repository rather than invoking it in place: running
# the real script from a temp directory would quietly deploy this laptop's
# actual eve checkout, and every guard would appear to pass while testing
# nothing.
DEPLOY_FILES = (
    "ship.sh",
    "eve@.service",
    "eve-speaker@.service",
    "eve-speaker@.timer",
    "render_service.py",
)


def git(*args: str, cwd: Path) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=True,
    ).stdout.strip()


def run_ship(*args: str, cwd: Path, **env: str):
    """Run the copy of ship.sh that lives inside `cwd`."""
    environment = dict(os.environ)
    # Anything inherited here would silently supply a host the test meant to
    # withhold, and the "no host" case would pass for the wrong reason.
    for name in list(environment):
        if name.startswith("EVE_DEPLOY_"):
            del environment[name]
    environment.update(env)
    environment["GIT_TERMINAL_PROMPT"] = "0"
    return subprocess.run(
        ["bash", str(cwd / "deploy" / "ship.sh"), *args],
        cwd=cwd, capture_output=True, text=True, env=environment,
    )


def plant_deploy_scripts(work: Path) -> None:
    """Copy the real deploy scripts in, so the real ones are what run."""
    (work / "deploy").mkdir(exist_ok=True)
    for name in DEPLOY_FILES:
        target = work / "deploy" / name
        target.write_text((REPO_ROOT / "deploy" / name).read_text())
        target.chmod(0o755)


@pytest.fixture
def deployable(tmp_path):
    """An origin with one commit on main, and a clone that tracks it."""
    origin = tmp_path / "origin.git"
    work = tmp_path / "work"
    subprocess.run(["git", "init", "--bare", "-b", "main", str(origin)],
                   check=True, capture_output=True)
    subprocess.run(["git", "clone", str(origin), str(work)],
                   check=True, capture_output=True)
    git("config", "user.email", "test@example.invalid", cwd=work)
    git("config", "user.name", "Test", cwd=work)
    # ship.sh reads deploy/eve@.service and deploy/render_service.py out of
    # the TARGET COMMIT to compute the unit contract, so the fixture repo has
    # to genuinely contain them at that commit.
    plant_deploy_scripts(work)
    git("add", "-A", cwd=work)
    git("commit", "-m", "first", cwd=work)
    git("push", "-u", "origin", "main", cwd=work)
    return work


class TestTheCommitMustBeFetchable:
    def test_a_pushed_commit_is_accepted(self, deployable):
        result = run_ship("--dry-run", "pi.invalid", cwd=deployable)
        assert result.returncode == 0, result.stderr
        assert "would deploy" in result.stdout

    def test_an_unpushed_commit_is_refused_with_the_fix(self, deployable):
        # THE guard. Without it the Pi is told to reset to a commit it can
        # never fetch, and the deploy fails halfway with the service stopped.
        (deployable / "new.txt").write_text("local only\n")
        git("add", "-A", cwd=deployable)
        git("commit", "-m", "not pushed", cwd=deployable)

        result = run_ship("--dry-run", "pi.invalid", cwd=deployable)
        assert result.returncode != 0
        assert "is not on origin/main" in result.stderr
        # The error carries the command that fixes it, because the person
        # reading it is mid-deploy and does not want to look it up.
        assert "git push origin HEAD:main" in result.stderr

    def test_an_unknown_ref_is_refused(self, deployable):
        result = run_ship("--dry-run", "--ref", "v9.9.9", "pi.invalid",
                          cwd=deployable)
        assert result.returncode != 0
        assert "no such commit: v9.9.9" in result.stderr

    def test_a_tag_on_main_can_be_shipped_by_name(self, deployable):
        # The release path: ship the tag, not whatever HEAD happens to be.
        git("tag", "-a", "v0.1.0", "-m", "first release", cwd=deployable)
        result = run_ship("--dry-run", "--ref", "v0.1.0", "pi.invalid",
                          cwd=deployable)
        assert result.returncode == 0, result.stderr
        assert git("rev-parse", "--short", "v0.1.0^{commit}",
                   cwd=deployable) in result.stdout


class TestArguments:
    def test_no_host_anywhere_is_an_error_that_names_both_ways_to_give_one(
            self, deployable):
        result = run_ship("--dry-run", cwd=deployable)
        assert result.returncode != 0
        assert "no host" in result.stderr
        assert "EVE_DEPLOY_HOST" in result.stderr

    def test_the_host_may_come_from_the_environment(self, deployable):
        result = run_ship("--dry-run", cwd=deployable,
                          EVE_DEPLOY_HOST="pi.invalid")
        assert result.returncode == 0, result.stderr
        assert "pi.invalid" in result.stdout

    def test_an_unknown_option_is_refused_rather_than_treated_as_a_host(
            self, deployable):
        # Otherwise `--dryrun` becomes the hostname and the script tries to
        # ssh to it, which is a real deploy attempt from a typo.
        result = run_ship("--dryrun", "pi.invalid", cwd=deployable)
        assert result.returncode != 0
        assert "unknown option: --dryrun" in result.stderr

    def test_help_prints_the_header_without_deploying(self, deployable):
        result = run_ship("--help", cwd=deployable)
        assert result.returncode == 0
        assert "EVE_DEPLOY_HOST" in result.stdout
        assert "would deploy" not in result.stdout

    def test_public_checkout_name_is_the_default(self, deployable):
        result = run_ship("--dry-run", "pi.invalid", cwd=deployable)
        assert "pi.invalid:hey-eve" in result.stdout

    def test_an_existing_private_checkout_can_keep_its_path(self, deployable):
        result = run_ship(
            "--dry-run", "pi.invalid", cwd=deployable, EVE_DEPLOY_PATH="eve"
        )
        assert "pi.invalid:eve" in result.stdout

    def test_a_missing_remote_is_reported_with_the_command_to_add_one(
            self, tmp_path):
        lonely = tmp_path / "lonely"
        lonely.mkdir()
        subprocess.run(["git", "init", "-b", "main", str(lonely)],
                       check=True, capture_output=True)
        git("config", "user.email", "test@example.invalid", cwd=lonely)
        git("config", "user.name", "Test", cwd=lonely)
        plant_deploy_scripts(lonely)
        git("add", "-A", cwd=lonely)
        git("commit", "-m", "first", cwd=lonely)

        result = run_ship("--dry-run", "pi.invalid", cwd=lonely)
        assert result.returncode != 0
        assert "no 'origin' remote" in result.stderr
        assert "git remote add origin" in result.stderr


class TestWhatTheHostIsAskedToDo:
    def test_uncommitted_work_is_left_behind_and_said_so(self, deployable):
        # Only committed work ships. Saying it out loud is the difference
        # between a surprise and an expectation.
        (deployable / "scratch.txt").write_text("work in progress\n")
        git("add", "-A", cwd=deployable)
        result = run_ship("--dry-run", "pi.invalid", cwd=deployable)
        assert result.returncode == 0, result.stderr
        assert "uncommitted changes stay behind" in result.stdout

    def test_the_host_resets_hard_rather_than_pulling(self, deployable):
        # The Pi is a deploy target, not a working copy: a merge conflict on
        # a machine nobody is sitting at helps no one.
        plan = run_ship("--dry-run", "pi.invalid", cwd=deployable).stdout
        assert "git reset --hard" in plan
        assert "git pull" not in plan

    def test_dependencies_are_synced_from_the_lock_before_restart(self,
                                                                  deployable):
        # So new code never starts against the previous release's packages.
        plan = run_ship("--dry-run", "pi.invalid", cwd=deployable).stdout
        assert "sync --locked" in plan
        assert plan.index("sync --locked") < plan.index("systemctl start")

    def test_uv_is_pinned_to_the_deployed_checkout(self, deployable):
        plan = run_ship("--dry-run", "pi.invalid", cwd=deployable).stdout
        assert 'export UV_PROJECT="$remote_checkout"' in plan
        assert 'export UV_PROJECT_ENVIRONMENT="$remote_checkout/.venv"' in plan

    def test_the_runtime_is_verified_after_sync_and_before_restart(self, deployable):
        plan = run_ship("--dry-run", "pi.invalid", cwd=deployable).stdout
        assert "EVE_CONFIG_DIR=" in plan
        assert "eve doctor" in plan
        assert plan.index("sync --locked") < plan.index("eve doctor")
        assert plan.index("eve doctor") < plan.index("systemctl start")

    def test_restart_must_remain_active_during_a_bounded_settle_period(
            self, deployable):
        plan = run_ship("--dry-run", "pi.invalid", cwd=deployable).stdout
        start = plan.rindex('sudo systemctl start "$unit"')
        settle = plan.index("for settle_second in 1 2 3 4 5")
        success = plan.index("live on $(hostname)")
        assert start < settle < success
        assert 'systemctl is-active --quiet "$unit"' in plan[settle:success]

    def test_an_interrupted_start_is_stopped_before_remote_exit(self, deployable):
        plan = run_ship("--dry-run", "pi.invalid", cwd=deployable).stdout
        handler = plan[plan.index("remote_deploy_exit()"):
                       plan.index("trap remote_deploy_exit EXIT")]
        attempted = plan.index("service_start_attempted=1")
        start = plan.rindex('sudo systemctl start "$unit"')
        complete = plan.index("deploy_complete=1")
        assert 'sudo systemctl stop "$unit"' in handler
        assert attempted < start < complete
        assert "trap 'exit 129' HUP" in plan

    def test_the_service_is_stopped_before_the_checkout_moves(self,
                                                             deployable):
        plan = run_ship("--dry-run", "pi.invalid", cwd=deployable).stdout
        assert plan.index("systemctl stop") < plan.index("git reset --hard")

    def test_the_unit_contract_is_checked_before_anything_is_touched(
            self, deployable):
        # An old process may still be mid-reply while this runs, and it must
        # not find half of the next release under it.
        plan = run_ship("--dry-run", "pi.invalid", cwd=deployable).stdout
        assert "eve-unit-contract-sha256" in plan
        assert plan.index("eve-unit-contract-sha256") \
            < plan.rindex('sudo systemctl stop "$unit"')

    def test_the_installed_unit_must_point_at_the_deployed_checkout(self, deployable):
        plan = run_ship("--dry-run", "pi.invalid", cwd=deployable).stdout
        assert "eve-checkout-dir" in plan
        assert "remote_checkout=$(pwd -P)" in plan
        assert plan.index("different Eve checkout") \
            < plan.rindex('sudo systemctl stop "$unit"')

    def test_the_contract_covers_every_installed_eve_unit(self, deployable):
        plan = run_ship("--dry-run", "pi.invalid", cwd=deployable).stdout
        assert "eve-speaker@.service" in plan
        assert "eve-speaker@.timer" in plan

    def test_ship_never_installs_the_unit_itself(self, deployable):
        # Applying a unit is root-equivalent, and this account's sudo is
        # deliberately scoped to stop and start only.
        plan = run_ship("--dry-run", "pi.invalid", cwd=deployable).stdout
        assert "/etc/systemd/system" not in plan
        assert "daemon-reload" not in plan

    def test_the_instance_name_is_resolved_on_the_host_not_here(
            self, deployable):
        # Interpolating $USER locally would deploy under whoever is sitting
        # at this laptop.
        plan = run_ship("--dry-run", "pi.invalid", cwd=deployable).stdout
        assert "id -un" in plan
        assert os.environ.get("USER", "\0impossible") not in plan

    def test_a_named_unit_is_used_verbatim_without_an_instance(
            self, deployable):
        plan = run_ship("--dry-run", "pi.invalid", cwd=deployable,
                        EVE_DEPLOY_SERVICE="eve-test.service").stdout
        assert "eve-test.service" in plan

    def test_a_dry_run_touches_nothing_and_contacts_no_one(self, deployable):
        before = git("rev-parse", "HEAD", cwd=deployable)
        result = run_ship("--dry-run", "pi.invalid", cwd=deployable)
        assert result.returncode == 0
        assert git("rev-parse", "HEAD", cwd=deployable) == before
        assert git("status", "--porcelain", cwd=deployable) == ""


class TestAnExistingInstallStillGetsNewSettings:
    """The interview only runs when the env file is absent.

    Anything introduced after a machine was set up would therefore never
    reach it — the key is simply missing and whatever depends on it does not
    install. That is precisely how the speaker reconnect would have missed
    the Pi it was written for, whose env file predates it and holds three
    keys.
    """

    def test_a_missing_new_setting_is_offered_rather_than_skipped(self):
        script = (REPO_ROOT / "deploy" / "install.sh").read_text()
        kept = script.index("already exists — keeping it")
        assert "VOICE_SPEAKER_MAC" in script[kept:], \
            "an existing install never gets settings added after it was made"

    def test_it_appends_rather_than_rewriting_the_file(self):
        # The file holds an API key; rewriting it to add one line is a much
        # bigger operation than appending one.
        script = (REPO_ROOT / "deploy" / "install.sh").read_text()
        kept = script.index("already exists — keeping it")
        assert ">> \"$ENV_FILE\"" in script[kept:]

    def test_it_only_asks_when_the_key_is_absent(self):
        script = (REPO_ROOT / "deploy" / "install.sh").read_text()
        assert "grep -q '^VOICE_SPEAKER_MAC=' \"$ENV_FILE\"" in script

    def test_the_speaker_units_install_only_when_an_address_is_set(self):
        # A wired speaker should gain nothing it has to reason about.
        script = (REPO_ROOT / "deploy" / "install.sh").read_text()
        guard = script.index('if [ -n "$CONFIGURED_SPEAKER" ]')
        install = script.index('sudo install -m 0644 "$SPEAKER_TIMER_TMP"')
        assert guard < install

    def test_clearing_the_address_disables_an_old_timer(self):
        script = (REPO_ROOT / "deploy" / "install.sh").read_text()
        assert 'disable --now "eve-speaker@$SERVICE_USER.timer"' in script
