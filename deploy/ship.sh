#!/usr/bin/env bash
# ship.sh — deploy a commit to the Pi running eve.
#
# The Pi pulls from the configured git remote (origin by default). This
# script's only job on your machine is to prove the commit is actually there,
# then tell the Pi to go and get it.
#
# That ordering is the whole point. The remote is the source of truth, so the
# Pi can always rebuild itself from a clean clone without this laptop.
# eve's origin used to be a bare repository ON the Pi, which meant the only
# complete copy of the project lived on whichever machine you happened to be
# sitting at — and the deploy target and the backup were the same SD card.
#
#   ./deploy/ship.sh                    # HEAD -> $EVE_DEPLOY_HOST
#   ./deploy/ship.sh --ref v0.2 raspberrypi   # a tag, to a named host
#   ./deploy/ship.sh --dry-run          # print what would happen, touch nothing
#
# Configuration, all optional except the host:
#   EVE_DEPLOY_HOST     ssh target (or pass it as the last argument)
#   EVE_DEPLOY_PATH     checkout on the host      (default: hey-eve)
#                       Existing private hosts may keep using
#                       EVE_DEPLOY_PATH=eve.
#   EVE_DEPLOY_SERVICE  systemd unit              (default: eve@$USER)
#   EVE_DEPLOY_REMOTE   git remote                (default: origin)
#   EVE_DEPLOY_BRANCH   branch it must be on      (default: main)
#
# Only committed work ships. Settings never travel: ~/.config/eve/env holds
# the API key and belongs to the Pi, and nothing here touches it.
set -euo pipefail
cd "$(dirname "$0")/.."

REMOTE="${EVE_DEPLOY_REMOTE:-origin}"
BRANCH="${EVE_DEPLOY_BRANCH:-main}"
DIR="${EVE_DEPLOY_PATH:-hey-eve}"
# A trailing '@' means "instance name is the user on the HOST", resolved
# there rather than here. Interpolating $USER locally would deploy under
# whoever is sitting at this laptop; single-quoting it for the remote shell
# is worse, because it never expands at all and systemd is handed a unit
# literally called eve@$USER. Ask the host.
SERVICE="${EVE_DEPLOY_SERVICE:-eve@}"
REF="HEAD"
DRY_RUN=0
HOST="${EVE_DEPLOY_HOST:-}"

die() { printf '\033[31merror:\033[0m %s\n' "$*" >&2; exit 1; }
note() { printf '  %b\n' "$*"; }

while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --ref)     REF="${2:?--ref needs a commit, tag or branch}"; shift 2 ;;
    -h|--help)
      sed -n '2,/^set -euo pipefail$/{ /^set -euo pipefail$/q; p; }' "$0" \
        | sed 's/^# \{0,1\}//'
      exit 0;;
    -*)        die "unknown option: $1" ;;
    *)         HOST="$1"; shift ;;
  esac
done

[ -n "$HOST" ] || die "no host. Pass one as an argument or set EVE_DEPLOY_HOST.
       e.g. ./deploy/ship.sh raspberrypi"

SHA=$(git rev-parse --verify --quiet "$REF^{commit}") \
  || die "no such commit: $REF"
SHORT=$(git rev-parse --short "$SHA")

if ! git diff --quiet || ! git diff --cached --quiet; then
  note "note: uncommitted changes stay behind — shipping $SHORT as committed."
fi

git remote get-url "$REMOTE" >/dev/null 2>&1 \
  || die "no '$REMOTE' remote. Add one:
       git remote add $REMOTE <url>"

if [ "$DRY_RUN" -eq 0 ]; then
  git fetch --quiet "$REMOTE" "$BRANCH" \
    || die "cannot reach $REMOTE. Check the network and deploy credentials."
fi

# THE guard. If the commit is not on the remote, the Pi cannot fetch it, and
# deploying it any other way would put the Pi somewhere unreproducible.
if ! git merge-base --is-ancestor "$SHA" "$REMOTE/$BRANCH" 2>/dev/null; then
  die "$SHORT is not on $REMOTE/$BRANCH, so the host cannot fetch it.

       Push it first:   git push $REMOTE HEAD:$BRANCH"
fi

# `git fetch` then `reset --hard` rather than `git pull`: the Pi is a deploy
# target, not a working copy. Anything edited there is discarded on purpose,
# and a merge conflict on a machine nobody is sitting at helps no one. Sync
# the exact locked environment before restart so new code never starts
# against dependencies from the previous release.
REMOTE_CMD="set -e
cd '$DIR'
remote_checkout=\$(pwd -P)
unset UV_WORKING_DIR UV_CONFIG_FILE UV_ENV_FILE UV_NO_PROJECT
export UV_PROJECT=\"\$remote_checkout\"
export UV_PROJECT_ENVIRONMENT=\"\$remote_checkout/.venv\"
unit='$SERVICE'
service_start_attempted=0
deploy_complete=0
remote_deploy_exit() {
  status=\$?
  trap - EXIT HUP INT TERM
  set +e
  if [ \"\$status\" -ne 0 ] \
     && [ \"\$service_start_attempted\" -eq 1 ] \
     && [ \"\$deploy_complete\" -eq 0 ]; then
    if ! sudo systemctl stop \"\$unit\" >/dev/null 2>&1; then
      echo \"error: could not confirm that unverified \$unit is stopped\" >&2
      echo \"stop it immediately: sudo systemctl stop \$unit\" >&2
    fi
  fi
  exit \"\$status\"
}
trap remote_deploy_exit EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM
case \"\$unit\" in *@) unit=\"\$unit\$(id -un)\";; esac
git fetch --quiet '$REMOTE'
if command -v uv >/dev/null 2>&1; then
  uv_bin=\$(command -v uv)
elif [ -x \"\$HOME/.local/bin/uv\" ]; then
  uv_bin=\"\$HOME/.local/bin/uv\"
else
  echo \"error: uv not found on PATH or at \$HOME/.local/bin/uv\" >&2
  exit 1
fi
# Applying a unit is root-equivalent and deliberately stays outside this
# deploy account's narrow stop/start permission. install.sh renders host
# paths and refreshes it. Check the TARGET commit's contract BEFORE changing
# the live checkout: eve may still be mid-reply while this runs, and it must
# not find half of the next release under it.
installed=\$(systemctl show --property=FragmentPath --value \"\$unit\" \\
  2>/dev/null || true)
expected_unit_hash=\$(
  {
    printf 'template\0'
    git show '$SHA:deploy/eve@.service'
    printf '\0renderer\0'
    git show '$SHA:deploy/render_service.py'
    printf '\0unit-file\0eve-speaker@.service\0'
    git show '$SHA:deploy/eve-speaker@.service'
    printf '\0unit-file\0eve-speaker@.timer\0'
    git show '$SHA:deploy/eve-speaker@.timer'
  } | \"\$uv_bin\" run --no-sync python -c \
    'import hashlib,sys; print(hashlib.sha256(sys.stdin.buffer.read()).hexdigest())'
)
installed_unit_hash=
config_dir=
installed_checkout=
if [ -n \"\$installed\" ] && [ -r \"\$installed\" ]; then
  {
    IFS= read -r installed_unit_header || true
    IFS= read -r installed_config_header || true
    IFS= read -r installed_checkout_header || true
  } < \"\$installed\"
  case \"\$installed_unit_header\" in
    '# eve-unit-contract-sha256='*) installed_unit_hash=\"\${installed_unit_header#*=}\";;
  esac
  case \"\$installed_config_header\" in
    '# eve-config-dir='*) config_dir=\"\${installed_config_header#*=}\";;
  esac
  case \"\$installed_checkout_header\" in
    '# eve-checkout-dir='*) installed_checkout=\"\${installed_checkout_header#*=}\";;
  esac
fi
if [ -z \"\$installed_unit_hash\" ] \\
   || [ \"\$expected_unit_hash\" != \"\$installed_unit_hash\" ]; then
  echo \"\" >&2
  echo \"  ============================================================\" >&2
  echo \"  ERROR: the installed service unit is missing or out of date.\" >&2
  echo \"  The live checkout was left unchanged.\" >&2
  echo \"  Apply this unit-changing release interactively on the host:\" >&2
  echo \"    sudo systemctl stop \\\"\$unit\\\"  # if it is currently running\" >&2
  echo \"    git reset --hard $SHA\" >&2
  echo \"    ./deploy/install.sh\" >&2
  echo \"  ============================================================\" >&2
  exit 1
fi
if [ -z \"\$installed_checkout\" ] \
   || [ \"\$installed_checkout\" != \"\$remote_checkout\" ]; then
  echo \"error: installed unit points at a different Eve checkout\" >&2
  echo \"  deploy checkout:    \$remote_checkout\" >&2
  echo \"  installed checkout: \${installed_checkout:-missing}\" >&2
  echo \"rerun ./deploy/install.sh from the deploy checkout\" >&2
  exit 1
fi
if [ -z \"\$config_dir\" ]; then
  echo \"error: installed unit does not identify Eve's config directory\" >&2
  echo \"rerun ./deploy/install.sh interactively on the host\" >&2
  exit 1
fi
sudo systemctl stop \"\$unit\"
git reset --hard --quiet $SHA
\"\$uv_bin\" sync --locked --no-dev
EVE_CONFIG_DIR=\"\$config_dir\" \"\$uv_bin\" run --no-sync eve doctor
service_start_attempted=1
sudo systemctl start \"\$unit\"
for settle_second in 1 2 3 4 5; do
  sleep 1
  if ! systemctl is-active --quiet \"\$unit\"; then
    sudo systemctl stop \"\$unit\" 2>/dev/null || true
    echo \"error: \$unit did not remain active during startup\" >&2
    echo \"inspect: journalctl -u \$unit -n 100 --no-pager\" >&2
    exit 1
  fi
done
deploy_complete=1
service_start_attempted=0
echo \"  \$(git rev-parse --short HEAD) live on \$(hostname) as \$unit\""

if [ "$DRY_RUN" -eq 1 ]; then
  echo "would deploy $SHORT to $HOST:$DIR"
  echo "would run on $HOST:"
  printf '%s\n' "$REMOTE_CMD" | sed 's/^/    /'
  exit 0
fi

# shellcheck disable=SC2029  # expanding SHA/DIR locally is the intent
ssh "$HOST" "$REMOTE_CMD"

# The instance name must resolve on the HOST (same reasoning as SERVICE
# above), so the hint quotes the whole command for the remote shell; a bare
# \$(id -un) here would expand on whatever laptop the operator pastes it into.
case "$SERVICE" in
  *@) echo "shipped $SHORT. watch: ssh $HOST 'journalctl -u \"${SERVICE}\$(id -un)\" -f'" ;;
  *)  echo "shipped $SHORT. watch: ssh $HOST 'journalctl -u \"$SERVICE\" -f'" ;;
esac
