#!/usr/bin/env bash
# Installer — from a fresh clone to a listening assistant.
#
# Sets up the Python environment, writes ~/.config/eve/env, fetches the
# Kokoro and Silero weights, and can install eve as a systemd service that
# comes back after a reboot.
#
# Run it as the unprivileged account that will own the checkout and run eve.
# Do NOT invoke it with sudo; it elevates only the individual package-manager
# and systemd commands that need it, and refuses to run as root.
set -euo pipefail
cd "$(dirname "$0")/.."
REPO_DIR=$(pwd -P)

CONFIG_DIR="${EVE_CONFIG_DIR:-${HEY_CLAUDE_CONFIG:-$HOME/.config/eve}}"
case "$CONFIG_DIR" in
  "~")   CONFIG_DIR="$HOME";;
  "~/"*) CONFIG_DIR="$HOME/${CONFIG_DIR#\~/}";;
esac
case "$CONFIG_DIR" in /*) ;; *) CONFIG_DIR="$REPO_DIR/$CONFIG_DIR";; esac
export EVE_CONFIG_DIR="$CONFIG_DIR"
ENV_FILE="$CONFIG_DIR/env"

# BEGIN TESTABLE RUNTIME ENVIRONMENT SCRUB
# The service receives EVE_CONFIG_DIR and reads every other setting from its
# data file. Make the installer do exactly the same. Otherwise an exported
# KOKORO_DIR, WHISPER_DIR, EVE_MODELS_DIR or VOICE_* value can make setup
# provision and verify one runtime while systemd starts with another.
scrub_inherited_runtime_environment() {
  local inherited_name
  INHERITED_RUNTIME_SETTINGS_FOUND=0
  while IFS= read -r inherited_name; do
    case "$inherited_name" in
      EVE_CONFIG_DIR) ;;
      ANTHROPIC_API_KEY|VOICE_*|KOKORO_*|WHISPER_*|EVE_*|\
      HEY_CLAUDE_*|BARKEEP_*|UV_PROJECT|UV_PROJECT_ENVIRONMENT|\
      UV_WORKING_DIR|UV_CONFIG_FILE|UV_ENV_FILE|UV_NO_PROJECT)
        INHERITED_RUNTIME_SETTINGS_FOUND=1
        unset "$inherited_name";;
    esac
  done < <(compgen -e)
}
# END TESTABLE RUNTIME ENVIRONMENT SCRUB

say() { printf '\n\033[1m%s\033[0m\n' "$*"; }
ask() { local p="$1" d="$2" v; read -r -p "$p [$d]: " v; echo "${v:-$d}"; }

scrub_inherited_runtime_environment
# CONFIG_DIR was selected before the scrub so the legacy bootstrap name can
# still locate an existing install. Only the current name crosses into Eve
# subprocesses and the installed units.
export EVE_CONFIG_DIR="$CONFIG_DIR"
export UV_PROJECT="$REPO_DIR"
export UV_PROJECT_ENVIRONMENT="$REPO_DIR/.venv"
if [ "$INHERITED_RUNTIME_SETTINGS_FOUND" -eq 1 ]; then
  say "Using $ENV_FILE as the authoritative runtime configuration."
  echo "  Ignoring inherited assistant and uv project settings other"
  echo "  than the EVE_CONFIG_DIR locator."
  echo "  Put persistent values in that data file instead."
fi

# The API key is the one setting with no sensible default and no way past it:
# without it eve starts, draws her face, and fails on the first question.
# Read with -s so it does not land in the terminal scrollback, and echo a
# newline ourselves because -s eats the one the user typed.
ask_secret() {
  local p="$1" v=""
  if ! read -r -s -p "$p: " v; then
    printf '\n  stdin closed before "%s" was answered. Run interactively,\n' "$p" >&2
    printf '  or create %s yourself from env.example first.\n' "$ENV_FILE" >&2
    exit 1
  fi
  printf '\n' >&2
  echo "$v"
}

ask_choice() {
  local p="$1" d="$2"; shift 2
  local allowed=("$@") v="" option
  while :; do
    v=$(ask "$p" "$d")
    for option in "${allowed[@]}"; do
      [ "$v" = "$option" ] && { echo "$v"; return; }
    done
    printf '  must be one of: %s\n' "${allowed[*]}" >&2
  done
}

ask_yes_no() {
  local p="$1" d="$2" v=""
  while :; do
    v=$(ask "$p" "$d")
    case "$v" in
      y|Y|yes|Yes|YES) echo 1; return;;
      n|N|no|No|NO)    echo 0; return;;
    esac
    printf '  answer y or n\n' >&2
  done
}

# The env file is a data file consumed by config.secret(), never shell code.
# That distinction is what lets an API key contain any character at all
# without becoming syntax. Keep the one-key-per-line invariant.
validate_env_value() {
  local key="$1" value="$2"
  case "$value" in
    *$'\n'*|*$'\r'*)
      printf '  %s must be a single-line value\n' "$key" >&2
      exit 1;;
  esac
}

validate_speaker_mac() {
  local value="$1"
  [ -z "$value" ] && return
  if [[ ! "$value" =~ ^([[:xdigit:]]{2}:){5}[[:xdigit:]]{2}$ ]]; then
    say "Invalid Bluetooth address: $value"
    echo "  Expected six hexadecimal octets separated by colons."
    exit 1
  fi
}

write_env_line() {
  validate_env_value "$1" "$2"
  printf '%s=%s\n' "$1" "$2"
}

ensure_private_directory() {
  local path="$1"
  ( umask 077; mkdir -p -- "$path" )
  if [ -L "$path" ] || [ ! -d "$path" ] || [ ! -O "$path" ]; then
    printf '  must be a real directory owned by %s: %s\n' "$(id -un)" "$path" >&2
    exit 1
  fi
  chmod 700 -- "$path"
}

say "eve installer"

if [ "${EUID:-$(id -u)}" -eq 0 ]; then
  say "Do not run this installer as root."
  echo "  Run ./deploy/install.sh as the unprivileged account that will own"
  echo "  the checkout and run eve. It needs that account's Bluetooth session"
  echo "  for the speakers; root gets a different one."
  exit 1
fi

# 1. Supported platform and dependencies, checked before uv can mutate .venv.
if [ "$(uname -s)" != "Linux" ]; then
  say "The supported service installer requires Linux."
  echo "  Direct development on other platforms is documented separately."
  exit 1
fi
case "$(uname -m)" in
  x86_64|aarch64) ;;
  *)
    say "Unsupported CPU architecture: $(uname -m)"
    echo "  Supported Linux service architectures are x86_64 and aarch64."
    exit 1;;
esac

GLIBC_INFO=$(getconf GNU_LIBC_VERSION 2>/dev/null || true)
case "$GLIBC_INFO" in
  "glibc "*) GLIBC_VERSION=${GLIBC_INFO#glibc };;
  *)
    say "The supported service installer requires glibc 2.28 or newer."
    echo "  Could not identify glibc with: getconf GNU_LIBC_VERSION"
    exit 1;;
esac
GLIBC_MAJOR=${GLIBC_VERSION%%.*}
GLIBC_REST=${GLIBC_VERSION#*.}
GLIBC_MINOR=${GLIBC_REST%%.*}
case "$GLIBC_MAJOR.$GLIBC_MINOR" in
  *[!0-9.]*|.*|*.) GLIBC_SUPPORTED=0;;
  *)
    if [ "$GLIBC_MAJOR" -gt 2 ] \
       || { [ "$GLIBC_MAJOR" -eq 2 ] && [ "$GLIBC_MINOR" -ge 28 ]; }; then
      GLIBC_SUPPORTED=1
    else
      GLIBC_SUPPORTED=0
    fi;;
esac
if [ "$GLIBC_SUPPORTED" -ne 1 ]; then
  say "Unsupported glibc: $GLIBC_VERSION"
  echo "  Eve requires glibc 2.28 or newer."
  exit 1
fi

#
# git is not optional even if you cloned by hand: deploy/ship.sh updates this
# machine by fetching into this checkout, so an install without git is an
# install that can never be updated.
missing=""
for tool in git curl cmake; do
  command -v "$tool" >/dev/null || missing="$missing $tool"
done
if ! command -v c++ >/dev/null && ! command -v g++ >/dev/null \
   && ! command -v clang++ >/dev/null; then
  missing="$missing C++-compiler"
fi
if ! command -v sha256sum >/dev/null && ! command -v shasum >/dev/null; then
  missing="$missing sha256sum-or-shasum"
fi
# ALSA. `arecord` and `aplay` are how she hears and speaks — there is no
# fallback and no Python substitute — and they ship on Raspberry Pi OS, which
# is exactly why nothing checked for them. On a minimal Debian or Fedora they
# are not installed, and without this the install succeeds, she starts, draws
# her face, and then cannot hear or speak.
alsa_missing=""
for tool in arecord aplay; do
  command -v "$tool" >/dev/null || alsa_missing="$alsa_missing $tool"
done
if [ -n "$alsa_missing" ]; then
  missing="$missing$alsa_missing (alsa-utils)"
fi
UV_BIN=$(type -P uv || true)
if [ -z "$UV_BIN" ] || [ ! -x "$UV_BIN" ]; then
  if [ -n "${HOME:-}" ] && [ -x "$HOME/.local/bin/uv" ]; then
    UV_BIN="$HOME/.local/bin/uv"
  else
    missing="$missing uv"
  fi
fi
if [ -n "$missing" ]; then
  say "Missing required tools:$missing"
  echo "  Install them before running this script again."
  echo "  uv: https://docs.astral.sh/uv/getting-started/installation/"
  case "$missing" in
    *alsa-utils*) echo "  alsa-utils: your distribution's package manager" ;;
  esac
  exit 1
fi

# Decide whether this run owns a systemd install BEFORE changing the machine.
# A missing sudo or an unusable system manager must be discovered now, not
# after an API key has already been written to disk.
SERVICE_USER=$(id -un)
SERVICE_UNIT="eve@$SERVICE_USER"
INSTALL_SVC=0
SERVICE_WAS_ACTIVE=0
SERVICE_WAS_ENABLED=0
if [ "$(uname)" = "Linux" ] && command -v systemctl >/dev/null; then
  if systemctl show --property=Version --value >/dev/null 2>&1; then
    if systemctl is-enabled --quiet "$SERVICE_UNIT" 2>/dev/null; then
      SERVICE_WAS_ENABLED=1
    fi
    if systemctl is-active --quiet "$SERVICE_UNIT" 2>/dev/null; then
      SERVICE_WAS_ACTIVE=1
      INSTALL_SVC=1
      say "eve is already installed — this run will refresh her unit."
    elif [ "$SERVICE_WAS_ENABLED" -eq 1 ]; then
      INSTALL_SVC=1
      say "eve is enabled but stopped — this run will refresh and start her."
    else
      INSTALL_SVC=$(ask_yes_no "Install eve so she runs at boot? (y/n)" "y")
    fi
  else
    say "systemctl is installed but no usable system manager is running."
    echo "  Skipping service installation; run eve by hand after setup."
  fi
fi

if [ "$INSTALL_SVC" -eq 1 ]; then
  SYSTEMD_VERSION=$(systemctl --version 2>/dev/null \
    | sed -n '1s/^systemd \([0-9][0-9]*\).*/\1/p')
  case "$SYSTEMD_VERSION" in
    ''|*[!0-9]*)
      say "Could not determine the installed systemd version."
      exit 1;;
  esac
  if [ "$SYSTEMD_VERSION" -lt 243 ]; then
    say "Unsupported systemd version: $SYSTEMD_VERSION"
    echo "  The managed service requires systemd 243 or newer."
    exit 1
  fi
  command -v sudo >/dev/null && sudo -v || {
    say "Service installation needs working sudo access."
    echo "  Run as the intended unprivileged service account with working"
    echo "  sudo, then run the installer again. Do not run it as root."
    exit 1
  }
  command -v systemd-analyze >/dev/null || {
    say "Service installation needs systemd-analyze."
    echo "  It validates the rendered unit before the live service is touched."
    exit 1
  }

  # render_service.py repeats this check, but reaching it after uv sync and
  # model downloads is too late. These values land verbatim in systemd
  # directives, where whitespace/quotes/backslashes split tokens and '%' is
  # a specifier. Fail before the first mutable setup step.
  validate_systemd_path() {
    local label="$1" value="$2"
    case "$value" in
      /*) ;;
      *)
        say "$label must be an absolute path: $value"
        exit 1;;
    esac
    case "$value" in
      *[!A-Za-z0-9_./+-]*)
        say "$label contains characters unsafe in a systemd unit: $value"
        echo "  Use a path containing only letters, digits, /, ., _, + and -."
        exit 1;;
    esac
  }
  UV_CACHE_DIR=$("$UV_BIN" cache dir)
  validate_systemd_path "checkout path" "$REPO_DIR"
  validate_systemd_path "uv executable path" "$UV_BIN"
  validate_systemd_path "config directory" "$CONFIG_DIR"
  validate_systemd_path "uv cache directory" "$UV_CACHE_DIR"
fi

# uv may download a managed interpreter when the host has no compatible
# system Python. That is the advertised fresh-host path; verify exactly what
# it selected before stopping an existing service or creating .venv.
PYTHON_BIN=$("$UV_BIN" python find '>=3.11,<3.14')
if [ -z "$PYTHON_BIN" ] || [ ! -x "$PYTHON_BIN" ]; then
  say "uv could not provision CPython 3.11, 3.12 or 3.13."
  exit 1
fi
PYTHON_IMPLEMENTATION=$("$PYTHON_BIN" -c 'import platform; print(platform.python_implementation())')
PYTHON_VERSION=$("$PYTHON_BIN" -c 'import platform; print(platform.python_version())')
case "$PYTHON_IMPLEMENTATION:$PYTHON_VERSION" in
  CPython:3.11.*|CPython:3.12.*|CPython:3.13.*) ;;
  *)
    say "Unsupported Python: $PYTHON_IMPLEMENTATION $PYTHON_VERSION"
    echo "  Eve requires CPython 3.11, 3.12 or 3.13."
    exit 1;;
esac

# Match eve.config's lexical checkout-relative path normalization. The raw
# spelling was already checked for systemd safety before uv was allowed to
# download this interpreter.
CONFIG_DIR=$(
  "$PYTHON_BIN" -c \
    'import os,sys; print(os.path.abspath(os.path.expanduser(sys.argv[1])))' \
    "$CONFIG_DIR"
)
export EVE_CONFIG_DIR="$CONFIG_DIR"
ENV_FILE="$CONFIG_DIR/env"

# An already-running service may have imported code or native libraries from
# .venv. Stop it before uv changes a byte. If anything later fails, leaving it
# stopped is deliberate: old code against a half-updated native environment
# is not a rollback.
SERVICE_STOPPED=0
SERVICE_START_ATTEMPTED=0
SERVICE_ENABLE_ATTEMPTED=0
UNIT_TMP_DIR=""
cleanup_unit_tmp() {
  if [ -n "$UNIT_TMP_DIR" ] && [ -d "$UNIT_TMP_DIR" ]; then
    rm -f -- \
      "$UNIT_TMP_DIR/eve@.service" \
      "$UNIT_TMP_DIR/eve-speaker@.service" \
      "$UNIT_TMP_DIR/eve-speaker@.timer"
    rmdir -- "$UNIT_TMP_DIR" 2>/dev/null || :
    UNIT_TMP_DIR=""
  fi
}
installer_exit() {
  status=$?
  # Once shutdown begins, do not let a second signal interrupt the fail-closed
  # stop or recursively enter another handler.
  trap - EXIT HUP INT TERM
  set +e
  cleanup_unit_tmp
  if [ "$status" -ne 0 ]; then
    stop_failed=0
    disable_failed=0
    if [ "$SERVICE_START_ATTEMPTED" -eq 1 ]; then
      # Covers failure or a signal after systemctl start but before the
      # settle loop completed. Stopping is idempotent, including when the
      # failed start already left the unit inactive.
      sudo systemctl stop "$SERVICE_UNIT" >/dev/null 2>&1 || stop_failed=1
      SERVICE_STOPPED=1
    fi
    if [ "$SERVICE_ENABLE_ATTEMPTED" -eq 1 ] \
       && [ "$SERVICE_WAS_ENABLED" -eq 0 ]; then
      # A fresh install that never passed startup verification must not come
      # back at the next boot merely because `enable` succeeded first.
      sudo systemctl disable "$SERVICE_UNIT" >/dev/null 2>&1 \
        || disable_failed=1
    fi
    if [ "$stop_failed" -eq 1 ]; then
      say "Setup failed; could not confirm that $SERVICE_UNIT is stopped."
      echo "  Stop the unverified service immediately:"
      echo "    sudo systemctl stop $SERVICE_UNIT"
    elif [ "$disable_failed" -eq 1 ]; then
      say "Setup failed; could not remove $SERVICE_UNIT from boot startup."
      echo "  Disable it immediately:"
      echo "    sudo systemctl disable $SERVICE_UNIT"
    elif [ "$SERVICE_STOPPED" -eq 1 ]; then
      say "Setup failed; $SERVICE_UNIT remains stopped."
      echo "  Correct the reported failure, rerun this installer, or recover with:"
      echo "    sudo systemctl start $SERVICE_UNIT"
    fi
  fi
  exit "$status"
}
trap installer_exit EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

if [ "$SERVICE_WAS_ACTIVE" -eq 1 ]; then
  say "Stopping $SERVICE_UNIT before changing its environment..."
  sudo systemctl stop "$SERVICE_UNIT"
  SERVICE_STOPPED=1
fi

# 2. Python environment.
say "Syncing the environment..."
"$UV_BIN" sync --locked --no-dev --python "$PYTHON_BIN"

# 3. Configuration.
ensure_private_directory "$CONFIG_DIR"
if [ ! -f "$ENV_FILE" ]; then
  say "Let's configure this install (Enter accepts the default)."
  echo "  Every key is documented in env.example."
  KEY=$(ask_secret "Anthropic API key (input hidden)")
  [ -n "$KEY" ] || { echo "  an API key is required" >&2; exit 1; }
  PRESENT=$(ask_choice "How she presents — female or male" "female" female male)
  PERSONA=$(ask_choice "Persona — glados or plain" "glados" glados plain)
  # Offer what this machine actually has. The default used to be one
  # particular Shure on one particular Pi, so pressing enter on somebody
  # else's box inherited a device that does not exist — and a wrong
  # microphone is total, silent deafness.
  MIC_LISTING=$(arecord -l 2>/dev/null || true)
  MIC_DEFAULT=$(sed -n \
    's/^card [0-9]*: \([^ ]*\) .*, device \([0-9]*\): .*/plughw:CARD=\1,DEV=\2/p' \
    <<< "$MIC_LISTING")
  MIC_DEFAULT=${MIC_DEFAULT%%$'\n'*}
  if [ -n "$MIC_DEFAULT" ]; then
    echo "  Capture devices found:"
    printf '%s\n' "$MIC_LISTING" | sed -n 's/^card /    card /p'
  else
    echo "  No capture hardware found. \`arecord -l\` lists them once one is"
    echo "  plugged in; leave this blank and eve will look again at startup."
  fi
  MIC=$(ask "Microphone (ALSA device)" "$MIC_DEFAULT")
  # A Bluetooth speaker has to be *connected*, not merely paired, and nothing
  # on a Linux box initiates that after a reboot — BlueZ only reconnects after
  # a link loss, and a fresh boot is not one. With an address here, a timer
  # keeps the link up; without one, nothing is installed and a wired speaker
  # works exactly as before.
  if command -v bluetoothctl >/dev/null; then
    echo "  Optional: a Bluetooth speaker's address, so it is reconnected after"
    echo "  a reboot. Blank for a wired speaker. Paired devices:"
    bluetoothctl devices Paired 2>/dev/null | sed 's/^/    /' || true
    SPEAKER=$(ask "Bluetooth speaker address (blank for none)" "")
    validate_speaker_mac "$SPEAKER"
  else
    echo "  bluetoothctl is not installed, so Bluetooth reconnect setup is"
    echo "  skipped. A wired/default ALSA speaker still works."
    SPEAKER=""
  fi
  echo "  Optional: a Barkeep token enables the BUSY Bar tool. Blank to skip,"
  echo "  and that tool is simply not offered to the model."
  BARTOKEN=$(ask_secret "Barkeep token (blank to skip, input hidden)")
  (
    # The key must be owner-only from the first byte, not merely after a
    # later chmod. Publish the complete file atomically from a same-directory
    # temporary path; an interruption can then leave only a 0600 temp file.
    umask 077
    TMP=$(mktemp "$CONFIG_DIR/.env.install.XXXXXX")
    trap 'rm -f -- "$TMP"' EXIT
    trap 'exit 1' HUP INT TERM
    {
      write_env_line ANTHROPIC_API_KEY "$KEY"
      write_env_line BARKEEP_TOKEN "$BARTOKEN"
      write_env_line VOICE_PRESENT "$PRESENT"
      write_env_line VOICE_PERSONA "$PERSONA"
      write_env_line VOICE_MIC "$MIC"
      write_env_line VOICE_SPEAKER_MAC "$SPEAKER"
      echo "# Every other key is documented in env.example."
    } > "$TMP"
    mv "$TMP" "$ENV_FILE"
    trap - EXIT HUP INT TERM
  )
  say "Wrote $ENV_FILE — edit it any time, then restart to apply."
else
  say "$ENV_FILE already exists — keeping it."
  echo "  (The interview runs only when that file is absent. To be asked"
  echo "  again, move it aside and rerun; or edit it by hand.)"

  # A setting introduced after this file was written would otherwise never
  # reach an existing install: the interview above is skipped, so the key is
  # simply absent and the feature that depends on it silently does not
  # install. That is exactly how the speaker reconnect would have missed the
  # machine it was written for.
  #
  # So: ask once, for settings that are new and absent, and append rather
  # than rewrite. Answering nothing leaves the file untouched.
  if ! grep -q '^VOICE_SPEAKER_MAC=' "$ENV_FILE" && command -v bluetoothctl >/dev/null; then
    say "New since this file was written: the Bluetooth speaker's address."
    echo "  Nothing on a Linux box reconnects a paired speaker after a reboot"
    echo "  — BlueZ only reconnects after a link loss, and a boot is not one."
    echo "  With an address here, a timer keeps the link up. Paired devices:"
    bluetoothctl devices Paired 2>/dev/null | sed 's/^/    /' || true
    SPEAKER=$(ask "Bluetooth speaker address (blank to skip)" "")
    if [ -n "$SPEAKER" ]; then
      validate_speaker_mac "$SPEAKER"
      validate_env_value VOICE_SPEAKER_MAC "$SPEAKER"
      printf '%s=%s\n' VOICE_SPEAKER_MAC "$SPEAKER" >> "$ENV_FILE"
      say "Added VOICE_SPEAKER_MAC to $ENV_FILE."
    fi
  fi
fi
# An install predating the umask above can leave the key world-readable.
chmod 600 "$ENV_FILE"
CONFIGURED_SPEAKER=$(
  "$UV_BIN" run --no-sync python -c \
    'from eve import config; print(config.setting("VOICE_SPEAKER_MAC"))'
)
validate_speaker_mac "$CONFIGURED_SPEAKER"
if [ "$INSTALL_SVC" -eq 1 ] && [ -n "$CONFIGURED_SPEAKER" ] \
   && ! command -v bluetoothctl >/dev/null; then
  say "VOICE_SPEAKER_MAC is set, but bluetoothctl is not installed."
  echo "  Install bluez/bluetoothctl, or clear VOICE_SPEAKER_MAC."
  exit 1
fi

# 4. Model weights and the complete local speech runtime.
#
# Offer to finish the rename first. hey-claude became eve, and the rename
# moved this default path without moving the 325MB that lived at it — which
# cost the assistant her voice on the first thing she was asked to say after
# the restart, an hour later, in a crash loop. Moving it is a rename on the
# same filesystem, so it is instant and it is reversible.
LEGACY_MODELS="$HOME/.local/share/hey-claude"
MODELS_DIR=$("$UV_BIN" run --no-sync python -m eve.doctor --show models-dir)
if [ -d "$LEGACY_MODELS" ] && [ "$LEGACY_MODELS" != "$MODELS_DIR" ]; then
  say "Found model weights under the project's former name."
  echo "  from: $LEGACY_MODELS"
  echo "  to:   $MODELS_DIR"
  echo "  Leaving them costs nothing — eve looks in both — but two copies of"
  echo "  325MB is worth tidying once."
  if [ "$(ask_yes_no "Move them? (y/n)" "y")" = "1" ]; then
    mkdir -p "$MODELS_DIR"
    for sub in kokoro models; do
      if [ -d "$LEGACY_MODELS/$sub" ]; then
        mkdir -p "$MODELS_DIR/$sub"
        # -n: never clobber a file already at the destination. If both exist
        # the new one is the one in use, and it wins.
        mv -n "$LEGACY_MODELS/$sub"/* "$MODELS_DIR/$sub/" 2>/dev/null || true
        rmdir "$LEGACY_MODELS/$sub" 2>/dev/null || true
      fi
    done
    rmdir "$LEGACY_MODELS" 2>/dev/null \
      || echo "  (left $LEGACY_MODELS in place — it still has something in it)"
    say "Moved."
  fi
fi

say "Fetching model weights..."
UV_BIN="$UV_BIN" scripts/fetch-models.sh

say "Provisioning the pinned whisper.cpp runtime..."
UV_BIN="$UV_BIN" scripts/provision-whisper.sh

# File presence is not readiness. Exercise Kokoro, Silero and Whisper against
# one generated utterance before installing or starting any service. This is
# intentionally hardware-free: it opens no microphone, speaker, framebuffer,
# BUSY Bar or network API.
say "Verifying the local voice pipeline..."
"$UV_BIN" run --no-sync eve doctor

# 5. The service.
if [ "$INSTALL_SVC" -eq 1 ]; then
  mkdir -p -- "$UV_CACHE_DIR"

  # SupplementaryGroups= fails the whole service with status=216/GROUP if
  # even one named group does not exist. Include every device group this host
  # actually defines, including input where distributions use it for USB
  # microphones and input nodes.
  group_exists() {
    if command -v getent >/dev/null; then
      getent group "$1" >/dev/null 2>&1
    else
      grep -q "^$1:" /etc/group
    fi
  }
  DEVICE_GROUPS=()
  for group in audio video tty bluetooth input; do
    group_exists "$group" && DEVICE_GROUPS+=("$group")
  done

  UNIT_TMP_DIR=$(mktemp -d "${TMPDIR:-/tmp}/eve-unit.XXXXXX")
  UNIT_TMP="$UNIT_TMP_DIR/eve@.service"

  "$UV_BIN" run --no-sync python deploy/render_service.py \
    --template deploy/eve@.service \
    --checkout "$REPO_DIR" \
    --uv "$UV_BIN" \
    --config-dir "$CONFIG_DIR" \
    --uv-cache-dir "$UV_CACHE_DIR" \
    --supplementary-groups "${DEVICE_GROUPS[@]}" \
    --output "$UNIT_TMP"

  if ! systemd-analyze verify "$UNIT_TMP"; then
    say "The rendered unit failed systemd validation."
    echo "  No unit or host-wide journal setting was changed."
    exit 1
  fi

  # The speaker keeper. Rendered the same way and for the same reason — the
  # checkout path is host-specific — but installed only when an address was
  # configured, so a wired setup gains nothing it has to reason about.
  INSTALL_SPEAKER=0
  if [ -n "$CONFIGURED_SPEAKER" ]; then
    INSTALL_SPEAKER=1
    SPEAKER_TMP="$UNIT_TMP_DIR/eve-speaker@.service"
    SPEAKER_TIMER_TMP="$UNIT_TMP_DIR/eve-speaker@.timer"
    "$UV_BIN" run --no-sync python deploy/render_service.py \
      --template deploy/eve-speaker@.service \
      --checkout "$REPO_DIR" \
      --uv "$UV_BIN" \
      --config-dir "$CONFIG_DIR" \
      --uv-cache-dir "$UV_CACHE_DIR" \
      --output "$SPEAKER_TMP"
    cp deploy/eve-speaker@.timer "$SPEAKER_TIMER_TMP"
    if ! systemd-analyze verify "$SPEAKER_TMP" "$SPEAKER_TIMER_TMP"; then
      say "The rendered speaker service or timer failed systemd validation."
      echo "  No unit or host-wide journal setting was changed."
      exit 1
    fi
  fi

  # Journald's default is 10% of the filesystem. This box runs an
  # always-listening assistant, so retention is a privacy horizon as much as
  # a disk one. This setting is host-wide, however, not scoped to Eve, so a
  # first install must never silently change retention for every service.
  # It is considered only after every selected unit has passed validation.
  JOURNAL_DROPIN=/etc/systemd/journald.conf.d/eve-retention.conf
  if [ -f "$JOURNAL_DROPIN" ]; then
    INSTALL_JOURNAL_RETENTION=1
    say "Refreshing the existing host-wide journald limit (14 days, 200MB)."
  else
    say "Optional host-wide journal retention"
    echo "  This limits logs for every service on this machine, not only Eve,"
    echo "  to at most 14 days and 200MB."
    INSTALL_JOURNAL_RETENTION=$(
      ask_yes_no "Apply that host-wide limit? (y/n)" "n"
    )
  fi
  if [ "$INSTALL_JOURNAL_RETENTION" -eq 1 ]; then
    sudo install -m 0644 -D deploy/journald-retention.conf "$JOURNAL_DROPIN"
    sudo systemctl restart systemd-journald
  else
    say "Keeping this host's existing journald policy."
  fi

  # Nothing under /etc is written until all runtime and unit checks above
  # have succeeded.
  sudo install -m 0644 "$UNIT_TMP" /etc/systemd/system/eve@.service
  if [ "$INSTALL_SPEAKER" -eq 1 ]; then
    sudo install -m 0644 "$SPEAKER_TMP" /etc/systemd/system/eve-speaker@.service
    sudo install -m 0644 "$SPEAKER_TIMER_TMP" \
      /etc/systemd/system/eve-speaker@.timer
  fi

  sudo systemctl daemon-reload
  if [ "$INSTALL_SPEAKER" -eq 1 ]; then
    sudo systemctl enable --now "eve-speaker@$SERVICE_USER.timer"
    say "Speaker keeper installed — it reconnects on boot and every minute."
  else
    # Refreshing an install after clearing VOICE_SPEAKER_MAC must also clear
    # the old desired state; otherwise yesterday's timer keeps reconnecting a
    # speaker the operator explicitly removed from configuration.
    sudo systemctl disable --now "eve-speaker@$SERVICE_USER.timer" \
      >/dev/null 2>&1 || true
  fi
  if [ "$SERVICE_WAS_ENABLED" -eq 0 ]; then
    SERVICE_ENABLE_ATTEMPTED=1
  fi
  sudo systemctl enable "$SERVICE_UNIT"
  SERVICE_START_ATTEMPTED=1
  sudo systemctl start "$SERVICE_UNIT"
  # A successful systemctl start only proves exec() was attempted. Import and
  # device failures can arrive a moment later, so require a bounded period of
  # stable activity before calling the install successful.
  for settle_second in 1 2 3 4 5; do
    sleep 1
    if ! systemctl is-active --quiet "$SERVICE_UNIT"; then
      sudo systemctl stop "$SERVICE_UNIT" 2>/dev/null || true
      say "$SERVICE_UNIT did not remain active during startup."
      echo "  Inspect: journalctl -u $SERVICE_UNIT -n 100 --no-pager"
      exit 1
    fi
  done
  SERVICE_STOPPED=0
  SERVICE_START_ATTEMPTED=0
  SERVICE_ENABLE_ATTEMPTED=0
  cleanup_unit_tmp

  say "Running. Watch her: journalctl -u $SERVICE_UNIT -f"

  cat <<EOF

  One more thing worth doing. ship.sh needs sudo only to stop and start this
  exact unit. If this account has a blanket NOPASSWD rule, scope it instead:

    # /etc/sudoers.d/eve   (edit with visudo -f, never a plain editor)
    $SERVICE_USER ALL=(root) NOPASSWD: $(command -v systemctl) stop $SERVICE_UNIT, \\
      $(command -v systemctl) start $SERVICE_UNIT
EOF
  if [ -e /dev/fb0 ]; then
    CHVT_BIN=$(type -P chvt || true)
    if [ -n "$CHVT_BIN" ]; then
      cat <<EOF

  This host has a framebuffer, so the reference face also needs permission
  for exactly one console switch. Add this separate narrow rule with:

    sudo visudo -f /etc/sudoers.d/eve

    $SERVICE_USER ALL=(root) NOPASSWD: $CHVT_BIN 8
EOF
    else
      say "The framebuffer face needs chvt, but chvt is not installed."
      echo "  Install your distribution's console-tools/kbd package, then add"
      echo "  a sudoers rule allowing only its absolute path followed by 8."
    fi
  fi
else
  say "Done. Run her with: uv run python -m eve.main"
fi

say 'Say "hey Eve".'
