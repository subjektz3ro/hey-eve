#!/usr/bin/env bash
# Bring up the Bluetooth speaker, and keep it up.
#
# BlueZ will not do this for you. A paired, bonded, *trusted* device is one
# BlueZ will happily ACCEPT a connection from — it never initiates one. Its
# [Policy] section only covers reconnecting after a link loss (supervision
# timeout); a fresh boot is not a link loss, so nothing ever asks. The result
# is a machine that comes back from a reboot with the speaker paired and
# silent, which is exactly what happened on 2026-08-15:
#
#   Paired: yes  Bonded: yes  Trusted: yes  Connected: no
#
# and zero mentions of the speaker's address in the whole boot journal.
#
# That matters more here than it would elsewhere because /etc/asound.conf
# pins ALSA's `default` at one specific address, so with no link every single
# aplay fails instantly rather than falling back to another sink.
#
# Idempotent on purpose: the timer runs this every minute, and the common case
# is "already connected", which must cost nothing and log nothing.
set -euo pipefail

CONFIG_DIR="${EVE_CONFIG_DIR:-${HEY_CLAUDE_CONFIG:-$HOME/.config/eve}}"
ENV_FILE="$CONFIG_DIR/env"

# Read one key the same way eve/config.py:secret() does — one KEY=value per
# line, never sourced, so a file holding an API key is never shell.
setting() {
  [ -f "$ENV_FILE" ] || return 0
  while IFS= read -r line; do
    case "$line" in
      "$1="*) printf '%s\n' "${line#*=}" | tr -d "\"'" ; return 0;;
    esac
  done < "$ENV_FILE"
}

MAC="${VOICE_SPEAKER_MAC:-$(setting VOICE_SPEAKER_MAC)}"
if [ -z "${MAC:-}" ]; then
  # Not an error: a wired speaker is a perfectly good configuration, and this
  # unit is installed unconditionally. Say so once and stop.
  echo "no VOICE_SPEAKER_MAC in $ENV_FILE — nothing to connect"
  exit 0
fi

command -v bluetoothctl >/dev/null || {
  echo "bluetoothctl not installed; cannot manage $MAC" >&2
  exit 1
}

if bluetoothctl info "$MAC" 2>/dev/null | grep -q "Connected: yes"; then
  exit 0          # the common case: already up, say nothing
fi

# `connect` blocks until the link is established or the attempt fails. The
# timeout matters: with the speaker powered off this otherwise hangs holding
# the unit active until the next timer tick lands on top of it. Guarded
# because `timeout` is coreutils — always present on the Pi, absent on a
# developer's macOS, and this must not become a Linux-only script for the
# sake of a safety margin.
connect=(bluetoothctl connect "$MAC")
if command -v timeout >/dev/null 2>&1; then
  connect=(timeout 25 "${connect[@]}")
fi

echo "connecting $MAC"
if "${connect[@]}" 2>&1 | grep -q "Connection successful"; then
  echo "connected $MAC"
  exit 0
fi

# Failing is normal and not worth a red unit: the speaker is simply switched
# off, and the timer will get it when someone turns it on. Exit 0 so systemd
# does not mark the unit failed for a condition that resolves itself.
echo "could not connect $MAC (powered off?) — will retry"
exit 0
