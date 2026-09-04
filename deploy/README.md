# Installing and deploying hey-eve

`deploy/install.sh` provisions the supported Linux voice runtime and can
install Eve as a systemd service. `deploy/ship.sh` is the maintainer path for
deploying an already-pushed commit to an existing host.

## Supported host

The managed service path supports:

- 64-bit `x86_64` or `aarch64` Linux with glibc 2.28 or newer;
- CPython 3.11–3.13, managed by `uv`;
- systemd 243 or newer for the managed service;
- at least 4 GiB of RAM and 2 GiB of free disk are recommended;
- `git`, `curl`, `bash`, CMake, a C/C++ compiler, a SHA-256 utility, and
  `alsa-utils`; and
- a microphone plus an ALSA playback device.

The tested reference host is an 8 GiB Raspberry Pi 5 running 64-bit Raspberry
Pi OS. Alpine Linux/musl, 32-bit operating systems, and Python 3.14 are not
supported by the locked neural-runtime dependencies.

On Debian or Raspberry Pi OS:

```bash
sudo apt update
sudo apt install -y git curl cmake build-essential alsa-utils
```

Install [`uv`](https://docs.astral.sh/uv/getting-started/installation/) before
running the repository installer.

## Initial installation

Run the installer as the unprivileged account that will own the checkout and
service. Do not invoke the script itself with `sudo`; it elevates only the
system changes that require it.

```bash
git clone https://github.com/subjektz3ro/hey-eve.git
cd hey-eve
./deploy/install.sh
```

The installer checks compatibility before installing the service. It then:

1. synchronizes the locked production dependency set;
2. creates or preserves `~/.config/eve/env` with owner-only permissions;
3. downloads and verifies the Kokoro model and voice bank and the Silero VAD
   model;
4. checks out a pinned whisper.cpp revision, builds `whisper-cli`, and
   downloads the pinned `base.en` model;
5. generates speech with Kokoro, runs Silero inference, and transcribes the
   generated phrase with Whisper; and
6. validates the rendered systemd unit before installing or starting it.

Any failed dependency, hash, build, import, inference, or synthesis check exits
nonzero before the service starts. Rerunning the installer is safe: valid
artifacts are reused, configuration is preserved, and corrupt downloads are
replaced through temporary files.

The installer ignores inherited assistant and uv project-selection overrides
other than the bootstrap `EVE_CONFIG_DIR`, then pins uv to the current checkout
and its `.venv`. Put persistent installer and service settings in the selected
env file. Direct manual runs still let explicit process environment values win.

The configuration interview runs only when the settings file does not exist.
Every setting is documented in [`env.example`](../env.example). Move the file
aside if you want to repeat the interview, or edit it and rerun the installer.

## Files outside the checkout

The defaults are:

| Path | Purpose |
|---|---|
| `~/.config/eve/env` | API key and operator settings, mode `0600` |
| `~/.config/eve/memory.json` | Bounded remembered facts |
| `~/.config/eve/reminders.json` | Pending reminders |
| `~/.local/share/eve/kokoro/` | Kokoro model and voice bank |
| `~/.local/share/eve/models/` | Silero VAD model |
| `~/whisper.cpp/` | Pinned Whisper source, build, and `base.en` model |
| `/etc/systemd/system/eve@.service` | Rendered service unit |
| `/etc/systemd/system/eve-speaker@.*` | Optional Bluetooth reconnect units |
| `/etc/systemd/journald.conf.d/eve-retention.conf` | Optional host-wide journal retention policy |

`EVE_CONFIG_DIR`, `EVE_MODELS_DIR`, `KOKORO_DIR`, and `WHISPER_DIR` can move
these locations. A leading `~` is expanded; relative paths are resolved from
the checkout. The installer records the resolved configuration directory in
the service unit so the managed service and deploy operations read the same
file. For a manual command with a custom location, supply it again:

```bash
EVE_CONFIG_DIR=/srv/eve-config uv run eve doctor
```

A managed systemd installation validates the resolved checkout, configuration,
`uv` executable, and `uv` cache paths, then embeds them in its unit. The unit
also pins uv's project environment to the checkout's `.venv`. Those paths may
contain only ASCII letters, digits, `/`, `.`, `_`, `+`, and `-`; whitespace,
`%`, quotes, and backslashes are rejected before installation begins. The
documented defaults satisfy this rule on a conventional Linux account. Manual
operation does not use the rendered service unit.

The journald policy affects the whole host, not only Eve. When selected, it
sets a 14-day age limit and 200 MB system journal limit and restarts journald.

## Validate the installation

The runtime verifier does not open the microphone, speaker, framebuffer,
touchscreen, Anthropic API, or any network connection:

```bash
uv run eve doctor
```

It verifies all model hashes and the Whisper source revision, then executes
Kokoro, Silero, and Whisper together. Test the configured playback separately:

```bash
uv run python -m eve.main --say "hello" --no-display
```

For microphone, Bluetooth, panel, touchscreen, and console-switch setup, see
[`docs/hardware.md`](../docs/hardware.md).

## Service operation

The instance name is the account that owns the checkout:

```bash
sudo systemctl start "eve@$USER"
sudo systemctl stop "eve@$USER"
systemctl status "eve@$USER"
journalctl -u "eve@$USER" -f
```

The unit runs as that user rather than root. It uses the host's existing
`audio`, `video`, `tty`, `bluetooth`, and `input` groups when they exist. A
missing optional group is omitted so it cannot prevent startup.

The unit includes a memory ceiling, OOM stop policy, and bounded restart loop.
It does not use a systemd filesystem sandbox because the supported runtime
loads user-owned models and opens several hardware devices. See
[`SECURITY.md`](../SECURITY.md) for the exact boundary.

### Manual operation

Skip the systemd prompt to run Eve directly:

```bash
uv run eve doctor
uv run python -m eve.main
```

Use `--no-display` on a headless host. The installer still provisions and
verifies the complete local voice pipeline before reporting success.

## Updating an installation

For a normal public checkout, stop Eve before changing code or its environment,
fast-forward to the current release, and rerun the installer:

```bash
cd ~/hey-eve
sudo systemctl stop "eve@$USER"
git pull --ff-only
./deploy/install.sh
```

Stopping first prevents the existing process from importing or spawning a
worker from a partially updated checkout. Because the unit remains enabled,
the installer starts it only after the locked sync and runtime checks succeed.
If setup fails, the service remains stopped and the error identifies the
failed stage.

## Maintainer deployment

`ship.sh` deploys a commit only after proving it is reachable from the host's
configured `origin/main`. Push first, then run a dry run:

```bash
git push origin HEAD:main
./deploy/ship.sh --dry-run raspberrypi
./deploy/ship.sh raspberrypi
```

Set `EVE_DEPLOY_HOST` instead of passing the SSH target each time. Other
settings are:

| Variable | Default | Meaning |
|---|---|---|
| `EVE_DEPLOY_PATH` | `hey-eve` | Checkout path relative to the SSH account's home |
| `EVE_DEPLOY_SERVICE` | `eve@` | Instance service; a trailing `@` resolves to the remote account |
| `EVE_DEPLOY_REMOTE` | `origin` | Git remote fetched by the host |
| `EVE_DEPLOY_BRANCH` | `main` | Branch that must contain the requested commit |

An existing private installation using `~/eve` can keep that path with
`EVE_DEPLOY_PATH=eve`.

On the host, the deploy script:

1. verifies the installed unit contract before changing anything;
2. stops Eve;
3. hard-resets the dedicated deployment checkout to the requested commit;
4. runs `uv sync --locked --no-dev`;
5. runs the complete offline runtime verifier; and
6. starts the service and checks its status.

The hard reset is intentional for a dedicated deployment checkout. Do not use
that checkout for uncommitted work. Configuration and model data are outside
the Git tree and are preserved.

`ship.sh` does not install systemd units because applying a unit is a
root-equivalent operation. If the template or renderer changed, the command
stops before touching the checkout and asks the operator to rerun
`./deploy/install.sh` interactively on the host.

### Read access to a public origin

A public checkout needs no GitHub credential. HTTPS is the simplest remote:

```bash
git remote set-url origin https://github.com/subjektz3ro/hey-eve.git
```

Use an SSH deploy key only for a private fork. Give it read-only access to that
one repository; do not store a broad personal access token on the service host.

### Narrow sudo permissions

After installation, run `sudo -n -l` and remove blanket passwordless sudo if
present. A maintainer deployment needs only stop/start access for Eve's exact
unit. The optional face also needs `chvt 8` to move the Linux console away
from the framebuffer.

Use `command -v systemctl` and `command -v chvt` to find the actual paths, then
edit the rule with `visudo -f /etc/sudoers.d/eve`. A headless installation
does not need the `chvt` entry.

For example, replace the account, unit, and command paths below with the values
for the host:

```text
eveuser ALL=(root) NOPASSWD: /usr/bin/systemctl start eve@eveuser, /usr/bin/systemctl stop eve@eveuser
eveuser ALL=(root) NOPASSWD: /usr/bin/chvt 8
```

## Rollback

Choose a known-good commit from the public repository, stop the service, reset
the dedicated host checkout to that commit, and rerun its installer:

```bash
cd ~/hey-eve
sudo systemctl stop "eve@$USER"
git fetch origin main
git reset --hard <KNOWN_GOOD_COMMIT>
./deploy/install.sh
```

This discards tracked-file edits in the deployment checkout. It does not remove
the settings, memory, reminders, or downloaded models. A revision with a
different service-unit contract must be installed interactively, which the
installer handles.

## Uninstalling the service

Disable Eve and the optional speaker timer before removing any files:

```bash
sudo systemctl disable --now "eve@$USER"
sudo systemctl disable --now "eve-speaker@$USER.timer"
```

The checkout, `~/.config/eve`, `~/.local/share/eve`, `~/whisper.cpp`, systemd
unit files, and optional journald drop-in are deliberately left in place so an
uninstall does not delete credentials, memories, models, or host policy without
an explicit operator decision.
