# Hardware and audio setup

Eve's voice pipeline needs a microphone and an ALSA playback device. The
480×320 face and touchscreen are optional. A manual run can use `--no-display`;
the managed service has no display flag and continues its voice loop when the
framebuffer is unavailable.

## Supported service host

The automated Linux installation supports:

- 64-bit `x86_64` or `aarch64` Linux with glibc 2.28 or newer;
- CPython 3.11, 3.12, or 3.13, installed through `uv`;
- systemd 243 or newer for the managed service;
- at least 4 GiB of RAM and 2 GiB of free disk space are recommended;
- `git`, `curl`, `bash`, CMake, a C/C++ toolchain, and ALSA utilities.

The tested reference build is an 8 GiB Raspberry Pi 5 running 64-bit Raspberry
Pi OS. Eve's core voice loop is not Pi-specific. Alpine/musl, 32-bit Raspberry
Pi OS, and Python 3.14 are outside the supported dependency set.

## Microphone and speaker

USB audio and a wired or USB speaker are the simplest configuration. Verify
both directions before installing the service:

```bash
arecord -l
aplay -L
arecord -D plughw:CARD=YOUR_CARD,DEV=0 -f S16_LE -r 16000 -c 1 -d 3 /tmp/eve-mic.wav
aplay /tmp/eve-mic.wav
```

Set `VOICE_MIC` to the capture device shown by `arecord -l`. If it is blank,
Eve discovers the first capture card at startup.

### Bluetooth playback

Bluetooth is optional. The working reference setup uses BlueZ and BlueALSA;
PipeWire or PulseAudio setups should expose their own ALSA-compatible default
device instead.

Install and start BlueZ and BlueALSA using the packages for your distribution,
then pair, trust, and connect the speaker:

```text
bluetoothctl
power on
scan on
pair <SPEAKER_MAC>
trust <SPEAKER_MAC>
connect <SPEAKER_MAC>
quit
```

The reference BlueALSA configuration routes ALSA's default playback to that
address:

```text
# /etc/asound.conf
defaults.bluealsa.service "org.bluealsa"
defaults.bluealsa.device "<SPEAKER_MAC>"
defaults.bluealsa.profile "a2dp"

pcm.!default {
    type plug
    slave.pcm {
        type bluealsa
        device "<SPEAKER_MAC>"
        profile "a2dp"
    }
}
```

After `VOICE_SPEAKER_MAC` is set, the installer can enable Eve's timer that
reconnects the speaker after boot and once per minute when disconnected.
Confirm playback with `aplay` before relying on the timer.

## 480×320 face

The face renderer expects a Linux framebuffer with these properties:

| Property | Required value |
|---|---|
| Device | `/dev/fb0` |
| Geometry | 480×320 |
| Pixel format | RGB565, little-endian |
| Line stride | 960 bytes |

The validated panel is an MHS35IPS-compatible 3.5-inch SPI display using an
ILI9486 display controller and ADS7846 touch controller. The working Raspberry
Pi boot configuration contains:

```text
dtparam=spi=on
dtoverlay=mhs35ips:rotate=90
```

Reboot after changing the boot overlay, then verify the driver and geometry.
If the commands are missing, install your distribution's `fbset` package and
the `kbd` or `console-tools` package that provides `chvt`.

```bash
ls -l /dev/fb0
fbset -fb /dev/fb0
grep -E 'fb_ili9486|ads7846' /proc/modules
```

The service account needs access to the framebuffer, console, sound devices,
Bluetooth stack when used, and touchscreen when enabled. The generated unit
adds only groups that exist on the host from this set: `video`, `tty`, `audio`,
`bluetooth`, and `input`.

The renderer switches away from the Linux console with `sudo -n chvt 8` so a
cursor is not drawn over the face. If the panel is enabled, allow that one
command for the service account using `visudo`; do not grant blanket passwordless
sudo. A headless managed service continues when no framebuffer is available
and does not need a `chvt` sudo rule. For a manual run, `--no-display` skips
the renderer entirely.

## Touchscreen

Touch input is enabled when a compatible device is available. Set
`VOICE_TOUCH=0` to disable it. After the ADS7846 input device appears, make
sure the service account belongs to the `input` group, then run:

```bash
uv run python scripts/touch-probe.py
```

Touch the requested corners and copy the printed `VOICE_TOUCH_SWAP`,
`VOICE_TOUCH_FLIP_X`, and `VOICE_TOUCH_FLIP_Y` values into
`~/.config/eve/env`. Set `VOICE_TOUCH=1` and restart Eve.

## First hardware check

After installation, use these checks in order:

```bash
uv run eve doctor
uv run python -m eve.main --say "hello" --no-display
systemctl status "eve@$USER"
journalctl -u "eve@$USER" -f
```

The doctor command validates software and model compatibility. The spoken
line validates the configured playback path. For a managed install, the
installer has already started and verified the service; its log then shows the
selected microphone, display status, and any unavailable optional hardware.
