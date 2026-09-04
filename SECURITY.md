# Security and privacy

Eve combines an always-open microphone, a remote language-model API, local
memory files, subprocesses, and an optional system service. This document
describes the implemented boundaries and the data that leaves the host.

## Supported versions

Security fixes are applied to the latest release and `main`. Older releases
are not maintained separately.

## Report a vulnerability

Use GitHub's **Report a vulnerability** button to open a private security
advisory for this repository. Do not include credentials, transcripts, private
hostnames, addresses, or other sensitive operator data in a public issue.

If private vulnerability reporting is unavailable, open a public issue that
contains no exploit or private details and asks the maintainer for a private
contact channel.

## What leaves the host

Speech is captured and transcribed locally. A transcript is sent to Anthropic
only after the wake matcher accepts it as an addressed request or during an
active follow-up window.

An addressed turn can send:

- the request transcript;
- the bounded recent conversation history;
- the remembered-facts block included in the system prompt;
- tool results returned to the model; and
- a search query and retrieved content when Anthropic's server-side web-search
  tool is used.

Silero voice activity detection, whisper.cpp transcription, Kokoro synthesis,
audio playback, reminders, memory storage, and face rendering run locally.
Eve does not expose an inbound network server.

## Local persistence

| Data | Location | Lifetime |
|---|---|---|
| Current capture | Owner-only temporary directory under `/dev/shm` on supported Linux; the OS temporary directory on other development platforms | Removed after normal turn cleanup; Linux tmpfs clears abrupt leftovers at reboot, while other platforms follow their OS temporary-file policy |
| Transcripts and replies | None in the managed service by default | Written to the journal with `VOICE_DEBUG=1`; interactive runs print them to the terminal, whose scrollback or session logging may retain them |
| Timings, token counts, and cost | systemd journal | Existing host policy; the installer can optionally apply a host-wide 14-day/200 MB limit |
| Remembered facts | `~/.config/eve/memory.json` | Until changed or forgotten; 40 entries maximum |
| Pending reminders | `~/.config/eve/reminders.json` | Until delivered or canceled; 20 entries maximum |
| Synthesized speech | Pipe to `aplay` | Not persisted |

The language model may automatically store a stable fact or preference when
it considers that information useful for future turns. Storage is not limited
to sentences that contain an explicit “remember” command. Stored values are
plain JSON and are included in future model prompts. Review or remove them
with the memory tools or by editing `memory.json` while Eve is stopped.

Reminder text is later read aloud verbatim. Review queued reminders with the
list-reminders tool when untrusted speech or web content may have influenced a
turn.

## Credentials and configuration

The supported service setup stores `ANTHROPIC_API_KEY` and the optional
`BARKEEP_TOKEN` in `~/.config/eve/env`, mode `0600`, inside an owner-only
directory. The file is parsed as data and is never sourced as shell code.

Credentials read from that file are returned on demand and are not copied into
`os.environ`, so Eve's `arecord`, `aplay`, Whisper, and display subprocesses do
not inherit them. If an operator supplies a credential directly in the process
environment, normal operating-system inheritance rules apply; the settings
file is the recommended service configuration.

All non-secret settings loaded from the file are restricted to documented Eve
prefixes. This prevents an accidental line such as `PATH=` or `LD_PRELOAD=`
from changing every child process.

Rotate a credential by editing the settings file and restarting the service:

```bash
sudo systemctl stop "eve@$USER"
sudo systemctl start "eve@$USER"
```

The optional Barkeep token is sent only to the configured `BARKEEP_URL`.
Loopback may use HTTP or Barkeep's generated self-signed HTTPS certificate.
Any non-loopback endpoint must use HTTPS with normal certificate verification;
install a private or self-signed certificate in the host trust store rather
than bypassing that check. API redirects are refused.

## Prompt injection and tool limits

Anyone within microphone range can address Eve. Web-search results also enter
the model context. Prompting alone cannot make either source trusted, so the
local tool surface is deliberately bounded:

| Tool | Local effect |
|---|---|
| Anthropic web search | Remote search, capped at three uses per turn |
| System status, local time, BUSY Bar status | Read-only |
| Remember / forget | One bounded JSON store |
| Set / list / cancel reminders | One bounded JSON store |

There is no model tool for arbitrary command execution, arbitrary filesystem
access, or arbitrary local/network writes. Tool names dispatch through an
explicit handler map, and unknown names are rejected.

## Service permissions

`eve@<user>.service` runs as the selected unprivileged login account. It needs
access to ALSA devices and, when enabled, the framebuffer, console, Bluetooth,
and input devices. The installer renders only supplementary groups that exist
on the host.

The service does not currently use systemd filesystem sandbox directives.
Eve loads model data from the user's home directory, opens hardware devices,
and may call `sudo -n chvt 8` for the optional framebuffer. Resource controls
still bound failure behavior: the unit has a memory ceiling, an OOM policy,
a reachable restart limit, and core dumps disabled. The optional Bluetooth
speaker helper also disables core dumps because it reads the shared settings
file.

If the display is enabled, grant only the exact `chvt 8` command required by
the renderer. `deploy/ship.sh` needs only stop/start access to the exact Eve
unit. Use `visudo` and avoid blanket `NOPASSWD: ALL` rules.

## Downloads and supply chain

The supported installer pins:

- Python packages through `uv.lock`;
- a specific whisper.cpp Git revision;
- the Whisper `base.en` model by SHA-256;
- the Kokoro model and voice bank by SHA-256; and
- the Silero VAD model by source revision and SHA-256.

Installation and deployment stop before service startup when a revision,
digest, import, or real inference/synthesis check fails. GitHub Actions pins
third-party actions by commit SHA, runs dependency auditing, and builds an
isolated wheel.

[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) records the upstream
projects and their licenses.

## Public-release hygiene

`scripts/check_release_hygiene.py` scans the exact staged Git blobs rather
than unsaved working-tree copies. It fails on an empty candidate, nonignored
untracked files, private/generated paths, common credential forms, private
network identifiers, personal coordinates, hostnames, MAC addresses, and
absolute home paths. It reports rule names and paths without printing matched
secret values.

The scanner complements review; it is not a general secret-management or
malware-analysis system. Before publishing a release, run the complete CI
suite from a clean checkout and inspect the Git commit metadata and remote
refs as well as the file tree.
