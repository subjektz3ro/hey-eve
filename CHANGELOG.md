# Changelog

## 0.1.2 — Initial public release

- Added the always-listening voice loop: local Silero voice activity detection,
  local whisper.cpp transcription, Claude responses and bounded tools, and
  local Kokoro speech synthesis.
- Added the optional 480×320 framebuffer face, visible microphone and sleep
  states, speech-timed animation, idle motion, and touchscreen reactions.
- Added bounded local memory and reminders plus an optional read-only BUSY Bar
  status integration through Barkeep.
- Added the supported glibc Linux installer for CPython 3.11–3.13 on `x86_64`
  and `aarch64`. It locks Python dependencies, pins and verifies downloaded
  models and whisper.cpp source, and runs the complete local inference chain
  before installing or starting the systemd service.
- Added privacy and security controls for temporary audio, credentials,
  content logging, model tools, downloaded artifacts, and service operation.
- Added hardware-free tests, static analysis, dependency auditing, release
  hygiene checks, isolated-wheel verification, and a stable CI gate.
- Added installation, hardware, security, contribution, support, and
  third-party licensing documentation for a fresh public checkout.
