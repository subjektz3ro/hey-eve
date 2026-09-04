"""An always-listening voice assistant with local speech processing and an animated face."""

import os as _os

# ONNX Runtime 1.29 made telemetry default-on in its official POSIX builds.
# This assignment must precede every Eve import that can reach either Silero
# or Kokoro: the runtime reads it during native initialization, and disabling
# telemetry through its Python API can already be too late. Assignment rather
# than setdefault is deliberate. Local inference without third-party runtime
# telemetry is an application privacy boundary, not an operator preference.
_os.environ["ORT_DISABLE_TELEMETRY"] = "1"

# Importing config first is deliberate: reading it applies ~/.config/eve/env,
# and most settings in that file are captured into a module constant at import
# — speech._HANG_S, tts.KOKORO_DIR, log.DEBUG. Anything later would be too
# late for them. See config.load_settings for what that file was doing until
# now, which was very little.
from eve import config as _config

SETTINGS_APPLIED = _config.SETTINGS_APPLIED
