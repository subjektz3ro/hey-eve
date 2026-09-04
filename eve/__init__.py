"""An always-listening voice assistant with local speech processing and an animated face."""

# Importing config first is deliberate: reading it applies ~/.config/eve/env,
# and most settings in that file are captured into a module constant at import
# — speech._HANG_S, tts.KOKORO_DIR, log.DEBUG. Anything later would be too
# late for them. See config.load_settings for what that file was doing until
# now, which was very little.
from eve import config as _config

SETTINGS_APPLIED = _config.SETTINGS_APPLIED
