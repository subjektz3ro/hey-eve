#!/usr/bin/env bash
# Fetch the model files the assistant needs. They are 356MB together,
# which is why they are not in git.
#
#   Kokoro-82M  -- text to speech, Apache-2.0
#   Silero VAD  -- tells speech from door slams, MIT
#
# Every file is pinned by SHA-256 and verified after download. The previous
# version tested `[ -s "$2" ]` — non-empty — and called that "have it", which
# means a download interrupted at 40MB of 325MB was indistinguishable from a
# complete one and could never be repaired by re-running this script. A bad
# Kokoro file does not fail here; it fails much later, inside the first spoken
# reply, which is the worst place to discover it.
set -euo pipefail
cd "$(dirname "$0")/.."
REPO_DIR=$(pwd -P)

# Always use the virtual environment owned by this checkout.
unset UV_WORKING_DIR UV_CONFIG_FILE UV_ENV_FILE UV_NO_PROJECT
export UV_PROJECT="$REPO_DIR"
export UV_PROJECT_ENVIRONMENT="$REPO_DIR/.venv"

UV_BIN="${UV_BIN:-$(type -P uv || true)}"
if [ -z "$UV_BIN" ] && [ -x "${HOME:-}/.local/bin/uv" ]; then
  UV_BIN="$HOME/.local/bin/uv"
fi
if [ -z "$UV_BIN" ] || [ ! -x "$UV_BIN" ]; then
  echo "error: uv is required; sync the locked Eve environment first" >&2
  exit 1
fi

# Python owns path expansion and the immutable artifact manifest.  Reading it
# here keeps a KOKORO_DIR override, an EVE_MODELS_DIR override, the runtime,
# and the installer on exactly the same files.
show() {
  "$UV_BIN" run --no-sync python -m eve.doctor --show "$1"
}

kokoro_model=$(show kokoro-model)
kokoro_voices=$(show kokoro-voices)
silero_model=$(show silero-model)
kokoro_model_url=$(show kokoro-model-url)
kokoro_voices_url=$(show kokoro-voices-url)
silero_model_url=$(show silero-model-url)
kokoro_model_sha=$(show kokoro-model-sha256)
kokoro_voices_sha=$(show kokoro-voices-sha256)
silero_sha=$(show silero-model-sha256)

mkdir -p "$(dirname "$kokoro_model")" "$(dirname "$silero_model")"

# BSD (macOS) and GNU (Linux) ship different tools for this, and neither is
# present everywhere. Pick one once rather than at every call site.
if command -v sha256sum >/dev/null; then
  digest() { sha256sum "$1" | cut -d" " -f1; }
elif command -v shasum >/dev/null; then
  digest() { shasum -a 256 "$1" | cut -d" " -f1; }
else
  echo "error: need sha256sum or shasum to verify downloads" >&2
  exit 1
fi

fetch() {  # url, path, expected-sha256
  local url="$1" path="$2" want="$3" part="$(dirname "$2")/.$(basename "$2").part"

  if [ -f "$path" ]; then
    if [ "$(digest "$path")" = "$want" ]; then
      echo "have $(basename "$path")"
      return
    fi
    # Ours to own, and its pinned digest did not match. Remove it before
    # retrying, so a failed repair leaves nothing rather than leaving bytes
    # the runtime would happily open and mis-decode.
    echo "$(basename "$path") is corrupt or outdated — refetching"
    rm -f "$path"
  fi

  echo "fetching $(basename "$path")"
  rm -f "$part"
  if curl -fL --progress-bar --retry 3 --retry-delay 2 -o "$part" "$url" \
     && [ "$(digest "$part")" = "$want" ]; then
    mv -f "$part" "$path"
  else
    rm -f "$part"
    echo "error: $(basename "$path") failed to download or did not match its" \
         "expected checksum" >&2
    exit 1
  fi
}

fetch "$kokoro_model_url" "$kokoro_model" "$kokoro_model_sha"
fetch "$kokoro_voices_url" "$kokoro_voices" "$kokoro_voices_sha"
fetch "$silero_model_url" "$silero_model" "$silero_sha"

echo "Kokoro models in $(dirname "$kokoro_model")"
echo "Silero model at $silero_model"
