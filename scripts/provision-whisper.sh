#!/usr/bin/env bash
# Provision the exact whisper.cpp source and base.en model Eve supports.
#
# This owns only the configured WHISPER_DIR.  An existing checkout with
# modified tracked files is refused rather than overwritten.
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

show() {
  "$UV_BIN" run --no-sync python -m eve.doctor --show "$1"
}

WHISPER_ROOT=$(show whisper-dir)
WHISPER_BINARY=$(show whisper-binary)
WHISPER_MODEL=$(show whisper-model)
SOURCE_URL=$(show whisper-source-url)
SOURCE_COMMIT=$(show whisper-source-commit)
MODEL_URL=$(show whisper-model-url)
MODEL_SHA=$(show whisper-model-sha256)

case "$WHISPER_ROOT" in
  ""|/|"${HOME:-}")
    echo "error: refusing unsafe WHISPER_DIR: $WHISPER_ROOT" >&2
    exit 1;;
esac
if [ "$WHISPER_ROOT" = "$REPO_DIR" ]; then
  echo "error: WHISPER_DIR resolves to the Eve checkout; refusing to replace it" >&2
  exit 1
fi

for tool in git curl cmake; do
  command -v "$tool" >/dev/null || {
    echo "error: $tool is required to provision whisper.cpp" >&2
    exit 1
  }
done
if ! command -v c++ >/dev/null && ! command -v g++ >/dev/null \
   && ! command -v clang++ >/dev/null; then
  echo "error: a C++ compiler is required to build whisper.cpp" >&2
  exit 1
fi

if command -v sha256sum >/dev/null; then
  digest() { sha256sum "$1" | cut -d" " -f1; }
elif command -v shasum >/dev/null; then
  digest() { shasum -a 256 "$1" | cut -d" " -f1; }
else
  echo "error: need sha256sum or shasum to verify the Whisper model" >&2
  exit 1
fi

if [ -e "$WHISPER_ROOT" ] && [ ! -d "$WHISPER_ROOT/.git" ]; then
  if [ ! -d "$WHISPER_ROOT" ] || [ -n "$(find "$WHISPER_ROOT" -mindepth 1 -maxdepth 1 -print -quit)" ]; then
    echo "error: WHISPER_DIR exists but is not a whisper.cpp git checkout:" >&2
    echo "  $WHISPER_ROOT" >&2
    echo "Move it aside or choose another WHISPER_DIR; nothing was overwritten." >&2
    exit 1
  fi
fi

if [ -d "$WHISPER_ROOT/.git" ]; then
  existing_origin=$(git -C "$WHISPER_ROOT" remote get-url origin 2>/dev/null || true)
  # A repository name is not an identity: an unrelated or attacker-controlled
  # host can also publish something ending in whisper.cpp.  Reuse only the
  # official upstream HTTPS and SSH spellings.  The Eve marker is evidence of
  # the last completed provision, not permission to ignore a changed remote.
  case "$existing_origin" in
    https://github.com/ggml-org/whisper.cpp|\
    https://github.com/ggml-org/whisper.cpp.git|\
    git@github.com:ggml-org/whisper.cpp|\
    git@github.com:ggml-org/whisper.cpp.git|\
    ssh://git@github.com/ggml-org/whisper.cpp|\
    ssh://git@github.com/ggml-org/whisper.cpp.git) ;;
    *)
      echo "error: WHISPER_DIR is a Git checkout, but not the official whisper.cpp checkout:" >&2
      echo "  $WHISPER_ROOT" >&2
      echo "  origin: ${existing_origin:-missing}" >&2
      echo "Set WHISPER_DIR correctly; no checkout was changed." >&2
      exit 1;;
  esac
fi

if [ ! -d "$WHISPER_ROOT/.git" ]; then
  mkdir -p "$(dirname "$WHISPER_ROOT")"
  git clone --filter=blob:none --no-checkout "$SOURCE_URL" "$WHISPER_ROOT"
elif [ -n "$(git -C "$WHISPER_ROOT" status --porcelain --untracked-files=no)" ]; then
  echo "error: whisper.cpp has modified tracked files; refusing to overwrite them" >&2
  exit 1
fi

echo "provisioning whisper.cpp at $SOURCE_COMMIT"
git -C "$WHISPER_ROOT" fetch --quiet --depth 1 "$SOURCE_URL" "$SOURCE_COMMIT"
git -C "$WHISPER_ROOT" checkout --quiet --detach "$SOURCE_COMMIT"
git -C "$WHISPER_ROOT" submodule update --init --recursive

jobs=$(getconf _NPROCESSORS_ONLN 2>/dev/null || true)
case "$jobs" in ""|*[!0-9]*) jobs=2;; esac
cmake -S "$WHISPER_ROOT" -B "$WHISPER_ROOT/build" \
  -DCMAKE_BUILD_TYPE=Release -DWHISPER_BUILD_TESTS=OFF
cmake --build "$WHISPER_ROOT/build" --parallel "$jobs" --target whisper-cli
if [ ! -x "$WHISPER_BINARY" ]; then
  echo "error: build completed without $WHISPER_BINARY" >&2
  exit 1
fi

mkdir -p "$(dirname "$WHISPER_MODEL")"
if [ -f "$WHISPER_MODEL" ] && [ "$(digest "$WHISPER_MODEL")" = "$MODEL_SHA" ]; then
  echo "have $(basename "$WHISPER_MODEL")"
else
  model_part="$(dirname "$WHISPER_MODEL")/.$(basename "$WHISPER_MODEL").part"
  echo "fetching $(basename "$WHISPER_MODEL")"
  rm -f -- "$model_part"
  if curl -fL --progress-bar --retry 3 --retry-delay 2 \
       -o "$model_part" "$MODEL_URL" \
     && [ "$(digest "$model_part")" = "$MODEL_SHA" ]; then
    mv -f -- "$model_part" "$WHISPER_MODEL"
  else
    rm -f -- "$model_part"
    echo "error: Whisper model download failed its pinned SHA-256" >&2
    exit 1
  fi
fi

marker="$WHISPER_ROOT/.eve-source-commit"
marker_part="$WHISPER_ROOT/.eve-source-commit.part"
printf '%s\n' "$SOURCE_COMMIT" > "$marker_part"
mv -f -- "$marker_part" "$marker"
echo "whisper.cpp runtime ready in $WHISPER_ROOT"
