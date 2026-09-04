"""Verify Eve's local speech runtime without opening any hardware.

The installer and deployer both call this module.  A successful check means
the exact model bytes are present, Kokoro can make non-silent speech, Silero
can execute an inference, and the pinned whisper.cpp binary can transcribe
that speech with the pinned model.  It never opens ALSA, the framebuffer, an
input device, or the Anthropic API.
"""
from __future__ import annotations

import argparse
import hashlib
import math
import os
import re
import subprocess
import sys
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path

from eve import config

KOKORO_MODEL_NAME = "kokoro-v1.0.onnx"
KOKORO_VOICES_NAME = "voices-v1.0.bin"
SILERO_MODEL_NAME = "silero_vad.onnx"
WHISPER_MODEL_NAME = "ggml-base.en.bin"
WHISPER_BINARY_NAME = "build/bin/whisper-cli"
WHISPER_SOURCE_MARKER = ".eve-source-commit"
RUNTIME_PHRASE = "This is Eve runtime verification."

KOKORO_BASE_URL = (
    "https://github.com/thewh1teagle/kokoro-onnx/releases/download/"
    "model-files-v1.0"
)
KOKORO_MODEL_URL = f"{KOKORO_BASE_URL}/{KOKORO_MODEL_NAME}"
KOKORO_VOICES_URL = f"{KOKORO_BASE_URL}/{KOKORO_VOICES_NAME}"
SILERO_SOURCE_COMMIT = "bfdc0193023f121ea5b3cc7b176dbed570a68a59"
SILERO_MODEL_URL = (
    "https://raw.githubusercontent.com/snakers4/silero-vad/"
    f"{SILERO_SOURCE_COMMIT}/src/silero_vad/data/{SILERO_MODEL_NAME}"
)
WHISPER_SOURCE_URL = "https://github.com/ggml-org/whisper.cpp.git"
# This is the exact source revision exercised by the live Pi on 2026-08-07.
# A newer release is not an upgrade until the same end-to-end check below has
# been run against it on the supported host.
WHISPER_SOURCE_COMMIT = "592feef04a1802b18cbeffd0fd0eb5d02570c2ec"
WHISPER_MODEL_REVISION = "80da2d8bfee42b0e836fc3a9890373e5defc00a6"
WHISPER_MODEL_URL = (
    "https://huggingface.co/ggerganov/whisper.cpp/resolve/"
    f"{WHISPER_MODEL_REVISION}/{WHISPER_MODEL_NAME}"
)

KOKORO_MODEL_SHA256 = (
    "7d5df8ecf7d4b1878015a32686053fd0eebe2bc377234608764cc0ef3636a6c5"
)
KOKORO_VOICES_SHA256 = (
    "bca610b8308e8d99f32e6fe4197e7ec01679264efed0cac9140fe9c29f1fbf7d"
)
SILERO_MODEL_SHA256 = (
    "1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3"
)
WHISPER_MODEL_SHA256 = (
    "a03779c86df3323075f5e796cb2ce5029f00ec8869eee3fdfb897afe36c6d002"
)


class RuntimeCheckError(RuntimeError):
    """The local model runtime is absent, corrupt, or incompatible."""


@dataclass(frozen=True)
class RuntimePaths:
    kokoro_model: Path
    kokoro_voices: Path
    silero_model: Path
    whisper_root: Path
    whisper_binary: Path
    whisper_model: Path


def runtime_paths() -> RuntimePaths:
    """Resolve every runtime artifact through config's shared path contract."""
    return RuntimePaths(
        kokoro_model=config.kokoro_dir() / KOKORO_MODEL_NAME,
        kokoro_voices=config.kokoro_dir() / KOKORO_VOICES_NAME,
        silero_model=config.silero_model(),
        whisper_root=config.whisper_dir(),
        whisper_binary=config.whisper_dir() / WHISPER_BINARY_NAME,
        whisper_model=config.whisper_dir() / "models" / WHISPER_MODEL_NAME,
    )


def sha256(path: Path) -> str:
    """Hash a large artifact without reading it all into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_digest(label: str, path: Path, expected: str) -> None:
    if not path.is_file():
        raise RuntimeCheckError(f"{label} is missing: {path}")
    actual = sha256(path)
    if actual != expected:
        raise RuntimeCheckError(
            f"{label} has sha256 {actual}, expected {expected}: {path}"
        )


def _verify_whisper_source(root: Path) -> None:
    marker = root / WHISPER_SOURCE_MARKER
    try:
        marked = marker.read_text().strip()
    except OSError as exc:
        raise RuntimeCheckError(
            f"whisper.cpp was not provisioned by Eve ({marker} is missing)"
        ) from exc
    if marked != WHISPER_SOURCE_COMMIT:
        raise RuntimeCheckError(
            f"whisper.cpp marker names {marked or 'nothing'}, expected "
            f"{WHISPER_SOURCE_COMMIT}"
        )

    try:
        head = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeCheckError(f"could not inspect whisper.cpp source: {exc}") from exc
    if head.returncode != 0 or head.stdout.strip() != WHISPER_SOURCE_COMMIT:
        raise RuntimeCheckError(
            "whisper.cpp source is not the pinned commit " + WHISPER_SOURCE_COMMIT
        )
    try:
        changed = subprocess.run(
            [
                "git", "-C", str(root), "diff", "--quiet",
                WHISPER_SOURCE_COMMIT, "--",
            ],
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeCheckError(f"could not inspect whisper.cpp source: {exc}") from exc
    if changed.returncode != 0:
        raise RuntimeCheckError("whisper.cpp has modified tracked source files")


def verify_files(paths: RuntimePaths | None = None) -> RuntimePaths:
    """Verify exact bytes and the pinned whisper.cpp source revision."""
    paths = paths or runtime_paths()
    _require_digest(
        "Kokoro model", paths.kokoro_model, KOKORO_MODEL_SHA256
    )
    _require_digest(
        "Kokoro voice bank", paths.kokoro_voices, KOKORO_VOICES_SHA256
    )
    _require_digest("Silero model", paths.silero_model, SILERO_MODEL_SHA256)
    _require_digest("Whisper model", paths.whisper_model, WHISPER_MODEL_SHA256)
    if not paths.whisper_binary.is_file() or not os.access(
        paths.whisper_binary, os.X_OK
    ):
        raise RuntimeCheckError(
            f"Whisper binary is missing or not executable: {paths.whisper_binary}"
        )
    _verify_whisper_source(paths.whisper_root)
    return paths


def _resample_s16(pcm: bytes, source_rate: int, target_rate: int) -> bytes:
    """Nearest-sample mono s16 conversion for Whisper's 16 kHz WAV input."""
    import numpy as np

    samples = np.frombuffer(pcm, dtype=np.int16)
    if not samples.size or source_rate == target_rate:
        return pcm
    output_count = int(samples.size * target_rate / source_rate)
    indexes = np.minimum(
        (
            np.arange(output_count, dtype=np.float64)
            * source_rate
            / target_rate
        ).astype(np.int64),
        samples.size - 1,
    )
    return samples[indexes].tobytes()


def exercise_runtime(paths: RuntimePaths) -> tuple[int, float, str]:
    """Run all three inference engines against one generated spoken phrase."""
    import numpy as np

    from eve import speech, tts, vad

    if tts.MODEL_FILE != KOKORO_MODEL_NAME:
        raise RuntimeCheckError(
            f"KOKORO_MODEL={tts.MODEL_FILE!r} is outside the supported runtime "
            f"contract; use {KOKORO_MODEL_NAME!r}"
        )
    found = tts.paths()
    if found != (paths.kokoro_model, paths.kokoro_voices):
        raise RuntimeCheckError("Kokoro is not using the verified model and voice bank")

    # VOICE_VOLUME=0 is a valid operator preference, not a silent model.  The
    # check needs the graph's output before that preference is applied.
    configured_volume = config.VOLUME
    try:
        config.VOLUME = 1.0
        pcm = tts.synth(RUNTIME_PHRASE)
    finally:
        config.VOLUME = configured_volume
    samples = np.frombuffer(pcm, dtype=np.int16)
    amplitudes = np.abs(samples.astype(np.int32))
    if samples.size < config.TTS_RATE // 4 or not np.any(amplitudes > 256):
        raise RuntimeCheckError("Kokoro produced empty or effectively silent audio")

    detector = vad.SileroVAD(paths.silero_model)
    probability = detector.speech_probability(
        np.zeros(vad.WINDOW, dtype=np.int16)
    )
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise RuntimeCheckError(f"Silero returned invalid probability {probability!r}")

    whisper_pcm = _resample_s16(pcm, config.TTS_RATE, config.MIC_RATE)
    with tempfile.TemporaryDirectory(prefix="eve-doctor-") as temporary:
        wav_path = Path(temporary) / "kokoro-check.wav"
        with wave.open(str(wav_path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(config.MIC_RATE)
            handle.writeframes(whisper_pcm)
        transcript = speech.WhisperTranscriber(
            paths.whisper_binary, paths.whisper_model
        ).transcribe(wav_path).strip()
    if not transcript:
        raise RuntimeCheckError(
            "Whisper loaded but returned no transcript for Kokoro speech"
        )
    expected_words = re.findall(r"[a-z0-9]+", RUNTIME_PHRASE.lower())
    heard_words = re.findall(r"[a-z0-9]+", transcript.lower())
    if heard_words != expected_words:
        raise RuntimeCheckError(
            "Whisper ran but did not recognize the generated verification "
            f"phrase: {transcript!r}"
        )
    return samples.size, probability, transcript


def verify_runtime() -> list[str]:
    """Verify artifact integrity and execute the complete local model chain."""
    paths = verify_files()
    samples, probability, transcript = exercise_runtime(paths)
    return [
        "artifact hashes match the supported runtime",
        f"Kokoro produced {samples} non-silent samples",
        f"Silero inference returned {probability:.6f}",
        f"Whisper transcribed generated speech as {transcript!r}",
    ]


def show_values() -> dict[str, str]:
    """Values shell provisioners need, without duplicating path or pin logic."""
    paths = runtime_paths()
    return {
        "config-dir": str(config.CONFIG_DIR),
        "models-dir": str(config.MODELS_DIR),
        "kokoro-dir": str(paths.kokoro_model.parent),
        "kokoro-model": str(paths.kokoro_model),
        "kokoro-voices": str(paths.kokoro_voices),
        "silero-model": str(paths.silero_model),
        "whisper-dir": str(paths.whisper_root),
        "whisper-binary": str(paths.whisper_binary),
        "whisper-model": str(paths.whisper_model),
        "kokoro-model-url": KOKORO_MODEL_URL,
        "kokoro-voices-url": KOKORO_VOICES_URL,
        "silero-model-url": SILERO_MODEL_URL,
        "whisper-source-url": WHISPER_SOURCE_URL,
        "whisper-source-commit": WHISPER_SOURCE_COMMIT,
        "whisper-model-url": WHISPER_MODEL_URL,
        "kokoro-model-sha256": KOKORO_MODEL_SHA256,
        "kokoro-voices-sha256": KOKORO_VOICES_SHA256,
        "silero-model-sha256": SILERO_MODEL_SHA256,
        "whisper-model-sha256": WHISPER_MODEL_SHA256,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--show",
        choices=sorted(show_values()),
        help="print one resolved path or immutable artifact value and exit",
    )
    args = parser.parse_args(argv)
    if args.show:
        print(show_values()[args.show])
        return 0
    try:
        messages = verify_runtime()
    # This is the install/deploy boundary: an ONNX provider error, an absent
    # shared library, or a failed child process must be one concise failed
    # check, not a traceback followed by an accidentally started service.
    except Exception as exc:
        print(f"eve runtime check failed: {exc}", file=sys.stderr)
        return 1
    for message in messages:
        print(f"ok: {message}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
