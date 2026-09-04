"""The hardware-free runtime check used by install and deploy."""
from __future__ import annotations

import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

from eve import config, doctor, speech, tts, vad


@pytest.fixture
def runtime_paths(tmp_path):
    kokoro = tmp_path / "models" / "kokoro"
    whisper = tmp_path / "whisper.cpp"
    paths = doctor.RuntimePaths(
        kokoro_model=kokoro / doctor.KOKORO_MODEL_NAME,
        kokoro_voices=kokoro / doctor.KOKORO_VOICES_NAME,
        silero_model=tmp_path / "models" / "models" / doctor.SILERO_MODEL_NAME,
        whisper_root=whisper,
        whisper_binary=whisper / doctor.WHISPER_BINARY_NAME,
        whisper_model=whisper / "models" / doctor.WHISPER_MODEL_NAME,
    )
    contents = {
        paths.kokoro_model: b"kokoro graph",
        paths.kokoro_voices: b"voice bank",
        paths.silero_model: b"silero graph",
        paths.whisper_model: b"whisper graph",
        paths.whisper_binary: b"#!/bin/sh\nexit 0\n",
    }
    for path, body in contents.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
    paths.whisper_binary.chmod(0o755)
    return paths


def _use_fixture_digests(monkeypatch, paths):
    monkeypatch.setattr(
        doctor, "KOKORO_MODEL_SHA256", doctor.sha256(paths.kokoro_model)
    )
    monkeypatch.setattr(
        doctor, "KOKORO_VOICES_SHA256", doctor.sha256(paths.kokoro_voices)
    )
    monkeypatch.setattr(
        doctor, "SILERO_MODEL_SHA256", doctor.sha256(paths.silero_model)
    )
    monkeypatch.setattr(
        doctor, "WHISPER_MODEL_SHA256", doctor.sha256(paths.whisper_model)
    )


class TestPinnedFiles:
    def test_every_artifact_is_hashed_and_the_source_is_checked(
        self, runtime_paths, monkeypatch
    ):
        _use_fixture_digests(monkeypatch, runtime_paths)
        inspected = []
        monkeypatch.setattr(
            doctor, "_verify_whisper_source", lambda root: inspected.append(root)
        )
        assert doctor.verify_files(runtime_paths) == runtime_paths
        assert inspected == [runtime_paths.whisper_root]

    def test_a_wrong_digest_is_a_hard_failure(self, runtime_paths, monkeypatch):
        _use_fixture_digests(monkeypatch, runtime_paths)
        runtime_paths.kokoro_voices.write_bytes(b"truncated")
        with pytest.raises(doctor.RuntimeCheckError, match="Kokoro voice bank has sha256"):
            doctor.verify_files(runtime_paths)

    def test_a_missing_file_names_the_exact_path(self, runtime_paths):
        runtime_paths.kokoro_model.unlink()
        with pytest.raises(doctor.RuntimeCheckError, match=str(runtime_paths.kokoro_model)):
            doctor.verify_files(runtime_paths)

    def test_the_whisper_binary_must_be_executable(self, runtime_paths, monkeypatch):
        _use_fixture_digests(monkeypatch, runtime_paths)
        runtime_paths.whisper_binary.chmod(0o644)
        with pytest.raises(doctor.RuntimeCheckError, match="not executable"):
            doctor.verify_files(runtime_paths)


class TestPinnedWhisperSource:
    @pytest.fixture
    def checkout(self, tmp_path, monkeypatch):
        root = tmp_path / "whisper.cpp"
        root.mkdir()
        subprocess.run(["git", "init", "-q", str(root)], check=True)
        subprocess.run(
            [
                "git", "-C", str(root), "config", "user.email",
                "test" + "@" + "example.invalid",
            ],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(root), "config", "user.name", "Test"], check=True
        )
        source = root / "source.cc"
        source.write_text("one\n")
        subprocess.run(["git", "-C", str(root), "add", "source.cc"], check=True)
        subprocess.run(
            ["git", "-C", str(root), "commit", "-qm", "fixture"], check=True
        )
        commit = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        monkeypatch.setattr(doctor, "WHISPER_SOURCE_COMMIT", commit)
        (root / doctor.WHISPER_SOURCE_MARKER).write_text(commit + "\n")
        return root, source, commit

    def test_a_clean_exact_checkout_passes(self, checkout):
        root, _, _ = checkout
        doctor._verify_whisper_source(root)

    def test_modified_tracked_source_is_refused(self, checkout):
        root, source, _ = checkout
        source.write_text("changed\n")
        with pytest.raises(doctor.RuntimeCheckError, match="modified tracked"):
            doctor._verify_whisper_source(root)

    def test_the_provisioning_marker_is_required(self, checkout):
        root, _, _ = checkout
        (root / doctor.WHISPER_SOURCE_MARKER).unlink()
        with pytest.raises(doctor.RuntimeCheckError, match="not provisioned by Eve"):
            doctor._verify_whisper_source(root)

    def test_the_marker_and_git_head_must_both_name_the_pin(
        self, checkout, monkeypatch
    ):
        root, _, commit = checkout
        other = "0" * 40
        monkeypatch.setattr(doctor, "WHISPER_SOURCE_COMMIT", other)
        (root / doctor.WHISPER_SOURCE_MARKER).write_text(other)
        with pytest.raises(doctor.RuntimeCheckError, match="source is not the pinned"):
            doctor._verify_whisper_source(root)
        (root / doctor.WHISPER_SOURCE_MARKER).write_text(commit)
        with pytest.raises(doctor.RuntimeCheckError, match="marker names"):
            doctor._verify_whisper_source(root)


class TestRealInferenceSeams:
    def _working_backends(self, runtime_paths, monkeypatch, *, probability=0.25,
                          transcript="This is Eve runtime verification."):
        pcm = np.full(config.TTS_RATE // 2, 1200, dtype=np.int16).tobytes()
        monkeypatch.setattr(tts, "MODEL_FILE", doctor.KOKORO_MODEL_NAME)
        monkeypatch.setattr(
            tts,
            "paths",
            lambda: (runtime_paths.kokoro_model, runtime_paths.kokoro_voices),
        )
        monkeypatch.setattr(tts, "synth", lambda text: pcm)

        seen = {}

        class Detector:
            def __init__(self, model):
                seen["silero_model"] = model

            def speech_probability(self, samples):
                seen["silero_samples"] = samples.copy()
                return probability

        class Transcriber:
            def __init__(self, binary, model):
                seen["whisper"] = (binary, model)

            def transcribe(self, wav_path):
                with wave.open(str(wav_path), "rb") as handle:
                    seen["wav"] = (
                        handle.getnchannels(),
                        handle.getframerate(),
                        handle.getnframes(),
                    )
                return transcript

        monkeypatch.setattr(vad, "SileroVAD", Detector)
        monkeypatch.setattr(speech, "WhisperTranscriber", Transcriber)
        return seen, len(pcm) // 2

    def test_one_generated_phrase_runs_through_all_three_engines(
        self, runtime_paths, monkeypatch
    ):
        seen, sample_count = self._working_backends(runtime_paths, monkeypatch)
        monkeypatch.setattr(config, "VOLUME", 0.0)
        samples, probability, transcript = doctor.exercise_runtime(runtime_paths)
        assert (samples, probability) == (sample_count, 0.25)
        assert transcript.startswith("This is Eve")
        assert config.VOLUME == 0.0
        assert seen["silero_model"] == runtime_paths.silero_model
        assert seen["silero_samples"].shape == (vad.WINDOW,)
        assert seen["whisper"] == (
            runtime_paths.whisper_binary,
            runtime_paths.whisper_model,
        )
        whisper_samples = int(sample_count * config.MIC_RATE / config.TTS_RATE)
        assert seen["wav"] == (1, config.MIC_RATE, whisper_samples)

    def test_resampling_preserves_signed_nontrivial_audio(self):
        source = np.array(
            [1200, -2400, 3600, -4800, 6000, -7200, 8400], dtype=np.int16
        )
        converted = np.frombuffer(
            doctor._resample_s16(source.tobytes(), 7, 4), dtype=np.int16
        )
        assert converted.tolist() == [1200, -2400, -4800, -7200]
        assert converted.dtype == np.int16
        assert np.any(converted > 0) and np.any(converted < 0)

    def test_silent_kokoro_output_is_refused(self, runtime_paths, monkeypatch):
        self._working_backends(runtime_paths, monkeypatch)
        monkeypatch.setattr(
            tts, "synth", lambda text: np.zeros(config.TTS_RATE, np.int16).tobytes()
        )
        with pytest.raises(doctor.RuntimeCheckError, match="effectively silent"):
            doctor.exercise_runtime(runtime_paths)

    def test_an_invalid_silero_result_is_refused(self, runtime_paths, monkeypatch):
        self._working_backends(runtime_paths, monkeypatch, probability=float("nan"))
        with pytest.raises(doctor.RuntimeCheckError, match="invalid probability"):
            doctor.exercise_runtime(runtime_paths)

    def test_an_empty_whisper_transcript_is_refused(
        self, runtime_paths, monkeypatch
    ):
        self._working_backends(runtime_paths, monkeypatch, transcript="   ")
        with pytest.raises(doctor.RuntimeCheckError, match="no transcript"):
            doctor.exercise_runtime(runtime_paths)

    def test_unrelated_whisper_output_does_not_count_as_compatibility(
        self, runtime_paths, monkeypatch
    ):
        self._working_backends(runtime_paths, monkeypatch, transcript="random words")
        with pytest.raises(doctor.RuntimeCheckError, match="did not recognize"):
            doctor.exercise_runtime(runtime_paths)

    def test_an_alternate_kokoro_graph_is_outside_the_contract(
        self, runtime_paths, monkeypatch
    ):
        monkeypatch.setattr(tts, "MODEL_FILE", "kokoro-v1.0.fp16.onnx")
        with pytest.raises(doctor.RuntimeCheckError, match="outside the supported"):
            doctor.exercise_runtime(runtime_paths)


class TestCommand:
    def test_show_exposes_the_single_path_manifest(self, monkeypatch, capsys):
        monkeypatch.setattr(config, "MODELS_DIR", Path("/models"))
        monkeypatch.setattr(config, "kokoro_dir", lambda: Path("/models/kokoro"))
        assert doctor.main(["--show", "kokoro-dir"]) == 0
        assert capsys.readouterr().out == "/models/kokoro\n"

    def test_success_reports_each_exercised_engine(self, monkeypatch, capsys):
        monkeypatch.setattr(
            doctor,
            "verify_runtime",
            lambda: ["Kokoro worked", "Silero worked", "Whisper worked"],
        )
        assert doctor.main([]) == 0
        assert capsys.readouterr().out.count("ok:") == 3

    def test_any_backend_failure_is_a_clean_nonzero_result(
        self, monkeypatch, capsys
    ):
        def fail():
            raise RuntimeError("provider would not load")

        monkeypatch.setattr(doctor, "verify_runtime", fail)
        assert doctor.main([]) == 1
        assert "provider would not load" in capsys.readouterr().err

    def test_verify_runtime_reports_concrete_results(self, runtime_paths, monkeypatch):
        monkeypatch.setattr(doctor, "verify_files", lambda: runtime_paths)
        monkeypatch.setattr(
            doctor,
            "exercise_runtime",
            lambda paths: (22050, 0.125, "runtime verification"),
        )
        messages = doctor.verify_runtime()
        assert len(messages) == 4
        assert any("22050" in message for message in messages)
        assert any("0.125000" in message for message in messages)
        assert any("runtime verification" in message for message in messages)

    def test_eve_doctor_is_the_canonical_entrypoint(self, monkeypatch):
        from eve import main

        called = []
        monkeypatch.setattr(doctor, "main", lambda argv: called.append(argv) or 7)
        monkeypatch.setattr(main.signal, "signal", lambda *args: None)
        monkeypatch.setattr(sys, "argv", ["eve", "doctor"])
        assert main.main() == 7
        assert called == [[]]

    def test_eve_doctor_verifies_files_before_silero_can_load(
        self, monkeypatch, capsys
    ):
        from eve import main

        loaded = []
        monkeypatch.setattr(speech, "_DETECTOR", speech._UNLOADED_DETECTOR)
        monkeypatch.setattr(vad, "load", lambda: loaded.append(True))

        def reject_unverified_files():
            raise doctor.RuntimeCheckError("bad model digest")

        monkeypatch.setattr(doctor, "verify_files", reject_unverified_files)
        monkeypatch.setattr(main.signal, "signal", lambda *args: None)
        monkeypatch.setattr(sys, "argv", ["eve", "doctor"])

        assert main.main() == 1
        assert loaded == []
        assert "bad model digest" in capsys.readouterr().err
