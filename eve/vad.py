"""Telling speech from everything else, with Silero VAD.

Replaces a loudness threshold, which cannot distinguish a spoken word from a
door closing, a keyboard, or music — all of which were waking the assistant
and costing two seconds of Whisper each time.

Runs on onnxruntime, which is already present for Kokoro, so this adds a
2.3MB model and no new dependency. It is stateful: the model carries context
between chunks, so chunks must be fed in order and the state reset between
utterances.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime

from eve import config

# Alongside Kokoro's weights, outside the repo.  config centralises path
# expansion and the compatibility lookup for pre-rename installs.
MODEL = config.silero_model()

# Silero v5 judges 512 new samples at a time — 32ms — but the ONNX graph
# expects the previous window's last 64 samples prepended as context, for 576
# in total. Feeding a bare 512 runs without error and returns ~0 for
# everything, which looks exactly like a microphone that has gone deaf.
WINDOW = 512
CONTEXT = 64
RATE = 16000


class SileroVAD:
    """Speech probability for one 32ms window at a time."""

    def __init__(self, model: Path = MODEL) -> None:
        options = onnxruntime.SessionOptions()
        options.log_severity_level = 4   # the graph warns benignly on every call
        # One thread: this runs beside Whisper and the display, and the model
        # is far too small for parallelism to pay for its overhead.
        options.inter_op_num_threads = 1
        options.intra_op_num_threads = 1
        self._session = onnxruntime.InferenceSession(
            str(model), options, providers=["CPUExecutionProvider"]
        )
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros(CONTEXT, dtype=np.float32)

    def reset(self) -> None:
        """Forget the previous utterance's context."""
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros(CONTEXT, dtype=np.float32)

    def speech_probability(self, samples: np.ndarray) -> float:
        """How likely this window is speech, from 0 to 1.

        Accepts int16 samples of any length; only the first full window is
        judged, and short input returns 0 rather than guessing.
        """
        if len(samples) < WINDOW:
            return 0.0
        window = samples[:WINDOW].astype(np.float32) / 32768.0
        padded = np.concatenate((self._context, window))
        self._context = window[-CONTEXT:]
        probability, self._state = self._session.run(
            None,
            {
                "input": padded.reshape(1, CONTEXT + WINDOW),
                "state": self._state,
                "sr": np.array(RATE, dtype=np.int64),
            },
        )
        return float(probability[0][0])


def load() -> "SileroVAD | None":
    """The detector, or None if its model is missing.

    Returning None rather than raising lets the caller fall back to the
    loudness threshold: a missing model should degrade the assistant, not
    stop it from listening at all.
    """
    try:
        return SileroVAD()
    except Exception:
        return None
