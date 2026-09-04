# Third-party notices

Eve's original source code is licensed under the MIT License in
[`LICENSE`](LICENSE). Dependencies, downloaded model data, and system tools
retain their own licenses.

This repository does not contain the model files or the Whisper source tree.
The installation scripts download them from their upstream projects, verify
their pinned revisions or SHA-256 digests, and store them outside the checkout.

| Component | Use | Pinned source | License |
|---|---|---|---|
| Anthropic Python SDK | Remote language-model client | Version in `uv.lock`; [anthropics/anthropic-sdk-python](https://github.com/anthropics/anthropic-sdk-python) | MIT |
| Kokoro ONNX | Local text-to-speech runtime | Version in `uv.lock`; [thewh1teagle/kokoro-onnx](https://github.com/thewh1teagle/kokoro-onnx) | MIT |
| Kokoro-82M model and voice bank | Downloaded speech model data | [model-files-v1.0](https://github.com/thewh1teagle/kokoro-onnx/releases/tag/model-files-v1.0), SHA-256 values in `eve/doctor.py` | Apache-2.0 |
| phonemizer-fork | Kokoro phoneme conversion dependency | Version in `uv.lock`; [bootphon/phonemizer](https://github.com/bootphon/phonemizer) | GPL-3.0-or-later |
| espeakng-loader | Locates the bundled eSpeak NG library and data | Version in `uv.lock`; [thewh1teagle/espeakng-loader](https://github.com/thewh1teagle/espeakng-loader) | MIT |
| eSpeak NG library and data | Phoneme-generation runtime bundled by espeakng-loader | Bytes pinned by the locked espeakng-loader wheel; [espeak-ng/espeak-ng](https://github.com/espeak-ng/espeak-ng) | GPL-3.0-or-later, with separately licensed files noted upstream |
| Silero VAD | Downloaded local voice-activity model | Commit `bfdc0193023f121ea5b3cc7b176dbed570a68a59`; [snakers4/silero-vad](https://github.com/snakers4/silero-vad) | MIT |
| whisper.cpp | Downloaded local speech-to-text runtime | Commit `592feef04a1802b18cbeffd0fd0eb5d02570c2ec`; [ggml-org/whisper.cpp](https://github.com/ggml-org/whisper.cpp) | MIT |
| Whisper `base.en` model | Downloaded speech-recognition model data | Revision and SHA-256 in `eve/doctor.py`; [ggerganov/whisper.cpp](https://huggingface.co/ggerganov/whisper.cpp) | MIT |
| ONNX Runtime | Local inference runtime | Version in `uv.lock`; [microsoft/onnxruntime](https://github.com/microsoft/onnxruntime) | MIT |
| NumPy | Audio, model, and rendering arrays | Version in `uv.lock`; [numpy/numpy](https://github.com/numpy/numpy) | BSD-3-Clause |
| Pillow | 480×320 face rasterization | Version in `uv.lock`; [python-pillow/Pillow](https://github.com/python-pillow/Pillow) | MIT-CMU |

Exact Python versions and artifact hashes are recorded in `uv.lock` and
`eve/doctor.py`. License texts and corresponding source are available
from the linked upstream projects; package distributions may also include
their license files. The Whisper checkout contains its license file.

The GIF demonstrations in `docs/` and the Eve face artwork rendered by this
repository were created for this project by its author.

The optional `glados` persona is an independent homage. GLaDOS and Portal are
properties of Valve Corporation. This project is not affiliated with or
endorsed by Valve, Anthropic, the Kokoro authors, or the other upstream
projects listed above.
