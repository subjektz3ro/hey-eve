# Contributing

Contributions may fix behavior, improve hardware support, add a model backend,
refine the face, strengthen tests, or improve documentation. Keep pull requests
focused and describe the behavior that changes.

By contributing, you agree that your contribution is licensed under the
project's MIT License. Third-party code, model data, and generated assets must
retain their original attribution and license terms in
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

## Development setup

The repository supports CPython 3.11–3.13. Tests do not require a Raspberry
Pi, microphone, speaker, framebuffer, model files, API key, or network access.

```bash
git clone https://github.com/subjektz3ro/hey-eve.git
cd hey-eve
uv sync --locked --dev
uv run pytest -q
```

Useful manual modes after configuring an Anthropic API key:

```bash
uv run python -m eve.main --text --quiet --no-display
uv run python -m eve.main --text --no-display
uv run python -m eve.main --push
uv run python -m eve.main --say "hello" --no-display
```

The first command needs only the API path. Spoken modes also need the verified
Kokoro model files and an ALSA playback device. Always-listening modes need the
Whisper and Silero assets plus an ALSA capture device.

## Run the CI checks

```bash
uv run python -m compileall -q eve scripts deploy tests
uv run ruff check --output-format github .
uv run mypy
uv run python scripts/check_release_hygiene.py
for f in deploy/*.sh scripts/*.sh; do bash -n "$f"; done
uv run pytest -q --cov=eve --cov=scripts --cov=deploy \
  --cov-report=json:coverage.json --cov-report=term:skip-covered
uv run python scripts/check_coverage.py coverage.json
```

Dependency audit and isolated-wheel checks require package-index access:

```bash
ci=$(mktemp -d)
uv export --locked --no-dev --no-emit-project --format requirements-txt \
  -o "$ci/requirements.txt"
uvx --from pip-audit==2.10.1 pip-audit \
  -r "$ci/requirements.txt" --progress-spinner off

uv build --wheel --out-dir "$ci/wheel"
uv venv "$ci/wheel-venv" --python 3.11
uv pip install --python "$ci/wheel-venv/bin/python" "$ci"/wheel/*.whl
(cd "$ci" && "$ci/wheel-venv/bin/eve" --help)
```

CI runs the main test suite on Python 3.11 and 3.13. Per-file coverage floors
are enforced by `scripts/check_coverage.py`; exceptions in `pyproject.toml`
must include a reason and should be raised when coverage improves.

## Architecture boundaries

- `Transcriber` owns speech recognition; `Responder` owns one language-model
  turn. Keep microphone, display, and playback code independent of the chosen
  backend.
- `eve/head.py` owns face behavior, timing, and state. `eve/void.py` owns the
  raster appearance. New behavior should not be coupled to pixel drawing.
- Hardware access belongs behind existing seams in `eve/speech.py`,
  `eve/touch.py`, and the face classes. Unit tests use files and fakes instead
  of real devices.
- The service and installer share verification code. A dependency, model, or
  path change must update install, deploy, doctor output, and tests together.
- Long-running work must respond to SIGINT and SIGTERM and release temporary
  audio, player, and renderer resources.

## Privacy and security rules

- Do not commit credentials, private hostnames, MAC addresses, coordinates,
  personal paths, memory files, audio captures, generated model files, or
  operator configuration.
- Send transcript/reply content through `eve.log.content` or
  `eve.log.spoken`. Do not print it on a service path unless `VOICE_DEBUG=1`
  is explicitly in effect.
- Keep captured speech in an owner-only directory under `/dev/shm` and remove
  it through normal process cleanup.
- Do not add an arbitrary command, filesystem, or network tool to the model
  without a separate security review.
- Do not source the settings file. It is parsed as data so credential values
  cannot become shell syntax.
- Keep all model downloads revision- or digest-pinned and verify them before
  use.

The release-hygiene checker scans the exact staged Git blobs and fails closed
on untracked public-candidate files. Run it after staging the intended change,
not only before `git add`.

## Tests and comments

Tests should state observable behavior and fail under the regression they are
meant to prevent. Prefer a small pure helper when timing, signal, geometry, or
selection logic can be separated from hardware I/O.

Comments should explain a constraint, measurement, or non-obvious decision.
Avoid restating the code. When fixing a hardware or lifecycle bug that could
return, add a focused regression and leave the relevant reason near the seam.

## Hardware checks

Software tests cannot establish microphone gain, speaker behavior, panel
contrast, touch orientation, or Bluetooth recovery. If a change affects one of
those, state exactly what was observed, on which configuration, and whether
the check used a real device. Leave the service and hardware in their original
state after testing.

## Pull requests

Before opening a pull request:

1. Stage the intended files and run the release-hygiene check.
2. Run the relevant focused tests and the complete suite.
3. Run Ruff, mypy, and shell syntax checks.
4. Update configuration examples and docs when behavior or requirements
   change.
5. List the commands run and distinguish automated evidence from physical
   hardware observations.

Do not include unrelated formatting or generated-file churn. Maintainers merge
through protected `main` after required CI and security checks pass.
