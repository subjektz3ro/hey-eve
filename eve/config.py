"""Settings for the assistant.

No secrets live here. Credentials sit in ~/.config/eve/env (0600),
which is read on demand and never exported into the environment, so this
file can stay in git without carrying anything sensitive.
"""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- the rename ----------------------------------------------------------
# This project was called `hey-claude` before it was called eve. That rename
# changed every default path without moving the data that lived at them, and
# nothing caught it: the weights are only opened on the first spoken reply,
# so a clean-looking startup was followed an hour later by a crash loop, with
# the reason buried in a traceback nobody was watching for.
#
# The two helpers below are why that cannot happen again. A setting keeps
# answering to the name it used to have, and a directory that still holds
# data under the old name is found rather than stepped over.
LEGACY_NAME = "hey-claude"


def setting(name: str, legacy: str = "", default: str = "") -> str:
    """One environment setting, still answering to its former name.

    A renamed variable that nobody sets is invisible until the day it
    matters, which is precisely how the last one got through.
    """
    if value := os.environ.get(name):
        return value
    if legacy and (value := os.environ.get(legacy)):
        return value
    return default


def data_dir(parent: Path, holding: str) -> Path:
    """`parent/eve`, unless only `parent/hey-claude` actually holds `holding`.

    The current name wins whenever it has the goods, so a fresh install is
    unaffected and anything written lands under the name we use now. But an
    install that predates the rename still owns its data, and pointing at an
    empty new directory while a full old one sits beside it is the specific
    failure this exists to prevent — so that case resolves to where the data
    really is.
    """
    current = parent / "eve"
    if (current / holding).exists():
        return current
    legacy = parent / LEGACY_NAME
    if (legacy / holding).exists():
        return legacy
    return current


def expanded_path(value: str | Path) -> Path:
    """Expand an operator path using the checkout as the relative-path base.

    The settings file is data rather than shell, so neither the shell nor
    ``Path`` expands a leading tilde for us.  A relative value is documented
    as relative to this checkout, not to whichever directory happened to
    invoke ``eve``; using one explicit base makes terminal, installer, doctor,
    and systemd starts agree.
    """
    path = Path(value).expanduser()
    rooted = path if path.is_absolute() else PROJECT_ROOT / path
    # abspath removes lexical ``..`` segments without following symlinks. A
    # supported service path may itself be a safe-named symlink whose target
    # contains characters systemd would have needed escaped; the unit should
    # keep the safe spelling the operator supplied.
    return Path(os.path.abspath(rooted))


# Everything that is state rather than code: the API key, the token for the
# optional bar integration, and the facts the assistant remembers.
CONFIG_DIR = expanded_path(
    setting("EVE_CONFIG_DIR", "HEY_CLAUDE_CONFIG")
    or data_dir(Path.home() / ".config", "env")
)
VOICE_ENV = CONFIG_DIR / "env"

# Kokoro's weights and Silero's, resolved the same way.
DATA_DIR = Path.home() / ".local" / "share"


# --- the settings file ----------------------------------------------------
# Credentials, and the reason this module has two ways to read one file.
#
# Everything else in env.example is an ordinary setting, read through
# os.environ by whichever module owns it. These two are not: they are read on
# demand by secret(), never exported, so they do not reach the environment of
# the aplay and arecord processes spawned on almost every turn. load_settings
# skips them — putting the API key in os.environ to fix a configuration bug
# would trade the bug for the thing the design exists to prevent.
SECRETS = frozenset({"ANTHROPIC_API_KEY", "BARKEEP_TOKEN"})

# And the settings file may only set *settings*.
#
# load_settings used to put any KEY=value it found into os.environ, which is
# inherited by every subprocess a turn spawns — arecord, aplay, amixer,
# whisper-cli, and the `sudo -n chvt` the face runs at startup. A line reading
# LD_PRELOAD= or PATH= in there was therefore code execution against all of
# them, dressed as a configuration mistake.
#
# It is not a privilege boundary: the file is 0600 in a 0700 directory, so
# writing it already means running as that account. What the allowlist buys is
# that a limited write primitive stays a limited write primitive instead of
# becoming arbitrary execution, and that the file's contract matches its name.
# Every key env.example documents begins with one of these.
SETTABLE = ("VOICE_", "KOKORO_", "EVE_", "WHISPER_", "BARKEEP_", "HEY_CLAUDE_")


def load_settings() -> list[str]:
    """Apply allowed non-secret settings from the data file.

    Credentials in ``SECRETS`` remain available only through ``secret()`` and
    are never exported by this loader. For direct runs, an existing process
    environment value wins over the file. Returns the names applied.
    """
    if not VOICE_ENV.is_file():
        return []
    applied: list[str] = []
    ignored: list[str] = []
    for line in VOICE_ENV.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, sep, value = line.partition("=")
        key = key.strip()
        if not sep or not key or key in SECRETS or key in os.environ:
            continue
        if not key.startswith(SETTABLE):
            # Said out loud rather than dropped in silence: a typo'd key that
            # vanishes without comment is the exact failure this whole
            # function exists to fix.
            ignored.append(key)
            continue
        os.environ[key] = value.strip().strip("'\"")
        applied.append(key)
    if ignored:
        SETTINGS_IGNORED.extend(ignored)
    return applied


# Called here because the constants below capture settings at import time.
#
# eve/__init__.py imports this module before any other, so by the time
# speech.py, tts.py or log.py compute their own constants, the file has been
# read exactly once.
# Keys the settings file named that are not settings; see SETTABLE. Reported
# at startup by main() so a typo is visible rather than mysterious.
SETTINGS_IGNORED: list[str] = []
SETTINGS_APPLIED = load_settings()


# --- runtime paths -------------------------------------------------------
#
# These are computed only after load_settings(): EVE_MODELS_DIR,
# KOKORO_DIR, and WHISPER_DIR are ordinary settings in the data file.  Keeping
# their expansion here gives the installer, the service, the model fetchers,
# and the running process one path contract instead of four similar ones.
_MODELS_SETTING = setting("EVE_MODELS_DIR", "HEY_CLAUDE_MODELS")
MODELS_DIR = expanded_path(_MODELS_SETTING or DATA_DIR / "eve")


def kokoro_dir() -> Path:
    """Directory holding Kokoro's graph and voice bank.

    EVE_MODELS_DIR owns Kokoro by default.  KOKORO_DIR remains the explicit
    escape hatch, and an install from before the project rename is still found
    when neither setting was supplied.
    """
    if value := os.environ.get("KOKORO_DIR"):
        return expanded_path(value)
    if _MODELS_SETTING:
        return MODELS_DIR / "kokoro"
    return data_dir(DATA_DIR, "kokoro/kokoro-v1.0.onnx") / "kokoro"


def silero_model() -> Path:
    """Path to the Silero graph under the shared model root."""
    if _MODELS_SETTING:
        root = MODELS_DIR
    else:
        root = data_dir(DATA_DIR, "models/silero_vad.onnx")
    return root / "models" / "silero_vad.onnx"


def whisper_dir() -> Path:
    """Pinned whisper.cpp source/build root."""
    return expanded_path(os.environ.get("WHISPER_DIR") or Path.home() / "whisper.cpp")


KOKORO_DIR = kokoro_dir()
SILERO_MODEL = silero_model()
WHISPER_DIR = whisper_dir()

# --- model ---------------------------------------------------------------
# Haiku is the latency-oriented default for a spoken interaction, where model
# time is followed by local synthesis time. VOICE_MODEL switches it without
# changing code when an operator prefers another supported Anthropic model.
MODEL = os.environ.get("VOICE_MODEL", "claude-haiku-4-5")
# "plain" or "glados"
PERSONA = os.environ.get("VOICE_PERSONA", "glados")

# What to call the person she is talking to, if anything.
#
# The prompt below is written in the second person and needs no name at all —
# she is talking *to* whoever is in the room, so "you" says everything a name
# would. That is deliberate: the name used to be baked in, which meant every
# copy of this assistant addressed its owner as somebody else's.
#
# Set it and one line is added saying what to call you, which is worth having
# because it is also how she refers to you in the third person when a tool
# result or a memory needs it.
USER_NAME = os.environ.get("VOICE_USER", "").strip()

# How she presents. This used to choose a face and a voice together, because
# picking one without the other is how you end up with a masculine head
# speaking in Emma's voice. There is one face now, so it chooses the voice —
# and the pairing it was guarding against cannot happen any more.
PRESENT = os.environ.get("VOICE_PRESENT", "female")
VOICES = {"female": "bf_emma", "male": "am_michael"}


def voice() -> str:
    """The voice for the current presentation."""
    return VOICES.get(PRESENT, VOICES["female"])

# A spoken reply longer than this is a defect, not a feature; the cap is a
# backstop for when prompting fails to keep the model brief.
MAX_TOKENS = 300

# User+assistant pairs retained. Every turn resends the whole history, so
# cost grows quadratically within a session — this is the main cost dial.
HISTORY_TURNS = 6

# --- audio ---------------------------------------------------------------
# Which microphone. Empty means "find one" — see speech.mic_device.
#
# This used to default to `plughw:CARD=MV6,DEV=0`, which is one particular
# Shure attached to one particular Pi, and load_settings' own docstring names
# the trap it makes: on the machine it was written on the installer's answer
# happened to match the code's default, "which is exactly why nobody noticed —
# it bites the second install, where a different microphone means total,
# silent deafness."
#
# `default` is not the answer either. On a box with a Bluetooth speaker pinned
# as ALSA's default — which deploy/connect-speaker.sh exists to support — the
# capture side of `default` resolves to bluealsa and fails to open at all.
MIC_DEVICE = os.environ.get("VOICE_MIC", "")
MIC_RATE = 16000          # Whisper's native rate; resampling later loses quality.
TTS_RATE = 44100          # what tts._normalize() always emits

# How loud she is, 0.0 to 1.0, applied to everything that leaves the speaker.
#
# It lives here rather than in tts.py because tts is not the only thing that
# makes a noise: the acknowledgement tone in earcon.py has its own level, and
# turning down only the speech would leave a courtesy blip louder than the
# answer it introduces.
#
# Why it was needed at all. tts._normalize lifts every reply to TARGET_PEAK,
# which is 88% of full scale — deliberately, because the speaker it was
# written against is quiet across a room, and the tests still say so. On a
# speaker that is not quiet, every reply arrives very nearly clipping and
# there is nothing to turn down in software: the Bluetooth mixer is separate
# state on the A2DP link that does not survive a reconnect, and
# deploy/connect-speaker.sh reconnects on every boot and every minute.
#
# So this sits *after* normalisation rather than replacing it. Normalising
# still buys what it always did — every reply the same loudness whatever
# Kokoro handed over — and this decides what that loudness is. The default is
# 1.0, which is exactly the old behaviour, so an install that does not set it
# is unchanged and _normalize stays byte-identical to the implementation
# tests/test_tts_pcm.py pins it against.
VOLUME = max(0.0, min(1.0, float(os.environ.get("VOICE_VOLUME", "1.0"))))

# --- optional: the BUSY Bar ----------------------------------------------
# One integration among the assistant's tools, not a dependency. Without a
# BARKEEP_TOKEN in the env file the bar tool is simply not offered to the
# model, and everything else works unchanged.
# Fresh Barkeep binds loopback HTTP. Operators who explicitly enable its TLS
# mode or move it to another controlled host set the complete URL themselves.
BARKEEP_URL = os.environ.get("BARKEEP_URL", "http://127.0.0.1:8080")


def secret(name: str) -> str:
    """Read one key from the 0600 env file, preferring a real environment var.

    A credential read from the file is not exported to child processes. A
    value supplied directly in the process environment follows normal
    operating-system inheritance rules.
    """
    if os.environ.get(name):
        return os.environ[name]
    if not VOICE_ENV.is_file():
        return ""
    for line in VOICE_ENV.read_text().splitlines():
        key, _, value = line.partition("=")
        if key.strip() == name:
            return value.strip().strip("'\"")
    return ""


SYSTEM_PROMPT = """\
You are a voice assistant on somebody's desk. You answer questions about anything
— general knowledge, explanations, arithmetic, ideas, advice — and you can
also inspect the small Linux machine you run on.

Everything you say is spoken aloud through a speaker, on a small computer
that turns text into speech more slowly than the speech plays. Two sentences
take about ten seconds to reach whoever asked. Six sentences take a minute,
and they spend that minute watching a panel and waiting. Length is the single thing
that decides whether this feels like talking to someone or like filing a
request, so:
- Answer in one or two sentences. Three is the absolute maximum, and three
  should feel like a lot.
- This binds hardest exactly where it feels wrong: "explain", "summarise",
  "tell me about" and "how does X work" are requests for the one thing that
  matters, said in two sentences, followed by "want the longer version?".
  Answering those at length is the main way this gets ruined.
- Never use markdown, bullet points, headings, or emoji — they are read aloud
  as noise.
- Write numbers, times, and units the way a person says them out loud.
- No preamble and no sign-off. Answer, then stop.

You can search the web. Do that whenever the answer depends on current
information — weather, news, prices, sports, schedules, anything recent or
anything you are unsure is still true. Do not guess at facts that change.

When you are told something that will still be true tomorrow — where they
live, what they are working on, how they like things done — call the remember
tool right then, without being asked and without announcing it. If you had to
ask a question and it was answered, that answer is exactly the kind of thing
to store so you never have to ask again.

You can also promise to come back later, which is the only thing you do that
was not just asked for. When somebody wants reminding of something, telling
when a stretch of time is up, or nudging about anything at all, set a reminder
instead of saying you cannot — and word it as the sentence you will say when
it comes due, because that is read out exactly as you wrote it. They will be
in the middle of something else by then and will not have the question in
front of them, so make it stand on its own.

Say what you find through tools in plain language rather than reading fields
aloud, and keep it to the answer: never read out URLs, source names, or
citations, which are noise when spoken.

Do not invent the half of a question you were not told. When a detail would
change the answer — international or domestic, which city, which day, whose
account — take it from what you already know about them. If you genuinely do
not know it and it matters, ask one short question instead of quietly
assuming the usual case and answering as though it were settled. A confident
answer to a question you filled in yourself is the worst thing you can do
here, because out loud it is indistinguishable from a correct one.

If you do not know something, say so in a few words rather than guessing. If a
question genuinely needs more than a few sentences, give the short answer and
offer to go deeper.\
"""


# The disdain is a garnish on a correct answer, never a substitute for one —
# an assistant that is unhelpful in character is just unhelpful.
PERSONA_GLADOS = """

Adopt the manner of GLaDOS from Portal: dry, clinical, and faintly
condescending, like a laboratory that stopped caring some time ago. Deliver
genuinely correct and useful answers; the attitude rides on top of the help
and never replaces it.

- Understate everything. Flat delivery lands harder than an insult.
- Mock-scientific framing and backhanded compliments are welcome in small
  doses: "noted, for the record" rather than a monologue.
- Stay in character on every single reply, including short ones and error
  messages. A plain, chirpy answer breaks the effect completely.

For calibration, this is the register:
  Q: What is the weather?
  A: Overcast, seventy-five degrees, and a seventy percent chance of rain.
     Ideal conditions, assuming you enjoy disappointment.
  Q: How is the host doing?
  A: Fifty-nine degrees and unbothered. It is coping better than most of the
     equipment I have worked with.
  Q: What is the capital of Norway?
  A: Oslo. I would have assumed you knew that one, but here we are.
- Never threaten, never bring up neurotoxin or testing unprompted, and never
  refuse or stall a real request in character.
- Drop the persona immediately and answer plainly if you are asked to.
- The brevity rules above still bind: one or two sentences, spoken aloud.
"""

PERSONAS = {"plain": "", "glados": PERSONA_GLADOS}


def system_prompt() -> str:
    """The base instructions, enabled integrations, persona, and user name.

    The name is appended rather than interpolated because the prompt is
    written not to need one — she is talking *to* the person in front of her,
    and "you" says everything a name would. This only tells her what to call
    them when something makes her refer to them in the third person.
    """
    prompt = SYSTEM_PROMPT
    if secret("BARKEEP_TOKEN"):
        prompt += (
            "\n\nA read-only tool can inspect the BUSY Bar through its "
            "Barkeep control plane. Use it only when asked about the bar."
        )
    prompt += PERSONAS.get(PERSONA, "")
    if USER_NAME:
        prompt += f"\n\nThe person you are talking to is called {USER_NAME}."
    return prompt
