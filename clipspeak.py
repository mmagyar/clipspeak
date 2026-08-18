#!/usr/bin/env python3
"""
clipspeak - watch the macOS clipboard and read new text aloud with a local neural TTS model.

Nothing leaves your machine. Copy some text, hear it. Copy something else, it
switches to the new text. Copy a single short word (e.g. "stop") to shut it up.

Usage:
    python clipspeak.py                     # watch the clipboard, menu bar icon
    python clipspeak.py --check             # load the model and say a test line
    python clipspeak.py --say "some text"   # one-shot, no clipboard involved
    python clipspeak.py --preset kokoro     # use the small/fast model instead

Config lives in CONFIG below; every value can be overridden with a
CLIPSPEAK_* environment variable or a CLI flag. The menu bar saves the model,
voice and speed you pick to ~/.config/clipspeak.json and reads them at startup.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import queue
import re
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Iterable, Iterator

# --------------------------------------------------------------------------
# Presets
# --------------------------------------------------------------------------
# `kwargs` are passed straight to mlx-audio's model.generate(). If a model
# rejects one of them we retry without it, so an unknown kwarg is not fatal.

PRESETS: dict[str, dict] = {
    # Best quality. ~1.7B params at 8-bit, roughly 2 GB resident, generates a
    # bit faster than realtime on Apple Silicon. The CustomVoice checkpoint is
    # the one with a speaker table; the Base checkpoint has none, so it invents a
    # new voice for every utterance. lang_code pins the language, otherwise
    # "auto" detection drifts into Chinese on short or ambiguous text.
    "qwen": {
        "model": "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit",
        # stream=True is what makes it start talking in under a second: the model
        # hands back audio every `streaming_interval` seconds instead of after the
        # whole chunk is decoded (measured: 0.6s to first sound instead of 4.9s).
        # temperature/top_p sit below the model defaults (0.9 / 1.0) because at the
        # defaults it sometimes samples laughter, crying or a different tone.
        # Do not go much lower. This is an autoregressive codec model, so a low
        # temperature makes it stick on the silence token and never emit EOS: at
        # 0.3 one run in six produced 4 seconds of speech and 5 minutes of silence,
        # and at 0.0 every run did. Measured over six runs of the same sentence,
        # 0.5 held pitch to +/-4 Hz against +/-7 at 0.3 and +/-10 at 0.7.
        # instruct pins the delivery. max_tokens 0 means "derive it from the text".
        "kwargs": {
            "voice": "ryan",
            "speed": 1.0,
            "lang_code": "english",
            "stream": True,
            "streaming_interval": 1.0,
            "temperature": 0.5,
            "top_p": 0.8,
            "max_tokens": 0,
            "instruct": "Read calmly and clearly in a neutral tone.",
        },
        "sample_rate": 24000,
        # Qwen accepts `speed` and ignores it, so clipspeak stretches the audio
        # itself. The speaker names come from the checkpoint at load time.
        "native_speed": False,
        "voices": [],
    },
    # Small, fast, very low latency. 82M params. Good fallback / good default
    # if you find the big model too slow to start talking.
    "kokoro": {
        "model": "mlx-community/Kokoro-82M-bf16",
        "kwargs": {"voice": "af_heart", "speed": 1.0, "lang_code": "a"},
        "sample_rate": 24000,
        "native_speed": True,
        "voices": [
            "af_heart", "af_bella", "af_nova", "af_sky", "am_adam", "am_echo",
            "bf_alice", "bf_emma", "bm_daniel", "bm_george",
        ],
    },
    # No install required at all - macOS built-in voice. Useful for proving the
    # clipboard plumbing works before you download gigabytes of weights.
    "system": {
        "model": None,
        "kwargs": {},
        "sample_rate": 22050,
        "native_speed": True,
        "voices": ["Alex", "Daniel", "Fiona", "Karen", "Moira", "Samantha", "Tessa"],
    },
}


@dataclass
class Config:
    preset: str = "qwen"
    model: str | None = None          # overrides the preset's model repo
    voice: str | None = None          # overrides the preset's voice
    speed: float | None = None

    poll_interval: float = 0.35       # seconds between clipboard checks
    min_chars: int = 25               # shorter than this = treated as a "stop"
    max_chars: int = 6000             # longer than this is skipped, not read
    chunk_chars: int = 900            # target size of each synthesis chunk
    min_alpha_ratio: float = 0.20     # below this it's data, not code or prose
    speak_on_start: bool = False      # read whatever is already on the clipboard

    log_level: str = "INFO"
    extra_kwargs: dict = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> "Config":
        cfg = cls()
        cfg.update(load_choices())
        for f in ("preset", "model", "voice", "log_level"):
            v = os.environ.get(f"CLIPSPEAK_{f.upper()}")
            if v:
                setattr(cfg, f, v)
        for f in ("speed", "poll_interval", "min_alpha_ratio"):
            v = os.environ.get(f"CLIPSPEAK_{f.upper()}")
            if v:
                setattr(cfg, f, float(v))
        for f in ("min_chars", "max_chars", "chunk_chars"):
            v = os.environ.get(f"CLIPSPEAK_{f.upper()}")
            if v:
                setattr(cfg, f, int(v))
        if os.environ.get("CLIPSPEAK_SPEAK_ON_START", "").lower() in ("1", "true", "yes"):
            cfg.speak_on_start = True

        raw = os.environ.get("CLIPSPEAK_EXTRA_KWARGS")
        if raw:
            try:
                extra = json.loads(raw)
            except ValueError as exc:
                raise SystemExit(f"CLIPSPEAK_EXTRA_KWARGS is not valid JSON: {exc}") from exc
            if not isinstance(extra, dict):
                raise SystemExit(
                    f"CLIPSPEAK_EXTRA_KWARGS must be a JSON object, got {type(extra).__name__}"
                )
            cfg.extra_kwargs = extra
        return cfg

    def update(self, values: dict) -> None:
        for k, v in values.items():
            if hasattr(self, k):
                setattr(self, k, v)


log = logging.getLogger("clipspeak")

# What the menu bar remembers between runs. Env vars and CLI flags still win.
CHOICES_PATH = os.path.expanduser("~/.config/clipspeak.json")


def load_choices() -> dict:
    try:
        with open(CHOICES_PATH) as fh:
            value = json.load(fh)
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as exc:
        log.warning("ignoring %s: %s", CHOICES_PATH, exc)
        return {}
    if not isinstance(value, dict):
        log.warning("ignoring %s: expected a JSON object, got %s", CHOICES_PATH, type(value).__name__)
        return {}
    return value


def save_choices(cfg: "Config") -> None:
    """Remember the menu bar picks. Best effort: a read-only home is not fatal."""
    try:
        os.makedirs(os.path.dirname(CHOICES_PATH), exist_ok=True)
        with open(CHOICES_PATH, "w") as fh:
            json.dump({"preset": cfg.preset, "voice": cfg.voice, "speed": cfg.speed}, fh)
    except OSError as exc:
        log.warning("could not save choices to %s: %s", CHOICES_PATH, exc)


# --------------------------------------------------------------------------
# Clipboard access
# --------------------------------------------------------------------------

class ClipboardReader:
    """Reads macOS clipboard text, preferring NSPasteboard so we can tell text
    from images/files and so we only wake up when the contents actually change."""

    def __init__(self) -> None:
        self._pb = None
        self._last_count = -1
        try:
            from AppKit import NSPasteboard, NSPasteboardTypeString  # type: ignore

            self._pb = NSPasteboard.generalPasteboard()
            self._string_type = NSPasteboardTypeString
            log.debug("using NSPasteboard")
        except Exception as exc:  # pragma: no cover - platform dependent
            log.warning("pyobjc not available (%s); falling back to pbpaste polling", exc)

    def read(self) -> str | None:
        """Current clipboard text, or None if it holds no text."""
        if self._pb is not None:
            types = self._pb.types()
            if self._string_type not in types:
                log.debug("clipboard holds no text (types=%s)", list(types))
                return None
            value = self._pb.stringForType_(self._string_type)
            return str(value) if value else None

        try:
            out = subprocess.run(
                ["pbpaste"], capture_output=True, text=True, timeout=5
            ).stdout
        except Exception as exc:
            log.error("pbpaste failed: %s", exc)
            return None
        return out or None

    def poll(self) -> str | None:
        """Return new clipboard text, or None if nothing changed / not text."""
        if self._pb is not None:
            count = self._pb.changeCount()
            if count == self._last_count:
                return None
            self._last_count = count
            return self.read()

        # pbpaste fallback: no change counter, so we diff the string.
        text = self.read()
        if text == getattr(self, "_last_text", None):
            return None
        self._last_text = text
        return text

# --------------------------------------------------------------------------
# Filtering: decide whether a clipboard entry is worth reading
# --------------------------------------------------------------------------

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.I)
BARE_URLISH_RE = re.compile(
    r"^(?:https?://|www\.|[\w.-]+\.(?:com|org|net|io|dev|ai|co|uk|hu|de|edu|gov)\b)\S*$",
    re.I,
)
PATH_RE = re.compile(r"^(?:~|/|\./|\.\./|[A-Za-z]:\\)\S*$")
BLOB_MEAN_TOKEN = 20              # above this it is hex, base64 or minified
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?;:])\s+|\n{2,}")
FENCE_RE = re.compile(r"```(?:[A-Za-z0-9_+.-]*\n)?([\s\S]*?)```")

# --------------------------------------------------------------------------
# Speaking technical text
# --------------------------------------------------------------------------
# Kokoro's G2P (misaki) treats the dot in "run.sh" as sentence punctuation and
# never says the word "dot", so the extension arrives as an unstressed mumble
# after a pause. Same for paths, operators and SCREAMING_SNAKE names. Fixing it
# here rather than in misaki also fixes the qwen and system presets.

# Only extensions misaki gets wrong. Checked and deliberately absent: json
# ("JAY-son"), yaml/yml ("yammel"), toml ("tommel") already read correctly.
# Single-letter .c and .h are absent too: too easy to fire on prose like "A.C.".
EXTENSIONS = {
    "sh": "S H", "bash": "bash", "zsh": "Z S H", "py": "P Y", "pyc": "P Y C",
    "ts": "T S", "tsx": "T S X", "js": "J S", "jsx": "J S X", "mjs": "M J S",
    "md": "M D", "rs": "R S", "rb": "R B", "go": "go", "cpp": "C plus plus",
    "hpp": "H plus plus", "css": "C S S", "scss": "S C S S", "html": "H T M L",
    "xml": "X M L", "csv": "C S V", "txt": "text", "cfg": "config",
    "ini": "I N I", "sql": "S Q L", "png": "P N G", "jpg": "J P G",
    "jpeg": "J P E G", "gif": "gif", "svg": "S V G", "pdf": "P D F",
    "mp3": "M P 3", "mp4": "M P 4", "wav": "wav", "gz": "G Z", "tgz": "T G Z",
    "zip": "zip", "lock": "lock", "env": "env",
}

# Longest first so "->" wins over "-" and "!=" over "=".
OPERATORS = (
    ("->", " arrow "), ("=>", " arrow "), ("!=", " not equals "),
    ("==", " equals "), (">=", " greater or equal "), ("<=", " less or equal "),
    ("&&", " and "), ("||", " or "), ("::", " colon colon "), ("|>", " pipe "),
)

# Case-sensitive on purpose: real extensions are lowercase, and matching
# uppercase would rewrite "U.S." and "A.C." in ordinary prose.
PATH_TOKEN_RE = re.compile(
    r"(?<![\w/.~-])[~.]{0,2}/?[\w.~/-]*\.(?:"
    + "|".join(sorted(EXTENSIONS, key=len, reverse=True))
    + r")(?!\w)"
)
FLAG_RE = re.compile(r"(?<!\w)--?([A-Za-z][\w-]*)(?!\w)")
SNAKE_RE = re.compile(r"(?<!\w)([A-Za-z_]\w*(?:_\w+)+)(?!\w)")

# Short names misaki spells out letter by letter or slurs into one syllable.
CODE_WORDS = {
    "str": "string", "int": "integer", "bool": "boolean", "len": "length",
    "args": "arguments", "kwargs": "keyword arguments", "init": "initialize",
    "src": "source", "dir": "directory", "env": "environment",
    "repo": "repository", "util": "utility", "impl": "implementation",
    "cfg": "config", "fn": "function", "ret": "return", "elif": "else if",
}
CODE_WORD_RE = re.compile(r"(?<![\w/.])[a-z]{2,6}(?![\w/.])")

JSON_MAX_DEPTH = 6
JSON_MAX_ITEMS = 50


def _say_ident(name: str) -> str:
    """Split an identifier into words. ALL_CAPS is lowercased, otherwise misaki
    spells every letter and CLIPSPEAK_MIN_CHARS becomes seventeen letters."""
    if "_" not in name and "-" not in name:
        return name
    if name.isupper():
        name = name.lower()
    return re.sub(r"[_-]+", " ", name).strip()


def _say_path(m: "re.Match[str]") -> str:
    stem, _, ext = m.group(0).rpartition(".")
    stem = stem.lstrip(".")                       # ./ and ../ carry no meaning
    if stem.startswith("~"):
        stem = "home" + stem[1:]
    parts = [_say_ident(p) for p in stem.split("/") if p]
    return " slash ".join(parts) + " dot " + EXTENSIONS[ext]


def say_tech(text: str) -> str:
    """Rewrite filenames, paths, identifiers and operators as spoken English.

    Shape matching is deliberately narrow, so "and/or", "e.g." and "3.5" in
    ordinary prose come through untouched.
    """
    t = PATH_TOKEN_RE.sub(_say_path, text)
    for symbol, word in OPERATORS:
        t = t.replace(symbol, word)
    t = FLAG_RE.sub(lambda m: _say_ident(m.group(1)), t)
    t = SNAKE_RE.sub(lambda m: _say_ident(m.group(1)), t)
    return re.sub(r"[ \t]{2,}", " ", t)


def _say_scalar(value: object) -> str:
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    return say_tech(str(value))


def say_json(value: object, depth: int = 0) -> str:
    """Narrate parsed JSON structurally: keys, values and list lengths, not
    braces and colons."""
    if depth > JSON_MAX_DEPTH:
        return "nested data."
    if isinstance(value, dict):
        out = ["object."]
        for i, (key, item) in enumerate(value.items()):
            if i >= JSON_MAX_ITEMS:
                out.append(f"and {len(value) - i} more.")
                break
            name = say_tech(str(key))
            if isinstance(item, (dict, list)):
                out.append(f"key {name}, {say_json(item, depth + 1)}")
            else:
                out.append(f"key {name}, value {_say_scalar(item)}.")
        return " ".join(out)
    if isinstance(value, list):
        out = [f"list of {len(value)}."]
        for i, item in enumerate(value):
            if i >= JSON_MAX_ITEMS:
                out.append(f"and {len(value) - i} more.")
                break
            out.append(
                say_json(item, depth + 1)
                if isinstance(item, (dict, list))
                else f"{_say_scalar(item)}."
            )
        return " ".join(out)
    return f"{_say_scalar(value)}."


def say_code(text: str) -> str:
    """Read source line by line. Brackets become pauses, not spoken symbols.

    ponytail: no per-language parsing, so this is literal-with-pauses rather
    than semantic. Add a tree-sitter pass if line-oriented reading proves too
    flat to follow.
    """
    lines = []
    for raw_line in text.split("\n"):
        line = say_tech(raw_line.strip())
        if not line:
            continue
        line = re.sub(r"^(#!|#+|//+|--)\s*", "comment, ", line)
        line = CODE_WORD_RE.sub(lambda m: CODE_WORDS.get(m.group(0), m.group(0)), line)
        line = re.sub(r"(?<=[\w.])/(?=[\w.~])", " slash ", line)  # unspaced = path
        line = re.sub(r"\.(?=[A-Za-z_])", " dot ", line)     # attribute access
        line = line.replace("=", " equals ")
        line = re.sub(r"(?<!\w)[frbu]{1,2}(?=[\"\'])", "", line)   # f"..." prefixes
        line = re.sub(r"[\"\'`]", "", line)
        line = re.sub(r"[(){}\[\]:;,]+", ",", line)          # brackets -> pause
        line = re.sub(r"\s*,\s*", ", ", line)
        line = re.sub(r"\s{2,}", " ", line).strip(" ,")
        if line:
            lines.append(line if line[-1] in ".!?" else line + ".")
    return " ".join(lines)


# Source code scores high on at least one of these. Prose and markdown score
# zero on both, including markdown that contains fenced code.
CODE_LINE_END_RE = re.compile(r"[:{};,)\]\\]$")


def looks_like_code(text: str) -> bool:
    """True when the clipboard holds raw source rather than prose.

    Measured across this repo: README.md and plain prose score 0.00/0.00, while
    run.sh scores 0.17/0.64, clipspeak.py 0.63/0.87, and Python and TypeScript
    snippets 1.00/0.50 or better.
    """
    lines = [ln.rstrip() for ln in text.split("\n") if ln.strip()]
    if not lines:
        return False
    ends = sum(bool(CODE_LINE_END_RE.search(ln)) for ln in lines) / len(lines)
    indented = sum(ln[0] in " \t" for ln in lines) / len(lines)
    return ends >= 0.3 or indented >= 0.4


def _as_json(text: str) -> object | None:
    """Parsed JSON if `text` is a JSON object or array, else None."""
    if text[:1] not in "{[":
        return None
    try:
        value = json.loads(text)
    except ValueError:
        return None
    return value if isinstance(value, (dict, list)) else None


def clean_text(text: str) -> str:
    """Light normalisation so the model doesn't read markdown scaffolding aloud."""
    data = _as_json(text.strip())
    if data is not None:
        return say_json(data)
    t = text.replace("\r\n", "\n").replace("\r", "\n")
    # A fence means this is a document that contains code, not source itself, so
    # the markdown path below handles it and reads only the fenced parts as code.
    if "```" not in t and looks_like_code(t):
        return say_code(t)
    t = FENCE_RE.sub(lambda m: " " + say_code(m.group(1)) + " ", t)
    t = re.sub(r"`([^`]+)`", r"\1", t)                       # inline code
    t = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", t)              # images
    t = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", t)           # links -> label
    t = URL_RE.sub(" link ", t)                              # bare urls
    t = re.sub(r"^\s{0,3}#{1,6}\s*", "", t, flags=re.M)      # headings
    t = re.sub(r"^\s{0,3}[-*+]\s+", "", t, flags=re.M)       # bullets
    t = re.sub(r"\*\*([^*]+)\*\*", r"\1", t)                 # bold
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{2,}", "\n\n", t)
    return say_tech(t).strip()


def alpha_ratio(text: str) -> float:
    dense = [c for c in text if not c.isspace()]
    if not dense:
        return 0.0
    return sum(c.isalpha() for c in dense) / len(dense)


def classify(raw: str, cleaned: str, cfg: Config) -> tuple[bool, str]:
    """Return (should_speak, reason).

    The shape checks all run against the *raw* clipboard text. Cleaning rewrites
    bare URLs to the word "link" and turns code into words, so measuring the
    cleaned text would hide exactly what we are trying to detect.
    """
    r = raw.strip()
    t = cleaned.strip()
    if not t:
        return False, "empty"
    if BARE_URLISH_RE.match(r):
        return False, "url"
    if PATH_RE.match(r) and " " not in r:
        return False, "file path"
    if len(t) > cfg.max_chars:
        return False, f"too long ({len(t)} chars > {cfg.max_chars})"
    if len(r) >= cfg.min_chars and _as_json(r) is not None:
        return True, "json"
    if len(t) < cfg.min_chars:
        return False, "too short"
    if " " not in r:
        # Raw, not cleaned: normalising "==" to " equals " would otherwise put
        # spaces into a base64 blob and make it look like prose.
        return False, "single token"
    # Mean token length separates content from blobs with a wide margin: prose
    # 3.9, shell 5.6, Python 6.8-10, against 36 for minified JS and 48 for
    # base64. A single long line in real source does not move the mean, which is
    # why this beats measuring the longest run. URLs are legitimately long and
    # already became "link", so measure without them.
    words = URL_RE.sub(" ", r).split()
    mean_token = sum(len(w) for w in words) / len(words)
    if mean_token > BLOB_MEAN_TOKEN:
        return False, f"unreadable blob (mean token {mean_token:.0f} chars)"
    ratio = alpha_ratio(r)
    if ratio < cfg.min_alpha_ratio:
        return False, f"looks like data, not code or prose (alpha ratio {ratio:.2f})"
    return True, "ok"


def chunk_text(text: str, target: int) -> list[str]:
    """Split into speakable chunks at sentence boundaries, ~`target` chars each.

    Chunking matters: it's what lets playback start after the first sentence
    instead of after the whole passage has been synthesised.
    """
    pieces = [p.strip() for p in SENTENCE_SPLIT_RE.split(text) if p and p.strip()]
    chunks: list[str] = []
    current = ""
    for piece in pieces:
        # A single sentence longer than the target gets hard-split on commas.
        if len(piece) > target * 2:
            sub = [s.strip() for s in re.split(r"(?<=,)\s+", piece) if s.strip()]
        else:
            sub = [piece]
        for s in sub:
            if not current:
                current = s
            elif len(current) + len(s) + 1 <= target:
                current = f"{current} {s}"
            else:
                chunks.append(current)
                current = s
    if current:
        chunks.append(current)
    return chunks


# --------------------------------------------------------------------------
# TTS backends
# --------------------------------------------------------------------------

class Backend:
    sample_rate = 24000
    voice: str | None = None      # set at runtime, read at synthesis time
    speed: float = 1.0
    voice_list: list[str] = []

    def synth(self, text: str) -> Iterator["object"]:
        raise NotImplementedError

    def play(self, chunks: Iterable[str], cancel: threading.Event) -> None:
        raise NotImplementedError


def time_stretch(audio, factor: float, sample_rate: int):
    """Change duration by `factor` without moving pitch, using WSOLA. A factor
    above 1.0 shortens the audio, so speech gets faster."""
    import numpy as np

    frame = int(sample_rate * 0.04)
    hop_out = frame // 2
    hop_in = int(round(hop_out * factor))
    search = int(sample_rate * 0.005)
    if abs(factor - 1.0) < 0.02 or len(audio) < frame + hop_in + 2 * search:
        return audio

    window = np.hanning(frame).astype(np.float32)
    out = np.zeros(int(len(audio) / factor) + frame, dtype=np.float32)
    weight = np.zeros_like(out)

    read, write, tail = 0, 0, None
    while read + frame + search <= len(audio) and write + frame <= len(out):
        pos = read
        if tail is not None:
            # Slide the read head to wherever it lines up best with the previous
            # frame's overlap. Without this the seams beat against each other.
            # `read` itself keeps its ideal spacing, otherwise the offsets
            # accumulate and the clip drifts away from the requested duration.
            lo = max(0, read - search)
            hi = min(len(audio) - frame, read + search)
            if hi > lo:
                scores = np.correlate(audio[lo : hi + frame], tail, mode="valid")
                pos = lo + int(np.argmax(scores[: hi - lo + 1]))
        out[write : write + frame] += audio[pos : pos + frame] * window
        weight[write : write + frame] += window
        tail = audio[pos + hop_out : pos + frame]
        read += hop_in
        write += hop_out

    out = out[: write + frame]
    weight = weight[: write + frame]
    return np.divide(out, weight, out=np.zeros_like(out), where=weight > 1e-6)


MAX_SILENCE = 2.5   # seconds of dead air inside one chunk before we give up on it


def token_budget(text: str, headroom: float = 2.5) -> int:
    """Token cap for one chunk: enough for the text, not enough for a runaway.

    Qwen3-TTS emits 12.5 codec tokens per second of audio and speech runs at
    roughly 15 characters per second. The library's own default of 4096 is five
    minutes, so a generation that loops on the silence token instead of stopping
    holds the speaker silent for minutes. This bounds that to a few seconds.
    """
    return max(64, int(len(text) / 15 * 12.5 * headroom))


def voice_warning(speakers: Iterable[str], voice: str | None) -> str | None:
    """Explain why a model will ignore `voice`, or None if it will honour it."""
    names = sorted(s.lower() for s in speakers)
    if not names:
        return "this model has no speaker table, so every utterance gets a random voice"
    if not voice:
        return f"no voice set, so every utterance gets a random voice; try one of {names}"
    if voice.lower() not in names:
        return f"voice {voice!r} is not in this model; try one of {names}"
    return None


class MLXBackend(Backend):
    """mlx-audio on Apple Silicon. Generates chunk N+1 while chunk N plays."""

    native_speed = True   # False when the model ignores `speed` and clipspeak stretches

    def __init__(
        self,
        model_repo: str,
        gen_kwargs: dict,
        sample_rate: int,
        native_speed: bool = True,
        voices: list[str] | None = None,
    ) -> None:
        import numpy as np  # noqa: F401  (checked early so failure is obvious)
        import sounddevice as sd  # noqa: F401
        from mlx_audio.tts.utils import load_model

        log.info("loading %s ...", model_repo)
        t0 = time.time()
        self.model = load_model(model_repo)
        self.gen_kwargs = dict(gen_kwargs)
        self.sample_rate = sample_rate
        log.info("model ready in %.1fs", time.time() - t0)

        talker = getattr(getattr(self.model, "config", None), "talker_config", None)
        speakers = sorted(getattr(talker, "spk_id", None) or {})
        self.native_speed = native_speed
        self.voice = gen_kwargs.get("voice")
        self.speed = float(gen_kwargs.get("speed") or 1.0)
        self.voice_list = speakers or list(voices or [])
        warning = voice_warning(speakers, self.voice)
        if warning:
            log.warning("%s", warning)

    def _generate(self, text: str) -> Iterator["object"]:
        """Yield audio as the model produces it, dropping kwargs it doesn't accept.

        Lazy on purpose. Collecting the results first would throw away the
        streaming models give us and put a whole chunk of silence at the front.
        """
        kwargs = dict(self.gen_kwargs)
        if self.voice:
            kwargs["voice"] = self.voice
        if self.native_speed:
            kwargs["speed"] = self.speed
        if kwargs.get("max_tokens") == 0:
            kwargs["max_tokens"] = token_budget(text)
        while True:
            # generate() is a generator, so a rejected kwarg only shows up on
            # the first item, not on the call itself.
            results = self.model.generate(text=text, **kwargs)
            try:
                first = next(results)
            except StopIteration:
                return
            except TypeError as exc:
                bad = re.search(r"'(\w+)'", str(exc))
                if bad and bad.group(1) in kwargs:
                    log.warning("model rejected kwarg %r, retrying without it", bad.group(1))
                    kwargs.pop(bad.group(1))
                    continue
                raise
            yield first
            yield from results
            return

    def play(self, chunks: Iterable[str], cancel: threading.Event) -> None:
        # Synthesis stays on the calling thread: MLX keeps thread-local state that
        # segfaults when the thread that used it exits. Playback gets the thread.
        import numpy as np
        import sounddevice as sd

        audio_q: queue.Queue = queue.Queue()

        def player() -> None:
            stream = None
            try:
                while True:
                    item = audio_q.get()
                    if item is None or cancel.is_set():
                        break
                    arr, sr = item
                    if stream is None:
                        stream = sd.OutputStream(samplerate=sr, channels=1, dtype="float32")
                        stream.start()
                    # Write in slices so a cancel lands within ~50ms.
                    step = max(1, sr // 20)
                    for i in range(0, len(arr), step):
                        if cancel.is_set():
                            break
                        stream.write(arr[i : i + step])
            except Exception as exc:
                log.error("playback failed: %s", exc)
            finally:
                if stream is not None:
                    try:
                        if cancel.is_set():
                            stream.abort()
                        stream.stop()
                        stream.close()
                    except Exception:
                        pass

        thread = threading.Thread(target=player, daemon=True)
        thread.start()
        try:
            for chunk in chunks:
                if cancel.is_set():
                    break
                silent = 0.0
                for result in self._generate(chunk):
                    if cancel.is_set():
                        break
                    arr = np.asarray(result.audio, dtype=np.float32).reshape(-1)
                    sr = getattr(result, "sample_rate", None) or self.sample_rate
                    # Roughly one chunk in eight loops on the silence token instead
                    # of stopping, and then holds the speaker quiet until it hits
                    # its token budget. Real speech never pauses this long inside
                    # one chunk, so treat it as the end of the chunk and move on.
                    if float(np.abs(arr).max()) < 0.01:
                        silent += len(arr) / sr
                        if silent > MAX_SILENCE:
                            log.warning("chunk stalled on silence, moving to the next one")
                            break
                    else:
                        silent = 0.0
                    if not self.native_speed:
                        arr = time_stretch(arr, self.speed, sr)
                    audio_q.put((arr, sr))
        except Exception as exc:
            log.error("synthesis failed: %s", exc)
        finally:
            audio_q.put(None)
        thread.join()


class SystemBackend(Backend):
    """macOS `say`. Zero install, obviously robotic - here to prove the pipeline."""

    def __init__(self, voice: str | None = None, speed: float | None = None,
                 voices: list[str] | None = None) -> None:
        self.voice = voice
        self.speed = float(speed or 1.0)
        self.voice_list = list(voices or [])
        self._proc: subprocess.Popen | None = None

    def play(self, chunks: Iterable[str], cancel: threading.Event) -> None:
        for chunk in chunks:
            if cancel.is_set():
                return
            cmd = ["say", "-r", str(int(180 * self.speed))]
            if self.voice:
                cmd += ["-v", self.voice]
            cmd.append(chunk)
            self._proc = subprocess.Popen(cmd)
            while self._proc.poll() is None:
                if cancel.is_set():
                    self._proc.terminate()
                    return
                time.sleep(0.05)


def build_backend(cfg: Config) -> Backend:
    preset = PRESETS.get(cfg.preset)
    if preset is None:
        raise SystemExit(f"unknown preset {cfg.preset!r}; choose from {list(PRESETS)}")

    if cfg.preset == "system":
        return SystemBackend(voice=cfg.voice, speed=cfg.speed, voices=preset["voices"])

    kwargs = dict(preset["kwargs"])
    if cfg.voice:
        kwargs["voice"] = cfg.voice
    if cfg.speed is not None:
        kwargs["speed"] = cfg.speed
    kwargs.update(cfg.extra_kwargs)
    return MLXBackend(
        cfg.model or preset["model"],
        kwargs,
        preset["sample_rate"],
        native_speed=preset["native_speed"],
        voices=preset["voices"],
    )


# --------------------------------------------------------------------------
# Speaker thread: one utterance at a time, new text interrupts old
# --------------------------------------------------------------------------

class Speaker:
    """One worker thread for every utterance. It is created once and never joined:
    a thread that has used MLX segfaults the process when it exits."""

    def __init__(self, backend: Backend, cfg: Config) -> None:
        self.backend = backend
        self.cfg = cfg
        self._jobs: queue.Queue = queue.Queue()
        self._cancel = threading.Event()
        self.busy = False
        self.loading = False
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self) -> None:
        while True:
            job = self._jobs.get()
            try:
                if callable(job):
                    job()
                else:
                    chunks, cancel = job
                    if not cancel.is_set():
                        self.backend.play(chunks, cancel)
            except Exception as exc:
                log.error("playback failed: %s", exc)
            finally:
                self.busy = not self._jobs.empty()

    def load(self, build) -> None:
        """Swap the backend. The load runs on the worker thread because MLX keeps
        thread-local state: the thread that loads a model must be the one that
        generates with it. Queued after whatever is playing, so `stop` first."""
        self.stop()
        self.loading = True

        def job() -> None:
            try:
                self.backend = build()
            except Exception as exc:
                log.error("load failed, keeping the current model: %s", exc)
            finally:
                self.loading = False

        self._jobs.put(job)

    def stop(self) -> None:
        self._cancel.set()
        if self._jobs.empty():
            self.busy = False

    def speak(self, text: str) -> None:
        self.stop()
        self._cancel = threading.Event()
        chunks = chunk_text(text, self.cfg.chunk_chars)
        log.info("speaking %d chars in %d chunk(s)", len(text), len(chunks))
        self.busy = True
        self._jobs.put((chunks, self._cancel))


# --------------------------------------------------------------------------
# Menu bar
# --------------------------------------------------------------------------

SPEED_CHOICES = (0.75, 1.0, 1.25, 1.5, 2.0)

MENUBAR_STATES = {
    "idle": ("waveform", "Watching clipboard"),
    "speaking": ("waveform.circle.fill", "Speaking..."),
    "paused": ("speaker.slash", "Paused"),
    "loading": ("arrow.down.circle", "Loading model..."),
}


def _submenu(parent, values, label, target, selector: bytes) -> list:
    """Fill a parent item's submenu with radio-style choices. Returns its items.
    Called again when the choices change, e.g. voices after a model switch."""
    from AppKit import NSMenu, NSMenuItem

    parent.setEnabled_(bool(values))
    if not values:
        parent.setSubmenu_(None)
        return []
    sub = NSMenu.alloc().init()
    items = []
    for value in values:
        item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(label(value), selector, "")
        item.setTarget_(target)
        item.setRepresentedObject_(value)
        sub.addItem_(item)
        items.append(item)
    parent.setSubmenu_(sub)
    return items


def build_menubar(cfg: Config, speaker: Speaker, clip: ClipboardReader, stopping: threading.Event):
    """Create the status item and its menu. Returns the controller so the caller
    (or a test) can drive it without an event loop."""
    from AppKit import (
        NSApplication,
        NSApplicationActivationPolicyAccessory,
        NSImage,
        NSMenu,
        NSMenuItem,
        NSStatusBar,
        NSVariableStatusItemLength,
    )
    from Foundation import NSObject, NSRunLoop, NSTimer

    class Controller(NSObject):
        # PyObjC turns underscores into selector colons, so method names stay camelCase.
        def currentState(self) -> str:
            if speaker.loading:
                return "loading"
            if self.paused:
                return "paused"
            return "speaking" if speaker.busy else "idle"

        def refreshUI(self) -> None:
            state = self.currentState()
            self.pauseItem.setTitle_("Resume Watching" if self.paused else "Pause Watching")
            if speaker.backend is not self.voicedBackend:
                # A model switch brings its own speaker names with it.
                self.voicedBackend = speaker.backend
                self.voiceItems = _submenu(
                    self.voiceParent, speaker.backend.voice_list, str, self, b"setVoice:"
                )
            for item in self.presetItems:
                item.setState_(int(item.representedObject() == cfg.preset))
            for item in self.voiceItems:
                item.setState_(int(item.representedObject() == speaker.backend.voice))
            for item in self.speedItems:
                item.setState_(int(item.representedObject() == speaker.backend.speed))
            if state == self.shown:
                return
            self.shown = state
            symbol, label = MENUBAR_STATES[state]
            icon = NSImage.imageWithSystemSymbolName_accessibilityDescription_(symbol, label)
            if icon is None:  # pre-Big Sur, or a symbol this macOS lacks
                self.item.button().setTitle_(label)
            else:
                icon.setTemplate_(True)
                self.item.button().setImage_(icon)
            self.item.button().setToolTip_(label)
            self.statusItem.setTitle_(label)

        def tick_(self, _timer) -> None:
            if stopping.is_set():
                NSApplication.sharedApplication().terminate_(None)
                return
            try:
                raw = clip.poll()
                if raw is not None and not self.paused:
                    cleaned = clean_text(raw)
                    ok, reason = classify(raw, cleaned, cfg)
                    if ok:
                        speaker.speak(cleaned)
                    else:
                        # A short copy doubles as a stop button.
                        log.info("skipped (%s)", reason)
                        speaker.stop()
            except Exception as exc:
                log.error("loop error: %s", exc)
            self.refreshUI()

        def speakNow_(self, _sender) -> None:
            """Read the clipboard aloud even if the filters would skip it."""
            text = clean_text(clip.read() or "")
            if text:
                speaker.speak(text)
            else:
                log.info("nothing to speak")
            self.refreshUI()

        def stopSpeaking_(self, _sender) -> None:
            speaker.stop()
            self.refreshUI()

        def togglePause_(self, _sender) -> None:
            self.paused = not self.paused
            if self.paused:
                speaker.stop()
            self.refreshUI()

        def setVoice_(self, sender) -> None:
            cfg.voice = speaker.backend.voice = sender.representedObject()
            log.info("voice set to %s", speaker.backend.voice)
            save_choices(cfg)
            self.refreshUI()

        def setPreset_(self, sender) -> None:
            name = sender.representedObject()
            if name == cfg.preset:
                return
            # Drop the old model's voice and repo override, keep the speed.
            cfg.preset, cfg.model, cfg.voice = name, None, None
            cfg.speed = speaker.backend.speed
            log.info("loading preset %s ...", name)
            speaker.load(lambda: build_backend(cfg))
            save_choices(cfg)
            self.refreshUI()

        def setSpeed_(self, sender) -> None:
            cfg.speed = speaker.backend.speed = float(sender.representedObject())
            log.info("speed set to %.2fx", speaker.backend.speed)
            save_choices(cfg)
            self.refreshUI()

        def quitApp_(self, _sender) -> None:
            NSApplication.sharedApplication().terminate_(None)

    app = NSApplication.sharedApplication()
    app.setActivationPolicy_(NSApplicationActivationPolicyAccessory)  # no Dock icon

    ctrl = Controller.alloc().init()
    ctrl.paused = False
    ctrl.shown = None
    ctrl.item = NSStatusBar.systemStatusBar().statusItemWithLength_(NSVariableStatusItemLength)

    menu = NSMenu.alloc().init()
    ctrl.statusItem = menu.addItemWithTitle_action_keyEquivalent_("", None, "")
    ctrl.statusItem.setEnabled_(False)
    menu.addItem_(NSMenuItem.separatorItem())
    for title, selector in (
        ("Speak Clipboard", b"speakNow:"),
        ("Stop Speaking", b"stopSpeaking:"),
        ("Pause Watching", b"togglePause:"),
    ):
        entry = menu.addItemWithTitle_action_keyEquivalent_(title, selector, "")
        entry.setTarget_(ctrl)
    ctrl.pauseItem = menu.itemWithTitle_("Pause Watching")
    menu.addItem_(NSMenuItem.separatorItem())

    # Voice and speed take effect on the next utterance, not the one playing now.
    # Model takes effect once the new weights finish loading.
    add = menu.addItemWithTitle_action_keyEquivalent_
    ctrl.presetItems = _submenu(add("Model", None, ""), list(PRESETS), str, ctrl, b"setPreset:")
    ctrl.voiceParent = add("Voice", None, "")
    ctrl.voiceItems = _submenu(ctrl.voiceParent, speaker.backend.voice_list, str, ctrl, b"setVoice:")
    ctrl.speedItems = _submenu(add("Speed", None, ""), SPEED_CHOICES, lambda s: f"{s:g}x", ctrl, b"setSpeed:")
    ctrl.voicedBackend = speaker.backend

    menu.addItem_(NSMenuItem.separatorItem())
    menu.addItemWithTitle_action_keyEquivalent_("Quit", b"quitApp:", "q").setTarget_(ctrl)
    ctrl.item.setMenu_(menu)
    ctrl.refreshUI()

    timer = NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
        cfg.poll_interval, ctrl, b"tick:", None, True
    )
    # Keep ticking while a menu is open, otherwise the icon freezes mid-utterance.
    NSRunLoop.currentRunLoop().addTimer_forMode_(timer, "NSEventTrackingRunLoopMode")
    return ctrl


def run_menubar(cfg: Config) -> int:
    """Status item on the main thread. Synthesis stays on Speaker's worker thread,
    which must never exit (see Speaker)."""
    from AppKit import NSApplication

    backend = build_backend(cfg)
    speaker = Speaker(backend, cfg)
    clip = ClipboardReader()
    if not cfg.speak_on_start:
        clip.poll()  # swallow whatever is already there

    stopping = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stopping.set())
    signal.signal(signal.SIGTERM, lambda *_: stopping.set())

    build_menubar(cfg, speaker, clip, stopping)
    log.info("menu bar ready (preset=%s). copy text to hear it.", cfg.preset)
    NSApplication.sharedApplication().run()
    speaker.stop()
    log.info("stopped")
    return 0


def main(argv: list[str] | None = None) -> int:
    cfg = Config.from_env()

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--preset", choices=list(PRESETS), default=cfg.preset)
    p.add_argument("--model", default=cfg.model, help="override the HF model repo")
    p.add_argument("--voice", default=cfg.voice)
    p.add_argument("--speed", type=float, default=cfg.speed)
    p.add_argument("--min-chars", type=int, default=cfg.min_chars)
    p.add_argument("--max-chars", type=int, default=cfg.max_chars)
    p.add_argument("--poll-interval", type=float, default=cfg.poll_interval)
    p.add_argument("--speak-on-start", action="store_true", default=cfg.speak_on_start)
    p.add_argument("--say", metavar="TEXT", help="speak this once and exit")
    p.add_argument("--check", action="store_true", help="load the model, say a test line, exit")
    p.add_argument("--verbose", "-v", action="store_true")
    args = p.parse_args(argv)

    cfg.preset = args.preset
    cfg.model = args.model
    cfg.voice = args.voice
    cfg.speed = args.speed
    cfg.min_chars = args.min_chars
    cfg.max_chars = args.max_chars
    cfg.poll_interval = args.poll_interval
    cfg.speak_on_start = args.speak_on_start

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else getattr(logging, cfg.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.check or args.say:
        text = args.say or "Clipboard reader is working. This is the local model speaking."
        backend = build_backend(cfg)
        cancel = threading.Event()
        backend.play(chunk_text(clean_text(text), cfg.chunk_chars), cancel)
        return 0

    return run_menubar(cfg)


if __name__ == "__main__":
    sys.exit(main())
