from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field

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
