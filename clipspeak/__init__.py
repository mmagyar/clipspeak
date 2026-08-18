"""clipspeak: watch the macOS clipboard and read new text aloud with a local
neural TTS model. Re-exports the package's public names."""
from .config import (
    CHOICES_PATH,
    PRESETS,
    Config,
    load_choices,
    save_choices,
)
from .clipboard import ClipboardReader
from .text import say_code, say_json, say_tech
from .filter import (
    alpha_ratio,
    chunk_text,
    classify,
    clean_text,
    looks_like_code,
)
from .backends import (
    MAX_SILENCE,
    Backend,
    MLXBackend,
    SystemBackend,
    build_backend,
    time_stretch,
    token_budget,
    voice_warning,
)
from .speaker import Speaker
from .menubar import (
    MENUBAR_STATES,
    SPEED_CHOICES,
    build_menubar,
    run_menubar,
)

__all__ = [
    "CHOICES_PATH",
    "PRESETS",
    "Config",
    "load_choices",
    "save_choices",
    "ClipboardReader",
    "say_code",
    "say_json",
    "say_tech",
    "alpha_ratio",
    "chunk_text",
    "classify",
    "clean_text",
    "looks_like_code",
    "MAX_SILENCE",
    "Backend",
    "MLXBackend",
    "SystemBackend",
    "build_backend",
    "time_stretch",
    "token_budget",
    "voice_warning",
    "Speaker",
    "MENUBAR_STATES",
    "SPEED_CHOICES",
    "build_menubar",
    "run_menubar",
]
