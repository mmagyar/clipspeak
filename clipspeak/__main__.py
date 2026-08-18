#!/usr/bin/env python3
"""
clipspeak - watch the macOS clipboard and read new text aloud with a local neural TTS model.

Nothing leaves your machine. Copy some text, hear it. Copy something else, it
switches to the new text. Copy a single short word (e.g. "stop") to shut it up.

Usage:
    python -m clipspeak                    # watch the clipboard, menu bar icon
    python -m clipspeak --check            # load the model and say a test line
    python -m clipspeak --say "some text"  # one-shot, no clipboard involved
    python -m clipspeak --preset kokoro    # use the small/fast model instead

Config lives in Config in clipspeak/config.py; every value can be overridden
with a CLIPSPEAK_* environment variable or a CLI flag. The menu bar saves the
model, voice and speed you pick to ~/.config/clipspeak.json and reads them at
startup.
"""

from __future__ import annotations

import argparse
import logging
import sys
import threading

from .backends import build_backend
from .config import PRESETS, Config
from .filter import chunk_text, clean_text
from .menubar import run_menubar


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
