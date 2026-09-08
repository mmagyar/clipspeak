#!/usr/bin/env bash
# clipspeak - prepares the venv on first run, then starts the menu bar app.
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"
PYBIN="$DIR/.venv/bin/python"
DEPS=(mlx-audio sounddevice numpy pyobjc-framework-Cocoa "misaki[en]" "spacy>=3.8.16,<3.9")
# Preinstall pinned blis/thinc wheels: newer thinc pins blis<1.1, which has no
# cp313 wheels and falls back to a source build that fails under Cython.
PINNED=("blis>=1.3,<1.4" "thinc>=8.3.12,<8.4")

say() { printf "\033[1;34m==>\033[0m %s\n" "$*"; }
warn() { printf "\033[1;33m!!\033[0m %s\n" "$*"; }

ready() { "$PYBIN" -c "import importlib.util as u,sys; sys.exit(any(u.find_spec(m) is None for m in ('mlx_audio','sounddevice','numpy','AppKit','misaki','spacy')))" 2>/dev/null; }

# misaki[en] (Kokoro's text processing) pulls spacy, which has no wheels above
# 3.13, so build the venv with an older interpreter.
pick_python() {
  for p in ${PYTHON:-} python3.13 python3.12 python3; do
    command -v "$p" >/dev/null 2>&1 || continue
    "$p" -c 'import sys; raise SystemExit(sys.version_info >= (3, 14))' 2>/dev/null || continue
    echo "$p"; return 0
  done
  return 1
}

if ! ready; then
  [[ "$(uname -s)" == "Darwin" ]] || warn "Built for macOS. MLX will not work here."
  [[ "$(uname -m)" == "arm64" ]] || warn "MLX needs Apple Silicon. Use --preset system."

  if command -v brew >/dev/null 2>&1; then
    # portaudio backs sounddevice; ffmpeg covers mlx-audio's decoding paths.
    # espeak-ng is misaki's fallback for words missing from its lexicon: without
    # it Kokoro crashes with "unsupported operand type(s) for +: 'NoneType' and
    # 'str'" on the first out-of-dictionary word instead of skipping it.
    for pkg in portaudio ffmpeg espeak-ng; do
      brew list "$pkg" >/dev/null 2>&1 && continue
      say "Installing $pkg (Homebrew)"
      brew install "$pkg"
    done
  else
    warn "No Homebrew. Install portaudio yourself, or audio output will fail."
  fi

  # An existing venv on 3.14+ cannot install spacy, so replace it.
  if [[ ! -x "$PYBIN" ]] || ! "$PYBIN" -c 'import sys; raise SystemExit(sys.version_info >= (3, 14))'; then
    PY="$(pick_python)" || { warn "Need Python 3.13 or older (spacy has no newer wheels). Try: brew install python@3.12"; exit 1; }
    say "Creating virtualenv at .venv with $("$PY" -V)"
    rm -rf .venv
    "$PY" -m venv .venv
  fi

  say "Installing Python packages (~1GB of wheels, this takes a few minutes)"
  "$PYBIN" -m pip install --upgrade --quiet pip wheel || true
  "$PYBIN" -m pip install "${PINNED[@]}" || true
  "$PYBIN" -m pip install "${DEPS[@]}" || true

  ready || { warn "Install failed, see the output above. Missing: ${DEPS[*]}"; exit 1; }
  say "Ready. The first run downloads ~2GB of model weights from Hugging Face."
fi

exec "$PYBIN" -m clipspeak "$@"
