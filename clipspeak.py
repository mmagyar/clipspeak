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
CLIPSPEAK_* environment variable or a CLI flag.
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
        "kwargs": {"voice": "ryan", "speed": 1.0, "lang_code": "english"},
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
    chunk_chars: int = 220            # target size of each synthesis chunk
    min_alpha_ratio: float = 0.45     # below this it's probably code/data, skip
    speak_on_start: bool = False      # read whatever is already on the clipboard

    log_level: str = "INFO"
    extra_kwargs: dict = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> "Config":
        cfg = cls()
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


log = logging.getLogger("clipspeak")


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
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?;:])\s+|\n{2,}")


def clean_text(text: str) -> str:
    """Light normalisation so the model doesn't read markdown scaffolding aloud."""
    t = text.replace("\r\n", "\n").replace("\r", "\n")
    t = re.sub(r"```[\s\S]*?```", " (code block) ", t)      # fenced code
    t = re.sub(r"`([^`]+)`", r"\1", t)                       # inline code
    t = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", t)              # images
    t = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", t)           # links -> label
    t = URL_RE.sub(" link ", t)                              # bare urls
    t = re.sub(r"^\s{0,3}#{1,6}\s*", "", t, flags=re.M)      # headings
    t = re.sub(r"^\s{0,3}[-*+]\s+", "", t, flags=re.M)       # bullets
    t = re.sub(r"\*\*([^*]+)\*\*", r"\1", t)                 # bold
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{2,}", "\n\n", t)
    return t.strip()


def alpha_ratio(text: str) -> float:
    dense = [c for c in text if not c.isspace()]
    if not dense:
        return 0.0
    return sum(c.isalpha() for c in dense) / len(dense)


def classify(raw: str, cleaned: str, cfg: Config) -> tuple[bool, str]:
    """Return (should_speak, reason).

    URL and path checks run against the *raw* clipboard text, because cleaning
    rewrites bare URLs to the word "link" and would hide them from us.
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
    if len(t) < cfg.min_chars:
        return False, "too short"
    if " " not in t:
        return False, "single token"
    ratio = alpha_ratio(t)
    if ratio < cfg.min_alpha_ratio:
        return False, f"looks like code or data (alpha ratio {ratio:.2f})"
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

    def _generate(self, text: str):
        """Call generate(), dropping kwargs the model doesn't accept."""
        kwargs = dict(self.gen_kwargs)
        if self.voice:
            kwargs["voice"] = self.voice
        if self.native_speed:
            kwargs["speed"] = self.speed
        while True:
            try:
                return list(self.model.generate(text=text, **kwargs))
            except TypeError as exc:
                bad = re.search(r"'(\w+)'", str(exc))
                if bad and bad.group(1) in kwargs:
                    log.warning("model rejected kwarg %r, retrying without it", bad.group(1))
                    kwargs.pop(bad.group(1))
                    continue
                raise

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
                for result in self._generate(chunk):
                    if cancel.is_set():
                        break
                    arr = np.asarray(result.audio, dtype=np.float32).reshape(-1)
                    sr = getattr(result, "sample_rate", None) or self.sample_rate
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
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self) -> None:
        while True:
            chunks, cancel = self._jobs.get()
            try:
                if not cancel.is_set():
                    self.backend.play(chunks, cancel)
            except Exception as exc:
                log.error("playback failed: %s", exc)
            finally:
                self.busy = not self._jobs.empty()

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
}


def _submenu(menu, title: str, values, label, target, selector: bytes) -> list:
    """Add a titled submenu of radio-style choices. Returns its items."""
    from AppKit import NSMenu, NSMenuItem

    parent = menu.addItemWithTitle_action_keyEquivalent_(title, None, "")
    if not values:
        parent.setEnabled_(False)
        return []
    sub = NSMenu.alloc().init()
    items = []
    for value in values:
        item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(label(value), selector, "")
        item.setTarget_(target)
        item.setRepresentedObject_(value)
        sub.addItem_(item)
        items.append(item)
    menu.setSubmenu_forItem_(sub, parent)
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
            if self.paused:
                return "paused"
            return "speaking" if speaker.busy else "idle"

        def refreshUI(self) -> None:
            state = self.currentState()
            self.pauseItem.setTitle_("Resume Watching" if self.paused else "Pause Watching")
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
            speaker.backend.voice = sender.representedObject()
            log.info("voice set to %s", speaker.backend.voice)
            self.refreshUI()

        def setSpeed_(self, sender) -> None:
            speaker.backend.speed = float(sender.representedObject())
            log.info("speed set to %.2fx", speaker.backend.speed)
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

    # Both settings take effect on the next utterance, not the one playing now.
    ctrl.voiceItems = _submenu(menu, "Voice", speaker.backend.voice_list, str, ctrl, b"setVoice:")
    ctrl.speedItems = _submenu(menu, "Speed", SPEED_CHOICES, lambda s: f"{s:g}x", ctrl, b"setSpeed:")

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
