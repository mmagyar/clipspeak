from __future__ import annotations

import logging
import signal
import threading

from . import backends
from .clipboard import ClipboardReader
from .config import PRESETS, Config, save_choices
from .filter import classify, clean_text
from .speaker import Speaker

log = logging.getLogger("clipspeak")


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
            speaker.load(lambda: backends.build_backend(cfg))
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

    backend = backends.build_backend(cfg)
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
