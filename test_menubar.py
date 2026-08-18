"""Menu bar state logic. Uses a fake backend, so no model and no audio device."""
import sys
import threading
import time

sys.path.insert(0, __file__.rsplit("/", 1)[0])

from clipspeak import Config, Speaker, build_menubar


class FakeBackend:
    """Blocks until released, so `busy` has a window to be observed in."""

    voice = "ryan"
    speed = 1.0
    voice_list = ["ryan", "serena"]

    def __init__(self) -> None:
        self.release = threading.Event()
        self.spoken: list[list[str]] = []

    def play(self, chunks, cancel) -> None:
        self.spoken.append(list(chunks))
        self.release.wait(timeout=5)


class FakeClipboard:
    def __init__(self) -> None:
        self.text: str | None = None
        self.fresh = False

    def poll(self) -> str | None:
        if not self.fresh:
            return None
        self.fresh = False
        return self.text

    def read(self) -> str | None:
        return self.text

    def put(self, text: str) -> None:
        self.text, self.fresh = text, True


def wait_for(predicate, what: str, timeout: float = 5.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError(f"timed out waiting for {what}")


cfg = Config()
backend = FakeBackend()
speaker = Speaker(backend, cfg)
clip = FakeClipboard()
ctrl = build_menubar(cfg, speaker, clip, threading.Event())

LONG = "This is a long enough sentence of ordinary prose that the filter lets it through."

assert ctrl.currentState() == "idle", ctrl.currentState()
assert ctrl.item.menu().numberOfItems() == 10, ctrl.item.menu().numberOfItems()

# voice and speed submenus, with the current value ticked
assert [i.title() for i in ctrl.voiceItems] == ["ryan", "serena"]
assert [i.title() for i in ctrl.speedItems] == ["0.75x", "1x", "1.25x", "1.5x", "2x"]
assert [i.state() for i in ctrl.voiceItems] == [1, 0], "current voice not ticked"
assert [i.state() for i in ctrl.speedItems] == [0, 1, 0, 0, 0], "current speed not ticked"

ctrl.setVoice_(ctrl.voiceItems[1])
assert backend.voice == "serena", backend.voice
assert [i.state() for i in ctrl.voiceItems] == [0, 1], "tick did not follow the voice"

ctrl.setSpeed_(ctrl.speedItems[3])
assert backend.speed == 1.5, backend.speed
assert [i.state() for i in ctrl.speedItems] == [0, 0, 0, 1, 0], "tick did not follow the speed"

# a clipboard change speaks and shows the speaking state
clip.put(LONG)
ctrl.tick_(None)
wait_for(lambda: backend.spoken, "the fake backend to receive text")
assert ctrl.currentState() == "speaking", ctrl.currentState()

# stop returns to idle
ctrl.stopSpeaking_(None)
backend.release.set()
wait_for(lambda: not speaker.busy, "speaker to go idle")
assert ctrl.currentState() == "idle", ctrl.currentState()

# paused: clipboard changes are swallowed, not spoken
ctrl.togglePause_(None)
assert ctrl.currentState() == "paused", ctrl.currentState()
assert ctrl.pauseItem.title() == "Resume Watching", ctrl.pauseItem.title()
before = len(backend.spoken)
clip.put("Another perfectly readable sentence that should be ignored while paused.")
ctrl.tick_(None)
time.sleep(0.2)
assert len(backend.spoken) == before, "spoke while paused"

# resuming does not replay what was copied while paused
ctrl.togglePause_(None)
ctrl.tick_(None)
time.sleep(0.2)
assert len(backend.spoken) == before, "replayed clipboard from the paused period"
assert ctrl.currentState() == "idle", ctrl.currentState()

# "Speak Clipboard" ignores the filters
clip.text = "short"
ctrl.speakNow_(None)
wait_for(lambda: len(backend.spoken) == before + 1, "forced speak")
assert backend.spoken[-1] == ["short"], backend.spoken[-1]

print("all menu bar tests passed")
