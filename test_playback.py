"""MLX segfaults when a thread that used it exits, so synthesis must stay on the
calling thread. Fake sounddevice/model keep this runnable without audio or MLX."""
import sys
import threading
import types

import numpy as np

fake_sd = types.ModuleType("sounddevice")
fake_sd.OutputStream = lambda **kw: types.SimpleNamespace(
    start=lambda: None, write=lambda a: None, stop=lambda: None,
    close=lambda: None, abort=lambda: None,
)
sys.modules["sounddevice"] = fake_sd

from clipspeak import MLXBackend

backend = object.__new__(MLXBackend)
backend.sample_rate = 24000
seen: list[str] = []


def fake_generate(text: str):
    seen.append(threading.current_thread().name)
    return [types.SimpleNamespace(audio=np.zeros(2400, dtype=np.float32), sample_rate=24000)]


backend._generate = fake_generate
backend.play(["one", "two"], threading.Event())

main = threading.current_thread().name
assert seen == [main, main], f"synthesis ran off the calling thread: {seen}"
print("all playback tests passed")


from clipspeak import voice_warning

assert voice_warning([], "ryan"), "empty speaker table must warn"
assert voice_warning(["ryan", "serena"], None), "missing voice must warn"
assert voice_warning(["ryan", "serena"], "chelsie"), "unknown voice must warn"
assert voice_warning(["Ryan", "serena"], "ryan") is None, "known voice must not warn"

print("all voice tests passed")


from clipspeak import time_stretch

SR = 24000
tone = np.sin(2 * np.pi * 140 * np.arange(SR) / SR).astype(np.float32)


def peak_hz(a):
    return np.fft.rfftfreq(len(a), 1 / SR)[np.argmax(np.abs(np.fft.rfft(a * np.hanning(len(a)))))]


for factor in (0.75, 1.25, 1.5, 2.0):
    out = time_stretch(tone, factor, SR)
    ratio = len(out) / len(tone)
    assert abs(ratio - 1 / factor) < 0.05, f"{factor}x gave a length ratio of {ratio:.3f}"
    assert abs(peak_hz(out) - 140) < 5, f"{factor}x moved the pitch to {peak_hz(out):.1f} Hz"

assert len(time_stretch(tone, 1.0, SR)) == len(tone), "1.0x must be a no-op"
assert len(time_stretch(tone[:100], 1.5, SR)) == 100, "too-short input must pass through"

print("all time stretch tests passed")
