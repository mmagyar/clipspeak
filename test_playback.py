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
from clipspeak.backends import PRE_ROLL, LEAD_KEEP

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


# _generate must stay lazy: the streaming models hand back audio a second at a
# time, and collecting it first would put that whole wait in front of the sound.
class FakeModel:
    def __init__(self, reject: str | None = None) -> None:
        self.reject = reject
        self.yielded = 0

    def generate(self, text: str, **kwargs):
        if self.reject and self.reject in kwargs:
            raise TypeError(f"generate() got an unexpected keyword argument '{self.reject}'")
        for _ in range(3):
            self.yielded += 1
            yield types.SimpleNamespace(audio=np.zeros(2400, dtype=np.float32), sample_rate=24000)


lazy = object.__new__(MLXBackend)
lazy.model = FakeModel()
lazy.gen_kwargs = {"stream": True}
lazy.voice = "ryan"
lazy.native_speed = True
lazy.speed = 1.0

it = lazy._generate("hello")
next(it)
assert lazy.model.yielded == 1, f"_generate ran ahead: {lazy.model.yielded} items before the first"
assert len(list(it)) == 2, "_generate dropped the remaining audio"

lazy.model = FakeModel(reject="stream")
assert len(list(lazy._generate("hello"))) == 3, "_generate must retry without a rejected kwarg"

print("all generate tests passed")


# A low temperature can make the model loop on the silence token forever, so the
# token budget has to cap a chunk near its real length, not at the 4096 default.
from clipspeak import token_budget

SENTENCE = "The model loads once at startup and stays resident in memory." * 3
assert token_budget(SENTENCE) < 4096, "budget must beat the library default"
assert token_budget(SENTENCE) / 12.5 > len(SENTENCE) / 15, "budget must fit the text it has to speak"
assert token_budget("hi") >= 64, "a short chunk still needs a floor"

print("all token budget tests passed")


# A stalled generation streams silence until it runs out of tokens. play() has to
# cut that chunk short rather than hold the speaker quiet for a minute.
import clipspeak

stall = object.__new__(MLXBackend)
stall.sample_rate = 24000
stall.native_speed = True
stall.speed = 1.0
played: list[float] = []
fake_sd.OutputStream = lambda **kw: types.SimpleNamespace(
    start=lambda: None, write=lambda a: played.append(len(a) / 24000),
    stop=lambda: None, close=lambda: None, abort=lambda: None,
)

def stalling(text: str):
    yield types.SimpleNamespace(audio=np.full(24000, 0.5, dtype=np.float32), sample_rate=24000)
    for _ in range(60):   # a minute of dead air
        yield types.SimpleNamespace(audio=np.zeros(24000, dtype=np.float32), sample_rate=24000)

stall._generate = stalling
stall.play(["one"], threading.Event())
assert sum(played) < 1 + clipspeak.MAX_SILENCE + 1, f"played {sum(played):.0f}s, so the stall was not cut"
assert sum(played) >= 1, "the real speech before the stall must still play"

print("all stall tests passed")


# A hard cut from a mid-waveform sample to silence is an audible click. Every
# ending has to ramp down first, and a cancel has to drain that ramp, not drop it.
def play_recording(generate, stop_after: int | None = None):
    """Play one chunk through a fake stream. Returns (samples, abort_calls)."""
    out: list[np.ndarray] = []
    aborts: list[bool] = []
    cancel = threading.Event()

    def write(a):
        out.append(np.asarray(a, dtype=np.float32).reshape(-1).copy())
        if stop_after is not None and len(out) == stop_after:
            cancel.set()

    fake_sd.OutputStream = lambda **kw: types.SimpleNamespace(
        start=lambda: None, write=write, stop=lambda: None,
        close=lambda: None, abort=lambda: aborts.append(True),
    )
    back = object.__new__(MLXBackend)
    back.sample_rate = SR
    back.native_speed = True
    back.speed = 1.0
    back._generate = generate
    back.play(["one"], cancel)
    return np.concatenate(out), aborts


def loud(text: str):
    for _ in range(4):
        yield types.SimpleNamespace(audio=np.full(SR, 0.5, dtype=np.float32), sample_rate=SR)


# Skip the pre-roll: it steps from silence into this fake's constant 0.5, which
# real speech (starting at zero) never does.
def after_pre_roll(a):
    return a[int(SR * PRE_ROLL) :]


played, _ = play_recording(loud)
assert abs(played[-1]) < 0.02, f"audio ends at {played[-1]:.2f}, so it clicks"
assert np.abs(np.diff(after_pre_roll(played))).max() < 0.02, "the ending must ramp, not step"
assert len(played) >= 4 * SR, "the ramp must not eat the speech"

cut, aborts = play_recording(loud, stop_after=3)
assert not aborts, "abort() drops the ramp, so a cancel must drain instead"
assert abs(cut[-1]) < 0.02, f"a cancelled ending sits at {cut[-1]:.2f}, so it clicks"
assert np.abs(np.diff(after_pre_roll(cut))).max() < 0.02, "a cancelled ending must ramp, not step"
assert len(cut) < 4 * SR, "cancel must still cut the audio short"

assert np.abs(played[: int(SR * PRE_ROLL)]).max() == 0, "the first audio needs silence in front"

print("all click tests passed")


# Kokoro renders a word-initial plosive with a weak burst, so the first chunk
# is prefixed with "Uh. " and the filler's audio cut out again. The content
# onset is the onset after the longest quiet stretch seen so far.
from clipspeak.backends import content_onset, FILLER

quiet = lambda sec: np.zeros(int(SR * sec), dtype=np.float32)
tone = lambda sec, hz: (0.3 * np.sin(2 * np.pi * hz * np.arange(int(SR * sec)) / SR)).astype(np.float32)

prefixed = np.concatenate([quiet(0.2), tone(0.1, 300), quiet(0.1), tone(0.5, 600)])
cut = content_onset(prefixed, SR)
assert cut is not None and abs(cut - int(SR * 0.35)) <= 2, \
    f"content onset should sit at the tone after the pause, got {cut}"
assert prefixed[cut - int(SR * LEAD_KEEP) : cut].max() == 0, "the cut must keep closure silence"
assert cut >= int(SR * 0.3), "the filler must be cut off"

long_single = np.concatenate([quiet(0.4), tone(1.2, 600)])
assert content_onset(long_single, SR) is None, \
    "no early gap after the first sound: the filler must have merged, so don't cut"

short_single = np.concatenate([quiet(0.2), tone(0.05, 600)])
assert content_onset(short_single, SR) is None, "a lone early onset must keep buffering"

# A quiet stretch shorter than LEAD_KEEP: keep all of it, never cut into the
# sound in front of it.
short_stretch = np.concatenate([quiet(0.2), tone(0.1, 300), quiet(0.04), tone(0.5, 600)])
cut = content_onset(short_stretch, SR)
assert cut is not None and abs(cut - int(SR * 0.3)) <= 2, \
    f"a short closure must be kept whole, cut at {cut/SR:.3f}s"
assert short_stretch[cut] == 0 and short_stretch[cut - 1] != 0, "the cut must not clip the filler"

# A pause deep inside the content must never read as the boundary.
deep_pause = np.concatenate([quiet(0.2), tone(0.4, 600), quiet(0.08), tone(0.4, 600)])
cut = content_onset(deep_pause, SR)
assert cut is not None and abs(cut - int(SR * 0.63)) <= 2, \
    f"the first gap after the first sound is the boundary, got {cut/SR:.2f}s"

# End-to-end through _synth_chunk: the prefix reaches the model, the filler
# audio does not reach the queue.
back = object.__new__(MLXBackend)
back.sample_rate = SR
back.native_speed = True
back.speed = 1.0
back.first_chunk_prefix = FILLER
got_text, out = [], []

def filler_generate(text: str):
    got_text.append(text)
    yield types.SimpleNamespace(
        audio=np.concatenate([quiet(0.2), tone(0.1, 300), quiet(0.1), tone(0.5, 600)]),
        sample_rate=SR,
    )

back._generate = filler_generate
import queue as _q
q = _q.Queue()
back._synth_chunk("hello", q, threading.Event(), FILLER)
assert got_text == [FILLER + "hello"], f"prefix missing from synthesis text: {got_text}"
arr, sr = q.get_nowait()
first_loud = int(np.argmax(np.abs(arr) > 0.02))
assert first_loud < int(SR * 0.2), f"content should start right after LEAD_KEEP, at {first_loud/SR:.2f}s"
assert abs(arr).max() > 0.2 and np.allclose(np.unique(np.round(np.abs(arr[:first_loud]), 4)), [0.0]), \
    "the filler audio must be gone"

print("all filler cut tests passed")
