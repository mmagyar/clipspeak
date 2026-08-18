from __future__ import annotations

import logging
import queue
import re
import subprocess
import threading
import time
from typing import Iterable, Iterator

from .config import PRESETS, Config

log = logging.getLogger("clipspeak")


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
FADE = 0.008        # seconds of ramp-down at the end, so the speaker does not click
PRE_ROLL = 0.08     # seconds of silence written into a fresh stream: a just-started
                    # stream renders its first block soft, and that must eat
                    # silence instead of the first syllable


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
        # Open and prime the stream before synthesis, so the first audio lands
        # in a warm device instead of racing its startup.
        stream = sd.OutputStream(samplerate=self.sample_rate, channels=1, dtype="float32")
        stream.start()
        stream.write(np.zeros(int(self.sample_rate * PRE_ROLL), dtype=np.float32))

        def player() -> None:
            rate = self.sample_rate
            tail = 0.0            # last sample sent, so the ending can ramp from it
            try:
                while True:
                    item = audio_q.get()
                    if item is None or cancel.is_set():
                        break
                    arr, sr = item
                    # Write in slices so a cancel lands within ~50ms.
                    step = max(1, sr // 20)
                    for i in range(0, len(arr), step):
                        if cancel.is_set():
                            break
                        block = arr[i : i + step]
                        if len(block):
                            stream.write(block)
                            tail = float(block[-1])
            except Exception as exc:
                log.error("playback failed: %s", exc)
            finally:
                try:
                    # Every ending here is a hard cut: cancelled, stalled on
                    # silence, or out of tokens. Dropping from mid-waveform
                    # straight to silence is an audible click, so slide the
                    # last sample down to zero first.
                    if tail:
                        stream.write(np.linspace(tail, 0.0, max(1, int(rate * FADE)),
                                                 dtype=np.float32))
                    stream.stop()   # drains, unlike abort(), so the ramp is heard
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
