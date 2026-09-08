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
PRE_ROLL = 0.1      # seconds of silence played ahead of the first syllable so
                    # the device starts on something quiet.
LEAD_KEEP = 0.05    # seconds of closure silence kept in front of the first
                    # plosive. A "d" without its closure reads as clipped.
FILLER = "Uh. "     # spoken lead-in prepended to the first chunk of an
                    # utterance. Kokoro renders a word-initial plosive with a
                    # weak burst when the word starts the utterance (measured
                    # on bf_emma: less than half the high-frequency onset
                    # energy of the same word mid-sentence), which hears as a
                    # clipped first consonant no matter what the playback path
                    # does: pre-rolls, noise bursts and guards all failed.
                    # Putting a filler word first moves the content to the
                    # middle of the utterance, where the model renders it
                    # properly. The filler's own audio is then cut out before
                    # playback, so nothing extra is heard and latency only
                    # grows by the closure.
FILLER_GAP = 0.035  # seconds of quiet separating the filler's audio from the
                    # content's first onset.
HEAD_WINDOW = 0.5   # seconds after the filler's first sound in which the
                    # filler-to-content gap must appear. Rendering is
                    # deterministic for a voice and speed (measured over
                    # repeated runs: gap lands within ~0.11-0.18s), so a gap
                    # this early is the boundary, not a pause inside the
                    # content.
CUT_BUFFER = 2.0    # seconds of audio to inspect for the content onset before
                    # giving up and playing the chunk with the filler intact.


def content_onset(arr, sr: int, threshold: float = 0.02):
    """Sample index where the real content starts in a filler-prefixed chunk,
    or None while that is still unclear. The filler is the first sound in the
    buffer; the first quiet stretch of FILLER_GAP after it is the boundary, and
    the content's onset is where that stretch ends. Only the head of the buffer
    is searched, so pauses between the content's own words can never be
    mistaken for the boundary. Called again after every new piece of audio
    arrives, so None just means 'keep buffering'; if no boundary shows up the
    caller plays the chunk with the filler heard, which beats cutting into the
    content."""
    import numpy as np

    loud = np.abs(arr) > threshold
    first = int(np.argmax(loud))
    if not loud[first]:            # still quiet at the head
        return None
    limit = min(len(arr), first + int(sr * HEAD_WINDOW))
    min_gap = int(sr * FILLER_GAP)
    i = first
    while i < limit:
        if not loud[i]:
            j = i
            while j < limit and not loud[j]:
                j += 1
            if j - i >= min_gap:
                # Keep up to LEAD_KEEP of closure in front of the burst, but
                # never reach back into the filler: a fast rendering leaves
                # less quiet there than LEAD_KEEP.
                return max(i, j - int(sr * LEAD_KEEP))
            i = j
        else:
            i += 1
    return None


def trim_lead_silence(arr, sr: int, threshold: float = 0.01):
    """Cut most of a chunk's leading silence, keeping LEAD_KEEP seconds of it.
    A plosive still needs its closure in front of the burst to sound right."""
    import numpy as np

    onset = int(np.argmax(np.abs(arr) > threshold))
    if np.abs(arr[onset]) <= threshold:   # all quiet, nothing to protect
        return arr
    return arr[max(0, onset - int(sr * LEAD_KEEP)):]


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

    @property
    def voice(self) -> str | None:
        return self._voice

    @voice.setter
    def voice(self, value: str | None) -> None:
        self._voice = value
        # Kokoro encodes the language in the voice name's first letter
        # (af_... American, bf_... British) and the gender in the second. The
        # preset pins lang_code for its default voice, so a menu-bar pick from
        # the other language leaves the pipeline's G2P running the wrong
        # lexicon: mlx-audio logs "Language mismatch" and renders British
        # voices with American pronunciations. Keep them in step. Voices for
        # other models don't match the two-letter shape, so they leave
        # lang_code alone; a lang_code explicitly set via
        # CLIPSPEAK_EXTRA_KWARGS is honoured only for the default voice.
        code = self.gen_kwargs.get("lang_code")
        if (value and isinstance(code, str) and len(code) == 1
                and value[:2] in {c + g for c in "abefhipjz" for g in "fm"}):
            self.gen_kwargs["lang_code"] = value[0]

    def __init__(
        self,
        model_repo: str,
        gen_kwargs: dict,
        sample_rate: int,
        native_speed: bool = True,
        voices: list[str] | None = None,
        first_chunk_prefix: str | None = None,
    ) -> None:
        import numpy as np  # noqa: F401  (checked early so failure is obvious)
        import sounddevice as sd  # noqa: F401
        from mlx_audio.tts.utils import load_model

        log.info("loading %s ...", model_repo)
        t0 = time.time()
        self.model = load_model(model_repo)
        self.gen_kwargs = dict(gen_kwargs)
        self.sample_rate = sample_rate
        self.voice = gen_kwargs.get("voice")   # setter syncs kokoro's lang_code
        log.info("model ready in %.1fs", time.time() - t0)

        talker = getattr(getattr(self.model, "config", None), "talker_config", None)
        speakers = sorted(getattr(talker, "spk_id", None) or {})
        self.native_speed = native_speed
        self.speed = float(gen_kwargs.get("speed") or 1.0)
        self.voice_list = speakers or list(voices or [])
        self.first_chunk_prefix = first_chunk_prefix
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
        # Open the stream before synthesis, so the first audio lands in a running
        # device instead of racing its startup.
        stream = sd.OutputStream(samplerate=self.sample_rate, channels=1, dtype="float32")
        stream.start()

        def player() -> None:
            rate = self.sample_rate
            tail = 0.0            # last sample sent, so the ending can ramp from it
            first = True
            try:
                while True:
                    item = audio_q.get()
                    if item is None or cancel.is_set():
                        break
                    arr, sr = item
                    if first:
                        first = False
                        stream.write(np.zeros(int(sr * PRE_ROLL), dtype=np.float32))
                        # Only the utterance's first chunk: later chunks are
                        # sentences whose lead silence is the pause between
                        # them, and that pacing should survive.
                        arr = trim_lead_silence(arr, sr)
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
        prefix = getattr(self, "first_chunk_prefix", None)
        first_chunk = True
        try:
            for chunk in chunks:
                if cancel.is_set():
                    break
                try:
                    self._synth_chunk(chunk, audio_q, cancel,
                                      prefix if first_chunk else None)
                except Exception as exc:
                    # One chunk failing (misaki hits a word its lexicon and its
                    # fallback both choke on) should not mute the rest of the
                    # utterance.
                    log.error("chunk failed, skipping it: %s", exc)
                finally:
                    first_chunk = False
        finally:
            audio_q.put(None)
        thread.join()

    def _synth_chunk(self, chunk: str, audio_q: queue.Queue,
                     cancel: threading.Event, prefix: str | None = None) -> None:
        import numpy as np

        silent = 0.0
        buf: list[np.ndarray] = []   # audio held back while hunting the content onset
        cutting = prefix is not None
        for result in self._generate(prefix + chunk if prefix else chunk):
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
            if cutting:
                # Hold the filler-prefixed audio until the content onset is
                # located, then drop everything before it. content_onset only
                # reads the head of the buffer, so buffered content never
                # skews the boundary. Failing to locate it within CUT_BUFFER
                # plays the chunk with the filler heard, which beats cutting
                # into the content.
                buf.append(arr)
                whole = buf[0] if len(buf) == 1 else np.concatenate(buf)
                cut = content_onset(whole, sr)
                if cut is not None:
                    audio_q.put((whole[cut:], sr))
                    cutting = False
                    buf = []
                elif len(whole) > int(sr * CUT_BUFFER):
                    audio_q.put((whole, sr))
                    cutting = False
                    buf = []
                continue
            audio_q.put((arr, sr))
        if buf:
            # The generation ended while still hunting; play what is there.
            audio_q.put((buf[0] if len(buf) == 1 else np.concatenate(buf), sr))


def system_voices() -> list[str]:
    """The voices `say` can actually use, discovered at runtime so system voices
    installed after this file was written show up in the menu."""
    try:
        out = subprocess.run(
            ["say", "-v", "?"], capture_output=True, text=True, timeout=5, check=True
        ).stdout
    except Exception as exc:
        log.warning("could not list system voices: %s", exc)
        return []
    voices = []
    for line in out.splitlines():
        # Voice names can contain spaces ("Bad News"); two or more spaces
        # separate the name from its locale, and "#" starts the sample text.
        m = re.match(r"^(.+?)\s{2,}(\S+)\s+#", line)
        if m:
            voices.append(m.group(1).strip())
    return voices


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
        voices = system_voices() or preset["voices"]
        return SystemBackend(voice=cfg.voice, speed=cfg.speed, voices=sorted(voices))

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
        first_chunk_prefix=preset.get("first_chunk_prefix"),
    )
