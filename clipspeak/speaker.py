from __future__ import annotations

import logging
import queue
import threading

from .backends import Backend
from .config import Config
from .filter import chunk_text

log = logging.getLogger("clipspeak")


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
        self.loading = False
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self) -> None:
        while True:
            job = self._jobs.get()
            try:
                if callable(job):
                    job()
                else:
                    chunks, cancel = job
                    if not cancel.is_set():
                        self.backend.play(chunks, cancel)
            except Exception as exc:
                log.error("playback failed: %s", exc)
            finally:
                self.busy = not self._jobs.empty()

    def load(self, build) -> None:
        """Swap the backend. The load runs on the worker thread because MLX keeps
        thread-local state: the thread that loads a model must be the one that
        generates with it. Queued after whatever is playing, so `stop` first."""
        self.stop()
        self.loading = True

        def job() -> None:
            try:
                self.backend = build()
            except Exception as exc:
                log.error("load failed, keeping the current model: %s", exc)
            finally:
                self.loading = False

        self._jobs.put(job)

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
