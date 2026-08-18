from __future__ import annotations

import logging
import subprocess

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
