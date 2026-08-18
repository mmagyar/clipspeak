from __future__ import annotations

import json
import re

from .config import Config
from .text import say_code, say_json, say_tech

# --------------------------------------------------------------------------
# Filtering: decide whether a clipboard entry is worth reading
# --------------------------------------------------------------------------

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.I)
BARE_URLISH_RE = re.compile(
    r"^(?:https?://|www\.|[\w.-]+\.(?:com|org|net|io|dev|ai|co|uk|hu|de|edu|gov)\b)\S*$",
    re.I,
)
PATH_RE = re.compile(r"^(?:~|/|\./|\.\./|[A-Za-z]:\\)\S*$")
BLOB_MEAN_TOKEN = 20              # above this it is hex, base64 or minified
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?;:])\s+|\n{2,}")
FENCE_RE = re.compile(r"```(?:[A-Za-z0-9_+.-]*\n)?([\s\S]*?)```")

# Source code scores high on at least one of these. Prose and markdown score
# zero on both, including markdown that contains fenced code.
CODE_LINE_END_RE = re.compile(r"[:{};,)\]\\]$")


def looks_like_code(text: str) -> bool:
    """True when the clipboard holds raw source rather than prose.

    Measured across this repo: README.md and plain prose score 0.00/0.00, while
    run.sh scores 0.17/0.64, clipspeak.py 0.63/0.87, and Python and TypeScript
    snippets 1.00/0.50 or better.
    """
    lines = [ln.rstrip() for ln in text.split("\n") if ln.strip()]
    if not lines:
        return False
    ends = sum(bool(CODE_LINE_END_RE.search(ln)) for ln in lines) / len(lines)
    indented = sum(ln[0] in " \t" for ln in lines) / len(lines)
    return ends >= 0.3 or indented >= 0.4


def _as_json(text: str) -> object | None:
    """Parsed JSON if `text` is a JSON object or array, else None."""
    if text[:1] not in "{[":
        return None
    try:
        value = json.loads(text)
    except ValueError:
        return None
    return value if isinstance(value, (dict, list)) else None


def clean_text(text: str) -> str:
    """Light normalisation so the model doesn't read markdown scaffolding aloud."""
    data = _as_json(text.strip())
    if data is not None:
        return say_json(data)
    t = text.replace("\r\n", "\n").replace("\r", "\n")
    # A fence means this is a document that contains code, not source itself, so
    # the markdown path below handles it and reads only the fenced parts as code.
    if "```" not in t and looks_like_code(t):
        return say_code(t)
    t = FENCE_RE.sub(lambda m: " " + say_code(m.group(1)) + " ", t)
    t = re.sub(r"`([^`]+)`", r"\1", t)                       # inline code
    t = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", t)              # images
    t = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", t)           # links -> label
    t = URL_RE.sub(" link ", t)                              # bare urls
    t = re.sub(r"^\s{0,3}#{1,6}\s*", "", t, flags=re.M)      # headings
    t = re.sub(r"^\s{0,3}[-*+]\s+", "", t, flags=re.M)       # bullets
    t = re.sub(r"\*\*([^*]+)\*\*", r"\1", t)                 # bold
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{2,}", "\n\n", t)
    return say_tech(t).strip()


def alpha_ratio(text: str) -> float:
    dense = [c for c in text if not c.isspace()]
    if not dense:
        return 0.0
    return sum(c.isalpha() for c in dense) / len(dense)


def classify(raw: str, cleaned: str, cfg: Config) -> tuple[bool, str]:
    """Return (should_speak, reason).

    The shape checks all run against the *raw* clipboard text. Cleaning rewrites
    bare URLs to the word "link" and turns code into words, so measuring the
    cleaned text would hide exactly what we are trying to detect.
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
    if len(r) >= cfg.min_chars and _as_json(r) is not None:
        return True, "json"
    if len(t) < cfg.min_chars:
        return False, "too short"
    if " " not in r:
        # Raw, not cleaned: normalising "==" to " equals " would otherwise put
        # spaces into a base64 blob and make it look like prose.
        return False, "single token"
    # Mean token length separates content from blobs with a wide margin: prose
    # 3.9, shell 5.6, Python 6.8-10, against 36 for minified JS and 48 for
    # base64. A single long line in real source does not move the mean, which is
    # why this beats measuring the longest run. URLs are legitimately long and
    # already became "link", so measure without them.
    words = URL_RE.sub(" ", r).split()
    mean_token = sum(len(w) for w in words) / len(words)
    if mean_token > BLOB_MEAN_TOKEN:
        return False, f"unreadable blob (mean token {mean_token:.0f} chars)"
    ratio = alpha_ratio(r)
    if ratio < cfg.min_alpha_ratio:
        return False, f"looks like data, not code or prose (alpha ratio {ratio:.2f})"
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
