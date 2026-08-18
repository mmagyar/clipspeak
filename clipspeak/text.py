from __future__ import annotations

import re

# --------------------------------------------------------------------------
# Speaking technical text
# --------------------------------------------------------------------------
# Kokoro's G2P (misaki) treats the dot in "run.sh" as sentence punctuation and
# never says the word "dot", so the extension arrives as an unstressed mumble
# after a pause. Same for paths, operators and SCREAMING_SNAKE names. Fixing it
# here rather than in misaki also fixes the qwen and system presets.

# Only extensions misaki gets wrong. Checked and deliberately absent: json
# ("JAY-son"), yaml/yml ("yammel"), toml ("tommel") already read correctly.
# Single-letter .c and .h are absent too: too easy to fire on prose like "A.C.".
EXTENSIONS = {
    "sh": "S H", "bash": "bash", "zsh": "Z S H", "py": "P Y", "pyc": "P Y C",
    "ts": "T S", "tsx": "T S X", "js": "J S", "jsx": "J S X", "mjs": "M J S",
    "md": "M D", "rs": "R S", "rb": "R B", "go": "go", "cpp": "C plus plus",
    "hpp": "H plus plus", "css": "C S S", "scss": "S C S S", "html": "H T M L",
    "xml": "X M L", "csv": "C S V", "txt": "text", "cfg": "config",
    "ini": "I N I", "sql": "S Q L", "png": "P N G", "jpg": "J P G",
    "jpeg": "J P E G", "gif": "gif", "svg": "S V G", "pdf": "P D F",
    "mp3": "M P 3", "mp4": "M P 4", "wav": "wav", "gz": "G Z", "tgz": "T G Z",
    "zip": "zip", "lock": "lock", "env": "env",
}

# Longest first so "->" wins over "-" and "!=" over "=".
OPERATORS = (
    ("->", " arrow "), ("=>", " arrow "), ("!=", " not equals "),
    ("==", " equals "), (">=", " greater or equal "), ("<=", " less or equal "),
    ("&&", " and "), ("||", " or "), ("::", " colon colon "), ("|>", " pipe "),
)

# Case-sensitive on purpose: real extensions are lowercase, and matching
# uppercase would rewrite "U.S." and "A.C." in ordinary prose.
PATH_TOKEN_RE = re.compile(
    r"(?<![\w/.~-])[~.]{0,2}/?[\w.~/-]*\.(?:"
    + "|".join(sorted(EXTENSIONS, key=len, reverse=True))
    + r")(?!\w)"
)
FLAG_RE = re.compile(r"(?<!\w)--?([A-Za-z][\w-]*)(?!\w)")
SNAKE_RE = re.compile(r"(?<!\w)([A-Za-z_]\w*(?:_\w+)+)(?!\w)")

# Short names misaki spells out letter by letter or slurs into one syllable.
CODE_WORDS = {
    "str": "string", "int": "integer", "bool": "boolean", "len": "length",
    "args": "arguments", "kwargs": "keyword arguments", "init": "initialize",
    "src": "source", "dir": "directory", "env": "environment",
    "repo": "repository", "util": "utility", "impl": "implementation",
    "cfg": "config", "fn": "function", "ret": "return", "elif": "else if",
}
CODE_WORD_RE = re.compile(r"(?<![\w/.])[a-z]{2,6}(?![\w/.])")

JSON_MAX_DEPTH = 6
JSON_MAX_ITEMS = 50


def _say_ident(name: str) -> str:
    """Split an identifier into words. ALL_CAPS is lowercased, otherwise misaki
    spells every letter and CLIPSPEAK_MIN_CHARS becomes seventeen letters."""
    if "_" not in name and "-" not in name:
        return name
    if name.isupper():
        name = name.lower()
    return re.sub(r"[_-]+", " ", name).strip()


def _say_path(m: "re.Match[str]") -> str:
    stem, _, ext = m.group(0).rpartition(".")
    stem = stem.lstrip(".")                       # ./ and ../ carry no meaning
    if stem.startswith("~"):
        stem = "home" + stem[1:]
    parts = [_say_ident(p) for p in stem.split("/") if p]
    return " slash ".join(parts) + " dot " + EXTENSIONS[ext]


def say_tech(text: str) -> str:
    """Rewrite filenames, paths, identifiers and operators as spoken English.

    Shape matching is deliberately narrow, so "and/or", "e.g." and "3.5" in
    ordinary prose come through untouched.
    """
    t = PATH_TOKEN_RE.sub(_say_path, text)
    for symbol, word in OPERATORS:
        t = t.replace(symbol, word)
    t = FLAG_RE.sub(lambda m: _say_ident(m.group(1)), t)
    t = SNAKE_RE.sub(lambda m: _say_ident(m.group(1)), t)
    return re.sub(r"[ \t]{2,}", " ", t)


def _say_scalar(value: object) -> str:
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    return say_tech(str(value))


def say_json(value: object, depth: int = 0) -> str:
    """Narrate parsed JSON structurally: keys, values and list lengths, not
    braces and colons."""
    if depth > JSON_MAX_DEPTH:
        return "nested data."
    if isinstance(value, dict):
        out = ["object."]
        for i, (key, item) in enumerate(value.items()):
            if i >= JSON_MAX_ITEMS:
                out.append(f"and {len(value) - i} more.")
                break
            name = say_tech(str(key))
            if isinstance(item, (dict, list)):
                out.append(f"key {name}, {say_json(item, depth + 1)}")
            else:
                out.append(f"key {name}, value {_say_scalar(item)}.")
        return " ".join(out)
    if isinstance(value, list):
        out = [f"list of {len(value)}."]
        for i, item in enumerate(value):
            if i >= JSON_MAX_ITEMS:
                out.append(f"and {len(value) - i} more.")
                break
            out.append(
                say_json(item, depth + 1)
                if isinstance(item, (dict, list))
                else f"{_say_scalar(item)}."
            )
        return " ".join(out)
    return f"{_say_scalar(value)}."


def say_code(text: str) -> str:
    """Read source line by line. Brackets become pauses, not spoken symbols.

    ponytail: no per-language parsing, so this is literal-with-pauses rather
    than semantic. Add a tree-sitter pass if line-oriented reading proves too
    flat to follow.
    """
    lines = []
    for raw_line in text.split("\n"):
        line = say_tech(raw_line.strip())
        if not line:
            continue
        line = re.sub(r"^(#!|#+|//+|--)\s*", "comment, ", line)
        line = CODE_WORD_RE.sub(lambda m: CODE_WORDS.get(m.group(0), m.group(0)), line)
        line = re.sub(r"(?<=[\w.])/(?=[\w.~])", " slash ", line)  # unspaced = path
        line = re.sub(r"\.(?=[A-Za-z_])", " dot ", line)     # attribute access
        line = line.replace("=", " equals ")
        line = re.sub(r"(?<!\w)[frbu]{1,2}(?=[\"\'])", "", line)   # f"..." prefixes
        line = re.sub(r"[\"\'`]", "", line)
        line = re.sub(r"[(){}\[\]:;,]+", ",", line)          # brackets -> pause
        line = re.sub(r"\s*,\s*", ", ", line)
        line = re.sub(r"\s{2,}", " ", line).strip(" ,")
        if line:
            lines.append(line if line[-1] in ".!?" else line + ".")
    return " ".join(lines)
