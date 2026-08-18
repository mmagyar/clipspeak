"""Logic tests for clipspeak's filtering and chunking. No macOS or MLX needed."""
import sys
sys.path.insert(0, __file__.rsplit("/", 1)[0])

from clipspeak import Config, clean_text, classify, chunk_text, alpha_ratio

cfg = Config()

SHOULD_SKIP = [
    ("", "empty clipboard"),
    ("https://example.com/some/long/path?x=1", "a bare URL"),
    ("www.nytimes.com", "a bare domain"),
    ("/Users/mate/Documents/report.pdf", "a file path"),
    ("~/code/project", "a home-relative path"),
    ("clipspeak", "a single word"),
    ("hello there", "too short"),
    ("a" * 7000, "too long"),
    ('{"id":1,"k":"v","n":[1,2,3],"z":0.5,"q":"::"}', "JSON"),
    ("8f3a2b1c9d4e5f60718293a4b5c6d7e8f9012345", "a hex blob"),
]

SHOULD_SPEAK = [
    "The quick brown fox jumps over the lazy dog, and then it does it again for good measure.",
    "Check out https://example.com for details. It has a full write-up of the benchmark results and methodology.",
    "# Heading\n\nSome **bold** text with a `code` span and a [link](https://x.com) in the middle of the paragraph.",
]

fails = []

for text, label in SHOULD_SKIP:
    ok, reason = classify(text, clean_text(text), cfg)
    status = "PASS" if not ok else "FAIL"
    if ok:
        fails.append(f"expected skip for {label!r}, got speak")
    print(f"{status}  skip {label:22} -> {reason}")

print()
for text in SHOULD_SPEAK:
    cleaned = clean_text(text)
    ok, reason = classify(text, cleaned, cfg)
    status = "PASS" if ok else "FAIL"
    if not ok:
        fails.append(f"expected speak for {text[:40]!r}, got skip ({reason})")
    print(f"{status}  speak -> {cleaned[:70]!r}")

print()
print("--- cleaning ---")
md = "## Results\n\n- item one\n- item two\n\nSee ![img](a.png) and [the docs](https://d.io).\n\n```\ncode here\n```\n"
print(repr(clean_text(md)))

print()
print("--- chunking ---")
passage = (
    "This is the first sentence of a longer passage. Here is a second one, which is "
    "somewhat longer and contains a few clauses, commas, and other punctuation. "
    "Third sentence! Fourth one? And a fifth that trails off at the end of the block "
    "so we can confirm the chunker groups things sensibly rather than emitting one "
    "chunk per sentence."
)
chunks = chunk_text(clean_text(passage), cfg.chunk_chars)
for i, c in enumerate(chunks):
    print(f"  [{i}] ({len(c):3d}) {c}")
if not chunks:
    fails.append("chunker returned nothing")
if any(len(c) > cfg.chunk_chars * 2.2 for c in chunks):
    fails.append("a chunk was far over target size")
rejoined = " ".join(chunks)
if len(rejoined.split()) != len(clean_text(passage).split()):
    fails.append("chunking lost or duplicated words")

print()
print("--- long single sentence hard-split ---")
long_one = "It was a long sentence, with many clauses, " * 12
lc = chunk_text(long_one.strip(), cfg.chunk_chars)
print(f"  {len(lc)} chunks, max len {max(len(c) for c in lc)}")

print()


print()
print("--- extra kwargs ---")
import os
os.environ["CLIPSPEAK_EXTRA_KWARGS"] = '{"lang_code": "german"}'
if Config.from_env().extra_kwargs != {"lang_code": "german"}:
    fails.append("CLIPSPEAK_EXTRA_KWARGS was not applied")
for bad in ('{"lang_code"', '["german"]'):
    os.environ["CLIPSPEAK_EXTRA_KWARGS"] = bad
    try:
        Config.from_env()
        fails.append(f"CLIPSPEAK_EXTRA_KWARGS={bad} was accepted")
    except SystemExit as exc:
        print(f"  rejected {bad!r} -> {exc}")
del os.environ["CLIPSPEAK_EXTRA_KWARGS"]

if fails:
    print("FAILURES:")
    for f in fails:
        print("  -", f)
    sys.exit(1)
print("all logic tests passed")
