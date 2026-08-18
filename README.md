# clipspeak

Copy text on macOS, hear it read aloud by a local neural TTS model. Nothing leaves the machine.

## Run

```bash
./run.sh
```

The first run installs portaudio and ffmpeg with Homebrew, builds `.venv`, and downloads ~2GB of model weights. Later runs start immediately. Arguments are passed through to `clipspeak.py`, so `./run.sh --preset system --check` proves the audio path with no downloads, and `./run.sh --check` proves the model.

The watcher always runs as a menu bar app. There is no headless mode.

## Use

The menu bar icon shows the state: waveform when watching, filled waveform when speaking, crossed-out speaker when paused, down arrow while a model loads. Its menu holds Speak Clipboard (reads the clipboard even if the filters would skip it), Stop Speaking, Pause Watching, Model, Voice, Speed, and Quit. Text copied while paused is not read on resume. There is no Dock icon.

Model switches between the presets without a restart. The switch stops the current utterance, loads the new weights, and rebuilds the voice list from the new model. A `--model` override is dropped. If the load fails, the old model stays.

Voice lists the speakers the loaded model declares. Speed offers 0.75x to 2x. Both apply to the next utterance, not the one playing. Speed survives a model switch, voice does not.

The menu bar writes each pick to `~/.config/clipspeak.json` and reads it at startup. Delete the file to go back to the defaults. An environment variable or a CLI flag still wins over the saved value.

`./run.sh --voice serena` and any other flag still starts the menu bar. Ctrl-C in the terminal quits it, as does the Quit item. `./run.sh --say "text"` and `./run.sh --check` are the two one-shot modes that speak once and exit without an icon.

Copying new text interrupts whatever is playing. Copying anything under 25 characters cancels playback and is not read, so a single word works as a stop button.

Skipped automatically: images and files, bare URLs, bare file paths, single tokens, and text over 6000 characters. Hex, base64 and minified bundles are skipped too, detected by mean token length rather than by counting letters, so real source code still gets through. Markdown scaffolding is stripped before synthesis, and bare URLs inside prose become the word "link".

## Technical text

Neural voices read `run.sh` as a mumble. The grapheme-to-phoneme stage treats the dot as sentence punctuation and never says the word "dot", so the extension arrives unstressed after a pause. clipspeak rewrites technical text into spoken English before synthesis, for every preset.

| Copied | Spoken |
|---|---|
| `run.sh`, `./run.sh` | run dot S H |
| `src/foo/bar.ts` | src slash foo slash bar dot T S |
| `~/code/notes.md` | home slash code slash notes dot M D |
| `min_chars`, `CLIPSPEAK_MIN_CHARS` | min chars, clipspeak min chars |
| `--poll-interval` | poll interval |
| `a -> b`, `x != y`, `c && d` | a arrow b, x not equals y, c and d |

Only known file extensions are rewritten, so `e.g.`, `U.S.`, `3.5` and `and/or` are left alone.

Copy a JSON object or array and it is narrated structurally, not symbol by symbol. `{"id": 1, "tags": ["a","b"]}` becomes "object. key id, value 1. key tags, list of 2. a. b."

Copy source code, or a markdown answer with a fenced block, and the code is read rather than announced. Brackets become pauses, operators become words, and identifiers are split, so `def load_model(path: str) -> Model:` becomes "def load model, path, string, arrow Model." This is line-oriented, not a parser, so it reads code literally with sensible pauses instead of describing what the code does.

## Models

| Preset | Model | Size | Notes |
|---|---|---|---|
| `qwen` (default) | `mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit` | ~2GB | Best quality, a little faster than realtime on Apple Silicon |
| `kokoro` | `mlx-community/Kokoro-82M-bf16` | ~330MB | Near-instant start, 54 voices |
| `system` | macOS `say` | none | For testing the plumbing |

Any other mlx-audio repo works via `--model`. If a model rejects `voice` or `speed`, clipspeak logs a warning and retries without it.

```bash
./run.sh --voice serena
./run.sh --preset kokoro --voice bf_emma --speed 1.15
```

Qwen voices: `aiden`, `dylan`, `eric`, `ryan`, `serena`, `vivian`, `ono_anna`, `sohee`, `uncle_fu`. Kokoro voices: `af_heart`, `af_bella`, `af_nova`, `af_sky`, `am_adam`, `am_echo`, `bf_alice`, `bf_emma`, `bm_daniel`, `bm_george`.

Use a CustomVoice checkpoint, not a Base one. Base checkpoints ship no speaker table, so they invent a new voice for every utterance. clipspeak logs a warning at startup when the loaded model will ignore the voice you asked for.

## Settings

Most settings have a flag and a `CLIPSPEAK_*` environment variable.

| Flag | Env | Default | Effect |
|---|---|---|---|
| `--preset` | `CLIPSPEAK_PRESET` | `qwen` | Model bundle |
| `--voice` | `CLIPSPEAK_VOICE` | preset default | Speaker |
| `--speed` | `CLIPSPEAK_SPEED` | `1.0` | Playback rate |
| `--min-chars` | `CLIPSPEAK_MIN_CHARS` | `25` | Below this, ignore and stop |
| `--max-chars` | `CLIPSPEAK_MAX_CHARS` | `6000` | Above this, ignore |
| `--poll-interval` | `CLIPSPEAK_POLL_INTERVAL` | `0.35` | Seconds between clipboard checks |
| `--speak-on-start` | `CLIPSPEAK_SPEAK_ON_START` | off | Read what is already copied at launch |

`CLIPSPEAK_CHUNK_CHARS` (900) sets the synthesis chunk size. The Qwen preset streams audio out of the model while it generates, so a large chunk does not delay the first sound. Keep it large: each chunk is a separate utterance, and the voice changes tone at every boundary. `CLIPSPEAK_MIN_ALPHA_RATIO` (0.20) rejects text that is neither prose nor code. Lower it if something you want read gets rejected.

Qwen accepts `speed` and then ignores it, so clipspeak stretches the audio itself with WSOLA, which changes duration without moving pitch. Kokoro and `say` handle speed natively. Measured error against the requested factor is under 1.5% across 0.75x to 2x.

The Qwen preset pins `lang_code` to `english`. Language auto-detection drifts into Chinese on short or ambiguous text. `CLIPSPEAK_EXTRA_KWARGS` takes a JSON object and is passed straight to the model, so `CLIPSPEAK_EXTRA_KWARGS='{"lang_code":"german"}'` switches language.

## Troubleshooting

Polling reads `NSPasteboard.changeCount()` and does nothing until that integer moves, so a low poll interval is cheap.

- **No sound, no error.** Check the output device: `./.venv/bin/python -c "import sounddevice; print(sounddevice.query_devices())"`.
- **`PortAudioError` on import.** `brew install portaudio`, then `pip install --force-reinstall sounddevice`.
- **"pyobjc not available".** It falls back to `pbpaste`, which cannot tell images from text. `pip install pyobjc-framework-Cocoa`.
- **Download stalls.** Weights land in `~/.cache/huggingface`. Fetch with resume: `hf download mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit`.
- **"Kokoro requires the optional 'misaki' package".** `pip install "misaki[en]"`. It needs spacy, which has no wheels above Python 3.13. On 3.14, delete `.venv` and let `run.sh` rebuild it with `python3.12`.
- **Slow to start talking.** Lower `CLIPSPEAK_CHUNK_CHARS`, or use `--preset kokoro`.

## Tests

Run them with the venv interpreter: `./.venv/bin/python test_logic.py`. `test_logic.py` covers filtering, chunking and config parsing. `test_playback.py` checks that synthesis runs on the calling thread, which MLX requires, and that an unusable voice is reported. `test_menubar.py` drives the menu bar controller with a fake backend. None of them need a model or an audio device.
