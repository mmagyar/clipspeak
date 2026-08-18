# clipspeak

Copy text on macOS, hear it read aloud by a local neural TTS model. Nothing leaves the machine.

## Run

```bash
./run.sh
```

The first run installs portaudio and ffmpeg with Homebrew, builds `.venv`, and downloads ~2GB of model weights. Later runs start immediately. Arguments are passed through to `clipspeak.py`, so `./run.sh --preset system --check` proves the audio path with no downloads, and `./run.sh --check` proves the model.

The watcher always runs as a menu bar app. There is no headless mode.

## Use

The menu bar icon shows the state: waveform when watching, filled waveform when speaking, crossed-out speaker when paused. Its menu holds Speak Clipboard (reads the clipboard even if the filters would skip it), Stop Speaking, Pause Watching, Voice, Speed, and Quit. Text copied while paused is not read on resume. There is no Dock icon.

Voice lists the speakers the loaded model declares. Speed offers 0.75x to 2x. Both apply to the next utterance, not the one playing, and reset to the preset defaults on restart.

`./run.sh --voice serena` and any other flag still starts the menu bar. Ctrl-C in the terminal quits it, as does the Quit item. `./run.sh --say "text"` and `./run.sh --check` are the two one-shot modes that speak once and exit without an icon.

Copying new text interrupts whatever is playing. Copying anything under 25 characters cancels playback and is not read, so a single word works as a stop button.

Skipped automatically: images and files, bare URLs, bare file paths, single tokens, text over 6000 characters, and anything below 45% letters, which catches JSON, hex, base64 and minified code. Markdown scaffolding is stripped before synthesis. Fenced code blocks become the words "code block", and bare URLs inside prose become the word "link".

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

`CLIPSPEAK_CHUNK_CHARS` (220) sets the synthesis chunk size. Lower it to about 120 for a faster first sound. `CLIPSPEAK_MIN_ALPHA_RATIO` (0.45) sets the code-detection threshold. Lower it if real prose gets rejected.

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
