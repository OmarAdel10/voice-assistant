# Voice Assistant

Offline-first voice assistant for academic field training.

## Overview

Voice Assistant is a privacy-focused, offline-first voice assistant designed for academic field training. It runs entirely on your local machine with no cloud dependencies for core functionality (STT, NLP, TTS).

**Key Features:**
- **Offline Speech-to-Text**: Uses `faster-whisper` (tiny.en model, int8 quantization, CPU)
- **Intent Recognition**: Regex-based NLP with fuzzy fallback (5 intents)
- **Offline Text-to-Speech**: `pyttsx3` (primary) with `gTTS` online fallback
- **System Actions**: Time, date, system info, app launching, web search
- **Latency Target**: <2.5s end-to-end (speech end → audio start)

## Installation

### System Dependencies (Fedora / PipeWire)

```bash
# Audio system (PipeWire/ALSA)
sudo dnf install pipewire pipewire-pulse alsa-utils

# For gTTS playback (optional but recommended)
sudo dnf install mpv ffmpeg
```

### Python Environment

```bash
# Using uv (recommended)
uv venv
uv pip install -e ".[dev]"

# Or using pip
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

The `[dev]` extra includes: `pytest`, `pytest-cov`, `ruff`, `mypy`, `build`, `pre-commit`.

## Usage

### CLI Modes

| Mode | Command | Description |
|------|---------|-------------|
| Listen (loop) | `voice-assistant` or `voice-assistant --listen` | Continuous listening mode (default) |
| Single shot | `voice-assistant --once` | One interaction: listen → process → speak → exit |
| Text input | `voice-assistant --once --text "query"` | Bypass STT, process text directly |
| List intents | `voice-assistant --list-intents` | Show available commands |
| Version | `voice-assistant --version` | Show version |
| Help | `voice-assistant --help` | Show usage |

### Example Interactions

```bash
# Continuous mode (Ctrl+C to exit)
$ voice-assistant
[INFO] voice_assistant.main: Starting voice loop. Press Ctrl+C to exit.
[INFO] voice_assistant.main: Listening...
...speak "what time is it"...
[INFO] core.nlp_engine: NLP: 2ms | Intent: get_time | Confidence: 1.00
[INFO] __main__: Intent: get_time | Entities: {} | Confidence: 1.00
[INFO] __main__: Response: 10:30 AM
[INFO] core.tts_engine: TTS: 45ms | Text length: 8 chars

# Single shot with voice
$ voice-assistant --once
[INFO] voice_assistant.main: Listening...
[INFO] voice_assistant.main: Transcribing...
[INFO] core.nlp_engine: NLP: 1ms | Intent: get_date | Confidence: 1.00
[INFO] __main__: Intent: get_date | Entities: {} | Confidence: 1.00
[INFO] __main__: Response: Monday, January 15, 2024
[INFO] core.tts_engine: TTS: 38ms | Text length: 25 chars

# Text mode (bypasses STT, great for testing)
$ voice-assistant --once --text "what time is it"
[INFO] core.nlp_engine: NLP: 0ms | Intent: get_time | Confidence: 1.00
[INFO] __main__: Intent: get_time | Entities: {} | Confidence: 1.00
[INFO] __main__: Response: 10:31 AM
[INFO] core.tts_engine: TTS: 21ms | Text length: 8 chars

# List available intents
$ voice-assistant --list-intents
Available intents:
  get_time: "what time is it", "current time", "tell me the time"...
  get_date: "what date is it", "today's date", "what day is it"...
  get_sys_info: "system info", "system status", "cpu usage"...
  open_app: "open {app}", "launch {app}", "start {app}"...
  web_search: "search for {query}", "google {query}", "look up {query}"...
```

## Supported Intents

| Intent | Example Phrases | Entities | Description |
|--------|-----------------|----------|-------------|
| `get_time` | "what time is it", "current time" | — | Current time (HH:MM AM/PM) |
| `get_date` | "what date is it", "today's date" | — | Current date (Weekday, Month DD, YYYY) |
| `get_sys_info` | "system info", "cpu usage", "memory" | — | CPU%, Memory%, Disk% |
| `open_app` | "open firefox", "launch vscode" | `app` | Launch app (must be in PATH) |
| `web_search` | "search for cats", "google python" | `query` | Open browser with search |

## Performance Targets

| Metric | Target | Measured (typical) |
|--------|--------|-------------------|
| STT (tiny.en, CPU) | <1.5s | ~1.2s for 3s audio |
| NLP parsing | <50ms | ~1-2ms |
| TTS (pyttsx3) | <500ms | ~20-50ms |
| **End-to-end** | **<2.5s** | **~1.5-2.0s** |

## Configuration

Create `config.yaml` in project root (optional — defaults are sensible):

```yaml
stt:
  model_size: "tiny.en"      # tiny.en, base.en, small.en, etc.
  device: "cpu"              # cpu, cuda
  compute_type: "int8"       # int8, float16, float32
  language: "en"
  vad_threshold: 0.5
  max_listen_seconds: 10

tts:
  engine: "pyttsx3"          # pyttsx3 | gTTS
  rate: 180                  # words per minute
  volume: 0.9                # 0.0 - 1.0
  language: "en"             # for gTTS

nlp:
  confidence_threshold: 0.6

audio:
  sample_rate: 16000
  channels: 1

log:
  level: "INFO"              # DEBUG, INFO, WARNING, ERROR
```

Environment variables override config (prefix `VA_`, nested with `__`):
```bash
VA_STT__MODEL_SIZE=base.en VA_TTS__ENGINE=gTTS voice-assistant
```

## Troubleshooting

### Microphone Permissions

```bash
# Check available devices
arecord -l

# Test recording
arecord -d 3 -f cd test.wav && aplay test.wav

# PipeWire permissions (if needed)
pavucontrol  # GUI to check input device selection
```

### Model Cache

Whisper models cached at `~/.cache/huggingface/hub/` (first run downloads ~75MB for tiny.en).

```bash
# Pre-download model
python -c "from faster_whisper import WhisperModel; WhisperModel('tiny.en', device='cpu', compute_type='int8')"
```

### Common Issues

| Symptom | Cause | Fix |
|---------|-------|-----|
| "No microphone found" | PipeWire/PulseAudio not exposing device | Run `pavucontrol`, check Input Devices tab |
| "Model load failed" | No internet for first download | Ensure internet, or pre-download (above) |
| "TTS fallback" | gTTS network error / pyttsx3 missing | Install `espeak-ng` for pyttsx3 |
| "App not found in PATH" | App not installed or not in $PATH | `which <app>` to verify, install if needed |
| High latency | Model too large / CPU slow | Use `tiny.en` with `int8` (default) |

## Development

### Running Tests

```bash
# All tests with coverage
pytest tests/ -v --cov=core --cov-fail-under=80

# Specific test file
pytest tests/test_stt.py -v

# With pre-commit hooks
pre-commit run --all-files
```

### Code Quality Gates

```bash
ruff format --check .
ruff check .
mypy core/ --ignore-missing-imports
pytest tests/ -v --cov=core --cov-fail-under=80
python -m build
```

### Pre-commit

```bash
pre-commit install
pre-commit run --all-files
```

## Architecture

```
┌─────────────┐     Audio      ┌──────────┐     Text     ┌────────┐     Text     ┌────────────┐     Audio     ┌──────────┐
│  Microphone │ ─────────────► │    STT   │ ───────────► │  NLP   │ ───────────► │  Actions   │ ───────────► │   TTS    │ ───────► │ Speakers │
└─────────────┘   (sounddevice) └──────────┘   (faster-   └────────┘   (regex +   └────────────┘   (pyttsx3/  └──────────┘
                                              whisper)              fuzzy)                gTTS)
```

## Specifications

- [PRD](docs/specs/PRD.md)
- [Architecture](docs/specs/ARCHITECTURE.md)
- [Design](docs/specs/DESIGN.md)
- [User Flow](docs/specs/USER_FLOW.md)
- [Development Environment](docs/specs/DEVELOPMENT_ENVIRONMENT.md)

## License

Academic field training project — no warranty implied.