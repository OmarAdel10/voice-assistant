# Voice Assistant

Offline-first voice assistant for academic field training with Egyptian Arabic + English support.

## Overview

Voice Assistant is a privacy-focused, offline-first voice assistant designed for academic field training. It runs entirely on your local machine with no cloud dependencies for core functionality (STT, NLP, TTS).

**Key Features:**
- **Offline Speech-to-Text**: Uses `faster-whisper` (Whisper Medium, int8 quantization, CUDA)
- **LLM-Powered Intent Parsing**: Qwen 2.5 1.5B (GGUF) for semantic understanding of Egyptian Arabic + English code-switching
- **Fallback NLP**: Regex-based patterns with fuzzy matching for reliability
- **Offline Text-to-Speech**: Piper (primary, Arabic + English voices) with pyttsx3/gTTS fallback
- **Smart App Launcher**: Fuzzy matching + dnf/flatpak install suggestions with voice confirmation
- **Languages**: Egyptian Arabic (العربية المصرية) + English with code-switching support
- **Latency Target**: <2.5s end-to-end (speech end → audio start)

## Installation

### System Dependencies (Fedora / PipeWire / CUDA)

```bash
# Audio system (PipeWire/ALSA)
sudo dnf install pipewire pipewire-pulse alsa-utils

# For Piper TTS playback
sudo dnf install mpv ffmpeg

# CUDA for GPU acceleration (RTX 3050 Ti 4GB tested)
sudo dnf install cuda-toolkit
# Or follow NVIDIA Fedora guide for your GPU
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

# Install llama-cpp-python with GPU support
pip install llama-cpp-python --verbose
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
| Voice enrollment | `voice-assistant --enroll` | Record phrases for speaker adaptation |
| Version | `voice-assistant --version` | Show version |
| Help | `voice-assistant --help` | Show usage |

### Example Interactions

```bash
# Continuous mode (Ctrl+C to exit)
$ voice-assistant
[INFO] voice_assistant.main: Starting voice loop. Press Ctrl+C to exit.
[INFO] voice_assistant.main: Listening...
...speak "ما الوقت"...
[INFO] core.llm_engine: LLM: intent=get_time | confidence=1.00 | lang=ar
[INFO] voice_assistant.main: Response: الساعة 11:30 صباحاً.
[INFO] core.tts_engine: TTS: 50ms | Text length: 20 chars

# Single shot with voice
$ voice-assistant --once
[INFO] voice_assistant.main: Listening...
[INFO] voice_assistant.main: Transcribing...
...speak "open code"...
[INFO] core.llm_engine: LLM: intent=open_app | confidence=1.00 | lang=en
[INFO] voice_assistant.main: Response: Opening VS Code for you.

# Text mode (bypasses STT, great for testing)
$ voice-assistant --once --text "what time is it"
[INFO] core.llm_engine: LLM: intent=get_time | confidence=1.00 | lang=en
[INFO] voice_assistant.main: Response: It is 2:30 PM.

# Arabic text mode
$ voice-assistant --once --text "افتح الكود"
[INFO] core.llm_engine: LLM: intent=open_app | confidence=1.00 | lang=ar
[INFO] voice_assistant.main: Response: تم تشغيل الكود.

# Code-switching (Arabic + English)
$ voice-assistant --once --text "open الكود"
[INFO] core.llm_engine: LLM: intent=open_app | confidence=1.00 | lang=ar
[INFO] voice_assistant.main: Response: Successfully launched code.

# Voice enrollment (improves STT accuracy for your voice)
$ voice-assistant --enroll
🎤 Voice Enrollment
==================================================
Please speak each phrase clearly when prompted.
Press Enter when ready to record each phrase.

Phrase 1/6:
  📝 "مرحبا، كيف حالك؟"
  Press Enter to start recording (5 seconds)...
```

### Voice Enrollment

Run `voice-assistant --enroll` to record 6 phrases (3 Arabic + 3 English) for speaker adaptation. This creates an `initial_prompt` that significantly improves STT accuracy for your voice and accent. Enrollment is saved to `~/.config/voice-assistant/enrollment.json` and persists across restarts.

## Supported Intents

| Intent | Example Phrases (EN) | Example Phrases (AR) | Entities | Description |
|--------|---------------------|---------------------|----------|-------------|
| `get_time` | "what time is it", "current time" | "ما الوقت", "الساعة كام" | — | Current time (HH:MM AM/PM) |
| `get_date` | "what date is it", "today's date" | "التاريخ", "النهاردة كام" | — | Current date (Weekday, Month DD, YYYY) |
| `get_sys_info` | "system info", "cpu usage" | "معلومات النظام", "النظام عامل إزاي" | — | CPU%, Memory%, Disk% |
| `open_app` | "open firefox", "launch vscode" | "افتح فايرفوكس", "شغل الكود" | `app` | Launch app (fuzzy match + install suggestions) |
| `web_search` | "search for cats", "google python" | "ابحث عن قطط", "جوجل بايثون" | `query` | Open browser with search |

**Egyptian dialect support**: "أبن الكود" → "افتح الكود", "قوت" → "كود", "الساعة كام" → get_time

**30+ Arabic app name mappings**: "كود/في إس كود/فيجوال ستوديو" → `code`, "فايرفوكس/المتصفح" → `firefox`, "تيرمينال/الطرفية" → `gnome-terminal`, etc.

**Custom aliases**: Add `~/.config/voice-assistant/app_aliases.json` for personal app names.

## Performance Targets

| Metric | Target | Measured (RTX 3050 Ti 4GB) |
|--------|--------|-------------------|
| STT (Medium, int8, CUDA) | <1.5s | ~0.8s for 5s audio |
| LLM parsing (Qwen 1.5B, int4, CUDA) | <500ms | ~300ms |
| TTS (Piper) | <500ms | ~300ms |
| **End-to-end** | **<2.5s** | **~1.5-2.0s** |

## Configuration

Create `config.yaml` in project root (optional — defaults are sensible):

```yaml
stt:
  model_size: "medium"           # tiny.en, base.en, small, medium, large-v3
  device: "cuda"                 # cpu, cuda
  compute_type: "int8"           # int8, float16, float32
  language: null                 # auto-detect (null), "ar", "en"
  allowed_languages: ["ar", "en"] # restrict transcription languages
  language_detection_threshold: 0.7 # stricter detection
  vad_filter: true
  vad_min_silence_ms: 500
  max_listen_seconds: 5
  initial_prompt: null           # set by voice enrollment (--enroll)

tts:
  engine: "piper"                # piper | pyttsx3
  rate: 180
  volume: 0.9
  piper_voice_dir: "models/tts"
  piper_voice_ar: "ar_JO-kareem-medium"
  piper_voice_en: "en_US-lessac-medium"
  piper_voice_ar_fallback: "ar_JO-kareem-low"
  piper_voice_en_fallback: "en_US-lessac-low"

nlp:
  confidence_threshold: 0.6
  confidence_threshold_ar: 0.5
  confidence_threshold_en: 0.6

llm:
  model_path: "models/llm/qwen2.5-1.5b-instruct-q4_k_m.gguf"
  enabled: true
  fallback_to_nlp: true
  confidence_threshold: 0.7
  max_tokens: 256
  temperature: 0.1
  n_gpu_layers: -1
  n_ctx: 4096
  n_threads: 4

audio:
  sample_rate: 16000
  channels: 1

log:
  level: "INFO"
```

Environment variables override config (prefix `VA_`, nested with `__`):
```bash
VA_STT__MODEL_SIZE=large-v3 VA_LLM__ENABLED=false voice-assistant
```

## Model Management

### STT Models
- Stored in `models/stt/`
- Auto-downloaded on first run if missing
- Convert to faster-whisper CT2 format: `ct2-transformers-converter --model openai/whisper-medium --output_dir models/stt/whisper-medium --quantization int8`

### LLM Model
- Place GGUF file at `models/llm/qwen2.5-1.5b-instruct-q4_k_m.gguf`
- Download from: `huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct-GGUF qwen2.5-1.5b-instruct-q4_k_m.gguf --local-dir models/llm`

### TTS Voices (Piper)
- Stored in `models/tts/`
- Pre-download: `python scripts/download_models.py`

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

### CUDA / GPU Issues

```bash
# Check CUDA
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"

# Check llama-cpp-python GPU layers
python -c "from llama_cpp import Llama; Llama(model_path='models/llm/...', n_gpu_layers=-1)"
```

### Common Issues

| Symptom | Cause | Fix |
|---------|-------|-----|
| "No microphone found" | PipeWire/PulseAudio not exposing device | Run `pavucontrol`, check Input Devices tab |
| "Model load failed" | No internet for first download | Ensure internet, or pre-download models |
| "TTS fallback" | Piper missing voices / pyttsx3 missing | Run `scripts/download_models.py`, install `espeak-ng` |
| "App not found in PATH" | App not installed or not in $PATH | `which <app>` to verify, install if needed |
| High STT latency | Model too large / CPU only | Use `medium` with `int8` on CUDA (default) |
| Low STT accuracy | No enrollment / dialect mismatch | Run `voice-assistant --enroll` |
| LLM not loading | llama-cpp-python build failed | Install build tools: `sudo dnf install build-essential cmake` |

### VRAM Usage (4GB GPU)

| Component | Model | Quantization | VRAM |
|-----------|-------|--------------|------|
| STT | Whisper Medium | int8 | ~2.0 GB |
| LLM | Qwen 2.5 1.5B | q4_k_m (int4) | ~1.0 GB |
| **Total** | | | **~3.0 GB** |

Leaves ~1GB headroom for KV cache and overhead.

## Development

### Running Tests

```bash
# All tests with coverage
pytest tests/ -v --cov=core --cov-fail-under=50

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
pytest tests/ -v --cov=core --cov-fail-under=50
python -m build
```

### Pre-commit

```bash
pre-commit install
pre-commit run --all-files
```

### Adding Tests for LLM Engine

```bash
# Run specific LLM tests (when added)
pytest tests/test_llm.py -v
```

## Architecture

```
┌─────────────┐     Audio      ┌──────────┐     Text     ┌────────────┐
│  Microphone │ ─────────────► │    STT   │ ───────────► │    LLM     │
│  (sounddev) │   (sounddevice) │ (faster- │   (raw +     │  (Qwen     │
└─────────────┘                │ whisper) │    stt_lang) │  2.5 1.5B) │
                               └──────────┘              └─────┬──────┘
                                    │                         │
                                    │              ┌──────────▼──────────┐
                                    │              │   Fallback NLP    │
                                    │              │  (Regex + Fuzzy)  │
                                    │              └──────────┬────────┘
                                    │                         │
                                    ▼                         ▼
                           ┌──────────────────┐    ┌──────────────────┐
                           │    Actions       │    │      TTS         │
                           │ (open_app, etc.) │    │  (Piper primary) │
                           └────────┬─────────┘    └────────┬─────────┘
                                    │                       │
                                    └───────────┬───────────┘
                                                ▼
                                         ┌──────────────┐
                                         │   Speakers   │
                                         └──────────────┘
```

## Specifications

- [PRD](docs/specs/PRD.md)
- [Architecture](docs/specs/ARCHITECTURE.md)
- [Design](docs/specs/DESIGN.md)
- [User Flow](docs/specs/USER_FLOW.md)
- [Development Environment](docs/specs/DEVELOPMENT_ENVIRONMENT.md)

## Phase History

| Phase | Branch | Description |
|-------|--------|-------------|
| 1 | `feature/stt-engine` | STT with GPU config, model downloader, multilingual |
| 2 | `feature/nlp-engine` | Bilingual NLP with Arabic/English intents |
| 3 | `feature/actions` | Smart app launcher with fuzzy matching + install |
| 4 | `feature/tts-engine` | Piper TTS with fallbacks, voice downloads |
| 5 | `feature/integration` | Full pipeline, CLI modes, error recovery |
| 6 | `feature/stt-phase1` | STT accuracy: language restrict, enrollment, Arabic mapping |
| 7 | `feature/stt-phase2` | **LLM intent parsing with Qwen 2.5 1.5B** |

## License

Academic field training project — no warranty implied.