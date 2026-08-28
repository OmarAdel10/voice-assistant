# Follow-ups & Deferred Items

All out-of-scope items discovered during development, triaged with rationale.

---

## Wake Word / Keyword Spotting
**Status**: DEFERRED — Out of scope for MVP
**Reason**: MVP targets push-to-talk / single-shot mode. Wake word requires persistent audio buffer and separate model (e.g., Porcupine, openWakeWord). Will be Phase 11+.

---

## Multi-language Support
**Status**: DEFERRED — Out of scope for MVP
**Reason**: MVP targets English only (`tiny.en`). Adding languages requires model selection, language detection, and locale-aware TTS. Future enhancement.

---

## GUI / System Tray
**Status**: DEFERRED — Out of scope for MVP
**Reason**: CLI-only per PRD. GUI adds dependency complexity (GTK/Qt) and platform-specific code. Future enhancement.

---

## Plugin System
**Status**: DEFERRED — Out of scope for MVP
**Reason**: Action registry is hardcoded. Plugin system requires dynamic loading, sandboxing, and API stability. Future enhancement.

---

## Conversation Memory / Context
**Status**: DEFERRED — Out of scope for MVP
**Reason**: Each interaction is stateless. Context requires dialogue management, entity persistence, session handling. Future enhancement.

---

## Windows/macOS Support
**Status**: DEFERRED — Out of scope for MVP
**Reason**: Target is Fedora Linux (Wayland/PipeWire). Cross-platform audio (sounddevice) works but untested. CI only runs on Linux.

---

## Alternative TTS Engines (Coqui, Piper, etc.)
**Status**: DEFERRED — Out of scope for MVP
**Reason**: pyttsx3 + gTTS cover offline + online. New engines add maintenance burden. Future enhancement.

---

## Streaming / Incremental STT
**Status**: DEFERRED — Out of scope for MVP
**Reason**: Current STT processes full utterance. Streaming requires VAD integration and partial results handling. Future enhancement.

---

## Custom Intent Training (ML-based NLP)
**Status**: DEFERRED — Out of scope for MVP
**Reason**: Regex-based NLP is deterministic and fast. ML intents need training data, model versioning, retraining pipeline. Future enhancement.

---

## Audio Feedback (Beeps/Chimes)
**Status**: PARTIAL — System bell (`\a`) implemented in CLI for start/end
**Reason**: Minimal UX feedback. Custom sounds add asset management. Can expand later.

---

## Config Hot Reload
**Status**: DEFERRED — Out of scope for MVP
**Reason**: Config loaded once at startup. Hot reload requires file watching and engine reinitialization. Low priority.

---

## Structured Logging (JSON)
**Status**: DEFERRED — Out of scope for MVP
**Reason**: Current logging is human-readable. Structured logs need log aggregation infrastructure. Future enhancement.

---

## Metrics / Telemetry
**Status**: DEFERRED — Out of scope for MVP
**Reason**: No metrics collection in MVP. Privacy-focused project avoids telemetry by default. Could be opt-in later.

---

## Docker/Podman Support
**Status**: DEFERRED — Out of scope for MVP
**Reason**: Audio device passthrough in containers is complex (--device=/dev/snd). CI tests on host.

---

## systemd Service / Auto-start
**Status**: DEFERRED — Out of scope for MVP
**Reason**: User-facing CLI tool, not a daemon. Service file template could be added later.

---

## Man Page / Completion Scripts
**Status**: DEFERRED — Out of scope for MVP
**Reason**: Click generates `--help` and basic completion. Full man pages need `click-man` or manual authoring. Low priority.