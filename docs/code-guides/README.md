# Voice Assistant: Friendly Code Guides

These guides explain the project in very simple language. Imagine the program is a small team:

- **Ears:** `core/stt_gemini.py` listens and writes down words.
- **Brain:** `core/llm_engine.py` and `core/nlp_engine.py` understand the words.
- **Hands:** `core/actions.py` does things on the computer.
- **Mouth:** `core/tts_engine.py` speaks back.
- **Teacher:** `config/settings.py` tells everyone how to work.
- **Director:** `voice_assistant/main.py` connects the team.

## Guides

### Core

- [core/__init__.py](core__init__.md)
- [core/exceptions.py](core-exceptions.md)
- [core/paths.py](core-paths.md)
- [core/stt_engine.py](core-stt-engine.md)
- [core/stt_gemini.py](core-stt-gemini.md)
- [core/nlp_engine.py](core-nlp-engine.md)
- [core/llm_engine.py](core-llm-engine.md)
- [core/actions.py](core-actions.md)
- [core/tts_engine.py](core-tts-engine.md)

### Configuration and setup

- [config/__init__.py](config__init__.md)
- [config/settings.py](config-settings.md)
- [config/intents.json](config-intents.md)
- [config.yaml](config-yaml.md)
- [scripts/download_models.py](scripts-download-models.md)

### Application layer

- [voice_assistant/__init__.py](voice-assistant__init__.md)
- [voice_assistant/main.py](voice-assistant-main.md)
- [voice_assistant/gui_mode.py](voice-assistant-gui-mode.md)
- [voice_assistant/enrollment.py](voice-assistant-enrollment.md)

## Complete flows

- [Microphone and chat flow](complete-flow.md)

## How to read

Read `main.py` first if you want the big picture. Then read the ear, brain, hands, and mouth guides. The final flow guide connects every piece from the first button press to the final answer.
