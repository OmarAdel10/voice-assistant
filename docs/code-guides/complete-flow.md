# Complete Voice Assistant Flow

This document connects the files together. There are two doors into the assistant: the microphone door and the text-chat door. After the input becomes text, both doors use the same understanding and action path.

## The whole team

```mermaid
flowchart LR
    U[User] --> I[Microphone or text]
    I --> S[STT only for microphone]
    S --> P[process_text in main.py]
    I --> P
    P --> L[LLM if available]
    L -->|low confidence or unavailable| N[NLP regex fallback]
    L -->|confident| X[Intent result]
    N --> X
    X --> A[actions.py]
    A --> R[Response text]
    R --> T[tts_engine.py]
    T --> O[Speaker or printed fallback]
```

## Flow A: pressing the microphone button

The exact GUI command is a JSON line with `type: "voice_command"`. The CLI can enter the same path through `run_once_voice` or its continuous loop.

```mermaid
sequenceDiagram
    participant User
    participant GUI as Flutter or CLI
    participant Main as VoiceAssistant
    participant STT as GeminiSTTEngine
    participant Gemini as Gemini API
    participant Brain as LLM/NLP
    participant Hands as actions.py
    participant Mouth as TTSEngine

    User->>GUI: Press microphone button
    GUI->>Main: Start one voice command
    Main->>STT: record_audio(max_listen_seconds)
    STT->>STT: Capture 16 kHz mono float audio
    STT->>STT: Measure RMS and adjust gain
    STT-->>Main: Audio samples
    Main->>STT: transcribe(audio)
    STT->>STT: Encode samples as 16-bit WAV
    STT->>Gemini: Upload WAV and request transcription
    Gemini-->>STT: Text response
    STT-->>Main: Text and optional language
    Main->>Brain: process_text(text, language)
    Brain->>Brain: Try local LLM
    Brain->>Brain: Fallback to regex NLP if needed
    Brain->>Hands: Run selected action
    Hands-->>Brain: Action result
    Brain-->>Main: Response and language
    Main->>Mouth: say(response, language)
    Mouth->>Mouth: Piper, then pyttsx3, then print fallback
    Mouth-->>User: Hear or read answer
```

### Step-by-step in very simple words

1. The user taps the microphone button.
2. The GUI sends `voice_command` to `GuiSession`, or the CLI calls `run_once_voice`.
3. `main.py` asks the STT factory for the Gemini engine if it is not ready yet.
4. `stt_gemini.py` asks `sounddevice` to listen.
5. It collects one channel of sound at 16,000 samples per second.
6. It checks whether the sound is too quiet or too loud and may adjust gain.
7. It changes the sound numbers into a WAV file held in memory.
8. It uploads that WAV to Gemini and asks for a transcription.
9. Gemini returns words. The engine returns those words to `main.py`.
10. If there are no words, the GUI reports `no-speech` and the journey ends there.
11. Otherwise, the GUI may emit a `transcription` event.
12. `process_text` first asks the local Qwen LLM to understand the words.
13. If the LLM is unavailable, fails, or is unsure, `NLPEngine` checks `intents.json`.
14. The brain finds an intent such as time, date, system information, open app, or web search.
15. It pulls out extra details, such as the app name or search query.
16. `actions.py` performs the real action.
17. A response sentence is returned.
18. `tts_engine.py` tries the correct Arabic or English Piper voice.
19. If Piper cannot speak, it tries pyttsx3, then prints the answer.
20. The user hears the answer, or sees it as a final fallback.

## Flow B: text chat

Text chat skips the ears. It begins with a typed string from the CLI `--text` option or a GUI `text_command` JSON line.

```mermaid
sequenceDiagram
    participant User
    participant UI as CLI or GUI
    participant Main as VoiceAssistant
    participant Brain as LLM/NLP
    participant Hands as actions.py
    participant Mouth as TTSEngine

    User->>UI: Type a command
    UI->>Main: process_text(text, no STT language)
    Main->>Brain: Try local LLM
    Brain-->>Main: Intent, entities, language
    Main->>Brain: Regex fallback if needed
    Brain->>Hands: Execute action
    Hands-->>Main: Result
    Main->>Mouth: Speak response
    Mouth-->>User: Hear or read response
```

### Step-by-step in very simple words

1. The user types something like `what time is it`.
2. The CLI or GUI sends the text to `VoiceAssistant.process_text`.
3. There is no microphone and no Gemini STT call.
4. Because there is no STT language, the LLM or NLP engine detects the language from the text.
5. The local LLM tries first when enabled and loaded.
6. NLP uses regex examples from `config/intents.json` if the LLM cannot help.
7. The chosen action runs in `core/actions.py`.
8. The result becomes a response sentence.
9. The response is spoken by Piper or a fallback engine.
10. In GUI mode, a `response` event and then `tts_playing` event are also sent back.

## Which files touch one request?

```mermaid
flowchart TD
    A[config.yaml] --> B[config/settings.py]
    B --> C[voice_assistant/main.py]
    C --> D[core/stt_engine.py]
    D --> E[core/stt_gemini.py]
    C --> F[core/llm_engine.py]
    C --> G[core/nlp_engine.py]
    G --> H[config/intents.json]
    C --> I[core/actions.py]
    C --> J[core/tts_engine.py]
    K[core/paths.py] --> B
    K --> G
    L[core/exceptions.py] --> C
    L --> D
    L --> G
    L --> I
```

## A tiny example

For `open vscode`:

```text
words -> open_app intent -> app=vscode -> find executable -> launch VS Code -> speak success
```

For `search for Python tutorials`:

```text
words -> web_search intent -> query=Python tutorials -> open Google -> speak search status
```

For `what time is it`:

```text
words -> get_time intent -> read current clock -> speak the time
```

The important connection is this: voice and text are different beginnings, but they share the same middle and ending.
