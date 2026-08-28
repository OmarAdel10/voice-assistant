# `core/stt_gemini.py`

## What is this file?

This is the assistant's ears. It records sound from the microphone and asks Gemini to turn that sound into words.

## Startup

`GeminiSTTEngine` reads the API key, model name, timeout, language hints, gain, and audio settings. It prefers IPv4 when needed, then creates a Google GenAI client.

## Recording

`record_audio`:

1. Imports `sounddevice`.
2. Calculates frames for the requested seconds at 16,000 samples per second.
3. Records one mono audio channel.
4. Waits until recording is finished.
5. Measures loudness using RMS.
6. Adjusts gain when automatic gain is enabled.
7. Returns a NumPy float32 array.

## Transcribing

`transcribe`:

1. Returns an empty answer for empty audio.
2. Converts float audio into a 16-bit mono WAV file in memory.
3. Uploads the WAV to Gemini Files API.
4. Sends the audio to the Interactions API with language hints.
5. Retries rate-limit errors up to three attempts.
6. Reads `output_text` and returns text plus language.
7. Converts failures into `STTError`.

## Other helpers

- `_encode_wav`: packages microphone numbers as a WAV file.
- `_map_language_code` and `_get_language_codes`: prepare BCP-47 language hints.
- `test_microphone`: reports RMS, peak, clipping, transcription, and suggested gain.
- `set_input_gain`, `set_auto_gain`, `set_language`: change live settings.
- `close` and context-manager methods: provide cleanup shape.

## Picture

```mermaid
flowchart LR
    M[Microphone] --> R[record_audio]
    R --> G[Gain and RMS check]
    G --> W[_encode_wav]
    W --> U[Upload WAV]
    U --> X[Gemini Transcribe]
    X --> T[Text and language]
```

The file does not decide what the sentence means. It only listens and writes it down.
