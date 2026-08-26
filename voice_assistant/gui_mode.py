"""Voice Assistant — GUI protocol session.

Bridges the Flutter GUI and the assistant core over stdin/stdout using
one JSON object per line.

Protocol:
  stdin commands : voice_command, text_command, enroll_start, enroll_record,
                   enroll_cancel, test_mic, settings
  stdout events  : transcription, response, error, mic_test_result,
                   tts_playing, status, enroll_state

stdout is reserved exclusively for protocol frames — all logging goes to
stderr so the GUI can parse stdout safely.
"""

from __future__ import annotations

import json
import logging
import signal
import sys
from typing import Any, TextIO

from config.settings import Settings
from core.stt_engine import STTEngine
from voice_assistant.enrollment import ENROLLMENT_PHRASES, save_enrollment

logger = logging.getLogger(__name__)


class GuiSession:
    """JSON-lines session driving the assistant from the Flutter GUI."""

    def __init__(self, assistant: Any, enroll_engine: STTEngine | None = None) -> None:
        """Initialize the session.

        Args:
            assistant: VoiceAssistant instance providing process_text(),
                settings, and lazily-initialized engines.
            enroll_engine: Optional pre-built STT engine dedicated to
                enrollment runs (primarily for testing).
        """
        self._assistant = assistant
        self._running = False
        self._in: TextIO | None = None
        self._out: TextIO | None = None
        self._enroll_engine = enroll_engine
        self._enroll_active = False
        self._enroll_index = 0
        self._enroll_captured: list[str] = []

    # ------------------------------------------------------------------
    # Output helpers
    # ------------------------------------------------------------------
    def _emit(self, payload: dict[str, Any]) -> None:
        assert self._out is not None
        self._out.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self._out.flush()

    def _emit_error(self, message: str) -> None:
        self._emit({"type": "error", "message": message})

    def _emit_status(self, status: str) -> None:
        self._emit({"type": "status", "status": status})

    def _emit_transcription(self, text: str, lang: str | None) -> None:
        payload: dict[str, Any] = {"type": "transcription", "text": text, "lang": lang}
        self._emit(payload)

    def _emit_response(self, text: str, lang: str | None) -> None:
        self._emit({"type": "response", "text": text, "lang": lang})

    def _safe_say(self, text: str, lang: str | None) -> bool:
        """Speak text, returning whether TTS succeeded."""
        try:
            self._assistant.tts_engine.say(text, lang=lang)
            return True
        except Exception as e:
            logger.error(f"TTS failed: {e}")
            return False

    # ------------------------------------------------------------------
    # Enrollment engine (shared instance; enrollment passes no prompt)
    # ------------------------------------------------------------------
    @property
    def enroll_engine(self) -> STTEngine:
        """STT engine used for enrollment runs.

        Reuses the assistant's engine so the whisper weights stay loaded
        once; enrollment calls transcribe() with an explicit neutral prompt.
        Tests may still inject a dedicated engine.
        """
        if self._enroll_engine is None:
            return self._assistant.stt_engine
        return self._enroll_engine

    # ------------------------------------------------------------------
    # Command handlers
    # ------------------------------------------------------------------
    def _handle_voice_command(self, command: dict[str, Any]) -> None:
        """One listen -> transcribe -> respond -> speak cycle."""
        stt = self._assistant.stt_engine
        self._emit_status("listening")
        audio = stt.record_audio(self._assistant.settings.stt.max_listen_seconds)
        text, lang = stt.transcribe(audio)

        if not text:
            self._emit_status("no-speech")
            return

        self._emit_transcription(text, lang)

        response_text, response_lang = self._assistant.process_text(text, lang=lang)
        self._emit_response(response_text, response_lang)

        if self._safe_say(response_text, response_lang):
            self._emit({"type": "tts_playing", "text": response_text})

    def _handle_text_command(self, command: dict[str, Any]) -> None:
        """Process a text query without STT."""
        text = command.get("text")
        if not text or not isinstance(text, str):
            self._emit_error("text_command requires a non-empty 'text' field")
            return

        response_text, response_lang = self._assistant.process_text(text, lang=None)
        self._emit_response(response_text, response_lang)

        if self._safe_say(response_text, response_lang):
            self._emit({"type": "tts_playing", "text": response_text})

    def _handle_test_mic(self, command: dict[str, Any]) -> None:
        """Record a sample and report audio-quality metrics."""
        duration = float(command.get("duration", 3.0))
        gain = command.get("gain")
        gain_f = float(gain) if gain is not None else None

        result = self._assistant.stt_engine.test_microphone(duration=duration, gain=gain_f)

        if "error" in result:
            self._emit_error(str(result["error"]))
            return

        payload: dict[str, Any] = {"type": "mic_test_result"}
        payload.update(result)
        self._emit(payload)

    def _handle_settings(self, command: dict[str, Any]) -> None:
        """Apply runtime settings to the live STT engine."""
        stt = self._assistant.stt_engine
        applied: list[str] = []

        input_gain = command.get("input_gain")
        if input_gain is not None:
            stt.set_input_gain(float(input_gain))
            applied.append(f"input_gain={float(input_gain):.2f}")

        auto_gain = command.get("auto_gain")
        if auto_gain is not None:
            stt.set_auto_gain(bool(auto_gain))
            applied.append(f"auto_gain={'on' if auto_gain else 'off'}")

        language = command.get("language")
        if language is not None:
            stt.set_language(None if language == "system" else language)
            applied.append(f"language={language}")

        self._emit_status("settings-applied" + (" (" + ", ".join(applied) + ")" if applied else ""))

    def _handle_enroll_start(self, command: dict[str, Any]) -> None:
        """Begin an interactive enrollment session."""
        del command
        self._enroll_active = True
        self._enroll_index = 0
        self._enroll_captured = []
        total = len(ENROLLMENT_PHRASES)
        self._emit({
            "type": "enroll_state",
            "phase": "started",
            "index": 0,
            "total": total,
            "phrase": ENROLLMENT_PHRASES[0],
            "transcription": None,
            "captured": 0,
            "phrases": list(ENROLLMENT_PHRASES),
        })

    def _handle_enroll_record(self, command: dict[str, Any]) -> None:
        """Capture the current phrase on user request."""
        del command
        if not self._enroll_active:
            self._emit_error("enroll_record before enroll_start")
            return

        engine = self.enroll_engine
        max_seconds = self._assistant.settings.stt.max_listen_seconds
        phrase = ENROLLMENT_PHRASES[self._enroll_index]

        try:
            audio = engine.record_audio(max_seconds)
            text, lang = engine.transcribe(audio, initial_prompt=None)
        except Exception as e:
            logger.error(f"Enrollment capture failed: {e}")
            self._emit_error(f"Enrollment capture failed: {e}")
            return

        if not text:
            self._emit({
                "type": "enroll_state",
                "phase": "no_speech",
                "index": self._enroll_index,
                "total": len(ENROLLMENT_PHRASES),
                "phrase": phrase,
                "transcription": None,
                "captured": len(self._enroll_captured),
            })
            return

        self._enroll_captured.append(text)
        self._emit_transcription(text, lang)

        done = len(self._enroll_captured) == len(ENROLLMENT_PHRASES)
        if not done:
            self._enroll_index += 1

        next_phrase = (
            ENROLLMENT_PHRASES[self._enroll_index] if not done else None
        )
        phase = "complete" if done else "captured"
        self._emit({
            "type": "enroll_state",
            "phase": phase,
            "index": self._enroll_index,
            "total": len(ENROLLMENT_PHRASES),
            "phrase": next_phrase,
            "transcription": text,
            "captured": len(self._enroll_captured),
        })

        if not done:
            return

        prompt = " ".join(self._enroll_captured)
        try:
            save_enrollment(prompt)
        except Exception as e:
            self._emit_error(f"Failed to save enrollment: {e}")
            self._enroll_active = False
            return

        self._enroll_active = False
        self._emit_response(
            f"Enrollment saved ({len(self._enroll_captured)}/{len(ENROLLMENT_PHRASES)} phrases). Restart to apply.",
            "en",
        )

    def _handle_enroll_cancel(self, command: dict[str, Any]) -> None:
        """Abort the interactive session, discarding captures."""
        del command
        self._enroll_active = False
        self._enroll_index = 0
        self._enroll_captured = []
        self._emit({
            "type": "enroll_state",
            "phase": "cancelled",
            "index": 0,
            "total": len(ENROLLMENT_PHRASES),
            "phrase": None,
            "transcription": None,
            "captured": 0,
        })

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------
    def _dispatch(self, command: dict[str, Any]) -> None:
        handlers = {
            "voice_command": self._handle_voice_command,
            "text_command": self._handle_text_command,
            "test_mic": self._handle_test_mic,
            "settings": self._handle_settings,
            "enroll_start": self._handle_enroll_start,
            "enroll_record": self._handle_enroll_record,
            "enroll_cancel": self._handle_enroll_cancel,
        }
        cmd_type = command.get("type")
        handler = handlers.get(cmd_type)
        if handler is None:
            self._emit_error(f"Unknown command type: {cmd_type}")
            return
        handler(command)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def run(self, stdin: TextIO | None = None, stdout: TextIO | None = None) -> None:
        """Serve commands until stdin EOF or a termination signal."""
        self._in = stdin if stdin is not None else sys.stdin
        self._out = stdout if stdout is not None else sys.stdout
        self._running = True

        def _stop(signum: int, frame: Any) -> None:  # noqa: ARG001
            logger.info(f"GUI session received signal {signum}, stopping")
            self._running = False

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, _stop)
            except ValueError:
                # Not in main thread (e.g. under some test runners)
                pass

        self._emit_status("ready")
        logger.info("GUI session started")

        while self._running:
            try:
                line = self._in.readline()
            except KeyboardInterrupt:
                break
            if not line:
                break  # EOF: GUI closed the pipe
            line = line.strip()
            if not line:
                continue

            try:
                command = json.loads(line)
            except json.JSONDecodeError as e:
                self._emit_error(f"Malformed JSON: {e}")
                continue
            if not isinstance(command, dict):
                self._emit_error("Command must be a JSON object")
                continue

            try:
                self._dispatch(command)
            except Exception as e:
                logger.exception(f"Command failed: {command.get('type')}")
                self._emit_error(str(e))

        self._emit_status("stopped")
        logger.info("GUI session ended")
