"""Tests for the GUI JSON-lines protocol session."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from config.settings import LLMConfig, Settings
from voice_assistant.gui_mode import GuiSession
from voice_assistant.main import VoiceAssistant

REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------
class FakeSTT:
    """Deterministic stand-in for STTEngine."""

    def __init__(self) -> None:
        self.input_gain: float = 1.0
        self.language: str | None = None
        self.auto_gain: bool = False
        self.record_calls = 0
        self.transcribe_prompts: list = []

    def record_audio(self, duration: float):
        self.record_calls += 1
        return b"fake-audio"

    def transcribe(self, audio, initial_prompt=None):
        self.transcribe_prompts.append(initial_prompt)
        if self.record_calls % 2 == 0:
            return ("", "en")  # simulate silence every other call
        return ("what time is it", "en")

    def test_microphone(self, duration: float, gain: float | None):
        return {
            "duration": duration,
            "rms": 0.05,
            "peak": 0.5,
            "clipped_samples": 0,
            "total_samples": int(duration * 16000),
            "clipping_percentage": 0.0,
            "detected_language": "en",
            "confidence": 0.9,
            "transcription": "testing one two three",
            "assessment": "good",
            "suggested_gain": 1.0,
        }

    def set_input_gain(self, gain: float) -> None:
        self.input_gain = gain

    def set_language(self, language: str | None) -> None:
        self.language = language

    def set_auto_gain(self, enabled: bool) -> None:
        self.auto_gain = enabled


class FakeTTS:
    def __init__(self) -> None:
        self.spoken: list[tuple[str, str | None]] = []

    def say(self, text: str, lang: str | None = None) -> None:
        self.spoken.append((text, lang))


class FakeEnrollSTT(FakeSTT):
    """Always captures speech, echoing a deterministic phrase per call."""

    def __init__(self) -> None:
        super().__init__()
        self.call_index = 0

    def record_audio(self, duration: float):
        return b"fake-audio"

    def transcribe(self, audio, initial_prompt=None):
        self.transcribe_prompts.append(initial_prompt)
        self.call_index += 1
        return (f"phrase {self.call_index}", "en")


def make_assistant() -> VoiceAssistant:
    """Real assistant core with fake audio engines and LLM disabled."""
    settings = Settings(llm=LLMConfig(enabled=False))
    assistant = VoiceAssistant(settings)
    assistant._stt_engine = FakeSTT()
    assistant._tts_engine = FakeTTS()
    return assistant


def run_session(
    commands: list[str], enroll_engine: FakeSTT | None = None
) -> tuple[list[dict], VoiceAssistant]:
    """Feed commands through a GuiSession and parse emitted events."""
    import io

    assistant = make_assistant()
    stdin_lines = "\n".join(commands) + "\n"
    out = io.StringIO()
    session = GuiSession(assistant)
    if enroll_engine is not None:
        session._enroll_engine = enroll_engine
    session.run(stdin=io.StringIO(stdin_lines), stdout=out)

    events = [json.loads(line) for line in out.getvalue().splitlines()]
    return events, assistant


def events_of_type(events: list[dict], type_: str) -> list[dict]:
    return [e for e in events if e.get("type") == type_]


# ---------------------------------------------------------------------------
# Protocol basics
# ---------------------------------------------------------------------------
def test_session_starts_and_stops_with_status_events():
    events, _ = run_session([])
    statuses = [e["status"] for e in events_of_type(events, "status")]
    assert statuses[0] == "ready"
    assert statuses[-1] == "stopped"


def test_malformed_json_yields_error_and_session_continues():
    events, _ = run_session(["not json at all", '{"type": "settings"}'])
    errors = events_of_type(events, "error")
    assert len(errors) == 1
    assert "Malformed JSON" in errors[0]["message"]
    # Session survived and handled the next command
    assert any("settings-applied" in e["status"] for e in events_of_type(events, "status"))


def test_non_object_json_yields_error():
    events, _ = run_session(['["array"]'])
    assert "JSON object" in events_of_type(events, "error")[0]["message"]


def test_unknown_command_type_yields_error():
    events, _ = run_session([json.dumps({"type": "bogus"})])
    assert "Unknown command type" in events_of_type(events, "error")[0]["message"]


# ---------------------------------------------------------------------------
# text_command
# ---------------------------------------------------------------------------
def test_text_command_emits_response_then_tts_playing():
    events, assistant = run_session(
        [json.dumps({"type": "text_command", "text": "what time is it"})]
    )
    responses = events_of_type(events, "response")
    tts = events_of_type(events, "tts_playing")
    assert len(responses) == 1
    assert isinstance(responses[0]["text"], str)
    assert len(tts) == 1
    assert tts[0]["text"] == responses[0]["text"]
    assert assistant._tts_engine.spoken  # TTS actually invoked


def test_text_command_without_text_yields_error():
    events, _ = run_session([json.dumps({"type": "text_command"})])
    assert events_of_type(events, "error")


# ---------------------------------------------------------------------------
# settings
# ---------------------------------------------------------------------------
def test_settings_applies_gain_autogain_and_language_live():
    cmd = {
        "type": "settings",
        "input_gain": 1.75,
        "auto_gain": True,
        "language": "ar",
    }
    _, assistant = run_session([json.dumps(cmd)])
    stt = assistant._stt_engine
    assert stt.input_gain == pytest.approx(1.75)
    assert stt.auto_gain is True
    assert stt.language == "ar"


def test_settings_system_language_maps_to_auto_detect():
    _, assistant = run_session([json.dumps({"type": "settings", "language": "system"})])
    assert assistant._stt_engine.language is None


# ---------------------------------------------------------------------------
# test_mic
# ---------------------------------------------------------------------------
def test_test_mic_emits_result_with_dart_contract_keys():
    events, _ = run_session([json.dumps({"type": "test_mic", "duration": 2.0})])
    results = events_of_type(events, "mic_test_result")
    assert len(results) == 1
    expected_keys = {
        "duration",
        "rms",
        "peak",
        "clipped_samples",
        "total_samples",
        "clipping_percentage",
        "detected_language",
        "transcription",
        "assessment",
        "suggested_gain",
    }
    assert expected_keys <= set(results[0].keys())


# ---------------------------------------------------------------------------
# voice_command
# ---------------------------------------------------------------------------
def test_voice_command_full_cycle_event_order():
    events, _ = run_session([json.dumps({"type": "voice_command", "mode": "once"})])

    # ready ... listening(status) ... transcription ... response ... tts_playing ... stopped
    def idx_of(pred):
        return next(i for i, e in enumerate(events) if pred(e))

    listening_idx = idx_of(lambda e: e["type"] == "status" and e.get("status") == "listening")
    assert listening_idx < idx_of(lambda e: e["type"] == "transcription")
    assert idx_of(lambda e: e["type"] == "transcription") < idx_of(
        lambda e: e["type"] == "response"
    )
    assert idx_of(lambda e: e["type"] == "response") < idx_of(
        lambda e: e["type"] == "tts_playing"
    )


def test_voice_command_silence_reports_no_speech():
    events, _ = run_session(
        [
            json.dumps({"type": "voice_command"}),  # record_calls=1 -> speech
            json.dumps({"type": "voice_command"}),  # record_calls=2 -> silence
        ]
    )
    assert any(e["status"] == "no-speech" for e in events_of_type(events, "status"))


# ---------------------------------------------------------------------------
# enroll
# ---------------------------------------------------------------------------
def test_enroll_full_flow_records_all_phrases(monkeypatch, tmp_path):
    import voice_assistant.enrollment as enrollment_mod

    monkeypatch.setattr(enrollment_mod, "ENROLLMENT_DIR", tmp_path)
    monkeypatch.setattr(enrollment_mod, "ENROLLMENT_FILE", tmp_path / "enrollment.json")

    enroll_fake = FakeEnrollSTT()
    events, _ = run_session([json.dumps({"type": "enroll"})], enroll_engine=enroll_fake)

    # Every phrase captured, each reported as a transcription event
    transcriptions = events_of_type(events, "transcription")
    assert [t["text"] for t in transcriptions] == [f"phrase {i}" for i in range(1, 7)]

    # Enrollment captures must not be biased by an enrollment prompt
    assert all(p is None for p in enroll_fake.transcribe_prompts)

    statuses = [e["status"] for e in events_of_type(events, "status")]
    assert "enroll-start" in statuses
    assert "enroll-complete" in statuses
    assert any("recording" in s for s in statuses)

    # Confirmation response closes the flow
    responses = events_of_type(events, "response")
    assert len(responses) == 1
    assert "Enrollment saved" in responses[0]["text"]

    # Combined prompt persisted to disk
    saved_file = tmp_path / "enrollment.json"
    assert saved_file.exists()
    saved = json.loads(saved_file.read_text(encoding="utf-8"))
    assert saved["initial_prompt"].startswith("phrase 1")
    assert saved["initial_prompt"].endswith("phrase 6")


def test_enroll_with_no_speech_yields_error(monkeypatch, tmp_path):
    import voice_assistant.enrollment as enrollment_mod

    monkeypatch.setattr(enrollment_mod, "ENROLLMENT_DIR", tmp_path)
    monkeypatch.setattr(enrollment_mod, "ENROLLMENT_FILE", tmp_path / "enrollment.json")

    class SilentSTT(FakeSTT):
        def transcribe(self, audio, initial_prompt=None):
            return ("", "en")

    events, _ = run_session([json.dumps({"type": "enroll"})], enroll_engine=SilentSTT())
    errors = events_of_type(events, "error")
    assert len(errors) == 1
    assert "no speech" in errors[0]["message"]
    assert not (tmp_path / "enrollment.json").exists()


# ---------------------------------------------------------------------------
# CWD independence (anchored paths regression)
# ---------------------------------------------------------------------------
def test_cli_works_from_foreign_cwd(tmp_path):
    venv_cli = REPO_ROOT / ".venv" / "bin" / "voice-assistant"
    if not venv_cli.exists():  # pragma: no cover - environment guard
        pytest.skip("venv console script not present")

    result = subprocess.run(
        [str(venv_cli), "--list-intents"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    assert "Available intents" in result.stdout
