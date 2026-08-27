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
# interactive enrollment
# ---------------------------------------------------------------------------
def _states(events):
    return events_of_type(events, "enroll_state")


def _cmd(obj):
    return json.dumps(obj)


def test_enroll_start_emits_first_phrase(monkeypatch, tmp_path):
    import voice_assistant.enrollment as enrollment_mod

    monkeypatch.setattr(enrollment_mod, "ENROLLMENT_DIR", tmp_path)
    monkeypatch.setattr(enrollment_mod, "ENROLLMENT_FILE", tmp_path / "enrollment.json")

    events, _ = run_session([_cmd({"type": "enroll_start"})])

    states = _states(events)
    assert len(states) == 1
    s = states[0]
    assert s["phase"] == "started"
    assert s["index"] == 0
    assert s["total"] == 6
    assert s["captured"] == 0
    assert len(s["phrases"]) == 6
    assert s["phrase"] == s["phrases"][0]
    assert s["phrases"][0] == enrollment_mod.ENROLLMENT_PHRASES[0]


def test_enroll_record_captures_and_advances(monkeypatch, tmp_path):
    import voice_assistant.enrollment as enrollment_mod

    monkeypatch.setattr(enrollment_mod, "ENROLLMENT_DIR", tmp_path)
    monkeypatch.setattr(enrollment_mod, "ENROLLMENT_FILE", tmp_path / "enrollment.json")

    events, _ = run_session(
        [
            _cmd({"type": "enroll_start"}),
            _cmd({"type": "enroll_record"}),
        ],
        enroll_engine=FakeEnrollSTT(),
    )

    transcriptions = events_of_type(events, "transcription")
    assert [t["text"] for t in transcriptions] == ["phrase 1"]

    s = _states(events)[-1]
    assert s["phase"] == "captured"
    assert s["index"] == 1
    assert s["captured"] == 1
    assert s["transcription"] == "phrase 1"

    from voice_assistant.enrollment import ENROLLMENT_PHRASES

    assert s["phrase"] == ENROLLMENT_PHRASES[1]


def test_enroll_no_speech_stays_on_phrase(monkeypatch, tmp_path):
    import voice_assistant.enrollment as enrollment_mod

    monkeypatch.setattr(enrollment_mod, "ENROLLMENT_DIR", tmp_path)
    monkeypatch.setattr(enrollment_mod, "ENROLLMENT_FILE", tmp_path / "enrollment.json")

    class SilentSTT(FakeSTT):
        def transcribe(self, audio, initial_prompt=None):
            return ("", "en")

    events, _ = run_session(
        [
            _cmd({"type": "enroll_start"}),
            _cmd({"type": "enroll_record"}),
            _cmd({"type": "enroll_record"}),
        ],
        enroll_engine=SilentSTT(),
    )

    assert not events_of_type(events, "error")
    states = _states(events)
    assert [s["phase"] for s in states[1:]] == ["no_speech", "no_speech"]
    assert all(s["index"] == 0 for s in states)
    assert all(s["phrase"] == states[0]["phrase"] for s in states[1:])


def test_enroll_cancel_resets_and_blocks_records(monkeypatch, tmp_path):
    import voice_assistant.enrollment as enrollment_mod

    monkeypatch.setattr(enrollment_mod, "ENROLLMENT_DIR", tmp_path)
    monkeypatch.setattr(enrollment_mod, "ENROLLMENT_FILE", tmp_path / "enrollment.json")

    events, _ = run_session(
        [
            _cmd({"type": "enroll_start"}),
            _cmd({"type": "enroll_record"}),
            _cmd({"type": "enroll_cancel"}),
            _cmd({"type": "enroll_record"}),
        ],
        enroll_engine=FakeEnrollSTT(),
    )

    states = _states(events)
    assert [s["phase"] for s in states] == ["started", "captured", "cancelled"]
    assert states[-1]["captured"] == 0

    errors = events_of_type(events, "error")
    assert len(errors) == 1
    assert "before enroll_start" in errors[0]["message"]
    assert not (tmp_path / "enrollment.json").exists()


def test_enroll_full_happy_path_saves_once(monkeypatch, tmp_path):
    import voice_assistant.enrollment as enrollment_mod

    monkeypatch.setattr(enrollment_mod, "ENROLLMENT_DIR", tmp_path)
    monkeypatch.setattr(enrollment_mod, "ENROLLMENT_FILE", tmp_path / "enrollment.json")

    commands = [_cmd({"type": "enroll_start"})]
    commands += [_cmd({"type": "enroll_record"}) for _ in range(6)]
    events, _ = run_session(commands, enroll_engine=FakeEnrollSTT())

    transcriptions = events_of_type(events, "transcription")
    assert [t["text"] for t in transcriptions] == [f"phrase {i}" for i in range(1, 7)]

    states = _states(events)
    assert [s["phase"] for s in states] == (
        ["started"] + ["captured"] * 5 + ["complete"]
    )
    final = states[-1]
    assert final["captured"] == 6
    assert final["phrase"] is None

    responses = events_of_type(events, "response")
    assert len(responses) == 1
    assert "Enrollment saved" in responses[0]["text"]

    saved_file = tmp_path / "enrollment.json"
    assert saved_file.exists()
    saved = json.loads(saved_file.read_text(encoding="utf-8"))
    assert saved["initial_prompt"].startswith("phrase 1")
    assert saved["initial_prompt"].endswith("phrase 6")


def test_enroll_record_before_start_is_error():
    events, _ = run_session([_cmd({"type": "enroll_record"})])
    errors = events_of_type(events, "error")
    assert len(errors) == 1
    assert "before enroll_start" in errors[0]["message"]
    assert not _states(events)


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
