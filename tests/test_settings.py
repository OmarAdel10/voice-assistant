"""Tests for config/settings.py."""

import tempfile
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from config.settings import Settings
from core.exceptions import VoiceAssistantError


class TestSettings:
    """Test Settings loading and validation."""

    def test_load_defaults_when_no_config_file(self):
        """Settings should load with sane defaults when config.yaml doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            # Don't create the file
            settings = Settings.load(config_path)

            # Check defaults from DESIGN.md
            assert settings.stt.model_size == "large-v3"
            assert settings.stt.language == "ar"
            assert settings.stt.device == "cuda"
            assert settings.stt.compute_type == "float16"
            assert settings.stt.model_dir == "models/stt"
            assert settings.stt.vad_threshold == 0.5
            assert settings.stt.max_listen_seconds == 5
            assert settings.stt.vad_filter is True
            assert settings.stt.vad_min_silence_ms == 500
            assert settings.tts.engine == "piper"
            assert settings.tts.rate == 180
            assert settings.tts.volume == 0.9
            assert settings.tts.voice_id is None
            assert settings.tts.piper_voice_dir == "models/tts"
            assert settings.tts.piper_voice_ar == "ar_EG-medium"
            assert settings.tts.piper_voice_en == "en_US-medium"
            assert settings.nlp.confidence_threshold == 0.6
            assert settings.audio.sample_rate == 16000
            assert settings.audio.channels == 1
            assert settings.log.level == "INFO"

    def test_load_from_yaml_file(self):
        """Settings should load values from config.yaml when present."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config_data = {
                "stt": {
                    "model_size": "base.en",
                    "language": "ar",
                    "vad_threshold": 0.6,
                    "max_listen_seconds": 5,
                },
                "tts": {
                    "engine": "pyttsx3",
                    "rate": 200,
                    "volume": 0.8,
                },
                "nlp": {
                    "confidence_threshold": 0.7,
                },
                "audio": {
                    "sample_rate": 44100,
                    "channels": 2,
                },
                "log": {
                    "level": "DEBUG",
                },
            }
            config_path.write_text(yaml.dump(config_data))

            settings = Settings.load(config_path)

            assert settings.stt.model_size == "base.en"
            assert settings.stt.language == "ar"
            assert settings.stt.vad_threshold == 0.6
            assert settings.stt.max_listen_seconds == 5
            assert settings.tts.engine == "pyttsx3"
            assert settings.tts.rate == 200
            assert settings.tts.volume == 0.8
            assert settings.nlp.confidence_threshold == 0.7
            assert settings.audio.sample_rate == 44100
            assert settings.audio.channels == 2
            assert settings.log.level == "DEBUG"

    def test_load_partial_yaml_merges_with_defaults(self):
        """Partial config.yaml should merge with defaults for missing values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config_data = {
                "stt": {
                    "model_size": "small.en",
                },
                "tts": {
                    "engine": "pyttsx3",
                },
            }
            config_path.write_text(yaml.dump(config_data))

            settings = Settings.load(config_path)

            # Overridden values
            assert settings.stt.model_size == "small.en"
            assert settings.tts.engine == "pyttsx3"
            # Default values preserved
            assert settings.stt.language == "ar"
            assert settings.stt.vad_threshold == 0.5
            assert settings.tts.rate == 180
            assert settings.tts.volume == 0.9
            assert settings.nlp.confidence_threshold == 0.6
            assert settings.audio.sample_rate == 16000

    def test_invalid_yaml_raises_voice_assistant_error(self):
        """Invalid YAML should raise VoiceAssistantError, not bare Exception."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config_path.write_text("invalid: yaml: [")

            with pytest.raises(VoiceAssistantError):
                Settings.load(config_path)

    def test_missing_file_returns_defaults(self):
        """Non-existent config file should return defaults, not raise."""
        settings = Settings.load(Path("/nonexistent/path/config.yaml"))
        assert settings.stt.model_size == "large-v3"

    def test_settings_are_immutable(self):
        """Settings should be frozen/immutable after creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            settings = Settings.load(config_path)

            with pytest.raises(ValidationError):
                settings.stt.model_size = "different"


class TestIntentsRegistry:
    """Test intents.json loading."""

    def test_intents_file_exists_and_valid(self):
        """intents.json should exist and contain exactly 5 intents."""
        import json

        intents_path = Path("config/intents.json")
        assert intents_path.exists(), "config/intents.json must exist"

        data = json.loads(intents_path.read_text())
        assert "intents" in data
        assert len(data["intents"]) == 5

        intent_names = {intent["name"] for intent in data["intents"]}
        expected = {"get_time", "get_date", "get_sys_info", "open_app", "web_search"}
        assert intent_names == expected

    def test_each_intent_has_required_fields(self):
        """Each intent must have name, patterns, entities, response_template."""
        import json

        intents_path = Path("config/intents.json")
        data = json.loads(intents_path.read_text())

        for intent in data["intents"]:
            assert "name" in intent
            assert "patterns" in intent
            assert "entities" in intent
            assert "response_template" in intent
            assert isinstance(intent["patterns"], list)
            assert len(intent["patterns"]) > 0
            assert isinstance(intent["entities"], list)
            assert isinstance(intent["response_template"], str)
