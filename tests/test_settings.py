"""Tests for config/settings.py."""

import os
import tempfile
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from config.settings import Settings, _load_dotenv
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
            assert settings.stt.language is None  # auto-detect
            assert settings.stt.max_listen_seconds == 5
            assert settings.tts.engine == "piper"
            assert settings.tts.rate == 180
            assert settings.tts.volume == 0.9
            assert settings.tts.voice_id is None
            assert settings.tts.piper_voice_dir.endswith("models/tts")
            assert Path(settings.tts.piper_voice_dir).is_absolute()
            assert settings.tts.piper_voice_ar == "ar_JO-kareem-medium"
            assert settings.tts.piper_voice_en == "en_US-lessac-medium"
            assert settings.tts.piper_voice_ar_fallback == "ar_JO-kareem-low"
            assert settings.tts.piper_voice_en_fallback == "en_US-lessac-low"
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
                    "language": "ar",
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

            assert settings.stt.language == "ar"
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
                    "language": "ar",
                },
                "tts": {
                    "engine": "pyttsx3",
                },
            }
            config_path.write_text(yaml.dump(config_data))

            settings = Settings.load(config_path)

            # Overridden values
            assert settings.stt.language == "ar"
            assert settings.tts.engine == "pyttsx3"
            # Default values preserved
            assert settings.stt.allowed_languages == ["ar", "en"]
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
        assert settings.stt.language is None

    def test_settings_are_immutable(self):
        """Settings should be frozen/immutable after creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            settings = Settings.load(config_path)

            with pytest.raises(ValidationError):
                settings.stt.language = "different"

    def test_relative_yaml_paths_anchor_to_project_root(self, tmp_path):
        """Relative model paths in config.yaml must resolve to PROJECT_ROOT."""
        cfg = tmp_path / "config.yaml"
        cfg.write_text(
            "stt:\n"
            "  language: ar\n"
            "tts:\n"
            "  piper_voice_dir: models/tts\n"
            "llm:\n"
            "  model_path: models/llm/model.gguf\n"
        )
        settings = Settings.load(cfg)

        assert settings.stt.language == "ar"
        assert Path(settings.tts.piper_voice_dir).is_absolute()
        assert settings.tts.piper_voice_dir.endswith("models/tts")
        assert Path(settings.llm.model_path).is_absolute()
        assert settings.llm.model_path.endswith("models/llm/model.gguf")

    def test_absolute_yaml_paths_left_untouched(self, tmp_path):
        """Absolute paths from config.yaml pass through unchanged."""
        cfg = tmp_path / "config.yaml"
        cfg.write_text("stt:\n  language: ar\n")
        settings = Settings.load(cfg)
        assert settings.stt.language == "ar"


class TestDotenvLoading:
    """Test .env auto-loading so secrets reach the process regardless of launcher.

    The Flutter GUI spawns the CLI as a subprocess and does not inherit a
    developer's shell exports, so GEMINI_API_KEY must be readable from a
    project-root .env file for every launch path (shell, python -m, GUI).
    """

    def test_load_dotenv_sets_environment_variable(self, tmp_path, monkeypatch):
        """A simple KEY=VALUE line in .env should populate os.environ."""
        monkeypatch.delenv("MY_TEST_KEY", raising=False)
        dotenv_path = tmp_path / ".env"
        dotenv_path.write_text("MY_TEST_KEY=super-secret\n")

        _load_dotenv(dotenv_path)

        assert os.environ["MY_TEST_KEY"] == "super-secret"

    def test_load_dotenv_skips_comments_and_blank_lines(self, tmp_path, monkeypatch):
        """Comments and blank lines must not raise or set spurious vars."""
        monkeypatch.delenv("REAL_KEY", raising=False)
        dotenv_path = tmp_path / ".env"
        dotenv_path.write_text("# a comment\n\nREAL_KEY=value\n   \n# another\n")

        _load_dotenv(dotenv_path)

        assert os.environ["REAL_KEY"] == "value"

    def test_load_dotenv_strips_quotes(self, tmp_path, monkeypatch):
        """Quoted values should have surrounding quotes stripped."""
        monkeypatch.delenv("QUOTED_KEY", raising=False)
        dotenv_path = tmp_path / ".env"
        dotenv_path.write_text('QUOTED_KEY="quoted-value"\n')

        _load_dotenv(dotenv_path)

        assert os.environ["QUOTED_KEY"] == "quoted-value"

    def test_load_dotenv_does_not_override_existing_env(self, tmp_path, monkeypatch):
        """A real environment variable must win over the .env file value."""
        monkeypatch.setenv("PRIORITY_KEY", "from-shell")
        dotenv_path = tmp_path / ".env"
        dotenv_path.write_text("PRIORITY_KEY=from-dotenv\n")

        _load_dotenv(dotenv_path)

        assert os.environ["PRIORITY_KEY"] == "from-shell"

    def test_load_dotenv_missing_file_is_noop(self, tmp_path):
        """A missing .env file must not raise."""
        _load_dotenv(tmp_path / "does-not-exist.env")  # should not raise

    def test_settings_load_reads_dotenv_from_project_root(self, tmp_path, monkeypatch):
        """Settings.load() should read GEMINI_API_KEY from the project-root .env."""
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.setattr("config.settings.PROJECT_ROOT", tmp_path)
        (tmp_path / ".env").write_text("GEMINI_API_KEY=from-project-env\n")

        Settings.load(tmp_path / "config.yaml")

        assert os.environ["GEMINI_API_KEY"] == "from-project-env"

    def test_settings_load_without_dotenv_file_does_not_raise(self, tmp_path, monkeypatch):
        """Settings.load() must work normally when no .env file is present."""
        monkeypatch.setattr("config.settings.PROJECT_ROOT", tmp_path)
        settings = Settings.load(tmp_path / "config.yaml")
        assert settings.stt.language is None


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
            assert "patterns_en" in intent
            assert "patterns_ar" in intent
            assert "entities" in intent
            assert "response_template_en" in intent
            assert "response_template_ar" in intent
            assert isinstance(intent["patterns_en"], list)
            assert len(intent["patterns_en"]) > 0
            assert isinstance(intent["patterns_ar"], list)
            assert len(intent["patterns_ar"]) > 0
            assert isinstance(intent["entities"], list)
            assert isinstance(intent["response_template_en"], str)
            assert isinstance(intent["response_template_ar"], str)
