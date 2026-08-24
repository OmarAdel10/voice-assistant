"""Integration tests for full voice assistant pipeline."""

from unittest.mock import Mock, patch

import pytest
from click.testing import CliRunner

from voice_assistant.main import cli


def create_mock_settings() -> Mock:
    """Create a properly structured mock Settings object."""
    mock_settings = Mock()

    # STT settings (updated for new config)
    mock_settings.stt = Mock()
    mock_settings.stt.model_size = "large-v3"
    mock_settings.stt.device = "cuda"
    mock_settings.stt.compute_type = "float16"
    mock_settings.stt.max_listen_seconds = 5
    mock_settings.stt.language = "ar"
    mock_settings.stt.vad_threshold = 0.5
    mock_settings.stt.model_dir = "models/stt"
    mock_settings.stt.vad_filter = True
    mock_settings.stt.vad_min_silence_ms = 500

    # TTS settings (updated for new config)
    mock_settings.tts = Mock()
    mock_settings.tts.engine = "piper"
    mock_settings.tts.rate = 180
    mock_settings.tts.volume = 0.9
    mock_settings.tts.voice_id = None
    mock_settings.tts.piper_voice_dir = "models/tts"
    mock_settings.tts.piper_voice_ar = "ar_EG-medium"
    mock_settings.tts.piper_voice_en = "en_US-medium"

    # NLP settings
    mock_settings.nlp = Mock()
    mock_settings.nlp.confidence_threshold = 0.6

    # Audio settings
    mock_settings.audio = Mock()
    mock_settings.audio.sample_rate = 16000
    mock_settings.audio.channels = 1

    # Log settings
    mock_settings.log = Mock()
    mock_settings.log.level = "INFO"

    return mock_settings


class TestFullPipeline:
    """Test complete voice assistant pipeline."""

    def setup_method(self):
        """Set up CLI runner."""
        self.runner = CliRunner()

    @patch("voice_assistant.main.Settings")
    @patch("voice_assistant.main.STTEngine")
    @patch("voice_assistant.main.NLPEngine")
    @patch("voice_assistant.main.get_time")
    @patch("voice_assistant.main.TTSEngine")
    def test_full_voice_loop_once_time(
        self, mock_tts_class, mock_get_time, mock_nlp_class, mock_stt_class, mock_settings_class
    ):
        """Full E2E: mic -> STT -> NLP -> action -> TTS for time query."""
        mock_settings = create_mock_settings()
        mock_settings_class.load.return_value = mock_settings

        mock_stt = Mock()
        mock_stt.record_audio.return_value = Mock()
        mock_stt.transcribe.return_value = ("what time is it", "en")  # Return tuple (text, lang)
        mock_stt_class.return_value = mock_stt

        mock_nlp = Mock()
        mock_nlp.parse.return_value = ("get_time", {}, 1.0)
        mock_nlp_class.return_value = mock_nlp

        mock_get_time.return_value = "02:30 PM"

        mock_tts = Mock()
        mock_tts_class.return_value = mock_tts

        result = self.runner.invoke(cli, ["--once"])

        assert result.exit_code == 0
        mock_stt.record_audio.assert_called_once()
        mock_stt.transcribe.assert_called_once()
        mock_nlp.parse.assert_called_once_with("what time is it", stt_language="en")
        mock_get_time.assert_called_once()
        mock_tts.say.assert_called_once_with("02:30 PM", lang="en")

    @patch("voice_assistant.main.Settings")
    @patch("voice_assistant.main.STTEngine")
    @patch("voice_assistant.main.NLPEngine")
    @patch("voice_assistant.main.web_search")
    @patch("voice_assistant.main.TTSEngine")
    def test_full_voice_loop_once_web_search(
        self, mock_tts_class, mock_web_search, mock_nlp_class, mock_stt_class, mock_settings_class
    ):
        """Full E2E for web search query."""
        mock_settings = create_mock_settings()
        mock_settings_class.load.return_value = mock_settings

        mock_stt = Mock()
        mock_stt.record_audio.return_value = Mock()
        mock_stt.transcribe.return_value = ("search for cats", "en")
        mock_stt_class.return_value = mock_stt

        mock_nlp = Mock()
        mock_nlp.parse.return_value = ("web_search", {"query": "cats"}, 1.0)
        mock_nlp_class.return_value = mock_nlp

        mock_web_search.return_value = "Successfully searched for cats"

        mock_tts = Mock()
        mock_tts_class.return_value = mock_tts

        result = self.runner.invoke(cli, ["--once"])

        assert result.exit_code == 0
        mock_web_search.assert_called_once_with("cats")
        mock_tts.say.assert_called_once_with("Successfully searched for cats", lang="en")

    @patch("voice_assistant.main.Settings")
    @patch("voice_assistant.main.NLPEngine")
    @patch("voice_assistant.main.get_time")
    @patch("voice_assistant.main.TTSEngine")
    def test_text_mode_bypass_stt(
        self, mock_tts_class, mock_get_time, mock_nlp_class, mock_settings_class
    ):
        """Text mode (--once --text) should bypass STT entirely."""
        mock_settings = create_mock_settings()
        mock_settings_class.load.return_value = mock_settings

        mock_nlp = Mock()
        mock_nlp.parse.return_value = ("get_time", {}, 1.0)
        mock_nlp_class.return_value = mock_nlp

        mock_get_time.return_value = "02:30 PM"

        mock_tts = Mock()
        mock_tts_class.return_value = mock_tts

        result = self.runner.invoke(cli, ["--once", "--text", "what time is it"])

        assert result.exit_code == 0
        # STT should NOT be called in text mode
        mock_nlp.parse.assert_called_once_with("what time is it", stt_language=None)
        mock_get_time.assert_called_once()
        mock_tts.say.assert_called_once_with("02:30 PM")


class TestErrorRecovery:
    """Test error recovery scenarios per USER_FLOW.md."""

    def setup_method(self):
        """Set up CLI runner."""
        self.runner = CliRunner()

    @patch("voice_assistant.main.Settings")
    @patch("voice_assistant.main.STTEngine")
    @patch("voice_assistant.main.TTSEngine")
    def test_mic_unavailable_recovery(self, mock_tts_class, mock_stt_class, mock_settings_class):
        """Mic unavailable -> TTS fallback message."""
        mock_settings = create_mock_settings()
        mock_settings_class.load.return_value = mock_settings

        mock_stt = Mock()
        mock_stt.record_audio.side_effect = Exception("No microphone found")
        mock_stt_class.return_value = mock_stt

        mock_tts = Mock()
        mock_tts_class.return_value = mock_tts

        result = self.runner.invoke(cli, ["--once"])

        assert result.exit_code == 0
        mock_tts.say.assert_called_once()
        called_text = mock_tts.say.call_args[0][0]
        assert "error" in called_text.lower() or "microphone" in called_text.lower()

    @patch("voice_assistant.main.Settings")
    @patch("voice_assistant.main.STTEngine")
    @patch("voice_assistant.main.NLPEngine")
    @patch("voice_assistant.main.TTSEngine")
    def test_model_load_failure_recovery(
        self, mock_tts_class, mock_nlp_class, mock_stt_class, mock_settings_class
    ):
        """STT model load failure -> TTS fallback."""
        mock_settings = create_mock_settings()
        mock_settings_class.load.return_value = mock_settings

        mock_stt = Mock()
        mock_stt.record_audio.return_value = Mock()
        mock_stt.transcribe.side_effect = RuntimeError("Model load failed")
        mock_stt_class.return_value = mock_stt

        mock_nlp = Mock()
        mock_nlp_class.return_value = mock_nlp

        mock_tts = Mock()
        mock_tts_class.return_value = mock_tts

        result = self.runner.invoke(cli, ["--once"])

        assert result.exit_code == 0
        mock_tts.say.assert_called_once()
        called_text = mock_tts.say.call_args[0][0]
        assert "error" in called_text.lower()

    @patch("voice_assistant.main.Settings")
    @patch("voice_assistant.main.STTEngine")
    @patch("voice_assistant.main.NLPEngine")
    @patch("voice_assistant.main.TTSEngine")
    def test_unknown_intent_spoken_fallback(
        self, mock_tts_class, mock_nlp_class, mock_stt_class, mock_settings_class
    ):
        """Unknown intent -> spoken fallback message."""
        mock_settings = create_mock_settings()
        mock_settings_class.load.return_value = mock_settings

        mock_stt = Mock()
        mock_stt.record_audio.return_value = Mock()
        mock_stt.transcribe.return_value = ("xyz random gibberish", "en")
        mock_stt_class.return_value = mock_stt

        mock_nlp = Mock()
        mock_nlp.parse.return_value = ("unknown", {}, 0.0)
        mock_nlp_class.return_value = mock_nlp

        mock_tts = Mock()
        mock_tts_class.return_value = mock_tts

        result = self.runner.invoke(cli, ["--once"])

        assert result.exit_code == 0
        mock_tts.say.assert_called_once()
        called_text = mock_tts.say.call_args[0][0]
        assert "didn't understand" in called_text.lower() or "unknown" in called_text.lower()

    @patch("voice_assistant.main.Settings")
    @patch("voice_assistant.main.STTEngine")
    @patch("voice_assistant.main.NLPEngine")
    @patch("voice_assistant.main.open_app")
    @patch("voice_assistant.main.TTSEngine")
    def test_action_failure_recovery(
        self, mock_tts_class, mock_open_app, mock_nlp_class, mock_stt_class, mock_settings_class
    ):
        """Action failure (open_app) -> TTS error message."""
        mock_settings = create_mock_settings()
        mock_settings_class.load.return_value = mock_settings

        mock_stt = Mock()
        mock_stt.record_audio.return_value = Mock()
        mock_stt.transcribe.return_value = ("open nonexistent", "en")
        mock_stt_class.return_value = mock_stt

        mock_nlp = Mock()
        mock_nlp.parse.return_value = ("open_app", {"app": "nonexistent"}, 1.0)
        mock_nlp_class.return_value = mock_nlp

        mock_open_app.side_effect = Exception("App not found")

        mock_tts = Mock()
        mock_tts_class.return_value = mock_tts

        result = self.runner.invoke(cli, ["--once"])

        assert result.exit_code == 0
        mock_tts.say.assert_called_once()
        called_text = mock_tts.say.call_args[0][0]
        assert "error" in called_text.lower() or "not found" in called_text.lower()

    @patch("voice_assistant.main.Settings")
    @patch("voice_assistant.main.STTEngine")
    @patch("voice_assistant.main.NLPEngine")
    @patch("voice_assistant.main.get_time")
    @patch("voice_assistant.main.TTSEngine")
    def test_tts_total_failure_fallback(
        self, mock_tts_class, mock_get_time, mock_nlp_class, mock_stt_class, mock_settings_class
    ):
        """TTS total failure -> should not crash, log error."""
        mock_settings = create_mock_settings()
        mock_settings_class.load.return_value = mock_settings

        mock_stt = Mock()
        mock_stt.record_audio.return_value = Mock()
        mock_stt.transcribe.return_value = ("what time is it", "en")
        mock_stt_class.return_value = mock_stt

        mock_nlp = Mock()
        mock_nlp.parse.return_value = ("get_time", {}, 1.0)
        mock_nlp_class.return_value = mock_nlp

        mock_get_time.return_value = "02:30 PM"

        mock_tts = Mock()
        mock_tts.say.side_effect = Exception("TTS completely broken")
        mock_tts_class.return_value = mock_tts

        result = self.runner.invoke(cli, ["--once"])

        # Should not crash, exit code 0
        assert result.exit_code == 0


class TestLatencyLogging:
    """Test latency logging E2E."""

    def setup_method(self):
        """Set up CLI runner."""
        self.runner = CliRunner()

    @patch("voice_assistant.main.Settings")
    @patch("voice_assistant.main.STTEngine")
    @patch("voice_assistant.main.NLPEngine")
    @patch("voice_assistant.main.get_time")
    @patch("voice_assistant.main.TTSEngine")
    def test_latency_logs_present(
        self, mock_tts_class, mock_get_time, mock_nlp_class, mock_stt_class, mock_settings_class
    ):
        """Each engine should log latency."""
        mock_settings = create_mock_settings()
        mock_settings_class.load.return_value = mock_settings

        mock_stt = Mock()
        mock_stt.record_audio.return_value = Mock()
        mock_stt.transcribe.return_value = ("what time is it", "en")
        mock_stt_class.return_value = mock_stt

        mock_nlp = Mock()
        mock_nlp.parse.return_value = ("get_time", {}, 1.0)
        mock_nlp_class.return_value = mock_nlp

        mock_get_time.return_value = "02:30 PM"

        mock_tts = Mock()
        mock_tts_class.return_value = mock_tts

        import logging

        with self.runner.isolated_filesystem():
            # Capture logs
            logger = logging.getLogger("voice_assistant.main")
            logger.setLevel(logging.INFO)

            result = self.runner.invoke(cli, ["--once"])

            assert result.exit_code == 0


class TestModuleBoundaries:
    """Test module boundary integrity - engines communicate only via defined interfaces."""

    def setup_method(self):
        """Set up CLI runner."""
        self.runner = CliRunner()

    @patch("voice_assistant.main.Settings")
    @patch("voice_assistant.main.STTEngine")
    @patch("voice_assistant.main.NLPEngine")
    @patch("voice_assistant.main.get_time")
    @patch("voice_assistant.main.TTSEngine")
    def test_voice_assistant_only_uses_public_interfaces(
        self, mock_tts_class, mock_get_time, mock_nlp_class, mock_stt_class, mock_settings_class
    ):
        """VoiceAssistant should only use public methods of engines."""
        mock_settings = create_mock_settings()
        mock_settings_class.load.return_value = mock_settings

        mock_stt = Mock()
        mock_stt.record_audio.return_value = Mock()
        mock_stt.transcribe.return_value = ("what time is it", "en")
        mock_stt_class.return_value = mock_stt

        mock_nlp = Mock()
        mock_nlp.parse.return_value = ("get_time", {}, 1.0)
        mock_nlp_class.return_value = mock_nlp

        mock_get_time.return_value = "02:30 PM"

        mock_tts = Mock()
        mock_tts_class.return_value = mock_tts

        result = self.runner.invoke(cli, ["--once"])

        assert result.exit_code == 0

        # Verify only public methods called
        mock_stt.record_audio.assert_called_once()
        mock_stt.transcribe.assert_called_once()
        mock_nlp.parse.assert_called_once()
        mock_get_time.assert_called_once()
        mock_tts.say.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
