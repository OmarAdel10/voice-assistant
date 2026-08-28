"""Tests for voice_assistant/main.py CLI."""

from unittest.mock import Mock, patch

from click.testing import CliRunner

from voice_assistant.main import cli


def create_mock_settings() -> Mock:
    """Create a properly structured mock Settings object."""
    mock_settings = Mock()

    # STT settings
    mock_settings.stt = Mock()
    mock_settings.stt.max_listen_seconds = 5
    mock_settings.stt.language = "ar"

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
    mock_settings.nlp.confidence_threshold_ar = 0.5
    mock_settings.nlp.confidence_threshold_en = 0.6

    # Audio settings
    mock_settings.audio = Mock()
    mock_settings.audio.sample_rate = 16000
    mock_settings.audio.channels = 1

    # Log settings
    mock_settings.log = Mock()
    mock_settings.log.level = "INFO"

    return mock_settings


class TestCLI:
    """Test CLI entrypoint."""

    def setup_method(self):
        """Set up CLI runner."""
        self.runner = CliRunner()

    @patch("voice_assistant.main.Settings")
    def test_list_intents(self, mock_settings_class):
        """--list-intents should show available intents."""
        mock_settings = create_mock_settings()
        mock_settings_class.load.return_value = mock_settings

        result = self.runner.invoke(cli, ["--list-intents"])

        assert result.exit_code == 0
        assert "get_time" in result.output
        assert "get_date" in result.output
        assert "get_sys_info" in result.output
        assert "open_app" in result.output
        assert "web_search" in result.output

    @patch("voice_assistant.main.Settings")
    @patch("voice_assistant.main.NLPEngine")
    @patch("voice_assistant.main.get_time")
    @patch("voice_assistant.main.TTSEngine")
    def test_once_text_mode_time(
        self, mock_tts_class, mock_get_time, mock_nlp_class, mock_settings_class
    ):
        """--once --text 'what time is it' should execute get_time."""
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
        mock_nlp.parse.assert_called_once_with("what time is it", stt_language=None)
        mock_get_time.assert_called_once()

    @patch("voice_assistant.main.Settings")
    @patch("voice_assistant.main.NLPEngine")
    @patch("voice_assistant.main.get_date")
    @patch("voice_assistant.main.TTSEngine")
    def test_once_text_mode_date(
        self, mock_tts_class, mock_get_date, mock_nlp_class, mock_settings_class
    ):
        """--once --text 'what date is it' should execute get_date."""
        mock_settings = create_mock_settings()
        mock_settings_class.load.return_value = mock_settings

        mock_nlp = Mock()
        mock_nlp.parse.return_value = ("get_date", {}, 1.0)
        mock_nlp_class.return_value = mock_nlp

        mock_get_date.return_value = "Monday, January 15, 2024"

        mock_tts = Mock()
        mock_tts_class.return_value = mock_tts

        result = self.runner.invoke(cli, ["--once", "--text", "what date is it"])

        assert result.exit_code == 0
        mock_get_date.assert_called_once()

    @patch("voice_assistant.main.Settings")
    @patch("voice_assistant.main.NLPEngine")
    @patch("voice_assistant.main.get_sysinfo")
    @patch("voice_assistant.main.TTSEngine")
    def test_once_text_mode_sysinfo(
        self, mock_tts_class, mock_get_sysinfo, mock_nlp_class, mock_settings_class
    ):
        """--once --text 'system info' should execute get_sysinfo."""
        mock_settings = create_mock_settings()
        mock_settings_class.load.return_value = mock_settings

        mock_nlp = Mock()
        mock_nlp.parse.return_value = ("get_sys_info", {}, 1.0)
        mock_nlp_class.return_value = mock_nlp

        mock_get_sysinfo.return_value = {
            "cpu_percent": 25.5,
            "memory_percent": 60.0,
            "disk_percent": 45.0,
        }

        mock_tts = Mock()
        mock_tts_class.return_value = mock_tts

        result = self.runner.invoke(cli, ["--once", "--text", "system info"])

        assert result.exit_code == 0
        mock_get_sysinfo.assert_called_once()

    @patch("voice_assistant.main.Settings")
    @patch("voice_assistant.main.NLPEngine")
    @patch("voice_assistant.main.open_app")
    @patch("voice_assistant.main.TTSEngine")
    def test_once_text_mode_open_app(
        self, mock_tts_class, mock_open_app, mock_nlp_class, mock_settings_class
    ):
        """--once --text 'open firefox' should execute open_app."""
        mock_settings = create_mock_settings()
        mock_settings_class.load.return_value = mock_settings

        mock_nlp = Mock()
        mock_nlp.parse.return_value = ("open_app", {"app": "firefox"}, 1.0)
        mock_nlp_class.return_value = mock_nlp

        mock_open_app.return_value = "Successfully launched firefox"

        mock_tts = Mock()
        mock_tts_class.return_value = mock_tts

        result = self.runner.invoke(cli, ["--once", "--text", "open firefox"])

        assert result.exit_code == 0
        mock_open_app.assert_called_once_with("firefox", lang="en")

    @patch("voice_assistant.main.Settings")
    @patch("voice_assistant.main.NLPEngine")
    @patch("voice_assistant.main.web_search")
    @patch("voice_assistant.main.TTSEngine")
    def test_once_text_mode_web_search(
        self, mock_tts_class, mock_web_search, mock_nlp_class, mock_settings_class
    ):
        """--once --text 'search cats' should execute web_search."""
        mock_settings = create_mock_settings()
        mock_settings_class.load.return_value = mock_settings

        mock_nlp = Mock()
        mock_nlp.parse.return_value = ("web_search", {"query": "cats"}, 1.0)
        mock_nlp_class.return_value = mock_nlp

        mock_web_search.return_value = "Successfully searched for cats"

        mock_tts = Mock()
        mock_tts_class.return_value = mock_tts

        result = self.runner.invoke(cli, ["--once", "--text", "search cats"])

        assert result.exit_code == 0
        mock_web_search.assert_called_once_with("cats")

    @patch("voice_assistant.main.Settings")
    @patch("voice_assistant.main.NLPEngine")
    @patch("voice_assistant.main.TTSEngine")
    def test_once_text_mode_unknown_intent(
        self, mock_tts_class, mock_nlp_class, mock_settings_class
    ):
        """Unknown intent should pass error message to TTS."""
        mock_settings = create_mock_settings()
        mock_settings_class.load.return_value = mock_settings

        mock_nlp = Mock()
        mock_nlp.parse.return_value = ("unknown", {}, 0.0)
        mock_nlp.get_response_template.return_value = (
            "I didn't understand that command. Try asking for time, date, "
            "system info, opening an app, or searching the web."
        )
        mock_nlp_class.return_value = mock_nlp

        mock_tts = Mock()
        mock_tts_class.return_value = mock_tts

        result = self.runner.invoke(cli, ["--once", "--text", "xyz gibberish"])

        assert result.exit_code == 0
        # Verify TTS was called with error message
        mock_tts.say.assert_called_once()
        called_text = mock_tts.say.call_args[0][0]
        assert "didn't understand" in called_text.lower() or "unknown" in called_text.lower()

    @patch("voice_assistant.main.Settings")
    @patch("voice_assistant.main.create_stt_engine")
    @patch("voice_assistant.main.NLPEngine")
    @patch("voice_assistant.main.get_time")
    @patch("voice_assistant.main.TTSEngine")
    def test_once_mode_voice(
        self, mock_tts_class, mock_get_time, mock_nlp_class, mock_stt_class, mock_settings_class
    ):
        """--once should listen, transcribe, parse, execute, speak."""
        mock_settings = create_mock_settings()
        mock_settings_class.load.return_value = mock_settings

        mock_stt = Mock()
        mock_stt.record_audio.return_value = Mock()
        mock_stt.transcribe.return_value = ("what time is it", "en")  # Return tuple (text, lang)
        mock_stt_class.return_value = mock_stt

        mock_nlp = Mock()
        mock_nlp.parse.return_value = ("get_time", {}, 1.0)
        mock_nlp._detect_language.return_value = "en"
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
    @patch("voice_assistant.main.create_stt_engine")
    @patch("voice_assistant.main.NLPEngine")
    @patch("voice_assistant.main.TTSEngine")
    @patch("signal.signal")
    def test_listen_mode_sigint(
        self, mock_signal, mock_tts_class, mock_nlp_class, mock_stt_class, mock_settings_class
    ):
        """SIGINT should trigger clean shutdown."""
        mock_settings = create_mock_settings()
        mock_settings_class.load.return_value = mock_settings

        mock_stt = Mock()
        mock_stt_class.return_value = mock_stt

        mock_nlp = Mock()
        mock_nlp_class.return_value = mock_nlp

        mock_tts = Mock()
        mock_tts_class.return_value = mock_tts

        # Simulate SIGINT on first iteration
        def raise_sigint(*args, **kwargs):
            raise KeyboardInterrupt()

        mock_stt.record_audio.side_effect = raise_sigint

        result = self.runner.invoke(cli, ["--listen"])

        assert result.exit_code == 0
        mock_signal.assert_called()

    def test_version_flag(self):
        """--version should show version."""
        result = self.runner.invoke(cli, ["--version"])

        assert result.exit_code == 0
        assert "0.1.0" in result.output

    def test_help_flag(self):
        """--help should show usage with examples."""
        result = self.runner.invoke(cli, ["--help"])

        assert result.exit_code == 0
        assert "--listen" in result.output
        assert "--once" in result.output
        assert "--text" in result.output
        assert "--list-intents" in result.output
        assert "examples" in result.output.lower()

    @patch("voice_assistant.main.Settings")
    @patch("voice_assistant.main.logging.basicConfig")
    def test_logging_config_from_settings(self, mock_basicConfig, mock_settings_class):
        """Logging should be configured from settings."""
        mock_settings = create_mock_settings()
        mock_settings.log.level = "DEBUG"
        mock_settings_class.load.return_value = mock_settings

        self.runner.invoke(cli, ["--list-intents"])

        mock_basicConfig.assert_called_once()
        call_kwargs = mock_basicConfig.call_args[1]
        assert call_kwargs["level"] == "DEBUG"

    @patch("voice_assistant.main.Settings")
    @patch("voice_assistant.main.NLPEngine")
    @patch("voice_assistant.main.TTSEngine")
    def test_nlp_error_handling(self, mock_tts_class, mock_nlp_class, mock_settings_class):
        """NLP errors should be handled gracefully and passed to TTS."""
        mock_settings = create_mock_settings()
        mock_settings_class.load.return_value = mock_settings

        mock_nlp = Mock()
        mock_nlp.parse.side_effect = Exception("NLP failed")
        mock_nlp_class.return_value = mock_nlp

        mock_tts = Mock()
        mock_tts_class.return_value = mock_tts

        result = self.runner.invoke(cli, ["--once", "--text", "test"])

        assert result.exit_code == 0
        # Verify TTS was called with error message
        mock_tts.say.assert_called_once()
        called_text = mock_tts.say.call_args[0][0]
        assert "error" in called_text.lower()

    @patch("voice_assistant.main.Settings")
    @patch("voice_assistant.main.create_stt_engine")
    @patch("voice_assistant.main.TTSEngine")
    def test_stt_error_handling(self, mock_tts_class, mock_stt_class, mock_settings_class):
        """STT errors should be handled gracefully and passed to TTS."""
        mock_settings = create_mock_settings()
        mock_settings_class.load.return_value = mock_settings

        mock_stt = Mock()
        mock_stt.record_audio.side_effect = Exception("No microphone")
        mock_stt_class.return_value = mock_stt

        mock_tts = Mock()
        mock_tts_class.return_value = mock_tts

        result = self.runner.invoke(cli, ["--once"])

        assert result.exit_code == 0
        # Verify TTS was called with error message
        mock_tts.say.assert_called_once()
        called_text = mock_tts.say.call_args[0][0]
        assert "error" in called_text.lower() or "microphone" in called_text.lower()


class TestCLIEntryPoint:
    """Test module execution."""

    def test_module_execution(self):
        """Module should be importable."""
        import voice_assistant.main

        assert hasattr(voice_assistant.main, "cli")
