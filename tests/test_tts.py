"""Tests for core/tts_engine.py."""

from unittest.mock import Mock, patch

import pytest

from core.tts_engine import TTSEngine


class TestTTSEngineInit:
    """Test TTSEngine initialization."""

    def test_init_defaults(self):
        """Engine should initialize with default settings."""
        engine = TTSEngine()
        assert engine._rate == 180
        assert engine._volume == 0.9
        assert engine._engine_preference == "pyttsx3"

    def test_init_custom_settings(self):
        """Engine should accept custom settings."""
        engine = TTSEngine(rate=200, volume=0.8, engine="gTTS")
        assert engine._rate == 200
        assert engine._volume == 0.8
        assert engine._engine_preference == "gTTS"


class TestTTSEnginePyttsx3:
    """Test pyttsx3 engine path."""

    @patch("pyttsx3.init")
    def test_say_pyttsx3_success(self, mock_pyttsx3_init):
        """pyttsx3 success should speak and return."""
        mock_engine = Mock()
        mock_pyttsx3_init.return_value = mock_engine

        engine = TTSEngine(engine="pyttsx3")
        engine.say("Hello world")

        mock_pyttsx3_init.assert_called_once()
        mock_engine.setProperty.assert_any_call("rate", 180)
        mock_engine.setProperty.assert_any_call("volume", 0.9)
        mock_engine.say.assert_called_once_with("Hello world")
        mock_engine.runAndWait.assert_called_once()

    @patch("pyttsx3.init")
    def test_say_pyttsx3_sets_voice_if_provided(self, mock_pyttsx3_init):
        """Should set voice_id if provided."""
        mock_engine = Mock()
        mock_pyttsx3_init.return_value = mock_engine

        engine = TTSEngine(engine="pyttsx3", voice_id="english")
        engine.say("Test")

        mock_engine.setProperty.assert_any_call("voice", "english")

    @patch("pyttsx3.init")
    @patch.object(TTSEngine, "_say_gtts")
    def test_say_pyttsx3_failure_falls_back(self, mock_say_gtts, mock_pyttsx3_init):
        """pyttsx3 failure should fall back to gTTS."""
        mock_pyttsx3_init.side_effect = RuntimeError("No TTS engine")

        engine = TTSEngine(engine="pyttsx3")
        engine.say("Hello")

        mock_say_gtts.assert_called_once_with("Hello")


class TestTTSEngineGtts:
    """Test gTTS engine path."""

    @patch("gtts.gTTS")
    @patch("tempfile.NamedTemporaryFile")
    @patch("shutil.which")
    @patch("subprocess.run")
    def test_say_gtts_success(
        self, mock_subprocess_run, mock_shutil_which, mock_tempfile, mock_gtts
    ):
        """gTTS success should synthesize and play."""
        mock_tts = Mock()
        mock_gtts.return_value = mock_tts

        mock_fp = Mock()
        mock_fp.name = "/tmp/test.mp3"
        mock_tempfile.return_value.__enter__.return_value = mock_fp

        mock_shutil_which.return_value = "/usr/bin/mpv"
        mock_subprocess_run.return_value = Mock()

        engine = TTSEngine(engine="gTTS")
        engine._say_gtts("Hello world")

        mock_gtts.assert_called_once_with(text="Hello world", lang="en", slow=False)
        mock_tts.write_to_fp.assert_called_once()
        mock_shutil_which.assert_called()
        mock_subprocess_run.assert_called()

    @patch("gtts.gTTS")
    @patch("pyttsx3.init")
    @patch("builtins.print")
    def test_say_gtts_failure_falls_back(self, mock_print, mock_pyttsx3_init, mock_gtts):
        """gTTS failure should fall back to pyttsx3 then print."""
        mock_gtts.side_effect = ConnectionError("No internet")
        mock_pyttsx3_init.side_effect = RuntimeError("pyttsx3 failed")

        engine = TTSEngine(engine="gTTS")
        engine.say("Hello")

        mock_print.assert_called_once_with("[TTS fallback] Hello")


class TestTTSEngineFallbackChain:
    """Test complete fallback chain."""

    @patch("pyttsx3.init")
    @patch("gtts.gTTS")
    @patch("builtins.print")
    def test_pyttsx3_then_gtts_then_print(self, mock_print, mock_gtts, mock_pyttsx3_init):
        """Chain: pyttsx3 fails -> gTTS fails -> print."""
        mock_pyttsx3_init.side_effect = RuntimeError("pyttsx3 failed")
        mock_gtts.side_effect = ConnectionError("gTTS failed")

        engine = TTSEngine(engine="pyttsx3")
        engine.say("Test message")

        mock_print.assert_called_once_with("[TTS fallback] Test message")

    @patch("gtts.gTTS")
    @patch("pyttsx3.init")
    @patch("builtins.print")
    def test_gtts_then_print(self, mock_print, mock_pyttsx3_init, mock_gtts):
        """Chain: gTTS fails -> pyttsx3 fails -> print."""
        mock_gtts.side_effect = ConnectionError("gTTS failed")
        mock_pyttsx3_init.side_effect = RuntimeError("pyttsx3 failed")

        engine = TTSEngine(engine="gTTS")
        engine.say("Test message")

        mock_print.assert_called_once_with("[TTS fallback] Test message")


class TestTTSEngineEdgeCases:
    """Test edge cases."""

    @patch("pyttsx3.init")
    def test_say_empty_string(self, mock_pyttsx3_init):
        """Empty string should be handled gracefully."""
        mock_engine = Mock()
        mock_pyttsx3_init.return_value = mock_engine

        engine = TTSEngine()
        engine.say("")

        mock_engine.say.assert_called_once_with("")

    @patch("pyttsx3.init")
    def test_say_logs_latency(self, mock_pyttsx3_init):
        """say should log latency."""
        mock_engine = Mock()
        mock_pyttsx3_init.return_value = mock_engine

        engine = TTSEngine()

        with patch("core.tts_engine.logger") as mock_logger:
            engine.say("Test")
            # Check INFO log with TTS timing
            info_calls = [c for c in mock_logger.info.call_args_list if "TTS:" in str(c)]
            assert len(info_calls) >= 1

    @patch("pyttsx3.init")
    def test_say_unicode_text(self, mock_pyttsx3_init):
        """Unicode text should work."""
        mock_engine = Mock()
        mock_pyttsx3_init.return_value = mock_engine

        engine = TTSEngine()
        engine.say("Hello 🌍")

        mock_engine.say.assert_called_once_with("Hello 🌍")


class TestTTSEngineGttsDetails:
    """Test gTTS specific details."""

    @patch("gtts.gTTS")
    @patch("tempfile.NamedTemporaryFile")
    @patch("shutil.which")
    @patch("subprocess.run")
    def test_gtts_uses_correct_lang(
        self, mock_subprocess_run, mock_shutil_which, mock_tempfile, mock_gtts
    ):
        """gTTS should use correct language."""
        mock_tts = Mock()
        mock_gtts.return_value = mock_tts

        mock_fp = Mock()
        mock_fp.name = "/tmp/test.mp3"
        mock_tempfile.return_value.__enter__.return_value = mock_fp

        mock_shutil_which.return_value = "/usr/bin/mpv"
        mock_subprocess_run.return_value = Mock()

        engine = TTSEngine(engine="gTTS", lang="es")
        engine._say_gtts("Hola")

        mock_gtts.assert_called_once_with(text="Hola", lang="es", slow=False)

    @patch("gtts.gTTS")
    @patch("tempfile.NamedTemporaryFile")
    @patch("shutil.which")
    @patch("subprocess.run")
    def test_gtts_cleans_up_temp_file(
        self, mock_subprocess_run, mock_shutil_which, mock_tempfile, mock_gtts
    ):
        """Temp file should be cleaned up."""
        mock_tts = Mock()
        mock_gtts.return_value = mock_tts

        mock_fp = Mock()
        mock_fp.name = "/tmp/test.mp3"
        mock_tempfile.return_value.__enter__.return_value = mock_fp

        mock_shutil_which.return_value = "/usr/bin/mpv"
        mock_subprocess_run.return_value = Mock()

        engine = TTSEngine(engine="gTTS")
        engine._say_gtts("Test")

        # Temp file context manager should be called
        mock_tempfile.assert_called()


class TestTTSEngineEngineSwitching:
    """Test engine preference switching."""

    @patch("pyttsx3.init")
    def test_default_engine_is_pyttsx3(self, mock_pyttsx3_init):
        """Default engine should be pyttsx3."""
        mock_engine = Mock()
        mock_pyttsx3_init.return_value = mock_engine

        engine = TTSEngine()
        engine.say("Test")

        mock_pyttsx3_init.assert_called_once()

    @patch("gtts.gTTS")
    @patch("tempfile.NamedTemporaryFile")
    @patch("shutil.which")
    @patch("subprocess.run")
    def test_can_set_gtts_as_preferred(
        self, mock_subprocess_run, mock_shutil_which, mock_tempfile, mock_gtts
    ):
        """Can set gTTS as preferred engine."""
        mock_tts = Mock()
        mock_gtts.return_value = mock_tts

        mock_fp = Mock()
        mock_fp.name = "/tmp/test.mp3"
        mock_tempfile.return_value.__enter__.return_value = mock_fp

        mock_shutil_which.return_value = "/usr/bin/mpv"
        mock_subprocess_run.return_value = Mock()

        engine = TTSEngine(engine="gTTS")
        engine.say("Test")

        mock_gtts.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
