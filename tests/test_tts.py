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
        assert engine._engine_preference == "piper"
        assert engine._piper_voices == {
            "ar": "ar_JO-kareem-medium",
            "en": "en_US-lessac-medium",
            "ar_fallback": "ar_JO-kareem-low",
            "en_fallback": "en_US-lessac-low",
        }

    def test_init_custom_settings(self):
        """Engine should accept custom settings."""
        engine = TTSEngine(
            rate=200,
            volume=0.8,
            engine="pyttsx3",
            piper_voice_dir="/custom/tts",
            piper_voice_ar="ar_EG-custom",
            piper_voice_en="en_US-custom",
            piper_voice_ar_fallback="ar_EG-custom-fallback",
            piper_voice_en_fallback="en_US-custom-fallback",
        )
        assert engine._rate == 200
        assert engine._volume == 0.8
        assert engine._engine_preference == "pyttsx3"
        assert str(engine._piper_voice_dir) == "/custom/tts"
        assert engine._piper_voices == {
            "ar": "ar_EG-custom",
            "en": "en_US-custom",
            "ar_fallback": "ar_EG-custom-fallback",
            "en_fallback": "en_US-custom-fallback",
        }


class TestTTSEnginePiper:
    """Test piper engine path (primary)."""

    @patch("core.tts_engine.PiperVoice")
    @patch("core.tts_engine.sd")
    @patch("core.tts_engine.np")
    @patch("core.tts_engine.Path.exists", return_value=True)
    def test_say_piper_success(self, mock_exists, mock_np, mock_sd, mock_piper_voice):
        """piper success should synthesize and play."""
        mock_voice = Mock()
        mock_voice.config.sample_rate = 22050
        mock_voice.synthesize.return_value = [
            Mock(audio_int16_bytes=b"\x00\x01" * 1000),
        ]
        mock_piper_voice.load.return_value = mock_voice

        mock_audio_array = Mock()
        mock_np.frombuffer.return_value = mock_audio_array
        mock_np.int16 = "int16"
        mock_np.float32 = "float32"
        mock_audio_array.astype.return_value = mock_audio_array
        mock_audio_array.__truediv__ = Mock(return_value=mock_audio_array)

        engine = TTSEngine(engine="piper")
        engine.say("Hello world", lang="en")

        mock_piper_voice.load.assert_called_once()
        mock_voice.synthesize.assert_called_once_with("Hello world")
        mock_sd.play.assert_called_once()
        mock_sd.wait.assert_called_once()

    @patch("core.tts_engine.PiperVoice", side_effect=FileNotFoundError("Voice not found"))
    @patch.object(TTSEngine, "_say_pyttsx3")
    def test_say_piper_failure_falls_back(self, mock_say_pyttsx3, mock_piper_voice):
        """piper failure should fall back to pyttsx3."""
        engine = TTSEngine(engine="piper")
        engine.say("Hello", lang="en")

        mock_say_pyttsx3.assert_called_once_with("Hello")


class TestTTSEnginePyttsx3:
    """Test pyttsx3 engine path (fallback)."""

    @patch("pyttsx3.init")
    def test_say_pyttsx3_success(self, mock_pyttsx3_init):
        """pyttsx3 success should speak and return."""
        mock_engine = Mock()
        mock_pyttsx3_init.return_value = mock_engine

        engine = TTSEngine(engine="pyttsx3")
        engine.say("Hello world", lang="en")

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
        engine.say("Test", lang="en")

        mock_engine.setProperty.assert_any_call("voice", "english")

    @patch("core.tts_engine.PiperVoice", side_effect=FileNotFoundError("piper failed"))
    @patch("pyttsx3.init", side_effect=RuntimeError("No TTS engine"))
    @patch.object(TTSEngine, "_say_fallback")
    def test_say_pyttsx3_failure_falls_back(
        self, mock_say_fallback, mock_pyttsx3_init, mock_piper_voice
    ):
        """pyttsx3 failure should fall back to piper, then print."""
        engine = TTSEngine(engine="pyttsx3")
        engine.say("Hello", lang="en")

        mock_say_fallback.assert_called_once_with("Hello")


class TestTTSEngineFallbackChain:
    """Test complete fallback chain."""

    @patch("core.tts_engine.PiperVoice", side_effect=FileNotFoundError("piper failed"))
    @patch("pyttsx3.init", side_effect=RuntimeError("pyttsx3 failed"))
    @patch.object(TTSEngine, "_say_fallback")
    def test_piper_then_pyttsx3_then_print(
        self, mock_say_fallback, mock_pyttsx3_init, mock_piper_voice
    ):
        """Chain: piper fails -> pyttsx3 fails -> print."""
        engine = TTSEngine(engine="piper")
        engine.say("Test message", lang="en")

        mock_say_fallback.assert_called_once_with("Test message")

    @patch("pyttsx3.init", side_effect=RuntimeError("pyttsx3 failed"))
    @patch("core.tts_engine.PiperVoice", side_effect=FileNotFoundError("piper failed"))
    @patch.object(TTSEngine, "_say_fallback")
    def test_pyttsx3_then_piper_then_print(
        self, mock_say_fallback, mock_piper_voice, mock_pyttsx3_init
    ):
        """Chain: pyttsx3 fails -> piper fails -> print."""
        engine = TTSEngine(engine="pyttsx3")
        engine.say("Test message", lang="en")

        mock_say_fallback.assert_called_once_with("Test message")


class TestTTSEngineEdgeCases:
    """Test edge cases."""

    @patch("pyttsx3.init")
    def test_say_empty_string(self, mock_pyttsx3_init):
        """Empty string should be handled gracefully."""
        mock_engine = Mock()
        mock_pyttsx3_init.return_value = mock_engine

        engine = TTSEngine(engine="pyttsx3")
        engine.say("")

        mock_engine.say.assert_called_once_with("")

    @patch("pyttsx3.init")
    def test_say_logs_latency(self, mock_pyttsx3_init):
        """say should log latency."""
        mock_engine = Mock()
        mock_pyttsx3_init.return_value = mock_engine

        engine = TTSEngine(engine="pyttsx3")

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

        engine = TTSEngine(engine="pyttsx3")
        engine.say("Hello 🌍")

        mock_engine.say.assert_called_once_with("Hello 🌍")

    @patch("pyttsx3.init")
    def test_say_arabic_text_uses_ar_voice(self, mock_pyttsx3_init):
        """Arabic text should be detected and use pyttsx3 (piper fallback)."""
        mock_engine = Mock()
        mock_pyttsx3_init.return_value = mock_engine

        engine = TTSEngine(engine="pyttsx3")
        engine.say("مرحبا", lang="ar")

        mock_engine.say.assert_called_once_with("مرحبا")


class TestTTSEngineLanguageDetection:
    """Test language detection."""

    def test_detect_language_arabic(self):
        """Arabic script should be detected as ar."""
        engine = TTSEngine()
        assert engine._detect_language("مرحبا") == "ar"
        assert engine._detect_language("السلام عليكم") == "ar"

    def test_detect_language_english(self):
        """Latin script should be detected as en."""
        engine = TTSEngine()
        assert engine._detect_language("Hello") == "en"
        assert engine._detect_language("How are you?") == "en"

    def test_detect_language_mixed(self):
        """Mixed text: any Arabic char -> ar, else en."""
        engine = TTSEngine()
        # Contains Arabic char -> ar
        assert engine._detect_language("مرحبا world") == "ar"
        # Contains Arabic char -> ar
        assert engine._detect_language("Hello عالم") == "ar"
        # No Arabic -> en
        assert engine._detect_language("Hello world") == "en"


class TestTTSEngineEngineSwitching:
    """Test engine preference switching."""

    @patch("core.tts_engine.PiperVoice")
    @patch("core.tts_engine.sd")
    @patch("core.tts_engine.np")
    @patch("core.tts_engine.Path.exists", return_value=True)
    def test_default_engine_is_piper(self, mock_exists, mock_np, mock_sd, mock_piper_voice):
        """Default engine should be piper."""
        mock_voice = Mock()
        mock_voice.config.sample_rate = 22050
        mock_voice.synthesize.return_value = [Mock(audio_int16_bytes=b"\x00\x01" * 1000)]
        mock_piper_voice.load.return_value = mock_voice

        mock_audio_array = Mock()
        mock_np.frombuffer.return_value = mock_audio_array
        mock_np.int16 = "int16"
        mock_np.float32 = "float32"
        mock_audio_array.astype.return_value = mock_audio_array
        mock_audio_array.__truediv__ = Mock(return_value=mock_audio_array)

        engine = TTSEngine()  # Default engine="piper"
        engine.say("Test", lang="en")

        mock_piper_voice.load.assert_called_once()

    @patch("pyttsx3.init")
    def test_can_set_pyttsx3_as_preferred(self, mock_pyttsx3_init):
        """Can set pyttsx3 as preferred engine."""
        mock_engine = Mock()
        mock_pyttsx3_init.return_value = mock_engine

        engine = TTSEngine(engine="pyttsx3")
        engine.say("Test", lang="en")

        mock_pyttsx3_init.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
