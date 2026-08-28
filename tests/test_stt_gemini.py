"""Tests for core/stt_gemini.py."""

from __future__ import annotations

from unittest.mock import Mock, patch

import numpy as np
import pytest

from config.settings import STTConfig, STTProvider
from core.exceptions import STTError
from core.stt_gemini import GeminiSTTEngine


class TestGeminiSTTEngine:
    """Test GeminiSTTEngine functionality."""

    def _make_config(self, **overrides) -> STTConfig:
        """Create STTConfig with Gemini settings."""
        defaults = {
            "provider": STTProvider.GEMINI,
            "gemini_api_key": "test-key",
            "gemini_model": "gemini-3.5-transcribe",
            "gemini_base_url": None,
            "gemini_timeout": 30.0,
            "gemini_language": None,
            "allowed_languages": ["ar", "en"],
            "language_detection_threshold": 0.7,
            "input_gain": 1.0,
            "auto_gain": False,
            "language": None,
            "initial_prompt": None,
        }
        defaults.update(overrides)
        return STTConfig(**defaults)

    def test_init_requires_api_key(self):
        """Init should fail without API key."""
        config = self._make_config(gemini_api_key=None)
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(STTError) as exc_info:
                GeminiSTTEngine(config)
            assert "Gemini API key not configured" in str(exc_info.value)

    def test_init_with_env_var(self):
        """Init should work with GEMINI_API_KEY env var."""
        config = self._make_config(gemini_api_key=None)
        with patch.dict("os.environ", {"GEMINI_API_KEY": "env-key"}):
            engine = GeminiSTTEngine(config)
            assert engine._api_key == "env-key"

    @patch("google.genai.Client")
    def test_init_converts_timeout_seconds_to_milliseconds(self, mock_client_class):
        """gemini_timeout is in seconds; the SDK expects milliseconds."""
        config = self._make_config(gemini_timeout=5.0, gemini_base_url="https://example.com/base")
        mock_client_class.return_value = Mock()
        with patch.dict("os.environ", {"GEMINI_API_KEY": "test"}):
            GeminiSTTEngine(config)

        http_options = mock_client_class.call_args.kwargs["http_options"]
        # Seconds -> milliseconds conversion, and base_url coexists (no overwrite).
        assert http_options["timeout"] == 5000
        assert http_options["base_url"] == "https://example.com/base"

    @patch("google.genai.Client")
    def test_init_default_timeout_becomes_30000_ms(self, mock_client_class):
        """Default 30.0s config should map to a 30000ms SDK timeout."""
        config = self._make_config()  # gemini_timeout default 30.0
        mock_client_class.return_value = Mock()
        with patch.dict("os.environ", {"GEMINI_API_KEY": "test"}):
            GeminiSTTEngine(config)

        http_options = mock_client_class.call_args.kwargs.get("http_options", {})
        assert http_options["timeout"] == 30000

    def test_prefer_ipv4_filters_ipv6(self, monkeypatch):
        """IPv4 filter should keep only AF_INET results and be idempotent."""
        import socket

        from core.stt_gemini import _prefer_ipv4

        def fake_getaddrinfo(*args, **kwargs):
            return [
                (socket.AF_INET6, socket.SOCK_STREAM, 6, "", ("2001:db8::1", 443, 0, 0)),
                (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("1.2.3.4", 443)),
            ]

        monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)
        _prefer_ipv4()
        result = socket.getaddrinfo("host", 443)
        assert all(entry[0] == socket.AF_INET for entry in result)
        assert len(result) == 1

        wrapped = socket.getaddrinfo
        _prefer_ipv4()  # idempotent: no double wrapping
        assert socket.getaddrinfo is wrapped

    def test_prefer_ipv4_falls_back_when_only_ipv6(self):
        """IPv6-only host should still return its original results."""
        import socket

        from core.stt_gemini import _prefer_ipv4

        def fake_get_laddrinfo(*args, **kwargs):
            return [(socket.AF_INET6, socket.SOCK_STREAM, 6, "", ("2001:db8::1", 443, 0, 0))]

        saved = socket.getaddrinfo
        socket.getaddrinfo = fake_get_laddrinfo  # type: ignore[assignment]
        try:
            _prefer_ipv4()
            result = socket.getaddrinfo("host", 443)
            assert result[0][0] == socket.AF_INET6  # untouched
        finally:
            socket.getaddrinfo = saved  # type: ignore[assignment]

    def test_encode_wav_format(self):
        """WAV encoding should produce valid 16kHz mono 16-bit PCM."""
        config = self._make_config()
        with patch.dict("os.environ", {"GEMINI_API_KEY": "test"}):
            engine = GeminiSTTEngine(config)

        # 1 second of silence at 16kHz
        audio = np.zeros(16000, dtype=np.float32)
        wav_bytes = engine._encode_wav(audio)

        # Verify WAV header
        assert wav_bytes[:4] == b"RIFF"
        assert wav_bytes[8:12] == b"WAVE"
        assert wav_bytes[12:16] == b"fmt "

        # Check format chunk: audio format (1=PCM), channels (1), sample rate (16000)
        import struct

        fmt_chunk = wav_bytes[20:36]
        audio_format, channels, sample_rate, _, _, bits_per_sample = struct.unpack(
            "<HHIIHH", fmt_chunk
        )
        assert audio_format == 1  # PCM
        assert channels == 1  # mono
        assert sample_rate == 16000
        assert bits_per_sample == 16

    def test_encode_wav_applies_gain(self):
        """WAV encoding should apply input gain."""
        config = self._make_config(input_gain=2.0)
        with patch.dict("os.environ", {"GEMINI_API_KEY": "test"}):
            engine = GeminiSTTEngine(config)

        # Signal at 0.5 amplitude
        audio = np.full(16000, 0.5, dtype=np.float32)
        wav_bytes = engine._encode_wav(audio)

        # With 2x gain, should clip at 1.0 -> 32767
        import io
        import wave

        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            frames = wf.readframes(wf.getnframes())
            samples = np.frombuffer(frames, dtype=np.int16)
            assert np.max(samples) == 32767  # clipped

    @patch("google.genai.Client")
    def test_transcribe_success(self, mock_client_class):
        """Successful transcription should return text and language."""
        config = self._make_config()
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        mock_file = Mock()
        mock_file.uri = "file://test-file-uri"
        mock_file.mime_type = "audio/wav"
        mock_client.files.upload.return_value = mock_file

        mock_interaction = Mock()
        mock_interaction.output_text = "Hello world"
        mock_client.interactions.create.return_value = mock_interaction

        with patch.dict("os.environ", {"GEMINI_API_KEY": "test"}):
            engine = GeminiSTTEngine(config)

        audio = np.zeros(16000, dtype=np.float32)
        text, lang = engine.transcribe(audio)

        assert text == "Hello world"
        assert lang is None  # Gemini does not return detected-language metadata
        mock_client.files.upload.assert_called_once()
        mock_client.interactions.create.assert_called_once()

    @patch("google.genai.Client")
    def test_transcribe_with_forced_language(self, mock_client_class):
        """Should pass language_codes when forced."""
        config = self._make_config(gemini_language="ar-EG")
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        mock_file = Mock()
        mock_file.uri = "file://test-file-uri"
        mock_file.mime_type = "audio/wav"
        mock_client.files.upload.return_value = mock_file

        mock_interaction = Mock()
        mock_interaction.output_text = "مرحبا"
        mock_client.interactions.create.return_value = mock_interaction

        with patch.dict("os.environ", {"GEMINI_API_KEY": "test"}):
            engine = GeminiSTTEngine(config)

        audio = np.zeros(16000, dtype=np.float32)
        text, lang = engine.transcribe(audio)

        assert text == "مرحبا"
        # Verify language_codes was passed in generation_config
        call_args = mock_client.interactions.create.call_args
        gen_config = call_args.kwargs.get("generation_config", {})
        trans_config = gen_config.get("transcription_config", {})
        assert trans_config.get("language_codes") == ["ar-EG"]

    @patch("google.genai.Client")
    def test_transcribe_with_prompt(self, mock_client_class):
        """Should pass initial_prompt when provided."""
        config = self._make_config()
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        mock_file = Mock()
        mock_file.uri = "file://test-file-uri"
        mock_file.mime_type = "audio/wav"
        mock_client.files.upload.return_value = mock_file

        mock_interaction = Mock()
        mock_interaction.output_text = "Hello"
        mock_client.interactions.create.return_value = mock_interaction

        with patch.dict("os.environ", {"GEMINI_API_KEY": "test"}):
            engine = GeminiSTTEngine(config)

        audio = np.zeros(16000, dtype=np.float32)
        engine.transcribe(audio, initial_prompt="test prompt")

        # Verify the call was made
        mock_client.interactions.create.assert_called_once()

    @patch("google.genai.Client")
    def test_transcribe_rate_limit_retry(self, mock_client_class):
        """Should retry on rate limit errors."""
        config = self._make_config()
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        mock_file = Mock()
        mock_file.uri = "file://test-file-uri"
        mock_file.mime_type = "audio/wav"
        mock_client.files.upload.return_value = mock_file

        mock_interaction = Mock()
        mock_interaction.output_text = "Hello"

        mock_client.interactions.create.side_effect = [
            Exception("Rate limit exceeded (429)"),
            Exception("Rate limit exceeded (429)"),
            mock_interaction,
        ]

        with patch.dict("os.environ", {"GEMINI_API_KEY": "test"}):
            engine = GeminiSTTEngine(config)

        audio = np.zeros(16000, dtype=np.float32)
        text, lang = engine.transcribe(audio)

        assert text == "Hello"
        assert mock_client.interactions.create.call_count == 3

    @patch("google.genai.Client")
    def test_transcribe_rate_limit_exhausted(self, mock_client_class):
        """Should raise after max retries exhausted."""
        config = self._make_config()
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        mock_file = Mock()
        mock_file.uri = "file://test-file-uri"
        mock_file.mime_type = "audio/wav"
        mock_client.files.upload.return_value = mock_file

        mock_client.interactions.create.side_effect = Exception("Rate limit exceeded (429)")

        with patch.dict("os.environ", {"GEMINI_API_KEY": "test"}):
            engine = GeminiSTTEngine(config)

        audio = np.zeros(16000, dtype=np.float32)
        with pytest.raises(STTError) as exc_info:
            engine.transcribe(audio)
        assert "rate limit" in str(exc_info.value).lower()

    @patch("google.genai.Client")
    def test_transcribe_api_error(self, mock_client_class):
        """Should raise STTError on API error."""
        config = self._make_config()
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        mock_file = Mock()
        mock_file.uri = "file://test-file-uri"
        mock_file.mime_type = "audio/wav"
        mock_client.files.upload.return_value = mock_file

        mock_client.interactions.create.side_effect = Exception("Unauthorized: permission denied")

        with patch.dict("os.environ", {"GEMINI_API_KEY": "test"}):
            engine = GeminiSTTEngine(config)

        audio = np.zeros(16000, dtype=np.float32)
        with pytest.raises(STTError) as exc_info:
            engine.transcribe(audio)
        assert (
            "unauthorized" in str(exc_info.value).lower()
            or "permission" in str(exc_info.value).lower()
        )

    def test_transcribe_empty_audio(self):
        """Empty audio should return empty string."""
        config = self._make_config()
        with patch.dict("os.environ", {"GEMINI_API_KEY": "test"}):
            engine = GeminiSTTEngine(config)

        audio = np.array([], dtype=np.float32)
        text, lang = engine.transcribe(audio)
        assert text == ""
        assert lang is None

    @patch("sounddevice.rec")
    @patch("sounddevice.wait")
    def test_record_audio(self, mock_wait, mock_rec):
        """record_audio should return float32 array at 16kHz mono."""
        mock_recording = np.random.rand(48000, 1).astype(np.float32)
        mock_rec.return_value = mock_recording
        mock_wait.return_value = None

        config = self._make_config()
        with patch.dict("os.environ", {"GEMINI_API_KEY": "test"}):
            engine = GeminiSTTEngine(config)
            audio = engine.record_audio(duration=3.0)

        assert isinstance(audio, np.ndarray)
        assert audio.dtype == np.float32
        assert audio.shape[1] == 1
        mock_rec.assert_called_once()
        assert mock_rec.call_args.kwargs["samplerate"] == 16000

    def test_set_input_gain(self):
        """set_input_gain should clamp value."""
        config = self._make_config()
        with patch.dict("os.environ", {"GEMINI_API_KEY": "test"}):
            engine = GeminiSTTEngine(config)

        engine.set_input_gain(5.0)
        assert engine._input_gain == 5.0

        engine.set_input_gain(15.0)  # above max
        assert engine._input_gain == 10.0

        engine.set_input_gain(0.001)  # below min
        assert engine._input_gain == 0.01

    def test_set_auto_gain(self):
        """set_auto_gain should update flag."""
        config = self._make_config(auto_gain=False)
        with patch.dict("os.environ", {"GEMINI_API_KEY": "test"}):
            engine = GeminiSTTEngine(config)

        engine.set_auto_gain(True)
        assert engine._auto_gain is True

    def test_set_language(self):
        """set_language should update forced language."""
        config = self._make_config()
        with patch.dict("os.environ", {"GEMINI_API_KEY": "test"}):
            engine = GeminiSTTEngine(config)

        engine.set_language("ar-EG")
        assert engine._forced_language == "ar-EG"

        engine.set_language(None)
        assert engine._forced_language is None

    def test_close(self):
        """close should clean up resources."""
        config = self._make_config()
        with patch.dict("os.environ", {"GEMINI_API_KEY": "test"}):
            engine = GeminiSTTEngine(config)

        engine.close()  # Should not raise

    def test_context_manager(self):
        """Engine should work as context manager."""
        config = self._make_config()
        with (
            patch.dict("os.environ", {"GEMINI_API_KEY": "test"}),
            GeminiSTTEngine(config) as engine,
        ):
            assert engine is not None
        # Context exit should not raise


class TestSTTFactory:
    """Test create_stt_engine factory function."""

    def _make_settings(self, provider: STTProvider, **overrides) -> STTConfig:
        """Create STTConfig with given provider."""
        defaults = {
            "provider": provider,
            "gemini_api_key": "test-key" if provider == STTProvider.GEMINI else None,
            "gemini_model": "gemini-3.5-transcribe",
            "language": None,
            "allowed_languages": ["ar", "en"],
            "language_detection_threshold": 0.7,
            "initial_prompt": None,
            "input_gain": 1.0,
            "auto_gain": False,
            "gemini_base_url": None,
            "gemini_timeout": 30.0,
            "gemini_language": None,
        }
        defaults.update(overrides)
        return STTConfig(**defaults)

    @patch("core.stt_gemini.GeminiSTTEngine.__init__", return_value=None)
    def test_factory_returns_gemini(self, mock_init):
        """Factory should return Gemini engine when provider=gemini."""
        from core.stt_engine import create_stt_engine

        config = self._make_settings(STTProvider.GEMINI)
        engine = create_stt_engine(config)

        from core.stt_gemini import GeminiSTTEngine

        assert isinstance(engine, GeminiSTTEngine)

    def test_factory_requires_gemini_api_key(self):
        """Factory should fail when Gemini credentials are unavailable."""
        from core.stt_engine import create_stt_engine

        config = self._make_settings(STTProvider.GEMINI, gemini_api_key=None)
        with (
            patch.dict("os.environ", {}, clear=True),
            pytest.raises(STTError, match="Gemini API key not configured"),
        ):
            create_stt_engine(config)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
