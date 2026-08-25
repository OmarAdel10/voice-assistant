"""Tests for core/stt_engine.py."""

from unittest.mock import Mock, patch

import numpy as np
import pytest

from core.exceptions import STTError
from core.stt_engine import STTEngine


class TestSTTEngine:
    """Test STTEngine functionality."""

    def test_init_does_not_load_model(self):
        """__init__ should not load model (lazy loading)."""
        with patch("faster_whisper.WhisperModel") as mock_model_class:
            engine = STTEngine()
            mock_model_class.assert_not_called()
            assert engine._model is None

    def test_load_model_creates_whisper_model(self):
        """load_model should create WhisperModel with correct params."""
        with patch("faster_whisper.WhisperModel") as mock_model_class:
            mock_model = Mock()
            mock_model_class.return_value = mock_model

            engine = STTEngine()
            engine.load_model()

            # Engine uses download_root for Hugging Face cache
            mock_model_class.assert_called_once_with(
                "large-v3",
                device="cuda",
                compute_type="float16",
                download_root="models/stt",
            )
            assert engine._model is mock_model

    def test_load_model_idempotent(self):
        """Multiple load_model calls should not reload."""
        with patch("faster_whisper.WhisperModel") as mock_model_class:
            mock_model = Mock()
            mock_model_class.return_value = mock_model

            engine = STTEngine()
            engine.load_model()
            engine.load_model()
            engine.load_model()

            mock_model_class.assert_called_once()

    def test_load_model_failure_raises_stt_error(self):
        """Model load failure should raise STTError with context."""
        with patch("faster_whisper.WhisperModel", side_effect=RuntimeError("CUDA OOM")):
            engine = STTEngine()
            with pytest.raises(STTError) as exc_info:
                engine.load_model()
            assert "Failed to load STT model" in str(exc_info.value)

    @patch("sounddevice.rec")
    @patch("sounddevice.wait")
    def test_record_audio_returns_float32_array(self, mock_wait, mock_rec):
        """record_audio should return float32 numpy array at 16kHz mono."""
        mock_recording = np.random.rand(48000, 1).astype(np.float32)  # 3s at 16kHz
        mock_rec.return_value = mock_recording
        mock_wait.return_value = None

        engine = STTEngine()
        audio = engine.record_audio(duration=3.0)

        assert isinstance(audio, np.ndarray)
        assert audio.dtype == np.float32
        assert audio.shape[1] == 1  # mono
        mock_rec.assert_called_once()
        call_args = mock_rec.call_args
        assert call_args.kwargs["samplerate"] == 16000
        assert call_args.kwargs["channels"] == 1
        assert call_args.kwargs["dtype"] == "float32"

    @patch("sounddevice.rec")
    def test_record_audio_failure_raises_stt_error(self, mock_rec):
        """Recording failure should raise STTError."""
        mock_rec.side_effect = OSError("No input device")

        engine = STTEngine()
        with pytest.raises(STTError) as exc_info:
            engine.record_audio(duration=3.0)
        assert "Failed to record audio" in str(exc_info.value)

    def test_transcribe_returns_text_and_lang(self):
        """transcribe should return transcribed text and detected language."""
        with patch("faster_whisper.WhisperModel") as mock_model_class:
            mock_model = Mock()
            mock_segments = [Mock(text="Hello world"), Mock(text="How are you")]
            mock_info = Mock(language="en", language_probability=0.99)
            mock_model.transcribe.return_value = (mock_segments, mock_info)
            mock_model_class.return_value = mock_model

            engine = STTEngine()
            audio = np.zeros(16000, dtype=np.float32)  # 1s silence
            result = engine.transcribe(audio)

            assert result == ("Hello world How are you", "en")
            mock_model.transcribe.assert_called_once()

    def test_transcribe_empty_audio_returns_empty_string(self):
        """Empty audio should return empty string and None language."""
        with patch("faster_whisper.WhisperModel") as mock_model_class:
            mock_model = Mock()
            mock_info = Mock(language="en", language_probability=0.99)
            mock_model.transcribe.return_value = ([], mock_info)
            mock_model_class.return_value = mock_model

            engine = STTEngine()
            audio = np.array([], dtype=np.float32)
            result = engine.transcribe(audio)

            assert result == ("", None)

    def test_transcribe_failure_raises_stt_error(self):
        """Transcription failure should raise STTError with context."""
        with patch("faster_whisper.WhisperModel") as mock_model_class:
            mock_model = Mock()
            mock_model.transcribe.side_effect = RuntimeError("Model error")
            mock_model_class.return_value = mock_model

            engine = STTEngine()
            audio = np.zeros(16000, dtype=np.float32)
            with pytest.raises(STTError) as exc_info:
                engine.transcribe(audio)
            assert "Transcription failed" in str(exc_info.value)

    def test_transcribe_logs_latency(self, caplog):
        """transcribe should log STT duration in milliseconds."""
        import logging

        caplog.set_level(logging.INFO)

        with patch("faster_whisper.WhisperModel") as mock_model_class:
            mock_model = Mock()
            mock_segments = [Mock(text="Test")]
            mock_info = Mock(language="en", language_probability=0.99)
            mock_model.transcribe.return_value = (mock_segments, mock_info)
            mock_model_class.return_value = mock_model

            engine = STTEngine()
            audio = np.zeros(16000, dtype=np.float32)
            engine.transcribe(audio)

            # Check for latency log
            assert any(
                "STT:" in record.message and "ms" in record.message for record in caplog.records
            )

    def test_full_pipeline_record_and_transcribe(self):
        """Integration: record_audio + transcribe works together."""
        with (
            patch("faster_whisper.WhisperModel") as mock_model_class,
            patch("sounddevice.rec") as mock_rec,
            patch("sounddevice.wait") as mock_wait,
        ):
            mock_model = Mock()
            mock_segments = [Mock(text="Test transcription")]
            mock_info = Mock(language="en", language_probability=0.99)
            mock_model.transcribe.return_value = (mock_segments, mock_info)
            mock_model_class.return_value = mock_model

            mock_recording = np.random.rand(16000, 1).astype(np.float32)
            mock_rec.return_value = mock_recording
            mock_wait.return_value = None

            engine = STTEngine()
            audio = engine.record_audio(duration=1.0)
            text, lang = engine.transcribe(audio)

            assert text == "Test transcription"
            assert lang == "en"


class TestSTTEngineModelCache:
    """Test model caching behavior."""

    def test_model_cached_after_first_load(self):
        """Model should be cached after first load."""
        with patch("faster_whisper.WhisperModel") as mock_model_class:
            mock_model = Mock()
            mock_model_class.return_value = mock_model

            engine = STTEngine()
            engine.load_model()
            first_model = engine._model
            engine.load_model()
            second_model = engine._model

            assert first_model is second_model
