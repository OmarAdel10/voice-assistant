"""Voice Assistant — Speech-to-Text Engine."""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
from numpy.typing import NDArray

from core.exceptions import STTError

logger = logging.getLogger(__name__)

# Type alias for Whisper model
WhisperModel = Any


class STTEngine:
    """Speech-to-Text engine using faster-whisper."""

    def __init__(
        self,
        model_size: str = "tiny.en",
        device: str = "cpu",
        compute_type: str = "int8",
        sample_rate: int = 16000,
    ) -> None:
        """Initialize STT engine (lazy model loading).

        Args:
            model_size: Whisper model size (tiny.en, base.en, small.en, etc.)
            device: Device to run on (cpu, cuda)
            compute_type: Quantization type (int8, float16, float32)
            sample_rate: Audio sample rate in Hz
        """
        self._model_size = model_size
        self._device = device
        self._compute_type = compute_type
        self._sample_rate = sample_rate
        self._model: WhisperModel | None = None

    def load_model(self) -> None:
        """Load the Whisper model (idempotent)."""
        if self._model is not None:
            return

        try:
            # Import here to avoid hard dependency at module load time
            from faster_whisper import WhisperModel  # type: ignore[import-untyped]

            logger.info(
                f"Loading STT model: {self._model_size} on {self._device} ({self._compute_type})"
            )
            self._model = WhisperModel(
                self._model_size,
                device=self._device,
                compute_type=self._compute_type,
            )
            logger.info("STT model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load STT model: {e}")
            raise STTError(f"Failed to load STT model: {e}") from e

    def record_audio(self, duration: float) -> NDArray[np.float32]:
        """Record audio from microphone.

        Args:
            duration: Recording duration in seconds

        Returns:
            Audio data as float32 numpy array (mono, 16kHz)

        Raises:
            STTError: If recording fails
        """
        try:
            import sounddevice as sd  # type: ignore[import-untyped]

            logger.info(f"Recording audio for {duration}s at {self._sample_rate}Hz")
            frames = int(duration * self._sample_rate)

            recording = sd.rec(
                frames=frames,
                samplerate=self._sample_rate,
                channels=1,
                dtype="float32",
            )
            sd.wait()

            # Ensure correct shape (n_samples, 1) for mono
            if recording.ndim == 1:
                recording = recording.reshape(-1, 1)

            logger.info(f"Recorded {len(recording)} frames")
            return np.asarray(recording, dtype=np.float32)  # type: ignore[return-value]
        except Exception as e:
            logger.error(f"Failed to record audio: {e}")
            raise STTError(f"Failed to record audio: {e}") from e

    def transcribe(self, audio: NDArray[np.float32]) -> str:
        """Transcribe audio to text.

        Args:
            audio: Audio data as float32 numpy array (mono, 16kHz)

        Returns:
            Transcribed text

        Raises:
            STTError: If transcription fails
        """
        if self._model is None:
            self.load_model()

        if audio.size == 0:
            logger.warning("Empty audio provided for transcription")
            return ""

        try:
            start_time = time.perf_counter()

            # faster-whisper expects float32 array
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)

            # Flatten if needed (n_samples, 1) -> (n_samples,)
            if audio.ndim == 2 and audio.shape[1] == 1:
                audio = audio.flatten()

            # At this point model is guaranteed to be loaded
            assert self._model is not None
            segments, info = self._model.transcribe(audio)

            elapsed_ms = (time.perf_counter() - start_time) * 1000
            lang = info.language
            prob = info.language_probability
            logger.info(f"STT: {elapsed_ms:.0f}ms | Language: {lang} (p={prob:.2f})")

            # Concatenate all segment texts
            text = " ".join(segment.text for segment in segments).strip()

            if not text:
                logger.warning("Transcription returned empty text")

            return text
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise STTError(f"Transcription failed: {e}") from e
