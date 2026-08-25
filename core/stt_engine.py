"""Voice Assistant — Speech-to-Text Engine."""

from __future__ import annotations

import logging
import time
from pathlib import Path
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
        model_size: str = "large-v3",
        device: str = "cuda",
        compute_type: str = "float16",
        sample_rate: int = 16000,
        model_dir: str = "models/stt",
        language: str | None = None,
        allowed_languages: list[str] | None = None,
        language_detection_threshold: float = 0.7,
        vad_filter: bool = True,
        vad_min_silence_ms: int = 500,
        initial_prompt: str | None = None,
    ) -> None:
        """Initialize STT engine (lazy model loading).

        Args:
            model_size: Whisper model size (tiny.en, base.en, small.en, large-v3, etc.)
            device: Device to run on (cpu, cuda)
            compute_type: Quantization type (int8, float16, float32)
            sample_rate: Audio sample rate in Hz
            model_dir: Local directory for model storage
            language: Language code (None for auto-detect)
            allowed_languages: List of allowed language codes (e.g., ["ar", "en"])
            language_detection_threshold: Minimum probability for language detection
            vad_filter: Enable VAD filtering
            vad_min_silence_ms: Minimum silence duration for VAD in milliseconds
            initial_prompt: Speaker adaptation prompt from voice enrollment
        """
        self._model_size = model_size
        self._device = device
        self._compute_type = compute_type
        self._sample_rate = sample_rate
        self._model_dir = Path(model_dir)
        self._language = language
        self._allowed_languages = allowed_languages
        self._language_detection_threshold = language_detection_threshold
        self._vad_filter = vad_filter
        self._vad_min_silence_ms = vad_min_silence_ms
        self._initial_prompt = initial_prompt
        self._model: WhisperModel | None = None

    def _get_model_path(self) -> Path:
        """Get the local model path."""
        return self._model_dir / self._model_size

    def load_model(self) -> None:
        """Load the Whisper model (idempotent)."""
        if self._model is not None:
            return

        model_path = self._get_model_path()
        try:
            # Import here to avoid hard dependency at module load time
            from faster_whisper import WhisperModel  # type: ignore[import-untyped]

            logger.info(
                f"Loading STT model: {self._model_size} on {self._device} ({self._compute_type})"
            )
            if model_path.exists():
                logger.info(f"Loading from local path: {model_path}")
                self._model = WhisperModel(
                    str(model_path),
                    device=self._device,
                    compute_type=self._compute_type,
                )
            else:
                logger.info("Model not found locally, downloading from Hugging Face Hub...")
                self._model = WhisperModel(
                    self._model_size,
                    device=self._device,
                    compute_type=self._compute_type,
                    download_root=str(self._model_dir),
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

    def transcribe(self, audio: NDArray[np.float32]) -> tuple[str, str | None]:
        """Transcribe audio to text with language detection.

        Args:
            audio: Audio data as float32 numpy array (mono, 16kHz)

        Returns:
            Tuple of (transcribed_text, detected_language_code)

        Raises:
            STTError: If transcription fails
        """
        if self._model is None:
            self.load_model()

        if audio.size == 0:
            logger.warning("Empty audio provided for transcription")
            return "", None

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
            segments, info = self._model.transcribe(
                audio,
                language=self._language,
                language_detection_threshold=self._language_detection_threshold,
                vad_filter=self._vad_filter,
                vad_parameters={"min_silence_duration_ms": self._vad_min_silence_ms},
                initial_prompt=self._initial_prompt,
            )

            elapsed_ms = (time.perf_counter() - start_time) * 1000
            lang = info.language
            prob = info.language_probability
            logger.info(f"STT: {elapsed_ms:.0f}ms | Language: {lang} (p={prob:.2f})")

            # Validate detected language against allowed list
            if self._allowed_languages and lang not in self._allowed_languages:
                logger.warning(
                    f"Detected language '{lang}' not in allowed languages "
                    f"{self._allowed_languages}. Probability: {prob:.2f}"
                )
                # If probability is low, try re-transcribing with first allowed language
                if prob < self._language_detection_threshold and self._allowed_languages:
                    fallback_lang = self._allowed_languages[0]
                    logger.info(f"Re-transcribing with fallback language: {fallback_lang}")
                    segments, info = self._model.transcribe(
                        audio,
                        language=fallback_lang,
                        vad_filter=self._vad_filter,
                        vad_parameters={"min_silence_duration_ms": self._vad_min_silence_ms},
                        initial_prompt=self._initial_prompt,
                    )
                    lang = fallback_lang
                    prob = info.language_probability
                    logger.info(f"STT (fallback): Language: {lang} (p={prob:.2f})")
                else:
                    # Language detected with high confidence but not allowed
                    logger.error(f"Language '{lang}' detected with high confidence but not allowed")
                    raise STTError(
                        f"Detected language '{lang}' is not supported. "
                        f"Allowed languages: {self._allowed_languages}"
                    )

            # Concatenate all segment texts
            text = " ".join(segment.text for segment in segments).strip()

            if not text:
                logger.warning("Transcription returned empty text")

            return text, lang
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise STTError(f"Transcription failed: {e}") from e
