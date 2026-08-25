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
        input_gain: float = 1.0,
        auto_gain: bool = False,
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
            input_gain: Input volume gain (0.1-1.0 to prevent clipping)
            auto_gain: Automatically adjust input gain based on audio levels
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
        self._input_gain = max(0.01, min(10.0, input_gain))
        self._auto_gain = auto_gain
        # Auto-gain state
        self._target_rms = 0.15  # Target RMS level
        self._gain_adjustment_factor = 1.2  # How aggressively to adjust
        self._min_gain = 0.05
        self._max_gain = 2.0
        self._model: WhisperModel | None = None

    def _get_model_path(self) -> Path:
        """Get the local model path."""
        return self._model_dir / self._model_size

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
            # Let faster-whisper handle its own Hugging Face cache
            # download_root tells it where to cache; it will reuse existing cache
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

            # Auto-gain adjustment based on RMS level
            rms = float(np.sqrt(np.mean(recording**2)))
            if self._auto_gain and rms > 0:
                # Calculate gain adjustment to reach target RMS
                ratio = self._target_rms / rms
                # Limit adjustment factor to prevent wild swings
                ratio = max(
                    1.0 / self._gain_adjustment_factor, min(self._gain_adjustment_factor, ratio)
                )
                new_gain = self._input_gain * ratio
                new_gain = max(self._min_gain, min(self._max_gain, new_gain))
                if abs(new_gain - self._input_gain) > 0.01:
                    logger.info(
                        f"Auto-gain: adjusting gain from {self._input_gain:.3f} "
                        f"to {new_gain:.3f} (RMS={rms:.3f})"
                    )
                    self._input_gain = new_gain

            # Apply input gain to prevent clipping
            if self._input_gain != 1.0:
                recording = recording * self._input_gain
                # Clip to prevent overflow
                recording = np.clip(recording, -1.0, 1.0)
                clipped_samples = np.sum(np.abs(recording) >= 1.0)
                if clipped_samples > 0:
                    logger.warning(
                        f"Audio clipping detected: {clipped_samples}/{len(recording)} "
                        f"samples clipped"
                    )

            # Ensure correct shape (n_samples, 1) for mono
            if recording.ndim == 1:
                recording = recording.reshape(-1, 1)

            logger.info(f"Recorded {len(recording)} frames")
            return np.asarray(recording, dtype=np.float32)  # type: ignore[return-value]
        except Exception as e:
            logger.error(f"Failed to record audio: {e}")
            raise STTError(f"Failed to record audio: {e}") from e

    def test_microphone(
        self,
        duration: float = 3.0,
        gain: float | None = None,
    ) -> dict:
        """Test microphone and report audio quality metrics.

        Args:
            duration: Recording duration in seconds
            gain: Optional input gain override (uses config default if None)

        Returns:
            Dictionary with audio quality metrics and transcription
        """
        # Temporarily override gain if provided
        original_gain = self._input_gain
        if gain is not None:
            self._input_gain = max(0.01, min(10.0, gain))

        try:
            logger.info(f"Testing microphone for {duration}s...")
            audio = self.record_audio(duration)

            # Calculate metrics
            rms = float(np.sqrt(np.mean(audio**2)))
            peak = float(np.max(np.abs(audio)))
            clipped = int(np.sum(np.abs(audio) >= 1.0))
            total = len(audio)
            clipping_pct = (clipped / total * 100) if total > 0 else 0.0

            # Transcribe
            text, lang = self.transcribe(audio)

            # Determine quality assessment
            if clipping_pct > 5.0:
                assessment = "❌ High clipping - lower microphone volume"
                suggested_gain = max(0.05, self._input_gain * 0.5)
            elif clipping_pct > 1.0:
                assessment = "⚠️ Some clipping - consider lowering volume"
                suggested_gain = max(0.05, self._input_gain * 0.7)
            elif rms < 0.01:
                assessment = "⚠️ Very low signal - increase microphone volume"
                suggested_gain = min(2.0, self._input_gain * 2.0)
            else:
                assessment = "✅ Audio levels look good!"
                suggested_gain = self._input_gain

            return {
                "duration": duration,
                "rms": rms,
                "peak": peak,
                "clipped_samples": clipped,
                "total_samples": total,
                "clipping_percentage": clipping_pct,
                "rms_level": rms,
                "detected_language": lang,
                "transcription": text,
                "assessment": assessment,
                "suggested_gain": suggested_gain,
            }
        except Exception as e:
            logger.error(f"Microphone test failed: {e}")
            return {
                "error": str(e),
                "duration": duration,
            }
        finally:
            # Restore original gain
            self._input_gain = original_gain

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
                # If probability is low, return empty text (no fallback retranscribe)
                # LLM will handle "couldn't hear" response
                if prob < self._language_detection_threshold:
                    logger.info(
                        f"Language detection confidence too low ({prob:.2f}), returning empty"
                    )
                    return "", lang

            # Concatenate all segment texts
            text = " ".join(segment.text for segment in segments).strip()

            if not text:
                logger.warning("Transcription returned empty text")

            return text, lang
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise STTError(f"Transcription failed: {e}") from e
