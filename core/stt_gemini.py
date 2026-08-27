"""Voice Assistant — Gemini Cloud STT Engine (Gemini 3.5 Transcribe)."""

from __future__ import annotations

import io
import logging
import os
import socket
import time
import wave
from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray

from config.settings import STTConfig
from core.exceptions import STTError

logger = logging.getLogger(__name__)

# Sentinel distinguishing "no override" from an explicit None in transcribe().
_UNSET_PROMPT = object()


def _prefer_ipv4() -> None:
    """Make ``socket.getaddrinfo`` return IPv4-only addresses.

    Hosts without an IPv6 route (e.g. WiFi clients that resolve -AAAA records
    but have no IPv6 default gateway) still advertise IPv6 addresses. The
    google-genai SDK's httpx/httpcore stack may then dial the unroutable IPv6
    address and abort with ``[Errno 101] Network is unreachable`` instead of
    falling back to IPv4, making cloud transcription intermittently fail.

    Reordering the lookup is insufficient because httpcore applies its own
    RFC 6724 preference order. Filtering to IPv4 addresses is deterministic and
    is safe on this class of host (IPv6 is unreachable anyway). If a host only
    offers IPv6, the original results are returned untouched so lookups still
    work there.
    """

    def _wrap(original: Callable[..., Any]) -> Callable[..., Any]:
        def getaddrinfo(*args: Any, **kwargs: Any) -> Any:
            results = original(*args, **kwargs)
            ipv4 = [entry for entry in results if entry[0] == socket.AF_INET]
            return ipv4 if ipv4 else results

        return getaddrinfo

    if getattr(socket.getaddrinfo, "_va_ipv4_only", False):
        return
    socket.getaddrinfo = _wrap(socket.getaddrinfo)  # type: ignore[assignment]
    socket.getaddrinfo._va_ipv4_only = True  # type: ignore[attr-defined]


class GeminiSTTEngine:
    """STT engine using Gemini 3.5 Transcribe via Google GenAI SDK."""

    def __init__(self, settings: STTConfig) -> None:
        """Initialize Gemini STT engine.

        Args:
            settings: STT configuration with Gemini settings
        """
        api_key = settings.gemini_api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise STTError(
                "Gemini API key not configured. Set gemini_api_key in config "
                "or GEMINI_API_KEY env var."
            )

        self._api_key = api_key
        self._model = settings.gemini_model
        self._base_url = settings.gemini_base_url
        self._timeout = settings.gemini_timeout
        self._forced_language = settings.gemini_language
        self._allowed_languages = settings.allowed_languages
        self._language_threshold = settings.language_detection_threshold

        # Audio settings (shared with local engine)
        self._sample_rate = 16000
        self._input_gain = max(0.01, min(10.0, settings.input_gain))
        self._auto_gain = settings.auto_gain
        self._target_rms = 0.15
        self._gain_adjustment_factor = 1.2
        self._min_gain = 0.05
        self._max_gain = 2.0

        # Initialize GenAI client
        _prefer_ipv4()
        from google import genai

        client_kwargs = {"api_key": self._api_key}
        # The GenAI SDK's HttpOptions.timeout is in *milliseconds*, while the
        # configuration value is in seconds. Convert and merge into a single
        # http_options dict (base_url + timeout must coexist, not overwrite).
        http_options: dict[str, Any] = {}
        if self._base_url:
            http_options["base_url"] = self._base_url
        if self._timeout:
            http_options["timeout"] = int(self._timeout * 1000)
        if http_options:
            client_kwargs["http_options"] = http_options

        self._client = genai.Client(**client_kwargs)

    def set_input_gain(self, gain: float) -> None:
        """Update input gain at runtime (clamped to safe bounds)."""
        clamped = max(0.01, min(10.0, gain))
        if clamped != self._input_gain:
            logger.info(f"Gemini STT input gain updated: {self._input_gain:.3f} -> {clamped:.3f}")
        self._input_gain = clamped

    def set_auto_gain(self, enabled: bool) -> None:
        """Enable or disable automatic gain adjustment at runtime."""
        if enabled != self._auto_gain:
            logger.info(f"Gemini STT auto-gain {'enabled' if enabled else 'disabled'}")
        self._auto_gain = enabled

    def set_language(self, language: str | None) -> None:
        """Update forced transcription language at runtime."""
        if language != self._forced_language:
            logger.info(
                f"Gemini STT forced language updated: {self._forced_language} -> {language}"
            )
        self._forced_language = language

    def _encode_wav(self, audio: NDArray[np.float32]) -> bytes:
        """Encode float32 numpy audio as 16-bit PCM WAV bytes."""
        # Ensure mono
        if audio.ndim == 2 and audio.shape[1] == 1:
            audio = audio.flatten()
        elif audio.ndim == 2:
            audio = audio[:, 0]

        # Apply gain
        if self._input_gain != 1.0:
            audio = audio * self._input_gain
            audio = np.clip(audio, -1.0, 1.0)

        # Convert float32 [-1, 1] to int16
        audio_int16 = (audio * 32767).astype(np.int16)

        # Write WAV to bytes
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(self._sample_rate)
            wf.writeframes(audio_int16.tobytes())
        return buffer.getvalue()

    def record_audio(self, duration: float) -> NDArray[np.float32]:
        """Record audio from microphone.

        Args:
            duration: Recording duration in seconds

        Returns:
            Audio data as float32 numpy array (mono, 16kHz)
        """
        try:
            import sounddevice as sd  # type: ignore[import-untyped]
        except ImportError as e:
            raise STTError("sounddevice not installed") from e

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
            ratio = self._target_rms / rms
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

        # Ensure correct shape (n_samples, 1) for mono
        if recording.ndim == 1:
            recording = recording.reshape(-1, 1)

        logger.info(f"Recorded {len(recording)} frames")
        return np.asarray(recording, dtype=np.float32)

    def test_microphone(
        self,
        duration: float = 3.0,
        gain: float | None = None,
    ) -> dict[str, Any]:
        """Test microphone and report audio quality metrics."""
        original_gain = self._input_gain
        if gain is not None:
            self._input_gain = max(0.01, min(10.0, gain))

        try:
            logger.info(f"Testing microphone for {duration}s...")
            audio = self.record_audio(duration)

            rms = float(np.sqrt(np.mean(audio**2)))
            peak = float(np.max(np.abs(audio)))
            clipped = int(np.sum(np.abs(audio) >= 1.0))
            total = len(audio)
            clipping_pct = (clipped / total * 100) if total > 0 else 0.0

            text, lang = self.transcribe(audio)

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
            return {"error": str(e), "duration": duration}
        finally:
            self._input_gain = original_gain

    def _map_language_code(self, lang: str | None) -> str | None:
        """Map language codes to BCP-47 format for Gemini."""
        if not lang:
            return None
        lang_map = {
            "ar": "ar-EG",
            "en": "en-US",
            "ar-EG": "ar-EG",
            "en-US": "en-US",
        }
        return lang_map.get(lang, lang)

    def _get_language_codes(self) -> list[str] | None:
        """Get language codes for Gemini API based on config."""
        if self._forced_language:
            mapped = self._map_language_code(self._forced_language)
            return [mapped] if mapped else None

        # Use allowed_languages as hints
        if self._allowed_languages:
            mapped = [self._map_language_code(code) for code in self._allowed_languages]
            return [code for code in mapped if code]

        return None

    def transcribe(
        self,
        audio: NDArray[np.float32],
        initial_prompt: Any = _UNSET_PROMPT,
    ) -> tuple[str, str | None]:
        """Transcribe audio to text using Gemini 3.5 Transcribe API (Interactions API).

        Args:
            audio: Audio data as float32 numpy array (mono, 16kHz)
            initial_prompt: Optional speaker adaptation prompt (enrollment)

        Returns:
            Tuple of (transcribed_text, detected_language_code)

        Raises:
            STTError: If transcription fails
        """
        if audio.size == 0:
            logger.warning("Empty audio provided for transcription")
            return "", None

        try:
            start_time = time.perf_counter()

            # Encode audio as WAV
            wav_bytes = self._encode_wav(audio)

            # Upload audio file to Files API
            import io

            from google.genai import types

            audio_file = io.BytesIO(wav_bytes)
            audio_file.name = "audio.wav"

            uploaded_file = self._client.files.upload(
                file=audio_file,
                config=types.UploadFileConfig(mime_type="audio/wav"),
            )

            # Prepare interaction input with transcription config
            language_codes = self._get_language_codes()

            # The genai SDK's Interactions API expects raw content parts,
            # each discriminated by a "type" field (not the legacy
            # {"role": "user", "content": [...]} chat envelope).
            input_content = {
                "type": "audio",
                "uri": uploaded_file.uri,
                "mime_type": uploaded_file.mime_type,
            }

            generation_config = {
                "transcription_config": {
                    "language_codes": language_codes,
                    "mode": {"type": "smart"},
                }
            }

            # Retry on rate-limit (429) errors with exponential backoff
            max_attempts = 3
            interaction = None
            for attempt in range(1, max_attempts + 1):
                try:
                    interaction = self._client.interactions.create(
                        model=self._model,
                        input=[input_content],
                        generation_config=generation_config,
                    )
                    break
                except Exception as create_err:
                    error_str = str(create_err).lower()
                    is_rate_limit = (
                        "rate limit" in error_str
                        or "quota" in error_str
                        or "429" in error_str
                        or type(create_err).__name__ == "ResourceExhausted"
                    )
                    if is_rate_limit and attempt < max_attempts:
                        logger.warning(
                            f"Gemini rate limit (attempt {attempt}/{max_attempts}), retrying..."
                        )
                        time.sleep(min(2.0**attempt, 8.0))
                        continue
                    raise

            assert interaction is not None
            elapsed_ms = (time.perf_counter() - start_time) * 1000

            # Extract text from response
            text = interaction.output_text.strip() if interaction.output_text else ""

            # Language detection: use forced language or default to first allowed language
            lang = None
            if self._forced_language:
                lang = self._forced_language
            elif self._allowed_languages:
                lang = self._allowed_languages[0]

            logger.info(f"Gemini STT: {elapsed_ms:.0f}ms | Language: {lang} | Text: {text[:50]}...")

            # Validate detected language against allowed list
            if self._allowed_languages and lang and lang not in self._allowed_languages:
                logger.warning(
                    f"Detected language '{lang}' not in allowed languages {self._allowed_languages}"
                )

            if not text:
                logger.warning("Transcription returned empty text")

            return text, lang

        except Exception as e:
            logger.error(f"Gemini transcription failed: {e}")
            error_str = str(e).lower()
            if "rate limit" in error_str or "quota" in error_str or "429" in error_str:
                raise STTError("Gemini API rate limit exceeded") from e
            raise STTError(f"Gemini transcription failed: {e}") from e

    def close(self) -> None:
        """Close the engine (cleanup if needed)."""
        # GenAI client doesn't need explicit close
        pass

    def __enter__(self) -> GeminiSTTEngine:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
