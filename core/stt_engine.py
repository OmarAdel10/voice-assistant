"""Voice Assistant - Speech-to-text engine contract and factory."""

from __future__ import annotations

import logging
import os
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from config.settings import STTConfig, STTProvider
from core.exceptions import STTError

logger = logging.getLogger(__name__)

_UNSET_PROMPT = object()


class STTEngineBase(Protocol):
    """Protocol shared by the configured speech-to-text engine."""

    def record_audio(self, duration: float) -> NDArray[np.float32]: ...

    def transcribe(
        self, audio: NDArray[np.float32], initial_prompt: Any = _UNSET_PROMPT
    ) -> tuple[str, str | None]: ...

    def test_microphone(self, duration: float = 3.0, gain: float | None = None) -> dict: ...

    def set_input_gain(self, gain: float) -> None: ...

    def set_auto_gain(self, enabled: bool) -> None: ...

    def set_language(self, language: str | None) -> None: ...

    def close(self) -> None: ...


def create_stt_engine(config: STTConfig) -> STTEngineBase:
    """Create the configured cloud STT engine.

    Gemini is the only supported provider. Missing credentials and invalid
    provider values fail explicitly instead of selecting a local backend.
    """
    if config.provider != STTProvider.GEMINI:
        raise STTError(f"Unsupported STT provider: {config.provider}")

    api_key = config.gemini_api_key or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise STTError(
            "Gemini API key not configured. Set gemini_api_key in config or GEMINI_API_KEY env var."
        )

    from core.stt_gemini import GeminiSTTEngine

    return GeminiSTTEngine(config)
