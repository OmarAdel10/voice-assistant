"""Voice Assistant — Configuration settings."""

from __future__ import annotations

import os
from enum import StrEnum
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from core.exceptions import VoiceAssistantError
from core.paths import PROJECT_ROOT


def _anchor_path(value: object) -> object:
    """Resolve a relative path against the project root.

    Config files may specify model locations as project-relative strings;
    anchoring them at load time keeps the CLI correct from any CWD.
    """
    if isinstance(value, str) and value and not Path(value).is_absolute():
        return str(PROJECT_ROOT / value)
    return value


def _load_dotenv(dotenv_path: Path) -> None:
    """Populate ``os.environ`` from a ``.env`` file, anchored at project root.

    Secrets (e.g. ``GEMINI_API_KEY``) must reach the process no matter who
    launches it — an interactive shell, ``python -m``, or a subprocess spawned
    by the Flutter GUI (which does not inherit a developer's shell exports).
    Reading a fixed, absolute path makes the key available in every case.

    Real environment variables always win: existing entries are left in
    place via ``setdefault``. Missing files, comments, and blank lines are
    silently ignored; this is intentionally minimal (no third-party
    dependency) and only supports simple ``KEY=VALUE`` lines.
    """
    if not dotenv_path.is_file():
        return

    try:
        lines = dotenv_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return

    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


class STTProvider(StrEnum):
    """STT provider selection."""

    LOCAL = "local"
    GEMINI = "gemini"


class STTConfig(BaseModel):
    """Speech-to-Text configuration."""

    model_config = {"frozen": True}

    @field_validator("model_dir", mode="before")
    @classmethod
    def _anchor_model_dir(cls, v: object) -> object:
        return _anchor_path(v)

    model_size: str = Field(default="large-v3", description="Whisper model size")
    language: str | None = Field(
        default=None,
        description="Language code (ar for Arabic, en for English, None for auto-detect)",
    )
    allowed_languages: list[str] = Field(
        default_factory=lambda: ["ar", "en"],
        description="Allowed language codes for transcription (restricts detection)",
    )
    language_detection_threshold: float = Field(
        default=0.7, ge=0.0, le=1.0, description="Minimum probability for language detection"
    )
    device: str = Field(default="cuda", description="Device to run on (cpu, cuda)")
    compute_type: str = Field(
        default="float16", description="Quantization type (int8, float16, float32)"
    )
    model_dir: str = Field(
        default=str(PROJECT_ROOT / "models" / "stt"),
        description="Local model directory",
    )
    vad_threshold: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Voice activity detection threshold"
    )
    max_listen_seconds: int = Field(default=5, gt=0, description="Maximum recording duration")
    vad_filter: bool = Field(default=True, description="Enable VAD filtering")
    vad_min_silence_ms: int = Field(default=500, gt=0, description="Min silence duration for VAD")
    initial_prompt: str | None = Field(
        default=None, description="Speaker adaptation prompt from voice enrollment"
    )
    input_gain: float = Field(
        default=1.0, ge=0.01, le=10.0, description="Input volume gain (0.1-1.0 to prevent clipping)"
    )
    auto_gain: bool = Field(
        default=False, description="Automatically adjust input gain based on audio levels"
    )
    offline: bool = Field(
        default=False,
        description="Require a locally cached whisper snapshot; never contact the Hub",
    )

    # Provider selection
    provider: STTProvider = Field(
        default=STTProvider.GEMINI, description="STT provider (local or gemini)"
    )

    # Gemini Cloud STT settings
    gemini_api_key: str | None = Field(
        default=None, description="Gemini API key (or set GEMINI_API_KEY env var)"
    )
    gemini_model: str = Field(
        default="gemini-3.5-transcribe", description="Gemini Transcribe model ID"
    )
    gemini_base_url: str | None = Field(
        default=None, description="Vertex AI override endpoint (optional)"
    )
    gemini_timeout: float = Field(
        default=30.0, gt=0, description="Gemini API request timeout in seconds"
    )
    gemini_language: str | None = Field(
        default=None,
        description="Force language for Gemini (BCP-47, e.g., en-US, ar-EG), None=auto-detect",
    )


class TTSConfig(BaseModel):
    """Text-to-Speech configuration."""

    model_config = {"frozen": True}

    @field_validator("piper_voice_dir", mode="before")
    @classmethod
    def _anchor_piper_voice_dir(cls, v: object) -> object:
        return _anchor_path(v)

    engine: Literal["piper", "pyttsx3"] = Field(default="piper", description="TTS engine")
    rate: int = Field(default=180, gt=0, description="Speech rate (words per minute)")
    volume: float = Field(default=0.9, ge=0.0, le=1.0, description="Volume level")
    voice_id: str | None = Field(default=None, description="Voice ID for pyttsx3")
    piper_voice_dir: str = Field(
        default=str(PROJECT_ROOT / "models" / "tts"),
        description="Local piper voices directory",
    )
    piper_voice_ar: str = Field(default="ar_JO-kareem-medium", description="Arabic piper voice")
    piper_voice_en: str = Field(default="en_US-lessac-medium", description="English piper voice")
    piper_voice_ar_fallback: str = Field(
        default="ar_JO-kareem-low", description="Arabic piper voice fallback"
    )
    piper_voice_en_fallback: str = Field(
        default="en_US-lessac-low", description="English piper voice fallback"
    )


class NLPConfig(BaseModel):
    """Natural Language Processing configuration."""

    model_config = {"frozen": True}

    confidence_threshold: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Intent matching confidence threshold (default for all languages)",
    )
    confidence_threshold_ar: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Arabic intent matching confidence threshold (lower for dialect support)",
    )
    confidence_threshold_en: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="English intent matching confidence threshold",
    )


class AudioConfig(BaseModel):
    """Audio I/O configuration."""

    model_config = {"frozen": True}

    sample_rate: int = Field(default=16000, gt=0, description="Audio sample rate")
    channels: int = Field(default=1, gt=0, description="Number of audio channels")


class LogConfig(BaseModel):
    """Logging configuration."""

    model_config = {"frozen": True}

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO", description="Log level"
    )


class LLMConfig(BaseModel):
    """LLM configuration for intent parsing and response generation."""

    model_config = {"frozen": True}

    @field_validator("model_path", mode="before")
    @classmethod
    def _anchor_model_path(cls, v: object) -> object:
        return _anchor_path(v)

    model_path: str = Field(
        default=str(PROJECT_ROOT / "models" / "llm" / "qwen2.5-1.5b-instruct-q4_k_m.gguf"),
        description="Path to GGUF model file",
    )
    enabled: bool = Field(default=True, description="Enable LLM intent parsing")
    fallback_to_nlp: bool = Field(default=True, description="Fall back to NLP regex on LLM failure")
    confidence_threshold: float = Field(
        default=0.7, ge=0.0, le=1.0, description="Minimum confidence for LLM intent"
    )
    max_tokens: int = Field(default=256, gt=0, description="Max tokens for generation")
    temperature: float = Field(default=0.1, ge=0.0, le=1.0, description="Sampling temperature")
    n_gpu_layers: int = Field(default=-1, description="GPU layers (-1 = all)")
    n_ctx: int = Field(default=4096, gt=0, description="Context window size")
    n_threads: int = Field(default=4, gt=0, description="CPU threads for inference")
    verbose: bool = Field(default=False, description="Verbose llama.cpp output")


class Settings(BaseSettings):
    """Application settings loaded from YAML with environment override."""

    model_config = SettingsConfigDict(
        env_prefix="VA_",
        env_nested_delimiter="__",
        extra="ignore",
        frozen=True,
    )

    stt: STTConfig = Field(default_factory=STTConfig)
    tts: TTSConfig = Field(default_factory=TTSConfig)
    nlp: NLPConfig = Field(default_factory=NLPConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    log: LogConfig = Field(default_factory=LogConfig)

    @classmethod
    def load(cls, config_path: Path | str) -> Settings:
        """Load settings from YAML file, falling back to defaults.

        Args:
            config_path: Path to config.yaml file

        Returns:
            Settings instance with merged config

        Raises:
            VoiceAssistantError: If YAML is invalid
        """
        _load_dotenv(PROJECT_ROOT / ".env")

        path = Path(config_path)

        if not path.exists():
            return cls()

        try:
            content = path.read_text(encoding="utf-8")
            data = yaml.safe_load(content) or {}
        except yaml.YAMLError as e:
            raise VoiceAssistantError(f"Invalid YAML in {config_path}: {e}") from e
        except OSError as e:
            raise VoiceAssistantError(f"Failed to read {config_path}: {e}") from e

        # Merge with defaults by creating instance with file data
        return cls(**data)
