"""Voice Assistant — Configuration settings."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from core.exceptions import VoiceAssistantError


class STTConfig(BaseModel):
    """Speech-to-Text configuration."""

    model_config = {"frozen": True}

    model_size: str = Field(default="large-v3", description="Whisper model size")
    language: str = Field(default="en", description="Language code")
device: str = Field(default="cuda", description="Device to run on (cpu, cuda)")
    compute_type: str = Field(
        default="float16", description="Quantization type (int8, float16, float32)"
    )
    vad_threshold: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Voice activity detection threshold"
    )
    max_listen_seconds: int = Field(default=5, gt=0, description="Maximum recording duration")


class TTSConfig(BaseModel):
    """Text-to-Speech configuration."""

    model_config = {"frozen": True}

    engine: Literal["pyttsx3", "gTTS"] = Field(default="pyttsx3", description="TTS engine")
    rate: int = Field(default=180, gt=0, description="Speech rate (words per minute)")
    volume: float = Field(default=0.9, ge=0.0, le=1.0, description="Volume level")
    language: str = Field(default="en", description="Language code for gTTS")


class NLPConfig(BaseModel):
    """Natural Language Processing configuration."""

    model_config = {"frozen": True}

    confidence_threshold: float = Field(
        default=0.6, ge=0.0, le=1.0, description="Intent matching confidence threshold"
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
