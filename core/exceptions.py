"""Voice Assistant — Core exceptions module."""

from __future__ import annotations


class VoiceAssistantError(Exception):
    """Base exception for all Voice Assistant errors."""


class STTError(VoiceAssistantError):
    """Speech-to-Text engine errors."""


class NLPError(VoiceAssistantError):
    """Natural Language Processing engine errors."""


class TTSError(VoiceAssistantError):
    """Text-to-Speech engine errors."""


class ActionError(VoiceAssistantError):
    """Action execution errors."""


class AppNotFoundError(ActionError):
    """Application not found in system."""


class InstallError(ActionError):
    """Package installation errors."""
