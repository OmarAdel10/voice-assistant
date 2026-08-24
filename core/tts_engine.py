"""Voice Assistant — Text-to-Speech Engine."""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Literal

import numpy as np
import sounddevice as sd
from piper import PiperVoice

logger = logging.getLogger(__name__)


class TTSEngine:
    """Text-to-Speech engine with fallback chain: piper -> piper(fallback) -> pyttsx3 -> print."""

    def __init__(
        self,
        rate: int = 180,
        volume: float = 0.9,
        engine: Literal["piper", "pyttsx3"] = "piper",
        voice_id: str | None = None,
        piper_voice_dir: str = "models/tts",
        piper_voice_ar: str = "ar_JO-kareem-medium",
        piper_voice_en: str = "en_US-lessac-medium",
        piper_voice_ar_fallback: str = "ar_JO-kareem-low",
        piper_voice_en_fallback: str = "en_US-lessac-low",
    ) -> None:
        """Initialize TTS engine.

        Args:
            rate: Speech rate (words per minute)
            volume: Volume level (0.0 - 1.0)
            engine: Preferred engine ("piper" or "pyttsx3")
            voice_id: Voice identifier for pyttsx3 (None = system default)
            piper_voice_dir: Directory containing piper voices
            piper_voice_ar: Arabic piper voice name
            piper_voice_en: English piper voice name
            piper_voice_ar_fallback: Arabic piper fallback voice name
            piper_voice_en_fallback: English piper fallback voice name
        """
        self._rate = rate
        self._volume = volume
        self._engine_preference = engine
        self._voice_id = voice_id
        self._piper_voice_dir = Path(piper_voice_dir)
        self._piper_voices = {
            "ar": piper_voice_ar,
            "en": piper_voice_en,
            "ar_fallback": piper_voice_ar_fallback,
            "en_fallback": piper_voice_en_fallback,
        }

    def say(self, text: str, lang: str = "auto") -> None:
        """Speak text using fallback chain.

        Args:
            text: Text to synthesize
            lang: Language code ("ar", "en", "auto" for auto-detect)
        """
        start_time = time.perf_counter()

        # Determine language
        if lang == "auto":
            lang = self._detect_language(text)

        try:
            if self._engine_preference == "piper":
                self._say_piper(text, lang)
            else:
                self._say_pyttsx3(text)
        except Exception as e:
            logger.warning(f"Primary TTS failed: {e}, trying fallback")
            try:
                if self._engine_preference == "piper":
                    self._say_pyttsx3(text)
                else:
                    self._say_piper(text, lang)
            except Exception as e2:
                logger.error(f"Fallback TTS failed: {e2}, using print fallback")
                self._say_fallback(text)

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        logger.info(f"TTS: {elapsed_ms:.0f}ms | Text length: {len(text)} chars")

    def _detect_language(self, text: str) -> str:
        """Simple language detection: Arabic script -> ar, else en."""
        for char in text:
            if "\u0600" <= char <= "\u06ff" or "\u0750" <= char <= "\u077f":
                return "ar"
        return "en"

    def _say_piper(self, text: str, lang: str) -> None:
        """Speak using piper-tts (offline, natural) with fallback voice support."""
        # Try primary voice first
        primary_voice = self._piper_voices.get(lang, self._piper_voices["en"])
        fallback_voice = self._piper_voices.get(
            f"{lang}_fallback", self._piper_voices.get("en_fallback")
        )

        for voice_name in (primary_voice, fallback_voice):
            if not voice_name:
                continue
            voice_path = self._piper_voice_dir / voice_name / f"{voice_name}.onnx"
            if not voice_path.exists():
                logger.warning(f"Piper voice not found: {voice_path}, trying fallback")
                continue

            try:
                logger.info(f"Using piper voice: {voice_name} for lang: {lang}")
                voice = PiperVoice.load(str(voice_path))

                # Synthesize audio
                audio_chunks = []
                for chunk in voice.synthesize(text):
                    audio_chunks.append(chunk.audio_int16_bytes)

                # Combine and play
                audio_data = b"".join(audio_chunks)
                audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

                sd.play(audio_array, samplerate=voice.config.sample_rate)
                sd.wait()
                return  # Success

            except Exception as e:
                logger.warning(f"Piper voice {voice_name} failed: {e}, trying fallback")
                continue

        # If both primary and fallback failed, raise to trigger engine fallback
        raise RuntimeError("All piper voices failed")

    def _say_pyttsx3(self, text: str) -> None:
        """Speak using pyttsx3 (offline)."""
        import pyttsx3

        engine = pyttsx3.init()
        engine.setProperty("rate", self._rate)
        engine.setProperty("volume", self._volume)

        if self._voice_id:
            engine.setProperty("voice", self._voice_id)

        engine.say(text)
        engine.runAndWait()

    def _say_gtts(self, text: str) -> None:
        """Speak using gTTS (online) - kept for compatibility."""
        from gtts import gTTS

        lang = self._detect_language(text)
        tts = gTTS(text=text, lang=lang, slow=False)

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as fp:
            tts.write_to_fp(fp)
            temp_path = fp.name

        try:
            players = [
                ["mpv", "--no-video", "--really-quiet"],
                ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet"],
                ["aplay"],
                ["paplay"],
            ]

            played = False
            for player_cmd in players:
                if shutil.which(player_cmd[0]):
                    try:
                        subprocess.run(player_cmd + [temp_path], check=True, timeout=30)
                        played = True
                        break
                    except (subprocess.SubprocessError, subprocess.TimeoutExpired):
                        continue

            if not played:
                logger.warning("No audio player found for gTTS playback")
                raise RuntimeError("No audio player available")

        finally:
            try:
                import os

                os.unlink(temp_path)
            except OSError:
                pass

    def _say_fallback(self, text: str) -> None:
        """Last resort: print to stdout."""
        print(f"[TTS fallback] {text}")  # noqa: T201
