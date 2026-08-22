"""Voice Assistant — Text-to-Speech Engine."""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
import time
from typing import Literal

logger = logging.getLogger(__name__)


class TTSEngine:
    """Text-to-Speech engine with fallback chain: pyttsx3 -> gTTS -> print."""

    def __init__(
        self,
        rate: int = 180,
        volume: float = 0.9,
        engine: Literal["pyttsx3", "gTTS"] = "pyttsx3",
        voice_id: str | None = None,
        lang: str = "en",
    ) -> None:
        """Initialize TTS engine.

        Args:
            rate: Speech rate (words per minute)
            volume: Volume level (0.0 - 1.0)
            engine: Preferred engine ("pyttsx3" or "gTTS")
            voice_id: Voice identifier for pyttsx3 (None = system default)
            lang: Language code for gTTS
        """
        self._rate = rate
        self._volume = volume
        self._engine_preference = engine
        self._voice_id = voice_id
        self._lang = lang

    def say(self, text: str) -> None:
        """Speak text using fallback chain.

        Args:
            text: Text to synthesize
        """
        start_time = time.perf_counter()

        try:
            if self._engine_preference == "pyttsx3":
                self._say_pyttsx3(text)
            else:
                self._say_gtts(text)
        except Exception as e:
            logger.warning(f"Primary TTS failed: {e}, trying fallback")
            try:
                if self._engine_preference == "pyttsx3":
                    self._say_gtts(text)
                else:
                    self._say_pyttsx3(text)
            except Exception as e2:
                logger.error(f"Fallback TTS failed: {e2}, using print fallback")
                self._say_fallback(text)

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        logger.info(f"TTS: {elapsed_ms:.0f}ms | Text length: {len(text)} chars")

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
        """Speak using gTTS (online)."""
        from gtts import gTTS

        tts = gTTS(text=text, lang=self._lang, slow=False)

        # Write to temp file and try to play with available system player
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as fp:
            tts.write_to_fp(fp)
            temp_path = fp.name

        try:
            # Try common Linux audio players in order
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
            # Clean up temp file
            try:
                import os

                os.unlink(temp_path)
            except OSError:
                pass

    def _say_fallback(self, text: str) -> None:
        """Last resort: print to stdout."""
        print(f"[TTS fallback] {text}")  # noqa: T201
