"""Voice Assistant — CLI Entry Point."""

from __future__ import annotations

import logging
import signal
import sys
from pathlib import Path

import click

from config.settings import Settings
from core.actions import get_date, get_sysinfo, get_time, open_app, web_search
from core.exceptions import ActionError, NLPError, STTError, TTSError
from core.nlp_engine import NLPEngine
from core.stt_engine import STTEngine
from core.tts_engine import TTSEngine

logger = logging.getLogger(__name__)


class VoiceAssistant:
    """Main Voice Assistant application class."""

    def __init__(self, settings: Settings) -> None:
        """Initialize Voice Assistant with settings.

        Args:
            settings: Application settings
        """
        self.settings = settings
        self._running = False

        # Initialize engines lazily
        self._stt_engine: STTEngine | None = None
        self._nlp_engine: NLPEngine | None = None
        self._tts_engine: TTSEngine | None = None

    @property
    def stt_engine(self) -> STTEngine:
        """Lazy STT engine initialization."""
        if self._stt_engine is None:
            self._stt_engine = STTEngine(
                model_size=self.settings.stt.model_size,
                device=self.settings.stt.device,
                compute_type=self.settings.stt.compute_type,
                sample_rate=self.settings.audio.sample_rate,
            )
        return self._stt_engine

    @property
    def nlp_engine(self) -> NLPEngine:
        """Lazy NLP engine initialization."""
        if self._nlp_engine is None:
            self._nlp_engine = NLPEngine()
        return self._nlp_engine

    @property
    def tts_engine(self) -> TTSEngine:
        """Lazy TTS engine initialization."""
        if self._tts_engine is None:
            self._tts_engine = TTSEngine(
                rate=self.settings.tts.rate,
                volume=self.settings.tts.volume,
                engine=self.settings.tts.engine,
                lang=self.settings.tts.language,
            )
        return self._tts_engine

    def _safe_say(self, text: str) -> None:
        """Safely speak text, suppressing any TTS errors."""
        try:
            self.tts_engine.say(text)
        except Exception as e:
            logger.error(f"TTS fallback failed: {e}")
            logger.error(f"[TTS Error] {e}")

    def process_text(self, text: str) -> str:
        """Process text input through NLP and execute action.

        Args:
            text: User input text

        Returns:
            Response text to speak
        """
        try:
            intent, entities, confidence = self.nlp_engine.parse(text)

            if intent == "unknown" or confidence < self.settings.nlp.confidence_threshold:
                return (
                    "I didn't understand that command. Try asking for time, date, "
                    "system info, opening an app, or searching the web."
                )

            logger.info(f"Intent: {intent} | Entities: {entities} | Confidence: {confidence:.2f}")

            # Execute action based on intent
            if intent == "get_time":
                return get_time()
            elif intent == "get_date":
                return get_date()
            elif intent == "get_sys_info":
                info = get_sysinfo()
                cpu = info["cpu_percent"]
                mem = info["memory_percent"]
                disk = info["disk_percent"]
                return f"CPU: {cpu:.1f}% | Memory: {mem:.1f}% | Disk: {disk:.1f}%"
            elif intent == "open_app":
                return open_app(entities["app"])
            elif intent == "web_search":
                return web_search(entities["query"])
            else:
                return f"Unknown intent: {intent}"

        except (NLPError, ActionError) as e:
            logger.error(f"Processing failed: {e}")
            return f"Error: {e}"
        except Exception as e:
            logger.exception(f"Unexpected error: {e}")
            return "An unexpected error occurred"

    def run_once_voice(self) -> None:
        """Single voice interaction: listen -> transcribe -> process -> speak."""
        try:
            logger.info("Listening...")
            audio = self.stt_engine.record_audio(self.settings.stt.max_listen_seconds)

            logger.info("Transcribing...")
            text = self.stt_engine.transcribe(audio)

            if not text:
                logger.warning("No speech detected")
                return

            logger.info(f"Transcribed: {text}")

            response = self.process_text(text)

            logger.info(f"Response: {response}")
            self.tts_engine.say(response)

        except STTError as e:
            logger.error(f"STT error: {e}")
            self._safe_say("Could not understand audio. Please try again.")
        except TTSError as e:
            logger.error(f"TTS error: {e}")
            logger.error(f"[TTS Error] {e}")
        except Exception as e:
            logger.exception(f"Unexpected error in voice interaction: {e}")
            self._safe_say("An error occurred. Please try again.")

    def run_once_text(self, text: str) -> None:
        """Single text interaction: process -> speak."""
        try:
            response = self.process_text(text)
            logger.info(f"Response: {response}")
            self.tts_engine.say(response)
        except TTSError as e:
            logger.error(f"TTS error: {e}")
            logger.error(f"[TTS Error] {e}")
        except Exception as e:
            logger.exception(f"Unexpected error in text interaction: {e}")
            self.tts_engine.say("An error occurred. Please try again.")

    def run_listen_loop(self) -> None:
        """Continuous voice loop until interrupted."""
        self._running = True
        logger.info("Starting voice loop. Press Ctrl+C to exit.")

        def signal_handler(signum, frame):
            logger.info("Shutdown signal received")
            self._running = False

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        while self._running:
            try:
                self.run_once_voice()
            except KeyboardInterrupt:
                logger.info("Interrupted")
                break
            except Exception as e:
                logger.exception(f"Error in voice loop: {e}")
                # Continue loop on error

        logger.info("Voice assistant stopped")


@click.command()
@click.option(
    "--listen",
    "-l",
    "mode",
    flag_value="listen",
    default=True,
    help="Continuous listening mode (default)",
)
@click.option("--once", "mode", flag_value="once", help="Single interaction mode")
@click.option("--text", "text_input", type=str, default=None, help="Text input mode (bypasses STT)")
@click.option(
    "--list-intents",
    "list_intents",
    is_flag=True,
    default=False,
    help="List available intents and exit",
)
@click.option("--version", is_flag=True, default=False, help="Show version and exit")
@click.help_option("-h", "--help")
def cli(mode: str, text_input: str | None, list_intents: bool, version: bool) -> None:
    """Voice Assistant - Offline-first voice assistant for academic field training.

    Examples:
      voice-assistant              # Start continuous listening mode
      voice-assistant --once       # Single voice interaction
      voice-assistant --once --text "what time is it"  # Text input
      voice-assistant --list-intents  # Show available commands
    """
    if version:
        click.echo("Voice Assistant 0.1.0")
        return

    # Load settings FIRST (for logging config)
    config_path = Path("config.yaml")
    settings = Settings.load(config_path)

    # Configure logging
    logging.basicConfig(
        level=settings.log.level,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if list_intents:
        nlp = NLPEngine()
        click.echo("Available intents:")
        for intent in nlp._intents:
            name = intent["name"]
            patterns = ", ".join(f'"{p}"' for p in intent["patterns"][:3])
            if len(intent["patterns"]) > 3:
                patterns += "..."
            click.echo(f"  {name}: {patterns}")
        return

    # Create and run assistant
    assistant = VoiceAssistant(settings)

    if mode == "listen":
        assistant.run_listen_loop()
    elif mode == "once":
        if text_input:
            assistant.run_once_text(text_input)
        else:
            assistant.run_once_voice()
    else:
        click.echo(f"Unknown mode: {mode}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    cli()
