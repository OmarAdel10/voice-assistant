"""Voice Assistant — CLI Entry Point."""

from __future__ import annotations

import json
import logging
import signal
import sys
from pathlib import Path

import click

from config.settings import Settings
from core.actions import get_date, get_sysinfo, get_time, open_app, web_search
from core.exceptions import ActionError, NLPError, STTError, TTSError
from core.llm_engine import LLMEngine
from core.nlp_engine import NLPEngine
from core.stt_engine import STTEngine
from core.tts_engine import TTSEngine

logger = logging.getLogger(__name__)

# Enrollment file path
ENROLLMENT_DIR = Path.home() / ".config" / "voice-assistant"
ENROLLMENT_FILE = ENROLLMENT_DIR / "enrollment.json"

ENROLLMENT_PHRASES = [
    "مرحبا، كيف حالك؟",
    "افتح الكود من فضلك",
    "ما الوقت الآن؟",
    "Hello, how are you?",
    "Open the browser please",
    "What time is it?",
]


def load_enrollment() -> str | None:
    """Load enrollment prompt from file."""
    try:
        if ENROLLMENT_FILE.exists():
            data = json.loads(ENROLLMENT_FILE.read_text(encoding="utf-8"))
            return data.get("initial_prompt")
    except Exception as e:
        logger.warning(f"Failed to load enrollment: {e}")
    return None


def save_enrollment(prompt: str) -> None:
    """Save enrollment prompt to file."""
    try:
        ENROLLMENT_DIR.mkdir(parents=True, exist_ok=True)
        data = {"initial_prompt": prompt}
        ENROLLMENT_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info(f"Enrollment saved to {ENROLLMENT_FILE}")
    except Exception as e:
        logger.error(f"Failed to save enrollment: {e}")
        raise


class VoiceAssistant:
    """Main Voice Assistant application class."""

    def __init__(self, settings: Settings) -> None:
        """Initialize Voice Assistant with settings.

        Args:
            settings: Application settings
        """
        self.settings = settings
        self._running = False
        self._enrollment_prompt = load_enrollment()

        # Initialize engines lazily
        self._stt_engine: STTEngine | None = None
        self._nlp_engine: NLPEngine | None = None
        self._tts_engine: TTSEngine | None = None
        self._llm_engine: LLMEngine | None = None

    @property
    def stt_engine(self) -> STTEngine:
        """Lazy STT engine initialization."""
        if self._stt_engine is None:
            # Use enrollment prompt if available, otherwise fall back to config
            initial_prompt = self._enrollment_prompt or self.settings.stt.initial_prompt
            self._stt_engine = STTEngine(
                model_size=self.settings.stt.model_size,
                device=self.settings.stt.device,
                compute_type=self.settings.stt.compute_type,
                sample_rate=self.settings.audio.sample_rate,
                model_dir=self.settings.stt.model_dir,
                language=self.settings.stt.language,
                allowed_languages=self.settings.stt.allowed_languages,
                language_detection_threshold=self.settings.stt.language_detection_threshold,
                vad_filter=self.settings.stt.vad_filter,
                vad_min_silence_ms=self.settings.stt.vad_min_silence_ms,
                initial_prompt=initial_prompt,
                input_gain=self.settings.stt.input_gain,
            )
        return self._stt_engine

    @property
    def nlp_engine(self) -> NLPEngine:
        """Lazy NLP engine initialization."""
        if self._nlp_engine is None:
            self._nlp_engine = NLPEngine(
                confidence_threshold=self.settings.nlp.confidence_threshold,
                confidence_threshold_ar=self.settings.nlp.confidence_threshold_ar,
                confidence_threshold_en=self.settings.nlp.confidence_threshold_en,
            )
        return self._nlp_engine

    @property
    def tts_engine(self) -> TTSEngine:
        """Lazy TTS engine initialization."""
        if self._tts_engine is None:
            self._tts_engine = TTSEngine(
                rate=self.settings.tts.rate,
                volume=self.settings.tts.volume,
                engine=self.settings.tts.engine,
                voice_id=self.settings.tts.voice_id,
                piper_voice_dir=self.settings.tts.piper_voice_dir,
                piper_voice_ar=self.settings.tts.piper_voice_ar,
                piper_voice_en=self.settings.tts.piper_voice_en,
            )
        return self._tts_engine

    @property
    def llm_engine(self) -> LLMEngine | None:
        """Lazy LLM engine initialization."""
        if not self.settings.llm.enabled:
            return None
        if self._llm_engine is None:
            self._llm_engine = LLMEngine(self.settings)
        return self._llm_engine

    def _safe_say(self, text: str) -> None:
        """Safely speak text, suppressing any TTS errors."""
        try:
            self.tts_engine.say(text)
        except Exception as e:
            logger.error(f"TTS fallback failed: {e}")
            logger.error(f"[TTS Error] {e}")

    def process_text(self, text: str, lang: str | None = None) -> tuple[str, str]:
        """Process text input through LLM (primary) or NLP (fallback) and execute action.

        Args:
            text: User input text
            lang: Detected language from STT (optional)

        Returns:
            Tuple of (response text, detected language)
        """
        # 1. Try LLM first (primary parser)
        if self.llm_engine and self.llm_engine.is_available():
            try:
                llm_result = self.llm_engine.parse_intent(text, lang)

                if llm_result.confidence >= self.settings.llm.confidence_threshold:
                    # LLM succeeded with high confidence
                    logger.info(
                        f"LLM: intent={llm_result.intent} | "
                        f"confidence={llm_result.confidence:.2f} | "
                        f"lang={llm_result.language}"
                    )
                    return self._execute_llm_intent(llm_result)

                # Low confidence - log and fall back to NLP
                logger.info(
                    f"LLM low confidence ({llm_result.confidence:.2f}), falling back to NLP"
                )

            except Exception as e:
                logger.warning(f"LLM parsing failed: {e}, falling back to NLP")

        # 2. NLP Regex Fallback
        try:
            intent, entities, confidence = self.nlp_engine.parse(text, stt_language=lang)

            # Use per-language threshold for unknown check
            detected_lang = lang or "en"
            threshold = (
                self.settings.nlp.confidence_threshold_ar
                if detected_lang == "ar"
                else self.settings.nlp.confidence_threshold_en
            )
            if intent == "unknown" or confidence < threshold:
                # Return localized "didn't understand" message
                template = self.nlp_engine.get_response_template("unknown", detected_lang)
                return template.format(text=text), detected_lang

            llm_available = (
                self.llm_engine
                and self.llm_engine.is_available()
                and self.settings.llm.fallback_to_nlp
            )
            if llm_available:
                assert self.llm_engine is not None
                try:
                    response_text = self.llm_engine.generate_response(
                        intent, entities, detected_lang, success=True
                    )
                    if response_text:
                        return self._execute_nlp_intent(intent, entities, detected_lang)
                except Exception as e:
                    logger.warning(f"LLM response generation failed: {e}")

            # NLP template fallback
            return self._execute_nlp_intent(intent, entities, detected_lang)

        except (NLPError, ActionError) as e:
            logger.error(f"Processing failed: {e}")
            return f"Error: {e}", lang or "en"
        except Exception as e:
            logger.exception(f"Unexpected error: {e}")
            return "An unexpected error occurred", lang or "en"

    def _execute_llm_intent(self, llm_result) -> tuple[str, str]:
        """Execute action based on LLM result and return LLM's response_text."""
        intent = llm_result.intent
        entities = llm_result.entities
        response_lang = llm_result.language

        if intent == "get_time" or intent == "get_date" or intent == "get_sys_info":
            return llm_result.response_text, response_lang
        elif intent == "open_app":
            app_name = entities.get("app")
            if not app_name:
                return "Which app would you like me to open?", response_lang
            result = open_app(app_name)
            # Use LLM's response_text if it mentions the app, otherwise use action result
            use_llm_text = app_name in llm_result.response_text
            response = llm_result.response_text if use_llm_text else result
            return response, response_lang
        elif intent == "web_search":
            query = entities.get("query", "")
            if not query:
                return "What would you like me to search for?", response_lang
            return web_search(query), response_lang
        else:
            return llm_result.response_text, response_lang

    def _execute_nlp_intent(self, intent: str, entities: dict, lang: str) -> tuple[str, str]:
        """Execute action based on NLP result."""
        if intent == "get_time":
            return get_time(), lang
        elif intent == "get_date":
            return get_date(), lang
        elif intent == "get_sys_info":
            info = get_sysinfo()
            cpu = info["cpu_percent"]
            mem = info["memory_percent"]
            disk = info["disk_percent"]
            return f"CPU: {cpu:.1f}% | Memory: {mem:.1f}% | Disk: {disk:.1f}%", lang
        elif intent == "open_app":
            app_name = entities.get("app")
            if not app_name:
                # Fallback: infer app from transcribed text for common cases
                return "Which app would you like me to open?", lang
            return open_app(app_name), lang
        elif intent == "web_search":
            return web_search(entities["query"]), lang
        else:
            return f"Unknown intent: {intent}", lang

    def run_once_voice(self) -> None:
        """Single voice interaction: listen -> transcribe -> process -> speak."""
        try:
            logger.info("Listening...")
            audio = self.stt_engine.record_audio(self.settings.stt.max_listen_seconds)

            logger.info("Transcribing...")
            text, stt_lang = self.stt_engine.transcribe(audio)

            if not text:
                logger.warning("No speech detected")
                return

            logger.info(f"Transcribed: {text} | Language: {stt_lang}")

            response, response_lang = self.process_text(text, lang=stt_lang)

            logger.info(f"Response: {response}")
            self.tts_engine.say(response, lang=response_lang)

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
            # For text mode, we don't have STT language, so let NLP detect
            response, response_lang = self.process_text(text, lang=None)
            logger.info(f"Response: {response}")
            self.tts_engine.say(response, lang=response_lang)
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


def run_enrollment(settings: Settings) -> None:
    """Run voice enrollment process."""
    click.echo("\n🎤 Voice Enrollment")
    click.echo("=" * 50)
    click.echo("Please speak each phrase clearly when prompted.")
    click.echo("Press Enter when ready to record each phrase.\n")

    # Create STT engine for enrollment (without initial_prompt to avoid circular dependency)
    stt_engine = STTEngine(
        model_size=settings.stt.model_size,
        device=settings.stt.device,
        compute_type=settings.stt.compute_type,
        sample_rate=settings.audio.sample_rate,
        model_dir=settings.stt.model_dir,
        language=None,
        allowed_languages=settings.stt.allowed_languages,
        language_detection_threshold=settings.stt.language_detection_threshold,
        vad_filter=settings.stt.vad_filter,
        vad_min_silence_ms=settings.stt.vad_min_silence_ms,
        initial_prompt=None,
    )

    all_transcriptions = []

    for i, phrase in enumerate(ENROLLMENT_PHRASES, 1):
        click.echo(f"Phrase {i}/{len(ENROLLMENT_PHRASES)}:")
        click.echo(f'  📝 "{phrase}"')
        click.echo("  Press Enter to start recording (5 seconds)...")
        click.prompt("", prompt_suffix="", show_default=False)

        try:
            click.echo("  🔴 Recording...")
            audio = stt_engine.record_audio(settings.stt.max_listen_seconds)
            click.echo("  🎙️  Transcribing...")
            text, lang = stt_engine.transcribe(audio)

            if text:
                click.echo(f'  ✅ Heard: "{text}" (lang: {lang})')
                all_transcriptions.append(text)
            else:
                click.echo("  ⚠️  No speech detected, skipping...")

        except Exception as e:
            click.echo(f"  ❌ Error: {e}")
            logger.error(f"Enrollment error on phrase {i}: {e}")

        click.echo("")

    if not all_transcriptions:
        click.echo("❌ No successful transcriptions. Enrollment cancelled.")
        return

    # Create initial_prompt from all transcriptions
    initial_prompt = " ".join(all_transcriptions)

    click.echo("=" * 50)
    click.echo(f'📋 Combined prompt: "{initial_prompt}"')
    click.echo("")

    if click.confirm("Save this enrollment?", default=True):
        try:
            save_enrollment(initial_prompt)
            click.echo("✅ Enrollment saved successfully!")
            click.echo(f"   File: {ENROLLMENT_FILE}")
            click.echo("\n💡 Restart voice-assistant to use the new enrollment.")
        except Exception as e:
            click.echo(f"❌ Failed to save enrollment: {e}")
    else:
        click.echo("Enrollment cancelled.")


def run_mic_test(settings: Settings, duration: float, gain: float | None) -> None:
    """Run microphone test and display audio quality metrics."""
    click.echo("\n🎤 Microphone Test")
    click.echo("=" * 50)
    click.echo(f"Duration: {duration}s")
    if gain is not None:
        click.echo(f"Gain override: {gain}")
    click.echo("")

    # Create STT engine for test (without initial_prompt to avoid circular dependency)
    stt_engine = STTEngine(
        model_size=settings.stt.model_size,
        device=settings.stt.device,
        compute_type=settings.stt.compute_type,
        sample_rate=settings.audio.sample_rate,
        model_dir=settings.stt.model_dir,
        language=None,
        allowed_languages=settings.stt.allowed_languages,
        language_detection_threshold=settings.stt.language_detection_threshold,
        vad_filter=settings.stt.vad_filter,
        vad_min_silence_ms=settings.stt.vad_min_silence_ms,
        initial_prompt=None,
    )

    click.echo("🔴 Recording...")
    try:
        result = stt_engine.test_microphone(duration=duration, gain=gain)

        if "error" in result:
            click.echo(f"❌ Error: {result['error']}")
            return

        click.echo("")
        click.echo("📊 Results")
        click.echo("-" * 50)
        click.echo(f"Duration:     {result['duration']}s")
        click.echo(f"RMS Level:    {result['rms']:.4f}")
        click.echo(f"Peak Level:   {result['peak']:.4f}")
        click.echo(
            f"Clipping:     {result['clipped_samples']}/{result['total_samples']} "
            f"({result['clipping_percentage']:.1f}%)"
        )
        click.echo(
            f"Language:     {result['detected_language']} (p={result.get('confidence', 0):.2f})"
        )
        click.echo(f'Transcription: "{result["transcription"]}"')
        click.echo("")
        click.echo(result["assessment"])
        if result["suggested_gain"] != settings.stt.input_gain:
            click.echo(
                f"💡 Suggested input_gain: {result['suggested_gain']:.3f} "
                f"(current: {settings.stt.input_gain})"
            )

    except Exception as e:
        click.echo(f"❌ Test failed: {e}")
        logger.error(f"Mic test failed: {e}")


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
@click.option(
    "--enroll",
    "do_enroll",
    is_flag=True,
    default=False,
    help="Run voice enrollment for speaker adaptation",
)
@click.option(
    "--test-mic",
    "do_test_mic",
    is_flag=True,
    default=False,
    help="Test microphone audio quality and transcription",
)
@click.option(
    "--test-mic-duration",
    type=float,
    default=3.0,
    help="Duration for microphone test in seconds",
)
@click.option(
    "--test-mic-gain",
    type=float,
    default=None,
    help="Override input gain for microphone test",
)
@click.option("--version", is_flag=True, default=False, help="Show version and exit")
@click.help_option("-h", "--help")
def cli(
    mode: str,
    text_input: str | None,
    list_intents: bool,
    do_enroll: bool,
    do_test_mic: bool,
    test_mic_duration: float,
    test_mic_gain: float | None,
    version: bool,
) -> None:
    """Voice Assistant - Offline-first voice assistant for academic field training.

    Examples:
      voice-assistant              # Start continuous listening mode
      voice-assistant --once       # Single voice interaction
      voice-assistant --once --text "what time is it"  # Text input
      voice-assistant --list-intents  # Show available commands
      voice-assistant --enroll     # Run voice enrollment
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
        nlp = NLPEngine(
            confidence_threshold=settings.nlp.confidence_threshold,
            confidence_threshold_ar=settings.nlp.confidence_threshold_ar,
            confidence_threshold_en=settings.nlp.confidence_threshold_en,
        )
        click.echo("Available intents:")
        for intent in nlp.patterns:
            name = intent["name"]
            patterns = ", ".join(f'"{p}"' for p in intent["patterns"][:3])
            if len(intent["patterns"]) > 3:
                patterns += "..."
            click.echo(f"  {name}: {patterns}")
        return

    if do_enroll:
        run_enrollment(settings)
        return

    if do_test_mic:
        run_mic_test(settings, test_mic_duration, test_mic_gain)
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
