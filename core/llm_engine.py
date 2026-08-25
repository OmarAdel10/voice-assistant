"""Voice Assistant — LLM Engine for Intent Parsing."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

from config.settings import Settings

logger = logging.getLogger(__name__)

# System prompt for intent parsing
SYSTEM_PROMPT = """You are a voice assistant for an Egyptian Arabic speaker with broken English.
Parse the user's speech (Arabic, English, or mixed) and output ONLY valid JSON.

Output format:
{
  "intent": "get_time|get_date|get_sys_info|open_app|web_search|unknown",
  "entities": {"app": "...", "query": "..."},
  "language": "ar|en",
  "response_text": "Natural response for TTS in detected language",
  "confidence": 0.0-1.0,
  "reasoning": "Brief explanation of decision"
}

INTENT DEFINITIONS:
- get_time: User asks for current time. entities: {}
- get_date: User asks for today's date. entities: {}
- get_sys_info: User asks for CPU/memory/disk usage. entities: {}
- open_app: User wants to launch an application. entities: {"app": "executable_name"}
- web_search: User wants to search the web. entities: {"query": "search terms"}
- unknown: Cannot determine intent. entities: {}

EGYPTIAN ARABIC APP MAPPINGS (infer executable for open_app):
- كود / الكود / في إس كود / فيجوال ستوديو / فيجوال → "code"
- فايرفوكس / المتصفح / فاير فوكس → "firefox"
- كروم / جوجل كروم / كروميوم → "chrome"
- تيرمينال / الطرفية / كونسول → "gnome-terminal"
- ناوتيليس / الملفات / دولفين / ثونار → "nautilus"
- في ال سي / فيإلسي / ام بي في → "vlc"
- سبوتيفاي → "spotify"
- ديسكورد → "discord"
- تليجرام / التيليجرام → "telegram"
- جيديت / محرر النصوص → "gedit"
- كيت → "kate"
- حاسبة / الآلة الحاسبة → "gnome-calculator"
- ليبر أوفيس → "libreoffice"
- (Infer others from context - Egyptian dialect common terms)

HANDLING RULES:
1. Code-switching: "open الكود" → intent=open_app, entities.app="code", language="en"
2. Dialect normalization: "أبن" → "افتح", "قوت" → "كود", "الساعة كام" → get_time
3. Partial speech: "كود..." → open_app with app="code" (high confidence)
4. Homophones: "ساعة" with time context → get_time, with watch context → unknown
5. If unsure after reasoning: confidence < 0.7, intent="unknown"

LANGUAGE DETECTION (CRITICAL):
- Use STT language as PRIMARY signal (it's based on audio analysis)
- Only OVERRIDE STT language if user text CLEARLY contradicts it:
  - STT says "en" but user text is entirely Arabic script → use "ar"
  - STT says "ar" but user text is entirely Latin script → use "en"
- Mixed code-switching → trust STT language
- When STT language is "auto-detect" or None → DETECT FROM TEXT SCRIPT:
  - Text contains Arabic script (Unicode 0600-06FF) → language="ar"
  - Text contains only Latin script → language="en"
  - Mixed scripts → use dominant script

ENTITY RULES:
- get_time, get_date, get_sys_info: entities = {}
- open_app: entities = {"app": "executable_name"}
- web_search: entities = {"query": "search terms"}

RESPONSE_TEXT GENERATION:
- get_time: "It is 2:30 PM." / "الساعة 2:30 ظهراً."
- get_date: "Today is Monday, January 15, 2024." / "اليوم الاثنين 15 يناير 2024."
- get_sys_info: "CPU: 25% | Memory: 60% | Disk: 45%." / "المعالج: 25% | الذاكرة: 60% | القرص: 45%."
- open_app: "Opening VS Code for you." / "تم تشغيل الكود يا بطل."
- web_search: "Searching for Python tutorial." / "جاري البحث عن بايثون."
- On failure: apologetic, helpful ("لم أفهم، ممكن تعيد؟" / "Sorry, I didn't understand.")

THINK STEP BY STEP. Output ONLY JSON."""


USER_PROMPT_TEMPLATE = """User said: "{text}"
STT detected language: {stt_language}

Parse and output JSON."""


@dataclass
class LLMResult:
    """Result from LLM intent parsing."""

    intent: str
    entities: dict[str, str]
    response_text: str
    language: str | None
    confidence: float
    reasoning: str = ""
    raw_response: str = ""


class LLMEngine:
    """LLM engine using llama-cpp-python for intent parsing and response generation."""

    def __init__(self, settings: Settings) -> None:
        """Initialize LLM engine.

        Args:
            settings: Application settings with LLM config
        """
        self.settings = settings
        self._llm = None
        self._load_model()

    def _load_model(self) -> None:
        """Load the GGUF model via llama-cpp-python."""
        if not self.settings.llm.enabled:
            logger.info("LLM disabled in config")
            return

        model_path = self.settings.llm.model_path
        try:
            from llama_cpp import Llama  # type: ignore[import-untyped]

            logger.info(f"Loading LLM model: {model_path}")
            self._llm = Llama(  # type: ignore[call-arg]
                model_path=model_path,
                n_gpu_layers=self.settings.llm.n_gpu_layers,
                n_ctx=self.settings.llm.n_ctx,
                n_threads=self.settings.llm.n_threads,
                verbose=self.settings.llm.verbose,
            )
            logger.info("LLM model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load LLM model: {e}")
            self._llm = None

    def is_available(self) -> bool:
        """Check if LLM is loaded and available."""
        return self._llm is not None

    def _detect_language_from_text(self, text: str) -> str:
        """Detect language from text script."""
        # Check for Arabic script (Unicode 0600-06FF, 0750-077F, 08A0-08FF, FB50-FDFF, FE70-FEFF)
        for char in text:
            code = ord(char)
            if (
                0x0600 <= code <= 0x06FF
                or 0x0750 <= code <= 0x077F
                or 0x08A0 <= code <= 0x08FF
                or 0xFB50 <= code <= 0xFDFF
                or 0xFE70 <= code <= 0xFEFF
            ):
                return "ar"
        return "en"

    def parse_intent(
        self,
        text: str,
        stt_language: str | None,
        context: dict | None = None,
    ) -> LLMResult:
        """Parse user intent from text using LLM.

        Args:
            text: User input text
            stt_language: Language detected by STT ("ar", "en", or None)
            context: Optional context from previous interactions

        Returns:
            LLMResult with intent, entities, response_text, language, confidence
        """
        if not self.is_available():
            logger.warning("LLM not available, returning unknown")
            return LLMResult(
                intent="unknown",
                entities={},
                response_text="",
                language=stt_language or "en",
                confidence=0.0,
                reasoning="LLM not loaded",
            )

        # Auto-detect language from text if STT language not provided
        stt_lang = stt_language
        if stt_lang is None:
            stt_lang = self._detect_language_from_text(text)

        # Build prompt
        prompt = self._build_prompt(text, stt_lang, context)

        try:
            # Generate response
            assert self._llm is not None
            response = self._llm(
                prompt,
                max_tokens=self.settings.llm.max_tokens,
                temperature=self.settings.llm.temperature,
                stop=["\n\n", "```"],
                echo=False,
            )

            raw_text = response["choices"][0]["text"].strip()

            logger.debug(f"LLM raw response: {raw_text}")

            # Parse JSON
            result = self._parse_json_response(raw_text)
            result.raw_response = raw_text

            # Validate and set defaults
            result = self._validate_result(result, stt_lang)

            logger.info(
                f"LLM: intent={result.intent} | confidence={result.confidence:.2f} | "
                f"lang={result.language} | reasoning={result.reasoning}"
            )

            return result

        except Exception as e:
            logger.error(f"LLM inference failed: {e}")
            return LLMResult(
                intent="unknown",
                entities={},
                response_text="",
                language=stt_lang,
                confidence=0.0,
                reasoning=f"LLM error: {e}",
            )

    def generate_response(
        self,
        intent: str,
        entities: dict,
        language: str,
        success: bool,
    ) -> str:
        """Generate TTS response text when NLP fallback succeeds.

        Args:
            intent: Intent from NLP fallback
            entities: Entities from NLP fallback
            language: Language code ("ar" or "en")
            success: Whether the action succeeded

        Returns:
            Natural response text for TTS
        """
        if not self.is_available():
            return ""

        entity_str = ", ".join(f"{k}={v}" for k, v in entities.items()) if entities else "none"

        prompt = f"""Generate a natural TTS response for a voice assistant.

Intent: {intent}
Entities: {entity_str}
Language: {language}
Action Success: {success}

Output ONLY the response text in the appropriate language (Egyptian Arabic or English).
Be natural and conversational."""

        try:
            assert self._llm is not None
            response = self._llm(
                prompt,
                max_tokens=128,
                temperature=0.3,
                stop=["\n"],
                echo=False,
            )
            text = response["choices"][0]["text"].strip()
            logger.debug(f"LLM generated response: {text}")
            return text
        except Exception as e:
            logger.warning(f"LLM response generation failed: {e}")
            return ""

    def _build_prompt(
        self,
        text: str,
        stt_language: str | None,
        context: dict | None = None,
    ) -> str:
        """Build the full prompt for intent parsing."""
        stt_lang_display = stt_language if stt_language else "auto-detect"
        user_prompt = USER_PROMPT_TEMPLATE.format(text=text, stt_language=stt_lang_display)

        if context:
            context_str = json.dumps(context, ensure_ascii=False)
            user_prompt += f"\nContext: {context_str}"

        return f"{SYSTEM_PROMPT}\n\n{user_prompt}"

    def _parse_json_response(self, raw_text: str) -> LLMResult:
        """Extract and parse JSON from LLM response."""
        # Find JSON object in response
        json_match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if not json_match:
            raise ValueError("No JSON found in response")

        json_str = json_match.group(0)
        data = json.loads(json_str)

        return LLMResult(
            intent=data.get("intent", "unknown"),
            entities=data.get("entities", {}),
            response_text=data.get("response_text", ""),
            language=data.get("language", "en"),
            confidence=float(data.get("confidence", 0.0)),
            reasoning=data.get("reasoning", ""),
        )

    def _validate_result(self, result: LLMResult, stt_lang: str | None) -> LLMResult:
        """Validate and set defaults for LLM result."""
        # Valid intents
        valid_intents = {
            "get_time",
            "get_date",
            "get_sys_info",
            "open_app",
            "web_search",
            "unknown",
        }

        if result.intent not in valid_intents:
            logger.warning(f"Invalid intent '{result.intent}', defaulting to unknown")
            result.intent = "unknown"

        # Validate language
        if result.language not in ("ar", "en"):
            result.language = stt_lang if stt_lang in ("ar", "en") else "en"

        # Validate confidence
        result.confidence = max(0.0, min(1.0, result.confidence))

        # Validate entities for open_app
        if result.intent == "open_app" and "app" not in result.entities:
            result.intent = "unknown"
            result.confidence = 0.0
            result.reasoning += " | Missing app entity"

        # Validate entities for web_search
        if result.intent == "web_search" and "query" not in result.entities:
            result.intent = "unknown"
            result.confidence = 0.0
            result.reasoning += " | Missing query entity"

        # Ensure response_text exists
        if not result.response_text:
            if result.language == "ar":
                result.response_text = "تم التنفيذ."
            else:
                result.response_text = "Done."

        return result
