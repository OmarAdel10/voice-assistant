"""Voice Assistant — Natural Language Processing Engine."""

from __future__ import annotations

import json
import logging
import re
import time
from difflib import SequenceMatcher
from pathlib import Path
from re import Pattern
from typing import Any

from core.exceptions import NLPError

try:
    from langdetect import detect
except ImportError:
    detect = None

logger = logging.getLogger(__name__)


class NLPEngine:
    """Natural Language Processing engine for intent classification."""

    def __init__(self, intents_path: str | Path = "config/intents.json") -> None:
        """Initialize NLP engine by loading intents from JSON.

        Args:
            intents_path: Path to intents.json file

        Raises:
            NLPError: If intents file cannot be loaded or parsed
        """
        self._intents_path = Path(intents_path)
        self._intents: list[dict[str, Any]] = []
        self._compiled_patterns_en: list[tuple[str, list[Pattern[str]], list[str]]] = []
        self._compiled_patterns_ar: list[tuple[str, list[Pattern[str]], list[str]]] = []
        self._response_templates_en: dict[str, str] = {}
        self._response_templates_ar: dict[str, str] = {}
        self._load_intents()

    def _load_intents(self) -> None:
        """Load and compile intent patterns from JSON file."""
        try:
            content = self._intents_path.read_text(encoding="utf-8")
            data = json.loads(content)
        except OSError as e:
            logger.error(f"Failed to read intents file: {e}")
            raise NLPError(f"Failed to read intents file: {e}") from e
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in intents file: {e}")
            raise NLPError(f"Invalid JSON in intents file: {e}") from e

        if "intents" not in data:
            raise NLPError("Intents file missing 'intents' key")

        self._intents = data["intents"]

        # Compile regex patterns for each intent (English and Arabic)
        self._compiled_patterns_en = []
        self._compiled_patterns_ar = []
        for intent in self._intents:
            name = intent["name"]
            entities = intent.get("entities", [])

            # English patterns
            patterns_en = intent.get("patterns_en", [])
            compiled_for_intent_en = []
            for p in patterns_en:
                parts = re.split(r"(\{\w+\})", p)
                escaped_parts = []
                for part in parts:
                    if re.match(r"^\{\w+\}$", part):
                        entity_name = part[1:-1]
                        if entity_name == "query":
                            escaped_parts.append(f"(?P<{entity_name}>.+)")
                        else:
                            escaped_parts.append(f"(?P<{entity_name}>[^\\s]+)")
                    else:
                        escaped_parts.append(re.escape(part))
                pattern_str = "".join(escaped_parts)
                compiled = re.compile(pattern_str, re.IGNORECASE)
                compiled_for_intent_en.append(compiled)
            self._compiled_patterns_en.append((name, compiled_for_intent_en, entities))

            # Arabic patterns
            patterns_ar = intent.get("patterns_ar", [])
            compiled_for_intent_ar = []
            for p in patterns_ar:
                parts = re.split(r"(\{\w+\})", p)
                escaped_parts = []
                for part in parts:
                    if re.match(r"^\{\w+\}$", part):
                        entity_name = part[1:-1]
                        if entity_name == "query":
                            escaped_parts.append(f"(?P<{entity_name}>.+)")
                        else:
                            escaped_parts.append(f"(?P<{entity_name}>[^\\s]+)")
                    else:
                        escaped_parts.append(re.escape(part))
                pattern_str = "".join(escaped_parts)
                compiled = re.compile(pattern_str, re.IGNORECASE)
                compiled_for_intent_ar.append(compiled)
            self._compiled_patterns_ar.append((name, compiled_for_intent_ar, entities))

            # Response templates
            if "response_template_en" in intent:
                self._response_templates_en[name] = intent["response_template_en"]
            if "response_template_ar" in intent:
                self._response_templates_ar[name] = intent["response_template_ar"]

    def _normalize(self, text: str) -> str:
        """Normalize text for matching."""
        return text.strip().lower()

    def _detect_language(self, text: str, stt_language: str | None = None) -> str:
        """Detect language from STT output or langdetect fallback.

        If STT language and text language differ, trust text language
        (more reliable for intent matching than audio-based STT guess).
        """
        # STT language takes priority IF it matches text language
        if stt_language:
            stt_lang = "ar" if stt_language.startswith("ar") else "en"
            # Verify with langdetect
            if detect:
                try:
                    text_lang = detect(text)
                    text_lang = "ar" if text_lang == "ar" else "en"
                    # If they match, use STT language
                    if text_lang == stt_lang:
                        return stt_lang
                    # If they differ, prefer text language (more reliable for intent matching)
                    logger.warning(
                        f"STT language ({stt_lang}) != text language "
                        f"({text_lang}), using text language"
                    )
                    return text_lang
                except Exception:
                    pass
            return stt_lang

        # Fallback to langdetect
        if detect:
            try:
                lang = detect(text)
                return "ar" if lang == "ar" else "en"
            except Exception:
                pass

        # Default to English
        return "en"

    def _get_patterns_and_templates(self, lang: str):
        """Get compiled patterns and response templates for language."""
        if lang == "ar":
            return self._compiled_patterns_ar, self._response_templates_ar
        return self._compiled_patterns_en, self._response_templates_en

    def _fuzzy_match(self, text: str, lang: str) -> tuple[str, float]:
        """Fuzzy match against example utterances for given language."""
        best_intent = "unknown"
        best_score = 0.0

        for intent in self._intents:
            patterns = intent.get(f"patterns_{lang}", [])
            for pattern in patterns:
                score = SequenceMatcher(None, text, pattern.lower()).ratio()
                if score > best_score:
                    best_score = score
                    best_intent = intent["name"]

        return best_intent, best_score

    def parse(
        self, text: str, stt_language: str | None = None
    ) -> tuple[str, dict[str, str], float]:
        """Parse text and return intent, entities, and confidence.

        Args:
            text: User input text
            stt_language: Language code from STT (e.g., "ar", "en")

        Returns:
            Tuple of (intent_name, entities_dict, confidence_float)
            Returns ("unknown", {}, 0.0) if no match

        Raises:
            NLPError: If internal parsing fails
        """
        start_time = time.perf_counter()

        try:
            normalized = self._normalize(text)

            if not normalized:
                return "unknown", {}, 0.0

            # Detect language
            lang = self._detect_language(text, stt_language)

            # Get patterns and templates for detected language
            compiled_patterns, response_templates = self._get_patterns_and_templates(lang)

            # Primary: regex pattern matching
            for intent_name, patterns, entities in compiled_patterns:
                for pattern in patterns:
                    match = pattern.search(normalized)
                    if match:
                        # Extract entities from named capture groups
                        extracted = {}
                        for entity in entities:
                            if entity in match.groupdict():
                                extracted[entity] = match.group(entity)

                        elapsed_ms = (time.perf_counter() - start_time) * 1000
                        logger.info(
                            f"NLP: {elapsed_ms:.0f}ms | Intent: {intent_name} | "
                            f"Confidence: 1.0 | Lang: {lang}"
                        )

                        return intent_name, extracted, 1.0

            # Fallback: fuzzy matching
            fuzzy_intent, fuzzy_score = self._fuzzy_match(normalized, lang)

            elapsed_ms = (time.perf_counter() - start_time) * 1000
            logger.info(
                f"NLP: {elapsed_ms:.0f}ms | Intent: {fuzzy_intent} | "
                f"Confidence: {fuzzy_score:.2f} (fuzzy) | Lang: {lang}"
            )

            if fuzzy_score >= 0.6:
                return fuzzy_intent, {}, fuzzy_score

            return "unknown", {}, 0.0

        except Exception as e:
            logger.error(f"NLP parsing failed: {e}")
            raise NLPError(f"NLP parsing failed: {e}") from e

    def get_response_template(self, intent_name: str, lang: str) -> str:
        """Get response template for intent and language."""
        if lang == "ar":
            return self._response_templates_ar.get(
                intent_name,
                (
                    "لم أفهم الأمر. جرب أن تسأل عن الوقت، التاريخ، "
                    "معلومات النظام، فتح تطبيق، أو البحث على الويب."
                ),
            )
        return self._response_templates_en.get(
            intent_name,
            (
                "I didn't understand that command. Try asking for time, date, "
                "system info, opening an app, or searching the web."
            ),
        )

    @property
    def patterns(self) -> list[dict[str, Any]]:
        """Backwards compatibility: return intents with combined patterns."""
        result = []
        for intent in self._intents:
            combined = intent.copy()
            # Combine patterns from both languages
            combined["patterns"] = intent.get("patterns_en", []) + intent.get("patterns_ar", [])
            result.append(combined)
        return result
