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
        self._compiled_patterns: list[tuple[str, list[Pattern[str]], list[str]]] = []
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

        # Compile regex patterns for each intent
        self._compiled_patterns = []
        for intent in self._intents:
            name = intent["name"]
            patterns = intent["patterns"]
            entities = intent.get("entities", [])

            # Compile each pattern separately to avoid duplicate named group issues
            compiled_for_intent = []
            for p in patterns:
                parts = re.split(r"(\{\w+\})", p)
                escaped_parts = []
                for part in parts:
                    if re.match(r"^\{\w+\}$", part):
                        # This is a placeholder like {app} or {query}
                        entity_name = part[1:-1]
                        # Use greedy capture for query, non-greedy for app
                        if entity_name == "query":
                            escaped_parts.append(f"(?P<{entity_name}>.+)")
                        else:
                            escaped_parts.append(f"(?P<{entity_name}>[^\\s]+)")
                    else:
                        escaped_parts.append(re.escape(part))
                pattern_str = "".join(escaped_parts)
                compiled = re.compile(pattern_str, re.IGNORECASE)
                compiled_for_intent.append(compiled)

            self._compiled_patterns.append((name, compiled_for_intent, entities))

    def _normalize(self, text: str) -> str:
        """Normalize text for matching."""
        return text.strip().lower()

    def _fuzzy_match(self, text: str) -> tuple[str, float]:
        """Fuzzy match against example utterances using difflib.

        Args:
            text: Normalized input text

        Returns:
            Tuple of (intent_name, confidence)
        """
        best_intent = "unknown"
        best_score = 0.0

        for intent in self._intents:
            for pattern in intent["patterns"]:
                # Use the pattern as example utterance
                score = SequenceMatcher(None, text, pattern.lower()).ratio()
                if score > best_score:
                    best_score = score
                    best_intent = intent["name"]

        return best_intent, best_score

    def parse(self, text: str) -> tuple[str, dict[str, str], float]:
        """Parse text and return intent, entities, and confidence.

        Args:
            text: User input text

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

            # Primary: regex pattern matching
            for intent_name, patterns, entities in self._compiled_patterns:
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
                            f"NLP: {elapsed_ms:.0f}ms | Intent: {intent_name} | Confidence: 1.0"
                        )

                        return intent_name, extracted, 1.0

            # Fallback: fuzzy matching
            fuzzy_intent, fuzzy_score = self._fuzzy_match(normalized)

            elapsed_ms = (time.perf_counter() - start_time) * 1000
            intent = fuzzy_intent
            conf = fuzzy_score
            logger.info(
                f"NLP: {elapsed_ms:.0f}ms | Intent: {intent} | Confidence: {conf:.2f} (fuzzy)"
            )

            if fuzzy_score >= 0.6:
                return fuzzy_intent, {}, fuzzy_score

            return "unknown", {}, 0.0

        except Exception as e:
            logger.error(f"NLP parsing failed: {e}")
            raise NLPError(f"NLP parsing failed: {e}") from e
