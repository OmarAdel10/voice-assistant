"""Tests for core/nlp_engine.py."""

from unittest.mock import patch

import pytest

from core.exceptions import NLPError
from core.nlp_engine import NLPEngine


class TestNLPEngine:
    """Test NLPEngine functionality."""

    def test_parse_get_time(self):
        """Parse 'what time is it' -> get_time intent."""
        engine = NLPEngine()
        intent, entities, confidence = engine.parse("what time is it")

        assert intent == "get_time"
        assert entities == {}
        assert confidence > 0.6

    def test_parse_get_date(self):
        """Parse 'what date is it' -> get_date intent."""
        engine = NLPEngine()
        intent, entities, confidence = engine.parse("what date is it")

        assert intent == "get_date"
        assert entities == {}
        assert confidence > 0.6

    def test_parse_get_sys_info(self):
        """Parse 'system info' -> get_sys_info intent."""
        engine = NLPEngine()
        intent, entities, confidence = engine.parse("system info")

        assert intent == "get_sys_info"
        assert entities == {}
        assert confidence > 0.6

    def test_parse_open_app_extracts_entity(self):
        """Parse 'open firefox' -> open_app with app entity."""
        engine = NLPEngine()
        intent, entities, confidence = engine.parse("open firefox")

        assert intent == "open_app"
        assert entities == {"app": "firefox"}
        assert confidence > 0.6

    def test_parse_web_search_extracts_entity(self):
        """Parse 'search for cats' -> web_search with query entity."""
        engine = NLPEngine()
        intent, entities, confidence = engine.parse("search for cats")

        assert intent == "web_search"
        assert entities == {"query": "cats"}
        assert confidence > 0.6

    def test_parse_unknown_input_returns_unknown(self):
        """Unknown input returns unknown intent with 0 confidence."""
        engine = NLPEngine()
        intent, entities, confidence = engine.parse("xyz random gibberish")

        assert intent == "unknown"
        assert entities == {}
        assert confidence == 0.0

    def test_parse_case_insensitive(self):
        """Matching should be case insensitive."""
        engine = NLPEngine()
        intent, _, confidence = engine.parse("WHAT TIME IS IT")

        assert intent == "get_time"
        assert confidence > 0.6

    def test_parse_with_punctuation(self):
        """Matching should handle punctuation."""
        engine = NLPEngine()
        intent, _, confidence = engine.parse("what time is it?")

        assert intent == "get_time"
        assert confidence > 0.6

    def test_parse_open_app_variations(self):
        """Parse various 'open' patterns."""
        engine = NLPEngine()

        for text in ["open firefox", "launch firefox", "start firefox", "run firefox"]:
            intent, entities, confidence = engine.parse(text)
            assert intent == "open_app", f"Failed for: {text}"
            assert entities == {"app": "firefox"}, f"Failed for: {text}"
            assert confidence > 0.6, f"Failed for: {text}"

    def test_parse_web_search_variations(self):
        """Parse various 'search' patterns."""
        engine = NLPEngine()

        for text in ["search for cats", "google cats", "look up cats", "find cats", "search cats"]:
            intent, entities, confidence = engine.parse(text)
            assert intent == "web_search", f"Failed for: {text}"
            assert entities == {"query": "cats"}, f"Failed for: {text}"
            assert confidence > 0.6, f"Failed for: {text}"

    def test_parse_logs_latency(self, caplog):
        """Parse should log latency in milliseconds."""
        import logging

        caplog.set_level(logging.INFO)

        engine = NLPEngine()
        engine.parse("what time is it")

        assert any("NLP:" in record.message and "ms" in record.message for record in caplog.records)

    def test_parse_empty_string_returns_unknown(self):
        """Empty string returns unknown."""
        engine = NLPEngine()
        intent, entities, confidence = engine.parse("")

        assert intent == "unknown"
        assert entities == {}
        assert confidence == 0.0

    def test_parse_whitespace_only_returns_unknown(self):
        """Whitespace-only string returns unknown."""
        engine = NLPEngine()
        intent, entities, confidence = engine.parse("   ")

        assert intent == "unknown"
        assert entities == {}
        assert confidence == 0.0

    def test_intents_loaded_from_config(self):
        """Engine should load intents from config/intents.json."""
        engine = NLPEngine()
        assert len(engine._intents) == 5
        intent_names = {i["name"] for i in engine._intents}
        assert intent_names == {"get_time", "get_date", "get_sys_info", "open_app", "web_search"}

    def test_invalid_intents_file_raises_nlp_error(self):
        """Invalid intents file should raise NLPError."""
        with patch("core.nlp_engine.Path") as mock_path:
            mock_path.return_value.read_text.side_effect = OSError("File not found")
            with pytest.raises(NLPError):
                NLPEngine()

    def test_malformed_json_raises_nlp_error(self):
        """Malformed JSON should raise NLPError."""
        with patch("core.nlp_engine.Path") as mock_path:
            mock_path.return_value.read_text.return_value = "{ invalid json"
            with pytest.raises(NLPError):
                NLPEngine()


class TestNLPEngineFuzzyMatching:
    """Test fuzzy matching fallback."""

    def test_fuzzy_match_get_time(self):
        """Fuzzy match should work for similar phrases."""
        engine = NLPEngine()
        intent, _, confidence = engine.parse("tell me the time please")

        assert intent == "get_time"
        assert confidence > 0.5  # Fuzzy match may have lower confidence

    def test_fuzzy_match_get_date(self):
        """Fuzzy match for date variations."""
        engine = NLPEngine()
        intent, _, confidence = engine.parse("what's today's date")

        assert intent == "get_date"
        assert confidence > 0.5


class TestNLPEngineEntities:
    """Test entity extraction."""

    def test_open_app_entity_extraction_various(self):
        """Extract app name from various patterns."""
        engine = NLPEngine()

        test_cases = [
            ("open firefox", "firefox"),
            ("launch vscode", "vscode"),
            ("start terminal", "terminal"),
            ("run calculator", "calculator"),
        ]
        for text, expected_app in test_cases:
            intent, entities, _ = engine.parse(text)
            assert intent == "open_app"
            assert entities == {"app": expected_app}

    def test_web_search_entity_extraction_various(self):
        """Extract query from various patterns."""
        engine = NLPEngine()

        test_cases = [
            ("search for python tutorial", "python tutorial"),
            ("google machine learning", "machine learning"),
            ("look up weather forecast", "weather forecast"),
            ("find restaurants nearby", "restaurants nearby"),
            ("search quantum physics", "quantum physics"),
        ]
        for text, expected_query in test_cases:
            intent, entities, _ = engine.parse(text)
            assert intent == "web_search"
            assert entities == {"query": expected_query}
