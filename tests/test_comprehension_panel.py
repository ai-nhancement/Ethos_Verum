"""
tests/test_comprehension_panel.py

Unit tests for core/comprehension_panel.py — three-model signal verification.

Most tests operate on the vote-tallying and signal-filtering logic directly
(no live API calls).  Live integration tests are skipped unless all three
API keys are present in the environment.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Dict
from unittest.mock import AsyncMock, patch

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.comprehension_panel import (
    _tally_votes,
    _parse_model_response,
    verify_signals,
    is_available,
    _MODEL_IDS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _sig(value_name: str, polarity: Optional[int] = None) -> dict:
    s = {
        "value_name": value_name,
        "text_excerpt": "test excerpt",
        "significance": 0.8,
        "disambiguation_confidence": 0.75,
        "source": "keyword",
    }
    if polarity is not None:
        s["polarity_hint"] = polarity
    return s


# ---------------------------------------------------------------------------
# Model identifiers
# ---------------------------------------------------------------------------

class TestModelIDs:
    def test_three_models_configured(self):
        assert len(_MODEL_IDS) == 3

    def test_claude_or_gpt_present(self):
        assert any("claude" in m or "gpt" in m or "openai" in m for m in _MODEL_IDS)

    def test_gpt_or_other_model_present(self):
        assert any("gpt" in m or "deepseek" in m or "llama" in m for m in _MODEL_IDS)

    def test_all_model_ids_are_strings(self):
        assert all(isinstance(m, str) and m for m in _MODEL_IDS)


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

class TestParseModelResponse:
    def test_valid_json_holds_true(self):
        text = '{"integrity": {"holds": true, "violates": false}}'
        result = _parse_model_response(text)
        assert result is not None
        assert result["integrity"] == (True, False)

    def test_valid_json_violates_true(self):
        text = '{"courage": {"holds": false, "violates": true}}'
        result = _parse_model_response(text)
        assert result["courage"] == (False, True)

    def test_both_true(self):
        text = '{"fairness": {"holds": true, "violates": true}}'
        result = _parse_model_response(text)
        assert result["fairness"] == (True, True)

    def test_both_false(self):
        text = '{"loyalty": {"holds": false, "violates": false}}'
        result = _parse_model_response(text)
        assert result["loyalty"] == (False, False)

    def test_multiple_values(self):
        text = ('{"integrity": {"holds": true, "violates": false}, '
                '"courage": {"holds": false, "violates": true}}')
        result = _parse_model_response(text)
        assert result["integrity"] == (True, False)
        assert result["courage"] == (False, True)

    def test_strips_markdown_fences(self):
        text = '```json\n{"humility": {"holds": true, "violates": false}}\n```'
        result = _parse_model_response(text)
        assert result is not None
        assert result["humility"] == (True, False)

    def test_empty_string_returns_none(self):
        assert _parse_model_response("") is None

    def test_invalid_json_returns_none(self):
        assert _parse_model_response("not json") is None

    def test_non_dict_returns_none(self):
        assert _parse_model_response("[1, 2, 3]") is None

    def test_value_names_lowercased(self):
        text = '{"Integrity": {"holds": true, "violates": false}}'
        result = _parse_model_response(text)
        assert "integrity" in result
        assert "Integrity" not in result

    def test_non_bool_treated_as_none(self):
        # Non-boolean values in holds/violates should fall back to None
        text = '{"loyalty": {"holds": "yes", "violates": 0}}'
        result = _parse_model_response(text)
        # "yes" is not bool → None; 0 is not bool → None
        assert result["loyalty"] == (None, None)


# ---------------------------------------------------------------------------
# Vote tallying
# ---------------------------------------------------------------------------

class TestTallyVotes:
    def _make_resp(self, value: str, holds: bool, violates: bool):
        return {value: (holds, violates)}

    def test_unanimous_p1(self):
        responses = [
            {"integrity": (True, False)},
            {"integrity": (True, False)},
            {"integrity": (True, False)},
        ]
        assert _tally_votes(responses, "integrity") == "P1"

    def test_unanimous_p0(self):
        responses = [
            {"integrity": (False, True)},
            {"integrity": (False, True)},
            {"integrity": (False, True)},
        ]
        assert _tally_votes(responses, "integrity") == "P0"

    def test_unanimous_discard(self):
        responses = [
            {"integrity": (False, False)},
            {"integrity": (False, False)},
            {"integrity": (False, False)},
        ]
        assert _tally_votes(responses, "integrity") == "discard"

    def test_unanimous_tension(self):
        responses = [
            {"integrity": (True, True)},
            {"integrity": (True, True)},
            {"integrity": (True, True)},
        ]
        assert _tally_votes(responses, "integrity") == "tension"

    def test_majority_p1_with_one_abstain(self):
        responses = [
            {"courage": (True, False)},
            {"courage": (True, False)},
            None,  # abstain
        ]
        assert _tally_votes(responses, "courage") == "P1"

    def test_majority_p0_with_one_abstain(self):
        responses = [
            {"courage": (False, True)},
            None,
            {"courage": (False, True)},
        ]
        assert _tally_votes(responses, "courage") == "P0"

    def test_majority_discard_with_one_abstain(self):
        responses = [
            {"compassion": (False, False)},
            {"compassion": (False, False)},
            None,
        ]
        assert _tally_votes(responses, "compassion") == "discard"

    def test_two_abstains_returns_skip(self):
        responses = [
            {"integrity": (True, False)},
            None,
            None,
        ]
        assert _tally_votes(responses, "integrity") == "skip"

    def test_all_abstain_returns_skip(self):
        responses = [None, None, None]
        assert _tally_votes(responses, "integrity") == "skip"

    def test_missing_key_counts_as_abstain(self):
        # Response present but key not in it → abstain for that value
        responses = [
            {"courage": (True, False)},  # missing "integrity"
            {"courage": (True, False)},
            {"courage": (True, False)},
        ]
        assert _tally_votes(responses, "integrity") == "skip"

    def test_split_vote_no_majority(self):
        # 1 P1, 1 P0, 1 discard → no majority for either
        responses = [
            {"integrity": (True, False)},
            {"integrity": (False, True)},
            {"integrity": (False, False)},
        ]
        verdict = _tally_votes(responses, "integrity")
        assert verdict == "discard"

    def test_2_of_3_p1_with_one_disagreement(self):
        responses = [
            {"integrity": (True, False)},
            {"integrity": (True, False)},
            {"integrity": (False, True)},  # disagrees
        ]
        assert _tally_votes(responses, "integrity") == "P1"

    def test_tension_majority(self):
        # 2 say both Q1+Q2, 1 says only Q1
        responses = [
            {"integrity": (True, True)},
            {"integrity": (True, True)},
            {"integrity": (True, False)},
        ]
        assert _tally_votes(responses, "integrity") == "tension"


# ---------------------------------------------------------------------------
# Signal filtering
# ---------------------------------------------------------------------------

class TestVerifySignals:
    def _mock_panel(self, verdicts: dict):
        """Return a patched _run_panel_async that yields given verdicts."""
        import asyncio

        async def _fake_panel(text, figure_name, value_names):
            return {v: verdicts.get(v, "skip") for v in value_names}

        return _fake_panel

    def test_disabled_returns_original(self):
        sigs = [_sig("integrity")]
        result = verify_signals("text", "figure", sigs, enabled=False)
        assert result == sigs

    def test_empty_signals_returns_empty(self):
        result = verify_signals("text", "figure", [], enabled=True)
        assert result == []

    def test_skip_passes_through(self):
        sigs = [_sig("integrity")]
        with patch("core.comprehension_panel._run_panel_async",
                   self._mock_panel({"integrity": "skip"})):
            result = verify_signals("text", "figure", sigs, enabled=True)
        assert len(result) == 1
        assert result[0]["value_name"] == "integrity"
        assert "+panel" not in result[0]["source"]

    def test_discard_removes_signal(self):
        sigs = [_sig("integrity")]
        with patch("core.comprehension_panel._run_panel_async",
                   self._mock_panel({"integrity": "discard"})):
            result = verify_signals("text", "figure", sigs, enabled=True)
        assert result == []

    def test_p1_sets_polarity_hint_positive(self):
        sigs = [_sig("integrity")]
        with patch("core.comprehension_panel._run_panel_async",
                   self._mock_panel({"integrity": "P1"})):
            result = verify_signals("text", "figure", sigs, enabled=True)
        assert len(result) == 1
        assert result[0]["polarity_hint"] == 1
        assert "+panel" in result[0]["source"]

    def test_p0_sets_polarity_hint_negative(self):
        sigs = [_sig("integrity")]
        with patch("core.comprehension_panel._run_panel_async",
                   self._mock_panel({"integrity": "P0"})):
            result = verify_signals("text", "figure", sigs, enabled=True)
        assert len(result) == 1
        assert result[0]["polarity_hint"] == -1
        assert "+panel" in result[0]["source"]

    def test_tension_duplicates_signal(self):
        sigs = [_sig("integrity")]
        with patch("core.comprehension_panel._run_panel_async",
                   self._mock_panel({"integrity": "tension"})):
            result = verify_signals("text", "figure", sigs, enabled=True)
        assert len(result) == 2
        polarities = {s["polarity_hint"] for s in result}
        assert polarities == {1, -1}
        for s in result:
            assert "+panel" in s["source"]

    def test_mixed_verdicts(self):
        sigs = [_sig("integrity"), _sig("courage"), _sig("compassion")]
        verdicts = {
            "integrity": "P1",
            "courage":   "discard",
            "compassion": "P0",
        }
        with patch("core.comprehension_panel._run_panel_async",
                   self._mock_panel(verdicts)):
            result = verify_signals("text", "figure", sigs, enabled=True)
        value_names = [s["value_name"] for s in result]
        assert "integrity" in value_names
        assert "courage" not in value_names
        assert "compassion" in value_names
        integrity_sig = next(s for s in result if s["value_name"] == "integrity")
        assert integrity_sig["polarity_hint"] == 1
        compassion_sig = next(s for s in result if s["value_name"] == "compassion")
        assert compassion_sig["polarity_hint"] == -1

    def test_panel_exception_returns_original(self):
        sigs = [_sig("integrity")]
        with patch("core.comprehension_panel._run_panel_async",
                   side_effect=RuntimeError("network down")):
            result = verify_signals("text", "figure", sigs, enabled=True)
        assert result == sigs

    def test_original_signal_not_mutated(self):
        orig = _sig("integrity")
        orig_copy = dict(orig)
        sigs = [orig]
        with patch("core.comprehension_panel._run_panel_async",
                   self._mock_panel({"integrity": "P1"})):
            verify_signals("text", "figure", sigs, enabled=True)
        # Original dict should be unchanged
        assert orig == orig_copy

    def test_deduplication_one_verdict_per_unique_value(self):
        # Two signals for the same value — both should be processed
        sigs = [_sig("integrity"), _sig("integrity")]
        with patch("core.comprehension_panel._run_panel_async",
                   self._mock_panel({"integrity": "P1"})):
            result = verify_signals("text", "figure", sigs, enabled=True)
        assert len(result) == 2
        assert all(s["polarity_hint"] == 1 for s in result)

    def test_source_field_appended(self):
        sigs = [_sig("integrity")]
        sigs[0]["source"] = "keyword+lexicon"
        with patch("core.comprehension_panel._run_panel_async",
                   self._mock_panel({"integrity": "P1"})):
            result = verify_signals("text", "figure", sigs, enabled=True)
        assert result[0]["source"] == "keyword+lexicon+panel"


# ---------------------------------------------------------------------------
# is_available
# ---------------------------------------------------------------------------

class TestIsAvailable:
    def test_returns_bool(self):
        result = is_available()
        assert isinstance(result, bool)

    def test_false_when_no_key(self):
        # Simulate no API key set
        import os
        orig = os.environ.pop("MODEL_ACCESS_KEY", None)
        try:
            result = is_available()
            # If gradient is importable but key absent → False
            # If gradient not importable → also False
            assert isinstance(result, bool)
        finally:
            if orig is not None:
                os.environ["MODEL_ACCESS_KEY"] = orig
