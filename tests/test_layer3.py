"""
tests/test_layer3.py

Tests for Layer 3 signal extraction:
  - Sub-layer A: structural adversity pattern detection
  - Sub-layer B: zero-shot DeBERTa entailment scoring
  - layer3_signals() integration
  - value_extractor.py Layer 3 wiring

Tests are organized to run sub-layer A (pure regex, always fast) first.
Sub-layer B (zero-shot) tests are marked so they can be skipped in
environments where the DeBERTa model is not cached.
"""

from __future__ import annotations

import pytest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.structural_layer import (
    structural_score,
    zeroshot_scores,
    layer3_signals,
    VALUE_HYPOTHESES,
    is_zeroshot_available,
    _ADVERSITY_RE,
    _AGENCY_RE,
    _RESISTANCE_RE,
    _STAKES_RE,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _zs_available():
    """True if DeBERTa model is cached and loadable."""
    return is_zeroshot_available()


# ---------------------------------------------------------------------------
# Sub-layer A — structural_score()
# ---------------------------------------------------------------------------

class TestStructuralScore:

    def test_no_signals_returns_zero(self):
        texts = [
            "He was known for his honesty.",
            "I love pizza and coffee.",
            "The meeting started at nine o'clock.",
        ]
        for t in texts:
            assert structural_score(t) == 0.0, f"Expected 0.0 for: {t!r}"

    def test_single_adversity_marker(self):
        # "despite" alone → 0.3
        score = structural_score("She continued despite the obstacles.")
        assert score == pytest.approx(0.3)

    def test_single_agency_marker(self):
        score = structural_score("I refused to leave my post.")
        # "I refused" → agency; "refused to" → resistance. Two signals.
        assert score >= 0.3

    def test_adversity_and_agency(self):
        score = structural_score("I stood firm despite the threat.")
        assert score >= 0.5

    def test_adversity_agency_resistance_three_signals(self):
        score = structural_score(
            "I refused to yield despite the pressure, knowing the cost."
        )
        assert score >= 0.8

    def test_all_four_signals_max(self):
        text = (
            "I refused to yield despite the threat to my life, "
            "knowing the cost of defiance. My life was at stake."
        )
        score = structural_score(text)
        assert score == pytest.approx(1.0)

    def test_adversity_regex_matches(self):
        adversity_phrases = [
            "despite the danger",
            "in spite of everything",
            "under pressure",
            "under fire",
            "in the face of opposition",
            "at great personal risk",
            "risking my life",
            "knowing the cost",
            "at the expense of",
            "against all opposition",
        ]
        for phrase in adversity_phrases:
            assert _ADVERSITY_RE.search(phrase), f"Should match: {phrase!r}"

    def test_agency_regex_matches(self):
        agency_phrases = [
            "I refused to",
            "I stood firm",
            "I carried on",
            "I pressed on",
            "I would not yield",
            "we did not abandon",
        ]
        for phrase in agency_phrases:
            assert _AGENCY_RE.search(phrase), f"Should match: {phrase!r}"

    def test_resistance_regex_matches(self):
        resistance_phrases = [
            "refused to yield",
            "refused to capitulate",
            "would not betray",
            "did not yield",
            "never yielded",
            "refused to surrender",
        ]
        for phrase in resistance_phrases:
            assert _RESISTANCE_RE.search(phrase), f"Should match: {phrase!r}"

    def test_stakes_regex_matches(self):
        stakes_phrases = [
            "my life was at stake",
            "threatened me to",
            "faced execution",
            "risked imprisonment",
        ]
        for phrase in stakes_phrases:
            assert _STAKES_RE.search(phrase), f"Should match: {phrase!r}"

    def test_score_is_monotonic_with_signal_count(self):
        """More signal types → higher score."""
        scores = [
            structural_score("He was honest."),                              # 0 signals
            structural_score("Despite everything, she continued."),          # 1 signal
            structural_score("I pressed on despite the danger."),            # 2 signals
            structural_score("I refused to yield despite the threat."),      # 3 signals
        ]
        for i in range(len(scores) - 1):
            assert scores[i] <= scores[i + 1], (
                f"Score [{i}]={scores[i]} should be <= [{i+1}]={scores[i+1]}"
            )

    def test_always_returns_float(self):
        for text in ["", "x", "a b c", None if False else ""]:
            result = structural_score(text)
            assert isinstance(result, float)
            assert 0.0 <= result <= 1.0


# ---------------------------------------------------------------------------
# Sub-layer B — zeroshot_scores() + is_zeroshot_available()
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _zs_available(), reason="DeBERTa model not cached")
class TestZeroshotScores:

    def test_returns_list_of_tuples(self):
        result = zeroshot_scores("I told the truth even when it cost me.", ["integrity"])
        assert isinstance(result, list)
        for item in result:
            assert len(item) == 2
            assert isinstance(item[0], str)
            assert isinstance(item[1], float)

    def test_integrity_high_on_truth_passage(self):
        text = "I refused to lie even knowing the consequences."
        scores = dict(zeroshot_scores(text, ["integrity"], threshold=0.0))
        assert "integrity" in scores
        assert scores["integrity"] > 0.80

    def test_courage_high_on_danger_passage(self):
        text = "Despite death threats, she continued her investigation alone."
        scores = dict(zeroshot_scores(text, ["courage"], threshold=0.0))
        assert "courage" in scores
        assert scores["courage"] > 0.70

    def test_responsibility_high_on_accountability_passage(self):
        text = "I accept full responsibility for the disaster. It was my order."
        scores = dict(zeroshot_scores(text, ["responsibility", "humility"], threshold=0.0))
        assert scores.get("responsibility", 0) > 0.70

    def test_threshold_filters_low_scores(self):
        text = "The weather was fair today."
        # No value should score high on this
        results = zeroshot_scores(text, list(VALUE_HYPOTHESES.keys()), threshold=0.90)
        assert len(results) == 0

    def test_sorted_descending(self):
        text = "I told the truth despite knowing they would arrest me."
        results = zeroshot_scores(text, list(VALUE_HYPOTHESES.keys()), threshold=0.0)
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)

    def test_returns_empty_on_empty_text(self):
        assert zeroshot_scores("", ["integrity"]) == []

    def test_returns_empty_on_unknown_values(self):
        result = zeroshot_scores("test text", ["nonexistent_value"])
        assert result == []

    def test_returns_empty_if_no_values(self):
        assert zeroshot_scores("test text", []) == []

    def test_all_15_hypotheses_defined(self):
        from core.semantic_store import VALUE_NAMES
        for v in VALUE_NAMES:
            assert v in VALUE_HYPOTHESES, f"Missing hypothesis for: {v}"

    def test_truncation_does_not_crash(self):
        long_text = "I spoke the truth despite the risk. " * 100
        result = zeroshot_scores(long_text, ["integrity", "courage"])
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# layer3_signals() integration
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _zs_available(), reason="DeBERTa model not cached")
class TestLayer3Signals:

    def test_returns_three_tuple(self):
        result = layer3_signals(
            text="I told the truth.",
            significance=0.5,
            doc_type="journal",
            candidate_values=[],
        )
        assert len(result) == 3
        struct_score, new_sigs, agreement = result
        assert isinstance(struct_score, float)
        assert isinstance(new_sigs, list)
        assert isinstance(agreement, dict)

    def test_structural_score_passthrough(self):
        struct_score, _, _ = layer3_signals(
            text="I refused to yield despite the threat to my life.",
            significance=0.8, doc_type="journal", candidate_values=[],
        )
        assert struct_score >= 0.8

    def test_agreement_populated_for_candidate_values(self):
        text = "I refused to lie even knowing the consequences."
        _, _, agreement = layer3_signals(
            text=text, significance=0.8, doc_type="journal",
            candidate_values=["integrity"], zeroshot_threshold=0.35,
        )
        assert "integrity" in agreement
        assert agreement["integrity"] > 0.70

    def test_standalone_detection_uses_higher_threshold(self):
        text = "I spoke the truth at great personal risk."
        # integrity not in candidates — should appear in new_sigs only if >= standalone threshold
        _, new_sigs, _ = layer3_signals(
            text=text, significance=0.8, doc_type="journal",
            candidate_values=[],  # empty — everything is standalone
            zeroshot_standalone_threshold=0.70,
        )
        # integrity should comfortably exceed 0.70 on this passage
        sig_names = {s["value_name"] for s in new_sigs}
        assert "integrity" in sig_names

    def test_no_standalone_below_high_threshold(self):
        # Use a passage with no clear value content — purely operational/logistical
        text = "The quarterly budget report was submitted on Tuesday at three o'clock."
        _, new_sigs, _ = layer3_signals(
            text=text, significance=0.5, doc_type="journal",
            candidate_values=[], zeroshot_standalone_threshold=0.95,
        )
        # With very high threshold, purely logistical text should produce no standalone detections
        assert len(new_sigs) == 0

    def test_new_signal_schema(self):
        text = "I accepted full responsibility for the failure and I did not yield."
        _, new_sigs, _ = layer3_signals(
            text=text, significance=0.7, doc_type="journal",
            candidate_values=[],  # responsibility not pre-detected
        )
        for sig in new_sigs:
            assert "value_name" in sig
            assert "text_excerpt" in sig
            assert "significance" in sig
            assert "disambiguation_confidence" in sig
            assert sig["source"] == "zeroshot"
            assert 0.0 <= sig["disambiguation_confidence"] <= 1.0

    def test_zeroshot_disabled_returns_no_new_sigs(self):
        struct_score, new_sigs, agreement = layer3_signals(
            text="I stood firm and refused to yield despite all threats.",
            significance=0.8, doc_type="journal", candidate_values=["courage"],
            zeroshot_enabled=False,
        )
        assert new_sigs == []
        assert agreement == {}
        # structural score still computed
        assert struct_score > 0.0

    def test_fail_open_on_bad_input(self):
        # Should not raise
        result = layer3_signals("", 0.5, "journal", [])
        assert len(result) == 3


# ---------------------------------------------------------------------------
# Structural-only (fast) layer3_signals tests
# ---------------------------------------------------------------------------

class TestLayer3StructuralOnly:
    """Tests that don't require the DeBERTa model."""

    def test_structural_score_zero_on_neutral_text(self):
        struct_score, _, _ = layer3_signals(
            text="He was a just and honest man.",
            significance=0.5, doc_type="journal",
            candidate_values=["integrity"],
            zeroshot_enabled=False,
        )
        assert struct_score == 0.0

    def test_structural_score_nonzero_on_adversity(self):
        struct_score, _, _ = layer3_signals(
            text="I stood firm despite the threat of execution.",
            significance=0.8, doc_type="journal",
            candidate_values=["courage"],
            zeroshot_enabled=False,
        )
        assert struct_score > 0.0

    def test_empty_candidates_with_no_zeroshot(self):
        struct_score, new_sigs, agreement = layer3_signals(
            text="I refused to yield despite everything.",
            significance=0.8, doc_type="journal",
            candidate_values=[],
            zeroshot_enabled=False,
        )
        assert isinstance(struct_score, float)
        assert new_sigs == []
        assert agreement == {}
