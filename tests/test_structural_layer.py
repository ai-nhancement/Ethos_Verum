"""
tests/test_structural_layer.py

Dedicated tests for core/structural_layer.py — the Layer 3 structural
pattern detector and zero-shot entailment sub-layer.

These are isolated unit tests. They do not require the full pipeline,
any database, or the DeBERTa model (zero-shot tests are mocked unless
the model is available).

Test structure:
  TestAdversityPattern      — _ADVERSITY_RE regex
  TestAgencyPattern         — _AGENCY_RE regex
  TestResistancePattern     — _RESISTANCE_RE regex
  TestStakesPattern         — _STAKES_RE regex
  TestStructuralScore       — structural_score() function
  TestZeroshotScores        — zeroshot_scores() with mocked pipeline
  TestLayer3Signals         — layer3_signals() integration
  TestRegexEdgeCases        — boundary conditions and non-matching text
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from core.structural_layer import (
    _ADVERSITY_RE,
    _AGENCY_RE,
    _RESISTANCE_RE,
    _STAKES_RE,
    structural_score,
    layer3_signals,
    zeroshot_scores,
    VALUE_HYPOTHESES,
)


# ---------------------------------------------------------------------------
# TestAdversityPattern
# ---------------------------------------------------------------------------

class TestAdversityPattern:
    def _hits(self, text: str) -> bool:
        return bool(_ADVERSITY_RE.search(text))

    def test_despite_matches(self):
        assert self._hits("Despite the danger, he continued.")

    def test_in_spite_of_matches(self):
        assert self._hits("In spite of the opposition, she held firm.")

    def test_although_matches(self):
        assert self._hits("Although they threatened him, he refused.")

    def test_even_though_matches(self):
        assert self._hits("Even though it cost him everything, he told the truth.")

    def test_notwithstanding_matches(self):
        assert self._hits("Notwithstanding the risks, she pressed on.")

    def test_at_great_risk_matches(self):
        assert self._hits("He spoke at great risk to his own safety.")

    def test_at_personal_cost_matches(self):
        assert self._hits("She acted at personal cost to herself.")

    def test_under_pressure_matches(self):
        assert self._hits("He maintained his position under pressure.")

    def test_under_threat_matches(self):
        assert self._hits("She refused to yield under threat.")

    def test_under_interrogation_matches(self):
        assert self._hits("He would not confess even under interrogation.")

    def test_against_all_opposition_matches(self):
        assert self._hits("She pressed forward against all opposition.")

    def test_in_the_face_of_matches(self):
        assert self._hits("In the face of death, he did not waver.")

    def test_sacrificing_matches(self):
        assert self._hits("Sacrificing everything, she spoke out.")

    def test_sacrificed_matches(self):
        assert self._hits("He sacrificed his career to expose the corruption.")

    def test_gave_up_everything_matches(self):
        assert self._hits("She gave up everything for the cause.")

    def test_laid_down_his_life_matches(self):
        assert self._hits("He laid down his life for his comrades.")

    def test_laid_down_her_life_matches(self):
        assert self._hits("She laid down her life protecting the children.")

    def test_knowing_the_cost_matches(self):
        assert self._hits("Knowing the cost, he signed the declaration.")

    def test_at_the_expense_of_matches(self):
        assert self._hits("He told the truth at the expense of his career.")

    def test_neutral_text_does_not_match(self):
        assert not self._hits("The meeting was scheduled for Tuesday morning.")

    def test_despite_not_duplicated(self):
        """
        Regression: 'despite' used to appear twice in the alternation.
        Verify simple 'despite' still matches after the deduplication fix.
        """
        assert self._hits("Despite the odds she succeeded.")
        assert self._hits("He persisted despite the threat of imprisonment.")


# ---------------------------------------------------------------------------
# TestAgencyPattern
# ---------------------------------------------------------------------------

class TestAgencyPattern:
    def _hits(self, text: str) -> bool:
        return bool(_AGENCY_RE.search(text))

    def test_i_refused_matches(self):
        assert self._hits("I refused to abandon my post.")

    def test_i_refuse_matches(self):
        assert self._hits("I refuse to betray my principles.")

    def test_i_stood_matches(self):
        assert self._hits("I stood despite the overwhelming pressure.")

    def test_i_stood_firm_matches(self):
        assert self._hits("I stood firm against every challenge.")

    def test_i_chose_matches(self):
        assert self._hits("I chose to speak even knowing the cost.")

    def test_i_held_matches(self):
        assert self._hits("I held my ground when they demanded I step aside.")

    def test_i_pressed_on_matches(self):
        assert self._hits("I pressed on despite exhaustion.")

    def test_i_still_refused_matches(self):
        assert self._hits("I still refused to yield.")

    def test_i_yet_continued_matches(self):
        assert self._hits("I yet continued even after the losses.")

    def test_i_nevertheless_persisted_matches(self):
        assert self._hits("I nevertheless persisted through the ordeal.")

    def test_we_endured_matches(self):
        assert self._hits("We endured every hardship without complaint.")

    def test_i_would_not_matches(self):
        assert self._hits("I would not abandon them.")

    def test_i_did_not_matches(self):
        assert self._hits("I did not yield to their demands.")

    def test_i_could_not_abandon_matches(self):
        assert self._hits("I could not abandon the people who trusted me.")

    def test_no_subject_does_not_match(self):
        assert not self._hits("Refused to yield and stood firm.")

    def test_third_person_does_not_match(self):
        assert not self._hits("He refused to yield.")

    def test_chose_does_not_match_chos(self):
        """
        Regression: 'chose?' used to allow 'chos' as a match. Verify
        that garbage like 'I chos' would not match 'chose' entries.
        The fix was to use 'chose' instead of 'chose?'.
        """
        # 'I chose' must match
        assert self._hits("I chose to remain.")
        # 'chose' must not be matchable as 'chos' (impossible in real text,
        # but verify the pattern doesn't accidentally widen to nonsense)
        assert not self._hits("I chos the harder path.")  # 'chos' not a word

    def test_stood_firm_not_required_separately(self):
        """
        Regression: 'stood' and 'stood firm' used to be separate alternatives.
        'stood' must match in both cases; the separate 'stood firm' was redundant.
        """
        assert self._hits("I stood and refused to move.")
        assert self._hits("I stood firm in the face of opposition.")


# ---------------------------------------------------------------------------
# TestResistancePattern
# ---------------------------------------------------------------------------

class TestResistancePattern:
    def _hits(self, text: str) -> bool:
        return bool(_RESISTANCE_RE.search(text))

    def test_refused_to_yield_matches(self):
        assert self._hits("He refused to yield even under torture.")

    def test_refused_to_surrender_matches(self):
        assert self._hits("She refused to surrender her principles.")

    def test_refused_to_betray_matches(self):
        assert self._hits("He refused to betray his comrades.")

    def test_refused_to_flee_matches(self):
        assert self._hits("She refused to flee when others ran.")

    def test_refused_to_give_up_matches(self):
        assert self._hits("I refused to give up despite everything.")

    def test_refused_to_back_down_matches(self):
        assert self._hits("He refused to back down from his position.")

    def test_would_not_yield_matches(self):
        assert self._hits("She would not yield to their demands.")

    def test_would_not_be_silenced_matches(self):
        assert self._hits("He would not be silenced by the authorities.")

    def test_did_not_waver_matches(self):
        assert self._hits("She did not waver throughout the ordeal.")

    def test_did_not_falter_matches(self):
        assert self._hits("He did not falter under the pressure.")

    def test_never_yielded_matches(self):
        assert self._hits("She never yielded on the matter of principle.")

    def test_never_betrayed_matches(self):
        assert self._hits("He never betrayed his friends.")

    def test_could_not_abandon_them_matches(self):
        assert self._hits("She could not abandon them in their hour of need.")

    def test_neutral_text_does_not_match(self):
        assert not self._hits("The project was completed on schedule.")


# ---------------------------------------------------------------------------
# TestStakesPattern
# ---------------------------------------------------------------------------

class TestStakesPattern:
    def _hits(self, text: str) -> bool:
        return bool(_STAKES_RE.search(text))

    def test_life_at_stake_matches(self):
        assert self._hits("His life was at stake when he made that choice.")

    def test_career_at_stake_matches(self):
        assert self._hits("Her career was at stake, yet she refused.")

    def test_freedom_at_stake_matches(self):
        assert self._hits("Their freedom depended on it.")

    def test_threatened_to_matches(self):
        assert self._hits("They threatened him to step aside.")

    def test_warned_them_to_matches(self):
        assert self._hits("The authorities warned them to stop immediately.")

    def test_risked_death_matches(self):
        assert self._hits("He risked death to carry the message.")

    def test_risked_imprisonment_matches(self):
        assert self._hits("She risked imprisonment to speak the truth.")

    def test_faced_execution_matches(self):
        assert self._hits("He faced execution rather than renounce his beliefs.")

    def test_faced_exile_matches(self):
        assert self._hits("She faced exile for her refusal to comply.")

    def test_at_gunpoint_matches(self):
        assert self._hits("He still refused at gunpoint.")

    def test_at_threat_of_death_matches(self):
        assert self._hits("Even at threat of death she would not recant.")

    def test_imprisonment_not_duplicated(self):
        """
        Regression: 'imprisonment' used to appear twice in the alternation.
        Verify it still matches after the deduplication fix.
        """
        assert self._hits("He risked imprisonment by speaking out.")
        assert self._hits("She faced imprisonment for her beliefs.")

    def test_neutral_text_does_not_match(self):
        assert not self._hits("The quarterly report showed steady growth.")


# ---------------------------------------------------------------------------
# TestStructuralScore
# ---------------------------------------------------------------------------

class TestStructuralScore:
    def test_no_signals_returns_zero(self):
        text = "The meeting was held on Tuesday and the committee adjourned."
        assert structural_score(text) == 0.0

    def test_one_signal_returns_point_three(self):
        # Only adversity — no agency, resistance, or stakes
        text = "Despite everything, the work continued."
        score = structural_score(text)
        assert score == 0.3

    def test_two_signals_returns_point_five(self):
        # adversity + agency
        text = "Despite the danger, I refused to turn back."
        score = structural_score(text)
        assert score == 0.5

    def test_three_signals_returns_point_eight(self):
        # adversity + agency + resistance
        text = (
            "Despite the threat, I stood my ground and refused to yield — "
            "I would not betray them."
        )
        score = structural_score(text)
        assert score == 0.8

    def test_four_signals_returns_one(self):
        # adversity + agency + resistance + stakes
        text = (
            "Despite the risk to my life, I refused to surrender. "
            "I would not yield even at gunpoint. "
            "My career was at stake, yet I pressed on."
        )
        score = structural_score(text)
        assert score == 1.0

    def test_return_type_is_float(self):
        assert isinstance(structural_score("text"), float)

    def test_empty_text_returns_zero(self):
        assert structural_score("") == 0.0

    def test_score_in_valid_range(self):
        texts = [
            "The report was filed.",
            "Despite all odds she continued.",
            "I refused to yield despite the danger.",
            "Despite everything I would not back down even at gunpoint.",
        ]
        for text in texts:
            s = structural_score(text)
            assert 0.0 <= s <= 1.0, f"structural_score out of range for: {text!r}"

    def test_longer_strong_passage_reaches_max(self):
        text = (
            "Although the enemies surrounded us, I refused to surrender. "
            "We would not abandon the wounded, even though our lives hung in the balance. "
            "They threatened me to capitulate at swordpoint, yet I held firm."
        )
        assert structural_score(text) == 1.0

    def test_fail_open_on_error(self):
        """structural_score must return 0.0, not raise, on unexpected input."""
        assert structural_score(None) == 0.0  # type: ignore


# ---------------------------------------------------------------------------
# TestZeroshotScores — mocked pipeline
# ---------------------------------------------------------------------------

class TestZeroshotScores:
    """Test the zeroshot_scores() interface using a mock pipeline."""

    def _make_mock_pipe(self, label_score_pairs: list) -> MagicMock:
        """Build a mock pipeline return value for the given (label, score) list."""
        # pipeline returns {"labels": [...], "scores": [...]}
        labels = [VALUE_HYPOTHESES[v] for v, _ in label_score_pairs]
        scores = [s for _, s in label_score_pairs]
        mock = MagicMock(return_value={"labels": labels, "scores": scores})
        return mock

    def test_returns_empty_when_no_pipeline(self):
        with patch("core.structural_layer._get_zeroshot_pipeline", return_value=None):
            result = zeroshot_scores("text", ["courage"])
            assert result == []

    def test_returns_empty_for_empty_text(self):
        mock_pipe = MagicMock()
        with patch("core.structural_layer._get_zeroshot_pipeline", return_value=mock_pipe):
            assert zeroshot_scores("", ["courage"]) == []

    def test_returns_empty_for_empty_values(self):
        mock_pipe = MagicMock()
        with patch("core.structural_layer._get_zeroshot_pipeline", return_value=mock_pipe):
            assert zeroshot_scores("brave text", []) == []

    def test_returns_empty_for_unknown_value(self):
        mock_pipe = MagicMock()
        with patch("core.structural_layer._get_zeroshot_pipeline", return_value=mock_pipe):
            result = zeroshot_scores("text", ["not_a_real_value"])
            assert result == []

    def test_above_threshold_value_returned(self):
        mock_pipe = self._make_mock_pipe([("courage", 0.85)])
        with patch("core.structural_layer._get_zeroshot_pipeline", return_value=mock_pipe):
            result = zeroshot_scores("brave text", ["courage"], threshold=0.50)
        assert len(result) == 1
        assert result[0][0] == "courage"
        assert result[0][1] == 0.85

    def test_below_threshold_value_excluded(self):
        mock_pipe = self._make_mock_pipe([("courage", 0.40)])
        with patch("core.structural_layer._get_zeroshot_pipeline", return_value=mock_pipe):
            result = zeroshot_scores("text", ["courage"], threshold=0.50)
        assert result == []

    def test_results_sorted_by_score_desc(self):
        mock_pipe = self._make_mock_pipe([
            ("courage", 0.60),
            ("integrity", 0.90),
            ("patience", 0.75),
        ])
        with patch("core.structural_layer._get_zeroshot_pipeline", return_value=mock_pipe):
            result = zeroshot_scores(
                "text", ["courage", "integrity", "patience"], threshold=0.30
            )
        scores = [s for _, s in result]
        assert scores == sorted(scores, reverse=True), \
            "zeroshot_scores must return results sorted by score DESC"

    def test_returns_value_name_not_hypothesis_string(self):
        mock_pipe = self._make_mock_pipe([("integrity", 0.95)])
        with patch("core.structural_layer._get_zeroshot_pipeline", return_value=mock_pipe):
            result = zeroshot_scores("text", ["integrity"], threshold=0.50)
        # Must return "integrity", not the hypothesis string
        assert result[0][0] == "integrity"
        assert result[0][0] in VALUE_HYPOTHESES

    def test_scores_rounded_to_4_decimal_places(self):
        mock_pipe = self._make_mock_pipe([("courage", 0.12345678)])
        with patch("core.structural_layer._get_zeroshot_pipeline", return_value=mock_pipe):
            result = zeroshot_scores("text", ["courage"], threshold=0.10)
        score_str = str(result[0][1])
        if "." in score_str:
            assert len(score_str.split(".")[1]) <= 4

    def test_inference_error_returns_empty(self):
        mock_pipe = MagicMock(side_effect=RuntimeError("OOM"))
        with patch("core.structural_layer._get_zeroshot_pipeline", return_value=mock_pipe):
            result = zeroshot_scores("text", ["courage"])
            assert result == []


# ---------------------------------------------------------------------------
# TestLayer3Signals
# ---------------------------------------------------------------------------

class TestLayer3Signals:
    def test_returns_triple(self):
        """layer3_signals must return (float, list, dict) regardless of input."""
        with patch("core.structural_layer._get_zeroshot_pipeline", return_value=None):
            result = layer3_signals("text", 0.90, "action", [])
        assert len(result) == 3
        struct_score, new_sigs, agreement = result
        assert isinstance(struct_score, float)
        assert isinstance(new_sigs, list)
        assert isinstance(agreement, dict)

    def test_structural_score_propagates(self):
        """Structural score must reflect actual adversity signal count."""
        strong = (
            "Despite the risk to my life, I refused to yield. "
            "They threatened me at gunpoint."
        )
        with patch("core.structural_layer._get_zeroshot_pipeline", return_value=None):
            score, _, _ = layer3_signals(strong, 0.90, "action", [])
        assert score > 0.0, "Strong adversity passage must produce non-zero structural score"

    def test_zeroshot_disabled_returns_empty_agreement(self):
        with patch("core.structural_layer._get_zeroshot_pipeline", return_value=None):
            _, new_sigs, agreement = layer3_signals(
                "text", 0.90, "action", ["courage"],
                zeroshot_enabled=False,
            )
        assert new_sigs == []
        assert agreement == {}

    def test_agreement_placed_in_correct_dict(self):
        """
        When zero-shot fires for a value already in candidate_values,
        it must appear in the agreement dict, not in new_sigs.
        """
        hyp = VALUE_HYPOTHESES["courage"]
        mock_pipe = MagicMock(return_value={
            "labels": [hyp],
            "scores": [0.80],
        })
        with patch("core.structural_layer._get_zeroshot_pipeline", return_value=mock_pipe):
            _, new_sigs, agreement = layer3_signals(
                "brave text", 0.90, "action", ["courage"],
                zeroshot_threshold=0.50,
            )
        assert "courage" in agreement, \
            "courage in candidate_values + high ZS score must appear in agreement"
        # Must NOT also be in new_sigs (it's not a new detection)
        new_names = {s["value_name"] for s in new_sigs}
        assert "courage" not in new_names

    def test_standalone_detection_above_threshold(self):
        """
        Zero-shot score above standalone threshold for a value NOT in
        candidate_values must produce a new signal in new_sigs.
        """
        hyp = VALUE_HYPOTHESES["integrity"]
        mock_pipe = MagicMock(return_value={
            "labels": [hyp],
            "scores": [0.85],
        })
        with patch("core.structural_layer._get_zeroshot_pipeline", return_value=mock_pipe):
            _, new_sigs, agreement = layer3_signals(
                "honest text", 0.90, "action", [],  # candidate_values empty
                zeroshot_threshold=0.35,
                zeroshot_standalone_threshold=0.70,
            )
        new_names = {s["value_name"] for s in new_sigs}
        assert "integrity" in new_names, \
            "High-confidence zero-shot standalone detection must appear in new_sigs"

    def test_standalone_below_threshold_excluded(self):
        """
        Zero-shot score below standalone threshold for a non-candidate value
        must NOT produce a new signal.
        """
        hyp = VALUE_HYPOTHESES["integrity"]
        mock_pipe = MagicMock(return_value={
            "labels": [hyp],
            "scores": [0.60],
        })
        with patch("core.structural_layer._get_zeroshot_pipeline", return_value=mock_pipe):
            _, new_sigs, _ = layer3_signals(
                "honest text", 0.90, "action", [],
                zeroshot_threshold=0.35,
                zeroshot_standalone_threshold=0.70,  # 0.60 < 0.70
            )
        assert new_sigs == [], \
            "Below-standalone-threshold zero-shot score must not create new signal"

    def test_new_signal_has_required_fields(self):
        hyp = VALUE_HYPOTHESES["loyalty"]
        mock_pipe = MagicMock(return_value={
            "labels": [hyp],
            "scores": [0.90],
        })
        with patch("core.structural_layer._get_zeroshot_pipeline", return_value=mock_pipe):
            _, new_sigs, _ = layer3_signals(
                "loyal text", 0.90, "action", [],
                zeroshot_threshold=0.35,
                zeroshot_standalone_threshold=0.70,
            )
        assert new_sigs
        sig = new_sigs[0]
        for field in ("value_name", "text_excerpt", "significance",
                      "disambiguation_confidence", "source"):
            assert field in sig, f"New signal missing field: {field}"

    def test_new_signal_source_is_zeroshot(self):
        hyp = VALUE_HYPOTHESES["loyalty"]
        mock_pipe = MagicMock(return_value={
            "labels": [hyp],
            "scores": [0.90],
        })
        with patch("core.structural_layer._get_zeroshot_pipeline", return_value=mock_pipe):
            _, new_sigs, _ = layer3_signals(
                "text", 0.90, "action", [],
                zeroshot_threshold=0.35,
                zeroshot_standalone_threshold=0.70,
            )
        assert new_sigs[0]["source"] == "zeroshot"


# ---------------------------------------------------------------------------
# TestRegexEdgeCases
# ---------------------------------------------------------------------------

class TestRegexEdgeCases:
    def test_all_four_patterns_case_insensitive(self):
        assert _ADVERSITY_RE.search("DESPITE the danger")
        assert _AGENCY_RE.search("I REFUSED to yield")
        assert _RESISTANCE_RE.search("REFUSED TO YIELD")
        assert _STAKES_RE.search("HIS LIFE WAS AT STAKE")

    def test_partial_word_not_matched_by_stakes_pattern(self):
        """
        'imprisonment' appears once in _STAKES_RE after the deduplication fix.
        Verify the regex still works correctly.
        """
        assert _STAKES_RE.search("She faced imprisonment for speaking out.")

    def test_value_hypotheses_covers_all_15_values(self):
        from core.value_extractor import VALUE_VOCAB
        vocab_keys = set(VALUE_VOCAB.keys())
        hyp_keys   = set(VALUE_HYPOTHESES.keys())
        assert hyp_keys == vocab_keys, \
            f"VALUE_HYPOTHESES must cover all 15 values.\n" \
            f"Missing from hypotheses: {vocab_keys - hyp_keys}\n" \
            f"Extra in hypotheses: {hyp_keys - vocab_keys}"

    def test_structural_score_is_deterministic(self):
        text = "Despite the danger I refused to yield even at gunpoint."
        s1 = structural_score(text)
        s2 = structural_score(text)
        assert s1 == s2
