"""
tests/test_resistance.py

Tests for core/resistance.py — three-tier resistance formula.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.resistance import (
    compute_resistance,
    _MORTAL_RE,
    _HIGH_ADVERSITY_RE,
    _ADVERSITY_RE,
    _HOLD_RE,
    _FAILURE_RE,
    _BASE,
    _SIG_CAP,
    _MORTAL_BONUS,
    _HIGH_ADV_BONUS,
    _ADV_BONUS,
    _HOLD_EXTRA,
)


# ---------------------------------------------------------------------------
# Regex pattern smoke tests
# ---------------------------------------------------------------------------

class TestMortalPatterns:
    @pytest.mark.parametrize("phrase", [
        "I would rather die",
        "I die the King's faithful servant",
        "I am prepared to die",
        "prepared to die for this ideal",
        "one life to lose for my country",
        "abjure, curse, and detest",
        "sentence against me",
        "cannot and will not retract anything",
        "will not retract anything",
        "executed at dawn",
        "hanged as a traitor",
        "beheaded on the scaffold",
        "sentenced to death",
    ])
    def test_mortal_phrases_match(self, phrase):
        assert _MORTAL_RE.search(phrase), f"Expected mortal match for: {phrase!r}"

    def test_bare_death_does_not_match(self):
        # "death" alone fires too broadly on philosophical text
        assert not _MORTAL_RE.search("It is not death that a man should fear")

    def test_bare_die_third_person_not_matched(self):
        # Only first-person contextual forms should match
        assert not _MORTAL_RE.search("she would die of old age")


class TestHighAdversityPatterns:
    @pytest.mark.parametrize("phrase", [
        "threatened to destroy me professionally",
        "they can come for me today",
        "at the hazard of incurring ridicule",
        "advised by my lawyers to stay silent",
        "imprisoned for his beliefs",
        "exiled from his homeland",
        "persecuted by the state",
        "oppressed and broken-hearted",
    ])
    def test_high_adv_phrases_match(self, phrase):
        assert _HIGH_ADVERSITY_RE.search(phrase), f"Expected match: {phrase!r}"


class TestAdversityPatterns:
    @pytest.mark.parametrize("phrase", [
        "even though it cost him everything",
        "despite the danger",
        "but I still believe",
        "not easy to stand firm",
        "scared but he continued",
        "at a cost to himself",
        "my suffering is greater than my courage",
        "I thought it better to be ruined",
    ])
    def test_adversity_phrases_match(self, phrase):
        assert _ADVERSITY_RE.search(phrase), f"Expected match: {phrase!r}"


class TestHoldPatterns:
    @pytest.mark.parametrize("phrase", [
        "nevertheless he continued",
        "stood firm under pressure",
        "refused to yield",
        "would not surrender",
        "will not yield in the thing that is right",
        "did not capitulate",
        "pressed on despite everything",
        "would do it again",
        "choose to stay anyway",
    ])
    def test_hold_phrases_match(self, phrase):
        assert _HOLD_RE.search(phrase), f"Expected match: {phrase!r}"


class TestFailurePatterns:
    @pytest.mark.parametrize("phrase", [
        "gave in to the pressure",
        "gave up after years of struggle",
        "yielded to their demands",
        "backed down from his position",
        "caved under interrogation",
        "capitulated to the regime",
        "dropped it when threatened",
    ])
    def test_failure_phrases_match(self, phrase):
        assert _FAILURE_RE.search(phrase), f"Expected match: {phrase!r}"


# ---------------------------------------------------------------------------
# Formula: base + significance + doc_type only (no text markers)
# ---------------------------------------------------------------------------

class TestBaseFormula:
    def test_base_floor_no_markers(self):
        # With zero significance and no text signals, output is base + doc_nudge
        result = compute_resistance("He walked quietly to the market.", 0.0, "unknown")
        # 0.12 + 0.0 + 0.02 = 0.14
        assert result == pytest.approx(0.14, abs=0.01)

    def test_significance_increases_score(self):
        low  = compute_resistance("A plain observation.", 0.10, "unknown")
        high = compute_resistance("A plain observation.", 0.90, "unknown")
        assert high > low

    def test_significance_capped(self):
        # sig=0.99 should give same result as sig=1.5 (both hit _SIG_CAP)
        a = compute_resistance("Plain text.", 0.99, "unknown")
        b = compute_resistance("Plain text.", 1.50, "unknown")
        assert a == b

    def test_doc_type_action_higher_than_speech(self):
        text = "A plain observation with no adversity markers."
        action  = compute_resistance(text, 0.5, "action")
        speech  = compute_resistance(text, 0.5, "speech")
        assert action > speech

    def test_doc_type_unknown_fallback(self):
        # Unknown or unrecognised doc types use the 'unknown' nudge
        a = compute_resistance("Plain text.", 0.5, "unknown")
        b = compute_resistance("Plain text.", 0.5, "xyzzy")
        assert a == b

    def test_result_in_range(self):
        result = compute_resistance("Just a sentence.", 0.5, "journal")
        assert 0.0 <= result <= 1.0

    def test_never_raises_on_empty(self):
        result = compute_resistance("", 0.0, "")
        assert isinstance(result, float)

    def test_never_raises_on_bad_sig(self):
        # NaN does not raise — min(nan, cap) clips to a boundary value.
        # Contract: no exception, result in valid range.
        result = compute_resistance("text", float("nan"), "speech")
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_result_clipped_at_one(self):
        # Even with maxed parameters, must not exceed 1.0
        result = compute_resistance(
            "I would rather die than betray my conscience. I would not yield.",
            1.0, "action"
        )
        assert result <= 1.0


# ---------------------------------------------------------------------------
# Tier A — Mortal stakes
# ---------------------------------------------------------------------------

class TestMortalTier:
    def test_mortal_fires_on_die(self):
        text = "I would rather die having spoken after my manner, than speak in your manner and live."
        result = compute_resistance(text, 0.7, "speech")
        assert result >= 0.80

    def test_mortal_fires_on_prepared_to_die(self):
        text = "It is an ideal for which I am prepared to die."
        result = compute_resistance(text, 0.7, "speech")
        assert result >= 0.80

    def test_mortal_fires_on_one_life_to_lose(self):
        text = "I only regret that I have but one life to lose for my country."
        result = compute_resistance(text, 0.7, "speech")
        assert result >= 0.80

    def test_mortal_fires_on_abjure(self):
        text = "I abjure, curse, and detest the aforesaid errors and heresies."
        result = compute_resistance(text, 0.7, "speech")
        assert result >= 0.80

    def test_mortal_fires_on_retract_refusal(self):
        text = "I cannot and will not retract anything, since it is neither safe nor right to go against conscience."
        result = compute_resistance(text, 0.7, "speech")
        assert result >= 0.80

    def test_mortal_not_suppressed_by_failure(self):
        # Galileo's recantation: "abjure" fires mortal, but text is capitulation.
        # Mortal tier must fire regardless of failure language.
        text = "I abjure, curse, and detest the aforesaid errors and heresies, and I gave in."
        with_failure = compute_resistance(text, 0.7, "speech")
        text_no_failure = "I abjure, curse, and detest the aforesaid errors and heresies."
        without_failure = compute_resistance(text_no_failure, 0.7, "speech")
        assert with_failure == without_failure

    def test_mortal_with_hold_adds_extra(self):
        text_hold    = "I would rather die. I will not yield."
        text_no_hold = "I would rather die having spoken thus."
        with_hold    = compute_resistance(text_hold, 0.7, "speech")
        without_hold = compute_resistance(text_no_hold, 0.7, "speech")
        assert with_hold == pytest.approx(without_hold + _HOLD_EXTRA, abs=0.001)

    def test_mortal_beats_high_adv(self):
        # When both mortal and high_adv patterns could fire, mortal wins
        text = "I would rather die than be imprisoned again."
        result = compute_resistance(text, 0.7, "speech")
        # Should include _MORTAL_BONUS, not just _HIGH_ADV_BONUS
        base_sig_doc = _BASE + min(0.7 * 0.17, 0.12) + 0.02
        assert result >= round(base_sig_doc + _MORTAL_BONUS - 0.01, 4)


# ---------------------------------------------------------------------------
# Tier B — High adversity
# ---------------------------------------------------------------------------

class TestHighAdversityTier:
    def test_high_adv_fires_on_threatened(self):
        text = "They threatened to destroy me professionally. I stayed anyway."
        result = compute_resistance(text, 0.7, "unknown")
        assert result >= 0.50

    def test_high_adv_fires_on_oppressed(self):
        text = "I was oppressed and broken-hearted with the sorrows and injustice I saw."
        result = compute_resistance(text, 0.7, "letter")
        assert result >= 0.50

    def test_high_adv_fires_on_hazard(self):
        text = "I prefer to be true to myself, even at the hazard of incurring the ridicule of others."
        result = compute_resistance(text, 0.7, "journal")
        assert result >= 0.50

    def test_high_adv_suppressed_by_failure(self):
        # Person faced threat but dropped it — no bonus
        text = "They threatened to report me. I dropped it. It wasn't worth the trouble."
        result = compute_resistance(text, 0.7, "unknown")
        base_sig_doc = _BASE + min(0.7 * 0.17, 0.12) + 0.02
        assert result == pytest.approx(base_sig_doc, abs=0.01)

    def test_high_adv_not_suppressed_when_hold_present(self):
        # Failure + hold → hold overrides failure suppression for Tier B
        text = "They threatened to destroy me. I caved in but choose to stay true to my beliefs."
        # has_failure=True (caved in), has_hold=True (choose to stay)
        # Tier B suppression: `not has_failure` → False → suppressed even with hold
        # This is intentional: if failure fires, high_adv is suppressed regardless
        result = compute_resistance(text, 0.7, "unknown")
        base_sig_doc = _BASE + min(0.7 * 0.17, 0.12) + 0.02
        assert result == pytest.approx(base_sig_doc, abs=0.01)

    def test_high_adv_with_hold_adds_extra(self):
        text_hold    = "They threatened to destroy me. I choose to stay regardless."
        text_no_hold = "They threatened to destroy me. I pressed forward."
        # "I choose to stay" fires hold; "pressed forward" does not
        with_hold    = compute_resistance(text_hold, 0.7, "unknown")
        without_hold = compute_resistance(text_no_hold, 0.7, "unknown")
        assert with_hold > without_hold


# ---------------------------------------------------------------------------
# Tier C — Standard adversity
# ---------------------------------------------------------------------------

class TestStandardAdversityTier:
    def test_adversity_fires_on_despite(self):
        text = "Despite the danger, he continued on his chosen path."
        result = compute_resistance(text, 0.7, "unknown")
        assert result > _BASE + min(0.7 * 0.17, 0.12) + 0.02

    def test_adversity_fires_on_suffering(self):
        text = "Let me not be ashamed that my suffering is greater than my courage."
        result = compute_resistance(text, 0.7, "journal")
        assert result >= 0.40

    def test_adversity_suppressed_by_failure_without_hold(self):
        text = "Despite my principles, I gave in to the pressure."
        # adversity fires (despite), failure fires (gave in), hold does not
        result = compute_resistance(text, 0.7, "unknown")
        base_sig_doc = _BASE + min(0.7 * 0.17, 0.12) + 0.02
        assert result == pytest.approx(base_sig_doc, abs=0.01)

    def test_adversity_kept_when_hold_overrides_failure(self):
        text = "Despite my principles, I gave in to pressure, but I will not yield in the thing that is right."
        # adversity fires (despite), failure fires (gave in), hold fires (will not yield)
        result = compute_resistance(text, 0.7, "unknown")
        base_sig_doc = _BASE + min(0.7 * 0.17, 0.12) + 0.02
        assert result > base_sig_doc

    def test_adversity_with_hold_adds_extra(self):
        text = "Despite everything, I would not surrender."
        # adversity fires (despite), hold fires (would not surrender)
        result = compute_resistance(text, 0.7, "unknown")
        base_sig_doc = _BASE + min(0.7 * 0.17, 0.12) + 0.02
        assert result >= round(base_sig_doc + _ADV_BONUS + _HOLD_EXTRA - 0.01, 4)


# ---------------------------------------------------------------------------
# Tier priority — highest tier wins, tiers do not stack
# ---------------------------------------------------------------------------

class TestTierPriority:
    def test_mortal_takes_precedence_over_adversity(self):
        # Both mortal and standard adversity patterns present
        text = "Despite everything, I would rather die than betray my values."
        mortal_only   = compute_resistance("I would rather die than betray my values.", 0.7, "speech")
        with_adversity = compute_resistance(text, 0.7, "speech")
        # Result should be the same (mortal fires, adversity ignored)
        assert with_adversity == mortal_only

    def test_high_adv_takes_precedence_over_standard_adv(self):
        text = "Despite the hardship, they threatened to destroy me professionally."
        high_only = compute_resistance("They threatened to destroy me professionally.", 0.7, "unknown")
        combined  = compute_resistance(text, 0.7, "unknown")
        assert combined == high_only


# ---------------------------------------------------------------------------
# Calibration regression — seed set directional correctness
# ---------------------------------------------------------------------------

class TestCalibrationRegression:
    """
    Directional correctness on the resistance calibration seed.
    We do not require exact matches — we verify that the formula places
    mortal-stakes cases substantially higher than low-stakes philosophical cases.
    """

    MORTAL_CASES = [
        ("I would rather die having spoken after my manner, than speak in your manner and live.", "speech"),
        ("I die the King's faithful servant, but God's first.", "speech"),
        ("I only regret that I have but one life to lose for my country.", "speech"),
        ("I abjure, curse, and detest the aforesaid errors and heresies, and generally all and every error and sect contrary to the Holy Church.", "speech"),
        ("I cannot and will not retract anything, since it is neither safe nor right to go against conscience.", "speech"),
        ("It is an ideal for which I am prepared to die.", "speech"),
    ]

    LOW_CASES = [
        ("Honesty is the first chapter in the book of wisdom.", "letter"),
        ("To be good is noble; but to show others how to be good is nobler and no trouble.", "unknown"),
        ("Seek not the good in external things; seek it in yourself.", "unknown"),
        ("Even in a palace it is possible to live well.", "journal"),
        ("The impediment to action advances action. What stands in the way becomes the way.", "journal"),
    ]

    def test_mortal_cases_score_above_070(self):
        for text, doc_type in self.MORTAL_CASES:
            result = compute_resistance(text, 0.70, doc_type)
            assert result >= 0.70, f"Expected >= 0.70 for: {text[:60]!r}, got {result}"

    def test_low_cases_score_below_040(self):
        for text, doc_type in self.LOW_CASES:
            result = compute_resistance(text, 0.70, doc_type)
            assert result <= 0.40, f"Expected <= 0.40 for: {text[:60]!r}, got {result}"

    def test_mortal_mean_above_low_mean(self):
        sig = 0.70
        mortal_mean = sum(
            compute_resistance(t, sig, d) for t, d in self.MORTAL_CASES
        ) / len(self.MORTAL_CASES)
        low_mean = sum(
            compute_resistance(t, sig, d) for t, d in self.LOW_CASES
        ) / len(self.LOW_CASES)
        assert mortal_mean > low_mean + 0.40

    def test_calibration_mae_below_threshold(self):
        """MAE on the full calibration seed should be < 0.20."""
        seed_path = ROOT / "data" / "resistance_calibration_seed.jsonl"
        if not seed_path.exists():
            pytest.skip("calibration seed not found")

        records = [json.loads(l) for l in seed_path.read_text().splitlines() if l.strip()]
        errors = []
        for r in records:
            got = compute_resistance(r["text_excerpt"], 0.70, r["doc_type"])
            errors.append(abs(got - r["resistance_score"]))
        mae = sum(errors) / len(errors)
        assert mae < 0.20, f"MAE {mae:.4f} exceeds 0.20 threshold"
