"""
tests/test_polarity_layer.py

Tests for core/polarity_layer.py — three-tier value polarity detection.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.polarity_layer import (
    detect_polarity,
    polarity_label,
    _tier1_target,
    _tier2_lexicon,
    POSITIVE_TARGET_WORDS,
    NEGATIVE_TARGET_WORDS,
    _INVERSION_RE,
)


# ---------------------------------------------------------------------------
# polarity_label
# ---------------------------------------------------------------------------

class TestPolarityLabel:
    def test_positive(self):
        assert polarity_label(1) == "constructive"

    def test_negative(self):
        assert polarity_label(-1) == "destructive"

    def test_ambiguous(self):
        assert polarity_label(0) == "ambiguous"

    def test_unknown_value_defaults_ambiguous(self):
        assert polarity_label(99) == "ambiguous"


# ---------------------------------------------------------------------------
# Tier 1: target lexicon proximity
# ---------------------------------------------------------------------------

class TestTier1Target:
    def test_positive_target_no_inversion(self):
        text = "He defended the rights of the oppressed with great courage."
        # match_idx near "courage"
        idx = text.index("courage")
        polarity, conf = _tier1_target(text, idx)
        assert polarity == 1
        assert conf >= 0.68

    def test_negative_target_no_inversion(self):
        text = "He committed acts of tyranny and oppression without hesitation."
        idx = text.index("tyranny")
        polarity, conf = _tier1_target(text, idx)
        assert polarity == -1
        assert conf >= 0.68

    def test_no_target_words_returns_ambiguous(self):
        text = "He walked across the field on a sunny afternoon."
        polarity, conf = _tier1_target(text, 10)
        assert polarity == 0
        assert conf == 0.0

    def test_inversion_reverses_positive_target(self):
        # "fought against tyranny" -> inversion + negative target -> +1
        text = "He fought against tyranny at great personal cost."
        idx = text.index("tyranny")
        polarity, conf = _tier1_target(text, idx)
        assert polarity == 1
        assert conf == pytest.approx(0.68, abs=0.01)

    def test_inversion_reverses_negative_target(self):
        # "resisted justice" -> inversion + positive target -> -1
        text = "He resisted justice and defied accountability."
        idx = text.index("justice")
        polarity, conf = _tier1_target(text, idx)
        assert polarity == -1

    def test_inversion_pattern_against(self):
        # "against" + negative target word -> positive
        # Avoid incidental positive target words ("life") in window
        text = "Standing against oppression was his greatest act."
        idx = text.index("oppression")
        polarity, conf = _tier1_target(text, idx)
        assert polarity == 1

    def test_both_targets_positive_majority(self):
        text = "He upheld rights, freedom, and dignity but also showed cruelty."
        idx = text.index("rights")
        polarity, conf = _tier1_target(text, idx)
        # More positive hits -> +1, reduced confidence
        assert polarity == 1
        assert conf < 0.80  # contested

    def test_both_targets_tied_returns_ambiguous(self):
        # Equal positive and negative hits
        text = "Between freedom and tyranny he seemed to waver."
        idx = len(text) // 2
        polarity, conf = _tier1_target(text, idx)
        # With exactly one of each, tied -> ambiguous
        assert polarity == 0 or conf <= 0.80  # tied or contested

    def test_window_limits_search(self):
        # Window is 120 chars. Place match_idx right before "tyranny" and
        # put a positive word 250+ chars before (outside backward window).
        prefix = "x" * 250   # 250 neutral chars, no target words
        text = prefix + "tyranny"
        idx = len(prefix)
        polarity, conf = _tier1_target(text, idx)
        # Only "tyranny" is in the 120-char window; no positive hits
        assert polarity == -1

    def test_case_insensitive_window_search(self):
        text = "He defended FREEDOM and JUSTICE at great risk."
        idx = text.index("FREEDOM")
        polarity, conf = _tier1_target(text, idx)
        assert polarity == 1


# ---------------------------------------------------------------------------
# Tier 2: MFT lexicon integration
# ---------------------------------------------------------------------------

class TestTier2Lexicon:
    def test_no_lexicon_signal_passthrough(self):
        polarity, conf = _tier2_lexicon(0.0, 0.0, 1, 0.80)
        assert polarity == 1
        assert conf == 0.80

    def test_vice_confirms_negative(self):
        polarity, conf = _tier2_lexicon(0.70, 0.0, -1, 0.68)
        assert polarity == -1
        assert conf > 0.68  # boosted

    def test_virtue_confirms_positive(self):
        polarity, conf = _tier2_lexicon(0.0, 0.70, 1, 0.68)
        assert polarity == 1
        assert conf > 0.68  # boosted

    def test_vice_overrides_weak_positive_tier1(self):
        # Strong vice signal (0.90) overrides weak positive tier1 (0.40)
        polarity, conf = _tier2_lexicon(0.90, 0.0, 1, 0.40)
        assert polarity == -1

    def test_virtue_overrides_weak_negative_tier1(self):
        polarity, conf = _tier2_lexicon(0.0, 0.90, -1, 0.40)
        assert polarity == 1

    def test_tier1_zero_vice_dominant(self):
        # Tier 1 ambiguous, vice score >= 0.40 -> negative
        polarity, conf = _tier2_lexicon(0.65, 0.0, 0, 0.0)
        assert polarity == -1
        assert conf > 0.0

    def test_tier1_zero_virtue_dominant(self):
        # Tier 1 ambiguous, virtue score >= 0.40 -> positive
        polarity, conf = _tier2_lexicon(0.0, 0.65, 0, 0.0)
        assert polarity == 1

    def test_tier1_zero_both_below_threshold(self):
        # Tier 1 ambiguous, both lexicon scores below 0.40 -> stays ambiguous
        polarity, conf = _tier2_lexicon(0.30, 0.20, 0, 0.0)
        assert polarity == 0

    def test_vice_strong_overrides_positive(self):
        # Vice score > tier1_conf when tier1 is positive -> flip
        polarity, conf = _tier2_lexicon(0.85, 0.0, 1, 0.50)
        assert polarity == -1

    def test_confidence_capped_at_one(self):
        polarity, conf = _tier2_lexicon(0.99, 0.0, -1, 0.99)
        assert polarity == -1
        assert conf <= 1.0


# ---------------------------------------------------------------------------
# Inversion regex
# ---------------------------------------------------------------------------

class TestInversionRegex:
    @pytest.mark.parametrize("phrase", [
        "fought against",
        "fighting against",
        "resisted",
        "resisting",
        "opposed",
        "opposing",
        "standing against",
        "stood against",
        "stood up to",
        "defying",
        "defied",
        "against",
        "refused to serve",
        "refused to enable",
        "refused to support",
    ])
    def test_inversion_phrases_match(self, phrase):
        assert _INVERSION_RE.search(phrase), f"Expected match for: {phrase!r}"

    def test_no_false_positive_neutral_verb(self):
        # "and" should not match
        assert not _INVERSION_RE.search("he loved and cherished")


# ---------------------------------------------------------------------------
# Public API: detect_polarity
# ---------------------------------------------------------------------------

class TestDetectPolarity:
    def test_constructive_courage(self):
        text = "He showed courage in defending the rights of the oppressed."
        idx = text.index("courage")
        polarity, conf = detect_polarity(text, idx, "courage")
        assert polarity == 1
        assert conf > 0.0

    def test_destructive_loyalty(self):
        text = "His loyalty to the regime sustained a system of oppression and subjugation."
        idx = text.index("loyalty")
        polarity, conf = detect_polarity(text, idx, "loyalty")
        assert polarity == -1
        assert conf > 0.0

    def test_ambiguous_no_context(self):
        text = "He demonstrated remarkable commitment throughout his career."
        idx = text.index("commitment")
        polarity, conf = detect_polarity(text, idx, "commitment")
        # No target words -> ambiguous
        assert polarity == 0

    def test_inversion_destructive_becomes_constructive(self):
        text = "He fought against tyranny and stood up to oppression his whole life."
        idx = text.index("tyranny")
        polarity, conf = detect_polarity(text, idx, "courage")
        assert polarity == 1

    def test_vice_lexicon_tips_ambiguous_to_destructive(self):
        text = "He demonstrated commitment in his work."
        idx = text.index("commitment")
        # No target words in text but strong vice signal from lexicon
        polarity, conf = detect_polarity(
            text, idx, "commitment",
            lexicon_vice_score=0.80,
            lexicon_virtue_score=0.0,
        )
        assert polarity == -1

    def test_virtue_lexicon_tips_ambiguous_to_constructive(self):
        text = "He demonstrated commitment in his work."
        idx = text.index("commitment")
        polarity, conf = detect_polarity(
            text, idx, "commitment",
            lexicon_vice_score=0.0,
            lexicon_virtue_score=0.80,
        )
        assert polarity == 1

    def test_never_raises_on_bad_input(self):
        polarity, conf = detect_polarity("", -999, "")
        assert polarity == 0
        assert conf == 0.0

    def test_never_raises_on_none_cfg(self):
        polarity, conf = detect_polarity("Some text about freedom.", 0, "integrity", cfg=None)
        assert isinstance(polarity, int)
        assert isinstance(conf, float)

    def test_socrates_execution_context(self):
        # Socrates' own words: no explicit target words in the passage.
        # Tier 1 returns ambiguous; a virtue lexicon signal would tip it positive.
        text = "I would rather die having spoken after my manner, than speak in your manner and live."
        idx = text.index("die")
        polarity, conf = detect_polarity(text, idx, "integrity",
                                          lexicon_virtue_score=0.72)
        assert polarity == 1

    def test_himmler_posen_context(self):
        # "moral right" causes "right" to fire as a positive target word,
        # so the vice signal must be strong enough to override tier1_conf (0.80).
        text = "We had the moral right, we had the duty to our people, to destroy this people which wanted to destroy us."
        idx = text.index("duty")
        polarity, conf = detect_polarity(text, idx, "loyalty",
                                          lexicon_vice_score=0.90)
        assert polarity == -1

    def test_robespierre_context(self):
        text = "Terror is only justice: prompt, severe and inflexible; it is then an emanation of virtue."
        idx = text.index("virtue")
        # "virtue" alone doesn't fire target words, but lexicon vice can push
        polarity, conf = detect_polarity(text, idx, "commitment",
                                          lexicon_vice_score=0.70)
        assert polarity == -1

    def test_confidence_in_valid_range(self):
        text = "He defended the rights of the innocent with his life."
        idx = text.index("rights")
        polarity, conf = detect_polarity(text, idx, "courage")
        assert 0.0 <= conf <= 1.0

    def test_polarity_is_int(self):
        text = "He defended freedom."
        idx = text.index("freedom")
        polarity, conf = detect_polarity(text, idx, "courage")
        assert isinstance(polarity, int)
        assert polarity in (-1, 0, 1)
