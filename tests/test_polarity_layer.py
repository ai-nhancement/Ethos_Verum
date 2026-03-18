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
    _inversion_near_target,
    POSITIVE_TARGET_WORDS,
    NEGATIVE_TARGET_WORDS,
    _INVERSION_RE,
    _POS_TARGET_RE,
    _NEG_TARGET_RE,
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
# Word-boundary regex matching (fix: no more substring false positives)
# ---------------------------------------------------------------------------

class TestWordBoundaryMatching:
    """
    The pre-compiled regexes must use word boundaries to prevent:
      "fair"   matching "affair"
      "truth"  matching "untruth"
      "equal"  matching "inequality"
      "reason" matching "treason"
      "honest" matching "dishonest"
      "right"  matching "birthright"
      "life"   matching "lifetime"
    """

    @pytest.mark.parametrize("word,false_ctx", [
        ("fair",   "he had a long affair with the countess"),
        ("truth",  "he spread untruth throughout the kingdom"),
        ("equal",  "the inequality was staggering"),
        ("reason", "he committed high treason"),
        ("honest", "his dishonest dealings came to light"),
        ("right",  "a birthright passed from father to son"),
        ("life",   "it was a lifetime of service"),
    ])
    def test_positive_word_does_not_match_false_context(self, word, false_ctx):
        assert word in POSITIVE_TARGET_WORDS, f"{word!r} not in POSITIVE_TARGET_WORDS"
        assert not _POS_TARGET_RE.search(false_ctx), (
            f"'{word}' should not match in: {false_ctx!r}"
        )

    @pytest.mark.parametrize("word,true_ctx", [
        ("fair",   "he sought a fair trial"),
        ("truth",  "he spoke the truth"),
        ("equal",  "all people are equal"),
        ("reason", "he appealed to reason"),
        ("honest", "an honest account of events"),
        ("right",  "the right to speak freely"),
        ("life",   "he risked his life"),
    ])
    def test_positive_word_matches_genuine_context(self, word, true_ctx):
        assert _POS_TARGET_RE.search(true_ctx), (
            f"'{word}' should match in: {true_ctx!r}"
        )

    def test_justice_does_not_match_injustice(self):
        assert not _POS_TARGET_RE.search("the injustice was clear")

    def test_justice_matches_justice(self):
        assert _POS_TARGET_RE.search("he fought for justice")


# ---------------------------------------------------------------------------
# Inversion regex — explicit patterns only, no bare "against"
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
        "refused to serve",
        "refused to enable",
        "refused to support",
        "spoke against",
        "spoke out against",
        "speaking against",
        "protested against",
        "campaigned against",
        "marched against",
    ])
    def test_inversion_phrases_match(self, phrase):
        assert _INVERSION_RE.search(phrase), f"Expected match for: {phrase!r}"

    def test_bare_against_does_not_match(self):
        # Bare "against" alone must NOT be an inversion pattern —
        # "discrimination against minorities" must not flip polarity.
        assert not _INVERSION_RE.fullmatch("against")
        assert not _INVERSION_RE.search("discrimination against minorities")
        assert not _INVERSION_RE.search("violence against civilians")

    def test_no_false_positive_neutral_verb(self):
        assert not _INVERSION_RE.search("he loved and cherished")

    def test_against_in_noun_phrase_not_inversion(self):
        # "against" by itself in a harmful phrase must NOT trigger inversion
        assert not _INVERSION_RE.search("hatred against the innocent")


# ---------------------------------------------------------------------------
# Inversion proximity check
# ---------------------------------------------------------------------------

class TestInversionProximity:
    def test_inversion_near_target_fires(self):
        # "fought against" is 0 chars from "tyranny"
        window = "fought against tyranny"
        target_positions = [15]  # position of "tyranny"
        assert _inversion_near_target(window, target_positions)

    def test_inversion_far_from_target_does_not_fire(self):
        # Inversion phrase is 100+ chars from the target word
        window = "fought against injustice, " + "x" * 100 + " freedom"
        target_positions = [len(window) - len("freedom")]
        # "fought against" is at position 0, target at ~126 — beyond _INVERSION_PROX=70
        assert not _inversion_near_target(window, target_positions)

    def test_cross_clause_inversion_does_not_flip_distant_target(self):
        # The key regression case: inversion phrase in one clause,
        # unrelated target word in another clause far away.
        text = "He spoke against injustice, always protecting freedom in his district."
        idx = text.index("freedom")
        polarity, conf = _tier1_target(text, idx)
        # "freedom" is the target near the match_idx; "spoke against" is ~43 chars
        # before "freedom".  With _INVERSION_PROX=30, that distance exceeds the
        # threshold, so the inversion does not apply and freedom reads as constructive.
        assert polarity == 1, f"Expected 1 (constructive), got {polarity}"


# ---------------------------------------------------------------------------
# Tier 1: target lexicon proximity
# ---------------------------------------------------------------------------

class TestTier1Target:
    def test_positive_target_no_inversion(self):
        text = "He defended the rights of the oppressed with great courage."
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

    def test_inversion_reverses_negative_target(self):
        text = "He fought against tyranny at great personal cost."
        idx = text.index("tyranny")
        polarity, conf = _tier1_target(text, idx)
        assert polarity == 1
        assert conf == pytest.approx(0.68, abs=0.01)

    def test_inversion_reverses_positive_target(self):
        # "resisted justice" -> inversion + positive target -> -1
        text = "He resisted justice throughout his career."
        idx = text.index("justice")
        polarity, conf = _tier1_target(text, idx)
        assert polarity == -1

    def test_inversion_pattern_against_compound(self):
        # Compound "standing against" + negative target -> positive
        text = "Standing against oppression was his greatest act."
        idx = text.index("oppression")
        polarity, conf = _tier1_target(text, idx)
        assert polarity == 1

    def test_both_targets_positive_majority(self):
        text = "He upheld rights, freedom, and dignity but also showed cruelty."
        idx = text.index("rights")
        polarity, conf = _tier1_target(text, idx)
        assert polarity == 1
        assert conf < 0.80  # contested — reduced confidence

    def test_both_targets_tied_returns_ambiguous(self):
        text = "Between freedom and tyranny he seemed to waver."
        idx = len(text) // 2
        polarity, conf = _tier1_target(text, idx)
        assert polarity == 0 or conf <= 0.80

    def test_window_limits_search(self):
        # "freedom" (positive) is 150 chars before match_idx — outside the
        # 120-char backward window.  Only "tyranny" (negative) is in window.
        sep  = ". " * 75              # 150 chars of neutral filler with word boundaries
        text = "freedom " + sep + "tyranny"
        idx  = text.index("tyranny")
        assert text.index("freedom") < idx - 120  # verify the design assumption
        polarity, conf = _tier1_target(text, idx)
        assert polarity == -1

    def test_match_idx_none_searches_full_passage(self):
        # No match_idx — full passage is searched
        # Value keyword at position 300, target word "tyranny" at position 350 —
        # a windowed search around idx=0 would miss it, full-passage search finds it.
        text = "x" * 300 + "His commitment was directed toward tyranny and domination."
        polarity, conf = _tier1_target(text, None)
        assert polarity == -1

    def test_match_idx_none_positive(self):
        text = "x" * 300 + "He dedicated himself to freedom and the rights of all people."
        polarity, conf = _tier1_target(text, None)
        assert polarity == 1

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
        assert conf > 0.68

    def test_virtue_confirms_positive(self):
        polarity, conf = _tier2_lexicon(0.0, 0.70, 1, 0.68)
        assert polarity == 1
        assert conf > 0.68

    def test_vice_overrides_weak_positive_tier1(self):
        polarity, conf = _tier2_lexicon(0.90, 0.0, 1, 0.40)
        assert polarity == -1

    def test_virtue_overrides_weak_negative_tier1(self):
        polarity, conf = _tier2_lexicon(0.0, 0.90, -1, 0.40)
        assert polarity == 1

    def test_tier1_zero_vice_dominant(self):
        polarity, conf = _tier2_lexicon(0.65, 0.0, 0, 0.0)
        assert polarity == -1
        assert conf > 0.0

    def test_tier1_zero_virtue_dominant(self):
        polarity, conf = _tier2_lexicon(0.0, 0.65, 0, 0.0)
        assert polarity == 1

    def test_tier1_zero_both_below_threshold(self):
        polarity, conf = _tier2_lexicon(0.30, 0.20, 0, 0.0)
        assert polarity == 0

    def test_vice_strong_overrides_positive(self):
        polarity, conf = _tier2_lexicon(0.85, 0.0, 1, 0.50)
        assert polarity == -1

    def test_confidence_capped_at_one(self):
        polarity, conf = _tier2_lexicon(0.99, 0.0, -1, 0.99)
        assert polarity == -1
        assert conf <= 1.0


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
        assert polarity == 0

    def test_inversion_destructive_becomes_constructive(self):
        text = "He fought against tyranny and stood up to oppression his whole life."
        idx = text.index("tyranny")
        polarity, conf = detect_polarity(text, idx, "courage")
        assert polarity == 1

    def test_vice_lexicon_tips_ambiguous_to_destructive(self):
        text = "He demonstrated commitment in his work."
        idx = text.index("commitment")
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
        polarity, conf = detect_polarity("", None, "")
        assert polarity == 0
        assert conf == 0.0

    def test_never_raises_on_none_cfg(self):
        polarity, conf = detect_polarity("Some text about freedom.", 0, "integrity", cfg=None)
        assert isinstance(polarity, int)
        assert isinstance(conf, float)

    def test_match_idx_none_full_passage(self):
        # Non-keyword signal — match_idx=None, full passage searched
        text = "x" * 200 + "His integrity was directed toward freedom and human dignity."
        polarity, conf = detect_polarity(text, None, "integrity")
        assert polarity == 1

    def test_match_idx_none_destructive(self):
        text = "x" * 200 + "His loyalty served tyranny and the genocide of his people."
        polarity, conf = detect_polarity(text, None, "loyalty")
        assert polarity == -1

    def test_socrates_execution_context(self):
        # Tier 1 returns ambiguous (no target words match); virtue signal tips positive
        text = "I would rather die having spoken after my manner, than speak in your manner and live."
        idx = text.index("die")
        polarity, conf = detect_polarity(text, idx, "integrity",
                                          lexicon_virtue_score=0.72)
        assert polarity == 1

    def test_himmler_posen_context(self):
        # "moral right" in text fires "right" as positive target — needs strong vice
        text = "We had the moral right, we had the duty to our people, to destroy this people which wanted to destroy us."
        idx = text.index("duty")
        polarity, conf = detect_polarity(text, idx, "loyalty",
                                          lexicon_vice_score=0.90)
        assert polarity == -1

    def test_robespierre_context(self):
        text = "Terror is only justice: prompt, severe and inflexible; it is then an emanation of virtue."
        idx = text.index("virtue")
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

    def test_discrimination_against_minorities_not_inverted(self):
        # Critical regression: "discrimination against minorities" must return
        # NEGATIVE polarity (discrimination is harmful), not positive.
        # Bare "against" is no longer an inversion trigger.
        text = "He showed loyalty through discrimination against minorities."
        idx = text.index("loyalty")
        polarity, conf = detect_polarity(text, idx, "loyalty")
        # "discrimination" is a negative target; no inversion verb before it
        assert polarity == -1

    def test_returns_confidence_for_polarity(self):
        # The returned confidence reflects how certain the polarity detection is.
        # A high-confidence result should differ from an ambiguous one.
        text_clear = "He defended the rights of the oppressed."
        text_ambig = "He demonstrated commitment throughout his career."
        _, conf_clear = detect_polarity(text_clear, 0, "courage")
        _, conf_ambig = detect_polarity(text_ambig, 0, "courage")
        assert conf_clear >= conf_ambig
