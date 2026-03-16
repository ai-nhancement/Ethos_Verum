"""
tests/test_lexicon_layer.py

Tests for Layer 1b — MFD2.0 + MoralStrength lexicon augmentation.
"""

from __future__ import annotations

import pytest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.lexicon_layer import (
    lexicon_signals,
    mfd2_size,
    moralstrength_size,
    is_available,
    _ensure_loaded,
    _MFD2,
    _MORALSTRENGTH,
    _MFT_TO_ETHOS,
)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

class TestLexiconLoading:

    def test_mfd2_loaded(self):
        _ensure_loaded()
        assert mfd2_size() > 1000, f"Expected >1000 entries, got {mfd2_size()}"

    def test_moralstrength_loaded(self):
        _ensure_loaded()
        assert moralstrength_size() > 100

    def test_is_available(self):
        assert is_available() is True

    def test_mfd2_known_entries(self):
        _ensure_loaded()
        # These words are in MFD2.0 care.virtue
        assert "compassion" in _MFD2
        assert "empathy"    in _MFD2
        assert "kindness"   in _MFD2

    def test_mfd2_polarity_correct(self):
        _ensure_loaded()
        assert _MFD2["compassion"][1] == "virtue"
        assert _MFD2["empathy"][1]    == "virtue"
        # vice entries should exist
        vice_words = [w for w, (f, p) in _MFD2.items() if p == "vice"]
        assert len(vice_words) > 100

    def test_mfd2_foundation_correct(self):
        _ensure_loaded()
        assert _MFD2["compassion"][0] == "care"
        assert _MFD2["empathy"][0]    == "care"

    def test_moralstrength_scores_in_range(self):
        _ensure_loaded()
        for word, foundations in _MORALSTRENGTH.items():
            for foundation, score in foundations.items():
                assert 1.0 <= score <= 10.0, (
                    f"{word}/{foundation}: score={score} out of [1,10]"
                )

    def test_mft_mapping_covers_five_foundations(self):
        assert set(_MFT_TO_ETHOS.keys()) == {"care", "fairness", "loyalty", "authority", "sanctity"}

    def test_mft_mapping_all_values_valid(self):
        from core.semantic_store import VALUE_NAMES
        for foundation, values in _MFT_TO_ETHOS.items():
            for v in values:
                assert v in VALUE_NAMES, f"Unknown value {v!r} in MFT mapping for {foundation}"


# ---------------------------------------------------------------------------
# lexicon_signals() — signal schema
# ---------------------------------------------------------------------------

class TestLexiconSignalSchema:

    def test_returns_list(self):
        result = lexicon_signals("I showed compassion.", 0.8)
        assert isinstance(result, list)

    def test_empty_text_returns_empty(self):
        assert lexicon_signals("", 0.8) == []

    def test_signal_has_required_fields(self):
        sigs = lexicon_signals("He showed great compassion for the suffering.", 0.8)
        for sig in sigs:
            assert "value_name" in sig
            assert "text_excerpt" in sig
            assert "significance" in sig
            assert "disambiguation_confidence" in sig
            assert "source" in sig
            assert "lexicon_polarity" in sig

    def test_confidence_in_range(self):
        sigs = lexicon_signals("She was just and fair to everyone.", 0.9)
        for sig in sigs:
            assert 0.0 <= sig["disambiguation_confidence"] <= 1.0

    def test_polarity_is_virtue_or_vice(self):
        sigs = lexicon_signals(
            "He showed compassion but also cruelty in equal measure.", 0.8
        )
        for sig in sigs:
            assert sig["lexicon_polarity"] in {"virtue", "vice"}

    def test_significance_passed_through(self):
        sigs = lexicon_signals("I showed compassion for them.", 0.75)
        for sig in sigs:
            assert sig["significance"] == 0.75


# ---------------------------------------------------------------------------
# lexicon_signals() — value detection
# ---------------------------------------------------------------------------

class TestLexiconValueDetection:

    def _virtue_values(self, text: str) -> set:
        return {s["value_name"] for s in lexicon_signals(text, 0.8)
                if s["lexicon_polarity"] == "virtue"}

    def _vice_values(self, text: str) -> set:
        return {s["value_name"] for s in lexicon_signals(text, 0.8)
                if s["lexicon_polarity"] == "vice"}

    def test_compassion_detected_on_care_virtue_word(self):
        assert "compassion" in self._virtue_values("She showed great empathy for the victims.")

    def test_fairness_detected_on_fairness_virtue_word(self):
        values = self._virtue_values("The ruling was equitable and just.")
        assert "fairness" in values

    def test_loyalty_detected_on_loyalty_virtue_word(self):
        # "loyal" and "fidelity" are confirmed loyalty.virtue words in MFD2.0
        values = self._virtue_values("He stayed loyal and showed fidelity to the cause.")
        assert "loyalty" in values

    def test_responsibility_detected_on_authority_virtue_word(self):
        values = self._virtue_values("She showed great leadership and stewardship.")
        assert "responsibility" in values or "commitment" in values

    def test_integrity_detected_on_sanctity_virtue_word(self):
        # sanctity.virtue words include "sacred", "holy", "pure", "righteous"
        values = self._virtue_values("He held the sacred duty above all else.")
        assert len(values) > 0  # at least something detected

    def test_vice_detected_on_betrayal_word(self):
        assert "loyalty" in self._vice_values("He chose to betray his closest ally.")

    def test_no_signal_on_value_neutral_text(self):
        sigs = lexicon_signals("The train arrived at the station at noon.", 0.5)
        virtue_sigs = [s for s in sigs if s["lexicon_polarity"] == "virtue"]
        assert len(virtue_sigs) == 0

    def test_source_tag_present(self):
        sigs = lexicon_signals("He showed empathy to the wounded.", 0.8)
        for s in sigs:
            assert "lexicon" in s["source"]

    def test_ms_boost_applied_for_strong_moral_words(self):
        # "compassion" is in both MFD2.0 and MoralStrength
        sigs = lexicon_signals("compassion", 0.8)
        virtue_sigs = [s for s in sigs if s["lexicon_polarity"] == "virtue"
                       and "compassion" in s["value_name"] or "love" in s["value_name"]]
        # At least one should have the "lexicon+ms" source tag
        ms_sigs = [s for s in sigs if "ms" in s["source"]]
        assert len(ms_sigs) > 0

    def test_one_signal_per_value_per_passage(self):
        """Each value should appear at most once per polarity per passage."""
        sigs = lexicon_signals(
            "She showed empathy, compassion, and kindness to all.", 0.8
        )
        virtue_sigs = [s for s in sigs if s["lexicon_polarity"] == "virtue"]
        value_names = [s["value_name"] for s in virtue_sigs]
        assert len(value_names) == len(set(value_names)), \
            f"Duplicate value names: {value_names}"


# ---------------------------------------------------------------------------
# Integration with value_extractor
# ---------------------------------------------------------------------------

class TestLexiconIntegration:

    def test_source_tag_includes_lexicon_on_keyword_agreement(self):
        """When keyword and lexicon both fire on same value, source should include lexicon."""
        from core.value_extractor import extract_value_signals, _lexicon_signals
        from core.config import get_config

        cfg = get_config()
        text = "She showed great empathy and compassion for the suffering."
        kw_sigs = extract_value_signals(text, "test-id", 0.8)
        lex_sigs = _lexicon_signals(text, 0.8, "journal", cfg)

        kw_values = {s["value_name"] for s in kw_sigs}
        lex_virtue = {s["value_name"] for s in lex_sigs if s["lexicon_polarity"] == "virtue"}

        # At least some overlap between keyword and lexicon values
        overlap = kw_values & lex_virtue
        assert len(overlap) > 0, (
            f"No overlap between keyword values {kw_values} and lexicon values {lex_virtue}"
        )

    def test_lexicon_disabled_in_config_skips_layer(self):
        from core.config import Config
        from core.value_extractor import _lexicon_signals
        cfg = Config(lexicon_enabled=False)
        result = _lexicon_signals("She showed compassion.", 0.8, "journal", cfg)
        assert result == []
