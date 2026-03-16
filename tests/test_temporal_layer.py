"""
tests/test_temporal_layer.py

Tests for Phase 3 — Translation and Temporal Handling.
  - Language detection (detect_language)
  - Source authenticity multipliers (source_authenticity)
  - Archaic preprocessing (preprocess_archaic)
  - pub_year discount (pub_year_discount)
  - calibrate_confidence integration
  - document_store Phase 3 columns
"""

from __future__ import annotations

import pytest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.temporal_layer import (
    detect_language,
    source_authenticity,
    preprocess_archaic,
    pub_year_discount,
    calibrate_confidence,
    AUTHENTICITY_ORIGINAL,
    AUTHENTICITY_TRANSLATION,
    AUTHENTICITY_UNCERTAIN,
)


# ---------------------------------------------------------------------------
# Language detection
# ---------------------------------------------------------------------------

class TestDetectLanguage:

    def test_english_detected(self):
        lang, conf = detect_language("I walked to the market and bought bread.")
        assert lang == "en"
        assert conf > 0.5

    def test_german_detected(self):
        lang, conf = detect_language(
            "Ich habe heute morgen einen langen Spaziergang gemacht."
        )
        assert lang == "de"
        assert conf > 0.5

    def test_french_detected(self):
        lang, conf = detect_language(
            "Nous avons discuté pendant des heures de la situation politique."
        )
        assert lang == "fr"
        assert conf > 0.5

    def test_returns_tuple(self):
        result = detect_language("Hello world.")
        assert len(result) == 2
        assert isinstance(result[0], str)
        assert isinstance(result[1], float)

    def test_empty_text_returns_unknown(self):
        lang, conf = detect_language("")
        assert lang == "unknown"
        assert conf == 0.0

    def test_confidence_in_range(self):
        _, conf = detect_language("The quick brown fox jumps over the lazy dog.")
        assert 0.0 <= conf <= 1.0

    def test_deterministic_output(self):
        """Same text should produce same result every time (seeded)."""
        text = "I stood firm and refused to yield despite the threat."
        r1 = detect_language(text)
        r2 = detect_language(text)
        assert r1 == r2


# ---------------------------------------------------------------------------
# Source authenticity
# ---------------------------------------------------------------------------

class TestSourceAuthenticity:

    def test_explicit_translation_returns_translation_multiplier(self):
        assert source_authenticity("en",  is_translation=True)  == AUTHENTICITY_TRANSLATION
        assert source_authenticity("fr",  is_translation=True)  == AUTHENTICITY_TRANSLATION
        assert source_authenticity("de",  is_translation=True)  == AUTHENTICITY_TRANSLATION
        assert source_authenticity("unknown", is_translation=True) == AUTHENTICITY_TRANSLATION

    def test_explicit_original_returns_original_multiplier(self):
        assert source_authenticity("en",  is_translation=False) == AUTHENTICITY_ORIGINAL
        assert source_authenticity("fr",  is_translation=False) == AUTHENTICITY_ORIGINAL
        assert source_authenticity("unknown", is_translation=False) == AUTHENTICITY_ORIGINAL

    def test_english_inferred_as_original(self):
        assert source_authenticity("en",  is_translation=None) == AUTHENTICITY_ORIGINAL

    def test_non_english_inferred_as_uncertain(self):
        for lang in ["fr", "de", "la", "el", "zh", "ar"]:
            result = source_authenticity(lang, is_translation=None)
            assert result == AUTHENTICITY_UNCERTAIN, f"Expected uncertain for lang={lang}"

    def test_unknown_lang_returns_uncertain(self):
        assert source_authenticity("unknown", is_translation=None) == AUTHENTICITY_UNCERTAIN

    def test_multipliers_are_ordered(self):
        assert AUTHENTICITY_TRANSLATION < AUTHENTICITY_ORIGINAL
        assert AUTHENTICITY_UNCERTAIN < AUTHENTICITY_TRANSLATION

    def test_all_multipliers_in_range(self):
        for m in [AUTHENTICITY_ORIGINAL, AUTHENTICITY_TRANSLATION, AUTHENTICITY_UNCERTAIN]:
            assert 0.0 < m <= 1.0


# ---------------------------------------------------------------------------
# Archaic preprocessing
# ---------------------------------------------------------------------------

class TestPreprocessArchaic:

    def test_thou_replaced(self):
        assert "you" in preprocess_archaic("Thou art brave.").lower()

    def test_thee_replaced(self):
        result = preprocess_archaic("I give thee this gift.")
        assert "thee" not in result.lower()
        assert "you" in result.lower()

    def test_thy_replaced(self):
        result = preprocess_archaic("Thy courage is great.")
        assert "thy" not in result.lower()
        assert "your" in result.lower()

    def test_hath_replaced(self):
        result = preprocess_archaic("He hath spoken the truth.")
        assert "has" in result.lower()
        assert "hath" not in result.lower()

    def test_doth_replaced(self):
        result = preprocess_archaic("He doth protest too much.")
        assert "does" in result.lower()

    def test_art_replaced(self):
        result = preprocess_archaic("Thou art just and true.")
        assert "art" not in result.lower() or "your" in result.lower()

    def test_twas_replaced(self):
        result = preprocess_archaic("'Twas the best of times.")
        assert "it was" in result.lower()

    def test_tis_replaced(self):
        result = preprocess_archaic("'Tis a truth universally acknowledged.")
        assert "it is" in result.lower()

    def test_shouldst_replaced(self):
        result = preprocess_archaic("Thou shouldst know better.")
        assert "should" in result.lower()

    def test_modern_text_unchanged(self):
        modern = "I stood firm and refused to yield despite everything."
        result = preprocess_archaic(modern)
        assert result == modern

    def test_empty_string_safe(self):
        assert preprocess_archaic("") == ""

    def test_mixed_archaic_and_modern(self):
        text = "I know thou art brave, and I trust you."
        result = preprocess_archaic(text)
        assert "thou" not in result.lower()
        assert "art" not in result.lower() or "you are" in result.lower()

    def test_preserves_capitalization(self):
        result = preprocess_archaic("Thou art courageous.")
        # "Thou" → "You" (capital preserved)
        assert result[0].isupper()

    def test_nay_replaced(self):
        result = preprocess_archaic("Nay, I will not yield.")
        assert "no" in result.lower()

    def test_historical_passage(self):
        """End-to-end test on a plausible archaic passage."""
        text = (
            "Thou hast spoken truly. 'Tis not for thee to decide "
            "what thy nation doth require of thee in this hour."
        )
        result = preprocess_archaic(text)
        assert "hath" not in result.lower()
        assert "thou" not in result.lower()
        assert "thee" not in result.lower()
        assert "thy" not in result.lower()
        # Meaning preserved
        assert "truly" in result
        assert "nation" in result


# ---------------------------------------------------------------------------
# pub_year discount
# ---------------------------------------------------------------------------

class TestPubYearDiscount:

    def test_none_returns_no_discount(self):
        assert pub_year_discount(None) == 1.0

    def test_modern_returns_no_discount(self):
        for year in [1850, 1900, 1963, 2000, 2024]:
            assert pub_year_discount(year) == 1.0

    def test_pre_1850_returns_discount(self):
        assert pub_year_discount(1800) < 1.0
        assert pub_year_discount(1600) < 1.0
        assert pub_year_discount(1400) < 1.0

    def test_discount_increases_with_age(self):
        """Older documents get larger discounts."""
        d1 = pub_year_discount(1840)   # near modern
        d2 = pub_year_discount(1750)   # 18th century
        d3 = pub_year_discount(1550)   # Early Modern
        d4 = pub_year_discount(1200)   # Medieval
        assert d1 >= d2 >= d3 >= d4

    def test_results_in_valid_range(self):
        for year in [0, 500, 1000, 1400, 1600, 1700, 1800, 1850, 2000]:
            result = pub_year_discount(year)
            assert 0.0 < result <= 1.0, f"year={year}: {result}"

    def test_boundary_1850_is_full(self):
        assert pub_year_discount(1850) == 1.0

    def test_just_before_boundary_is_discounted(self):
        assert pub_year_discount(1849) < 1.0


# ---------------------------------------------------------------------------
# calibrate_confidence
# ---------------------------------------------------------------------------

class TestCalibrateConfidence:

    def test_full_multipliers_unchanged(self):
        result = calibrate_confidence(0.8, 1.0, 1.0)
        assert result == pytest.approx(0.8)

    def test_translation_reduces_confidence(self):
        original  = calibrate_confidence(0.8, AUTHENTICITY_ORIGINAL,    1.0)
        translated = calibrate_confidence(0.8, AUTHENTICITY_TRANSLATION, 1.0)
        assert translated < original

    def test_temporal_discount_reduces_confidence(self):
        modern  = calibrate_confidence(0.8, 1.0, 1.0)
        ancient = calibrate_confidence(0.8, 1.0, 0.70)
        assert ancient < modern

    def test_both_multipliers_compound(self):
        both = calibrate_confidence(1.0, AUTHENTICITY_TRANSLATION, 0.80)
        assert both == pytest.approx(0.85 * 0.80, abs=1e-4)

    def test_clipped_to_1(self):
        result = calibrate_confidence(1.0, 1.0, 1.0)
        assert result <= 1.0

    def test_clipped_to_0(self):
        result = calibrate_confidence(0.0, 0.5, 0.5)
        assert result >= 0.0

    def test_result_is_float(self):
        assert isinstance(calibrate_confidence(0.7, 0.85, 0.93), float)


# ---------------------------------------------------------------------------
# DocumentStore Phase 3 columns
# ---------------------------------------------------------------------------

class TestDocumentStorePhase3:

    def _make_store(self):
        from core.document_store import DocumentStore
        return DocumentStore(db_path=":memory:")

    def test_insert_with_phase3_fields(self):
        store = self._make_store()
        pid = store.insert_passage(
            figure_name="seneca",
            session_id="figure:seneca",
            text="I choose virtue over safety at great personal cost.",
            doc_type="letter",
            significance=0.9,
            source_lang="la",
            source_authenticity=0.85,
            pub_year=65,
        )
        assert pid is not None

    def test_phase3_fields_persisted(self):
        store = self._make_store()
        store.insert_passage(
            figure_name="seneca",
            session_id="figure:seneca",
            text="I choose virtue over safety.",
            doc_type="letter",
            source_lang="la",
            source_authenticity=0.85,
            pub_year=65,
        )
        rows = store.get_passages_since("figure:seneca", -1e12)
        assert len(rows) == 1
        row = rows[0]
        assert row["source_lang"] == "la"
        assert float(row["source_authenticity"]) == pytest.approx(0.85)
        assert row["pub_year"] == 65

    def test_defaults_for_missing_phase3_fields(self):
        store = self._make_store()
        store.insert_passage(
            figure_name="lincoln",
            session_id="figure:lincoln",
            text="We shall not perish from the earth.",
            doc_type="speech",
        )
        rows = store.get_passages_since("figure:lincoln", -1e12)
        row = rows[0]
        assert row["source_lang"] == "unknown"
        assert float(row["source_authenticity"]) == pytest.approx(1.0)
        assert row["pub_year"] is None
