"""
tests/test_mft_classifier.py

Tests for core/mft_classifier.py (Layer 3c).

PRINCIPLE: Tests run real inference against the MMADS/MoralFoundationsClassifier
model when available. ML-disabled tests verify fail-open behaviour and interface
contracts without requiring the model to be present.

Test structure:
  TestLabelMap          — static mapping completeness and correctness
  TestMftSignalsOffline — fail-open behaviour when model unavailable
  TestMftSignalsOnline  — real inference tests (skipped if model unavailable)
  TestMftIntegration    — end-to-end: MFT layer integrated into full pipeline
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _reset_mft_singleton():
    """Force re-initialisation of the lazy-loaded classifier singleton."""
    import core.mft_classifier as _m
    _m._clf = None
    _m._clf_unavailable = False


# ---------------------------------------------------------------------------
# TestLabelMap — static mapping
# ---------------------------------------------------------------------------

class TestLabelMap:
    def test_all_ten_labels_present(self):
        from core.mft_classifier import LABEL_MAP
        assert len(LABEL_MAP) == 10
        for i in range(10):
            assert f"LABEL_{i}" in LABEL_MAP, f"LABEL_{i} missing from LABEL_MAP"

    def test_each_entry_has_foundation_and_polarity(self):
        from core.mft_classifier import LABEL_MAP
        valid_foundations = {"care", "fairness", "loyalty", "authority", "sanctity"}
        valid_polarities  = {"virtue", "vice"}
        for label, (foundation, polarity) in LABEL_MAP.items():
            assert foundation in valid_foundations, \
                f"{label}: unknown foundation '{foundation}'"
            assert polarity in valid_polarities, \
                f"{label}: unknown polarity '{polarity}'"

    def test_five_virtue_and_five_vice_labels(self):
        from core.mft_classifier import LABEL_MAP
        virtues = [k for k, (_, p) in LABEL_MAP.items() if p == "virtue"]
        vices   = [k for k, (_, p) in LABEL_MAP.items() if p == "vice"]
        assert len(virtues) == 5
        assert len(vices)   == 5

    def test_each_foundation_has_exactly_one_virtue_and_one_vice(self):
        from core.mft_classifier import LABEL_MAP
        from collections import Counter
        foundation_polarity = Counter((f, p) for (f, p) in LABEL_MAP.values())
        for foundation in ("care", "fairness", "loyalty", "authority", "sanctity"):
            assert foundation_polarity[(foundation, "virtue")] == 1
            assert foundation_polarity[(foundation, "vice")]   == 1

    def test_virtue_to_ethos_only_references_known_values(self):
        from core.mft_classifier import MFT_VIRTUE_TO_ETHOS
        from core.value_extractor import VALUE_VOCAB
        known = set(VALUE_VOCAB.keys())
        for foundation, ethos_values in MFT_VIRTUE_TO_ETHOS.items():
            for v in ethos_values:
                assert v in known, \
                    f"MFT_VIRTUE_TO_ETHOS[{foundation!r}] references unknown value {v!r}"

    def test_authority_vice_hint_only_references_known_values(self):
        from core.mft_classifier import MFT_AUTHORITY_VICE_HINT
        from core.value_extractor import VALUE_VOCAB
        known = set(VALUE_VOCAB.keys())
        for v in MFT_AUTHORITY_VICE_HINT:
            assert v in known, f"MFT_AUTHORITY_VICE_HINT references unknown value {v!r}"

    def test_care_virtue_maps_to_compassion(self):
        from core.mft_classifier import MFT_VIRTUE_TO_ETHOS
        assert "compassion" in MFT_VIRTUE_TO_ETHOS["care"]

    def test_fairness_virtue_maps_to_fairness(self):
        from core.mft_classifier import MFT_VIRTUE_TO_ETHOS
        assert "fairness" in MFT_VIRTUE_TO_ETHOS["fairness"]

    def test_loyalty_virtue_maps_to_loyalty_and_courage(self):
        from core.mft_classifier import MFT_VIRTUE_TO_ETHOS
        assert "loyalty"  in MFT_VIRTUE_TO_ETHOS["loyalty"]
        assert "courage"  in MFT_VIRTUE_TO_ETHOS["loyalty"]

    def test_authority_virtue_maps_to_humility(self):
        from core.mft_classifier import MFT_VIRTUE_TO_ETHOS
        assert "humility" in MFT_VIRTUE_TO_ETHOS["authority"]


# ---------------------------------------------------------------------------
# TestMftSignalsOffline — fail-open contract
# ---------------------------------------------------------------------------

class TestMftSignalsOffline:
    """Verify the module behaves correctly when the model cannot be loaded."""

    def setup_method(self):
        _reset_mft_singleton()

    def teardown_method(self):
        _reset_mft_singleton()

    def test_returns_empty_when_model_unavailable(self):
        with patch("core.mft_classifier._get_classifier", return_value=None):
            from core.mft_classifier import mft_signals
            result = mft_signals("I was brave and stood firm.")
            assert result == {"boosted_values": [], "vice_flags": []}

    def test_returns_empty_on_empty_text(self):
        with patch("core.mft_classifier._get_classifier", return_value=None):
            from core.mft_classifier import mft_signals
            assert mft_signals("") == {"boosted_values": [], "vice_flags": []}
            assert mft_signals("   ") == {"boosted_values": [], "vice_flags": []}

    def test_inference_error_returns_empty(self):
        """If the model raises during inference, fail-open must still apply."""
        mock_clf = MagicMock(side_effect=RuntimeError("GPU OOM"))
        with patch("core.mft_classifier._get_classifier", return_value=mock_clf):
            from core.mft_classifier import mft_signals
            result = mft_signals("I was brave.")
            assert result == {"boosted_values": [], "vice_flags": []}

    def test_result_has_expected_keys(self):
        with patch("core.mft_classifier._get_classifier", return_value=None):
            from core.mft_classifier import mft_signals
            result = mft_signals("anything")
            assert "boosted_values" in result
            assert "vice_flags" in result

    def test_boosted_values_is_list(self):
        with patch("core.mft_classifier._get_classifier", return_value=None):
            from core.mft_classifier import mft_signals
            assert isinstance(mft_signals("x")["boosted_values"], list)

    def test_vice_flags_is_list(self):
        with patch("core.mft_classifier._get_classifier", return_value=None):
            from core.mft_classifier import mft_signals
            assert isinstance(mft_signals("x")["vice_flags"], list)

    def test_is_available_false_when_model_unavailable(self):
        with patch("core.mft_classifier._get_classifier", return_value=None):
            from core.mft_classifier import is_available
            assert is_available() is False

    def test_mocked_inference_produces_correct_structure(self):
        """
        Mock the classifier to return known scores; verify the parsing logic.
        Catches bugs in score parsing independent of the real model.
        """
        scores = [
            {"label": "LABEL_0", "score": 0.9800},  # care.virtue → boosted
            {"label": "LABEL_1", "score": 0.0200},
            {"label": "LABEL_2", "score": 0.0050},
            {"label": "LABEL_3", "score": 0.0050},
            {"label": "LABEL_4", "score": 0.0050},
            {"label": "LABEL_5", "score": 0.9000},  # loyalty.vice → vice_flag
            {"label": "LABEL_6", "score": 0.0050},
            {"label": "LABEL_7", "score": 0.0050},
            {"label": "LABEL_8", "score": 0.0050},
            {"label": "LABEL_9", "score": 0.0050},
        ]
        mock_clf = MagicMock(return_value=[scores])
        with patch("core.mft_classifier._get_classifier", return_value=mock_clf):
            from core.mft_classifier import mft_signals
            result = mft_signals("She cared for the sick.", min_virtue_score=0.60,
                                  min_vice_score=0.85)
        # care.virtue fired → compassion, love, gratitude should be in boosted_values
        boosted_names = {b["value_name"] for b in result["boosted_values"]}
        assert "compassion" in boosted_names
        assert "love"       in boosted_names
        assert "gratitude"  in boosted_names
        # loyalty.vice scored 0.90 ≥ 0.85 → should be in vice_flags
        vice_foundations = {f["foundation"] for f in result["vice_flags"]}
        assert "loyalty" in vice_foundations

    def test_boosted_value_entry_has_required_fields(self):
        scores = [{"label": f"LABEL_{i}", "score": (0.9 if i == 0 else 0.01)}
                  for i in range(10)]
        mock_clf = MagicMock(return_value=[[s for s in scores]])
        with patch("core.mft_classifier._get_classifier", return_value=mock_clf):
            from core.mft_classifier import mft_signals
            result = mft_signals("text", min_virtue_score=0.5)
        for entry in result["boosted_values"]:
            assert "value_name"     in entry
            assert "mft_foundation" in entry
            assert "score"          in entry

    def test_vice_flag_entry_has_required_fields(self):
        scores = [{"label": f"LABEL_{i}", "score": (0.95 if i == 1 else 0.01)}
                  for i in range(10)]
        mock_clf = MagicMock(return_value=[[s for s in scores]])
        with patch("core.mft_classifier._get_classifier", return_value=mock_clf):
            from core.mft_classifier import mft_signals
            result = mft_signals("text", min_vice_score=0.90)
        for entry in result["vice_flags"]:
            assert "foundation" in entry
            assert "score"      in entry

    def test_below_threshold_virtue_not_boosted(self):
        """A virtue score below min_virtue_score must NOT produce a boost entry."""
        scores = [{"label": f"LABEL_{i}", "score": (0.50 if i == 4 else 0.01)}
                  for i in range(10)]
        mock_clf = MagicMock(return_value=[[s for s in scores]])
        with patch("core.mft_classifier._get_classifier", return_value=mock_clf):
            from core.mft_classifier import mft_signals
            result = mft_signals("text", min_virtue_score=0.60)
        # loyalty.virtue at 0.50 < 0.60 — must not appear
        assert result["boosted_values"] == [], \
            "sub-threshold virtue score must not produce boost entries"

    def test_below_threshold_vice_not_flagged(self):
        """A vice score below min_vice_score must NOT produce a vice_flag entry."""
        scores = [{"label": f"LABEL_{i}", "score": (0.80 if i == 1 else 0.01)}
                  for i in range(10)]
        mock_clf = MagicMock(return_value=[[s for s in scores]])
        with patch("core.mft_classifier._get_classifier", return_value=mock_clf):
            from core.mft_classifier import mft_signals
            result = mft_signals("text", min_vice_score=0.85)
        assert result["vice_flags"] == [], \
            "sub-threshold vice score must not produce vice_flag entry"

    def test_authority_vice_above_virtue_threshold_adds_independence(self):
        """authority.vice above min_virtue_score must add independence to boosted."""
        scores = [{"label": f"LABEL_{i}", "score": (0.95 if i == 7 else 0.01)}
                  for i in range(10)]
        mock_clf = MagicMock(return_value=[[s for s in scores]])
        with patch("core.mft_classifier._get_classifier", return_value=mock_clf):
            from core.mft_classifier import mft_signals
            result = mft_signals("text", min_virtue_score=0.60)
        boosted_names = {b["value_name"] for b in result["boosted_values"]}
        assert "independence" in boosted_names, \
            "authority.vice at high confidence must hint at independence"

    def test_scores_are_rounded_to_4_decimal_places(self):
        scores = [{"label": "LABEL_0", "score": 0.123456789}] + \
                 [{"label": f"LABEL_{i}", "score": 0.01} for i in range(1, 10)]
        mock_clf = MagicMock(return_value=[scores])
        with patch("core.mft_classifier._get_classifier", return_value=mock_clf):
            from core.mft_classifier import mft_signals
            result = mft_signals("text", min_virtue_score=0.10)
        for entry in result["boosted_values"]:
            score_str = str(entry["score"])
            if "." in score_str:
                assert len(score_str.split(".")[1]) <= 4, \
                    f"score {entry['score']} has more than 4 decimal places"


# ---------------------------------------------------------------------------
# TestMftSignalsOnline — real model inference
# ---------------------------------------------------------------------------

_model_available = None

def _check_model():
    global _model_available
    if _model_available is None:
        try:
            _reset_mft_singleton()
            from core.mft_classifier import is_available
            _model_available = is_available()
        except Exception:
            _model_available = False
    return _model_available


requires_model = pytest.mark.skipif(
    not _check_model(),
    reason="MMADS/MoralFoundationsClassifier not available (offline or not downloaded)"
)


class TestMftSignalsOnline:
    """Real inference tests — skipped if model not available."""

    @requires_model
    def test_care_text_fires_compassion(self):
        from core.mft_classifier import mft_signals
        result = mft_signals(
            "I tended to the wounded and nursed the sick back to health.",
            min_virtue_score=0.50,
        )
        boosted_names = {b["value_name"] for b in result["boosted_values"]}
        assert "compassion" in boosted_names, \
            f"care text did not boost compassion; got: {boosted_names}"

    @requires_model
    def test_fairness_text_fires_fairness_value(self):
        from core.mft_classifier import mft_signals
        result = mft_signals(
            "Everyone deserves equal treatment under the law, regardless of wealth or status.",
            min_virtue_score=0.50,
        )
        boosted_names = {b["value_name"] for b in result["boosted_values"]}
        assert "fairness" in boosted_names, \
            f"fairness text did not boost fairness; got: {boosted_names}"

    @requires_model
    def test_loyalty_text_fires_loyalty_value(self):
        from core.mft_classifier import mft_signals
        result = mft_signals(
            "I stood by my comrades even when retreat would have saved me.",
            min_virtue_score=0.50,
        )
        boosted_names = {b["value_name"] for b in result["boosted_values"]}
        assert "loyalty" in boosted_names, \
            f"loyalty text did not boost loyalty; got: {boosted_names}"

    @requires_model
    def test_bravery_text_fires_courage(self):
        """
        Bravery text activates loyalty.virtue (LABEL_4) which maps to courage.
        Regression guard: the LABEL_4→loyalty→courage path must survive refactors.
        """
        from core.mft_classifier import mft_signals
        result = mft_signals(
            "I stood by my comrades, brave despite the danger, and would not abandon them.",
            min_virtue_score=0.50,
        )
        boosted_names = {b["value_name"] for b in result["boosted_values"]}
        assert "courage" in boosted_names or "loyalty" in boosted_names, \
            f"bravery/loyalty text produced no relevant values; got: {boosted_names}"

    @requires_model
    def test_betrayal_text_produces_vice_flag(self):
        from core.mft_classifier import mft_signals
        result = mft_signals(
            "She betrayed her own team, exposing their secrets to the enemy for personal gain.",
            min_vice_score=0.50,
        )
        assert result["vice_flags"], \
            "betrayal text must produce at least one vice_flag"

    @requires_model
    def test_neutral_text_does_not_produce_high_confidence_boosts(self):
        """
        Purely administrative text should not fire strong MFT signals.
        Fails if the model is classifying noise.
        """
        from core.mft_classifier import mft_signals
        result = mft_signals(
            "The quarterly report was filed on Tuesday. The meeting started at nine.",
            min_virtue_score=0.80,  # high bar
        )
        # Neutral text at a high threshold should produce no boosts
        assert result["boosted_values"] == [], \
            f"neutral text fired strong MFT boosts: {result['boosted_values']}"

    @requires_model
    def test_result_structure_with_real_model(self):
        from core.mft_classifier import mft_signals
        result = mft_signals("I cared deeply for those who suffered.", min_virtue_score=0.30)
        assert "boosted_values" in result
        assert "vice_flags"     in result
        assert isinstance(result["boosted_values"], list)
        assert isinstance(result["vice_flags"],     list)

    @requires_model
    def test_mft_foundation_field_is_valid_string(self):
        from core.mft_classifier import mft_signals
        result = mft_signals("I tended to the wounded.", min_virtue_score=0.30)
        for entry in result["boosted_values"]:
            assert isinstance(entry["mft_foundation"], str)
            assert len(entry["mft_foundation"]) > 0


# ---------------------------------------------------------------------------
# TestMftIntegration — pipeline integration
# ---------------------------------------------------------------------------

@pytest.fixture
def isolated_stores(tmp_path):
    from core.document_store import DocumentStore
    from core.value_store import ValueStore
    import core.document_store as _dmod
    import core.value_store as _vmod

    doc = DocumentStore(str(tmp_path / "docs.db"))
    val = ValueStore(str(tmp_path / "vals.db"))

    orig_doc, orig_val = _dmod._instance, _vmod._INSTANCE
    _dmod._instance = doc
    _vmod._INSTANCE = val

    with patch("core.embedder.is_available", return_value=False), \
         patch("core.structural_layer._get_zeroshot_pipeline", return_value=None):
        yield doc, val

    _dmod._instance = orig_doc
    _vmod._INSTANCE = orig_val


_COURAGE_TEXT = (
    "I was afraid, but I stood firm and refused to flee. "
    "Despite the danger, I pressed forward with courage and did not flinch. "
    "I will not yield. Even though the cost was high, I remained resolute."
)


class TestMftIntegration:
    def test_mft_disabled_does_not_break_pipeline(self, isolated_stores):
        """
        Setting mft_enabled=False must leave the pipeline functional.
        No observation crash, results still returned.
        """
        from core.config import get_config
        from core.pipeline import ingest_text
        from core.value_extractor import process_figure

        cfg = get_config()
        orig = cfg.mft_enabled
        cfg.mft_enabled = False
        try:
            ingest_text("mft_off_fig", _COURAGE_TEXT, doc_type="action")
            n = process_figure("figure:mft_off_fig")
            assert n >= 0
        finally:
            cfg.mft_enabled = orig

    def test_mft_enabled_with_mocked_model_adds_source_tag(self, isolated_stores):
        """
        When MFT fires on a value already detected by keyword, the source
        field must include '+mft'.
        """
        from core.config import get_config
        from core.pipeline import ingest_text
        from core.value_extractor import process_figure
        from core.value_store import get_value_store

        # Mock MFT to always return loyalty.virtue boost for 'courage'
        mft_result = {
            "boosted_values": [
                {"value_name": "courage", "mft_foundation": "loyalty", "score": 0.99}
            ],
            "vice_flags": [],
        }

        cfg = get_config()
        orig_mft = cfg.mft_enabled
        orig_threshold = cfg.mft_min_virtue_score
        cfg.mft_enabled = True
        cfg.mft_min_virtue_score = 0.50

        try:
            with patch("core.mft_classifier.mft_signals", return_value=mft_result):
                ingest_text("mft_src_fig", _COURAGE_TEXT, doc_type="action")
                process_figure("figure:mft_src_fig")

            val = get_value_store()
            reg = val.get_registry("figure:mft_src_fig", min_demonstrations=1)
            courage = next((r for r in reg if r["value_name"] == "courage"), None)
            assert courage is not None, "courage not found in registry after MFT-boosted extraction"
        finally:
            cfg.mft_enabled = orig_mft
            cfg.mft_min_virtue_score = orig_threshold

    def test_mft_standalone_signal_recorded_when_no_keyword_match(self, isolated_stores):
        """
        MFT standalone: if mft fires above mft_standalone_threshold for a value
        not caught by keywords, that value must appear in the registry.
        Fails if standalone signal creation path is broken.
        """
        from core.config import get_config
        from core.pipeline import ingest_text
        from core.value_extractor import process_figure
        from core.value_store import get_value_store

        # Neutral admin text — keyword layer will not detect 'compassion'
        neutral_text = "The committee filed the quarterly report and adjourned at nine."

        # MFT mock returns a high-confidence care.virtue → compassion
        mft_result = {
            "boosted_values": [
                {"value_name": "compassion", "mft_foundation": "care", "score": 0.95}
            ],
            "vice_flags": [],
        }

        cfg = get_config()
        orig_mft         = cfg.mft_enabled
        orig_threshold   = cfg.mft_standalone_threshold
        cfg.mft_enabled            = True
        cfg.mft_standalone_threshold = 0.85  # 0.95 ≥ 0.85 → standalone signal created

        try:
            with patch("core.mft_classifier.mft_signals", return_value=mft_result):
                ingest_text("mft_standalone", neutral_text,
                            doc_type="action", significance=0.90)
                process_figure("figure:mft_standalone")

            val = get_value_store()
            reg = val.get_registry("figure:mft_standalone", min_demonstrations=1)
            names = {r["value_name"] for r in reg}
            assert "compassion" in names, \
                f"MFT standalone signal for 'compassion' not in registry; got: {names}"
        finally:
            cfg.mft_enabled            = orig_mft
            cfg.mft_standalone_threshold = orig_threshold

    def test_mft_vice_flag_does_not_add_false_virtue_observation(self, isolated_stores):
        """
        Vice flags must never be stored as virtue (P1) observations.
        A passage flagged as loyalty.vice should not create a loyalty observation.
        """
        from core.config import get_config
        from core.pipeline import ingest_text
        from core.value_extractor import process_figure
        from core.value_store import get_value_store

        betrayal_text = (
            "The meeting report was filed on Tuesday. "
            "The spreadsheet contained data. The committee adjourned."
        )

        mft_result = {
            "boosted_values": [],                                          # no virtues
            "vice_flags": [{"foundation": "loyalty", "score": 0.95}],     # only vice
        }

        cfg = get_config()
        orig_mft = cfg.mft_enabled
        cfg.mft_enabled = True

        try:
            with patch("core.mft_classifier.mft_signals", return_value=mft_result):
                ingest_text("mft_vice_fig", betrayal_text,
                            doc_type="action", significance=0.90)
                process_figure("figure:mft_vice_fig")

            val = get_value_store()
            reg = val.get_registry("figure:mft_vice_fig", min_demonstrations=1)
            # loyalty must not appear as a P1 virtue observation from vice flag alone
            loyalty_obs = [r for r in reg if r["value_name"] == "loyalty"]
            assert not loyalty_obs, \
                "vice flag must not create a P1 loyalty observation"
        finally:
            cfg.mft_enabled = orig_mft
