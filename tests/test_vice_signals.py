"""
tests/test_vice_signals.py

Tests for vice signal → P0-candidate observation flow.

Vice signals are lexicon detections where the MFD2.0 / MoralStrength word
maps to the *vice* side of a moral foundation (e.g. "cruelty" → care-vice
→ compassion P0 candidate).

Suite 1: Lexicon layer emits vice signals with correct shape
Suite 2: Standalone vice observations recorded when no other layer fires
Suite 3: Deduplication — vice not double-counted when virtue already in merged
Suite 4: Confidence threshold gate
Suite 5: Integration — full pipeline run, vice passage → polarity=-1 in DB
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.value_extractor import _lexicon_signals
from core.config import get_config, Config


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def cfg():
    return get_config()


@pytest.fixture
def stores(tmp_path):
    """Isolated real stores with ML layers disabled."""
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


# ---------------------------------------------------------------------------
# Suite 1: Lexicon layer vice signal shape
# ---------------------------------------------------------------------------

class TestLexiconViceShape:
    """Verify that the lexicon layer correctly emits vice signals."""

    def test_vice_signals_have_vice_polarity(self, cfg):
        sigs = _lexicon_signals(
            "He showed nothing but cruelty and callousness.", 0.8, "journal", cfg
        )
        vice = [s for s in sigs if s["lexicon_polarity"] == "vice"]
        assert len(vice) > 0, "Expected at least one vice signal for cruelty/callousness"

    def test_vice_signal_source_has_vice_suffix(self, cfg):
        sigs = _lexicon_signals(
            "He showed nothing but cruelty and callousness.", 0.8, "journal", cfg
        )
        for s in sigs:
            if s["lexicon_polarity"] == "vice":
                assert s["source"].endswith("_vice"), (
                    f"Expected source ending '_vice', got {s['source']!r}"
                )

    def test_vice_signal_confidence_at_base(self, cfg):
        sigs = _lexicon_signals(
            "He showed cruelty to his prisoners.", 0.8, "journal", cfg
        )
        vice = [s for s in sigs if s["lexicon_polarity"] == "vice"]
        assert len(vice) > 0
        for s in vice:
            assert s["disambiguation_confidence"] >= 0.40, (
                f"Vice signal confidence too low: {s['disambiguation_confidence']}"
            )

    def test_vice_signal_has_required_fields(self, cfg):
        sigs = _lexicon_signals(
            "He showed cruelty to his prisoners.", 0.8, "journal", cfg
        )
        vice = [s for s in sigs if s["lexicon_polarity"] == "vice"]
        assert len(vice) > 0
        required = {"value_name", "text_excerpt", "significance",
                    "disambiguation_confidence", "source", "lexicon_polarity"}
        for s in vice:
            assert required <= set(s.keys()), f"Missing fields in vice signal: {s.keys()}"

    def test_mixed_passage_produces_both_virtue_and_vice(self, cfg):
        # "cruelty" → care-vice; "compassion" → care-virtue
        text = "He showed cruelty, yet also compassion for the wounded."
        sigs = _lexicon_signals(text, 0.8, "journal", cfg)
        vice_values   = {s["value_name"] for s in sigs if s["lexicon_polarity"] == "vice"}
        virtue_values = {s["value_name"] for s in sigs if s["lexicon_polarity"] == "virtue"}
        overlap = vice_values & virtue_values
        assert len(overlap) > 0, (
            "Expected overlapping values between vice and virtue signals"
        )

    def test_neutral_passage_no_vice_signals(self, cfg):
        # No moral loading
        text = "He walked to the market and bought some bread."
        sigs = _lexicon_signals(text, 0.8, "journal", cfg)
        vice = [s for s in sigs if s["lexicon_polarity"] == "vice"]
        assert len(vice) == 0, f"Unexpected vice signals on neutral text: {vice}"


# ---------------------------------------------------------------------------
# Suite 2: Standalone vice observations recorded when no other layer fires
# ---------------------------------------------------------------------------

class TestStandaloneViceObservation:
    """When only vice words fire, standalone observations with polarity=-1 are stored."""

    def test_pure_vice_passage_produces_negative_polarity_observations(self, stores):
        """A passage with only vice words (cruelty) → polarity=-1 observations."""
        _, val_store = stores
        from core.pipeline import ingest_text

        result = ingest_text(
            figure_name="test",
            text="He showed nothing but cruelty and callousness toward his subordinates.",
            pronoun="he",
            doc_type="journal",
            significance=0.8,
        )
        assert result.error is None

        obs = val_store.get_observations("figure:test")
        # At least some observations should have polarity=-1 from lexicon vice
        negative = [o for o in obs if o["value_polarity"] == -1]
        assert len(negative) > 0, (
            f"Expected polarity=-1 observations, got: "
            f"{[(o['value_name'], o['value_polarity'], o['source']) for o in obs]}"
        )

    def test_vice_observations_have_vice_source_tag(self, stores):
        """Observations from vice path should have source containing '_vice'."""
        _, val_store = stores
        from core.pipeline import ingest_text

        ingest_text(
            figure_name="test",
            text="He showed nothing but cruelty and callousness toward his subordinates.",
            pronoun="he",
            doc_type="journal",
            significance=0.8,
        )

        obs = val_store.get_observations("figure:test")
        vice_obs = [o for o in obs if "_vice" in o.get("source", "")]
        assert len(vice_obs) > 0, (
            "Expected at least one observation with '_vice' in source"
        )
        for o in vice_obs:
            assert o["value_polarity"] == -1, (
                f"Vice-sourced observation should have polarity=-1, got {o['value_polarity']}"
            )

    def test_cheating_passage_produces_fairness_vice_observation(self, stores):
        """'cheated' maps to fairness vice — should produce fairness observation."""
        _, val_store = stores
        from core.pipeline import ingest_text

        ingest_text(
            figure_name="test",
            text="He cheated and deceived his partners to gain an advantage.",
            pronoun="he",
            doc_type="journal",
            significance=0.8,
        )

        obs = val_store.get_observations("figure:test")
        fairness_negative = [
            o for o in obs
            if o["value_name"] == "fairness" and o["value_polarity"] == -1
        ]
        assert len(fairness_negative) > 0, (
            f"Expected fairness P0-candidate, got: "
            f"{[(o['value_name'], o['value_polarity']) for o in obs]}"
        )


# ---------------------------------------------------------------------------
# Suite 3: Deduplication — vice not double-counted when virtue present
# ---------------------------------------------------------------------------

class TestViceDeduplication:
    """When virtue for the same value already fired, no standalone vice added."""

    def test_mixed_passage_no_duplicate_observations(self, stores):
        """
        Passage with both 'cruelty' (vice) and 'compassion' (virtue) for same
        values → exactly one observation per value (the virtue/keyword one),
        not two.
        """
        _, val_store = stores
        from core.pipeline import ingest_text

        ingest_text(
            figure_name="test",
            text="Despite his cruelty, he showed great compassion for the wounded.",
            pronoun="he",
            doc_type="journal",
            significance=0.8,
        )

        obs = val_store.get_observations("figure:test")
        # Count observations per value_name — should be 1 each
        from collections import Counter
        by_value = Counter(o["value_name"] for o in obs)
        duplicates = {v: n for v, n in by_value.items() if n > 1}
        assert not duplicates, (
            f"Duplicate observations found (vice+virtue double-counted): {duplicates}"
        )

    def test_virtue_keyword_present_vice_does_not_add_second_observation(self, stores):
        """
        Keyword 'courage' fires (virtue); if a vice signal for the same value
        also fires, the vice should NOT add a second standalone observation.
        """
        _, val_store = stores
        from core.pipeline import ingest_text

        ingest_text(
            figure_name="test",
            text="His courage in the face of cowardly opposition was noted.",
            pronoun="he",
            doc_type="journal",
            significance=0.8,
        )

        obs = val_store.get_observations("figure:test")
        courage_obs = [o for o in obs if o["value_name"] == "courage"]
        assert len(courage_obs) <= 1, (
            f"Expected at most 1 courage observation, got {len(courage_obs)}: {courage_obs}"
        )


# ---------------------------------------------------------------------------
# Suite 4: Confidence threshold gate
# ---------------------------------------------------------------------------

class TestViceConfidenceThreshold:
    """Vice signals below lexicon_standalone_min_conf are not added."""

    def test_vice_below_threshold_not_recorded(self, stores):
        """Raise threshold above base confidence → no vice observations."""
        _, val_store = stores
        from core.pipeline import ingest_text
        from core.config import Config

        # Set threshold above 0.55 (base vice conf) so no vice signal passes
        high_threshold_cfg = Config(
            lexicon_standalone_min_conf=0.80,
            lexicon_enabled=True,
        )
        with patch("core.config._default", high_threshold_cfg):
            ingest_text(
                figure_name="test",
                text="He showed nothing but cruelty and callousness.",
                pronoun="he",
                doc_type="journal",
                significance=0.8,
            )

        obs = val_store.get_observations("figure:test")
        vice_obs = [o for o in obs if "_vice" in o.get("source", "")]
        assert len(vice_obs) == 0, (
            f"Expected no vice observations above threshold=0.80, got: {vice_obs}"
        )

    def test_vice_at_threshold_is_recorded(self, stores):
        """At default threshold (0.55), vice signals at exactly 0.55 pass."""
        _, val_store = stores
        from core.pipeline import ingest_text

        ingest_text(
            figure_name="test",
            text="He showed nothing but cruelty and callousness.",
            pronoun="he",
            doc_type="journal",
            significance=0.8,
        )

        obs = val_store.get_observations("figure:test")
        vice_obs = [o for o in obs if "_vice" in o.get("source", "")]
        assert len(vice_obs) > 0, (
            "Expected vice observations at default threshold"
        )


# ---------------------------------------------------------------------------
# Suite 5: Export integration — vice observations become P0 candidates
# ---------------------------------------------------------------------------

class TestViceExportIntegration:
    """Vice observations with polarity=-1 should export as P0 or panel-confirmed P0."""

    def test_vice_observation_exports_as_p0_candidate(self, stores):
        """
        End-to-end: ingest vice passage → build_training_records → vice observations
        have value_polarity=-1.

        The heuristic classifier may label it P1, P0, or AMBIGUOUS depending on
        resistance. The key invariant is value_polarity=-1 in the training record.
        """
        _, val_store = stores
        from core.pipeline import ingest_text
        from cli.export import _read_figure_observations, build_training_records

        ingest_text(
            figure_name="test",
            text="He showed nothing but cruelty and callousness toward his subordinates.",
            pronoun="he",
            doc_type="journal",
            significance=0.8,
        )

        rows = _read_figure_observations(val_store._db_path, figure_filter="test")
        assert len(rows) > 0, "Expected observations from vice passage"

        vice_rows = [r for r in rows if "_vice" in r.get("source", "")]
        assert len(vice_rows) > 0, (
            f"Expected rows with '_vice' source, got sources: "
            f"{[r.get('source') for r in rows]}"
        )

        # Build training records and verify value_polarity is preserved
        training = build_training_records(
            rows,
            p1_threshold=0.55,
            p0_threshold=0.35,
            min_observations=1,
        )
        vice_training = [
            t for t in training if "_vice" in t.get("source", "")
        ]
        assert len(vice_training) > 0, (
            "Expected training records from vice observations"
        )
        for t in vice_training:
            assert t["value_polarity"] == -1, (
                f"Vice training record should have value_polarity=-1, got {t['value_polarity']}"
            )

    def test_lexicon_disabled_produces_no_vice_observations(self, stores):
        """When lexicon is disabled, no vice observations are created."""
        _, val_store = stores
        from core.pipeline import ingest_text
        from core.config import Config

        no_lex_cfg = Config(lexicon_enabled=False)
        with patch("core.config._default", no_lex_cfg):
            ingest_text(
                figure_name="test",
                text="He showed nothing but cruelty and callousness.",
                pronoun="he",
                doc_type="journal",
                significance=0.8,
            )

        obs = val_store.get_observations("figure:test")
        vice_obs = [o for o in obs if "_vice" in o.get("source", "")]
        assert len(vice_obs) == 0, (
            f"Expected no vice observations when lexicon disabled, got: {vice_obs}"
        )
