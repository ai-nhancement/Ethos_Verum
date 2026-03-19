"""
tests/test_export.py

Tests for cli/export.py — classify_observation(), build_training_records(),
and the panel polarity override logic.

Suite 1: classify_observation — base heuristic
Suite 2: build_training_records — panel override (value_polarity + source)
Suite 3: build_training_records — source and polarity fields in output
Suite 4: build_training_records — APY label is never overridden by panel
Suite 5: build_training_records — non-panel observations unaffected
Suite 6: _read_figure_observations — source/polarity columns fetched
"""
from __future__ import annotations

import os
import sys
import tempfile
import time
import uuid

import pytest

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from cli.export import classify_observation, build_training_records
from core.value_store import ValueStore

NOW = time.time()


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _obs(
    value_name: str = "courage",
    text_excerpt: str = "I stood firm despite the threats.",
    resistance: float = 0.70,
    source: str = "keyword",
    value_polarity: int = 0,
    polarity_confidence: float = 0.0,
    observation_consistency: float = 0.5,
    disambiguation_confidence: float = 1.0,
    doc_type: str = "journal",
    session_id: str = "figure:test",
    record_id: str = None,
) -> dict:
    return {
        "id": str(uuid.uuid4()),
        "session_id": session_id,
        "record_id": record_id or str(uuid.uuid4()),
        "ts": NOW,
        "value_name": value_name,
        "text_excerpt": text_excerpt,
        "significance": 0.90,
        "resistance": resistance,
        "source": source,
        "value_polarity": value_polarity,
        "polarity_confidence": polarity_confidence,
        "disambiguation_confidence": disambiguation_confidence,
        "figure_name": "test",
        "document_type": doc_type,
        "observation_consistency": observation_consistency,
    }


def _build(obs_list, p1=0.55, p0=0.35):
    return build_training_records(obs_list, p1_threshold=p1, p0_threshold=p0,
                                  min_observations=1)


# ---------------------------------------------------------------------------
# Suite 1 — classify_observation base heuristic
# ---------------------------------------------------------------------------

class TestClassifyObservation:
    def test_high_resistance_hold_markers_p1(self):
        label, reason, conf = classify_observation(
            "I stood firm despite the threats.", 0.70, 0.55, 0.35)
        assert label == "P1"
        assert conf >= 0.75

    def test_failure_markers_p0(self):
        label, reason, conf = classify_observation(
            "I gave in and told them what they wanted.", 0.60, 0.55, 0.35)
        assert label == "P0"
        assert conf >= 0.80

    def test_apy_pressure_plus_failure(self):
        label, reason, conf = classify_observation(
            "Under pressure they forced me and I gave in.", 0.50, 0.55, 0.35)
        assert label == "APY"

    def test_apy_pressure_no_failure_gives_p1(self):
        label, reason, conf = classify_observation(
            "Under pressure they demanded it but I held firm.", 0.65, 0.55, 0.35)
        assert label == "P1"
        assert reason == "apy_resistance_held_under_pressure"

    def test_low_resistance_no_markers_p0(self):
        label, reason, conf = classify_observation(
            "He walked to the store.", 0.25, 0.55, 0.35)
        assert label == "P0"

    def test_middle_resistance_ambiguous(self):
        label, reason, conf = classify_observation(
            "He walked to the store.", 0.45, 0.55, 0.35)
        assert label == "AMBIGUOUS"

    def test_high_resistance_no_markers_p1_lower_confidence(self):
        label, reason, conf = classify_observation(
            "He walked to the store.", 0.70, 0.55, 0.35)
        assert label == "P1"
        assert conf == 0.75  # no hold markers

    def test_returns_tuple_of_three(self):
        result = classify_observation("text", 0.60, 0.55, 0.35)
        assert len(result) == 3
        label, reason, conf = result
        assert isinstance(label, str)
        assert isinstance(reason, str)
        assert isinstance(conf, float)


# ---------------------------------------------------------------------------
# Suite 2 — panel override in build_training_records
# ---------------------------------------------------------------------------

class TestPanelOverride:
    def test_panel_p1_overrides_ambiguous(self):
        """Panel confirmed P1 on an observation that would be AMBIGUOUS by heuristic."""
        obs = _obs(
            text_excerpt="The matter was noted.",  # no markers → AMBIGUOUS
            resistance=0.45,
            source="keyword+panel",
            value_polarity=1,
            polarity_confidence=0.95,
        )
        records = _build([obs])
        assert records[0]["label"] == "P1"
        assert records[0]["label_reason"] == "panel_confirmed_p1"
        assert records[0]["confidence"] == 0.90

    def test_panel_p0_overrides_ambiguous(self):
        """Panel confirmed P0 on an observation that would be AMBIGUOUS by heuristic."""
        obs = _obs(
            text_excerpt="The matter was noted.",
            resistance=0.45,
            source="keyword+panel",
            value_polarity=-1,
            polarity_confidence=0.88,
        )
        records = _build([obs])
        assert records[0]["label"] == "P0"
        assert records[0]["label_reason"] == "panel_confirmed_p0"

    def test_panel_p0_overrides_heuristic_p1(self):
        """Panel says P0 but heuristic says P1 (high resistance, hold markers).
        Panel verdict takes precedence."""
        obs = _obs(
            text_excerpt="I stood firm despite the threats.",
            resistance=0.80,
            source="keyword+semantic+panel",
            value_polarity=-1,  # panel saw this as a failure
            polarity_confidence=0.90,
        )
        records = _build([obs])
        assert records[0]["label"] == "P0"
        assert records[0]["label_reason"] == "panel_confirmed_p0"

    def test_panel_p1_overrides_heuristic_p0(self):
        """Panel says P1 but heuristic says P0 (failure markers in text).
        Panel verdict takes precedence."""
        obs = _obs(
            text_excerpt="I gave in and told them what they wanted.",
            resistance=0.70,
            source="keyword+panel",
            value_polarity=1,
            polarity_confidence=0.92,
        )
        records = _build([obs])
        assert records[0]["label"] == "P1"
        assert records[0]["label_reason"] == "panel_confirmed_p1"

    def test_panel_p1_confirms_heuristic_p1_no_change(self):
        """Panel P1 on already-P1 obs: label stays P1 (no needless override)."""
        obs = _obs(
            text_excerpt="I stood firm despite the threats.",
            resistance=0.80,
            source="keyword+panel",
            value_polarity=1,
        )
        records = _build([obs])
        assert records[0]["label"] == "P1"

    def test_panel_p0_confirms_heuristic_p0_no_change(self):
        """Panel P0 on already-P0 obs: label stays P0."""
        obs = _obs(
            text_excerpt="I gave in and told them what they wanted.",
            resistance=0.25,
            source="keyword+panel",
            value_polarity=-1,
        )
        records = _build([obs])
        assert records[0]["label"] == "P0"

    def test_panel_zero_polarity_no_override(self):
        """Panel source tag but polarity=0 (skip verdict): no override."""
        obs = _obs(
            text_excerpt="The matter was noted.",
            resistance=0.45,
            source="keyword+panel",
            value_polarity=0,  # skip — panel did not set polarity
        )
        records = _build([obs])
        assert records[0]["label"] == "AMBIGUOUS"


# ---------------------------------------------------------------------------
# Suite 3 — source and polarity fields present in output
# ---------------------------------------------------------------------------

class TestOutputFields:
    def test_source_field_in_output(self):
        obs = _obs(source="keyword+semantic+panel")
        records = _build([obs])
        assert records[0]["source"] == "keyword+semantic+panel"

    def test_value_polarity_in_output(self):
        obs = _obs(value_polarity=1, polarity_confidence=0.85,
                   source="keyword+panel")
        records = _build([obs])
        assert records[0]["value_polarity"] == 1
        assert records[0]["polarity_confidence"] == pytest.approx(0.85, abs=1e-4)

    def test_source_empty_string_for_no_panel(self):
        obs = _obs(source="keyword")
        records = _build([obs])
        assert records[0]["source"] == "keyword"

    def test_all_expected_fields_present(self):
        obs = _obs(source="keyword+panel", value_polarity=1)
        r = _build([obs])[0]
        expected = [
            "id", "source_obs_id", "figure", "session_id", "record_id", "ts",
            "value_name", "text_excerpt", "document_type", "significance",
            "resistance", "label", "label_reason", "fail_mode",
            "training_weight", "confidence", "pressure_markers",
            "failure_markers", "hold_markers", "source",
            "disambiguation_confidence", "observation_consistency",
            "value_polarity", "polarity_confidence",
            "pressure_source_id", "pressure_context",
            "deferred_apy_lag_s", "deferred_apy_lag_n",
        ]
        for f in expected:
            assert f in r, f"Missing field: {f}"


# ---------------------------------------------------------------------------
# Suite 4 — APY label is never overridden by panel
# ---------------------------------------------------------------------------

class TestPanelDoesNotOverrideAPY:
    def test_apy_not_overridden_by_panel_p1(self):
        """APY from text markers is kept even if panel set polarity=1."""
        obs = _obs(
            text_excerpt="Under pressure they forced me and I gave in.",
            resistance=0.60,
            source="keyword+panel",
            value_polarity=1,  # panel says holds — but APY wins
        )
        records = _build([obs])
        assert records[0]["label"] == "APY"

    def test_apy_not_overridden_by_panel_p0(self):
        """APY from text markers is kept even if panel set polarity=-1."""
        obs = _obs(
            text_excerpt="Under pressure they forced me and I gave in.",
            resistance=0.60,
            source="keyword+panel",
            value_polarity=-1,
        )
        records = _build([obs])
        assert records[0]["label"] == "APY"


# ---------------------------------------------------------------------------
# Suite 5 — non-panel observations unaffected
# ---------------------------------------------------------------------------

class TestNonPanelObservations:
    def test_no_panel_source_no_override(self):
        """Polarity set without panel tag: no override (polarity came from
        phrase/lexicon layer, not panel)."""
        obs = _obs(
            text_excerpt="The matter was noted.",
            resistance=0.45,
            source="keyword+lexicon",  # no +panel
            value_polarity=-1,
        )
        records = _build([obs])
        assert records[0]["label"] == "AMBIGUOUS"

    def test_keyword_only_source_normal_classification(self):
        obs = _obs(
            text_excerpt="I stood firm despite the threats.",
            resistance=0.80,
            source="keyword",
        )
        records = _build([obs])
        assert records[0]["label"] == "P1"

    def test_min_observations_filter_still_works(self):
        """min_observations filter operates independently of panel logic."""
        obs = _obs(source="keyword+panel", value_polarity=1)
        records = build_training_records([obs], p1_threshold=0.55,
                                         p0_threshold=0.35,
                                         min_observations=2)
        assert records == []


# ---------------------------------------------------------------------------
# Suite 6 — _read_figure_observations fetches source/polarity columns
# ---------------------------------------------------------------------------

class TestReadFigureObservations:
    """Verify that source, value_polarity, polarity_confidence are fetched
    from the DB — not silently defaulted to empty/zero."""

    def _make_store(self):
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        return ValueStore(db_path=path), path

    def test_source_column_fetched(self):
        store, path = self._make_store()
        store.register_figure_source("figure:test", "test", "journal", 1)
        store.record_observation(
            session_id="figure:test",
            turn_id="t1", record_id="r1", ts=NOW,
            value_name="courage",
            text_excerpt="stood firm",
            significance=0.9, resistance=0.7,
            disambiguation_confidence=1.0,
            doc_type="journal",
            value_polarity=1,
            polarity_confidence=0.88,
            source="keyword+semantic+panel",
        )

        from cli.export import _read_figure_observations
        rows = _read_figure_observations(path, figure_filter="test")
        assert len(rows) == 1
        row = rows[0]
        assert row["source"] == "keyword+semantic+panel"
        assert row["value_polarity"] == 1
        assert abs(row["polarity_confidence"] - 0.88) < 1e-4

    def test_missing_columns_default_gracefully(self):
        """Even on a DB where new columns don't exist yet (pre-migration),
        COALESCE defaults prevent KeyError."""
        store, path = self._make_store()
        store.register_figure_source("figure:test", "test", "journal", 1)
        # Write an observation with no source/polarity (defaults)
        store.record_observation(
            session_id="figure:test",
            turn_id="t1", record_id="r1", ts=NOW,
            value_name="courage",
            text_excerpt="stood firm despite threats",
            significance=0.9, resistance=0.7,
            disambiguation_confidence=1.0,
            doc_type="journal",
        )

        from cli.export import _read_figure_observations
        rows = _read_figure_observations(path, figure_filter="test")
        assert rows[0]["source"] == ""
        assert rows[0]["value_polarity"] == 0
        assert rows[0]["polarity_confidence"] == 0.0

    def test_panel_polarity_flows_through_to_training_record(self):
        """End-to-end: DB write with panel source → read → build_training_records
        → panel override applies correctly."""
        store, path = self._make_store()
        store.register_figure_source("figure:test", "test", "journal", 1)
        # Ambiguous text but panel confirmed P1
        store.record_observation(
            session_id="figure:test",
            turn_id="t1", record_id="r1", ts=NOW,
            value_name="integrity",
            text_excerpt="The matter was noted in the record.",
            significance=0.9, resistance=0.45,
            disambiguation_confidence=1.0,
            doc_type="journal",
            value_polarity=1,
            polarity_confidence=0.91,
            source="keyword+panel",
        )

        from cli.export import _read_figure_observations
        rows = _read_figure_observations(path, figure_filter="test")
        records = _build(rows)
        assert len(records) == 1
        assert records[0]["label"] == "P1"
        assert records[0]["label_reason"] == "panel_confirmed_p1"
        assert records[0]["source"] == "keyword+panel"
