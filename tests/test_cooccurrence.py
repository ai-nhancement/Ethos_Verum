"""
tests/test_cooccurrence.py

§7.8 Value Co-occurrence and Interaction Modeling — test suite

Suites:
  1. core/config.py — VALUE_TENSION_PAIRS + is_tension_pair()
  2. ValueStore.record_tension() / get_tensions()
  3. _compute_cooccurrence() — matrix structure and counts
  4. _compute_cooccurrence() — label breakdown (both_p1 / mixed / both_p0)
  5. _detect_tensions() — tension event detection from classified records
  6. _detect_tensions() — non-events (same label, non-tension pair)
  7. Tension events carry 1.5× training weight
  8. co_occurrence in export report (via export() integration)
  9. Edge cases (single value, no co-occurring pairs, empty records)
"""
from __future__ import annotations

import os
import sys
import sqlite3
import tempfile
import time
import uuid
import unittest

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.config import VALUE_TENSION_PAIRS, is_tension_pair
from core.value_store import ValueStore

NOW = time.time()


def _tmp_db() -> str:
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    return path


def _tmp_store() -> ValueStore:
    return ValueStore(db_path=_tmp_db())


def _obs(
    session_id: str = "figure:test",
    record_id: str = "r1",
    ts: float = NOW,
    value_name: str = "courage",
    label: str = "P1",
    resistance: float = 0.7,
    text_excerpt: str = "I stood firm despite the threat.",
    figure: str = "Test Figure",
    training_weight: float = 1.0,
) -> dict:
    return {
        "id":              str(uuid.uuid4()),
        "session_id":      session_id,
        "record_id":       record_id,
        "ts":              ts,
        "value_name":      value_name,
        "label":           label,
        "resistance":      resistance,
        "text_excerpt":    text_excerpt,
        "figure":          figure,
        "training_weight": training_weight,
        "document_type":   "journal",
        "significance":    0.8,
        "observation_consistency": 0.5,
        "disambiguation_confidence": 1.0,
    }


# ==============================================================================
# Suite 1: Tension pair configuration
# ==============================================================================

class TestTensionPairs(unittest.TestCase):

    def test_five_default_pairs(self):
        self.assertEqual(len(VALUE_TENSION_PAIRS), 5)

    def test_all_spec_pairs_present(self):
        pair_set = {frozenset(p) for p in VALUE_TENSION_PAIRS}
        expected = [
            ("independence", "loyalty"),
            ("fairness",     "compassion"),
            ("courage",      "patience"),
            ("responsibility", "humility"),
            ("commitment",   "growth"),
        ]
        for a, b in expected:
            self.assertIn(frozenset({a, b}), pair_set, f"Missing pair: ({a}, {b})")

    def test_is_tension_pair_symmetric(self):
        self.assertTrue(is_tension_pair("independence", "loyalty"))
        self.assertTrue(is_tension_pair("loyalty", "independence"))

    def test_is_tension_pair_all_five(self):
        for a, b in VALUE_TENSION_PAIRS:
            self.assertTrue(is_tension_pair(a, b), f"({a},{b}) not a tension pair")
            self.assertTrue(is_tension_pair(b, a), f"({b},{a}) not a tension pair")

    def test_non_tension_pair_returns_false(self):
        self.assertFalse(is_tension_pair("courage", "loyalty"))
        self.assertFalse(is_tension_pair("integrity", "compassion"))
        self.assertFalse(is_tension_pair("courage", "courage"))

    def test_unknown_value_returns_false(self):
        self.assertFalse(is_tension_pair("nonexistent", "loyalty"))


# ==============================================================================
# Suite 2: ValueStore tension table
# ==============================================================================

class TestTensionStore(unittest.TestCase):

    def setUp(self):
        self.store = _tmp_store()

    def test_record_and_retrieve(self):
        self.store.record_tension("sess", "rec1", NOW, "courage", "patience", 0.75, "text")
        rows = self.store.get_tensions("sess")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["value_held"], "courage")
        self.assertEqual(rows[0]["value_failed"], "patience")

    def test_get_all_sessions(self):
        self.store.record_tension("sess_a", "r1", NOW, "fairness", "compassion", 0.6, "t")
        self.store.record_tension("sess_b", "r2", NOW, "courage",  "patience",   0.7, "t")
        all_rows = self.store.get_tensions()
        self.assertEqual(len(all_rows), 2)

    def test_filter_by_session(self):
        self.store.record_tension("sess_a", "r1", NOW, "fairness", "compassion", 0.6, "t")
        self.store.record_tension("sess_b", "r2", NOW, "courage",  "patience",   0.7, "t")
        rows_a = self.store.get_tensions("sess_a")
        self.assertEqual(len(rows_a), 1)
        self.assertEqual(rows_a[0]["value_held"], "fairness")

    def test_text_excerpt_truncated_to_300(self):
        long_text = "x" * 500
        self.store.record_tension("sess", "r1", NOW, "courage", "patience", 0.7, long_text)
        rows = self.store.get_tensions("sess")
        self.assertLessEqual(len(rows[0]["text_excerpt"]), 300)

    def test_empty_session_returns_empty(self):
        rows = self.store.get_tensions("nonexistent")
        self.assertEqual(rows, [])

    def test_never_raises(self):
        try:
            self.store.record_tension(None, None, None, None, None, None, None)  # type: ignore
        except Exception:
            self.fail("record_tension raised unexpectedly")


# ==============================================================================
# Suite 3: _compute_cooccurrence() — matrix structure
# ==============================================================================

class TestCooccurrenceMatrix(unittest.TestCase):

    def _build(self, records):
        from cli.export import _compute_cooccurrence
        return _compute_cooccurrence(records)

    def test_single_passage_two_values(self):
        records = [
            _obs(record_id="r1", value_name="courage",   label="P1"),
            _obs(record_id="r1", value_name="integrity", label="P1"),
        ]
        matrix = self._build(records)
        self.assertIn("courage||integrity", matrix)
        self.assertEqual(matrix["courage||integrity"]["both_detected"], 1)

    def test_pair_key_always_sorted(self):
        records = [
            _obs(record_id="r1", value_name="patience", label="P1"),
            _obs(record_id="r1", value_name="courage",  label="P1"),
        ]
        matrix = self._build(records)
        # courage < patience alphabetically
        self.assertIn("courage||patience", matrix)
        self.assertNotIn("patience||courage", matrix)

    def test_no_cooccurrence_for_single_value_passage(self):
        records = [_obs(record_id="r1", value_name="courage", label="P1")]
        matrix = self._build(records)
        self.assertEqual(len(matrix), 0)

    def test_three_values_same_passage_produces_three_pairs(self):
        records = [
            _obs(record_id="r1", value_name="courage",     label="P1"),
            _obs(record_id="r1", value_name="integrity",   label="P1"),
            _obs(record_id="r1", value_name="compassion",  label="P1"),
        ]
        matrix = self._build(records)
        self.assertEqual(len(matrix), 3)

    def test_different_passages_not_paired(self):
        records = [
            _obs(record_id="r1", value_name="courage",   label="P1"),
            _obs(record_id="r2", value_name="integrity", label="P1"),
        ]
        matrix = self._build(records)
        self.assertEqual(len(matrix), 0)

    def test_session_isolation(self):
        records = [
            _obs(session_id="figure:a", record_id="r1", value_name="courage",   label="P1"),
            _obs(session_id="figure:b", record_id="r1", value_name="integrity", label="P1"),
        ]
        matrix = self._build(records)
        # same record_id but different session_id → different key → not paired
        self.assertEqual(len(matrix), 0)

    def test_multiple_passages_accumulate(self):
        records = [
            _obs(record_id="r1", value_name="courage",   label="P1"),
            _obs(record_id="r1", value_name="integrity", label="P1"),
            _obs(record_id="r2", value_name="courage",   label="P0"),
            _obs(record_id="r2", value_name="integrity", label="P0"),
        ]
        matrix = self._build(records)
        self.assertEqual(matrix["courage||integrity"]["both_detected"], 2)


# ==============================================================================
# Suite 4: _compute_cooccurrence() — label breakdown
# ==============================================================================

class TestCooccurrenceLabelBreakdown(unittest.TestCase):

    def _build(self, records):
        from cli.export import _compute_cooccurrence
        return _compute_cooccurrence(records)

    def test_both_p1(self):
        records = [
            _obs(record_id="r1", value_name="courage",   label="P1"),
            _obs(record_id="r1", value_name="integrity", label="P1"),
        ]
        entry = self._build(records)["courage||integrity"]
        self.assertEqual(entry["both_p1"], 1)
        self.assertEqual(entry["mixed"],   0)
        self.assertEqual(entry["both_p0"], 0)

    def test_both_p0(self):
        records = [
            _obs(record_id="r1", value_name="courage",   label="P0"),
            _obs(record_id="r1", value_name="integrity", label="P0"),
        ]
        entry = self._build(records)["courage||integrity"]
        self.assertEqual(entry["both_p1"], 0)
        self.assertEqual(entry["mixed"],   0)
        self.assertEqual(entry["both_p0"], 1)

    def test_mixed_one_p1_one_p0(self):
        records = [
            _obs(record_id="r1", value_name="courage",   label="P1"),
            _obs(record_id="r1", value_name="integrity", label="P0"),
        ]
        entry = self._build(records)["courage||integrity"]
        self.assertEqual(entry["both_p1"], 0)
        self.assertEqual(entry["mixed"],   1)
        self.assertEqual(entry["both_p0"], 0)

    def test_apy_treated_as_non_p1(self):
        records = [
            _obs(record_id="r1", value_name="courage",   label="P1"),
            _obs(record_id="r1", value_name="integrity", label="APY"),
        ]
        entry = self._build(records)["courage||integrity"]
        self.assertEqual(entry["mixed"], 1)

    def test_counts_sum_to_both_detected(self):
        records = [
            _obs(record_id="r1", value_name="courage",   label="P1"),
            _obs(record_id="r1", value_name="integrity", label="P1"),
            _obs(record_id="r2", value_name="courage",   label="P0"),
            _obs(record_id="r2", value_name="integrity", label="P0"),
            _obs(record_id="r3", value_name="courage",   label="P1"),
            _obs(record_id="r3", value_name="integrity", label="P0"),
        ]
        entry = self._build(records)["courage||integrity"]
        self.assertEqual(entry["both_detected"], 3)
        total = entry["both_p1"] + entry["mixed"] + entry["both_p0"]
        self.assertEqual(total, entry["both_detected"])


# ==============================================================================
# Suite 5: _detect_tensions() — tension event detection
# ==============================================================================

class TestTensionDetection(unittest.TestCase):

    def _detect(self, records, db_path=None):
        from cli.export import _detect_tensions
        if db_path is None:
            db_path = _tmp_db()
            # Init schema
            store = ValueStore(db_path=db_path)
        return _detect_tensions(records, db_path)

    def test_detects_independence_loyalty_tension(self):
        records = [
            _obs(record_id="r1", value_name="independence", label="P1", resistance=0.8),
            _obs(record_id="r1", value_name="loyalty",      label="P0", resistance=0.3),
        ]
        events = self._detect(records)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["value_held"],   "independence")
        self.assertEqual(events[0]["value_failed"], "loyalty")

    def test_detects_all_five_tension_pairs(self):
        pairs = VALUE_TENSION_PAIRS
        for a, b in pairs:
            records = [
                _obs(record_id="r1", value_name=a, label="P1"),
                _obs(record_id="r1", value_name=b, label="P0"),
            ]
            events = self._detect(records)
            self.assertEqual(len(events), 1, f"Missed tension for ({a},{b})")

    def test_held_is_p1_side(self):
        records = [
            _obs(record_id="r1", value_name="fairness",   label="P0"),
            _obs(record_id="r1", value_name="compassion", label="P1"),
        ]
        events = self._detect(records)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["value_held"],   "compassion")
        self.assertEqual(events[0]["value_failed"], "fairness")

    def test_tension_apy_counts_as_non_p1(self):
        records = [
            _obs(record_id="r1", value_name="courage",  label="P1"),
            _obs(record_id="r1", value_name="patience", label="APY"),
        ]
        events = self._detect(records)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["value_failed"], "patience")

    def test_no_tension_for_different_passages(self):
        records = [
            _obs(record_id="r1", value_name="independence", label="P1"),
            _obs(record_id="r2", value_name="loyalty",      label="P0"),
        ]
        events = self._detect(records)
        self.assertEqual(len(events), 0)


# ==============================================================================
# Suite 6: _detect_tensions() — non-event cases
# ==============================================================================

class TestTensionNonEvents(unittest.TestCase):

    def _detect(self, records):
        from cli.export import _detect_tensions
        db_path = _tmp_db()
        ValueStore(db_path=db_path)
        return _detect_tensions(records, db_path)

    def test_both_p1_not_a_tension(self):
        records = [
            _obs(record_id="r1", value_name="independence", label="P1"),
            _obs(record_id="r1", value_name="loyalty",      label="P1"),
        ]
        events = self._detect(records)
        self.assertEqual(len(events), 0)

    def test_both_p0_not_a_tension(self):
        records = [
            _obs(record_id="r1", value_name="independence", label="P0"),
            _obs(record_id="r1", value_name="loyalty",      label="P0"),
        ]
        events = self._detect(records)
        self.assertEqual(len(events), 0)

    def test_non_tension_pair_p1_p0_not_an_event(self):
        records = [
            _obs(record_id="r1", value_name="courage",   label="P1"),
            _obs(record_id="r1", value_name="integrity", label="P0"),
        ]
        events = self._detect(records)
        self.assertEqual(len(events), 0)

    def test_single_value_no_event(self):
        records = [_obs(record_id="r1", value_name="courage", label="P1")]
        events = self._detect(records)
        self.assertEqual(len(events), 0)

    def test_empty_records_no_event(self):
        events = self._detect([])
        self.assertEqual(len(events), 0)


# ==============================================================================
# Suite 7: Tension training weight
# ==============================================================================

class TestTensionWeight(unittest.TestCase):

    def _detect(self, records):
        from cli.export import _detect_tensions
        db_path = _tmp_db()
        ValueStore(db_path=db_path)
        return _detect_tensions(records, db_path)

    def test_tension_weight_is_1point5x_held_weight(self):
        records = [
            _obs(record_id="r1", value_name="independence", label="P1", training_weight=1.2),
            _obs(record_id="r1", value_name="loyalty",      label="P0", training_weight=0.8),
        ]
        events = self._detect(records)
        self.assertEqual(len(events), 1)
        self.assertAlmostEqual(events[0]["training_weight"], 1.2 * 1.5, places=4)

    def test_tension_weight_uses_held_not_failed(self):
        records = [
            _obs(record_id="r1", value_name="fairness",   label="P1", training_weight=2.0),
            _obs(record_id="r1", value_name="compassion", label="P0", training_weight=0.5),
        ]
        events = self._detect(records)
        # held = fairness (P1, weight=2.0) → tension weight = 2.0 * 1.5 = 3.0
        self.assertAlmostEqual(events[0]["training_weight"], 3.0, places=4)


# ==============================================================================
# Suite 8: co_occurrence in export report
# ==============================================================================

class TestCooccurrenceInReport(unittest.TestCase):

    def test_build_training_records_works_with_cooccurrence(self):
        """_compute_cooccurrence works on output of build_training_records."""
        from cli.export import build_training_records, _compute_cooccurrence
        records = build_training_records(
            [
                {
                    "id": str(uuid.uuid4()),
                    "session_id": "figure:test", "record_id": "r1", "ts": NOW,
                    "value_name": "courage", "text_excerpt": "I stood firm.",
                    "significance": 0.8, "resistance": 0.7, "figure_name": "Test",
                    "document_type": "journal", "observation_consistency": 0.5,
                    "disambiguation_confidence": 1.0,
                },
                {
                    "id": str(uuid.uuid4()),
                    "session_id": "figure:test", "record_id": "r1", "ts": NOW,
                    "value_name": "integrity", "text_excerpt": "I stood firm.",
                    "significance": 0.8, "resistance": 0.7, "figure_name": "Test",
                    "document_type": "journal", "observation_consistency": 0.5,
                    "disambiguation_confidence": 1.0,
                },
            ],
            p1_threshold=0.55, p0_threshold=0.35,
            min_observations=1, min_consistency=0.0,
        )
        matrix = _compute_cooccurrence(records)
        self.assertIn("courage||integrity", matrix)
        self.assertEqual(matrix["courage||integrity"]["both_detected"], 1)
        self.assertEqual(matrix["courage||integrity"]["both_p1"], 1)


# ==============================================================================
# Suite 9: Edge cases
# ==============================================================================

class TestEdgeCases(unittest.TestCase):

    def test_cooccurrence_empty_input(self):
        from cli.export import _compute_cooccurrence
        matrix = _compute_cooccurrence([])
        self.assertEqual(matrix, {})

    def test_cooccurrence_no_multi_value_passages(self):
        from cli.export import _compute_cooccurrence
        records = [
            _obs(record_id="r1", value_name="courage"),
            _obs(record_id="r2", value_name="integrity"),
        ]
        matrix = _compute_cooccurrence(records)
        self.assertEqual(len(matrix), 0)

    def test_detect_tensions_persists_to_db(self):
        from cli.export import _detect_tensions
        db_path = _tmp_db()
        store = ValueStore(db_path=db_path)
        records = [
            _obs(record_id="r1", value_name="responsibility", label="P1"),
            _obs(record_id="r1", value_name="humility",       label="P0"),
        ]
        _detect_tensions(records, db_path)
        tensions = store.get_tensions()
        self.assertEqual(len(tensions), 1)
        self.assertEqual(tensions[0]["value_held"], "responsibility")

    def test_is_tension_pair_case_sensitive(self):
        # Values in the pipeline are lowercased — uppercase should not match
        self.assertFalse(is_tension_pair("Independence", "Loyalty"))

    def test_cooccurrence_sorted_by_count_descending(self):
        from cli.export import _compute_cooccurrence
        records = [
            # courage||integrity appears twice
            _obs(record_id="r1", value_name="courage",   label="P1"),
            _obs(record_id="r1", value_name="integrity", label="P1"),
            _obs(record_id="r2", value_name="courage",   label="P1"),
            _obs(record_id="r2", value_name="integrity", label="P1"),
            # fairness||compassion appears once
            _obs(record_id="r3", value_name="fairness",   label="P1"),
            _obs(record_id="r3", value_name="compassion", label="P1"),
        ]
        matrix = _compute_cooccurrence(records)
        keys = list(matrix.keys())
        self.assertEqual(keys[0], "courage||integrity")  # highest count first


if __name__ == "__main__":
    unittest.main(verbosity=2)
