"""
tests/test_dataset_quality.py

Tests for the dataset quality scorecard (cli/dataset_quality.py).

Validates that:
  - Metrics are computed correctly from known inputs
  - Grading thresholds produce correct pass/fail verdicts
  - Edge cases (empty data, single record, all-same-label) are handled
  - Reproducibility hash is stable across runs
  - The scorecard correctly fails bad data and passes good data
"""

import json
import math
import os
import tempfile
from pathlib import Path

import pytest

import sys
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from cli.dataset_quality import _compute_metrics, grade_dataset, _load_records, THRESHOLDS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_record(
    figure="lincoln",
    value_name="integrity",
    label="P1",
    confidence=0.90,
    resistance=0.65,
    document_type="journal",
    disambiguation_confidence=1.0,
    observation_consistency=0.50,
    text_excerpt="He stood firm despite the threat.",
    ts=1000.0,
):
    """Create a minimal export record for testing."""
    return {
        "id": f"{figure}-{value_name}-{ts}",
        "source_obs_id": f"obs-{figure}-{ts}",
        "figure": figure,
        "value_name": value_name,
        "label": label,
        "confidence": confidence,
        "resistance": resistance,
        "document_type": document_type,
        "disambiguation_confidence": disambiguation_confidence,
        "observation_consistency": observation_consistency,
        "text_excerpt": text_excerpt,
        "ts": ts,
    }


def _make_good_dataset():
    """Create a dataset that should pass all quality checks."""
    records = []
    values = ["integrity", "courage", "compassion", "resilience", "patience",
              "fairness", "loyalty", "responsibility"]
    labels = ["P1", "P1", "P1", "P0", "P1", "AMBIGUOUS", "P1", "APY"]
    resistances = [0.85, 0.70, 0.55, 0.30, 0.65, 0.45, 0.78, 0.40]
    doc_types = ["journal", "letter", "speech", "journal", "action", "letter", "journal", "speech"]

    for i, (v, l, r, d) in enumerate(zip(values, labels, resistances, doc_types)):
        records.append(_make_record(
            figure="lincoln" if i < 4 else "gandhi",
            value_name=v,
            label=l,
            confidence=0.75 + (i * 0.02),
            resistance=r,
            document_type=d,
            ts=1000.0 + i,
        ))
    return records


def _make_bad_dataset():
    """Create a dataset that should fail multiple checks."""
    # All same value, all same label, all same resistance, one doc type
    return [
        _make_record(value_name="integrity", label="P1", confidence=0.40,
                     resistance=0.50, document_type="speech", ts=1000 + i)
        for i in range(10)
    ]


# ---------------------------------------------------------------------------
# Tests: _compute_metrics
# ---------------------------------------------------------------------------

class TestComputeMetrics:

    def test_empty_records(self):
        m = _compute_metrics([])
        assert "error" in m
        assert m["record_count"] == 0

    def test_record_count(self):
        records = [_make_record(ts=i) for i in range(5)]
        m = _compute_metrics(records)
        assert m["record_count"] == 5

    def test_value_coverage(self):
        records = [
            _make_record(value_name="integrity", ts=1),
            _make_record(value_name="courage", ts=2),
            _make_record(value_name="compassion", ts=3),
        ]
        m = _compute_metrics(records)
        assert m["value_coverage"] == 3
        assert set(m["values_detected"]) == {"integrity", "courage", "compassion"}

    def test_label_counts(self):
        records = [
            _make_record(label="P1", ts=1),
            _make_record(label="P1", ts=2),
            _make_record(label="P0", ts=3),
        ]
        m = _compute_metrics(records)
        assert m["label_counts"]["P1"] == 2
        assert m["label_counts"]["P0"] == 1

    def test_label_balance_all_same(self):
        records = [_make_record(label="P1", ts=i) for i in range(10)]
        m = _compute_metrics(records)
        assert m["max_label_pct"] == 1.0
        assert m["max_label_name"] == "P1"

    def test_label_balance_mixed(self):
        records = [
            _make_record(label="P1", ts=1),
            _make_record(label="P0", ts=2),
            _make_record(label="APY", ts=3),
            _make_record(label="AMBIGUOUS", ts=4),
        ]
        m = _compute_metrics(records)
        assert m["max_label_pct"] == 0.25

    def test_signal_density(self):
        records = [
            _make_record(confidence=0.90, ts=1),
            _make_record(confidence=0.70, ts=2),
            _make_record(confidence=0.40, ts=3),  # below 0.65
            _make_record(confidence=0.30, ts=4),  # below 0.65
        ]
        m = _compute_metrics(records)
        assert m["signal_density"] == 0.5  # 2 of 4

    def test_avg_confidence(self):
        records = [
            _make_record(confidence=0.80, ts=1),
            _make_record(confidence=0.60, ts=2),
        ]
        m = _compute_metrics(records)
        assert m["avg_confidence"] == 0.70

    def test_resistance_spread(self):
        records = [
            _make_record(resistance=0.20, ts=1),
            _make_record(resistance=0.80, ts=2),
        ]
        m = _compute_metrics(records)
        # mean = 0.50, values are 0.30 from mean each, variance = 0.09, std = 0.30
        assert abs(m["resistance_spread"] - 0.30) < 0.01

    def test_resistance_spread_zero(self):
        records = [_make_record(resistance=0.50, ts=i) for i in range(5)]
        m = _compute_metrics(records)
        assert m["resistance_spread"] == 0.0

    def test_source_diversity(self):
        records = [
            _make_record(document_type="journal", ts=1),
            _make_record(document_type="letter", ts=2),
            _make_record(document_type="speech", ts=3),
        ]
        m = _compute_metrics(records)
        assert m["source_diversity"] == 3

    def test_disambiguation_rate(self):
        records = [
            _make_record(disambiguation_confidence=1.0, ts=1),
            _make_record(disambiguation_confidence=0.9, ts=2),
            _make_record(disambiguation_confidence=0.5, ts=3),  # below 0.8
        ]
        m = _compute_metrics(records)
        assert abs(m["disambiguation_rate"] - 0.6667) < 0.01

    def test_consistency_floor(self):
        records = [
            _make_record(observation_consistency=0.80, ts=1),
            _make_record(observation_consistency=0.30, ts=2),
            _make_record(observation_consistency=0.60, ts=3),
        ]
        m = _compute_metrics(records)
        assert m["consistency_floor"] == 0.30

    def test_reproducibility_hash_stable(self):
        records = _make_good_dataset()
        m1 = _compute_metrics(records)
        m2 = _compute_metrics(records)
        assert m1["reproducibility_hash"] == m2["reproducibility_hash"]

    def test_reproducibility_hash_order_independent(self):
        records = _make_good_dataset()
        m1 = _compute_metrics(records)
        m2 = _compute_metrics(list(reversed(records)))
        assert m1["reproducibility_hash"] == m2["reproducibility_hash"]

    def test_by_figure_breakdown(self):
        records = [
            _make_record(figure="lincoln", label="P1", ts=1),
            _make_record(figure="lincoln", label="P0", ts=2),
            _make_record(figure="gandhi", label="P1", ts=3),
        ]
        m = _compute_metrics(records)
        assert "lincoln" in m["by_figure"]
        assert m["by_figure"]["lincoln"]["P1"] == 1
        assert m["by_figure"]["lincoln"]["P0"] == 1
        assert m["by_figure"]["gandhi"]["P1"] == 1

    def test_by_value_breakdown(self):
        records = [
            _make_record(value_name="integrity", label="P1", ts=1),
            _make_record(value_name="integrity", label="P0", ts=2),
            _make_record(value_name="courage", label="P1", ts=3),
        ]
        m = _compute_metrics(records)
        assert m["by_value"]["integrity"]["P1"] == 1
        assert m["by_value"]["integrity"]["P0"] == 1
        assert m["by_value"]["courage"]["P1"] == 1


# ---------------------------------------------------------------------------
# Tests: grade_dataset
# ---------------------------------------------------------------------------

class TestGradeDataset:

    def test_good_dataset_certified(self):
        records = _make_good_dataset()
        m = _compute_metrics(records)
        result = grade_dataset(m)
        assert result["grade"] == "CERTIFIED"
        assert result["failed"] == 0

    def test_bad_dataset_fails(self):
        records = _make_bad_dataset()
        m = _compute_metrics(records)
        result = grade_dataset(m)
        assert result["grade"] == "FAILED"
        assert result["failed"] > 0

    def test_empty_dataset_error(self):
        m = _compute_metrics([])
        result = grade_dataset(m)
        assert result["grade"] == "ERROR"

    def test_single_label_fails_balance(self):
        records = [_make_record(label="P1", value_name=v, ts=i,
                                confidence=0.85, resistance=0.2 * (i + 1),
                                document_type=["journal", "letter", "speech"][i % 3])
                   for i, v in enumerate(["integrity", "courage", "compassion",
                                          "resilience", "patience"])]
        m = _compute_metrics(records)
        result = grade_dataset(m)
        failed_names = [c["metric"] for c in result["checks"] if not c["passed"]]
        assert "Label Balance" in failed_names

    def test_low_value_coverage_fails(self):
        records = [_make_record(value_name="integrity", label=l, ts=i,
                                confidence=0.85, resistance=0.3 + i * 0.1,
                                document_type=["journal", "letter"][i % 2])
                   for i, l in enumerate(["P1", "P0", "P1", "APY"])]
        m = _compute_metrics(records)
        result = grade_dataset(m)
        failed_names = [c["metric"] for c in result["checks"] if not c["passed"]]
        assert "Value Coverage" in failed_names

    def test_low_confidence_fails(self):
        records = [_make_record(confidence=0.40, value_name=v, ts=i,
                                label=["P1", "P0"][i % 2],
                                resistance=0.2 + i * 0.1,
                                document_type=["journal", "letter", "speech"][i % 3])
                   for i, v in enumerate(["integrity", "courage", "compassion",
                                          "resilience", "patience"])]
        m = _compute_metrics(records)
        result = grade_dataset(m)
        failed_names = [c["metric"] for c in result["checks"] if not c["passed"]]
        assert "Avg Confidence" in failed_names
        assert "Signal Density" in failed_names

    def test_check_count(self):
        records = _make_good_dataset()
        m = _compute_metrics(records)
        result = grade_dataset(m)
        assert result["total_checks"] == 8

    def test_all_checks_have_required_fields(self):
        records = _make_good_dataset()
        m = _compute_metrics(records)
        result = grade_dataset(m)
        for c in result["checks"]:
            assert "metric" in c
            assert "value" in c
            assert "threshold" in c
            assert "passed" in c
            assert "detail" in c


# ---------------------------------------------------------------------------
# Tests: _load_records
# ---------------------------------------------------------------------------

class TestLoadRecords:

    def test_load_jsonl_file(self, tmp_path):
        f = tmp_path / "test.jsonl"
        records = [_make_record(ts=i) for i in range(3)]
        f.write_text("\n".join(json.dumps(r) for r in records))
        loaded = _load_records(str(f))
        assert len(loaded) == 3

    def test_load_directory(self, tmp_path):
        for name, count in [("ric_positive.jsonl", 2), ("ric_negative.jsonl", 3)]:
            f = tmp_path / name
            records = [_make_record(ts=i, figure=name, value_name=f"v{i}") for i in range(count)]
            f.write_text("\n".join(json.dumps(r) for r in records))
        loaded = _load_records(str(tmp_path))
        assert len(loaded) == 5

    def test_deduplication(self, tmp_path):
        rec = _make_record(ts=1)
        f1 = tmp_path / "ric_a.jsonl"
        f2 = tmp_path / "ric_b.jsonl"
        f1.write_text(json.dumps(rec))
        f2.write_text(json.dumps(rec))
        loaded = _load_records(str(tmp_path))
        assert len(loaded) == 1

    def test_empty_file(self, tmp_path):
        f = tmp_path / "empty.jsonl"
        f.write_text("")
        loaded = _load_records(str(f))
        assert len(loaded) == 0

    def test_nonexistent_path(self):
        loaded = _load_records("/nonexistent/path")
        assert len(loaded) == 0

    def test_blank_lines_skipped(self, tmp_path):
        f = tmp_path / "test.jsonl"
        rec = _make_record(ts=1)
        f.write_text(json.dumps(rec) + "\n\n\n" + json.dumps(_make_record(ts=2)))
        loaded = _load_records(str(f))
        assert len(loaded) == 2
