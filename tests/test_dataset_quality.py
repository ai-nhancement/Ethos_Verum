"""
tests/test_dataset_quality.py

Tests for the dataset quality scorecard (cli/dataset_quality.py).

Validates that:
  - Metrics are computed correctly from known inputs
  - Grading thresholds produce correct pass/fail verdicts
  - Edge cases (empty data, single record, all-same-label) are handled
  - Reproducibility hash is stable and order-independent
  - Per-line JSON errors are reported, not swallowed
  - Safe float conversion handles garbage gracefully
  - Per-figure minimum is enforced
  - Text quality check catches empty/short excerpts
  - Custom thresholds override defaults without mutation
"""

import json
from pathlib import Path

import pytest

import sys
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from cli.dataset_quality import (
    _compute_metrics, grade_dataset, _load_records,
    _safe_float, _percentile, DEFAULT_THRESHOLDS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rec(
    figure="lincoln",
    value_name="integrity",
    label="P1",
    confidence=0.90,
    resistance=0.65,
    document_type="journal",
    disambiguation_confidence=1.0,
    observation_consistency=0.50,
    text_excerpt="He stood firm despite the threat to his life and career.",
    ts=1000.0,
):
    """Create a minimal export record for testing."""
    return {
        "id": f"{figure}-{value_name}-{label}-{ts}",
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


def _good_dataset():
    """A dataset that passes all quality checks."""
    values = [
        "integrity", "courage", "compassion", "resilience", "patience",
        "fairness", "loyalty", "responsibility", "growth", "independence",
    ]
    labels =      ["P1", "P1", "P0", "P1", "AMBIGUOUS", "P1", "APY", "P0", "P1", "P1"]
    resistances = [0.85, 0.70, 0.30, 0.55, 0.45, 0.78, 0.40, 0.25, 0.65, 0.72]
    doc_types =   ["journal", "letter", "speech", "journal", "action", "letter",
                   "journal", "speech", "letter", "action"]
    records = []
    for i in range(30):
        idx = i % len(values)
        fig = ["lincoln", "gandhi", "douglass"][i % 3]
        records.append(_rec(
            figure=fig,
            value_name=values[idx],
            label=labels[idx],
            confidence=0.70 + (i % 5) * 0.05,
            resistance=resistances[idx] + (i % 3) * 0.05,
            document_type=doc_types[idx],
            ts=1000.0 + i,
        ))
    return records


def _bad_dataset():
    """A dataset that fails multiple checks."""
    return [
        _rec(value_name="integrity", label="P1", confidence=0.40,
             resistance=0.50, document_type="speech",
             text_excerpt="short", ts=1000 + i)
        for i in range(10)
    ]


# ---------------------------------------------------------------------------
# Tests: _safe_float
# ---------------------------------------------------------------------------

class TestSafeFloat:
    def test_normal(self):
        assert _safe_float(0.75) == 0.75

    def test_string_number(self):
        assert _safe_float("0.85") == 0.85

    def test_none(self):
        assert _safe_float(None) == 0.0

    def test_none_with_default(self):
        assert _safe_float(None, 0.5) == 0.5

    def test_garbage_string(self):
        assert _safe_float("high") == 0.0

    def test_nan(self):
        assert _safe_float(float("nan"), 0.5) == 0.5

    def test_inf(self):
        assert _safe_float(float("inf"), 0.5) == 0.5

    def test_int(self):
        assert _safe_float(1) == 1.0


# ---------------------------------------------------------------------------
# Tests: _percentile
# ---------------------------------------------------------------------------

class TestPercentile:
    def test_empty(self):
        assert _percentile([], 50) == 0.0

    def test_single(self):
        assert _percentile([0.7], 50) == 0.7

    def test_median_odd(self):
        assert _percentile([1, 2, 3, 4, 5], 50) == 3.0

    def test_5th_percentile(self):
        vals = list(range(100))
        p5 = _percentile(vals, 5)
        assert abs(p5 - 4.95) < 0.1

    def test_0th(self):
        assert _percentile([10, 20, 30], 0) == 10

    def test_100th(self):
        assert _percentile([10, 20, 30], 100) == 30


# ---------------------------------------------------------------------------
# Tests: _compute_metrics
# ---------------------------------------------------------------------------

class TestComputeMetrics:

    def test_empty_records(self):
        m = _compute_metrics([])
        assert "error" in m
        assert m["record_count"] == 0

    def test_record_count(self):
        records = [_rec(ts=i) for i in range(5)]
        m = _compute_metrics(records)
        assert m["record_count"] == 5

    def test_value_coverage(self):
        records = [
            _rec(value_name="integrity", ts=1),
            _rec(value_name="courage", ts=2),
            _rec(value_name="compassion", ts=3),
        ]
        m = _compute_metrics(records)
        assert m["value_coverage"] == 3
        assert set(m["values_detected"]) == {"integrity", "courage", "compassion"}

    def test_empty_value_name_excluded(self):
        records = [_rec(value_name="", ts=1), _rec(value_name="courage", ts=2)]
        m = _compute_metrics(records)
        assert m["value_coverage"] == 1

    def test_label_counts(self):
        records = [
            _rec(label="P1", ts=1),
            _rec(label="P1", ts=2),
            _rec(label="P0", ts=3),
        ]
        m = _compute_metrics(records)
        assert m["label_counts"]["P1"] == 2
        assert m["label_counts"]["P0"] == 1

    def test_label_balance_all_same(self):
        records = [_rec(label="P1", ts=i) for i in range(10)]
        m = _compute_metrics(records)
        assert m["max_label_pct"] == 1.0
        assert m["max_label_name"] == "P1"
        assert m["has_p1"] is True
        assert m["has_p0"] is False

    def test_label_balance_mixed(self):
        records = [
            _rec(label="P1", ts=1), _rec(label="P0", ts=2),
            _rec(label="APY", ts=3), _rec(label="AMBIGUOUS", ts=4),
        ]
        m = _compute_metrics(records)
        assert m["max_label_pct"] == 0.25
        assert m["has_p1"] is True
        assert m["has_p0"] is True

    def test_avg_confidence(self):
        records = [_rec(confidence=0.80, ts=1), _rec(confidence=0.60, ts=2)]
        m = _compute_metrics(records)
        assert m["avg_confidence"] == 0.70

    def test_confidence_with_garbage_value(self):
        records = [
            _rec(confidence=0.80, ts=1),
            _rec(ts=2),  # default 0.90
        ]
        records[1]["confidence"] = "broken"
        m = _compute_metrics(records)
        # "broken" → 0.0 via _safe_float, avg = (0.80 + 0.0) / 2 = 0.40
        assert m["avg_confidence"] == 0.40

    def test_resistance_spread(self):
        records = [_rec(resistance=0.20, ts=1), _rec(resistance=0.80, ts=2)]
        m = _compute_metrics(records)
        assert abs(m["resistance_spread"] - 0.30) < 0.01

    def test_resistance_spread_zero(self):
        records = [_rec(resistance=0.50, ts=i) for i in range(5)]
        m = _compute_metrics(records)
        assert m["resistance_spread"] == 0.0

    def test_source_diversity_excludes_unknown(self):
        records = [
            _rec(document_type="journal", ts=1),
            _rec(document_type="unknown", ts=2),
        ]
        m = _compute_metrics(records)
        assert m["source_diversity"] == 1  # only journal counts
        assert "unknown" in m["doc_types"]
        assert "unknown" not in m["doc_types_real"]

    def test_source_diversity_real(self):
        records = [
            _rec(document_type="journal", ts=1),
            _rec(document_type="letter", ts=2),
            _rec(document_type="speech", ts=3),
        ]
        m = _compute_metrics(records)
        assert m["source_diversity"] == 3

    def test_disambiguation_rate(self):
        records = [
            _rec(disambiguation_confidence=1.0, ts=1),
            _rec(disambiguation_confidence=0.9, ts=2),
            _rec(disambiguation_confidence=0.5, ts=3),
        ]
        m = _compute_metrics(records)
        assert abs(m["disambiguation_rate"] - 0.6667) < 0.01

    def test_consistency_5th_percentile(self):
        # 20 records at 0.80, 1 record at 0.10 — min would be 0.10 but 5th pct is higher
        records = [_rec(observation_consistency=0.80, ts=i) for i in range(20)]
        records.append(_rec(observation_consistency=0.10, ts=100))
        m = _compute_metrics(records)
        assert m["consistency_pct5"] > 0.10  # 5th percentile is NOT the min

    def test_text_quality_rate(self):
        records = [
            _rec(text_excerpt="This is a long enough excerpt for quality check.", ts=1),
            _rec(text_excerpt="This is also a long enough one for the check.", ts=2),
            _rec(text_excerpt="short", ts=3),
        ]
        m = _compute_metrics(records)
        assert abs(m["text_quality_rate"] - 0.6667) < 0.01

    def test_per_figure_counts(self):
        records = [
            _rec(figure="lincoln", ts=1),
            _rec(figure="lincoln", ts=2),
            _rec(figure="gandhi", ts=3),
        ]
        m = _compute_metrics(records)
        assert m["figure_counts"]["lincoln"] == 2
        assert m["figure_counts"]["gandhi"] == 1
        assert m["min_figure_records"] == 1
        assert m["weakest_figure"] == "gandhi"

    def test_reproducibility_hash_stable(self):
        records = _good_dataset()
        m1 = _compute_metrics(records)
        m2 = _compute_metrics(records)
        assert m1["reproducibility_hash"] == m2["reproducibility_hash"]

    def test_reproducibility_hash_order_independent(self):
        records = _good_dataset()
        m1 = _compute_metrics(records)
        m2 = _compute_metrics(list(reversed(records)))
        assert m1["reproducibility_hash"] == m2["reproducibility_hash"]

    def test_reproducibility_hash_changes_with_data(self):
        r1 = _good_dataset()
        r2 = _good_dataset()
        r2[0]["label"] = "P0"  # mutate one record
        m1 = _compute_metrics(r1)
        m2 = _compute_metrics(r2)
        assert m1["reproducibility_hash"] != m2["reproducibility_hash"]

    def test_by_figure_breakdown(self):
        records = [
            _rec(figure="lincoln", label="P1", ts=1),
            _rec(figure="lincoln", label="P0", ts=2),
            _rec(figure="gandhi", label="P1", ts=3),
        ]
        m = _compute_metrics(records)
        assert m["by_figure"]["lincoln"]["P1"] == 1
        assert m["by_figure"]["lincoln"]["P0"] == 1
        assert m["by_figure"]["gandhi"]["P1"] == 1

    def test_by_value_breakdown(self):
        records = [
            _rec(value_name="integrity", label="P1", ts=1),
            _rec(value_name="integrity", label="P0", ts=2),
            _rec(value_name="courage", label="P1", ts=3),
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
        records = _good_dataset()
        m = _compute_metrics(records)
        result = grade_dataset(m)
        assert result["grade"] == "CERTIFIED", \
            f"Expected CERTIFIED but got FAILED: " \
            f"{[c['metric'] for c in result['checks'] if not c['passed']]}"
        assert result["failed"] == 0

    def test_bad_dataset_fails(self):
        records = _bad_dataset()
        m = _compute_metrics(records)
        result = grade_dataset(m)
        assert result["grade"] == "FAILED"
        assert result["failed"] > 0

    def test_empty_dataset_error(self):
        m = _compute_metrics([])
        result = grade_dataset(m)
        assert result["grade"] == "ERROR"

    def test_all_p1_fails_balance_and_p0_check(self):
        records = _good_dataset()
        for r in records:
            r["label"] = "P1"
        m = _compute_metrics(records)
        result = grade_dataset(m)
        failed_names = {c["metric"] for c in result["checks"] if not c["passed"]}
        assert "Label Balance" in failed_names
        assert "P1+P0 Present" in failed_names

    def test_low_value_coverage_fails(self):
        records = [_rec(value_name="integrity", label=["P1", "P0"][i % 2], ts=i,
                        confidence=0.85, resistance=0.3 + (i % 5) * 0.1,
                        document_type=["journal", "letter"][i % 2])
                   for i in range(30)]
        m = _compute_metrics(records)
        result = grade_dataset(m)
        failed_names = {c["metric"] for c in result["checks"] if not c["passed"]}
        assert "Value Coverage" in failed_names

    def test_low_confidence_fails(self):
        records = _good_dataset()
        for r in records:
            r["confidence"] = 0.30
        m = _compute_metrics(records)
        result = grade_dataset(m)
        failed_names = {c["metric"] for c in result["checks"] if not c["passed"]}
        assert "Avg Confidence" in failed_names

    def test_single_doc_type_fails(self):
        records = _good_dataset()
        for r in records:
            r["document_type"] = "unknown"
        m = _compute_metrics(records)
        result = grade_dataset(m)
        failed_names = {c["metric"] for c in result["checks"] if not c["passed"]}
        assert "Source Diversity" in failed_names

    def test_short_text_fails(self):
        records = _good_dataset()
        for r in records:
            r["text_excerpt"] = "x"
        m = _compute_metrics(records)
        result = grade_dataset(m)
        failed_names = {c["metric"] for c in result["checks"] if not c["passed"]}
        assert "Text Quality" in failed_names

    def test_too_few_records_fails(self):
        records = _good_dataset()[:5]
        m = _compute_metrics(records)
        result = grade_dataset(m)
        failed_names = {c["metric"] for c in result["checks"] if not c["passed"]}
        assert "Record Count" in failed_names

    def test_weak_figure_fails(self):
        records = _good_dataset()
        # Add a figure with only 1 record
        records.append(_rec(figure="weak_figure", label="P1", ts=9999))
        m = _compute_metrics(records)
        result = grade_dataset(m)
        failed_names = {c["metric"] for c in result["checks"] if not c["passed"]}
        assert "Per-Figure Minimum" in failed_names

    def test_check_count(self):
        records = _good_dataset()
        m = _compute_metrics(records)
        result = grade_dataset(m)
        assert result["total_checks"] == 11

    def test_all_checks_have_required_fields(self):
        records = _good_dataset()
        m = _compute_metrics(records)
        result = grade_dataset(m)
        for c in result["checks"]:
            assert "metric" in c
            assert "value" in c
            assert "threshold" in c
            assert "passed" in c
            assert "detail" in c
            assert "op" in c

    def test_custom_thresholds_override(self):
        records = _good_dataset()
        m = _compute_metrics(records)
        # With impossibly high threshold, should fail
        result = grade_dataset(m, thresholds={"min_records": 99999})
        failed_names = {c["metric"] for c in result["checks"] if not c["passed"]}
        assert "Record Count" in failed_names

    def test_custom_thresholds_dont_mutate_defaults(self):
        original_min = DEFAULT_THRESHOLDS["min_records"]
        records = _good_dataset()
        m = _compute_metrics(records)
        grade_dataset(m, thresholds={"min_records": 99999})
        assert DEFAULT_THRESHOLDS["min_records"] == original_min


# ---------------------------------------------------------------------------
# Tests: _load_records
# ---------------------------------------------------------------------------

class TestLoadRecords:

    def test_load_jsonl_file(self, tmp_path):
        f = tmp_path / "test.jsonl"
        records = [_rec(ts=i) for i in range(3)]
        f.write_text("\n".join(json.dumps(r) for r in records))
        loaded, warnings = _load_records(str(f))
        assert len(loaded) == 3
        assert len(warnings) == 0

    def test_load_directory(self, tmp_path):
        for name, count in [("ric_positive.jsonl", 2), ("ric_negative.jsonl", 3)]:
            f = tmp_path / name
            records = [_rec(ts=i, figure=name, value_name=f"v{i}") for i in range(count)]
            f.write_text("\n".join(json.dumps(r) for r in records))
        loaded, _ = _load_records(str(tmp_path))
        assert len(loaded) == 5

    def test_deduplication(self, tmp_path):
        rec = _rec(ts=1)
        f1 = tmp_path / "ric_a.jsonl"
        f2 = tmp_path / "ric_b.jsonl"
        f1.write_text(json.dumps(rec))
        f2.write_text(json.dumps(rec))
        loaded, _ = _load_records(str(tmp_path))
        assert len(loaded) == 1

    def test_empty_file(self, tmp_path):
        f = tmp_path / "empty.jsonl"
        f.write_text("")
        loaded, _ = _load_records(str(f))
        assert len(loaded) == 0

    def test_nonexistent_path(self):
        loaded, warnings = _load_records("/nonexistent/path")
        assert len(loaded) == 0
        assert len(warnings) > 0

    def test_blank_lines_skipped(self, tmp_path):
        f = tmp_path / "test.jsonl"
        f.write_text(json.dumps(_rec(ts=1)) + "\n\n\n" + json.dumps(_rec(ts=2)))
        loaded, _ = _load_records(str(f))
        assert len(loaded) == 2

    def test_malformed_json_reports_warning(self, tmp_path):
        f = tmp_path / "test.jsonl"
        f.write_text(
            json.dumps(_rec(ts=1)) + "\n"
            + "this is not json\n"
            + json.dumps(_rec(ts=2)) + "\n"
        )
        loaded, warnings = _load_records(str(f))
        assert len(loaded) == 2  # good lines still loaded
        assert len(warnings) == 1  # bad line reported
        assert "malformed JSON" in warnings[0]

    def test_records_without_id_not_deduped(self, tmp_path):
        rec1 = _rec(ts=1)
        rec2 = _rec(ts=1)
        del rec1["id"]
        del rec1["source_obs_id"]
        del rec2["id"]
        del rec2["source_obs_id"]
        f = tmp_path / "test.jsonl"
        f.write_text(json.dumps(rec1) + "\n" + json.dumps(rec2))
        loaded, _ = _load_records(str(f))
        assert len(loaded) == 2  # both loaded since no ID to dedup on
