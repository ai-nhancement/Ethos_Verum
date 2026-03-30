#!/usr/bin/env python3
"""
cli/dataset_quality.py

Dataset Quality Scorecard — deterministic quality assessment for exported JSONL.

Reads a finished export (JSONL files) and produces a pass/fail grade with
specific, reproducible metrics. No human judgment. The numbers decide.

Metrics:
   1. Record Count        — minimum records to qualify as a dataset (>= 20)
   2. Value Coverage      — distinct values with signals (>= 5 to pass)
   3. Label Distribution  — at least 2 labels present, none > 80%, P1 and P0 both present
   4. Avg Confidence      — mean classification confidence (>= 0.65 to pass)
   5. Resistance Spread   — std dev of resistance scores (>= 0.10 to pass)
   6. Source Diversity     — distinct non-"unknown" document types (>= 2 to pass)
   7. Disambiguation Rate — % of signals with disambiguation_confidence >= 0.8 (>= 85%)
   8. Consistency Floor   — 5th percentile of observation_consistency (>= 0.25)
   9. Text Quality        — % of records with text_excerpt >= 30 chars (>= 90%)
  10. Per-Figure Minimum  — every figure has >= 5 records

Usage:
  python -m cli.dataset_quality --input output/ric/
  python -m cli.dataset_quality --input output/ric/ --json
  python -m cli.dataset_quality --input export.jsonl

Exit codes:
  0 — CERTIFIED (all metrics pass)
  1 — FAILED (one or more metrics failed)
  2 — ERROR (could not read input)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Thresholds — deterministic, no human judgment
# ---------------------------------------------------------------------------

DEFAULT_THRESHOLDS = {
    "min_records":              20,     # minimum records to be a dataset
    "value_coverage_min":       5,      # distinct values with signals
    "label_max_pct":            0.80,   # no single label > 80%
    "label_min_distinct":       2,      # at least 2 distinct labels with count >= 2
    "require_p1_and_p0":        True,   # both P1 and P0 must be present
    "avg_confidence_min":       0.65,   # mean confidence
    "resistance_spread_min":    0.10,   # std dev of resistance
    "source_diversity_min":     2,      # distinct non-"unknown" doc types
    "disambiguation_rate_min":  0.85,   # % with disambig >= 0.8
    "consistency_pct5_min":     0.25,   # 5th percentile of observation_consistency
    "text_quality_min_chars":   30,     # minimum chars for a text_excerpt to count
    "text_quality_rate_min":    0.90,   # % of records meeting text_quality_min_chars
    "per_figure_min_records":   5,      # every figure must have at least this many
}


# ---------------------------------------------------------------------------
# Safe numeric conversion
# ---------------------------------------------------------------------------

def _safe_float(val: Any, default: float = 0.0) -> float:
    """Convert to float without raising. Returns default on any failure."""
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return default
        return f
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Load records
# ---------------------------------------------------------------------------

def _load_records(input_path: str) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Load records from a JSONL file or all JSONL files in a directory.
    Returns (records, warnings) where warnings lists any issues encountered.
    """
    path = Path(input_path)
    records: List[Dict[str, Any]] = []
    warnings: List[str] = []

    if path.is_file() and path.suffix == ".jsonl":
        files = [path]
    elif path.is_dir():
        files = sorted(path.glob("ric_*.jsonl"))
        if not files:
            files = sorted(path.glob("*.jsonl"))
    else:
        return [], [f"Path does not exist or is not a file/directory: {input_path}"]

    seen_ids: set = set()
    for f in files:
        try:
            lines = f.read_text(encoding="utf-8").splitlines()
        except Exception as e:
            warnings.append(f"Could not read {f.name}: {e}")
            continue

        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                warnings.append(f"{f.name}:{line_num}: malformed JSON: {e}")
                continue

            # Deduplicate across files
            rec_id = rec.get("id") or rec.get("source_obs_id") or ""
            if rec_id:
                if rec_id in seen_ids:
                    continue
                seen_ids.add(rec_id)

            records.append(rec)

    return records, warnings


# Need this for the type hint above — placed after use to avoid forward ref issues
from typing import Tuple  # noqa: E402


# ---------------------------------------------------------------------------
# Percentile helper
# ---------------------------------------------------------------------------

def _percentile(values: List[float], pct: float) -> float:
    """Return the pct-th percentile (0-100) of a sorted list. Linear interpolation."""
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    if n == 1:
        return s[0]
    k = (pct / 100.0) * (n - 1)
    lo = int(k)
    hi = min(lo + 1, n - 1)
    frac = k - lo
    return s[lo] + frac * (s[hi] - s[lo])


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _compute_metrics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute all quality metrics from a list of export records."""
    n = len(records)
    if n == 0:
        return {"error": "No records to evaluate", "record_count": 0}

    # ── Value Coverage ──
    values_seen = set()
    for r in records:
        vn = r.get("value_name", "")
        if vn:
            values_seen.add(vn)
    value_coverage = len(values_seen)

    # ── Label Distribution ──
    label_counts: Dict[str, int] = defaultdict(int)
    for r in records:
        label_counts[r.get("label", "UNKNOWN")] += 1
    max_label_pct = max(label_counts.values()) / n
    max_label_name = max(label_counts, key=label_counts.get)
    distinct_labels_with_mass = sum(1 for c in label_counts.values() if c >= 2)
    has_p1 = label_counts.get("P1", 0) > 0
    has_p0 = label_counts.get("P0", 0) > 0

    # ── Average Confidence ──
    confidences = [_safe_float(r.get("confidence"), 0.0) for r in records]
    avg_confidence = sum(confidences) / n

    # ── Resistance Spread ──
    resistances = [_safe_float(r.get("resistance"), 0.0) for r in records]
    mean_resistance = sum(resistances) / n
    variance = sum((x - mean_resistance) ** 2 for x in resistances) / n
    resistance_spread = math.sqrt(variance)

    # ── Source Diversity (exclude "unknown") ──
    doc_types_all = set()
    doc_types_real = set()
    for r in records:
        dt = r.get("document_type", "unknown")
        if dt:
            doc_types_all.add(dt)
            if dt.lower() != "unknown":
                doc_types_real.add(dt)
    source_diversity = len(doc_types_real)

    # ── Disambiguation Rate ──
    disambig_scores = [_safe_float(r.get("disambiguation_confidence"), 1.0) for r in records]
    disambig_high = sum(1 for d in disambig_scores if d >= 0.8)
    disambiguation_rate = disambig_high / n

    # ── Consistency Floor (5th percentile, not min) ──
    consistencies = [_safe_float(r.get("observation_consistency"), 0.5) for r in records]
    consistency_pct5 = _percentile(consistencies, 5)

    # ── Text Quality ──
    text_lengths = [len(r.get("text_excerpt", "")) for r in records]
    text_ok_count = sum(1 for tl in text_lengths if tl >= 30)
    text_quality_rate = text_ok_count / n

    # ── Per-Figure Record Counts ──
    figure_counts: Dict[str, int] = defaultdict(int)
    for r in records:
        figure_counts[r.get("figure", "unknown")] += 1
    min_figure_records = min(figure_counts.values()) if figure_counts else 0
    weakest_figure = min(figure_counts, key=figure_counts.get) if figure_counts else "N/A"

    # ── Reproducibility Hash ──
    # Sort by deterministic key, hash with rounded floats for stability
    sorted_records = sorted(records, key=lambda r: (
        r.get("figure", ""),
        r.get("value_name", ""),
        r.get("text_excerpt", "")[:80],
        _safe_float(r.get("ts"), 0),
    ))
    content_for_hash = json.dumps(
        [
            {
                "figure": r.get("figure", ""),
                "value_name": r.get("value_name", ""),
                "text_excerpt": r.get("text_excerpt", "")[:80],
                "label": r.get("label", ""),
                "resistance": round(_safe_float(r.get("resistance")), 4),
                "confidence": round(_safe_float(r.get("confidence")), 2),
            }
            for r in sorted_records
        ],
        sort_keys=True,
        ensure_ascii=False,
    )
    reproducibility_hash = hashlib.sha256(content_for_hash.encode()).hexdigest()[:16]

    # ── Per-figure breakdown ──
    by_figure: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in records:
        fig = r.get("figure", "unknown")
        by_figure[fig][r.get("label", "UNKNOWN")] += 1
        by_figure[fig]["_total"] += 1

    # ── Per-value breakdown ──
    by_value: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in records:
        vn = r.get("value_name", "unknown")
        by_value[vn][r.get("label", "UNKNOWN")] += 1
        by_value[vn]["_total"] += 1

    return {
        "record_count":             n,
        "value_coverage":           value_coverage,
        "values_detected":          sorted(values_seen),
        "label_counts":             dict(label_counts),
        "max_label_pct":            round(max_label_pct, 4),
        "max_label_name":           max_label_name,
        "distinct_labels_with_mass": distinct_labels_with_mass,
        "has_p1":                   has_p1,
        "has_p0":                   has_p0,
        "avg_confidence":           round(avg_confidence, 4),
        "resistance_mean":          round(mean_resistance, 4),
        "resistance_spread":        round(resistance_spread, 4),
        "source_diversity":         source_diversity,
        "doc_types":                sorted(doc_types_all),
        "doc_types_real":           sorted(doc_types_real),
        "disambiguation_rate":      round(disambiguation_rate, 4),
        "consistency_pct5":         round(consistency_pct5, 4),
        "text_quality_rate":        round(text_quality_rate, 4),
        "min_figure_records":       min_figure_records,
        "weakest_figure":           weakest_figure,
        "figure_counts":            dict(figure_counts),
        "reproducibility_hash":     reproducibility_hash,
        "by_figure":                {k: dict(v) for k, v in sorted(by_figure.items())},
        "by_value":                 {k: dict(v) for k, v in sorted(by_value.items())},
    }


# ---------------------------------------------------------------------------
# Layer 1: Statistical Shape (grade_layer1)
# ---------------------------------------------------------------------------

def _check(checks: List[Dict], name: str, value, threshold, op: str = ">=", detail: str = ""):
    """Append a check result to the checks list."""
    if op == ">=":
        passed = value >= threshold
    elif op == "<=":
        passed = value <= threshold
    elif op == "==":
        passed = value == threshold
    else:
        passed = False
    checks.append({
        "metric": name, "value": value, "threshold": threshold,
        "op": op, "passed": passed, "detail": detail,
    })


def _grade_layer1(metrics: Dict[str, Any], T: Dict[str, Any]) -> Dict[str, Any]:
    """Layer 1: Statistical shape — does the dataset have the right structure?"""
    checks: List[Dict[str, Any]] = []

    try:
        _check(checks, "Record Count", metrics["record_count"], T["min_records"],
               detail=f"{metrics['record_count']} records in dataset")

        _check(checks, "Value Coverage", metrics["value_coverage"], T["value_coverage_min"],
               detail=f"{metrics['value_coverage']} of 15 values detected: "
                      f"{', '.join(metrics['values_detected'][:8])}"
                      f"{'...' if len(metrics['values_detected']) > 8 else ''}")

        _check(checks, "Label Balance", metrics["max_label_pct"], T["label_max_pct"], op="<=",
               detail=f"Dominant label: {metrics['max_label_name']} at {metrics['max_label_pct']*100:.1f}%")

        both_present = metrics["has_p1"] and metrics["has_p0"]
        _check(checks, "P1+P0 Present", both_present, True, op="==",
               detail=f"P1: {metrics['label_counts'].get('P1', 0)}, "
                      f"P0: {metrics['label_counts'].get('P0', 0)} — "
                      f"{'both present' if both_present else 'MISSING ' + ('P0' if metrics['has_p1'] else 'P1') + ' signals'}")

        _check(checks, "Avg Confidence", metrics["avg_confidence"], T["avg_confidence_min"],
               detail=f"Mean classification confidence: {metrics['avg_confidence']:.4f}")

        _check(checks, "Resistance Spread", metrics["resistance_spread"], T["resistance_spread_min"],
               detail=f"Std dev: {metrics['resistance_spread']:.4f}, mean: {metrics['resistance_mean']:.4f}")

        _check(checks, "Source Diversity", metrics["source_diversity"], T["source_diversity_min"],
               detail=f"Real document types: {', '.join(metrics['doc_types_real']) or 'none'}"
                      f"{' (unknown excluded)' if 'unknown' in metrics['doc_types'] else ''}")

        _check(checks, "Disambiguation Rate", metrics["disambiguation_rate"], T["disambiguation_rate_min"],
               detail=f"{metrics['disambiguation_rate']*100:.1f}% of signals have disambiguation >= 0.8")

        _check(checks, "Consistency Floor", metrics["consistency_pct5"], T["consistency_pct5_min"],
               detail=f"5th percentile of observation_consistency: {metrics['consistency_pct5']:.4f}")

        _check(checks, "Text Quality", metrics["text_quality_rate"], T["text_quality_rate_min"],
               detail=f"{metrics['text_quality_rate']*100:.1f}% of records have text_excerpt >= {T['text_quality_min_chars']} chars")

        _check(checks, "Per-Figure Minimum", metrics["min_figure_records"], T["per_figure_min_records"],
               detail=f"Weakest figure: {metrics['weakest_figure']} with {metrics['min_figure_records']} records")

    except Exception as e:
        _log.warning("Layer 1 partial failure: %s", e)
        checks.append({"metric": "Layer 1 Error", "value": str(e), "threshold": "N/A",
                        "op": "N/A", "passed": False, "detail": f"Unexpected error: {e}"})

    passed = sum(1 for c in checks if c["passed"])
    failed = sum(1 for c in checks if not c["passed"])
    return {"layer": "L1:Statistical", "checks": checks, "passed": passed, "failed": failed}


# ---------------------------------------------------------------------------
# Layer 2: Internal Consistency (grade_layer2)
# ---------------------------------------------------------------------------

def _grade_layer2(records: List[Dict[str, Any]], T: Dict[str, Any]) -> Dict[str, Any]:
    """
    Layer 2: Internal consistency — do the numbers within each record agree?

    Checks:
      - P1 records should have resistance >= p1_threshold (default 0.55)
        OR hold/pressure markers present (label_reason indicates markers)
      - P0 records should have resistance < p1_threshold
        OR failure markers present
      - Label and label_reason should be consistent
      - High confidence + low disambiguation = conflicting signal
    """
    checks: List[Dict[str, Any]] = []
    n = len(records)
    if n == 0:
        return {"layer": "L2:Consistency", "checks": [], "passed": 0, "failed": 0}

    try:
        p1_threshold = T.get("p1_resistance_threshold", 0.55)

        # Check 1: P1 records with resistance below threshold and no marker-based reason
        marker_reasons = {"apy_resistance_held_under_pressure", "high_resistance_hold_marker",
                          "pressure_detected_value_failed", "panel_confirmed_p1"}
        p1_records = [r for r in records if r.get("label") == "P1"]
        if p1_records:
            p1_low_resistance = sum(
                1 for r in p1_records
                if _safe_float(r.get("resistance")) < p1_threshold
                and r.get("label_reason", "") not in marker_reasons
            )
            p1_coherence = 1.0 - (p1_low_resistance / len(p1_records)) if p1_records else 1.0
            _check(checks, "P1 Resistance Coherence", round(p1_coherence, 4), 0.85,
                   detail=f"{p1_low_resistance}/{len(p1_records)} P1 records have resistance < {p1_threshold} without marker justification")

        # Check 2: label_reason matches label
        mismatches = 0
        for r in records:
            label = r.get("label", "")
            reason = r.get("label_reason", "")
            if not reason:
                continue
            if label == "P1" and "p0" in reason.lower() and "pressure" not in reason.lower():
                mismatches += 1
            elif label == "P0" and "held" in reason.lower() and "panel" not in reason.lower():
                mismatches += 1
            elif label == "APY" and "held" in reason.lower() and "pressure" not in reason.lower():
                mismatches += 1
        label_reason_rate = 1.0 - (mismatches / n)
        _check(checks, "Label-Reason Coherence", round(label_reason_rate, 4), 0.95,
               detail=f"{mismatches}/{n} records have label/reason mismatches")

        # Check 3: High confidence + low disambiguation (conflicting signals)
        conflicting = sum(
            1 for r in records
            if _safe_float(r.get("confidence")) >= 0.80
            and _safe_float(r.get("disambiguation_confidence"), 1.0) < 0.50
        )
        conflict_rate = conflicting / n
        _check(checks, "Signal Conflict Rate", round(conflict_rate, 4), 0.05, op="<=",
               detail=f"{conflicting}/{n} records have high confidence but low disambiguation")

    except Exception as e:
        _log.warning("Layer 2 partial failure: %s", e)
        checks.append({"metric": "Layer 2 Error", "value": str(e), "threshold": "N/A",
                        "op": "N/A", "passed": False, "detail": f"Unexpected error: {e}"})

    passed = sum(1 for c in checks if c["passed"])
    failed = sum(1 for c in checks if not c["passed"])
    return {"layer": "L2:Consistency", "checks": checks, "passed": passed, "failed": failed}


# ---------------------------------------------------------------------------
# Layer 3: Stability (grade_layer3)
# ---------------------------------------------------------------------------

def _grade_layer3(records: List[Dict[str, Any]], metrics: Dict[str, Any], T: Dict[str, Any]) -> Dict[str, Any]:
    """
    Layer 3: Stability — is the dataset robust or fragile?

    Checks:
      - Split-half: both random halves should independently pass Layer 1 core checks
      - Leave-one-figure-out: removing any single figure shouldn't drop value coverage
        below threshold
    """
    checks: List[Dict[str, Any]] = []
    n = len(records)
    if n < 10:
        checks.append({"metric": "Stability", "value": "N/A", "threshold": "N/A",
                        "op": "N/A", "passed": True,
                        "detail": f"Skipped: only {n} records (need >= 10 for stability checks)"})
        return {"layer": "L3:Stability", "checks": checks, "passed": 1, "failed": 0}

    try:
        # Check 1: Split-half consistency
        # Use deterministic split (even/odd by index after sorting) — no randomness
        sorted_recs = sorted(records, key=lambda r: (
            r.get("figure", ""), r.get("value_name", ""),
            r.get("text_excerpt", "")[:40], _safe_float(r.get("ts")),
        ))
        half_a = [sorted_recs[i] for i in range(0, n, 2)]
        half_b = [sorted_recs[i] for i in range(1, n, 2)]

        m_a = _compute_metrics(half_a)
        m_b = _compute_metrics(half_b)

        halves_ok = True
        half_issues = []

        for half_name, m_half in [("A", m_a), ("B", m_b)]:
            if "error" in m_half:
                halves_ok = False
                half_issues.append(f"Half {half_name}: error computing metrics")
                continue
            if m_half["value_coverage"] < T["value_coverage_min"]:
                halves_ok = False
                half_issues.append(f"Half {half_name}: value coverage {m_half['value_coverage']} < {T['value_coverage_min']}")
            if m_half["avg_confidence"] < T["avg_confidence_min"]:
                halves_ok = False
                half_issues.append(f"Half {half_name}: avg confidence {m_half['avg_confidence']:.3f} < {T['avg_confidence_min']}")

        _check(checks, "Split-Half Consistency", halves_ok, True, op="==",
               detail="; ".join(half_issues) if half_issues else "Both halves pass core checks independently")

        # Check 2: Leave-one-figure-out
        figure_counts = metrics.get("figure_counts", {})
        if len(figure_counts) >= 2:
            fragile_figures = []
            for fig in figure_counts:
                remaining = [r for r in records if r.get("figure") != fig]
                if not remaining:
                    continue
                m_remaining = _compute_metrics(remaining)
                if "error" in m_remaining:
                    continue
                if m_remaining["value_coverage"] < T["value_coverage_min"]:
                    fragile_figures.append(f"{fig} (coverage drops to {m_remaining['value_coverage']})")

            lofo_ok = len(fragile_figures) == 0
            _check(checks, "Figure Independence", lofo_ok, True, op="==",
                   detail=f"Fragile on: {', '.join(fragile_figures)}" if fragile_figures
                   else "No single figure removal breaks value coverage")
        else:
            _check(checks, "Figure Independence", True, True, op="==",
                   detail="Skipped: only 1 figure in dataset")

    except Exception as e:
        _log.warning("Layer 3 partial failure: %s", e)
        checks.append({"metric": "Layer 3 Error", "value": str(e), "threshold": "N/A",
                        "op": "N/A", "passed": False, "detail": f"Unexpected error: {e}"})

    passed = sum(1 for c in checks if c["passed"])
    failed = sum(1 for c in checks if not c["passed"])
    return {"layer": "L3:Stability", "checks": checks, "passed": passed, "failed": failed}


# ---------------------------------------------------------------------------
# Grade (all three layers)
# ---------------------------------------------------------------------------

def grade_dataset(
    metrics: Dict[str, Any],
    thresholds: Dict[str, Any] | None = None,
    records: List[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    """
    Three-layer quality assessment.

    Layer 1 (Statistical Shape): Does the dataset have the right structure?
    Layer 2 (Internal Consistency): Do individual records agree internally?
    Layer 3 (Stability): Is the dataset robust to splits and figure removal?

    All three layers must pass for CERTIFIED. Any layer failure = FAILED.
    Any layer exception = degraded result, not a crash.
    """
    if "error" in metrics:
        return {"grade": "ERROR", "reason": metrics["error"], "checks": [],
                "layers": [], "passed": 0, "failed": 1}

    T = {**DEFAULT_THRESHOLDS, **(thresholds or {})}

    # Run all three layers — each is fault-isolated
    l1 = _grade_layer1(metrics, T)
    l2 = _grade_layer2(records or [], T) if records else {
        "layer": "L2:Consistency", "checks": [], "passed": 0, "failed": 0
    }
    l3 = _grade_layer3(records or [], metrics, T) if records else {
        "layer": "L3:Stability", "checks": [], "passed": 0, "failed": 0
    }

    all_checks = l1["checks"] + l2["checks"] + l3["checks"]
    total_passed = l1["passed"] + l2["passed"] + l3["passed"]
    total_failed = l1["failed"] + l2["failed"] + l3["failed"]

    # All three layers must pass
    l1_ok = l1["failed"] == 0
    l2_ok = l2["failed"] == 0
    l3_ok = l3["failed"] == 0
    grade = "CERTIFIED" if (l1_ok and l2_ok and l3_ok) else "FAILED"

    return {
        "grade": grade,
        "passed": total_passed,
        "failed": total_failed,
        "total_checks": len(all_checks),
        "checks": all_checks,
        "layers": [
            {"name": l1["layer"], "passed": l1["passed"], "failed": l1["failed"],
             "verdict": "PASS" if l1_ok else "FAIL"},
            {"name": l2["layer"], "passed": l2["passed"], "failed": l2["failed"],
             "verdict": "PASS" if l2_ok else "FAIL"},
            {"name": l3["layer"], "passed": l3["passed"], "failed": l3["failed"],
             "verdict": "PASS" if l3_ok else "FAIL"},
        ],
        "record_count": metrics["record_count"],
        "reproducibility_hash": metrics["reproducibility_hash"],
    }


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def print_scorecard(result: Dict[str, Any], metrics: Dict[str, Any], warnings: List[str] | None = None) -> None:
    """Print a human-readable quality scorecard."""
    n = result.get("record_count", 0)

    print()
    print("=" * 68)
    print("  DATASET QUALITY SCORECARD — THREE-LAYER ASSESSMENT")
    print("=" * 68)
    print(f"  Records evaluated:  {n}")
    print(f"  Figures:            {len(metrics.get('figure_counts', {}))}")
    print(f"  Reproducibility:    {result.get('reproducibility_hash', 'N/A')}")

    if warnings:
        print(f"  Warnings:           {len(warnings)}")
        for w in warnings[:5]:
            print(f"    ! {w}")
        if len(warnings) > 5:
            print(f"    ... and {len(warnings) - 5} more")

    # Layer verdicts summary
    layers = result.get("layers", [])
    if layers:
        print()
        for layer in layers:
            v = layer["verdict"]
            sym = "+" if v == "PASS" else "X"
            print(f"  [{sym}] {layer['name']:<24} {v}  ({layer['passed']} passed, {layer['failed']} failed)")

    print("-" * 68)

    # Group checks by layer
    layer_names = [l["name"] for l in layers] if layers else [""]
    check_idx = 0
    for layer in layers:
        layer_check_count = layer["passed"] + layer["failed"]
        if layer_check_count == 0:
            continue
        print(f"\n  --- {layer['name']} ---")
        layer_checks = result["checks"][check_idx:check_idx + layer_check_count]
        check_idx += layer_check_count
        for c in layer_checks:
            status = "PASS" if c["passed"] else "FAIL"
            symbol = " + " if c["passed"] else " X "
            op_str = c["op"]
            val_str = str(c["value"])
            thr_str = str(c["threshold"])
            print(f"  {symbol} {c['metric']:<24}  {val_str:>8}  {op_str:>2} {thr_str:>6}  [{status}]")
            if c["detail"]:
                print(f"       {c['detail']}")

    print()
    print("-" * 68)

    grade = result["grade"]
    if grade == "CERTIFIED":
        print(f"  GRADE: CERTIFIED  ({result['passed']}/{result['total_checks']} checks passed across 3 layers)")
        print(f"  This dataset meets quality standards for export and publication.")
    else:
        print(f"  GRADE: FAILED  ({result['passed']}/{result['total_checks']} passed, {result['failed']} failed)")
        failed_layers = [l["name"] for l in layers if l["verdict"] == "FAIL"]
        if failed_layers:
            print(f"  Failed layers: {', '.join(failed_layers)}")

    print("=" * 68)

    # Per-figure summary
    if "by_figure" in metrics and metrics["by_figure"]:
        print(f"\n  Per-figure breakdown:")
        for fig, counts in metrics["by_figure"].items():
            total = counts.get("_total", 0)
            p1 = counts.get("P1", 0)
            p0 = counts.get("P0", 0)
            apy = counts.get("APY", 0)
            amb = counts.get("AMBIGUOUS", 0)
            print(f"    {fig:<22}  total={total:>4}  P1={p1:>3}  P0={p0:>3}  APY={apy:>3}  AMB={amb:>3}")

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Dataset Quality Scorecard — deterministic quality assessment for exported JSONL.",
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to a JSONL file or directory containing ric_*.jsonl exports.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON instead of formatted text.",
    )
    parser.add_argument(
        "--thresholds",
        type=str,
        default=None,
        help="Optional JSON file with custom thresholds (overrides defaults).",
    )
    args = parser.parse_args()

    # Custom thresholds (never mutates the global default)
    custom_thresholds = None
    if args.thresholds:
        try:
            custom_thresholds = json.loads(Path(args.thresholds).read_text())
        except Exception as e:
            print(f"Error loading thresholds: {e}", file=sys.stderr)
            sys.exit(2)

    records, warnings = _load_records(args.input)
    if not records:
        if args.json:
            print(json.dumps({"grade": "ERROR", "reason": "No records found at " + args.input,
                              "warnings": warnings}))
        else:
            print(f"\n  ERROR: No JSONL records found at {args.input}")
            for w in warnings:
                print(f"    ! {w}")
            print()
        sys.exit(2)

    metrics = _compute_metrics(records)
    result = grade_dataset(metrics, thresholds=custom_thresholds, records=records)

    if args.json:
        output = {**result, "metrics": metrics, "warnings": warnings}
        print(json.dumps(output, indent=2, ensure_ascii=False))
    else:
        print_scorecard(result, metrics, warnings)

    sys.exit(0 if result["grade"] == "CERTIFIED" else 1)


if __name__ == "__main__":
    main()
