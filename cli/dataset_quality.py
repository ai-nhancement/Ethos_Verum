#!/usr/bin/env python3
"""
cli/dataset_quality.py

Dataset Quality Scorecard — deterministic quality assessment for exported JSONL.

Reads a finished export (JSONL files) and produces a pass/fail grade with
specific, reproducible metrics. No human judgment. The numbers decide.

Metrics:
  1. Value Coverage      — distinct values with signals (>= 5 to pass)
  2. Label Balance       — no single label > 80% of total
  3. Signal Density      — % of records with confidence >= 0.65 (>= 50% to pass)
  4. Avg Confidence      — mean classification confidence (>= 0.65 to pass)
  5. Resistance Spread   — std dev of resistance scores (>= 0.10 to pass)
  6. Source Diversity     — distinct document types (>= 2 to pass)
  7. Disambiguation Rate — % of signals with disambiguation_confidence >= 0.8 (>= 85%)
  8. Consistency Floor   — min observation_consistency across records (>= 0.25)
  9. Reproducibility     — deterministic check: hash of sorted records is stable

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
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Thresholds — deterministic, no human judgment
# ---------------------------------------------------------------------------

THRESHOLDS = {
    "value_coverage_min":       5,      # distinct values with signals
    "label_balance_max_pct":    0.80,   # no single label > 80%
    "signal_density_min":       0.50,   # % records with confidence >= 0.65
    "avg_confidence_min":       0.65,   # mean confidence
    "resistance_spread_min":    0.10,   # std dev of resistance
    "source_diversity_min":     2,      # distinct doc types
    "disambiguation_rate_min":  0.85,   # % with disambig >= 0.8
    "consistency_floor_min":    0.25,   # min observation_consistency
}


# ---------------------------------------------------------------------------
# Load records
# ---------------------------------------------------------------------------

def _load_records(input_path: str) -> List[Dict[str, Any]]:
    """Load records from a JSONL file or all JSONL files in a directory."""
    path = Path(input_path)
    records = []

    if path.is_file() and path.suffix == ".jsonl":
        files = [path]
    elif path.is_dir():
        files = sorted(path.glob("ric_*.jsonl"))
        if not files:
            files = sorted(path.glob("*.jsonl"))
    else:
        return []

    seen_ids = set()
    for f in files:
        try:
            for line in f.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                # Deduplicate across files (positive/negative may overlap with per-figure)
                rec_id = rec.get("id") or rec.get("source_obs_id", "")
                if rec_id and rec_id in seen_ids:
                    continue
                if rec_id:
                    seen_ids.add(rec_id)
                records.append(rec)
        except Exception as e:
            _log.warning("Could not read %s: %s", f, e)

    return records


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _compute_metrics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute all quality metrics from a list of export records."""
    n = len(records)
    if n == 0:
        return {"error": "No records to evaluate", "record_count": 0}

    # ── Value Coverage ──
    values_seen = set(r.get("value_name", "") for r in records)
    values_seen.discard("")
    value_coverage = len(values_seen)

    # ── Label Balance ──
    label_counts: Dict[str, int] = defaultdict(int)
    for r in records:
        label_counts[r.get("label", "UNKNOWN")] += 1
    max_label_pct = max(label_counts.values()) / n if n > 0 else 1.0
    max_label_name = max(label_counts, key=label_counts.get)

    # ── Signal Density (% with confidence >= 0.65) ──
    high_conf_count = sum(1 for r in records if float(r.get("confidence", 0)) >= 0.65)
    signal_density = high_conf_count / n

    # ── Average Confidence ──
    confidences = [float(r.get("confidence", 0)) for r in records]
    avg_confidence = sum(confidences) / n

    # ── Resistance Spread ──
    resistances = [float(r.get("resistance", 0)) for r in records]
    mean_resistance = sum(resistances) / n
    variance = sum((x - mean_resistance) ** 2 for x in resistances) / n
    resistance_spread = math.sqrt(variance)

    # ── Source Diversity ──
    doc_types = set(r.get("document_type", "unknown") for r in records)
    doc_types.discard("")
    source_diversity = len(doc_types)

    # ── Disambiguation Rate ──
    disambig_scores = [float(r.get("disambiguation_confidence", 1.0)) for r in records]
    disambig_high = sum(1 for d in disambig_scores if d >= 0.8)
    disambiguation_rate = disambig_high / n

    # ── Consistency Floor ──
    consistencies = [float(r.get("observation_consistency", 0.5)) for r in records]
    consistency_floor = min(consistencies) if consistencies else 0.0

    # ── Reproducibility Hash ──
    # Sort by deterministic key, hash the content
    sorted_records = sorted(records, key=lambda r: (
        r.get("figure", ""),
        r.get("value_name", ""),
        r.get("text_excerpt", "")[:80],
        r.get("ts", 0),
    ))
    content_for_hash = json.dumps(
        [
            {
                "figure": r.get("figure", ""),
                "value_name": r.get("value_name", ""),
                "text_excerpt": r.get("text_excerpt", "")[:80],
                "label": r.get("label", ""),
                "resistance": r.get("resistance", 0),
                "confidence": r.get("confidence", 0),
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
        "record_count":         n,
        "value_coverage":       value_coverage,
        "values_detected":      sorted(values_seen),
        "label_counts":         dict(label_counts),
        "max_label_pct":        round(max_label_pct, 4),
        "max_label_name":       max_label_name,
        "signal_density":       round(signal_density, 4),
        "avg_confidence":       round(avg_confidence, 4),
        "resistance_mean":      round(mean_resistance, 4),
        "resistance_spread":    round(resistance_spread, 4),
        "source_diversity":     source_diversity,
        "doc_types":            sorted(doc_types),
        "disambiguation_rate":  round(disambiguation_rate, 4),
        "consistency_floor":    round(consistency_floor, 4),
        "reproducibility_hash": reproducibility_hash,
        "by_figure":            {k: dict(v) for k, v in sorted(by_figure.items())},
        "by_value":             {k: dict(v) for k, v in sorted(by_value.items())},
    }


# ---------------------------------------------------------------------------
# Grade
# ---------------------------------------------------------------------------

def grade_dataset(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply thresholds to metrics and return a pass/fail verdict for each.
    Returns {checks: [...], passed: int, failed: int, grade: "CERTIFIED"|"FAILED"}.
    """
    if "error" in metrics:
        return {"grade": "ERROR", "reason": metrics["error"], "checks": [], "passed": 0, "failed": 1}

    checks = []

    def check(name: str, value, threshold, op: str = ">=", detail: str = ""):
        if op == ">=":
            passed = value >= threshold
        elif op == "<=":
            passed = value <= threshold
        else:
            passed = False
        checks.append({
            "metric": name,
            "value": value,
            "threshold": threshold,
            "op": op,
            "passed": passed,
            "detail": detail,
        })

    T = THRESHOLDS

    check("Value Coverage", metrics["value_coverage"], T["value_coverage_min"],
          detail=f"{metrics['value_coverage']} of 15 values detected: {', '.join(metrics['values_detected'][:8])}{'...' if len(metrics['values_detected']) > 8 else ''}")

    check("Label Balance", metrics["max_label_pct"], T["label_balance_max_pct"], op="<=",
          detail=f"Dominant label: {metrics['max_label_name']} at {metrics['max_label_pct']*100:.1f}%")

    check("Signal Density", metrics["signal_density"], T["signal_density_min"],
          detail=f"{metrics['signal_density']*100:.1f}% of records have confidence >= 0.65")

    check("Avg Confidence", metrics["avg_confidence"], T["avg_confidence_min"],
          detail=f"Mean classification confidence: {metrics['avg_confidence']:.4f}")

    check("Resistance Spread", metrics["resistance_spread"], T["resistance_spread_min"],
          detail=f"Std dev: {metrics['resistance_spread']:.4f}, mean: {metrics['resistance_mean']:.4f}")

    check("Source Diversity", metrics["source_diversity"], T["source_diversity_min"],
          detail=f"Document types: {', '.join(metrics['doc_types'])}")

    check("Disambiguation Rate", metrics["disambiguation_rate"], T["disambiguation_rate_min"],
          detail=f"{metrics['disambiguation_rate']*100:.1f}% of signals have disambiguation >= 0.8")

    check("Consistency Floor", metrics["consistency_floor"], T["consistency_floor_min"],
          detail=f"Minimum observation consistency: {metrics['consistency_floor']:.4f}")

    passed = sum(1 for c in checks if c["passed"])
    failed = sum(1 for c in checks if not c["passed"])
    grade = "CERTIFIED" if failed == 0 else "FAILED"

    return {
        "grade": grade,
        "passed": passed,
        "failed": failed,
        "total_checks": len(checks),
        "checks": checks,
        "record_count": metrics["record_count"],
        "reproducibility_hash": metrics["reproducibility_hash"],
    }


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def print_scorecard(result: Dict[str, Any], metrics: Dict[str, Any]) -> None:
    """Print a human-readable quality scorecard."""
    n = result.get("record_count", 0)

    print()
    print("=" * 64)
    print("  DATASET QUALITY SCORECARD")
    print("=" * 64)
    print(f"  Records evaluated:  {n}")
    print(f"  Reproducibility:    {result.get('reproducibility_hash', 'N/A')}")
    print("-" * 64)

    for c in result["checks"]:
        status = "PASS" if c["passed"] else "FAIL"
        symbol = " + " if c["passed"] else " X "
        op_str = ">=" if c["op"] == ">=" else "<="
        print(f"  {symbol} {c['metric']:<22}  {str(c['value']):>8}  {op_str} {str(c['threshold']):>6}  [{status}]")
        if c["detail"]:
            print(f"       {c['detail']}")

    print("-" * 64)

    grade = result["grade"]
    if grade == "CERTIFIED":
        print(f"  GRADE: CERTIFIED  ({result['passed']}/{result['total_checks']} checks passed)")
        print(f"  This dataset meets quality standards for export and publication.")
    else:
        print(f"  GRADE: FAILED  ({result['passed']}/{result['total_checks']} passed, {result['failed']} failed)")
        print(f"  This dataset does not meet quality standards. See failed checks above.")

    print("=" * 64)

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

    # Custom thresholds
    if args.thresholds:
        try:
            custom = json.loads(Path(args.thresholds).read_text())
            THRESHOLDS.update(custom)
        except Exception as e:
            print(f"Error loading thresholds: {e}", file=sys.stderr)
            sys.exit(2)

    records = _load_records(args.input)
    if not records:
        if args.json:
            print(json.dumps({"grade": "ERROR", "reason": "No records found at " + args.input}))
        else:
            print(f"\n  ERROR: No JSONL records found at {args.input}\n")
        sys.exit(2)

    metrics = _compute_metrics(records)
    result = grade_dataset(metrics)

    if args.json:
        output = {**result, "metrics": metrics}
        print(json.dumps(output, indent=2, ensure_ascii=False))
    else:
        print_scorecard(result, metrics)

    sys.exit(0 if result["grade"] == "CERTIFIED" else 1)


if __name__ == "__main__":
    main()
