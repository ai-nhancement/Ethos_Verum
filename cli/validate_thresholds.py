#!/usr/bin/env python3
"""
cli/validate_thresholds.py

Threshold validation framework — compute P/R/F1 for P1/P0/APY classification
across a range of p1_threshold and p0_threshold settings against a gold-standard
annotation file.

Usage:
    # Sweep thresholds automatically
    python -m cli.validate_thresholds --gold data/gold_annotations.jsonl

    # Evaluate a single threshold pair
    python -m cli.validate_thresholds --gold data/gold_annotations.jsonl \\
        --p1 0.55 --p0 0.35

    # Write sweep results to CSV
    python -m cli.validate_thresholds --gold data/gold_annotations.jsonl \\
        --output output/threshold_sweep.csv

Gold annotation format (one JSON object per line):
    {
      "observation_id": "<source_obs_id from export>",
      "label": "P1" | "P0" | "APY" | "AMBIGUOUS"
    }

    OR, if matching by text excerpt:
    {
      "text_excerpt": "<passage text>",
      "value_name":   "<value>",
      "label": "P1"
    }

The validator re-classifies all observations from values.db using
build_training_records(), then computes precision/recall/F1 for each
threshold pair against the gold labels.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
_log = logging.getLogger(__name__)

_VALUES_DB = str(_ROOT / "data" / "values.db")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _precision_recall_f1(
    predicted: List[str],
    gold: List[str],
    positive_labels: Tuple[str, ...] = ("P1",),
) -> Dict[str, float]:
    """
    Binary P/R/F1 treating positive_labels as the positive class.
    Macro-averages if multiple positive labels are given.
    """
    tp = sum(1 for p, g in zip(predicted, gold) if p in positive_labels and g in positive_labels)
    fp = sum(1 for p, g in zip(predicted, gold) if p in positive_labels and g not in positive_labels)
    fn = sum(1 for p, g in zip(predicted, gold) if p not in positive_labels and g in positive_labels)
    tn = sum(1 for p, g in zip(predicted, gold) if p not in positive_labels and g not in positive_labels)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    accuracy  = (tp + tn) / len(predicted) if predicted else 0.0

    return {
        "precision": round(precision, 4),
        "recall":    round(recall, 4),
        "f1":        round(f1, 4),
        "accuracy":  round(accuracy, 4),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "n_positive_gold": tp + fn,
        "n_positive_pred": tp + fp,
    }


def _all_class_metrics(
    predicted: List[str],
    gold: List[str],
    labels: Tuple[str, ...] = ("P1", "P0", "APY"),
) -> Dict[str, Dict]:
    """Per-class P/R/F1 for each label."""
    return {lbl: _precision_recall_f1(predicted, gold, positive_labels=(lbl,))
            for lbl in labels}


# ---------------------------------------------------------------------------
# Gold loader
# ---------------------------------------------------------------------------

def load_gold(gold_path: str) -> Dict[str, str]:
    """
    Load gold annotations from a JSONL file.

    Returns a dict keyed by observation_id (preferred) or
    "<text_excerpt[:80]>|<value_name>" (fallback).
    """
    gold: Dict[str, str] = {}
    path = Path(gold_path)
    if not path.exists():
        _log.error("Gold file not found: %s", gold_path)
        return gold

    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                _log.warning("Line %d: JSON parse error: %s", lineno, e)
                continue

            label = obj.get("label", "").upper()
            if label not in ("P1", "P0", "APY", "AMBIGUOUS"):
                _log.warning("Line %d: unknown label %r — skipping", lineno, label)
                continue

            if "observation_id" in obj:
                gold[str(obj["observation_id"])] = label
            elif "text_excerpt" in obj and "value_name" in obj:
                key = f"{obj['text_excerpt'][:80]}|{obj['value_name']}"
                gold[key] = label
            else:
                _log.warning("Line %d: no 'observation_id' or 'text_excerpt'+'value_name' — skipping", lineno)

    _log.info("Loaded %d gold annotations from %s", len(gold), gold_path)
    return gold


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_thresholds(
    gold: Dict[str, str],
    db_path: str,
    p1_threshold: float,
    p0_threshold: float,
) -> Optional[Dict]:
    """
    Classify all observations at the given thresholds and compare to gold.
    Returns metrics dict or None if no matches found.
    """
    from cli.export import _read_figure_observations, _load_apy_context, build_training_records

    observations = _read_figure_observations(db_path, figure_filter=None)
    if not observations:
        return None

    apy_ctx = _load_apy_context(db_path)
    records = build_training_records(
        observations, p1_threshold, p0_threshold,
        min_observations=1, min_consistency=0.0,
        apy_context=apy_ctx,
    )

    # Match records to gold by observation_id or text/value key
    matched_pred: List[str] = []
    matched_gold: List[str] = []
    unmatched = 0

    for rec in records:
        # Try observation_id first
        gold_label = gold.get(rec.get("source_obs_id", ""))
        if gold_label is None:
            # Fallback: text_excerpt + value_name key
            key = f"{rec['text_excerpt'][:80]}|{rec['value_name']}"
            gold_label = gold.get(key)
        if gold_label is None:
            unmatched += 1
            continue
        matched_pred.append(rec["label"])
        matched_gold.append(gold_label)

    if not matched_pred:
        return None

    per_class = _all_class_metrics(matched_pred, matched_gold)
    p1_metrics = per_class["P1"]

    return {
        "p1_threshold":   round(p1_threshold, 3),
        "p0_threshold":   round(p0_threshold, 3),
        "n_matched":      len(matched_pred),
        "n_unmatched":    unmatched,
        "p1_precision":   p1_metrics["precision"],
        "p1_recall":      p1_metrics["recall"],
        "p1_f1":          p1_metrics["f1"],
        "p0_precision":   per_class["P0"]["precision"],
        "p0_recall":      per_class["P0"]["recall"],
        "p0_f1":          per_class["P0"]["f1"],
        "apy_precision":  per_class["APY"]["precision"],
        "apy_recall":     per_class["APY"]["recall"],
        "apy_f1":         per_class["APY"]["f1"],
        "accuracy":       p1_metrics["accuracy"],
        "tp": p1_metrics["tp"], "fp": p1_metrics["fp"],
        "fn": p1_metrics["fn"], "tn": p1_metrics["tn"],
    }


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

_P1_SWEEP = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
_P0_SWEEP = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45]


def sweep(gold: Dict[str, str], db_path: str) -> List[Dict]:
    """
    Evaluate all (p1_threshold, p0_threshold) combinations in the sweep grid.
    Only includes pairs where p0_threshold < p1_threshold.
    Returns rows sorted by P1 F1 descending.
    """
    rows = []
    total = sum(1 for p1 in _P1_SWEEP for p0 in _P0_SWEEP if p0 < p1)
    _log.info("Sweeping %d threshold combinations...", total)

    for p1 in _P1_SWEEP:
        for p0 in _P0_SWEEP:
            if p0 >= p1:
                continue
            result = evaluate_thresholds(gold, db_path, p1, p0)
            if result is not None:
                rows.append(result)

    rows.sort(key=lambda r: -r["p1_f1"])
    return rows


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

_CSV_FIELDS = [
    "p1_threshold", "p0_threshold", "n_matched",
    "p1_precision", "p1_recall", "p1_f1",
    "p0_precision", "p0_recall", "p0_f1",
    "apy_precision", "apy_recall", "apy_f1",
    "accuracy", "tp", "fp", "fn", "tn",
]


def _print_table(rows: List[Dict]) -> None:
    if not rows:
        print("No results.")
        return
    hdr = (f"{'p1_thr':>6}  {'p0_thr':>6}  {'matched':>7}  "
           f"{'P1_pre':>6}  {'P1_rec':>6}  {'P1_F1':>6}  "
           f"{'P0_F1':>6}  {'APY_F1':>6}  {'acc':>5}")
    print("\n" + "=" * len(hdr))
    print("  THRESHOLD VALIDATION RESULTS  (sorted by P1 F1)")
    print("=" * len(hdr))
    print(hdr)
    print("-" * len(hdr))
    for r in rows[:20]:
        print(f"  {r['p1_threshold']:>4.2f}   {r['p0_threshold']:>4.2f}  "
              f"{r['n_matched']:>7}  "
              f"{r['p1_precision']:>6.4f}  {r['p1_recall']:>6.4f}  {r['p1_f1']:>6.4f}  "
              f"{r['p0_f1']:>6.4f}  {r['apy_f1']:>6.4f}  "
              f"{r['accuracy']:>5.4f}")
    if rows:
        best = rows[0]
        print(f"\n  Best P1 F1: p1_threshold={best['p1_threshold']}  "
              f"p0_threshold={best['p0_threshold']}  F1={best['p1_f1']:.4f}")
    print("=" * len(hdr) + "\n")


def _write_csv(path: str, rows: List[Dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    _log.info("Sweep results written to %s", path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate classification thresholds against gold annotations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--gold", required=True,
                        help="Path to gold annotation JSONL file")
    parser.add_argument("--p1", type=float, default=None,
                        help="P1 threshold to evaluate (if omitted, runs full sweep)")
    parser.add_argument("--p0", type=float, default=None,
                        help="P0 threshold to evaluate (if omitted, runs full sweep)")
    parser.add_argument("--output", default=None,
                        help="Write sweep CSV to this path")
    parser.add_argument("--db", default=_VALUES_DB,
                        help=f"Path to values.db (default: {_VALUES_DB})")
    args = parser.parse_args()

    gold = load_gold(args.gold)
    if not gold:
        _log.error("No gold annotations loaded — check file format")
        return 1

    if args.p1 is not None and args.p0 is not None:
        # Single-pair evaluation
        result = evaluate_thresholds(gold, args.db, args.p1, args.p0)
        if result is None:
            _log.error("No matching observations found — check that db has ingested figures")
            return 1
        _print_table([result])
        if args.output:
            _write_csv(args.output, [result])
    else:
        # Full sweep
        rows = sweep(gold, args.db)
        if not rows:
            _log.error("No results — check db path and gold file")
            return 1
        _print_table(rows)
        if args.output:
            _write_csv(args.output, rows)

    return 0


if __name__ == "__main__":
    sys.exit(main())
