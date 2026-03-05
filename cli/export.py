#!/usr/bin/env python3
"""
cli/export.py

Ethos — export RIC training data from ingested historical figure observations.

Reads value_observations from data/values.db for all ingested historical figures
(session_id LIKE 'figure:%') and classifies each observation as:

  P1   — value demonstrated under meaningful resistance (held under pressure)
  P0   — value failed or corrupted (yielded, rationalized, compromised)
  APY  — Answer-Pressure Yield (pressure to abandon detected; value may or may not have held)

The resistance score is the pressure measurement (computed during ingestion).
The text markers are the pressure event detectors (already in text_excerpt).
The document_type_bonus is authenticity weighting (already in figure_sources).

Output (default: output/ric/):
  ric_historical_positive.jsonl  — all P1 examples across figures
  ric_historical_negative.jsonl  — all P0 + APY examples across figures
  ric_figure_<name>.jsonl        — per-figure, all labels
  ric_historical_report.json     — summary statistics

Usage:
  python -m cli.export [options]

Options:
  --figure NAME        Export only this figure (e.g. gandhi, lincoln)
  --p1-threshold N     Min resistance for P1 classification (default 0.55)
  --p0-threshold N     Max resistance for P0 when no hold markers present (default 0.35)
  --min-observations N Min observations per value per figure to include (default 1)
  --output-dir PATH    Output directory (default output/ric/)
  --dry-run            Print classification stats only — no files written
  --no-ambiguous       Exclude AMBIGUOUS observations from per-figure files
  --min-consistency N  Min registry consistency score [0.0-1.0] to include (default 0.0)
  --db PATH            Path to values.db (default data/values.db)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
import time
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
from core.config import is_tension_pair

_VALUES_DB = str(_ROOT / "data" / "values.db")
_DEFAULT_OUTPUT_DIR = str(_ROOT / "output" / "ric")

# ---------------------------------------------------------------------------
# Document-type training weights
# Higher weight = more authentic source signal
# ---------------------------------------------------------------------------

_DOC_TYPE_WEIGHT = {
    "action":  1.5,
    "journal": 1.4,
    "letter":  1.2,
    "speech":  0.8,
    "unknown": 1.0,
}

# ---------------------------------------------------------------------------
# Text marker sets
# ---------------------------------------------------------------------------

_FAILURE_RE = re.compile(
    r'\b(gave in|gave up|yielded|surrendered|backed down|stepped back|'
    r'compromised my|failed to|could not maintain|could not keep|'
    r'i betrayed|i regret|i was wrong to|i should not have|'
    r'i lied|i deceived|i misled|i hid the|i concealed|'
    r'i was weak|cowardice|fear stopped|i gave up|'
    r'i rationalized|made excuses|i caved|i relented|'
    r'i told them what they wanted|i said what they wanted|'
    r'i pretended|i falsely|i fabricated)\b',
    re.IGNORECASE,
)

_HOLD_RE = re.compile(
    r'\b(despite|even though|nevertheless|nonetheless|still|'
    r'continued to|stood firm|refused to give|persevered|'
    r'maintained|held to|kept to|stayed true|did not yield|'
    r'would not surrender|would not give|refused to back|'
    r'i admitted|i confessed|i acknowledged my|i accepted responsibility|'
    r'i owned|pushed through|carried on|pressed on|went ahead)\b',
    re.IGNORECASE,
)

from core.apy_patterns import APY_PRESSURE_RE as _APY_PRESSURE_RE


def _find_markers(text: str, pattern: re.Pattern) -> List[str]:
    return [m.group(0).lower() for m in pattern.finditer(text)]


# ---------------------------------------------------------------------------
# Labeling
# ---------------------------------------------------------------------------

def classify_observation(
    text_excerpt: str,
    resistance: float,
    p1_threshold: float,
    p0_threshold: float,
) -> Tuple[str, str, float]:
    """
    Classify one value_observation record.
    Returns (label, reason, confidence):
      label:      "P1" | "P0" | "APY" | "AMBIGUOUS"
      reason:     human-readable explanation
      confidence: [0.0, 1.0]
    """
    failure_hits = _find_markers(text_excerpt, _FAILURE_RE)
    hold_hits    = _find_markers(text_excerpt, _HOLD_RE)
    apy_hits     = _find_markers(text_excerpt, _APY_PRESSURE_RE)

    if apy_hits:
        if failure_hits:
            return "APY", "pressure_detected_value_failed", 0.95
        else:
            return "P1", "apy_resistance_held_under_pressure", 0.95

    if failure_hits:
        return "P0", "failure_markers_present", 0.85

    if resistance >= p1_threshold and hold_hits:
        return "P1", "high_resistance_hold_marker", 0.90

    if resistance >= p1_threshold:
        return "P1", "high_resistance_held", 0.75

    if resistance < p0_threshold:
        return "P0", "low_resistance_no_hold", 0.55

    return "AMBIGUOUS", "insufficient_signal", 0.40


# ---------------------------------------------------------------------------
# Read from values.db
# ---------------------------------------------------------------------------

def _read_figure_observations(
    db_path: str,
    figure_filter: Optional[str] = None,
) -> List[Dict[str, Any]]:
    if not Path(db_path).exists():
        print(f"[export] ERROR: values.db not found at {db_path}")
        print("[export]  Run: python -m cli.ingest --figure <name> --file <path> --doc-type <type>")
        return []

    conn = sqlite3.connect(db_path, timeout=10.0)
    conn.row_factory = sqlite3.Row
    try:
        if figure_filter:
            session_id_filter = f"figure:{figure_filter.lower().strip()}"
            rows = conn.execute(
                """SELECT
                       vo.id, vo.session_id, vo.record_id, vo.ts,
                       vo.value_name, vo.text_excerpt,
                       vo.significance, vo.resistance,
                       COALESCE(vo.disambiguation_confidence, 1.0) AS disambiguation_confidence,
                       COALESCE(fs.figure_name, SUBSTR(vo.session_id, 8)) AS figure_name,
                       COALESCE(fs.document_type, 'unknown')              AS document_type,
                       COALESCE(vr.consistency, 0.5)                      AS observation_consistency
                   FROM value_observations vo
                   LEFT JOIN figure_sources fs ON fs.session_id = vo.session_id
                   LEFT JOIN value_registry vr
                       ON vr.session_id = vo.session_id AND vr.value_name = vo.value_name
                   WHERE vo.session_id = ?
                   ORDER BY vo.session_id, vo.ts""",
                (session_id_filter,),
            ).fetchall()
        else:
            rows = conn.execute(
                """SELECT
                       vo.id, vo.session_id, vo.record_id, vo.ts,
                       vo.value_name, vo.text_excerpt,
                       vo.significance, vo.resistance,
                       COALESCE(vo.disambiguation_confidence, 1.0) AS disambiguation_confidence,
                       COALESCE(fs.figure_name, SUBSTR(vo.session_id, 8)) AS figure_name,
                       COALESCE(fs.document_type, 'unknown')              AS document_type,
                       COALESCE(vr.consistency, 0.5)                      AS observation_consistency
                   FROM value_observations vo
                   LEFT JOIN figure_sources fs ON fs.session_id = vo.session_id
                   LEFT JOIN value_registry vr
                       ON vr.session_id = vo.session_id AND vr.value_name = vo.value_name
                   WHERE vo.session_id LIKE 'figure:%'
                   ORDER BY vo.session_id, vo.ts""",
            ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# APY context loader (cross-passage pressure window)
# ---------------------------------------------------------------------------

def _load_apy_context(db_path: str) -> Dict[str, List[Dict]]:
    """
    Load all apy_context rows grouped by session_id.
    Returns {session_id: [{record_id, ts, passage_idx, markers}, ...]}.
    """
    result: Dict[str, List[Dict]] = defaultdict(list)
    if not Path(db_path).exists():
        return result
    try:
        conn = sqlite3.connect(db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                "SELECT session_id, record_id, ts, passage_idx, markers FROM apy_context"
            ).fetchall()
            for r in rows:
                result[r["session_id"]].append(dict(r))
        except Exception:
            pass
        finally:
            conn.close()
    except Exception:
        pass
    return result


# ---------------------------------------------------------------------------
# Co-occurrence matrix
# ---------------------------------------------------------------------------

def _compute_cooccurrence(
    records: List[Dict[str, Any]],
) -> Dict[str, Dict]:
    """
    Build a co-occurrence matrix for all unordered value pairs (C(15,2) = 105 max).

    Groups records by (session_id, record_id) to find passages with 2+ values.
    For each pair counts: both_detected, both_p1, mixed (one P1/one not), both_p0.
    Returns {"V1||V2": {"both_detected": n, "both_p1": n, "mixed": n, "both_p0": n}}.
    """
    from itertools import combinations

    passages: Dict[str, List[Dict]] = defaultdict(list)
    for r in records:
        key = r["session_id"] + "||" + r["record_id"]
        passages[key].append(r)

    matrix: Dict[str, Dict] = {}
    for recs in passages.values():
        if len(recs) < 2:
            continue
        for r1, r2 in combinations(recs, 2):
            v1, v2 = sorted([r1["value_name"], r2["value_name"]])
            pair_key = f"{v1}||{v2}"
            if pair_key not in matrix:
                matrix[pair_key] = {"both_detected": 0, "both_p1": 0, "mixed": 0, "both_p0": 0}
            entry = matrix[pair_key]
            entry["both_detected"] += 1
            r1_pos = (r1["label"] == "P1")
            r2_pos = (r2["label"] == "P1")
            if r1_pos and r2_pos:
                entry["both_p1"] += 1
            elif not r1_pos and not r2_pos:
                entry["both_p0"] += 1
            else:
                entry["mixed"] += 1

    return dict(sorted(matrix.items(), key=lambda kv: -kv[1]["both_detected"]))


# ---------------------------------------------------------------------------
# Value tension detection
# ---------------------------------------------------------------------------

def _detect_tensions(
    records: List[Dict[str, Any]],
    db_path: str,
) -> List[Dict[str, Any]]:
    """
    Detect value tension events: two tension-pair values in the same passage
    where one is P1 and the other is P0/APY.

    Persists events to value_tension table. Returns list for JSONL export.
    Tension events carry 1.5× training weight per spec.
    """
    from itertools import combinations
    import sqlite3 as _sql3
    import uuid as _uuid

    passages: Dict[str, List[Dict]] = defaultdict(list)
    for r in records:
        key = r["session_id"] + "||" + r["record_id"]
        passages[key].append(r)

    events: List[Dict[str, Any]] = []

    try:
        conn = _sql3.connect(db_path, timeout=10.0)
        conn.execute("PRAGMA journal_mode=WAL")
        for recs in passages.values():
            if len(recs) < 2:
                continue
            for r1, r2 in combinations(recs, 2):
                if not is_tension_pair(r1["value_name"], r2["value_name"]):
                    continue
                r1_pos = (r1["label"] == "P1")
                r2_pos = (r2["label"] == "P1")
                if r1_pos == r2_pos:
                    continue  # both same direction
                held   = r1 if r1_pos else r2
                failed = r2 if r1_pos else r1
                uid = str(_uuid.uuid4())
                event = {
                    "id":             uid,
                    "session_id":     held["session_id"],
                    "record_id":      held["record_id"],
                    "ts":             held["ts"],
                    "figure":         held["figure"],
                    "value_held":     held["value_name"],
                    "value_failed":   failed["value_name"],
                    "resistance":     round(float(held.get("resistance", 0.0)), 4),
                    "text_excerpt":   str(held.get("text_excerpt", ""))[:300],
                    "held_label":     held["label"],
                    "failed_label":   failed["label"],
                    "training_weight": round(float(held.get("training_weight", 1.0)) * 1.5, 4),
                }
                events.append(event)
                try:
                    conn.execute(
                        """INSERT OR IGNORE INTO value_tension
                           (id, session_id, record_id, ts, value_held, value_failed,
                            resistance, text_excerpt)
                           VALUES (?,?,?,?,?,?,?,?)""",
                        (uid, event["session_id"], event["record_id"], float(event["ts"]),
                         event["value_held"], event["value_failed"],
                         event["resistance"], event["text_excerpt"]),
                    )
                except Exception:
                    pass
        conn.commit()
        conn.close()
    except Exception:
        pass

    return events


# ---------------------------------------------------------------------------
# Build training records
# ---------------------------------------------------------------------------

def build_training_records(
    observations: List[Dict[str, Any]],
    p1_threshold: float,
    p0_threshold: float,
    min_observations: int,
    min_consistency: float = 0.0,
    apy_context: Optional[Dict[str, List[Dict]]] = None,
    apy_passage_window: int = 5,
    apy_time_window_s: float = 259200.0,  # 72 hours
) -> List[Dict[str, Any]]:
    obs_count: Dict[Tuple[str, str], int] = defaultdict(int)
    for obs in observations:
        obs_count[(obs["session_id"], obs["value_name"])] += 1

    records: List[Dict[str, Any]] = []

    for obs in observations:
        if obs_count[(obs["session_id"], obs["value_name"])] < min_observations:
            continue
        if float(obs.get("observation_consistency", 0.5)) < min_consistency:
            continue

        text_excerpt = str(obs.get("text_excerpt") or "")
        resistance   = float(obs.get("resistance") or 0.0)
        doc_type     = str(obs.get("document_type") or "unknown")

        label, reason, confidence = classify_observation(
            text_excerpt, resistance, p1_threshold, p0_threshold
        )

        failure_hits = _find_markers(text_excerpt, _FAILURE_RE)
        hold_hits    = _find_markers(text_excerpt, _HOLD_RE)
        apy_hits     = _find_markers(text_excerpt, _APY_PRESSURE_RE)

        # ── Cross-passage APY upgrade ─────────────────────────────────────
        # If this passage was classified P0 (failure, no same-passage pressure)
        # check the APY context window for a recent pressure passage.
        pressure_source_id  = ""
        deferred_apy_lag_s  = 0.0
        deferred_apy_lag_n  = 0
        if label == "P0" and apy_context:
            sess_ctx = apy_context.get(obs.get("session_id", ""), [])
            obs_ts   = float(obs.get("ts", 0.0))
            obs_pidx = int(obs.get("_passage_idx", -1))  # -1 if not tracked
            for ctx_entry in sess_ctx:
                ctx_ts   = float(ctx_entry.get("ts", 0.0))
                ctx_pidx = int(ctx_entry.get("passage_idx", -1))
                # Skip context entries that come AFTER this passage
                if ctx_ts > obs_ts:
                    continue
                # Check time window (dated) or passage window (undated)
                within_time = (obs_ts - ctx_ts) <= apy_time_window_s
                within_passages = (
                    obs_pidx >= 0
                    and ctx_pidx >= 0
                    and (obs_pidx - ctx_pidx) <= apy_passage_window
                )
                if within_time or within_passages:
                    # Promote to cross-passage APY
                    label              = "APY"
                    reason             = "cross_passage_apy_pressure_context"
                    confidence         = 0.85
                    pressure_source_id = str(ctx_entry.get("record_id", ""))
                    deferred_apy_lag_s = round(obs_ts - ctx_ts, 1)
                    deferred_apy_lag_n = max(0, obs_pidx - ctx_pidx) if obs_pidx >= 0 and ctx_pidx >= 0 else 0
                    break  # Use the most recent (first in DESC order) qualifying entry

        fail_mode = ""
        if label == "APY":
            fail_mode = "APY"
        elif label == "P0":
            fail_mode = "UA" if not failure_hits else "SO"

        training_weight = round(
            _DOC_TYPE_WEIGHT.get(doc_type, 1.0) * float(obs.get("significance", 0.9)),
            4,
        )

        records.append({
            "id":               str(uuid.uuid4()),
            "source_obs_id":    obs["id"],
            "figure":           obs["figure_name"],
            "session_id":       obs["session_id"],
            "record_id":        obs.get("record_id", ""),
            "ts":               obs["ts"],
            "value_name":       obs["value_name"],
            "text_excerpt":     text_excerpt,
            "document_type":    doc_type,
            "significance":     round(float(obs.get("significance", 0.0)), 4),
            "resistance":       round(resistance, 4),
            "label":            label,
            "label_reason":     reason,
            "fail_mode":        fail_mode,
            "training_weight":  training_weight,
            "confidence":       round(confidence, 2),
            "pressure_markers": apy_hits,
            "failure_markers":  failure_hits,
            "hold_markers":             hold_hits,
            "disambiguation_confidence": round(float(obs.get("disambiguation_confidence", 1.0)), 4),
            "observation_consistency":   round(float(obs.get("observation_consistency", 0.5)), 4),
            "pressure_source_id":        pressure_source_id,
            "deferred_apy_lag_s":        deferred_apy_lag_s,
            "deferred_apy_lag_n":        deferred_apy_lag_n,
        })

    return records


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _write_jsonl(path: str, records: List[Dict[str, Any]]) -> int:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return len(records)


def _print_stats(records: List[Dict[str, Any]], label: str = "All") -> None:
    by_label: Dict[str, int] = defaultdict(int)
    by_figure: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    by_value: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for r in records:
        by_label[r["label"]] += 1
        by_figure[r["figure"]][r["label"]] += 1
        by_value[r["value_name"]][r["label"]] += 1

    print(f"\n{'-'*60}")
    print(f"  {label} — {len(records)} observations")
    print(f"{'-'*60}")
    print(f"  {'Label':<12}  {'Count':>6}")
    for lbl in ["P1", "P0", "APY", "AMBIGUOUS"]:
        n = by_label.get(lbl, 0)
        if n:
            bar = "#" * min(30, n // max(1, len(records) // 30))
            print(f"  {lbl:<12}  {n:>6}  {bar}")

    print(f"\n  By figure:")
    for fig in sorted(by_figure):
        counts = by_figure[fig]
        p1  = counts.get("P1", 0)
        p0  = counts.get("P0", 0)
        apy = counts.get("APY", 0)
        amb = counts.get("AMBIGUOUS", 0)
        total = p1 + p0 + apy + amb
        print(f"  {fig:<22}  total={total:>4}  P1={p1:>3}  P0={p0:>3}  APY={apy:>3}  AMB={amb:>3}")

    print(f"\n  Top values by P1 count:")
    p1_by_value = sorted(
        [(v, by_value[v].get("P1", 0)) for v in by_value],
        key=lambda x: -x[1],
    )
    for val, count in p1_by_value[:10]:
        if count > 0:
            print(f"  {val:<20}  P1={count:>3}  P0={by_value[val].get('P0', 0):>3}")


# ---------------------------------------------------------------------------
# Export main
# ---------------------------------------------------------------------------

def export(
    db_path: str = _VALUES_DB,
    figure_filter: Optional[str] = None,
    p1_threshold: float = 0.55,
    p0_threshold: float = 0.35,
    min_observations: int = 1,
    output_dir: str = _DEFAULT_OUTPUT_DIR,
    dry_run: bool = False,
    include_ambiguous: bool = True,
    min_consistency: float = 0.0,
    value_tension: bool = False,
) -> int:
    print(f"[export] Reading observations from {db_path}")
    observations = _read_figure_observations(db_path, figure_filter)

    if not observations:
        print("[export] No historical figure observations found.")
        print("[export]  Run: python -m cli.ingest --figure <name> --file <path> --doc-type <type>")
        return 0

    print(f"[export] Found {len(observations)} raw observations")
    print(f"[export] P1 threshold:    resistance >= {p1_threshold}")
    if min_consistency > 0.0:
        print(f"[export] Min consistency: {min_consistency}")
    print(f"[export] P0 threshold: resistance <  {p0_threshold}")

    apy_ctx = _load_apy_context(db_path)
    records = build_training_records(
        observations, p1_threshold, p0_threshold, min_observations, min_consistency,
        apy_context=apy_ctx,
    )
    _print_stats(records, label="Classification results")

    positive  = [r for r in records if r["label"] == "P1"]
    negative  = [r for r in records if r["label"] in ("P0", "APY")]
    ambiguous = [r for r in records if r["label"] == "AMBIGUOUS"]

    print(f"\n[export] Training set: {len(positive)} positive  {len(negative)} negative  {len(ambiguous)} ambiguous")

    if dry_run:
        if not positive and not negative:
            print("\n[export] NOTE: 0 negative examples found.")
            print("[export]   Ingest texts from figures with documented failures,")
            print("[export]   or use --p0-threshold to loosen the threshold.")
        print("\n[export] DRY RUN — no files written.")
        return 0

    if not positive and not negative:
        print("[export] Nothing to export. Check thresholds or run ingestion first.")
        return 0

    by_figure: Dict[str, List[Dict]] = defaultdict(list)
    for r in records:
        if r["label"] == "AMBIGUOUS" and not include_ambiguous:
            continue
        by_figure[r["figure"]].append(r)

    for fig, fig_records in by_figure.items():
        path = os.path.join(output_dir, f"ric_figure_{fig.lower()}.jsonl")
        n = _write_jsonl(path, fig_records)
        p1n = sum(1 for r in fig_records if r["label"] == "P1")
        p0n = sum(1 for r in fig_records if r["label"] in ("P0", "APY"))
        print(f"[export] {fig:<22}  {n:>4} records  (P1={p1n}  P0/APY={p0n})  -> {path}")

    pos_path = os.path.join(output_dir, "ric_historical_positive.jsonl")
    n_pos = _write_jsonl(pos_path, positive)
    print(f"\n[export] Positive  {n_pos:>4} records  -> {pos_path}")

    neg_path = os.path.join(output_dir, "ric_historical_negative.jsonl")
    n_neg = _write_jsonl(neg_path, negative)
    print(f"[export] Negative  {n_neg:>4} records  -> {neg_path}")

    by_label: Dict[str, int] = defaultdict(int)
    by_fig_label: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    by_val_label: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    total_weight_p1 = 0.0
    total_weight_p0 = 0.0

    for r in records:
        by_label[r["label"]] += 1
        by_fig_label[r["figure"]][r["label"]] += 1
        by_val_label[r["value_name"]][r["label"]] += 1
        if r["label"] == "P1":
            total_weight_p1 += r["training_weight"]
        elif r["label"] in ("P0", "APY"):
            total_weight_p0 += r["training_weight"]

    report = {
        "generated_at":          time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "db_path":               db_path,
        "p1_threshold":          p1_threshold,
        "p0_threshold":          p0_threshold,
        "min_observations":      min_observations,
        "total_observations":    len(observations),
        "total_classified":      len(records),
        "by_label": {
            "P1":        by_label.get("P1", 0),
            "P0":        by_label.get("P0", 0),
            "APY":       by_label.get("APY", 0),
            "AMBIGUOUS": by_label.get("AMBIGUOUS", 0),
        },
        "total_weight_positive": round(total_weight_p1, 3),
        "total_weight_negative": round(total_weight_p0, 3),
        "by_figure": {
            fig: {
                "P1":  by_fig_label[fig].get("P1", 0),
                "P0":  by_fig_label[fig].get("P0", 0),
                "APY": by_fig_label[fig].get("APY", 0),
            }
            for fig in sorted(by_fig_label)
        },
        "by_value": {
            val: {
                "P1":  by_val_label[val].get("P1", 0),
                "P0":  by_val_label[val].get("P0", 0),
                "APY": by_val_label[val].get("APY", 0),
            }
            for val in sorted(by_val_label, key=lambda v: -by_val_label[v].get("P1", 0))
        },
        "output_files": {
            "positive": pos_path,
            "negative": neg_path,
            "per_figure": {
                fig: os.path.join(output_dir, f"ric_figure_{fig.lower()}.jsonl")
                for fig in sorted(by_figure)
            },
        },
    }

    # Co-occurrence matrix (always computed, added to report)
    cooccurrence = _compute_cooccurrence(records)
    report["co_occurrence"] = cooccurrence
    print(f"[export] Co-occurrence: {len(cooccurrence)} value pairs co-observed")

    # Value tension detection (opt-in via --value-tension)
    if value_tension and not dry_run:
        tension_events = _detect_tensions(records, db_path)
        if tension_events:
            tension_path = os.path.join(output_dir, "ric_value_tensions.jsonl")
            n_t = _write_jsonl(tension_path, tension_events)
            print(f"[export] Tensions  {n_t:>4} events   -> {tension_path}")
            report["output_files"]["tensions"] = tension_path
            report["tension_events"] = len(tension_events)
        else:
            print("[export] Tensions: none detected")

    report_path = os.path.join(output_dir, "ric_historical_report.json")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"[export] Report    -> {report_path}")

    total = n_pos + n_neg
    print(f"\n[export] Done. {total} training examples written to {output_dir}")
    print(f"[export] Balance:  P1={n_pos}  P0/APY={n_neg}  ratio={n_pos/(n_neg or 1):.2f}:1")
    return total


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export RIC training data from historical figure observations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--figure", default=None,
                        help="Export only this figure (e.g. gandhi, lincoln)")
    parser.add_argument("--p1-threshold", type=float, default=0.55,
                        help="Min resistance score for P1 (default 0.55)")
    parser.add_argument("--p0-threshold", type=float, default=0.35,
                        help="Max resistance score for P0 (default 0.35)")
    parser.add_argument("--min-observations", type=int, default=1,
                        help="Min observations per value per figure (default 1)")
    parser.add_argument("--output-dir", default=_DEFAULT_OUTPUT_DIR,
                        help=f"Output directory (default {_DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print stats only — no files written")
    parser.add_argument("--no-ambiguous", action="store_true",
                        help="Exclude AMBIGUOUS observations from per-figure files")
    parser.add_argument("--min-consistency", type=float, default=0.0,
                        help="Min registry consistency score to include (default 0.0)")
    parser.add_argument("--db", default=_VALUES_DB,
                        help=f"Path to values.db (default {_VALUES_DB})")
    parser.add_argument("--value-tension", action="store_true",
                        help="Detect and export value tension events to ric_value_tensions.jsonl")
    args = parser.parse_args()

    export(
        db_path=args.db,
        figure_filter=args.figure,
        p1_threshold=args.p1_threshold,
        p0_threshold=args.p0_threshold,
        min_observations=args.min_observations,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
        include_ambiguous=not args.no_ambiguous,
        min_consistency=args.min_consistency,
        value_tension=args.value_tension,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
