#!/usr/bin/env python3
"""
cli/corpus_stats.py

Ethos — print corpus-level statistics for all ingested figures.

Usage:
    python -m cli.corpus_stats
    python -m cli.corpus_stats --format json
    python -m cli.corpus_stats --output report.json
    python -m cli.corpus_stats --min-figures 2   # cross-figure values filter
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from core.corpus import get_full_report

_DOC_DB  = os.path.normpath(os.path.join(_ROOT, "data", "documents.db"))
_VAL_DB  = os.path.normpath(os.path.join(_ROOT, "data", "values.db"))


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _bar(n: int, total: int, width: int = 20) -> str:
    if total == 0:
        return " " * width
    filled = int(round(n / total * width))
    return "█" * filled + "░" * (width - filled)


def _pct(n: int | float, total: int | float) -> str:
    if total == 0:
        return "  0%"
    return f"{n/total*100:4.1f}%"


def print_report(report: dict, min_figures: int = 2) -> None:
    ov  = report["overview"]
    figs = report["figures"]
    vd   = report["value_distribution"]
    res  = report["resistance"]
    sig  = report.get("significance", {})
    cfv  = report["cross_figure_values"]

    # ── Overview ────────────────────────────────────────────────────────────
    print("\n══════════════════════════════════════════════════")
    print("  ETHOS CORPUS STATISTICS")
    print("══════════════════════════════════════════════════")
    print(f"  Figures:          {ov['figure_count']}")
    print(f"  Passages:         {ov['total_passages']:,}")
    print(f"  Observations:     {ov['total_observations']:,}")
    print(f"  Unique values:    {ov['unique_values']}")
    cov_pct = ov["coverage_rate"] * 100
    print(f"  Coverage rate:    {cov_pct:.1f}%  "
          f"(passages producing ≥1 observation)")

    if ov["doc_types"]:
        print(f"\n  Doc type breakdown:")
        total_p = ov["total_passages"] or 1
        for dt, cnt in sorted(ov["doc_types"].items(),
                               key=lambda x: -x[1]):
            bar = _bar(cnt, total_p)
            print(f"    {dt:<10s}  {bar}  {cnt:6,}  {_pct(cnt, total_p)}")

    # ── Per-figure summary ──────────────────────────────────────────────────
    if figs:
        print("\n──────────────────────────────────────────────────")
        print("  FIGURES")
        print("──────────────────────────────────────────────────")
        for f in figs:
            top = ", ".join(v["value_name"] for v in f["top_values"][:3])
            print(f"\n  {f['figure_name']:<20s}  [{f['doc_type']}]")
            print(f"    passages={f['passage_count']:,}  "
                  f"observations={f['observations']:,}  "
                  f"values={f['unique_values']}")
            if f["avg_resistance"]:
                print(f"    avg_resistance={f['avg_resistance']:.3f}  "
                      f"avg_significance={f['avg_significance']:.3f}")
            if top:
                print(f"    top values: {top}")

    # ── Corpus quality gate ─────────────────────────────────────────────────
    cq = report.get("corpus_quality", [])
    if cq:
        print("\n──────────────────────────────────────────────────")
        print("  CORPUS QUALITY GATE")
        print("──────────────────────────────────────────────────")
        col_w = max((len(q["figure_name"]) for q in cq), default=10) + 2
        print(f"  {'Figure':<{col_w}}  {'Docs':>4}  {'Types':>5}  {'Tier':<12}  {'Export'}")
        print(f"  {'-'*col_w}  {'-'*4}  {'-'*5}  {'-'*12}  {'-'*6}")
        for q in cq:
            fig    = q["figure_name"]
            docs   = q["document_count"]
            types  = q["distinct_doc_type_count"]
            tier   = q["confidence_tier"]
            ok     = "YES" if q["approved_for_export"] else "NO"
            marker = "  " if q["approved_for_export"] else "! "
            print(f"  {marker}{fig:<{col_w-2}}  {docs:>4}  {types:>5}  {tier:<12}  {ok}")
            for note in q["notes"]:
                print(f"    -> {note}")
        n_approved = sum(1 for q in cq if q["approved_for_export"])
        n_blocked  = len(cq) - n_approved
        print(f"\n  {n_approved} approved  {n_blocked} need more documents")
        if n_blocked:
            print("  Run: python -m cli.ingest --figure <name> --doc-type <type> --doc-title <title> ...")

    # ── Value distribution ──────────────────────────────────────────────────
    if vd:
        print("\n──────────────────────────────────────────────────")
        print("  VALUE DISTRIBUTION  (cross-figure)")
        print("──────────────────────────────────────────────────")
        max_demos = vd[0]["total_demonstrations"] if vd else 1
        for v in vd:
            bar = _bar(v["total_demonstrations"], max_demos)
            print(f"  {v['value_name']:<18s}  {bar}  "
                  f"demos={v['total_demonstrations']:4d}  "
                  f"figures={v['figure_count']}  "
                  f"avg_w={v['avg_weight']:.3f}")

    # ── Cross-figure values ─────────────────────────────────────────────────
    if cfv:
        print(f"\n──────────────────────────────────────────────────")
        print(f"  CROSS-FIGURE VALUES  (≥{min_figures} figures)")
        print("──────────────────────────────────────────────────")
        for v in cfv:
            print(f"  {v['value_name']:<18s}  "
                  f"figures={v['figure_count']}  "
                  f"total_demos={v['total_demonstrations']:4d}  "
                  f"avg_w={v['avg_weight']:.3f}")
        if not cfv:
            print(f"  (no values appear in ≥{min_figures} figures yet)")

    # ── Resistance distribution ─────────────────────────────────────────────
    if res.get("histogram"):
        print("\n──────────────────────────────────────────────────")
        print("  RESISTANCE DISTRIBUTION")
        print("──────────────────────────────────────────────────")
        total_obs = sum(res["histogram"].values()) or 1
        for bucket, cnt in res["histogram"].items():
            bar = _bar(cnt, total_obs)
            print(f"  {bucket}  {bar}  {cnt:5,}  {_pct(cnt, total_obs)}")
        print(f"\n  mean={res['mean']:.3f}  std={res['std']:.3f}  "
              f"min={res['min']:.3f}  median={res['median']:.3f}  max={res['max']:.3f}")

    # ── Significance distribution ────────────────────────────────────────────
    if sig.get("histogram"):
        print("\n──────────────────────────────────────────────────")
        print("  SIGNIFICANCE DISTRIBUTION")
        print("──────────────────────────────────────────────────")
        total_obs = sum(sig["histogram"].values()) or 1
        for bucket, cnt in sig["histogram"].items():
            bar = _bar(cnt, total_obs)
            print(f"  {bucket}  {bar}  {cnt:5,}  {_pct(cnt, total_obs)}")
        print(f"\n  mean={sig['mean']:.3f}  std={sig['std']:.3f}  "
              f"min={sig['min']:.3f}  median={sig['median']:.3f}  max={sig['max']:.3f}")
        if sig["std"] < 0.01:
            print("  NOTE: low std suggests all passages use a single significance value.")
            print("        Run cli/ingest.py without --significance to use doc-type defaults.")

    print("\n══════════════════════════════════════════════════\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Print corpus-level statistics for all ingested figures."
    )
    parser.add_argument("--format", choices=["text", "json"], default="text",
                        help="Output format (default: text)")
    parser.add_argument("--output", default=None,
                        help="Write JSON output to this file")
    parser.add_argument("--min-figures", type=int, default=2,
                        help="Min figures for cross-figure value section (default: 2)")
    parser.add_argument("--doc-db", default=_DOC_DB,
                        help="Path to documents.db (default: data/documents.db)")
    parser.add_argument("--val-db", default=_VAL_DB,
                        help="Path to values.db (default: data/values.db)")
    args = parser.parse_args()

    from core.corpus import get_full_report, get_cross_figure_values

    report = get_full_report(args.doc_db, args.val_db)
    # Re-filter cross-figure values with user-specified min
    report["cross_figure_values"] = get_cross_figure_values(
        args.val_db, min_figures=args.min_figures
    )

    if args.format == "json" or args.output:
        json_str = json.dumps(report, indent=2)
        if args.output:
            Path(args.output).write_text(json_str, encoding="utf-8")
            print(f"Report written to: {args.output}")
        if args.format == "json":
            print(json_str)
    else:
        print_report(report, min_figures=args.min_figures)

    return 0


if __name__ == "__main__":
    sys.exit(main())
