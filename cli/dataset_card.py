#!/usr/bin/env python3
"""
cli/dataset_card.py

Ethos — generate a HuggingFace-compatible dataset card from corpus statistics.

Usage:
    python -m cli.dataset_card
    python -m cli.dataset_card --output DATASET_CARD.md
    python -m cli.dataset_card --corpus-name "Ethos Historical Values v1"
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

_DOC_DB = os.path.normpath(os.path.join(_ROOT, "data", "documents.db"))
_VAL_DB = os.path.normpath(os.path.join(_ROOT, "data", "values.db"))


def generate_card(
    corpus_name: str,
    doc_db: str = _DOC_DB,
    val_db: str = _VAL_DB,
) -> str:
    from core.corpus import get_full_report

    r    = get_full_report(doc_db, val_db)
    ov   = r["overview"]
    figs = r["figures"]
    vd   = r["value_distribution"]
    cfv  = r["cross_figure_values"]
    res  = r["resistance"]

    date = time.strftime("%Y-%m-%d")

    # ── YAML frontmatter ────────────────────────────────────────────────────
    lines = [
        "---",
        "language:",
        "- en",
        "task_categories:",
        "- text-classification",
        "task_ids:",
        "- multi-label-classification",
        "tags:",
        "- ethics",
        "- values",
        "- moral-psychology",
        "- historical-figures",
        "- value-alignment",
        f"pretty_name: {corpus_name}",
        "---",
        "",
    ]

    # ── Title + description ─────────────────────────────────────────────────
    lines += [
        f"# {corpus_name}",
        "",
        "## Dataset Description",
        "",
        "This dataset was produced by the **Ethos Universal Value Extraction Pipeline** (UVEP).",
        "It contains value observations extracted from the documented behavior and writing of",
        "historical figures — labeled P1 (value held under resistance), P0 (value failed),",
        "and APY (yielded under pressure).",
        "",
        "> *'We didn't invent values. We extracted them from people who lived them —",
        "> and people who didn't.'*",
        "",
        "### Core claim",
        "",
        "Human values are behavioral patterns, extractable from documented history using",
        "deterministic computation and verifiable through accumulation.",
        "Resistance score measures the cost of holding a value — a value stated in comfort",
        "is weak signal; a value demonstrated at personal cost is strong signal.",
        "",
    ]

    # ── Corpus statistics ───────────────────────────────────────────────────
    lines += [
        "## Corpus Statistics",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Figures | {ov['figure_count']} |",
        f"| Total passages | {ov['total_passages']:,} |",
        f"| Total observations | {ov['total_observations']:,} |",
        f"| Unique values | {ov['unique_values']} |",
        f"| Coverage rate | {ov['coverage_rate']*100:.1f}% |",
        f"| Generated | {date} |",
        "",
    ]

    if ov["doc_types"]:
        lines += [
            "### Document type breakdown",
            "",
            "| Doc type | Passages | % |",
            "|----------|----------|---|",
        ]
        total_p = ov["total_passages"] or 1
        for dt, cnt in sorted(ov["doc_types"].items(), key=lambda x: -x[1]):
            lines.append(f"| {dt} | {cnt:,} | {cnt/total_p*100:.1f}% |")
        lines.append("")

    # ── Figures ─────────────────────────────────────────────────────────────
    if figs:
        lines += [
            "## Figures",
            "",
            "| Figure | Doc type | Passages | Observations | Top values |",
            "|--------|----------|----------|--------------|------------|",
        ]
        for f in figs:
            top = ", ".join(v["value_name"] for v in f["top_values"][:3])
            lines.append(
                f"| {f['figure_name']} | {f['doc_type']} "
                f"| {f['passage_count']:,} | {f['observations']:,} | {top} |"
            )
        lines.append("")

    # ── Value distribution ──────────────────────────────────────────────────
    if vd:
        lines += [
            "## Value Distribution",
            "",
            "| Value | Total demonstrations | Figure count | Avg weight | Avg resistance |",
            "|-------|---------------------|--------------|------------|----------------|",
        ]
        for v in vd:
            lines.append(
                f"| {v['value_name']} | {v['total_demonstrations']} "
                f"| {v['figure_count']} | {v['avg_weight']:.4f} "
                f"| {v['avg_resistance']:.4f} |"
            )
        lines.append("")

    # ── Cross-figure values ─────────────────────────────────────────────────
    if cfv:
        lines += [
            "## Cross-Figure Values",
            "",
            "Values that appear in ≥2 distinct figures are candidates for the universal value set.",
            "",
            "| Value | Figures | Total demonstrations | Avg weight |",
            "|-------|---------|---------------------|------------|",
        ]
        for v in cfv:
            lines.append(
                f"| {v['value_name']} | {v['figure_count']} "
                f"| {v['total_demonstrations']} | {v['avg_weight']:.4f} |"
            )
        lines.append("")

    # ── Resistance ──────────────────────────────────────────────────────────
    if res.get("mean"):
        lines += [
            "## Resistance Score Distribution",
            "",
            f"| Statistic | Value |",
            f"|-----------|-------|",
            f"| Mean | {res['mean']:.4f} |",
            f"| Std | {res['std']:.4f} |",
            f"| Median | {res['median']:.4f} |",
            f"| Min | {res['min']:.4f} |",
            f"| Max | {res['max']:.4f} |",
            "",
        ]

    # ── Schema ──────────────────────────────────────────────────────────────
    lines += [
        "## Schema",
        "",
        "Each record in the exported JSONL files contains:",
        "",
        "```json",
        "{",
        '  "figure": "gandhi",',
        '  "value_name": "courage",',
        '  "label": "P1",',
        '  "resistance": 0.95,',
        '  "significance": 0.90,',
        '  "consistency": 0.85,',
        '  "weight": 0.7225,',
        '  "doc_type": "journal",',
        '  "text_excerpt": "...",',
        '  "pub_year": 1927,',
        '  "source_authenticity": 1.0',
        "}",
        "```",
        "",
        "### Labels",
        "",
        "| Label | Meaning |",
        "|-------|---------|",
        "| P1 | Value demonstrated under resistance (positive signal) |",
        "| P0 | Value failed — documented lapse or contradiction |",
        "| APY | Answer-Pressure Yield — capitulated under pressure |",
        "",
    ]

    # ── Pipeline ────────────────────────────────────────────────────────────
    lines += [
        "## Pipeline",
        "",
        "Multi-layer extraction architecture:",
        "",
        "| Layer | Method | Notes |",
        "|-------|--------|-------|",
        "| Layer 1 | Keyword vocabulary | 15 values × ~30 triggers, disambiguation + first-person gate |",
        "| Layer 1b | MFD2.0 + MoralStrength lexicons | 2,493 entries; MFT→value mapping |",
        "| Layer 2 | BGE-large semantic similarity | Qdrant prototype store, 322 seed sentences |",
        "| Layer 3a | Structural regex | Adversity context, agency, resistance-to-failure patterns |",
        "| Layer 3b | DeBERTa zero-shot entailment | Independent value hypotheses, no Ethos vocabulary |",
        "",
        "Resistance scoring (per observation):",
        "```",
        "resistance = base(0.25) + significance_bonus + doc_type_bonus + adversity_marker_bonus",
        "             × source_authenticity × temporal_discount",
        "```",
        "",
        "Registry weight formula:",
        "```",
        "weight = demonstrations × avg_significance × avg_resistance × consistency",
        "```",
        "",
    ]

    lines += [
        "## License",
        "",
        "Dataset content: CC BY 4.0",
        "Pipeline code: MIT",
        "",
    ]

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate a HuggingFace dataset card from corpus statistics."
    )
    parser.add_argument("--output", default="DATASET_CARD.md",
                        help="Output markdown file (default: DATASET_CARD.md)")
    parser.add_argument("--corpus-name", default="Ethos Historical Values Corpus",
                        help="Dataset name for the card header")
    parser.add_argument("--doc-db", default=_DOC_DB)
    parser.add_argument("--val-db", default=_VAL_DB)
    args = parser.parse_args()

    card = generate_card(
        corpus_name=args.corpus_name,
        doc_db=args.doc_db,
        val_db=args.val_db,
    )

    Path(args.output).write_text(card, encoding="utf-8")
    print(f"Dataset card written to: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
