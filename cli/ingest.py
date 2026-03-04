#!/usr/bin/env python3
"""
cli/ingest.py

Ethos — ingest a historical figure text corpus into the value extraction pipeline.

Usage:
    python -m cli.ingest --figure gandhi --file samples/gandhi.txt --doc-type journal
    python -m cli.ingest --figure lincoln --file samples/lincoln.txt --doc-type speech --pub-year 1863
    python -m cli.ingest --figure nixon --file samples/nixon.txt --doc-type speech --dry-run

Document types:
  journal  — private writing, no audience pressure      (+0.35 resistance bonus)
  letter   — directed correspondence                    (+0.30)
  speech   — public address                             (+0.10)
  action   — documented real-world behavior             (+0.40)
  unknown  — default                                    (+0.20)

The script:
  1. Reads and segments the source text (~450 chars per passage, sentence-aware)
  2. Registers the figure in values.db (figure_sources table)
  3. Inserts each passage into documents.db with session_id = "figure:<name>"
  4. Resets the processing watermark so value_extractor picks up all passages
  5. Immediately runs value extraction and prints the profile summary

Run with --dry-run to preview segmentation without writing to any DB.
"""

from __future__ import annotations

import argparse
import calendar
import re
import sys
import time
from pathlib import Path
from typing import List

# Ensure project root is importable
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from core.document_store import get_document_store
from core.value_extractor import process_figure
from core.value_store import get_value_store

_MAX_PASSAGE_CHARS = 450
_DEFAULT_SIGNIFICANCE = 0.90
_SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?])\s+')


# ---------------------------------------------------------------------------
# Text segmentation
# ---------------------------------------------------------------------------

def segment_text(text: str, max_chars: int = _MAX_PASSAGE_CHARS) -> List[str]:
    """
    Split text into sentence-bounded passages of at most max_chars.
    Preserves sentence boundaries — never cuts mid-sentence.
    Filters out passages shorter than 30 chars.
    """
    sentences = _SENTENCE_SPLIT_RE.split(text.strip())
    passages: List[str] = []
    current = ""

    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        if current and len(current) + 1 + len(sent) > max_chars:
            passages.append(current.strip())
            current = sent
        else:
            current = (current + " " + sent).strip() if current else sent

    if current.strip():
        passages.append(current.strip())

    return [p for p in passages if len(p) >= 30]


# ---------------------------------------------------------------------------
# Main ingestion
# ---------------------------------------------------------------------------

def ingest(
    figure_name: str,
    source_file: str,
    document_type: str,
    pub_year: int | None = None,
    significance: float = _DEFAULT_SIGNIFICANCE,
    dry_run: bool = False,
    extract: bool = True,
) -> int:
    """
    Ingest passages from source_file for figure_name.
    Returns number of passages processed (0 on error).
    """
    path = Path(source_file)
    if not path.exists():
        print(f"[ingest] ERROR: file not found: {source_file}")
        return 0

    text = path.read_text(encoding="utf-8")
    passages = segment_text(text)

    if not passages:
        print(f"[ingest] No passages extracted from {source_file}")
        return 0

    session_id = f"figure:{figure_name.lower().strip()}"

    if pub_year:
        base_ts = float(calendar.timegm((pub_year, 1, 1, 0, 0, 0, 0, 0, 0)))
    else:
        base_ts = time.time() - (365 * 24 * 3600)

    print(f"[ingest] figure      = {figure_name!r}")
    print(f"[ingest] session_id  = {session_id!r}")
    print(f"[ingest] doc_type    = {document_type!r}")
    print(f"[ingest] source      = {source_file}")
    print(f"[ingest] passages    = {len(passages)}")
    print(f"[ingest] significance= {significance}")

    if dry_run:
        print(f"[ingest] DRY RUN — no writes")
        print()
        for i, p in enumerate(passages[:8]):
            print(f"  [{i:02d}] {p[:100]}{'...' if len(p) > 100 else ''}")
        if len(passages) > 8:
            print(f"  ... and {len(passages) - 8} more passages")
        return len(passages)

    doc_store = get_document_store()
    val_store = get_value_store()

    # Register figure metadata in values.db
    val_store.register_figure_source(
        session_id=session_id,
        figure_name=figure_name,
        document_type=document_type,
        passage_count=len(passages),
    )

    # Reset watermark — sentinel predates all of human history so
    # value_extractor picks up every passage regardless of pub_year.
    doc_store.set_watermark(session_id, -1_000_000_000_000.0)

    # Insert passages into documents.db
    inserted = 0
    for i, passage in enumerate(passages):
        ts = base_ts + i  # 1-second apart to preserve ordering
        doc_store.insert_passage(
            figure_name=figure_name,
            session_id=session_id,
            text=passage,
            doc_type=document_type,
            significance=significance,
            ts=ts,
        )
        inserted += 1

    print(f"[ingest] inserted {inserted} passages into documents.db")

    if extract and inserted > 0:
        print(f"[ingest] running value extraction for {session_id!r} ...")
        _run_and_print(session_id)

    return inserted


def _run_and_print(session_id: str) -> None:
    try:
        n = process_figure(session_id)
        val_store = get_value_store()
        reg = val_store.get_registry(session_id=session_id, min_demonstrations=1)
        universal = val_store.get_universal_registry(min_demonstrations=1)

        print(f"[extract] {session_id} — {len(reg)} values observed ({n} observations recorded):")
        for r in reg[:10]:
            print(f"  {r['value_name']:20s}  demos={r['demonstrations']}  "
                  f"weight={r['weight']:.4f}  resistance={r['avg_resistance']:.3f}")

        if universal:
            print()
            print(f"[extract] Universal Registry ({len(universal)} values across all figures):")
            for r in universal[:10]:
                print(f"  {r['value_name']:20s}  total_demos={r['total_demonstrations']}  "
                      f"figures={r['figure_count']}  avg_weight={r['avg_weight']:.4f}")
    except Exception as exc:
        print(f"[extract] ERROR: {exc}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Ingest a historical figure text corpus into the Ethos pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--figure", required=True,
                        help="Figure identifier, e.g. 'gandhi', 'lincoln', 'nixon'")
    parser.add_argument("--file", required=True,
                        help="Path to source text file (UTF-8)")
    parser.add_argument(
        "--doc-type", default="unknown",
        choices=["journal", "letter", "speech", "action", "unknown"],
        help="Document type — affects resistance bonus weighting",
    )
    parser.add_argument("--pub-year", type=int, default=None,
                        help="Publication/composition year (for timestamp ordering)")
    parser.add_argument("--significance", type=float, default=_DEFAULT_SIGNIFICANCE,
                        help=f"Significance score for all passages (default {_DEFAULT_SIGNIFICANCE})")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview segmentation only — no DB writes")
    parser.add_argument("--no-extract", action="store_true",
                        help="Skip immediate value extraction after ingestion")
    args = parser.parse_args()

    count = ingest(
        figure_name=args.figure,
        source_file=args.file,
        document_type=args.doc_type,
        pub_year=args.pub_year,
        significance=args.significance,
        dry_run=args.dry_run,
        extract=not args.no_extract,
    )

    if not args.dry_run and count > 0:
        print()
        print("[done] Next steps:")
        print(f"  python -m cli.export --figure {args.figure.lower()}")
        print(f"  python -m cli.export  (all figures)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
