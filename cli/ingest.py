#!/usr/bin/env python3
"""
cli/ingest.py

Ethos — ingest a historical figure text corpus into the value extraction pipeline.

Usage:
    python -m cli.ingest --figure gandhi --file samples/gandhi.txt --doc-type journal
    python -m cli.ingest --figure lincoln --file samples/lincoln.txt --doc-type speech --pub-year 1863
    python -m cli.ingest --figure seneca --file samples/seneca.txt --doc-type letter --pub-year 65 --translation
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
import logging
import re
import sys
import time
from pathlib import Path
from typing import List

# Ensure project root is importable
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from core.document_store import get_document_store
from core.pipeline import ingest_text as _ingest_text
from core.value_extractor import process_figure
from core.value_store import get_value_store

_log = logging.getLogger(__name__)

_MAX_PASSAGE_CHARS = 450
_MAX_FILE_BYTES = 50 * 1024 * 1024  # 50 MB
_SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?])\s+')
_FIGURE_NAME_RE = re.compile(r'^[a-zA-Z0-9][a-zA-Z0-9_\-]{0,63}$')


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
    significance: float | None = None,
    is_translation: bool | None = None,
    dry_run: bool = False,
    extract: bool = True,
    doc_title: str = "",
    pronoun: str = "unknown",
) -> int:
    """
    Ingest passages from source_file for figure_name.
    Returns number of passages processed (0 on error).
    """
    if not _FIGURE_NAME_RE.match(figure_name):
        _log.error("Invalid figure name %r — must be alphanumeric/underscore/hyphen, 1-64 chars", figure_name)
        return 0

    path = Path(source_file)
    if not path.exists():
        _log.error("File not found: %s", source_file)
        return 0

    file_size = path.stat().st_size
    if file_size > _MAX_FILE_BYTES:
        _log.error("File too large: %d bytes (limit %d)", file_size, _MAX_FILE_BYTES)
        return 0

    text = path.read_text(encoding="utf-8")

    # Dry-run: preview segmentation without writing
    if dry_run:
        passages = segment_text(text)
        _log.info("DRY RUN — figure=%r doc_type=%r passages=%d",
                  figure_name, document_type, len(passages))
        print()
        for i, p in enumerate(passages[:8]):
            print(f"  [{i:02d}] {p[:100]}{'...' if len(p) > 100 else ''}")
        if len(passages) > 8:
            print(f"  ... and {len(passages) - 8} more passages")
        return len(passages)

    result = _ingest_text(
        figure_name=figure_name,
        text=text,
        doc_type=document_type,
        pub_year=pub_year,
        is_translation=is_translation,
        significance=significance,
        run_extract=extract,
        doc_title=doc_title,
        pronoun=pronoun,
    )

    if not result.ok:
        _log.error("Ingest failed: %s", result.error)
        return 0

    _log.info("figure=%r lang=%r authenticity=%.2f passages=%d observations=%d",
              figure_name, result.source_lang, result.source_authenticity,
              result.passages_ingested, result.observations_recorded)

    if extract and result.passages_ingested > 0:
        _run_and_print(result.session_id)

    return result.passages_ingested


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
                        help="Publication/composition year (for timestamp ordering + temporal discount)")
    parser.add_argument("--translation", action="store_true", default=False,
                        help="Declare this document is a translation (sets source_authenticity=0.85)")
    parser.add_argument("--significance", type=float, default=None,
                        help="Significance score for all passages (default: auto from doc-type)")
    parser.add_argument("--doc-title", default="",
                        help="Document title for corpus tracking (e.g. 'Meditations', 'Letter to Atticus')")
    parser.add_argument(
        "--pronoun", required=True,
        choices=["he", "she", "they", "i"],
        help="Figure's pronoun — used for subject/object resolution in phrase detection",
    )
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
        is_translation=True if args.translation else None,
        dry_run=args.dry_run,
        extract=not args.no_extract,
        doc_title=args.doc_title,
        pronoun=args.pronoun,
    )

    if not args.dry_run and count > 0:
        print()
        print("[done] Next steps:")
        print(f"  python -m cli.export --figure {args.figure.lower()}")
        print(f"  python -m cli.export  (all figures)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
