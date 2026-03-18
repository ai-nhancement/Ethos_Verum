#!/usr/bin/env python3
"""
cli/batch_ingest.py

Ethos — batch ingest a multi-figure corpus from a manifest file.

Usage:
    python -m cli.batch_ingest --manifest corpus/manifest.json
    python -m cli.batch_ingest --manifest corpus/manifest.json --dry-run
    python -m cli.batch_ingest --manifest corpus/manifest.json --figure gandhi
    python -m cli.batch_ingest --manifest corpus/manifest.json --no-extract

Manifest format (JSON):
    {
      "corpus_name": "historical_figures_v1",
      "description": "optional description",
      "default_significance": 0.90,
      "figures": [
        {
          "name": "gandhi",
          "pronoun": "he",
          "sources": [
            {
              "file": "samples/gandhi_autobiography.txt",
              "doc_type": "journal",
              "pub_year": 1927,
              "significance": 0.95,
              "translation": false
            },
            {
              "file": "samples/gandhi_speeches.txt",
              "doc_type": "speech"
            }
          ]
        }
      ]
    }

Behaviour:
  - All source files for a figure are ingested with run_extract=False.
  - Value extraction runs ONCE per figure after all its sources are loaded.
  - Continues on per-file errors (reports failures in summary).
  - Dry-run shows segmentation preview for each source without DB writes.
  - Exits non-zero if any figure had one or more failures.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
_log = logging.getLogger(__name__)

_MAX_FILE_BYTES = 50 * 1024 * 1024  # 50 MB


# ---------------------------------------------------------------------------
# Manifest loading + validation
# ---------------------------------------------------------------------------

def load_manifest(path: str) -> Dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"Manifest is not valid JSON: {e}") from e


def _resolve_source_path(raw_path: str, manifest_dir: Path) -> Path:
    """Resolve source file path relative to the manifest directory."""
    p = Path(raw_path)
    if p.is_absolute():
        return p
    # Try relative to manifest directory first, then CWD, then project root
    for base in (manifest_dir, Path.cwd(), _ROOT):
        candidate = base / p
        if candidate.exists():
            return candidate
    return manifest_dir / p  # return as-is; caller will report missing


# ---------------------------------------------------------------------------
# Per-figure ingestion
# ---------------------------------------------------------------------------

def ingest_figure(
    figure_name: str,
    sources: List[Dict],
    manifest_dir: Path,
    default_significance: float = 0.90,
    dry_run: bool = False,
    extract: bool = True,
    pronoun: str = "unknown",
) -> Dict:
    """
    Ingest all sources for one figure.

    Returns:
        {
          figure_name, sources_ok, sources_failed,
          total_passages, total_observations, errors: [str]
        }
    """
    from core.pipeline import ingest_text
    from core.value_extractor import process_figure

    result = {
        "figure_name":        figure_name,
        "sources_ok":         0,
        "sources_failed":     0,
        "total_passages":     0,
        "total_observations": 0,
        "errors":             [],
    }

    for src in sources:
        raw_file   = src.get("file", "")
        doc_type   = src.get("doc_type", "unknown")
        pub_year   = src.get("pub_year", None)
        sig        = float(src.get("significance", default_significance))
        translation = src.get("translation", None)
        is_translation = True if translation is True else (None if translation is None else False)

        file_path = _resolve_source_path(raw_file, manifest_dir)

        if not file_path.exists():
            msg = f"File not found: {file_path}"
            _log.error("[%s] %s", figure_name, msg)
            result["errors"].append(msg)
            result["sources_failed"] += 1
            continue

        file_size = file_path.stat().st_size
        if file_size > _MAX_FILE_BYTES:
            msg = f"File too large ({file_size} bytes): {file_path.name}"
            _log.error("[%s] %s", figure_name, msg)
            result["errors"].append(msg)
            result["sources_failed"] += 1
            continue

        try:
            text = file_path.read_text(encoding="utf-8")
        except Exception as e:
            msg = f"Could not read {file_path.name}: {e}"
            _log.error("[%s] %s", figure_name, msg)
            result["errors"].append(msg)
            result["sources_failed"] += 1
            continue

        if dry_run:
            from cli.ingest import segment_text
            passages = segment_text(text)
            print(f"  [dry-run] {file_path.name} ({doc_type}) → {len(passages)} passages")
            for i, p in enumerate(passages[:3]):
                print(f"    [{i:02d}] {p[:90]}{'...' if len(p) > 90 else ''}")
            if len(passages) > 3:
                print(f"    ... and {len(passages) - 3} more")
            result["sources_ok"] += 1
            result["total_passages"] += len(passages)
            continue

        # Real ingest — run_extract=False; we'll extract once for the figure after all files
        ingest_result = ingest_text(
            figure_name=figure_name,
            text=text,
            doc_type=doc_type,
            pub_year=pub_year,
            is_translation=is_translation,
            significance=sig,
            run_extract=False,
            pronoun=pronoun,
        )

        if not ingest_result.ok:
            msg = f"{file_path.name}: {ingest_result.error}"
            _log.error("[%s] %s", figure_name, msg)
            result["errors"].append(msg)
            result["sources_failed"] += 1
            continue

        _log.info("[%s] %s (%s) → %d passages  lang=%s auth=%.2f",
                  figure_name, file_path.name, doc_type,
                  ingest_result.passages_ingested,
                  ingest_result.source_lang,
                  ingest_result.source_authenticity)
        result["sources_ok"] += 1
        result["total_passages"] += ingest_result.passages_ingested

    # Run extraction once for the figure (covers all sources)
    if extract and not dry_run and result["sources_ok"] > 0:
        try:
            session_id = f"figure:{figure_name.lower().strip()}"
            n = process_figure(session_id)
            result["total_observations"] = n
            _log.info("[%s] extraction → %d observations", figure_name, n)
        except Exception as e:
            msg = f"Extraction failed: {e}"
            _log.error("[%s] %s", figure_name, msg)
            result["errors"].append(msg)

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def batch_ingest(
    manifest_path: str,
    figure_filter: Optional[str] = None,
    dry_run: bool = False,
    extract: bool = True,
) -> List[Dict]:
    """
    Process a corpus manifest. Returns list of per-figure result dicts.
    """
    manifest = load_manifest(manifest_path)
    manifest_dir = Path(manifest_path).resolve().parent

    corpus_name         = manifest.get("corpus_name", "unnamed")
    default_significance = float(manifest.get("default_significance", 0.90))
    figures             = manifest.get("figures", [])

    if not figures:
        _log.warning("Manifest contains no figures.")
        return []

    # Apply figure filter
    if figure_filter:
        figures = [f for f in figures if f.get("name", "").lower() == figure_filter.lower()]
        if not figures:
            _log.error("Figure %r not found in manifest.", figure_filter)
            return []

    print(f"\nCorpus: {corpus_name}  ({len(figures)} figure(s))")
    if dry_run:
        print("  [DRY RUN — no DB writes]\n")

    t0 = time.time()
    results = []

    for fig in figures:
        figure_name = fig.get("name", "").strip()
        sources     = fig.get("sources", [])
        pronoun     = (fig.get("pronoun") or "unknown").lower().strip()

        if not figure_name:
            _log.warning("Skipping figure entry with no 'name'.")
            continue
        if not sources:
            _log.warning("Figure %r has no sources — skipping.", figure_name)
            continue

        print(f"\n  {figure_name}  ({len(sources)} source(s))")

        r = ingest_figure(
            figure_name=figure_name,
            sources=sources,
            manifest_dir=manifest_dir,
            default_significance=default_significance,
            dry_run=dry_run,
            extract=extract,
            pronoun=pronoun,
        )
        results.append(r)

        status = "✓" if r["sources_failed"] == 0 else "✗"
        print(f"  {status}  passages={r['total_passages']}  "
              f"observations={r['total_observations']}  "
              f"failed={r['sources_failed']}")
        for err in r["errors"]:
            print(f"     ! {err}")

    # Summary
    total_passages    = sum(r["total_passages"]    for r in results)
    total_obs         = sum(r["total_observations"] for r in results)
    total_failed_figs = sum(1 for r in results if r["sources_failed"] > 0)
    elapsed           = round(time.time() - t0, 1)

    print(f"\n{'─' * 56}")
    print(f"  Corpus:      {corpus_name}")
    print(f"  Figures:     {len(results)}  ({total_failed_figs} with errors)")
    print(f"  Passages:    {total_passages}")
    print(f"  Observations:{total_obs}")
    print(f"  Elapsed:     {elapsed}s")
    if not dry_run and total_failed_figs == 0:
        print(f"\n  Next: python -m cli.corpus_stats")
        print(f"        python -m cli.export")
    print()

    return results


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Batch ingest a multi-figure corpus from a manifest JSON.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--manifest", required=True,
                        help="Path to manifest JSON file")
    parser.add_argument("--figure", default=None,
                        help="Ingest only this figure name (from manifest)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview segmentation — no DB writes")
    parser.add_argument("--no-extract", action="store_true",
                        help="Skip value extraction after ingestion")
    args = parser.parse_args()

    try:
        results = batch_ingest(
            manifest_path=args.manifest,
            figure_filter=args.figure,
            dry_run=args.dry_run,
            extract=not args.no_extract,
        )
    except (FileNotFoundError, ValueError) as e:
        _log.error("%s", e)
        return 1

    # Non-zero exit if any figure had failures
    had_errors = any(r["sources_failed"] > 0 for r in results)
    return 1 if had_errors else 0


if __name__ == "__main__":
    sys.exit(main())
