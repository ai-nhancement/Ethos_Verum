#!/usr/bin/env python3
"""
cli/build_gold_annotations.py

Build gold annotation files for threshold validation from external sources.
Outputs JSONL consumed by cli/validate_thresholds.py.

Sources
-------
mftc     Moral Foundations Twitter Corpus (CSV with binary foundation columns)
         Download: https://osf.io/k5n7y/
         Also handles Moral Foundations Reddit Corpus (MFRC) in same format.

semeval  SemEval 2023 Task 4 — Human Value Detection (Schwartz values)
         Download: https://zenodo.org/records/10564870
         Requires two TSV files: arguments + labels.

sep      Stanford Encyclopedia of Philosophy entry scraper.
         Fetches an entry, extracts quoted passages with virtue analysis.
         No download needed — fetches live from plato.stanford.edu.

merge    Merge multiple gold JSONL files, resolving conflicts by majority vote.

Output format (one JSON per line)
----------------------------------
  {"text_excerpt": "...", "value_name": "integrity", "label": "P1"}

  OR (if matching against ingested observations by ID):
  {"observation_id": "<source_obs_id>", "label": "P1"}

Note on ingestion
-----------------
For mftc and semeval sources the gold texts must also be ingested into Ethos
before validate_thresholds.py can match them. Use --export-texts to write a
plain text file, then:

  python -m cli.ingest --figure mftc_calibration \\
      --file data/mftc_texts.txt --doc-type unknown

Usage
-----
  python -m cli.build_gold_annotations --source mftc \\
      --input data/MFTC.csv --output data/gold_mftc.jsonl

  python -m cli.build_gold_annotations --source semeval \\
      --arguments data/semeval/arguments-training.tsv \\
      --labels    data/semeval/labels-training.tsv \\
      --output    data/gold_semeval.jsonl

  python -m cli.build_gold_annotations --source sep \\
      --entry marcus-aurelius --output data/gold_sep_marcus.jsonl

  python -m cli.build_gold_annotations --merge \\
      data/gold_mftc.jsonl data/gold_sep.jsonl \\
      --output data/gold_merged.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
import urllib.request
from collections import defaultdict
from html.parser import HTMLParser
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
_log = logging.getLogger(__name__)

_MAP_PATH = _ROOT / "data" / "mft_ethos_map.json"
_SEP_BASE  = "https://plato.stanford.edu/entries/{entry}/"
_SEP_UA    = "Mozilla/5.0 (compatible; EthosResearch/1.0)"

# ---------------------------------------------------------------------------
# Load mapping config
# ---------------------------------------------------------------------------

def _load_map() -> Dict:
    if not _MAP_PATH.exists():
        _log.error("Mapping file not found: %s", _MAP_PATH)
        return {}
    with _MAP_PATH.open(encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# MFTC / MFRC parser
# ---------------------------------------------------------------------------

# All known MFT foundation column name variants across corpus versions
_MFT_VIRTUE_COLS = {
    "care", "fairness", "loyalty", "authority", "purity", "liberty", "equality",
    "proportionality",
}
_MFT_VICE_COLS = {
    "harm", "cheating", "inequality", "betrayal", "subversion", "degradation",
    "oppression",
}
_MFT_ALL_COLS  = _MFT_VIRTUE_COLS | _MFT_VICE_COLS
_TEXT_COLS     = {"text", "tweet", "tweet_text", "post", "body", "content"}
_MIN_TEXT_LEN  = 40


def _parse_mftc(
    path: str,
    mapping: Dict,
    min_annotators: int = 1,
    max_records: Optional[int] = None,
) -> List[Dict]:
    """
    Parse a Moral Foundations Twitter/Reddit Corpus CSV file.

    Handles two common formats:
      A) Aggregated: one row per text; foundation columns contain binary 0/1
         or an annotator-count integer (e.g., care=3 means 3 of 5 annotators).
      B) Per-annotator: one row per (text, annotator); foundation columns binary.

    Args:
        path:           Path to CSV file.
        mapping:        Loaded mft_ethos_map.json dict.
        min_annotators: Min annotator count to treat a label as present (format A).
        max_records:    Cap on output records (useful for testing).

    Returns:
        List of {text_excerpt, value_name, label} dicts.
    """
    virtue_map = mapping.get("mft_virtue_to_ethos", {})
    vice_map   = mapping.get("mft_vice_to_ethos", {})
    p1_label   = mapping.get("mft_virtue_label", "P1")
    p0_label   = mapping.get("mft_vice_label",   "P0")

    records: List[Dict] = []
    # Track (text_excerpt[:80], value_name) to deduplicate
    seen: set = set()

    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            _log.error("MFTC file has no header: %s", path)
            return []

        # Detect text column
        fieldnames_lower = {fn.lower(): fn for fn in reader.fieldnames}
        text_col = next(
            (fieldnames_lower[c] for c in _TEXT_COLS if c in fieldnames_lower),
            None,
        )
        if text_col is None:
            _log.error("Could not find text column in %s. Columns: %s",
                       path, list(reader.fieldnames))
            return []

        # Detect foundation columns present in this file
        found_foundations = {
            fn.lower(): fn for fn in reader.fieldnames
            if fn.lower() in _MFT_ALL_COLS
        }
        if not found_foundations:
            _log.error("No MFT foundation columns found in %s. Columns: %s",
                       path, list(reader.fieldnames))
            return []

        _log.info("MFTC: text_col=%r  foundations=%s", text_col,
                  sorted(found_foundations.keys()))

        for row in reader:
            text = (row.get(text_col) or "").strip()
            if len(text) < _MIN_TEXT_LEN:
                continue

            excerpt = text[:200]

            for foundation_lower, col_name in found_foundations.items():
                raw = row.get(col_name, "0") or "0"
                try:
                    val = float(raw)
                except ValueError:
                    continue

                # Format A: aggregated count; Format B: binary
                present = val >= min_annotators

                if not present:
                    continue

                if foundation_lower in _MFT_VIRTUE_COLS:
                    ethos_values = virtue_map.get(foundation_lower, [])
                    label = p1_label
                elif foundation_lower in _MFT_VICE_COLS:
                    ethos_values = vice_map.get(foundation_lower, [])
                    label = p0_label
                else:
                    continue

                for value_name in ethos_values:
                    key = (excerpt[:80], value_name)
                    if key in seen:
                        continue
                    seen.add(key)
                    records.append({
                        "text_excerpt": excerpt,
                        "value_name":   value_name,
                        "label":        label,
                        "source":       "mftc",
                        "foundation":   foundation_lower,
                    })
                    if max_records and len(records) >= max_records:
                        return records

    _log.info("MFTC: %d gold annotations extracted from %s", len(records), path)
    return records


# ---------------------------------------------------------------------------
# SemEval 2023 Task 4 parser
# ---------------------------------------------------------------------------

# SemEval column header variants
_SEMEVAL_ARG_ID_COL  = {"argument_id", "argumentid", "id"}
_SEMEVAL_TEXT_COL    = {"premise", "text", "argument"}
_SEMEVAL_STANCE_COL  = {"stance"}


def _parse_semeval(
    arguments_path: str,
    labels_path: str,
    mapping: Dict,
    max_records: Optional[int] = None,
) -> List[Dict]:
    """
    Parse SemEval 2023 Task 4 'Human Value Detection' TSV files.

    Args:
        arguments_path: arguments-training.tsv (or validation/test)
        labels_path:    labels-training.tsv (or validation)
        mapping:        Loaded mft_ethos_map.json dict.

    Returns:
        List of {text_excerpt, value_name, label} dicts.
    """
    semeval_map = mapping.get("semeval_to_ethos", {})

    # --- Load arguments ---
    args_by_id: Dict[str, Dict] = {}
    with open(arguments_path, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f, delimiter="\t")
        fn_lower = {fn.lower(): fn for fn in (reader.fieldnames or [])}
        id_col   = next((fn_lower[c] for c in _SEMEVAL_ARG_ID_COL if c in fn_lower), None)
        text_col = next((fn_lower[c] for c in _SEMEVAL_TEXT_COL   if c in fn_lower), None)
        if not id_col or not text_col:
            _log.error("SemEval arguments file missing id/text columns. Cols: %s",
                       list(reader.fieldnames or []))
            return []
        for row in reader:
            arg_id = row.get(id_col, "").strip()
            if arg_id:
                args_by_id[arg_id] = {
                    "text":   (row.get(text_col) or "").strip(),
                    "stance": (row.get(fn_lower.get("stance", ""), "") or "").strip(),
                }

    # --- Load labels ---
    records: List[Dict] = []
    seen: set = set()

    with open(labels_path, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f, delimiter="\t")
        fn_lower = {fn.lower(): fn for fn in (reader.fieldnames or [])}
        id_col   = next((fn_lower[c] for c in _SEMEVAL_ARG_ID_COL if c in fn_lower), None)
        if not id_col:
            _log.error("SemEval labels file missing id column. Cols: %s",
                       list(reader.fieldnames or []))
            return []

        # Find which of the 20 Schwartz value columns are present
        # SemEval header names use spaces in the original
        value_cols: List[Tuple[str, str]] = []  # (header_in_file, canonical_name)
        for fn in (reader.fieldnames or []):
            if fn == id_col:
                continue
            # Match against semeval_map keys — try exact, then space→colon substitution
            if fn in semeval_map:
                value_cols.append((fn, fn))
            else:
                # Some releases use underscore or different spacing
                normalized = fn.replace("_", " ").replace("-", ": ").strip()
                # Try title-cased versions
                for key in semeval_map:
                    if key.lower() == normalized.lower():
                        value_cols.append((fn, key))
                        break

        if not value_cols:
            _log.warning("No matching Schwartz value columns found in labels file. "
                         "Columns: %s", list(reader.fieldnames or []))

        for row in reader:
            arg_id = row.get(id_col, "").strip()
            arg    = args_by_id.get(arg_id)
            if not arg or len(arg["text"]) < _MIN_TEXT_LEN:
                continue

            excerpt = arg["text"][:200]
            # For "against" stance, the premise argues against the conclusion —
            # values present are still P1 (the premise expresses them).
            # SemEval labels don't encode direction of holding vs. failing.
            label = "P1"

            for col, canonical in value_cols:
                raw = row.get(col, "0") or "0"
                try:
                    present = int(raw) == 1
                except ValueError:
                    present = False
                if not present:
                    continue

                for value_name in semeval_map.get(canonical, []):
                    key = (excerpt[:80], value_name)
                    if key in seen:
                        continue
                    seen.add(key)
                    records.append({
                        "text_excerpt": excerpt,
                        "value_name":   value_name,
                        "label":        label,
                        "source":       "semeval",
                        "semeval_value": canonical,
                    })
                    if max_records and len(records) >= max_records:
                        return records

    _log.info("SemEval: %d gold annotations from %s + %s",
              len(records), arguments_path, labels_path)
    return records


# ---------------------------------------------------------------------------
# Stanford Encyclopedia of Philosophy scraper
# ---------------------------------------------------------------------------

class _SEPParser(HTMLParser):
    """
    Extract visible text from a Stanford SEP HTML page.
    Collects paragraph text and marks section headings.
    """

    def __init__(self):
        super().__init__()
        self.paragraphs: List[str] = []
        self._current: List[str] = []
        self._in_p  = False
        self._in_h  = False
        self._depth = 0
        self._skip_tags = {"script", "style", "noscript", "nav", "header", "footer"}
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in self._skip_tags:
            self._skip_depth += 1
            return
        if self._skip_depth:
            return
        if tag == "p":
            self._in_p = True
            self._current = []
        elif tag in ("h2", "h3"):
            self._in_h = True
            self._current = []

    def handle_endtag(self, tag):
        if tag in self._skip_tags:
            self._skip_depth = max(0, self._skip_depth - 1)
            return
        if self._skip_depth:
            return
        if tag == "p" and self._in_p:
            self._in_p = False
            text = " ".join(self._current).strip()
            if len(text) > 40:
                self.paragraphs.append(text)
            self._current = []
        elif tag in ("h2", "h3") and self._in_h:
            self._in_h = False
            text = " ".join(self._current).strip()
            if text:
                self.paragraphs.append(f"[SECTION: {text}]")
            self._current = []

    def handle_data(self, data):
        if self._skip_depth:
            return
        if self._in_p or self._in_h:
            self._current.append(data.strip())


_QUOTE_RE = re.compile(
    r'[\u201c\u2018\u0022\u0027]'           # opening quote char
    r'([^\u201d\u2019\u0022\u0027]{30,300})'  # 30-300 chars of content
    r'[\u201d\u2019\u0022\u0027]'           # closing quote char
)

# Window around a quoted passage to search for virtue indicators (chars)
_CONTEXT_WINDOW = 400

# Minimum word-overlap between paragraph and virtue map to use a whole paragraph
_PARA_MIN_VIRTUE_HITS = 1


def _extract_sep_pairs(
    paragraphs: List[str],
    virtue_map: Dict[str, List[str]],
) -> List[Dict]:
    """
    Find (passage, ethos_value) pairs from SEP paragraphs.

    Two extraction modes:

    Mode A — Inline quotes:
      Find paragraphs that contain an inline quoted passage (30-300 chars).
      Check surrounding text for virtue indicator words.
      Use the quote as text_excerpt.
      Good when SEP directly quotes the primary source.

    Mode B — Whole paragraph:
      Find paragraphs that contain virtue indicator words and are long enough
      to represent a substantive claim. Use the full paragraph as text_excerpt.
      Noisier — captures scholarly paraphrase of primary source, not direct quote.
      Useful when ingesting the SEP text itself as a secondary source.

    Limitations:
      - Cannot distinguish P1 from P0 — all SEP annotations are P1.
      - Mode B paragraphs are scholar's language, not primary source text.
      - Cross-reference with primary source ingestion for highest quality.
    """
    results: List[Dict] = []
    seen: set = set()

    for para in paragraphs:
        if para.startswith("[SECTION:"):
            continue

        para_lower = para.lower()

        # ── Mode A: inline quotes ────────────────────────────────────────────
        for m in _QUOTE_RE.finditer(para):
            quote = m.group(1).strip()
            if len(quote) < 30:
                continue
            start   = max(0, m.start() - _CONTEXT_WINDOW)
            end     = min(len(para), m.end() + _CONTEXT_WINDOW)
            context = para[start:m.start()] + " " + para[m.end():end]
            context_lower = context.lower()

            matched_values: List[str] = []
            for stem, values in virtue_map.items():
                if stem in context_lower:
                    matched_values.extend(values)

            for value_name in dict.fromkeys(matched_values):
                key = (quote[:80], value_name)
                if key in seen:
                    continue
                seen.add(key)
                results.append({
                    "text_excerpt": quote[:200],
                    "value_name":   value_name,
                    "label":        "P1",
                    "source":       "sep",
                    "mode":         "quote",
                })

        # ── Mode B: whole paragraph with virtue language ─────────────────────
        if len(para) < 80:
            continue

        para_values: List[str] = []
        for stem, values in virtue_map.items():
            if stem in para_lower:
                para_values.extend(values)

        if len(para_values) < _PARA_MIN_VIRTUE_HITS:
            continue

        excerpt = para[:300]
        for value_name in dict.fromkeys(para_values):
            key = (excerpt[:80], value_name)
            if key in seen:
                continue
            seen.add(key)
            results.append({
                "text_excerpt": excerpt,
                "value_name":   value_name,
                "label":        "P1",
                "source":       "sep",
                "mode":         "paragraph",
            })

    return results


def _fetch_sep(entry: str) -> List[Dict]:
    """
    Fetch a Stanford Encyclopedia of Philosophy entry and extract
    (quoted passage, Ethos value) annotation pairs.

    Args:
        entry: SEP entry slug, e.g. 'marcus-aurelius', 'seneca', 'epictetus'.

    Returns:
        List of {text_excerpt, value_name, label, source} dicts. Empty on error.
    """
    mapping = _load_map()
    virtue_map = mapping.get("sep_virtue_words_to_ethos", {})

    url = _SEP_BASE.format(entry=entry.lower().strip())
    _log.info("Fetching SEP entry: %s", url)

    try:
        req = urllib.request.Request(url, headers={"User-Agent": _SEP_UA})
        with urllib.request.urlopen(req, timeout=15) as resp:
            html = resp.read().decode("utf-8", errors="replace")
    except Exception as e:
        _log.error("Failed to fetch %s: %s", url, e)
        return []

    parser = _SEPParser()
    parser.feed(html)

    _log.info("SEP %r: %d paragraphs parsed", entry, len(parser.paragraphs))

    pairs = _extract_sep_pairs(parser.paragraphs, virtue_map)
    _log.info("SEP %r: %d annotation pairs extracted", entry, len(pairs))
    return pairs


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------

def _merge_gold(files: List[str], output_path: str) -> int:
    """
    Merge multiple gold JSONL files.

    For each (text_excerpt[:80], value_name) key, collect all labels.
    Apply majority vote; if tied, mark as AMBIGUOUS.
    Writes merged JSONL to output_path. Returns count written.
    """
    votes: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    meta:  Dict[Tuple[str, str], Dict]      = {}

    for fpath in files:
        p = Path(fpath)
        if not p.exists():
            _log.warning("Merge: file not found: %s", fpath)
            continue
        with p.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text   = obj.get("text_excerpt", "")
                vname  = obj.get("value_name", "")
                label  = obj.get("label", "")
                if not (text and vname and label):
                    continue
                key = (text[:80], vname)
                votes[key].append(label)
                if key not in meta:
                    meta[key] = {"text_excerpt": text, "value_name": vname,
                                 "source": obj.get("source", "unknown")}

    records: List[Dict] = []
    for key, lbls in votes.items():
        counts: Dict[str, int] = defaultdict(int)
        for lbl in lbls:
            counts[lbl] += 1
        top_count = max(counts.values())
        winners   = [lbl for lbl, n in counts.items() if n == top_count]
        final_label = winners[0] if len(winners) == 1 else "AMBIGUOUS"

        rec = dict(meta[key])
        rec["label"]        = final_label
        rec["vote_counts"]  = dict(counts)
        rec["n_annotators"] = len(lbls)
        records.append(rec)

    _write_gold(records, output_path)
    agreement = sum(1 for k, v in votes.items()
                    if len(set(v)) == 1) / max(len(votes), 1)
    _log.info("Merge: %d records, %.1f%% agreement across sources",
              len(records), agreement * 100)
    return len(records)


# ---------------------------------------------------------------------------
# Gold writer
# ---------------------------------------------------------------------------

def _write_gold(records: List[Dict], path: str) -> int:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    _log.info("Wrote %d gold annotations -> %s", len(records), path)
    return len(records)


def _write_texts(records: List[Dict], path: str) -> int:
    """Write unique text excerpts to a plain text file for ingestion."""
    seen: set = set()
    lines: List[str] = []
    for rec in records:
        t = rec.get("text_excerpt", "").strip()
        if t and t not in seen:
            seen.add(t)
            lines.append(t)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(lines))
    _log.info("Wrote %d unique passages -> %s", len(lines), path)
    return len(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build gold annotation files for Ethos threshold validation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="source", required=True)

    # ── mftc ────────────────────────────────────────────────────────────────
    p_mftc = sub.add_parser("mftc",
        help="Parse Moral Foundations Twitter/Reddit Corpus CSV")
    p_mftc.add_argument("--input",  required=True,
                        help="Path to MFTC or MFRC CSV file")
    p_mftc.add_argument("--output", required=True,
                        help="Output gold JSONL path")
    p_mftc.add_argument("--min-annotators", type=int, default=1,
                        help="Min annotator count to treat a label as present (default 1)")
    p_mftc.add_argument("--max-records", type=int, default=None,
                        help="Cap output at this many records (for testing)")
    p_mftc.add_argument("--export-texts", default=None,
                        help="Also write a plain text file of passages for cli/ingest.py")

    # ── semeval ──────────────────────────────────────────────────────────────
    p_sem = sub.add_parser("semeval",
        help="Parse SemEval 2023 Task 4 Human Value Detection TSV files")
    p_sem.add_argument("--arguments", required=True,
                       help="arguments-training.tsv (or validation/test)")
    p_sem.add_argument("--labels",    required=True,
                       help="labels-training.tsv (or validation)")
    p_sem.add_argument("--output",    required=True,
                       help="Output gold JSONL path")
    p_sem.add_argument("--max-records", type=int, default=None)
    p_sem.add_argument("--export-texts", default=None,
                       help="Also write a plain text file of passages for cli/ingest.py")

    # ── sep ──────────────────────────────────────────────────────────────────
    p_sep = sub.add_parser("sep",
        help="Scrape a Stanford Encyclopedia of Philosophy entry")
    p_sep.add_argument("--entry",  required=True,
                       help="SEP entry slug, e.g. marcus-aurelius, seneca, epictetus")
    p_sep.add_argument("--output", required=True,
                       help="Output gold JSONL path")
    p_sep.add_argument("--export-texts", default=None,
                       help="Also write a plain text file of passages for cli/ingest.py")

    # ── merge ────────────────────────────────────────────────────────────────
    p_merge = sub.add_parser("merge",
        help="Merge multiple gold JSONL files (majority vote on conflicts)")
    p_merge.add_argument("files", nargs="+",
                         help="Gold JSONL files to merge")
    p_merge.add_argument("--output", required=True,
                         help="Output merged gold JSONL path")

    args = parser.parse_args()
    mapping = _load_map()

    if args.source == "mftc":
        records = _parse_mftc(
            args.input, mapping,
            min_annotators=args.min_annotators,
            max_records=args.max_records,
        )
        if not records:
            _log.error("No records extracted — check file format and column names")
            return 1
        _write_gold(records, args.output)
        if args.export_texts:
            _write_texts(records, args.export_texts)
        _print_summary(records)

    elif args.source == "semeval":
        records = _parse_semeval(
            args.arguments, args.labels, mapping,
            max_records=args.max_records,
        )
        if not records:
            _log.error("No records extracted — check TSV format and column names")
            return 1
        _write_gold(records, args.output)
        if args.export_texts:
            _write_texts(records, args.export_texts)
        _print_summary(records)

    elif args.source == "sep":
        records = _fetch_sep(args.entry)
        if not records:
            _log.warning("No annotation pairs extracted from SEP entry %r", args.entry)
            _log.warning("This may mean the entry doesn't contain quoted primary-source text.")
            return 1
        _write_gold(records, args.output)
        if args.export_texts:
            _write_texts(records, args.export_texts)
        _print_summary(records)

    elif args.source == "merge":
        n = _merge_gold(args.files, args.output)
        if n == 0:
            _log.error("Merge produced 0 records — check input files")
            return 1

    print(f"\nNext step: ingest the texts into Ethos, then run:")
    print(f"  python -m cli.validate_thresholds --gold {args.output}")
    return 0


def _print_summary(records: List[Dict]) -> None:
    by_label: Dict[str, int]       = defaultdict(int)
    by_value: Dict[str, int]       = defaultdict(int)
    by_source: Dict[str, int]      = defaultdict(int)
    for r in records:
        by_label[r["label"]]            += 1
        by_value[r["value_name"]]       += 1
        by_source[r.get("source", "?")] += 1

    print(f"\n{'-'*50}")
    print(f"  Gold annotations: {len(records)}")
    print(f"  Labels:  " + "  ".join(f"{l}={n}" for l, n in sorted(by_label.items())))
    print(f"  Sources: " + "  ".join(f"{s}={n}" for s, n in sorted(by_source.items())))
    print(f"\n  Top values:")
    for v, n in sorted(by_value.items(), key=lambda x: -x[1])[:10]:
        print(f"    {v:<20s}  {n}")
    print(f"{'-'*50}\n")


if __name__ == "__main__":
    sys.exit(main())
