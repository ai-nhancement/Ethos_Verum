#!/usr/bin/env python3
"""
cli/score_resistance.py

Interactive resistance scoring tool — review, correct, and extend the
resistance calibration dataset.

Two modes:

  review   Load an existing calibration file, display each example with the
           Ethos formula's computed resistance score, and let the researcher
           confirm or correct the human score and label.

  annotate Load a plain text file or JSONL of candidate passages, score each
           one interactively, and append results to a calibration output file.

Output format (JSONL):
  {
    "text_excerpt":       str,
    "value_name":         str,
    "label":              "P1" | "P0" | "APY",
    "resistance_score":   float,   <- human-judged ground truth
    "formula_score":      float,   <- Ethos formula output (for comparison)
    "resistance_rationale": str,
    "figure":             str,
    "source":             str,
    "doc_type":           str
  }

Scoring rubric
--------------
  0.00-0.20  Floor — value stated in comfort, no adversity present.
             "Honesty is the first chapter in the book of wisdom."

  0.20-0.40  Low — mild friction, social awkwardness, minor inconvenience.
             Disagreeing with a peer. Admitting a small mistake.

  0.40-0.60  Moderate — real stakes. Career risk, relationship damage,
             social exclusion, financial loss.
             "I knew speaking up would cost me the promotion, but I said it."

  0.60-0.80  High — serious personal cost. Loss of livelihood, freedom,
             major relationships. Sustained threat over time.
             "Even under arrest, I refused to name my associates."

  0.80-1.00  Extreme — life-threatening, total personal destruction.
             Death accepted or risked for the value.
             "I will not recant, even at the stake."

Usage
-----
  # Review and correct existing calibration file
  python -m cli.score_resistance review \\
      --input data/resistance_calibration_seed.jsonl \\
      --output data/resistance_calibration_reviewed.jsonl

  # Annotate new passages from a plain text file (one passage per blank line)
  python -m cli.score_resistance annotate \\
      --input data/candidates.txt \\
      --output data/resistance_calibration_seed.jsonl

  # Quick check: compute formula scores for all seed examples (no interaction)
  python -m cli.score_resistance compare \\
      --input data/resistance_calibration_seed.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

_RUBRIC = """
RESISTANCE RUBRIC
-----------------
0.00-0.20  Floor    — value stated in comfort, no adversity whatsoever
0.20-0.40  Low      — mild friction, social awkwardness, minor inconvenience
0.40-0.60  Moderate — real stakes: career risk, financial loss, social exclusion
0.60-0.80  High     — serious cost: livelihood, freedom, major relationships
0.80-1.00  Extreme  — life-threatening, total destruction, death accepted
"""

_VALUES = [
    "integrity", "courage", "compassion", "commitment", "patience",
    "responsibility", "fairness", "gratitude", "curiosity", "resilience",
    "love", "growth", "independence", "loyalty", "humility",
]

_LABELS = ["P1", "P0", "APY", "AMBIGUOUS", "SKIP"]


# ---------------------------------------------------------------------------
# Formula wrapper
# ---------------------------------------------------------------------------

def _formula_score(text: str, significance: float, doc_type: str) -> float:
    """Compute resistance using the current Ethos formula. Returns 0.5 on error."""
    try:
        from core.resistance import compute_resistance
        return compute_resistance(text, significance, doc_type)
    except Exception:
        return 0.5


# ---------------------------------------------------------------------------
# Input helpers
# ---------------------------------------------------------------------------

def _load_jsonl(path: str) -> List[Dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def _load_text_passages(path: str) -> List[str]:
    """Load plain text file — passages separated by blank lines."""
    text = Path(path).read_text(encoding="utf-8")
    blocks = [b.strip() for b in text.split("\n\n")]
    return [b for b in blocks if len(b) >= 30]


def _append_jsonl(path: str, record: Dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Interaction helpers
# ---------------------------------------------------------------------------

def _prompt(prompt: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    try:
        val = input(f"{prompt}{suffix}: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        sys.exit(0)
    return val if val else default


def _prompt_float(prompt: str, default: float, lo: float = 0.0, hi: float = 1.0) -> float:
    while True:
        raw = _prompt(prompt, str(round(default, 2)))
        try:
            v = float(raw)
            if lo <= v <= hi:
                return round(v, 2)
            print(f"  Must be between {lo} and {hi}.")
        except ValueError:
            print("  Enter a number.")


def _prompt_choice(prompt: str, choices: List[str], default: str) -> str:
    opts = "/".join(choices)
    while True:
        raw = _prompt(f"{prompt} ({opts})", default).upper()
        if raw in [c.upper() for c in choices]:
            return raw
        print(f"  Choose from: {opts}")


def _display_passage(rec: Dict, idx: int, total: int, formula: Optional[float] = None) -> None:
    print(f"\n{'='*60}")
    print(f"  [{idx}/{total}]  {rec.get('figure', '?')}  |  {rec.get('source', '?')}")
    print(f"{'='*60}")
    text = rec.get("text_excerpt", "")
    # Word-wrap at 60 chars
    words = text.split()
    line, lines = "", []
    for w in words:
        if line and len(line) + 1 + len(w) > 58:
            lines.append(line)
            line = w
        else:
            line = (line + " " + w).strip()
    if line:
        lines.append(line)
    for l in lines:
        print(f"  {l}")
    print()

    current_label    = rec.get("label", "?")
    current_human_r  = rec.get("resistance_score", "?")
    current_rationale = rec.get("resistance_rationale", "")

    print(f"  Value:          {rec.get('value_name', '?')}")
    print(f"  Label:          {current_label}")
    print(f"  Human score:    {current_human_r}")
    if formula is not None:
        delta = ""
        if isinstance(current_human_r, float):
            diff = formula - current_human_r
            delta = f"  (delta {diff:+.2f})"
        print(f"  Formula score:  {round(formula, 4)}{delta}")
    if current_rationale:
        print(f"  Rationale:      {current_rationale[:120]}")
    print(f"  Doc type:       {rec.get('doc_type', 'unknown')}")
    print()


# ---------------------------------------------------------------------------
# Review mode
# ---------------------------------------------------------------------------

def cmd_review(args) -> int:
    records = _load_jsonl(args.input)
    if not records:
        print(f"No records found in {args.input}")
        return 1

    out_path = args.output or args.input.replace(".jsonl", "_reviewed.jsonl")
    Path(out_path).unlink(missing_ok=True)  # start fresh

    print(_RUBRIC)
    print(f"Reviewing {len(records)} records from {args.input}")
    print("Press Enter to accept defaults. Type 's' to skip. Ctrl+C to quit.\n")

    changed = 0
    for i, rec in enumerate(records, 1):
        sig  = float(rec.get("significance", 0.80))
        dt   = rec.get("doc_type", "unknown")
        text = rec.get("text_excerpt", "")

        formula = _formula_score(text, sig, dt)
        _display_passage(rec, i, len(records), formula)

        action = _prompt("  Accept / Edit / Skip (a/e/s)", "a").lower()
        if action == "s":
            _append_jsonl(out_path, rec)
            continue

        if action == "e":
            # Value name
            print(f"  Values: {', '.join(_VALUES)}")
            new_val = _prompt("  Value", rec.get("value_name", "")).lower()
            if new_val and new_val not in _VALUES:
                print(f"  WARNING: '{new_val}' is not in the 15-value vocabulary")
            rec["value_name"] = new_val or rec.get("value_name", "")

            # Label
            new_label = _prompt_choice("  Label", _LABELS, rec.get("label", "P1"))
            if new_label == "SKIP":
                continue
            rec["label"] = new_label

            # Resistance score
            current_r = float(rec.get("resistance_score", 0.5))
            print(f"  Formula computed: {round(formula, 4)}   Your current: {current_r}")
            new_r = _prompt_float("  Resistance score", current_r)
            if new_r != current_r:
                changed += 1
            rec["resistance_score"] = new_r

            # Rationale
            new_rationale = _prompt("  Rationale (Enter to keep)",
                                    rec.get("resistance_rationale", ""))
            rec["resistance_rationale"] = new_rationale

            rec["formula_score"] = round(formula, 4)

        _append_jsonl(out_path, rec)

    print(f"\nDone. {len(records)} records -> {out_path}  ({changed} scores changed)")
    return 0


# ---------------------------------------------------------------------------
# Annotate mode
# ---------------------------------------------------------------------------

def cmd_annotate(args) -> int:
    # Load passages from text file or JSONL
    path = Path(args.input)
    if path.suffix == ".jsonl":
        recs = _load_jsonl(args.input)
        passages = [(r.get("text_excerpt", ""), r) for r in recs if r.get("text_excerpt")]
    else:
        raw = _load_text_passages(args.input)
        passages = [(p, {}) for p in raw]

    if not passages:
        print(f"No passages found in {args.input}")
        return 1

    out_path = args.output
    print(_RUBRIC)
    print(f"Annotating {len(passages)} passages. Output: {out_path}")
    print("Type 's' at any prompt to skip a passage. Ctrl+C to quit.\n")

    for i, (text, meta) in enumerate(passages, 1):
        print(f"\n{'='*60}")
        print(f"  [{i}/{len(passages)}]")
        print(f"{'='*60}")
        # Wrap text
        words = text.split()
        line, lines = "", []
        for w in words:
            if line and len(line) + 1 + len(w) > 58:
                lines.append(line)
                line = w
            else:
                line = (line + " " + w).strip()
        if line:
            lines.append(line)
        for l in lines:
            print(f"  {l}")
        print()

        # Value name
        default_val = meta.get("value_name", "")
        print(f"  Values: {', '.join(_VALUES)}")
        value_name = _prompt("  Value name (s=skip)", default_val).lower()
        if value_name == "s":
            continue
        if value_name not in _VALUES:
            print(f"  WARNING: '{value_name}' not in vocabulary — saved anyway")

        # Label
        label = _prompt_choice("  Label", _LABELS, meta.get("label", "P1"))
        if label == "SKIP":
            continue

        # Formula score (for reference)
        doc_type = _prompt("  Doc type (journal/letter/speech/action/unknown)", "unknown")
        formula = _formula_score(text, 0.80, doc_type)
        print(f"  Formula score: {round(formula, 4)}")

        # Human resistance score
        default_r = float(meta.get("resistance_score", formula))
        resistance_score = _prompt_float("  Your resistance score", default_r)

        # Rationale
        rationale = _prompt("  Brief rationale (why this score?)",
                            meta.get("resistance_rationale", ""))

        # Figure and source
        figure = _prompt("  Figure/speaker", meta.get("figure", ""))
        source = _prompt("  Source (doc, year)", meta.get("source", ""))

        record = {
            "text_excerpt":          text[:500],
            "value_name":            value_name,
            "label":                 label,
            "resistance_score":      resistance_score,
            "formula_score":         round(formula, 4),
            "resistance_rationale":  rationale,
            "figure":                figure,
            "source":                source,
            "doc_type":              doc_type,
        }
        _append_jsonl(out_path, record)
        print(f"  Saved.")

    print(f"\nDone. Results in {out_path}")
    return 0


# ---------------------------------------------------------------------------
# Compare mode — non-interactive formula audit
# ---------------------------------------------------------------------------

def cmd_compare(args) -> int:
    records = _load_jsonl(args.input)
    if not records:
        print(f"No records in {args.input}")
        return 1

    print(f"\n{'='*72}")
    print(f"  RESISTANCE FORMULA AUDIT  ({len(records)} examples)")
    print(f"{'='*72}")
    print(f"  {'Figure':<22} {'Val':<14} {'Lbl':<4} {'Human':>6} {'Formula':>8} {'Delta':>7}  Rationale")
    print(f"  {'-'*68}")

    errors: List[float] = []
    p1_errors: List[float] = []
    p0_errors: List[float] = []

    for rec in records:
        text  = rec.get("text_excerpt", "")
        dt    = rec.get("doc_type", "unknown")
        sig   = float(rec.get("significance", 0.80))
        human = rec.get("resistance_score")
        if human is None:
            continue

        formula = _formula_score(text, sig, dt)
        delta   = formula - float(human)
        errors.append(abs(delta))
        if rec.get("label") == "P1":
            p1_errors.append(abs(delta))
        elif rec.get("label") in ("P0", "APY"):
            p0_errors.append(abs(delta))

        flag = "  " if abs(delta) < 0.15 else ">>" if abs(delta) >= 0.30 else " >"
        fig  = rec.get("figure", "?")[:22]
        val  = rec.get("value_name", "?")[:14]
        lbl  = rec.get("label", "?")[:4]
        rat  = rec.get("resistance_rationale", "")[:45]

        print(f"{flag} {fig:<22} {val:<14} {lbl:<4} {float(human):>6.2f} {formula:>8.4f} {delta:>+7.4f}  {rat}")

    if errors:
        mae = sum(errors) / len(errors)
        print(f"\n  MAE (all):   {mae:.4f}  ({len(errors)} examples)")
        if p1_errors:
            print(f"  MAE (P1):    {sum(p1_errors)/len(p1_errors):.4f}  ({len(p1_errors)} examples)")
        if p0_errors:
            print(f"  MAE (P0):    {sum(p0_errors)/len(p0_errors):.4f}  ({len(p0_errors)} examples)")
        print(f"\n  >> = delta >= 0.30 (large disagreement — candidate for formula adjustment)")
        print(f"   > = delta >= 0.15 (moderate disagreement)")

        overestimates = sum(1 for r in records
                           if r.get("resistance_score") is not None
                           and _formula_score(r["text_excerpt"],
                                              float(r.get("significance", 0.80)),
                                              r.get("doc_type", "unknown"))
                                > float(r["resistance_score"]) + 0.10)
        underestimates = sum(1 for r in records
                            if r.get("resistance_score") is not None
                            and _formula_score(r["text_excerpt"],
                                               float(r.get("significance", 0.80)),
                                               r.get("doc_type", "unknown"))
                                 < float(r["resistance_score"]) - 0.10)
        print(f"\n  Formula overestimates:  {overestimates} examples")
        print(f"  Formula underestimates: {underestimates} examples")

    print(f"{'='*72}\n")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Interactive resistance scoring and formula audit tool.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_review = sub.add_parser("review",
        help="Review and correct an existing calibration JSONL file")
    p_review.add_argument("--input",  required=True,
                          help="Calibration JSONL to review")
    p_review.add_argument("--output", default=None,
                          help="Output path (default: input_reviewed.jsonl)")

    p_ann = sub.add_parser("annotate",
        help="Annotate new passages from a text or JSONL file")
    p_ann.add_argument("--input",  required=True,
                       help="Plain text (blank-line separated) or JSONL of candidate passages")
    p_ann.add_argument("--output", required=True,
                       help="Output calibration JSONL (appended)")

    p_cmp = sub.add_parser("compare",
        help="Non-interactive audit: show formula vs. human scores for all examples")
    p_cmp.add_argument("--input", required=True,
                       help="Calibration JSONL with resistance_score field")

    args = parser.parse_args()

    if args.cmd == "review":
        return cmd_review(args)
    elif args.cmd == "annotate":
        return cmd_annotate(args)
    elif args.cmd == "compare":
        return cmd_compare(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
