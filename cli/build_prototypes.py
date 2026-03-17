#!/usr/bin/env python3
"""
cli/build_prototypes.py

Build or rebuild the BGE-large value prototype vectors in Qdrant.

Builds two collections:
  ethos_value_prototypes   — hold/demonstration seeds (positive evidence)
  ethos_failure_prototypes — failure/violation seeds  (negative evidence)

Run this once after installing sentence-transformers and qdrant-client,
and again whenever value_seeds.py is updated.

Usage:
    python -m cli.build_prototypes
    python -m cli.build_prototypes --dry-run       # count seeds, no embedding
    python -m cli.build_prototypes --hold-only     # skip failure prototypes
    python -m cli.build_prototypes --value integrity courage  # rebuild specific values
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
_log = logging.getLogger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build Ethos value prototype vectors.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print seed counts only — do not embed or store.")
    ap.add_argument("--hold-only", action="store_true",
                    help="Build hold prototypes only; skip failure prototypes.")
    ap.add_argument("--value", nargs="+", metavar="VALUE",
                    help="Rebuild only these values (default: all 15).")
    args = ap.parse_args()

    from core.value_seeds import (
        get_seeds, seed_count_summary,
        get_failure_seeds, failure_seed_count_summary,
    )
    from core.semantic_store import VALUE_NAMES

    seeds         = get_seeds()
    failure_seeds = get_failure_seeds()
    summary         = seed_count_summary()
    failure_summary = failure_seed_count_summary()

    # Filter if --value specified
    if args.value:
        unknown = [v for v in args.value if v not in seeds]
        if unknown:
            _log.error("Unknown value names: %s", unknown)
            sys.exit(1)
        seeds         = {v: seeds[v]         for v in args.value}
        failure_seeds = {v: failure_seeds[v] for v in args.value if v in failure_seeds}

    print("\nHold seed sentence counts:")
    for v in VALUE_NAMES:
        if v in seeds:
            print(f"  {v:20s}  {summary[v]} seeds")

    if not args.hold_only:
        print("\nFailure seed sentence counts:")
        for v in VALUE_NAMES:
            if v in failure_seeds:
                print(f"  {v:20s}  {failure_summary.get(v, 0)} seeds")

    if args.dry_run:
        print("\n--dry-run: no embedding performed.")
        return

    # Check dependencies
    try:
        from core.embedder import is_available, _load_model
        _load_model()  # trigger load so we get the log message
        if not is_available():
            _log.error(
                "sentence-transformers not available. "
                "Install with: pip install sentence-transformers"
            )
            sys.exit(1)
    except Exception as e:
        _log.error("Embedder load failed: %s", e)
        sys.exit(1)

    try:
        from core.semantic_store import get_semantic_store, is_available as sem_ok
        if not sem_ok():
            _log.error(
                "Qdrant not available. Ensure Qdrant is running on localhost:6333. "
                "Install client with: pip install qdrant-client"
            )
            sys.exit(1)
    except Exception as e:
        _log.error("Semantic store init failed: %s", e)
        sys.exit(1)

    store = get_semantic_store()

    # --- Hold prototypes ---
    print(f"\nBuilding hold prototypes for {len(seeds)} value(s)...")
    hold_results = store.build_prototypes(seeds)

    ok  = [v for v, success in hold_results.items() if success]
    bad = [v for v, success in hold_results.items() if not success]
    print(f"  Stored:  {len(ok)}/{len(seeds)}")
    if ok:
        print(f"  Values:  {', '.join(ok)}")
    if bad:
        print(f"  Failed:  {', '.join(bad)}")

    total_hold = store.prototype_count()
    print(f"\nHold collection: {total_hold}/15 values ready.")

    # --- Failure prototypes ---
    if not args.hold_only and failure_seeds:
        print(f"\nBuilding failure prototypes for {len(failure_seeds)} value(s)...")
        fail_results = store.build_failure_prototypes(failure_seeds)

        fok  = [v for v, success in fail_results.items() if success]
        fbad = [v for v, success in fail_results.items() if not success]
        print(f"  Stored:  {len(fok)}/{len(failure_seeds)}")
        if fok:
            print(f"  Values:  {', '.join(fok)}")
        if fbad:
            print(f"  Failed:  {', '.join(fbad)}")

        total_fail = store.failure_prototype_count()
        print(f"\nFailure collection: {total_fail}/15 values ready.")

        if bad or fbad:
            sys.exit(1)
        if total_hold == 15 and total_fail == 15:
            print("\nBoth prototype collections fully built and ready.")
        else:
            missing = (15 - total_hold) + (15 - total_fail)
            print(f"Warning: {missing} prototype(s) still missing.")
    else:
        if bad:
            sys.exit(1)
        if total_hold == 15:
            print("Hold prototype collection fully built and ready.")
        else:
            print(f"Warning: {15 - total_hold} value(s) still missing from store.")


if __name__ == "__main__":
    main()
