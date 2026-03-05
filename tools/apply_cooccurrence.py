"""
tools/apply_cooccurrence.py

Implements §7.8 Value Co-occurrence and Interaction Modeling.

Changes:
  1. core/config.py     — VALUE_TENSION_PAIRS (5 default pairs from spec)
  2. core/value_store.py — value_tension table + record_tension() / get_tensions()
  3. cli/export.py      — _compute_cooccurrence(), _detect_tensions(),
                          --value-tension flag, co_occurrence in report JSON,
                          ric_value_tensions.jsonl output
"""

# ── 1. core/config.py ────────────────────────────────────────────────────────

with open("core/config.py", encoding="utf-8") as f:
    src = f.read()

old_config_end = """\
_default: Config | None = None


def get_config() -> Config:
    global _default
    if _default is None:
        _default = Config()
    return _default"""

new_config_end = """\
_default: Config | None = None


def get_config() -> Config:
    global _default
    if _default is None:
        _default = Config()
    return _default


# ---------------------------------------------------------------------------
# Value tension pairs (researcher-configurable)
# From Schwartz (1992) and Rokeach (1973) — default 5 pairs per §7.8 spec.
# Each tuple is (value_a, value_b); tension is symmetric.
# ---------------------------------------------------------------------------

VALUE_TENSION_PAIRS: list[tuple[str, str]] = [
    ("independence", "loyalty"),    # Autonomy vs. belonging
    ("fairness",     "compassion"), # Impartiality vs. mercy
    ("courage",      "patience"),   # Action vs. deliberation
    ("responsibility", "humility"), # Ownership vs. deference
    ("commitment",   "growth"),     # Persistence vs. revision
]

# Frozenset form for O(1) lookup
_TENSION_PAIR_SET: frozenset[frozenset] = frozenset(
    frozenset(p) for p in VALUE_TENSION_PAIRS
)


def is_tension_pair(v1: str, v2: str) -> bool:
    """Return True if (v1, v2) is a configured tension pair."""
    return frozenset({v1, v2}) in _TENSION_PAIR_SET"""

assert src.count(old_config_end) == 1, "config_end not found"
src = src.replace(old_config_end, new_config_end)

with open("core/config.py", "w", encoding="utf-8") as f:
    f.write(src)
print("core/config.py OK")

# ── 2. core/value_store.py ───────────────────────────────────────────────────

with open("core/value_store.py", encoding="utf-8") as f:
    src = f.read()

# 2a. Add value_tension table to schema (after figure_sources, before apy_context)
old_schema_apy = """\
            CREATE TABLE IF NOT EXISTS apy_context ("""

new_schema_with_tension = """\
            CREATE TABLE IF NOT EXISTS value_tension (
                id           TEXT PRIMARY KEY,
                session_id   TEXT NOT NULL DEFAULT '',
                record_id    TEXT NOT NULL DEFAULT '',
                ts           REAL NOT NULL,
                value_held   TEXT NOT NULL,
                value_failed TEXT NOT NULL,
                resistance   REAL NOT NULL DEFAULT 0.0,
                text_excerpt TEXT NOT NULL DEFAULT ''
            );
            CREATE INDEX IF NOT EXISTS idx_vtension_session
                ON value_tension(session_id, ts DESC);

            CREATE TABLE IF NOT EXISTS apy_context ("""

assert src.count(old_schema_apy) == 1, "schema_apy anchor not found"
src = src.replace(old_schema_apy, new_schema_with_tension)

# 2b. Add record_tension() and get_tensions() methods just before the APY Context section
old_before_apy_section = """\
    # ------------------------------------------------------------------
    # APY Context (cross-passage pressure window)
    # ------------------------------------------------------------------"""

new_tension_methods = """\
    # ------------------------------------------------------------------
    # Value Tension (co-occurrence interaction events)
    # ------------------------------------------------------------------

    def record_tension(
        self,
        session_id: str,
        record_id: str,
        ts: float,
        value_held: str,
        value_failed: str,
        resistance: float,
        text_excerpt: str,
    ) -> str:
        uid = str(__import__("uuid").uuid4())
        try:
            conn = self._conn()
            conn.execute(
                \"\"\"INSERT OR IGNORE INTO value_tension
                   (id, session_id, record_id, ts, value_held, value_failed,
                    resistance, text_excerpt)
                   VALUES (?,?,?,?,?,?,?,?)\"\"\",
                (uid, session_id, record_id, float(ts),
                 str(value_held), str(value_failed),
                 float(resistance), str(text_excerpt)[:300]),
            )
            conn.commit()
        except Exception:
            pass
        return uid

    def get_tensions(
        self,
        session_id: Optional[str] = None,
    ) -> List[Dict]:
        try:
            conn = self._conn()
            if session_id:
                rows = conn.execute(
                    \"\"\"SELECT * FROM value_tension
                       WHERE session_id=? ORDER BY ts\"\"\",
                    (session_id,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM value_tension ORDER BY ts"
                ).fetchall()
            return [dict(r) for r in rows]
        except Exception:
            return []

    # ------------------------------------------------------------------
    # APY Context (cross-passage pressure window)
    # ------------------------------------------------------------------"""

assert src.count(old_before_apy_section) == 1, "apy_section anchor not found"
src = src.replace(old_before_apy_section, new_tension_methods)

with open("core/value_store.py", "w", encoding="utf-8") as f:
    f.write(src)
print("core/value_store.py OK")

# ── 3. cli/export.py ─────────────────────────────────────────────────────────

with open("cli/export.py", encoding="utf-8") as f:
    src = f.read()

# 3a. Add import for is_tension_pair and ValueStore
old_root_import = "_ROOT = Path(__file__).resolve().parent.parent\n"
new_root_import = (
    "_ROOT = Path(__file__).resolve().parent.parent\n"
    "sys.path.insert(0, str(_ROOT))\n"
    "from core.config import is_tension_pair\n"
    "from core.value_store import get_value_store\n"
)
# sys.path.insert already present — only add what's missing
assert src.count(old_root_import) == 1
src = src.replace(old_root_import, new_root_import)

# 3b. Insert two new functions after _load_apy_context() and before Build section
old_build_section = """\
# ---------------------------------------------------------------------------
# Build training records
# ---------------------------------------------------------------------------"""

new_functions = """\
# ---------------------------------------------------------------------------
# Co-occurrence matrix
# ---------------------------------------------------------------------------

def _compute_cooccurrence(
    records: List[Dict[str, Any]],
) -> Dict[str, Dict]:
    \"\"\"
    Build a co-occurrence matrix for all 105 unordered value pairs.

    Groups records by (session_id, record_id) to find passages with 2+ values.
    For each pair, counts: both_detected, both_p1, mixed, both_p0.
    Returns {\"V1||V2\": {\"both_detected\": n, \"both_p1\": n, \"mixed\": n, \"both_p0\": n}}.
    \"\"\"
    from itertools import combinations

    # Group records by passage (session_id + record_id)
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
            l1, l2 = r1["label"], r2["label"]
            is_p1_1 = (l1 == "P1")
            is_p1_2 = (l2 == "P1")
            if is_p1_1 and is_p1_2:
                entry["both_p1"] += 1
            elif not is_p1_1 and not is_p1_2:
                entry["both_p0"] += 1
            else:
                entry["mixed"] += 1

    # Sort by both_detected descending
    return dict(sorted(matrix.items(), key=lambda kv: -kv[1]["both_detected"]))


# ---------------------------------------------------------------------------
# Value tension detection
# ---------------------------------------------------------------------------

def _detect_tensions(
    records: List[Dict[str, Any]],
    db_path: str,
) -> List[Dict[str, Any]]:
    \"\"\"
    Detect value tension events in classified records.

    A tension event occurs when two tension-pair values appear in the same
    passage and one is labeled P1 while the other is labeled P0/APY.

    Writes events to the value_tension table in values.db.
    Returns list of tension event dicts for JSONL output.
    \"\"\"
    import uuid as _uuid

    # Group records by passage
    passages: Dict[str, List[Dict]] = defaultdict(list)
    for r in records:
        key = r["session_id"] + "||" + r["record_id"]
        passages[key].append(r)

    events: List[Dict[str, Any]] = []

    try:
        store = get_value_store.__func__(db_path) if False else None  # avoid singleton
        import sqlite3 as _sql3
        conn = _sql3.connect(db_path, timeout=10.0)
        conn.execute("PRAGMA journal_mode=WAL")

        for recs in passages.values():
            if len(recs) < 2:
                continue
            from itertools import combinations
            for r1, r2 in combinations(recs, 2):
                v1, v2 = r1["value_name"], r2["value_name"]
                if not is_tension_pair(v1, v2):
                    continue
                l1, l2 = r1["label"], r2["label"]
                # One must be P1, the other P0 or APY
                r1_pos = (l1 == "P1")
                r2_pos = (l2 == "P1")
                if r1_pos == r2_pos:
                    continue  # both same direction — not a tension event
                held   = r1 if r1_pos else r2
                failed = r2 if r1_pos else r1
                uid = str(_uuid.uuid4())
                event = {
                    "id":           uid,
                    "session_id":   held["session_id"],
                    "record_id":    held["record_id"],
                    "ts":           held["ts"],
                    "figure":       held["figure"],
                    "value_held":   held["value_name"],
                    "value_failed": failed["value_name"],
                    "resistance":   round(float(held.get("resistance", 0.0)), 4),
                    "text_excerpt": str(held.get("text_excerpt", ""))[:300],
                    "held_label":   held["label"],
                    "failed_label": failed["label"],
                    "training_weight": round(
                        float(held.get("training_weight", 1.0)) * 1.5, 4
                    ),  # tension events carry 1.5× weight per spec
                }
                events.append(event)
                # Persist to DB
                try:
                    conn.execute(
                        \"\"\"INSERT OR IGNORE INTO value_tension
                           (id, session_id, record_id, ts, value_held, value_failed,
                            resistance, text_excerpt)
                           VALUES (?,?,?,?,?,?,?,?)\"\"\",
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
# ---------------------------------------------------------------------------"""

assert src.count(old_build_section) == 1, "build_section not found"
src = src.replace(old_build_section, new_functions)

# 3c. Add export() param: value_tension: bool = False
old_export_sig = """\
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
) -> int:"""

new_export_sig = """\
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
) -> int:"""

assert src.count(old_export_sig) == 1, "export_sig not found"
src = src.replace(old_export_sig, new_export_sig)

# 3d. After the report dict is built, add co_occurrence and tension output
old_report_path_block = """\
    report_path = os.path.join(output_dir, "ric_historical_report.json")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"[export] Report    -> {report_path}")"""

new_report_path_block = """\
    # Co-occurrence matrix
    cooccurrence = _compute_cooccurrence(records)
    report["co_occurrence"] = cooccurrence
    print(f"[export] Co-occurrence: {len(cooccurrence)} value pairs co-observed")

    # Value tension detection (optional)
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
    print(f"[export] Report    -> {report_path}")"""

assert src.count(old_report_path_block) == 1, "report_path_block not found"
src = src.replace(old_report_path_block, new_report_path_block)

# 3e. Add --value-tension flag to argparse and pass to export()
old_argparse_end = """\
    parser.add_argument("--db", default=_VALUES_DB,
                        help=f"Path to values.db (default {_VALUES_DB})")
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
    )"""

new_argparse_end = """\
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
    )"""

assert src.count(old_argparse_end) == 1, "argparse_end not found"
src = src.replace(old_argparse_end, new_argparse_end)

with open("cli/export.py", "w", encoding="utf-8") as f:
    f.write(src)
print("cli/export.py OK")

# ── Syntax check ─────────────────────────────────────────────────────────────
import py_compile
for path in ["core/config.py", "core/value_store.py", "cli/export.py"]:
    try:
        py_compile.compile(path, doraise=True)
        print(f"  syntax OK: {path}")
    except py_compile.PyCompileError as e:
        print(f"  SYNTAX ERROR in {path}: {e}")
        raise

print("\nAll §7.8 changes applied.")
