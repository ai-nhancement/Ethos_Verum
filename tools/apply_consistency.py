"""
Apply §7.9 — 4-component consistency scoring.

Changes:
  1. value_store.py: add doc_type column, replace _compute_consistency(), thread doc_type through
  2. value_extractor.py: pass doc_type to both store calls
  3. cli/export.py: observation_consistency + disambiguation_confidence in output, --min-consistency flag
"""
import math
import py_compile
import os

ROOT = 'C:/Ethos'

# ─────────────────────────────────────────────────────────────────────────────
# 1. value_store.py
# ─────────────────────────────────────────────────────────────────────────────
with open(f'{ROOT}/core/value_store.py', encoding='utf-8') as f:
    vs = f.read()

# 1a. Add doc_type to CREATE TABLE value_observations
vs = vs.replace(
    '                resistance                REAL NOT NULL DEFAULT 0.5,\n'
    '                disambiguation_confidence REAL NOT NULL DEFAULT 1.0\n'
    '            );',
    '                resistance                REAL NOT NULL DEFAULT 0.5,\n'
    '                disambiguation_confidence REAL NOT NULL DEFAULT 1.0,\n'
    '                doc_type                  TEXT NOT NULL DEFAULT \'unknown\'\n'
    '            );'
)

# 1b. Add migration for doc_type column alongside disambiguation_confidence migration
vs = vs.replace(
    '        try:\n'
    '            conn.execute(\n'
    '                "ALTER TABLE value_observations ADD COLUMN disambiguation_confidence REAL NOT NULL DEFAULT 1.0"\n'
    '            )\n'
    '            conn.commit()\n'
    '        except Exception:\n'
    '            pass  # Column already exists',
    '        try:\n'
    '            conn.execute(\n'
    '                "ALTER TABLE value_observations ADD COLUMN disambiguation_confidence REAL NOT NULL DEFAULT 1.0"\n'
    '            )\n'
    '            conn.commit()\n'
    '        except Exception:\n'
    '            pass  # Column already exists\n'
    '        try:\n'
    '            conn.execute(\n'
    '                "ALTER TABLE value_observations ADD COLUMN doc_type TEXT NOT NULL DEFAULT \'unknown\'"\n'
    '            )\n'
    '            conn.commit()\n'
    '        except Exception:\n'
    '            pass  # Column already exists'
)

# 1c. Add doc_type param to record_observation() + INSERT
vs = vs.replace(
    '        resistance: float,\n'
    '        disambiguation_confidence: float = 1.0,\n'
    '    ) -> str:\n'
    '        uid = str(uuid.uuid4())\n'
    '        try:\n'
    '            conn = self._conn()\n'
    '            conn.execute(\n'
    '                """INSERT INTO value_observations\n'
    '                   (id, session_id, turn_id, record_id, ts, value_name,\n'
    '                    text_excerpt, significance, resistance, disambiguation_confidence)\n'
    '                   VALUES (?,?,?,?,?,?,?,?,?,?)""",\n'
    '                (uid, session_id, turn_id, record_id, ts, value_name,\n'
    '                 text_excerpt[:200], float(significance), float(resistance),\n'
    '                 float(disambiguation_confidence)),\n'
    '            )',
    '        resistance: float,\n'
    '        disambiguation_confidence: float = 1.0,\n'
    '        doc_type: str = "unknown",\n'
    '    ) -> str:\n'
    '        uid = str(uuid.uuid4())\n'
    '        try:\n'
    '            conn = self._conn()\n'
    '            conn.execute(\n'
    '                """INSERT INTO value_observations\n'
    '                   (id, session_id, turn_id, record_id, ts, value_name,\n'
    '                    text_excerpt, significance, resistance, disambiguation_confidence, doc_type)\n'
    '                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",\n'
    '                (uid, session_id, turn_id, record_id, ts, value_name,\n'
    '                 text_excerpt[:200], float(significance), float(resistance),\n'
    '                 float(disambiguation_confidence), str(doc_type or "unknown")),\n'
    '            )'
)

# 1d. Add doc_type param to upsert_registry() and pass to _compute_consistency()
vs = vs.replace(
    '        significance: float,\n'
    '        resistance: float,\n'
    '        ts: float,\n'
    '    ) -> None:\n'
    '        try:\n'
    '            conn = self._conn()\n'
    '            row = conn.execute(\n'
    '                "SELECT * FROM value_registry WHERE session_id=? AND value_name=?",',
    '        significance: float,\n'
    '        resistance: float,\n'
    '        ts: float,\n'
    '        doc_type: str = "unknown",\n'
    '    ) -> None:\n'
    '        try:\n'
    '            conn = self._conn()\n'
    '            row = conn.execute(\n'
    '                "SELECT * FROM value_registry WHERE session_id=? AND value_name=?",',
    1
)
vs = vs.replace(
    '                consistency = _compute_consistency(conn, session_id, value_name, resistance)',
    '                consistency = _compute_consistency(conn, session_id, value_name, resistance, ts, str(doc_type or "unknown"))'
)

# 1e. Replace _compute_consistency() — 4-component formula
old_fn = '''def _compute_consistency(
    conn: sqlite3.Connection,
    session_id: str,
    value_name: str,
    new_resistance: float,
) -> float:
    try:
        rows = conn.execute(
            "SELECT resistance FROM value_observations WHERE session_id=? AND value_name=?",
            (session_id, value_name),
        ).fetchall()
        values = [float(r[0]) for r in rows] + [new_resistance]
        n = len(values)
        if n < 2:
            return 0.5
        mean = sum(values) / n
        if mean == 0.0:
            return 0.5
        variance = sum((v - mean) ** 2 for v in values) / n
        std_dev = math.sqrt(variance)
        consistency = 1.0 - (std_dev / mean)
        return max(0.0, min(1.0, consistency))
    except Exception:
        return 0.5'''

new_fn = '''def _compute_consistency(
    conn: sqlite3.Connection,
    session_id: str,
    value_name: str,
    new_resistance: float,
    new_ts: float,
    new_doc_type: str = "unknown",
) -> float:
    """
    Four-component consistency score for a (session_id, value_name) pair.

    Components (paper §7.9):
      0.30 × min(1, n / 10)                  — observation volume (saturates at 10)
      0.30 × max(0, 1 − σ_r / 0.40)          — resistance stability (low variance → high)
      0.25 × min(1, span_s / 31_536_000)     — temporal spread (saturates at 1 year)
      0.15 × min(1, distinct_doc_types / 3)  — source diversity (saturates at 3 types)

    The new observation (not yet committed) is included in the calculation.
    Never raises — returns 0.5 on any error.
    """
    try:
        rows = conn.execute(
            "SELECT resistance, ts, doc_type FROM value_observations "
            "WHERE session_id=? AND value_name=?",
            (session_id, value_name),
        ).fetchall()

        resistances = [float(r[0]) for r in rows] + [new_resistance]
        timestamps  = [float(r[1]) for r in rows] + [new_ts]
        doc_types   = {str(r[2]) for r in rows} | {new_doc_type}

        n = len(resistances)

        # Component 1: observation volume
        vol = min(1.0, n / 10.0)

        # Component 2: resistance stability
        mean_r = sum(resistances) / n
        if mean_r == 0.0 or n < 2:
            stab = 0.0
        else:
            variance = sum((v - mean_r) ** 2 for v in resistances) / n
            std_dev  = math.sqrt(variance)
            stab = max(0.0, 1.0 - std_dev / 0.40)

        # Component 3: temporal spread
        span_s = max(timestamps) - min(timestamps)
        spread = min(1.0, span_s / 31_536_000.0)  # saturates at 1 year

        # Component 4: source diversity
        diversity = min(1.0, len(doc_types) / 3.0)  # saturates at 3 types

        consistency = min(1.0,
            0.30 * vol      +
            0.30 * stab     +
            0.25 * spread   +
            0.15 * diversity
        )
        return round(max(0.0, consistency), 4)
    except Exception:
        return 0.5'''

assert vs.count(old_fn) == 1, f"Expected 1 _compute_consistency match"
vs = vs.replace(old_fn, new_fn)

with open(f'{ROOT}/core/value_store.py', 'w', encoding='utf-8') as f:
    f.write(vs)
py_compile.compile(f'{ROOT}/core/value_store.py', doraise=True)
print("value_store.py OK")


# ─────────────────────────────────────────────────────────────────────────────
# 2. value_extractor.py — pass doc_type to both store calls
# ─────────────────────────────────────────────────────────────────────────────
with open(f'{ROOT}/core/value_extractor.py', encoding='utf-8') as f:
    ve = f.read()

# Pass doc_type to record_observation
ve = ve.replace(
    '                disambiguation_confidence=sig.get("disambiguation_confidence", 1.0),\n'
    '            )',
    '                disambiguation_confidence=sig.get("disambiguation_confidence", 1.0),\n'
    '                doc_type=doc_type,\n'
    '            )'
)

# Pass doc_type to both upsert_registry calls
# First call (per-figure):
ve = ve.replace(
    '            val_store.upsert_registry(\n'
    '                session_id=session_id,\n'
    '                value_name=sig["value_name"],\n'
    '                significance=significance,\n'
    '                resistance=resistance,\n'
    '                ts=ts,\n'
    '            )\n'
    '            # Cross-figure aggregate (session_id=\'\')\n'
    '            val_store.upsert_registry(\n'
    '                session_id="",\n'
    '                value_name=sig["value_name"],\n'
    '                significance=significance,\n'
    '                resistance=resistance,\n'
    '                ts=ts,\n'
    '            )',
    '            val_store.upsert_registry(\n'
    '                session_id=session_id,\n'
    '                value_name=sig["value_name"],\n'
    '                significance=significance,\n'
    '                resistance=resistance,\n'
    '                ts=ts,\n'
    '                doc_type=doc_type,\n'
    '            )\n'
    '            # Cross-figure aggregate (session_id=\'\')\n'
    '            val_store.upsert_registry(\n'
    '                session_id="",\n'
    '                value_name=sig["value_name"],\n'
    '                significance=significance,\n'
    '                resistance=resistance,\n'
    '                ts=ts,\n'
    '                doc_type=doc_type,\n'
    '            )'
)

with open(f'{ROOT}/core/value_extractor.py', 'w', encoding='utf-8') as f:
    f.write(ve)
py_compile.compile(f'{ROOT}/core/value_extractor.py', doraise=True)
print("value_extractor.py OK")


# ─────────────────────────────────────────────────────────────────────────────
# 3. cli/export.py — observation_consistency + disambiguation_confidence + --min-consistency
# ─────────────────────────────────────────────────────────────────────────────
with open(f'{ROOT}/cli/export.py', encoding='utf-8') as f:
    ex = f.read()

# 3a. Update docstring options section
ex = ex.replace(
    '  --no-ambiguous       Exclude AMBIGUOUS observations from per-figure files\n'
    '  --db PATH            Path to values.db (default data/values.db)',
    '  --no-ambiguous       Exclude AMBIGUOUS observations from per-figure files\n'
    '  --min-consistency N  Min registry consistency score [0.0-1.0] to include (default 0.0)\n'
    '  --db PATH            Path to values.db (default data/values.db)'
)

# 3b. Update _read_figure_observations query to include disambiguation_confidence + consistency
old_q_all = (
    '                   FROM value_observations vo\n'
    '                   LEFT JOIN figure_sources fs ON fs.session_id = vo.session_id\n'
    '                   WHERE vo.session_id LIKE \'figure:%\'\n'
    '                   ORDER BY vo.session_id, vo.ts'
)
new_q_all = (
    '                   FROM value_observations vo\n'
    '                   LEFT JOIN figure_sources fs ON fs.session_id = vo.session_id\n'
    '                   LEFT JOIN value_registry vr\n'
    '                       ON vr.session_id = vo.session_id AND vr.value_name = vo.value_name\n'
    '                   WHERE vo.session_id LIKE \'figure:%\'\n'
    '                   ORDER BY vo.session_id, vo.ts'
)

old_cols_all = (
    '                       vo.id, vo.session_id, vo.record_id, vo.ts,\n'
    '                       vo.value_name, vo.text_excerpt,\n'
    '                       vo.significance, vo.resistance,\n'
    '                       COALESCE(fs.figure_name, SUBSTR(vo.session_id, 8)) AS figure_name,\n'
    '                       COALESCE(fs.document_type, \'unknown\')              AS document_type\n'
    '                   FROM value_observations vo\n'
    '                   LEFT JOIN figure_sources fs ON fs.session_id = vo.session_id\n'
    '                   WHERE vo.session_id LIKE \'figure:%\''
)
new_cols_all = (
    '                       vo.id, vo.session_id, vo.record_id, vo.ts,\n'
    '                       vo.value_name, vo.text_excerpt,\n'
    '                       vo.significance, vo.resistance,\n'
    '                       COALESCE(vo.disambiguation_confidence, 1.0) AS disambiguation_confidence,\n'
    '                       COALESCE(fs.figure_name, SUBSTR(vo.session_id, 8)) AS figure_name,\n'
    '                       COALESCE(fs.document_type, \'unknown\')              AS document_type,\n'
    '                       COALESCE(vr.consistency, 0.5)                      AS observation_consistency\n'
    '                   FROM value_observations vo\n'
    '                   LEFT JOIN figure_sources fs ON fs.session_id = vo.session_id\n'
    '                   LEFT JOIN value_registry vr\n'
    '                       ON vr.session_id = vo.session_id AND vr.value_name = vo.value_name\n'
    '                   WHERE vo.session_id LIKE \'figure:%\''
)
# Also update the figure-filtered query
old_cols_fig = (
    '                       vo.id, vo.session_id, vo.record_id, vo.ts,\n'
    '                       vo.value_name, vo.text_excerpt,\n'
    '                       vo.significance, vo.resistance,\n'
    '                       COALESCE(fs.figure_name, SUBSTR(vo.session_id, 8)) AS figure_name,\n'
    '                       COALESCE(fs.document_type, \'unknown\')              AS document_type\n'
    '                   FROM value_observations vo\n'
    '                   LEFT JOIN figure_sources fs ON fs.session_id = vo.session_id\n'
    '                   WHERE vo.session_id = ?'
)
new_cols_fig = (
    '                       vo.id, vo.session_id, vo.record_id, vo.ts,\n'
    '                       vo.value_name, vo.text_excerpt,\n'
    '                       vo.significance, vo.resistance,\n'
    '                       COALESCE(vo.disambiguation_confidence, 1.0) AS disambiguation_confidence,\n'
    '                       COALESCE(fs.figure_name, SUBSTR(vo.session_id, 8)) AS figure_name,\n'
    '                       COALESCE(fs.document_type, \'unknown\')              AS document_type,\n'
    '                       COALESCE(vr.consistency, 0.5)                      AS observation_consistency\n'
    '                   FROM value_observations vo\n'
    '                   LEFT JOIN figure_sources fs ON fs.session_id = vo.session_id\n'
    '                   LEFT JOIN value_registry vr\n'
    '                       ON vr.session_id = vo.session_id AND vr.value_name = vo.value_name\n'
    '                   WHERE vo.session_id = ?'
)

assert ex.count(old_cols_all) == 1
assert ex.count(old_cols_fig) == 1
ex = ex.replace(old_cols_all, new_cols_all)
ex = ex.replace(old_cols_fig, new_cols_fig)

# 3c. Add observation_consistency + disambiguation_confidence to training record
ex = ex.replace(
    '            "hold_markers":     hold_hits,\n'
    '        })',
    '            "hold_markers":             hold_hits,\n'
    '            "disambiguation_confidence": round(float(obs.get("disambiguation_confidence", 1.0)), 4),\n'
    '            "observation_consistency":   round(float(obs.get("observation_consistency", 0.5)), 4),\n'
    '        })'
)

# 3d. Add min_consistency param to build_training_records() + filter
ex = ex.replace(
    'def build_training_records(\n'
    '    observations: List[Dict[str, Any]],\n'
    '    p1_threshold: float,\n'
    '    p0_threshold: float,\n'
    '    min_observations: int,\n'
    ') -> List[Dict[str, Any]]:',
    'def build_training_records(\n'
    '    observations: List[Dict[str, Any]],\n'
    '    p1_threshold: float,\n'
    '    p0_threshold: float,\n'
    '    min_observations: int,\n'
    '    min_consistency: float = 0.0,\n'
    ') -> List[Dict[str, Any]]:'
)
ex = ex.replace(
    '        if obs_count[(obs["session_id"], obs["value_name"])] < min_observations:\n'
    '            continue',
    '        if obs_count[(obs["session_id"], obs["value_name"])] < min_observations:\n'
    '            continue\n'
    '        if float(obs.get("observation_consistency", 0.5)) < min_consistency:\n'
    '            continue'
)

# 3e. Add min_consistency param to export() + pass through
ex = ex.replace(
    '    include_ambiguous: bool = True,\n'
    ') -> int:\n'
    '    print(f"[export] Reading observations from {db_path}")',
    '    include_ambiguous: bool = True,\n'
    '    min_consistency: float = 0.0,\n'
    ') -> int:\n'
    '    print(f"[export] Reading observations from {db_path}")'
)
ex = ex.replace(
    '    records = build_training_records(observations, p1_threshold, p0_threshold, min_observations)',
    '    records = build_training_records(observations, p1_threshold, p0_threshold, min_observations, min_consistency)'
)
ex = ex.replace(
    '    print(f"[export] P1 threshold: resistance >= {p1_threshold}")',
    '    print(f"[export] P1 threshold:    resistance >= {p1_threshold}")\n'
    '    if min_consistency > 0.0:\n'
    '        print(f"[export] Min consistency: {min_consistency}")'
)

# 3f. Add --min-consistency to argparse + call
ex = ex.replace(
    '    parser.add_argument("--no-ambiguous", action="store_true",\n'
    '                        help="Exclude AMBIGUOUS observations from per-figure files")',
    '    parser.add_argument("--no-ambiguous", action="store_true",\n'
    '                        help="Exclude AMBIGUOUS observations from per-figure files")\n'
    '    parser.add_argument("--min-consistency", type=float, default=0.0,\n'
    '                        help="Min registry consistency score to include (default 0.0)")'
)
ex = ex.replace(
    '        include_ambiguous=not args.no_ambiguous,\n'
    '    )',
    '        include_ambiguous=not args.no_ambiguous,\n'
    '        min_consistency=args.min_consistency,\n'
    '    )'
)

with open(f'{ROOT}/cli/export.py', 'w', encoding='utf-8') as f:
    f.write(ex)
py_compile.compile(f'{ROOT}/cli/export.py', doraise=True)
print("cli/export.py OK")

print("\nAll changes applied successfully.")
