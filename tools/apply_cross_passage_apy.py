"""
tools/apply_cross_passage_apy.py

Implements §7.6 Cross-Passage APY Detection.

Changes:
  1. core/value_store.py  — apy_context table + write/read/prune methods
  2. core/value_extractor.py — write APY pressure context during ingestion
  3. cli/export.py — cross-passage upgrade to classify_observation +
                    pressure_source_id / deferred_apy_lag fields
"""
import re

# ── 1. value_store.py ─────────────────────────────────────────────────────────

with open("core/value_store.py", encoding="utf-8") as f:
    src = f.read()

# 1a. Add apy_context table to CREATE TABLE block (before the closing """)
old_schema_close = """\
            CREATE TABLE IF NOT EXISTS figure_sources (
                session_id    TEXT PRIMARY KEY,
                figure_name   TEXT NOT NULL,
                document_type TEXT NOT NULL DEFAULT 'unknown',
                ingested_at   REAL NOT NULL,
                passage_count INTEGER NOT NULL DEFAULT 0
            );
        \"\"\")"""

new_schema_close = """\
            CREATE TABLE IF NOT EXISTS figure_sources (
                session_id    TEXT PRIMARY KEY,
                figure_name   TEXT NOT NULL,
                document_type TEXT NOT NULL DEFAULT 'unknown',
                ingested_at   REAL NOT NULL,
                passage_count INTEGER NOT NULL DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS apy_context (
                id           TEXT PRIMARY KEY,
                session_id   TEXT NOT NULL DEFAULT '',
                record_id    TEXT NOT NULL DEFAULT '',
                ts           REAL NOT NULL,
                passage_idx  INTEGER NOT NULL DEFAULT 0,
                markers      TEXT NOT NULL DEFAULT ''
            );
            CREATE INDEX IF NOT EXISTS idx_apyctx_session
                ON apy_context(session_id, ts DESC);
        \"\"\")"""

assert src.count(old_schema_close) == 1, f"schema_close not found (count={src.count(old_schema_close)})"
src = src.replace(old_schema_close, new_schema_close)

# 1b. Add three methods after the get_registry method block.
# Anchor: the line "    def get_registry(" — insert new methods before the
# module-level section separator that follows all ValueStore methods.
# The separator is:   \n\n\ndef get_value_store(
old_store_end = "\n\n# ------------------------------------------------------------------\n# Singleton factory\n# ------------------------------------------------------------------\n\ndef get_value_store("
new_methods = '''
    # ------------------------------------------------------------------
    # APY Context (cross-passage pressure window)
    # ------------------------------------------------------------------

    def write_apy_context(
        self,
        session_id: str,
        record_id: str,
        ts: float,
        passage_idx: int,
        markers: str,
        window_n: int = 5,
    ) -> None:
        """Write a pressure context entry and prune to the N most recent."""
        import uuid as _uuid
        try:
            conn = self._conn()
            conn.execute(
                """INSERT OR REPLACE INTO apy_context
                   (id, session_id, record_id, ts, passage_idx, markers)
                   VALUES (?,?,?,?,?,?)""",
                (str(_uuid.uuid4()), session_id, record_id, float(ts),
                 int(passage_idx), str(markers)),
            )
            # Prune: keep only the N most recent by passage_idx
            conn.execute(
                """DELETE FROM apy_context
                   WHERE session_id=?
                   AND id NOT IN (
                       SELECT id FROM apy_context
                       WHERE session_id=?
                       ORDER BY passage_idx DESC, ts DESC
                       LIMIT ?
                   )""",
                (session_id, session_id, int(window_n)),
            )
            conn.commit()
        except Exception:
            pass

    def get_apy_context(
        self,
        session_id: str,
        since_passage_idx: int = 0,
        since_ts: float = 0.0,
    ) -> List[Dict]:
        """Return recent pressure context entries for a session."""
        try:
            conn = self._conn()
            rows = conn.execute(
                """SELECT record_id, ts, passage_idx, markers
                   FROM apy_context
                   WHERE session_id=?
                     AND (passage_idx >= ? OR ts >= ?)
                   ORDER BY passage_idx DESC""",
                (session_id, int(since_passage_idx), float(since_ts)),
            ).fetchall()
            return [dict(r) for r in rows]
        except Exception:
            return []

    def prune_apy_context(self, session_id: str, keep_n: int = 5) -> None:
        """Keep only the N most recent pressure context entries."""
        try:
            conn = self._conn()
            conn.execute(
                """DELETE FROM apy_context
                   WHERE session_id=?
                   AND id NOT IN (
                       SELECT id FROM apy_context
                       WHERE session_id=?
                       ORDER BY passage_idx DESC, ts DESC
                       LIMIT ?
                   )""",
                (session_id, session_id, int(keep_n)),
            )
            conn.commit()
        except Exception:
            pass

'''

assert src.count(old_store_end) == 1, f"store_end not found"
src = src.replace(old_store_end, new_methods + "\n\n# ------------------------------------------------------------------\n# Singleton factory\n# ------------------------------------------------------------------\n\ndef get_value_store(")

with open("core/value_store.py", "w", encoding="utf-8") as f:
    f.write(src)
print("value_store.py OK")

# ── 2. value_extractor.py ─────────────────────────────────────────────────────

with open("core/value_extractor.py", encoding="utf-8") as f:
    src = f.read()

# 2a. Import _APY_PRESSURE_RE from export — no, better: define the pattern
# locally in value_extractor (same pattern as export.py _APY_PRESSURE_RE).
# Insert after the existing imports block (after "import time").
old_import_time = "import time\n"
new_import_block = (
    "import time\n"
    "import re as _re\n"
    "\n"
    "# APY pressure pattern — mirrors cli/export._APY_PRESSURE_RE (kept in sync manually)\n"
    "_APY_PRESSURE_RE_INGEST = _re.compile(\n"
    "    r'\\b(under pressure|when pressed|when threatened|when they demanded|'\n"
    "    r'to avoid punishment|to save myself|to protect my position|'\n"
    "    r'they insisted|they demanded|forced to|compelled to|'\n"
    "    r'or face consequences|or be punished|or lose everything|'\n"
    "    r'i told them what they wanted to hear|said what was expected)\\b',\n"
    "    _re.IGNORECASE,\n"
    ")\n"
)
assert src.count(old_import_time) >= 1, "import time not found"
# Only replace first occurrence (the imports, not any string literal)
src = src.replace(old_import_time, new_import_block, 1)

# 2b. In _run_extraction(), track passage_idx and write to apy_context.
# Anchor on the passage loop setup:
old_loop_init = "    recorded = 0\n    now = time.time()\n    latest_ts = watermark\n\n    for row in passages[:_MAX_PASSAGES_PER_RUN]:"
new_loop_init = (
    "    recorded = 0\n"
    "    now = time.time()\n"
    "    latest_ts = watermark\n"
    "    _passage_idx = 0\n"
    "    _apy_window_n = cfg.__dict__.get('apy_context_window_n', 5)\n"
    "\n"
    "    for row in passages[:_MAX_PASSAGES_PER_RUN]:"
)
assert src.count(old_loop_init) == 1, f"loop_init not found"
src = src.replace(old_loop_init, new_loop_init)

# 2c. After "latest_ts = max(latest_ts, ts)" at the bottom of the loop,
# increment passage_idx. Anchor on the unique closing of the passage loop:
old_loop_end = "        latest_ts = max(latest_ts, ts)\n\n    doc_store.set_watermark("
new_loop_end = (
    "        # Write APY pressure context if this passage has pressure markers\n"
    "        apy_markers = [m.group(0).lower() for m in _APY_PRESSURE_RE_INGEST.finditer(text)]\n"
    "        if apy_markers:\n"
    "            val_store.write_apy_context(\n"
    "                session_id=session_id,\n"
    "                record_id=record_id,\n"
    "                ts=ts,\n"
    "                passage_idx=_passage_idx,\n"
    "                markers=', '.join(apy_markers),\n"
    "                window_n=_apy_window_n,\n"
    "            )\n"
    "        _passage_idx += 1\n"
    "        latest_ts = max(latest_ts, ts)\n"
    "\n"
    "    doc_store.set_watermark("
)
assert src.count(old_loop_end) == 1, f"loop_end not found"
src = src.replace(old_loop_end, new_loop_end)

with open("core/value_extractor.py", "w", encoding="utf-8") as f:
    f.write(src)
print("value_extractor.py OK")

# ── 3. cli/export.py ─────────────────────────────────────────────────────────

with open("cli/export.py", encoding="utf-8") as f:
    src = f.read()

# 3a. Add _load_apy_context() function after _read_figure_observations()
old_build_header = "# ---------------------------------------------------------------------------\n# Build training records\n# ---------------------------------------------------------------------------"
new_apy_loader = '''\
# ---------------------------------------------------------------------------
# APY context loader (cross-passage pressure window)
# ---------------------------------------------------------------------------

def _load_apy_context(db_path: str) -> Dict[str, List[Dict]]:
    """
    Load all apy_context rows grouped by session_id.
    Returns {session_id: [{record_id, ts, passage_idx, markers}, ...]}.
    """
    result: Dict[str, List[Dict]] = defaultdict(list)
    if not Path(db_path).exists():
        return result
    try:
        conn = sqlite3.connect(db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                "SELECT session_id, record_id, ts, passage_idx, markers FROM apy_context"
            ).fetchall()
            for r in rows:
                result[r["session_id"]].append(dict(r))
        except Exception:
            pass
        finally:
            conn.close()
    except Exception:
        pass
    return result


# ---------------------------------------------------------------------------
# Build training records
# ---------------------------------------------------------------------------'''

assert src.count(old_build_header) == 1, "build_header not found"
src = src.replace(old_build_header, new_apy_loader)

# 3b. Add apy_context param to build_training_records() signature
old_btr_sig = (
    "def build_training_records(\n"
    "    observations: List[Dict[str, Any]],\n"
    "    p1_threshold: float,\n"
    "    p0_threshold: float,\n"
    "    min_observations: int,\n"
    "    min_consistency: float = 0.0,\n"
    ") -> List[Dict[str, Any]]:"
)
new_btr_sig = (
    "def build_training_records(\n"
    "    observations: List[Dict[str, Any]],\n"
    "    p1_threshold: float,\n"
    "    p0_threshold: float,\n"
    "    min_observations: int,\n"
    "    min_consistency: float = 0.0,\n"
    "    apy_context: Optional[Dict[str, List[Dict]]] = None,\n"
    "    apy_passage_window: int = 5,\n"
    "    apy_time_window_s: float = 259200.0,  # 72 hours\n"
    ") -> List[Dict[str, Any]]:"
)
assert src.count(old_btr_sig) == 1, "btr_sig not found"
src = src.replace(old_btr_sig, new_btr_sig)

# 3c. Replace the classify_observation call inside build_training_records
# with cross-passage-aware logic.
# Current code after the obs loop filter:
#   label, reason, confidence = classify_observation(
#       text_excerpt, resistance, p1_threshold, p0_threshold
#   )
old_classify_call = (
    "        label, reason, confidence = classify_observation(\n"
    "            text_excerpt, resistance, p1_threshold, p0_threshold\n"
    "        )\n"
    "\n"
    "        failure_hits = _find_markers(text_excerpt, _FAILURE_RE)\n"
    "        hold_hits    = _find_markers(text_excerpt, _HOLD_RE)\n"
    "        apy_hits     = _find_markers(text_excerpt, _APY_PRESSURE_RE)"
)
new_classify_call = '''\
        label, reason, confidence = classify_observation(
            text_excerpt, resistance, p1_threshold, p0_threshold
        )

        failure_hits = _find_markers(text_excerpt, _FAILURE_RE)
        hold_hits    = _find_markers(text_excerpt, _HOLD_RE)
        apy_hits     = _find_markers(text_excerpt, _APY_PRESSURE_RE)

        # ── Cross-passage APY upgrade ─────────────────────────────────────
        # If this passage was classified P0 (failure, no same-passage pressure)
        # check the APY context window for a recent pressure passage.
        pressure_source_id  = ""
        deferred_apy_lag_s  = 0.0
        deferred_apy_lag_n  = 0
        if label == "P0" and apy_context:
            sess_ctx = apy_context.get(obs.get("session_id", ""), [])
            obs_ts   = float(obs.get("ts", 0.0))
            obs_pidx = int(obs.get("_passage_idx", -1))  # -1 if not tracked
            for ctx_entry in sess_ctx:
                ctx_ts   = float(ctx_entry.get("ts", 0.0))
                ctx_pidx = int(ctx_entry.get("passage_idx", -1))
                # Skip context entries that come AFTER this passage
                if ctx_ts > obs_ts:
                    continue
                # Check time window (dated) or passage window (undated)
                within_time = (obs_ts - ctx_ts) <= apy_time_window_s
                within_passages = (
                    obs_pidx >= 0
                    and ctx_pidx >= 0
                    and (obs_pidx - ctx_pidx) <= apy_passage_window
                )
                if within_time or within_passages:
                    # Promote to cross-passage APY
                    label              = "APY"
                    reason             = "cross_passage_apy_pressure_context"
                    confidence         = 0.85
                    pressure_source_id = str(ctx_entry.get("record_id", ""))
                    deferred_apy_lag_s = round(obs_ts - ctx_ts, 1)
                    deferred_apy_lag_n = max(0, obs_pidx - ctx_pidx) if obs_pidx >= 0 and ctx_pidx >= 0 else 0
                    break  # Use the most recent (first in DESC order) qualifying entry'''

assert src.count(old_classify_call) == 1, f"classify_call not found"
src = src.replace(old_classify_call, new_classify_call)

# 3d. Add pressure_source_id and deferred_apy_lag fields to the records.append() dict
old_records_append_end = (
    '            "disambiguation_confidence": round(float(obs.get("disambiguation_confidence", 1.0)), 4),\n'
    '            "observation_consistency":   round(float(obs.get("observation_consistency", 0.5)), 4),\n'
    "        })\n"
    "\n"
    "    return records"
)
new_records_append_end = (
    '            "disambiguation_confidence": round(float(obs.get("disambiguation_confidence", 1.0)), 4),\n'
    '            "observation_consistency":   round(float(obs.get("observation_consistency", 0.5)), 4),\n'
    '            "pressure_source_id":        pressure_source_id,\n'
    '            "deferred_apy_lag_s":        deferred_apy_lag_s,\n'
    '            "deferred_apy_lag_n":        deferred_apy_lag_n,\n'
    "        })\n"
    "\n"
    "    return records"
)
assert src.count(old_records_append_end) == 1, "records_append_end not found"
src = src.replace(old_records_append_end, new_records_append_end)

# 3e. In export(), load apy_context and pass it to build_training_records
old_build_call = (
    "    records = build_training_records(observations, p1_threshold, p0_threshold, min_observations, min_consistency)"
)
new_build_call = (
    "    apy_ctx = _load_apy_context(db_path)\n"
    "    records = build_training_records(\n"
    "        observations, p1_threshold, p0_threshold, min_observations, min_consistency,\n"
    "        apy_context=apy_ctx,\n"
    "    )"
)
assert src.count(old_build_call) == 1, "build_call not found"
src = src.replace(old_build_call, new_build_call)

with open("cli/export.py", "w", encoding="utf-8") as f:
    f.write(src)
print("export.py OK")

# ── Syntax check ─────────────────────────────────────────────────────────────
import py_compile
for path in ["core/value_store.py", "core/value_extractor.py", "cli/export.py"]:
    try:
        py_compile.compile(path, doraise=True)
        print(f"  syntax OK: {path}")
    except py_compile.PyCompileError as e:
        print(f"  SYNTAX ERROR in {path}: {e}")
        raise

print("\nAll §7.6 changes applied.")
