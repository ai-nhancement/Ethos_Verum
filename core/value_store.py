"""
core/value_store.py

Value Data Capture — persistent store for value observations and registry.
SQLite singleton at data/values.db, WAL mode.

Four tables:
  value_observations  — raw per-passage value detections (append-only)
  value_registry      — aggregated value weights, per-session + cross-session
  value_watermarks    — per-session processing watermarks (managed by document_store)
  figure_sources      — metadata for ingested historical figure corpora

Historical figure sessions use session_id = "figure:<name>" (e.g. "figure:gandhi").
Cross-figure aggregation: WHERE session_id LIKE 'figure:%' GROUP BY value_name.
Universal cross-session aggregate: session_id = '' (covers all figure sessions).

Never raises. All write methods fail silently.
"""

from __future__ import annotations

import math
import os
import sqlite3
import threading
import time
import uuid
from typing import Dict, List, Optional

_DB_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "data", "values.db")
)

_LOCK = threading.Lock()
_INSTANCE: Optional["ValueStore"] = None


class ValueStore:
    """
    Thread-safe SQLite store for value observations and registry.
    Obtain via get_value_store() singleton factory.
    """

    def __init__(self, db_path: str = _DB_PATH):
        self._db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._local = threading.local()
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            conn = sqlite3.connect(self._db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            self._local.conn = conn
        return self._local.conn

    def _init_db(self) -> None:
        conn = self._conn()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS value_observations (
                id                        TEXT PRIMARY KEY,
                session_id                TEXT NOT NULL DEFAULT '',
                turn_id                   TEXT NOT NULL DEFAULT '',
                record_id                 TEXT NOT NULL DEFAULT '',
                ts                        REAL NOT NULL,
                value_name                TEXT NOT NULL,
                text_excerpt              TEXT NOT NULL DEFAULT '',
                significance              REAL NOT NULL DEFAULT 0.0,
                resistance                REAL NOT NULL DEFAULT 0.5,
                disambiguation_confidence REAL NOT NULL DEFAULT 1.0,
                doc_type                  TEXT NOT NULL DEFAULT 'unknown'
            );
            CREATE INDEX IF NOT EXISTS idx_vobs_session
                ON value_observations(session_id, ts);
            CREATE INDEX IF NOT EXISTS idx_vobs_value
                ON value_observations(value_name, ts);

            CREATE TABLE IF NOT EXISTS value_registry (
                session_id       TEXT NOT NULL DEFAULT '',
                value_name       TEXT NOT NULL,
                demonstrations   INTEGER NOT NULL DEFAULT 0,
                avg_significance REAL NOT NULL DEFAULT 0.0,
                avg_resistance   REAL NOT NULL DEFAULT 0.0,
                consistency      REAL NOT NULL DEFAULT 0.5,
                weight           REAL NOT NULL DEFAULT 0.0,
                first_seen_ts    REAL NOT NULL,
                last_seen_ts     REAL NOT NULL,
                updated_at       REAL NOT NULL,
                PRIMARY KEY (session_id, value_name)
            );
            CREATE INDEX IF NOT EXISTS idx_vreg_weight
                ON value_registry(session_id, weight DESC);

            CREATE TABLE IF NOT EXISTS value_watermarks (
                session_id        TEXT PRIMARY KEY,
                last_processed_ts REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS figure_sources (
                session_id    TEXT PRIMARY KEY,
                figure_name   TEXT NOT NULL,
                document_type TEXT NOT NULL DEFAULT 'unknown',
                ingested_at   REAL NOT NULL,
                passage_count INTEGER NOT NULL DEFAULT 0
            );
        """)
        conn.commit()
        # Migration: add column to existing DBs (SQLite ignores if already present via try/except)
        try:
            conn.execute(
                "ALTER TABLE value_observations ADD COLUMN disambiguation_confidence REAL NOT NULL DEFAULT 1.0"
            )
            conn.commit()
        except Exception:
            pass  # Column already exists
        try:
            conn.execute(
                "ALTER TABLE value_observations ADD COLUMN doc_type TEXT NOT NULL DEFAULT 'unknown'"
            )
            conn.commit()
        except Exception:
            pass  # Column already exists

    # ------------------------------------------------------------------
    # Observations (append-only)
    # ------------------------------------------------------------------

    def record_observation(
        self,
        session_id: str,
        turn_id: str,
        record_id: str,
        ts: float,
        value_name: str,
        text_excerpt: str,
        significance: float,
        resistance: float,
        disambiguation_confidence: float = 1.0,
        doc_type: str = "unknown",
    ) -> str:
        uid = str(uuid.uuid4())
        try:
            conn = self._conn()
            conn.execute(
                """INSERT INTO value_observations
                   (id, session_id, turn_id, record_id, ts, value_name,
                    text_excerpt, significance, resistance, disambiguation_confidence, doc_type)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (uid, session_id, turn_id, record_id, ts, value_name,
                 text_excerpt[:200], float(significance), float(resistance),
                 float(disambiguation_confidence), str(doc_type or "unknown")),
            )
            conn.commit()
        except Exception:
            pass
        return uid

    def get_observations(
        self,
        session_id: Optional[str] = None,
        value_name: Optional[str] = None,
        since_ts: Optional[float] = None,
        limit: int = 100,
    ) -> List[Dict]:
        try:
            conn = self._conn()
            clauses, params = [], []
            if session_id is not None:
                clauses.append("session_id=?"); params.append(session_id)
            if value_name:
                clauses.append("value_name=?"); params.append(value_name)
            if since_ts is not None:
                clauses.append("ts>=?"); params.append(since_ts)
            where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
            params.append(min(limit, 500))
            rows = conn.execute(
                f"SELECT * FROM value_observations {where} ORDER BY ts DESC LIMIT ?",
                params,
            ).fetchall()
            return [dict(r) for r in rows]
        except Exception:
            return []

    # ------------------------------------------------------------------
    # Registry (UPSERT — aggregated per session + cross-session)
    # ------------------------------------------------------------------

    def upsert_registry(
        self,
        session_id: str,
        value_name: str,
        significance: float,
        resistance: float,
        ts: float,
        doc_type: str = "unknown",
    ) -> None:
        try:
            conn = self._conn()
            row = conn.execute(
                "SELECT * FROM value_registry WHERE session_id=? AND value_name=?",
                (session_id, value_name),
            ).fetchone()

            now = time.time()
            if row is None:
                consistency = _compute_consistency(conn, session_id, value_name, resistance, ts, str(doc_type or "unknown"))
                conn.execute(
                    """INSERT INTO value_registry
                       (session_id, value_name, demonstrations, avg_significance,
                        avg_resistance, consistency, weight,
                        first_seen_ts, last_seen_ts, updated_at)
                       VALUES (?,?,?,?,?,?,?,?,?,?)""",
                    (session_id, value_name, 1,
                     float(significance), float(resistance),
                     consistency,
                     float(significance) * float(resistance) * consistency,
                     ts, ts, now),
                )
            else:
                n = int(row["demonstrations"]) + 1
                prev_sig = float(row["avg_significance"])
                prev_res = float(row["avg_resistance"])
                new_sig = prev_sig + (float(significance) - prev_sig) / n
                new_res = prev_res + (float(resistance) - prev_res) / n
                consistency = _compute_consistency(conn, session_id, value_name, resistance, ts, str(doc_type or "unknown"))
                weight = round(n * new_sig * new_res * consistency, 4)
                conn.execute(
                    """UPDATE value_registry SET
                       demonstrations=?, avg_significance=?, avg_resistance=?,
                       consistency=?, weight=?, last_seen_ts=?, updated_at=?
                       WHERE session_id=? AND value_name=?""",
                    (n, round(new_sig, 4), round(new_res, 4),
                     round(consistency, 4), weight,
                     ts, now, session_id, value_name),
                )
            conn.commit()
        except Exception:
            pass

    def get_registry(
        self,
        session_id: str = "",
        min_demonstrations: int = 1,
    ) -> List[Dict]:
        try:
            conn = self._conn()
            rows = conn.execute(
                """SELECT * FROM value_registry
                   WHERE session_id=? AND demonstrations>=?
                   ORDER BY weight DESC""",
                (session_id, min_demonstrations),
            ).fetchall()
            return [dict(r) for r in rows]
        except Exception:
            return []

    # ------------------------------------------------------------------
    # Watermarks
    # ------------------------------------------------------------------

    def get_watermark(self, session_id: str) -> float:
        try:
            conn = self._conn()
            row = conn.execute(
                "SELECT last_processed_ts FROM value_watermarks WHERE session_id=?",
                (session_id,),
            ).fetchone()
            return float(row[0]) if row else 0.0
        except Exception:
            return 0.0

    def set_watermark(self, session_id: str, ts: float) -> None:
        try:
            conn = self._conn()
            conn.execute(
                """INSERT INTO value_watermarks (session_id, last_processed_ts)
                   VALUES (?,?)
                   ON CONFLICT(session_id) DO UPDATE SET last_processed_ts=excluded.last_processed_ts""",
                (session_id, ts),
            )
            conn.commit()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Figure sources
    # ------------------------------------------------------------------

    def register_figure_source(
        self,
        session_id: str,
        figure_name: str,
        document_type: str,
        passage_count: int = 0,
    ) -> None:
        try:
            conn = self._conn()
            conn.execute(
                """INSERT INTO figure_sources
                   (session_id, figure_name, document_type, ingested_at, passage_count)
                   VALUES (?,?,?,?,?)
                   ON CONFLICT(session_id) DO UPDATE SET
                       figure_name=excluded.figure_name,
                       document_type=excluded.document_type,
                       ingested_at=excluded.ingested_at,
                       passage_count=excluded.passage_count""",
                (session_id, figure_name, document_type, time.time(), passage_count),
            )
            conn.commit()
        except Exception:
            pass

    def get_figure_source(self, session_id: str) -> Dict:
        try:
            conn = self._conn()
            row = conn.execute(
                "SELECT * FROM figure_sources WHERE session_id=?", (session_id,)
            ).fetchone()
            return dict(row) if row else {}
        except Exception:
            return {}

    def get_figures_list(self) -> List[Dict]:
        try:
            conn = self._conn()
            rows = conn.execute(
                """SELECT
                       fs.session_id,
                       fs.figure_name,
                       fs.document_type,
                       fs.ingested_at,
                       fs.passage_count,
                       COALESCE(r.total_demos, 0)    AS total_demonstrations,
                       COALESCE(r.value_count, 0)    AS values_observed,
                       COALESCE(r.top_value, '')     AS top_value
                   FROM figure_sources fs
                   LEFT JOIN (
                       SELECT session_id,
                              SUM(demonstrations) AS total_demos,
                              COUNT(*)            AS value_count,
                              MAX(value_name)     AS top_value
                       FROM value_registry
                       WHERE session_id LIKE 'figure:%'
                       GROUP BY session_id
                   ) r ON r.session_id = fs.session_id
                   ORDER BY fs.ingested_at DESC"""
            ).fetchall()
            return [dict(r) for r in rows]
        except Exception:
            return []

    def get_universal_registry(self, min_demonstrations: int = 1) -> List[Dict]:
        try:
            conn = self._conn()
            rows = conn.execute(
                """SELECT
                       value_name,
                       SUM(demonstrations)          AS total_demonstrations,
                       AVG(weight)                  AS avg_weight,
                       AVG(avg_significance)        AS avg_significance,
                       AVG(avg_resistance)          AS avg_resistance,
                       COUNT(DISTINCT session_id)   AS figure_count
                   FROM value_registry
                   WHERE session_id LIKE 'figure:%'
                   GROUP BY value_name
                   HAVING SUM(demonstrations) >= ?
                   ORDER BY total_demonstrations DESC""",
                (min_demonstrations,),
            ).fetchall()
            return [dict(r) for r in rows]
        except Exception:
            return []

    def get_stats(self) -> Dict:
        try:
            conn = self._conn()
            total = conn.execute(
                "SELECT COUNT(*) FROM value_observations"
            ).fetchone()[0]
            top = conn.execute(
                """SELECT value_name, SUM(demonstrations) AS total_demos
                   FROM value_registry WHERE session_id=''
                   ORDER BY total_demos DESC LIMIT 5"""
            ).fetchall()
            figures = conn.execute(
                "SELECT COUNT(*) FROM figure_sources"
            ).fetchone()[0]
            return {
                "total_observations": total,
                "top_values": [
                    {"value_name": r["value_name"], "demonstrations": r["total_demos"]}
                    for r in top
                ],
                "figure_count": figures,
            }
        except Exception:
            return {"total_observations": 0, "top_values": [], "figure_count": 0}


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _compute_consistency(
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
        return 0.5


# ------------------------------------------------------------------
# Singleton factory
# ------------------------------------------------------------------

def get_value_store() -> ValueStore:
    global _INSTANCE
    if _INSTANCE is None:
        with _LOCK:
            if _INSTANCE is None:
                _INSTANCE = ValueStore(_DB_PATH)
    return _INSTANCE
