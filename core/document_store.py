from __future__ import annotations

import os
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import List, Optional

_DB_PATH = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "data", "documents.db"))

_DDL = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS passages (
    id          TEXT PRIMARY KEY,
    figure_name TEXT NOT NULL,
    session_id  TEXT NOT NULL,
    text        TEXT NOT NULL,
    doc_type    TEXT NOT NULL,
    significance REAL NOT NULL DEFAULT 0.90,
    ts          REAL NOT NULL,
    ingested_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_passages_session ON passages (session_id, ts);

CREATE TABLE IF NOT EXISTS watermarks (
    session_id          TEXT PRIMARY KEY,
    last_processed_ts   REAL NOT NULL
);
"""


class DocumentStore:
    def __init__(self, db_path: str = _DB_PATH) -> None:
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._db_path = db_path
        self._lock = threading.Lock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._lock:
            conn = self._connect()
            try:
                conn.executescript(_DDL)
                conn.commit()
            finally:
                conn.close()

    def insert_passage(
        self,
        figure_name: str,
        session_id: str,
        text: str,
        doc_type: str,
        significance: float = 0.90,
        ts: Optional[float] = None,
    ) -> str:
        passage_id = str(uuid.uuid4())
        now = time.time()
        ts = ts if ts is not None else now
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    INSERT INTO passages (id, figure_name, session_id, text, doc_type, significance, ts, ingested_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (passage_id, figure_name, session_id, text, doc_type, significance, ts, now),
                )
                conn.commit()
            finally:
                conn.close()
        return passage_id

    def get_passages_since(self, session_id: str, since_ts: float) -> List[sqlite3.Row]:
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    "SELECT * FROM passages WHERE session_id = ? AND ts > ? ORDER BY ts ASC",
                    (session_id, since_ts),
                ).fetchall()
                return list(rows)
            finally:
                conn.close()

    def get_watermark(self, session_id: str) -> float:
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT last_processed_ts FROM watermarks WHERE session_id = ?",
                    (session_id,),
                ).fetchone()
                return row["last_processed_ts"] if row else -1_000_000_000_000.0
            finally:
                conn.close()

    def set_watermark(self, session_id: str, ts: float) -> None:
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    "INSERT INTO watermarks (session_id, last_processed_ts) VALUES (?, ?) "
                    "ON CONFLICT(session_id) DO UPDATE SET last_processed_ts = excluded.last_processed_ts",
                    (session_id, ts),
                )
                conn.commit()
            finally:
                conn.close()

    def count_passages(self, session_id: str) -> int:
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT COUNT(*) as n FROM passages WHERE session_id = ?",
                    (session_id,),
                ).fetchone()
                return row["n"] if row else 0
            finally:
                conn.close()

    def list_figures(self) -> List[sqlite3.Row]:
        with self._lock:
            conn = self._connect()
            try:
                return list(conn.execute(
                    "SELECT figure_name, session_id, doc_type, COUNT(*) as passage_count, "
                    "MIN(ingested_at) as first_ingested "
                    "FROM passages GROUP BY session_id ORDER BY first_ingested DESC"
                ).fetchall())
            finally:
                conn.close()


_instance: Optional[DocumentStore] = None
_instance_lock = threading.Lock()


def get_document_store() -> DocumentStore:
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = DocumentStore()
    return _instance
