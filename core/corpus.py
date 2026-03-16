"""
core/corpus.py

Corpus-level statistics for the Ethos pipeline.

Queries both DocumentStore (documents.db) and ValueStore (values.db) and
returns structured dicts. Used by cli/corpus_stats.py and the REST API.

All public functions never raise — they return empty structures on error.
"""

from __future__ import annotations

import logging
import sqlite3
from typing import Dict, List, Optional

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------------

def get_overview(doc_db_path: str, val_db_path: str) -> Dict:
    """
    Return top-level corpus counts.

    {
      figure_count, total_passages, total_observations, unique_values,
      coverage_rate,   # fraction of passages that produced ≥1 observation
      doc_types,       # {doc_type: passage_count}
      ingested_range,  # [earliest_ts, latest_ts] or None
    }
    """
    try:
        vconn = _open(val_db_path)
        dconn = _open(doc_db_path)

        figure_count = _scalar(vconn,
            "SELECT COUNT(*) FROM figure_sources")
        total_passages = _scalar(dconn,
            "SELECT COUNT(*) FROM passages WHERE session_id LIKE 'figure:%'")
        total_obs = _scalar(vconn,
            "SELECT COUNT(*) FROM value_observations WHERE session_id LIKE 'figure:%'")
        unique_values = _scalar(vconn,
            """SELECT COUNT(DISTINCT value_name) FROM value_registry
               WHERE session_id LIKE 'figure:%'""")

        # Coverage: passages that generated at least one observation
        covered = _scalar(vconn,
            """SELECT COUNT(DISTINCT record_id) FROM value_observations
               WHERE session_id LIKE 'figure:%'""")
        coverage_rate = round(covered / total_passages, 4) if total_passages else 0.0

        doc_type_rows = dconn.execute(
            """SELECT doc_type, COUNT(*) AS n FROM passages
               WHERE session_id LIKE 'figure:%'
               GROUP BY doc_type ORDER BY n DESC"""
        ).fetchall()
        doc_types = {r["doc_type"]: r["n"] for r in doc_type_rows}

        range_row = vconn.execute(
            "SELECT MIN(ingested_at), MAX(ingested_at) FROM figure_sources"
        ).fetchone()
        ingested_range = (
            [range_row[0], range_row[1]] if range_row and range_row[0] else None
        )

        return {
            "figure_count":      figure_count,
            "total_passages":    total_passages,
            "total_observations": total_obs,
            "unique_values":     unique_values,
            "coverage_rate":     coverage_rate,
            "doc_types":         doc_types,
            "ingested_range":    ingested_range,
        }
    except Exception:
        _log.warning("get_overview failed", exc_info=True)
        return {
            "figure_count": 0, "total_passages": 0,
            "total_observations": 0, "unique_values": 0,
            "coverage_rate": 0.0, "doc_types": {}, "ingested_range": None,
        }
    finally:
        _safe_close(vconn); _safe_close(dconn)


# ---------------------------------------------------------------------------
# Per-figure summaries
# ---------------------------------------------------------------------------

def get_figure_summaries(doc_db_path: str, val_db_path: str) -> List[Dict]:
    """
    Return one dict per ingested figure, sorted by passage_count DESC.

    Each dict: {
      figure_name, session_id, doc_type, passage_count,
      observations, unique_values,
      top_values: [{value_name, demonstrations, weight}],
      avg_resistance, avg_significance,
    }
    """
    try:
        vconn = _open(val_db_path)
        dconn = _open(doc_db_path)

        figures = vconn.execute(
            "SELECT * FROM figure_sources ORDER BY passage_count DESC"
        ).fetchall()

        result = []
        for f in figures:
            sid = f["session_id"]
            obs = _scalar(vconn,
                "SELECT COUNT(*) FROM value_observations WHERE session_id=?", sid)
            uvals = _scalar(vconn,
                """SELECT COUNT(DISTINCT value_name) FROM value_registry
                   WHERE session_id=?""", sid)

            top_rows = vconn.execute(
                """SELECT value_name, demonstrations, weight
                   FROM value_registry WHERE session_id=?
                   ORDER BY weight DESC LIMIT 5""",
                (sid,),
            ).fetchall()
            top_values = [dict(r) for r in top_rows]

            avg_res = _scalar(vconn,
                "SELECT AVG(resistance) FROM value_observations WHERE session_id=?", sid)
            avg_sig = _scalar(vconn,
                "SELECT AVG(significance) FROM value_observations WHERE session_id=?", sid)

            # actual passage count from document store
            actual_passages = _scalar(dconn,
                "SELECT COUNT(*) FROM passages WHERE session_id=?", sid)

            result.append({
                "figure_name":    f["figure_name"],
                "session_id":     sid,
                "doc_type":       f["document_type"],
                "passage_count":  actual_passages,
                "observations":   obs,
                "unique_values":  uvals,
                "top_values":     top_values,
                "avg_resistance": round(float(avg_res or 0), 4),
                "avg_significance": round(float(avg_sig or 0), 4),
            })

        return result
    except Exception:
        _log.warning("get_figure_summaries failed", exc_info=True)
        return []
    finally:
        _safe_close(vconn); _safe_close(dconn)


# ---------------------------------------------------------------------------
# Value distribution
# ---------------------------------------------------------------------------

def get_value_distribution(val_db_path: str) -> List[Dict]:
    """
    Return cross-figure value distribution, sorted by total_demonstrations DESC.

    Each dict: {
      value_name, total_demonstrations, figure_count,
      avg_weight, avg_resistance, avg_significance,
    }
    """
    try:
        vconn = _open(val_db_path)
        rows = vconn.execute(
            """SELECT
                   value_name,
                   SUM(demonstrations)        AS total_demonstrations,
                   COUNT(DISTINCT session_id) AS figure_count,
                   AVG(weight)                AS avg_weight,
                   AVG(avg_resistance)        AS avg_resistance,
                   AVG(avg_significance)      AS avg_significance
               FROM value_registry
               WHERE session_id LIKE 'figure:%'
               GROUP BY value_name
               ORDER BY total_demonstrations DESC"""
        ).fetchall()
        return [
            {
                "value_name":           r["value_name"],
                "total_demonstrations": r["total_demonstrations"],
                "figure_count":         r["figure_count"],
                "avg_weight":           round(float(r["avg_weight"] or 0), 4),
                "avg_resistance":       round(float(r["avg_resistance"] or 0), 4),
                "avg_significance":     round(float(r["avg_significance"] or 0), 4),
            }
            for r in rows
        ]
    except Exception:
        _log.warning("get_value_distribution failed", exc_info=True)
        return []
    finally:
        _safe_close(vconn)


# ---------------------------------------------------------------------------
# Resistance distribution
# ---------------------------------------------------------------------------

def get_resistance_distribution(val_db_path: str) -> Dict:
    """
    Resistance histogram + summary stats across all figure observations.

    {
      mean, std, min, max, median,
      histogram: {
        "0.0-0.2": n, "0.2-0.4": n, "0.4-0.6": n, "0.6-0.8": n, "0.8-1.0": n
      }
    }
    """
    try:
        vconn = _open(val_db_path)
        rows = vconn.execute(
            """SELECT resistance FROM value_observations
               WHERE session_id LIKE 'figure:%'"""
        ).fetchall()
        if not rows:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "median": 0.0,
                    "histogram": {}}

        vals = [float(r["resistance"]) for r in rows]
        n = len(vals)
        mean = sum(vals) / n
        variance = sum((v - mean) ** 2 for v in vals) / n
        std = variance ** 0.5
        sorted_vals = sorted(vals)
        mid = n // 2
        median = sorted_vals[mid] if n % 2 else (sorted_vals[mid-1] + sorted_vals[mid]) / 2

        buckets = {"0.0-0.2": 0, "0.2-0.4": 0, "0.4-0.6": 0, "0.6-0.8": 0, "0.8-1.0": 0}
        for v in vals:
            if   v < 0.2: buckets["0.0-0.2"] += 1
            elif v < 0.4: buckets["0.2-0.4"] += 1
            elif v < 0.6: buckets["0.4-0.6"] += 1
            elif v < 0.8: buckets["0.6-0.8"] += 1
            else:         buckets["0.8-1.0"] += 1

        return {
            "mean":      round(mean, 4),
            "std":       round(std, 4),
            "min":       round(min(vals), 4),
            "max":       round(max(vals), 4),
            "median":    round(median, 4),
            "histogram": buckets,
        }
    except Exception:
        _log.warning("get_resistance_distribution failed", exc_info=True)
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "median": 0.0,
                "histogram": {}}
    finally:
        _safe_close(vconn)


# ---------------------------------------------------------------------------
# Cross-figure values (values that appear in ≥ N figures)
# ---------------------------------------------------------------------------

def get_cross_figure_values(val_db_path: str, min_figures: int = 2) -> List[Dict]:
    """
    Return values that appear in at least min_figures distinct figures.
    These are the candidates for the universal value set.
    """
    try:
        vconn = _open(val_db_path)
        rows = vconn.execute(
            """SELECT
                   value_name,
                   COUNT(DISTINCT session_id) AS figure_count,
                   SUM(demonstrations)        AS total_demonstrations,
                   AVG(weight)                AS avg_weight
               FROM value_registry
               WHERE session_id LIKE 'figure:%'
               GROUP BY value_name
               HAVING COUNT(DISTINCT session_id) >= ?
               ORDER BY figure_count DESC, total_demonstrations DESC""",
            (min_figures,),
        ).fetchall()
        return [
            {
                "value_name":           r["value_name"],
                "figure_count":         r["figure_count"],
                "total_demonstrations": r["total_demonstrations"],
                "avg_weight":           round(float(r["avg_weight"] or 0), 4),
            }
            for r in rows
        ]
    except Exception:
        _log.warning("get_cross_figure_values failed", exc_info=True)
        return []
    finally:
        _safe_close(vconn)


# ---------------------------------------------------------------------------
# Full report (all sections combined)
# ---------------------------------------------------------------------------

def get_full_report(doc_db_path: str, val_db_path: str) -> Dict:
    """
    Return all corpus statistics in one call. Used by CLI and API.
    """
    return {
        "overview":             get_overview(doc_db_path, val_db_path),
        "figures":              get_figure_summaries(doc_db_path, val_db_path),
        "value_distribution":   get_value_distribution(val_db_path),
        "resistance":           get_resistance_distribution(val_db_path),
        "cross_figure_values":  get_cross_figure_values(val_db_path, min_figures=2),
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _open(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def _scalar(conn: sqlite3.Connection, sql: str, *params) -> int | float:
    row = conn.execute(sql, params).fetchone()
    if row is None:
        return 0
    v = row[0]
    return v if v is not None else 0


def _safe_close(conn) -> None:
    try:
        if conn:
            conn.close()
    except Exception:
        pass
