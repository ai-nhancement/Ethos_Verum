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

def get_significance_distribution(val_db_path: str) -> Dict:
    """
    Significance histogram + summary stats across all figure observations.

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
            """SELECT significance FROM value_observations
               WHERE session_id LIKE 'figure:%'"""
        ).fetchall()
        if not rows:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "median": 0.0,
                    "histogram": {}}

        vals = [float(r["significance"]) for r in rows]
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
        _log.warning("get_significance_distribution failed", exc_info=True)
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "median": 0.0,
                "histogram": {}}
    finally:
        _safe_close(vconn)


# ---------------------------------------------------------------------------
# Corpus quality gate
# ---------------------------------------------------------------------------

# Thresholds — researcher-configurable
CORPUS_MIN_DOCS_CONFIDENT    = 3
CORPUS_MIN_TYPES_CONFIDENT   = 2
CORPUS_MIN_DOCS_PARTIAL      = 2


def get_figure_corpus_quality(val_db_path: str, figure_name: str) -> Dict:
    """
    Return a corpus quality assessment for a single figure.

    Tiers:
      "preliminary"  — 1 document (single source, low confidence)
      "partial"      — 2 documents (better, but potentially skewed)
      "confident"    — 3+ documents across ≥2 doc_types (approved for export)

    Return dict:
      {
        figure_name, document_count, doc_types, distinct_doc_type_count,
        confidence_tier, approved_for_export,
        documents: [{doc_title, doc_type, passage_count, ingested_at}],
        notes: [str],   # human-readable warnings
      }
    """
    try:
        vconn = _open(val_db_path)
        rows = vconn.execute(
            """SELECT doc_title, doc_type, passage_count, ingested_at
               FROM figure_documents
               WHERE figure_name=?
               ORDER BY ingested_at""",
            (figure_name,),
        ).fetchall()

        doc_count        = len(rows)
        doc_types        = sorted({r["doc_type"] for r in rows})
        distinct_types   = len(doc_types)
        documents        = [dict(r) for r in rows]

        if doc_count >= CORPUS_MIN_DOCS_CONFIDENT and distinct_types >= CORPUS_MIN_TYPES_CONFIDENT:
            tier     = "confident"
            approved = True
        elif doc_count >= CORPUS_MIN_DOCS_PARTIAL:
            tier     = "partial"
            approved = False
        else:
            tier     = "preliminary"
            approved = False

        notes: list = []
        if doc_count == 0:
            notes.append("No documents tracked. Re-ingest with --doc-title to enable tracking.")
        elif tier == "preliminary":
            need_docs  = CORPUS_MIN_DOCS_CONFIDENT - doc_count
            need_types = max(0, CORPUS_MIN_TYPES_CONFIDENT - distinct_types)
            notes.append(
                f"Need {need_docs} more document(s)"
                + (f" and {need_types} more doc-type(s)" if need_types else "")
                + " before export is approved."
            )
        elif tier == "partial":
            need_docs  = CORPUS_MIN_DOCS_CONFIDENT - doc_count
            need_types = max(0, CORPUS_MIN_TYPES_CONFIDENT - distinct_types)
            parts = []
            if need_docs > 0:
                parts.append(f"{need_docs} more document(s)")
            if need_types > 0:
                parts.append(f"{need_types} more doc-type(s)")
            if parts:
                notes.append("Need " + " and ".join(parts) + " before export is approved.")

        return {
            "figure_name":           figure_name,
            "document_count":        doc_count,
            "doc_types":             doc_types,
            "distinct_doc_type_count": distinct_types,
            "confidence_tier":       tier,
            "approved_for_export":   approved,
            "documents":             documents,
            "notes":                 notes,
        }
    except Exception:
        _log.warning("get_figure_corpus_quality failed for %r", figure_name, exc_info=True)
        return {
            "figure_name": figure_name, "document_count": 0, "doc_types": [],
            "distinct_doc_type_count": 0, "confidence_tier": "preliminary",
            "approved_for_export": False, "documents": [], "notes": [],
        }
    finally:
        _safe_close(vconn)


def get_all_corpus_quality(val_db_path: str) -> List[Dict]:
    """
    Return corpus quality for every figure that has any ingested documents,
    sorted by document_count DESC.
    """
    try:
        vconn = _open(val_db_path)
        figures = [
            r[0] for r in vconn.execute(
                "SELECT DISTINCT figure_name FROM figure_documents ORDER BY figure_name"
            ).fetchall()
        ]
        # Also include figures in figure_sources with no document rows (legacy ingests)
        legacy = [
            r[0] for r in vconn.execute(
                """SELECT DISTINCT figure_name FROM figure_sources
                   WHERE figure_name NOT IN (
                       SELECT DISTINCT figure_name FROM figure_documents
                   )"""
            ).fetchall()
        ]
        _safe_close(vconn)
        results = [get_figure_corpus_quality(val_db_path, f) for f in figures + legacy]
        return sorted(results, key=lambda r: -r["document_count"])
    except Exception:
        _log.warning("get_all_corpus_quality failed", exc_info=True)
        return []


def get_full_report(doc_db_path: str, val_db_path: str) -> Dict:
    """
    Return all corpus statistics in one call. Used by CLI and API.
    """
    return {
        "overview":               get_overview(doc_db_path, val_db_path),
        "figures":                get_figure_summaries(doc_db_path, val_db_path),
        "corpus_quality":         get_all_corpus_quality(val_db_path),
        "value_distribution":     get_value_distribution(val_db_path),
        "resistance":             get_resistance_distribution(val_db_path),
        "significance":           get_significance_distribution(val_db_path),
        "cross_figure_values":    get_cross_figure_values(val_db_path, min_figures=2),
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
