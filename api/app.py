"""
api/app.py

Ethos REST API — FastAPI application.

Endpoints:
  POST /figures/{name}/ingest   — ingest text for a figure
  GET  /figures                 — list all ingested figures
  GET  /figures/{name}/profile  — value registry for a figure
  GET  /figures/universal       — cross-figure aggregate registry
  POST /export/ric              — trigger RIC export, return report

Run with:
  python -m api.server          (development)
  uvicorn api.app:app --port 8000 --reload
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from fastapi import FastAPI, HTTPException, Path as FPath, Query
from fastapi.responses import JSONResponse

from api.models import (
    IngestRequest, IngestResponse,
    FigureProfileResponse, ValueRegistryEntry,
    FigureListResponse, FigureListItem,
    UniversalProfileResponse, UniversalValueEntry,
    ExportRequest, ExportResponse,
)

_log = logging.getLogger(__name__)

app = FastAPI(
    title="Ethos API",
    description=(
        "Universal Value Extraction Pipeline — REST interface.\n\n"
        "Ingest historical figure corpora, query value profiles, "
        "and export RIC training data."
    ),
    version="1.0.0",
)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health", tags=["system"])
def health():
    """Returns 200 if the API is running."""
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# POST /figures/{name}/ingest
# ---------------------------------------------------------------------------

@app.post(
    "/figures/{name}/ingest",
    response_model=IngestResponse,
    tags=["figures"],
    summary="Ingest text for a historical figure",
)
def ingest_figure(
    name: str = FPath(..., description="Figure identifier, e.g. 'gandhi', 'lincoln'"),
    body: IngestRequest = ...,
):
    """
    Segment, store, and extract value signals from raw text for a figure.

    - **name**: figure identifier (alphanumeric, underscores, hyphens, 1–64 chars)
    - **text**: raw UTF-8 source text
    - **doc_type**: document type affects resistance scoring
    - **pub_year**: publication year — triggers archaic preprocessing and temporal discount
    - **is_translation**: `true` = known translation (0.85×), `null` = auto-detect
    """
    from core.pipeline import ingest_text
    result = ingest_text(
        figure_name=name,
        text=body.text,
        doc_type=body.doc_type,
        pub_year=body.pub_year,
        is_translation=body.is_translation,
        significance=body.significance,
        run_extract=body.run_extract,
    )
    if not result.ok:
        raise HTTPException(status_code=422, detail=result.error)
    return IngestResponse(
        figure=result.figure_name,
        session_id=result.session_id,
        passages_ingested=result.passages_ingested,
        observations_recorded=result.observations_recorded,
        source_lang=result.source_lang,
        source_authenticity=result.source_authenticity,
        pub_year=result.pub_year,
        ok=True,
    )


# ---------------------------------------------------------------------------
# GET /figures
# ---------------------------------------------------------------------------

@app.get(
    "/figures",
    response_model=FigureListResponse,
    tags=["figures"],
    summary="List all ingested figures",
)
def list_figures():
    """Return all figures that have been ingested into the pipeline."""
    from core.value_store import get_value_store
    rows = get_value_store().get_figures_list()
    items = [
        FigureListItem(
            figure_name=r.get("figure_name", ""),
            session_id=r.get("session_id", ""),
            doc_type=r.get("document_type", "unknown"),
            passage_count=r.get("passage_count", 0),
            first_ingested=r.get("ingested_at"),
        )
        for r in rows
    ]
    return FigureListResponse(figures=items, total=len(items))


# ---------------------------------------------------------------------------
# GET /figures/universal  (must be declared BEFORE /figures/{name}/profile
#                          to avoid route shadowing)
# ---------------------------------------------------------------------------

@app.get(
    "/figures/universal",
    response_model=UniversalProfileResponse,
    tags=["figures"],
    summary="Cross-figure universal value registry",
)
def universal_profile(
    min_demonstrations: int = Query(1, ge=1, description="Min demonstrations to include"),
):
    """
    Aggregate value registry across all ingested figures.
    Shows which values appear most consistently across the corpus.
    """
    from core.pipeline import universal_profile as _up
    from core.value_store import get_value_store

    rows = _up(min_demonstrations=min_demonstrations)
    figure_count = len(get_value_store().get_figures_list())

    entries = [
        UniversalValueEntry(
            value_name=r.get("value_name", ""),
            total_demonstrations=r.get("total_demonstrations", 0),
            figure_count=r.get("figure_count", 0),
            avg_weight=round(float(r.get("avg_weight", 0.0)), 4),
        )
        for r in rows
    ]
    return UniversalProfileResponse(values=entries, figure_count=figure_count)


# ---------------------------------------------------------------------------
# GET /figures/{name}/profile
# ---------------------------------------------------------------------------

@app.get(
    "/figures/{name}/profile",
    response_model=FigureProfileResponse,
    tags=["figures"],
    summary="Value registry for a specific figure",
)
def figure_profile(
    name: str = FPath(..., description="Figure identifier"),
    min_demonstrations: int = Query(1, ge=1),
):
    """
    Return the value registry for a figure — all values observed,
    sorted by weight (demonstrations × significance × resistance × consistency).
    """
    from core.pipeline import figure_profile as _fp
    rows = _fp(figure_name=name, min_demonstrations=min_demonstrations)
    if not rows:
        raise HTTPException(
            status_code=404,
            detail=f"No profile found for figure '{name}'. Has it been ingested?",
        )
    entries = [
        ValueRegistryEntry(
            value_name=r.get("value_name", ""),
            demonstrations=r.get("demonstrations", 0),
            avg_significance=round(float(r.get("avg_significance", 0)), 4),
            avg_resistance=round(float(r.get("avg_resistance", 0)), 4),
            consistency=round(float(r.get("consistency", 0)), 4),
            weight=round(float(r.get("weight", 0)), 4),
            first_seen_ts=r.get("first_seen_ts"),
            last_seen_ts=r.get("last_seen_ts"),
        )
        for r in rows
    ]
    return FigureProfileResponse(
        figure=name,
        session_id=f"figure:{name.lower()}",
        values=entries,
    )


# ---------------------------------------------------------------------------
# POST /export/ric
# ---------------------------------------------------------------------------

@app.post(
    "/export/ric",
    response_model=ExportResponse,
    tags=["export"],
    summary="Export RIC training data",
)
def export_ric(body: ExportRequest = ExportRequest()):
    """
    Classify value observations as P1 / P0 / APY and write JSONL training files.

    Returns a summary report. Pass `dry_run=true` to get stats without writing files.
    """
    try:
        result = _run_export(body)
        return result
    except Exception as exc:
        _log.exception("export_ric failed")
        raise HTTPException(status_code=500, detail=str(exc))


def _run_export(req: ExportRequest) -> ExportResponse:
    """Invoke the CLI export logic programmatically."""
    import subprocess, json, os

    cmd = [sys.executable, "-m", "cli.export",
           "--p1-threshold", str(req.p1_threshold),
           "--p0-threshold", str(req.p0_threshold),
           "--min-observations", str(req.min_observations),
           "--min-consistency", str(req.min_consistency),
           "--output-dir", req.output_dir,
           ]
    if req.figure:
        cmd += ["--figure", req.figure]
    if req.dry_run:
        cmd.append("--dry-run")

    proc = subprocess.run(
        cmd,
        cwd=str(_ROOT),
        capture_output=True,
        text=True,
        timeout=120,
    )

    if proc.returncode != 0:
        raise RuntimeError(f"Export failed: {proc.stderr[:500]}")

    # Parse report JSON if it was written
    report_path = Path(req.output_dir) / "ric_historical_report.json"
    if not report_path.is_absolute():
        report_path = _ROOT / report_path

    files_written: list[str] = []
    p1 = p0 = apy = ambig = total = 0

    if report_path.exists() and not req.dry_run:
        try:
            data = json.loads(report_path.read_text())
            p1    = data.get("p1_count",        0)
            p0    = data.get("p0_count",        0)
            apy   = data.get("apy_count",       0)
            ambig = data.get("ambiguous_count", 0)
            total = data.get("total_count",     0)
            files_written = data.get("files_written", [])
        except Exception:
            pass
    else:
        # dry-run: parse stdout for counts
        for line in proc.stdout.splitlines():
            line = line.strip()
            if line.startswith("P1:"):
                try: p1 = int(line.split()[1])
                except Exception: pass
            elif line.startswith("P0:"):
                try: p0 = int(line.split()[1])
                except Exception: pass
            elif line.startswith("APY:"):
                try: apy = int(line.split()[1])
                except Exception: pass
        total = p1 + p0 + apy

    return ExportResponse(
        ok=True,
        figure=req.figure,
        p1_count=p1, p0_count=p0, apy_count=apy,
        ambiguous_count=ambig, total_count=total,
        output_dir=None if req.dry_run else req.output_dir,
        files_written=files_written,
    )
