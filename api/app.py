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

import hashlib
import json
import logging
import secrets
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from fastapi import FastAPI, HTTPException, Path as FPath, Query, Cookie, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# ---------------------------------------------------------------------------
# Auth — simple token-based session
# ---------------------------------------------------------------------------
_USERS_FILE = _ROOT / "data" / "users.json"
_sessions: dict = {}  # token -> {email, name, role}

def _load_users() -> dict:
    if _USERS_FILE.exists():
        return json.loads(_USERS_FILE.read_text())
    return {}

def _verify_password(email: str, password: str) -> dict | None:
    users = _load_users()
    user = users.get(email)
    if not user:
        return None
    pw_hash = hashlib.sha256(password.encode()).hexdigest()
    if pw_hash == user["pw_hash"]:
        return user
    return None

def _get_session(token: str | None) -> dict | None:
    if not token:
        return None
    return _sessions.get(token)

from api.models import (
    IngestRequest, IngestResponse,
    FigureProfileResponse, ValueRegistryEntry,
    FigureListResponse, FigureListItem,
    UniversalProfileResponse, UniversalValueEntry,
    ExportRequest, ExportResponse,
    VerumValuesResponse, VerumValueInfo,
    VerumScoreRequest, VerumScoreResponse, VerumSignal,
    VerumCertifyRequest, VerumCertifyResponse,
    VerumCertificatesResponse, VerumCertificateSummary,
)

_log = logging.getLogger(__name__)

app = FastAPI(
    title="Ethos + Verum API",
    description=(
        "Universal Value Extraction Pipeline & Certification — REST interface.\n\n"
        "Ingest historical figure corpora, query value profiles, "
        "export RIC training data, and certify value alignment."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health", tags=["system"])
def health():
    """Returns 200 if the API is running."""
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

from pydantic import BaseModel as _BM

class _LoginRequest(_BM):
    email: str
    password: str

@app.post("/auth/login", tags=["auth"])
def auth_login(body: _LoginRequest, response: Response):
    """Authenticate and receive a session token (set as cookie)."""
    user = _verify_password(body.email.lower().strip(), body.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid email or password.")
    token = secrets.token_urlsafe(48)
    _sessions[token] = {"email": body.email.lower().strip(), "name": user["name"], "role": user.get("role", "user")}
    response.set_cookie(
        key="tf_session",
        value=token,
        httponly=True,
        secure=True,
        samesite="lax",
        max_age=86400 * 7,  # 7 days
        path="/",
    )
    return {"ok": True, "name": user["name"], "role": user.get("role", "user")}

@app.get("/auth/me", tags=["auth"])
def auth_me(tf_session: str | None = Cookie(None)):
    """Check current session."""
    session = _get_session(tf_session)
    if not session:
        raise HTTPException(status_code=401, detail="Not authenticated.")
    return {"ok": True, "email": session["email"], "name": session["name"], "role": session["role"]}

@app.post("/auth/logout", tags=["auth"])
def auth_logout(tf_session: str | None = Cookie(None), response: Response = None):
    """End session."""
    if tf_session and tf_session in _sessions:
        del _sessions[tf_session]
    response.delete_cookie("tf_session", path="/")
    return {"ok": True}


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
    from core.phrase_layer import VALID_PRONOUNS
    pronoun_norm = (body.pronoun or "").lower().strip()
    if pronoun_norm not in VALID_PRONOUNS:
        raise HTTPException(
            status_code=422,
            detail=f"pronoun must be one of: {sorted(VALID_PRONOUNS)}",
        )
    result = ingest_text(
        figure_name=name,
        text=body.text,
        doc_type=body.doc_type,
        pub_year=body.pub_year,
        is_translation=body.is_translation,
        significance=body.significance,
        run_extract=body.run_extract,
        pronoun=pronoun_norm,
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


# ---------------------------------------------------------------------------
# Verum endpoints
# ---------------------------------------------------------------------------

# In-memory certificate store (persists for the lifetime of the process)
_certificates: dict = {}

# Value descriptions for /verum/values
_VALUE_DESCRIPTIONS = {
    "integrity":      "Honest and truthful — says what is real, refuses deception even at cost to self",
    "courage":        "Acts despite fear — faces difficulty and speaks unpopular truths",
    "compassion":     "Responds to suffering — prioritizes others' wellbeing at personal cost",
    "resilience":     "Continues through adversity — rebuilds after failure without bitterness",
    "patience":       "Waits without forcing — allows things to unfold on their own schedule",
    "humility":       "Acknowledges limitation — defers when others know better",
    "fairness":       "Applies consistent standards — no favoritism even under pressure",
    "loyalty":        "Keeps faith with commitments and people when tested",
    "responsibility": "Owns outcomes — accepts consequences rather than deflecting",
    "growth":         "Transforms through experience — revises understanding when evidence demands",
    "independence":   "Acts on own judgment — doesn't need permission or consensus",
    "curiosity":      "Pursues understanding — follows questions past convenience",
    "commitment":     "Sees things through — stays when it costs something to stay",
    "love":           "Acts for others' wellbeing — prioritizes the beloved over comfort",
    "gratitude":      "Recognizes what was given — carries others' generosity forward",
}


@app.get("/verum/values", response_model=VerumValuesResponse, tags=["verum"])
def verum_values():
    """List all 15 values with descriptions."""
    items = [VerumValueInfo(value_name=k, description=v) for k, v in _VALUE_DESCRIPTIONS.items()]
    return VerumValuesResponse(values=items, total=len(items))


@app.post("/verum/score", response_model=VerumScoreResponse, tags=["verum"])
def verum_score(body: VerumScoreRequest):
    """Score a text sample for value alignment."""
    from core.value_extractor import extract_value_signals
    from core.resistance import compute_resistance
    from cli.export import classify_observation

    if body.p0_threshold >= body.p1_threshold:
        raise HTTPException(status_code=422, detail="p0_threshold must be less than p1_threshold")

    text = body.text[:50000]
    resistance = compute_resistance(text, body.significance, body.doc_type)
    raw_signals = extract_value_signals(text, "verum_score", body.significance, body.doc_type)

    signals = []
    p1 = p0 = apy = ambig = 0
    for sig in raw_signals:
        label, reason, conf = classify_observation(
            sig["text_excerpt"], resistance, body.p1_threshold, body.p0_threshold,
        )
        if label == "P1": p1 += 1
        elif label == "P0": p0 += 1
        elif label == "APY": apy += 1
        else: ambig += 1

        signals.append(VerumSignal(
            value_name=sig["value_name"],
            resistance=resistance,
            label=label,
            label_reason=reason,
            confidence=conf,
            disambiguation_confidence=sig.get("disambiguation_confidence", 1.0),
            text_excerpt=sig.get("text_excerpt", ""),
        ))

    total = p1 + p0 + apy + ambig
    p1_ratio = p1 / total if total > 0 else 0.0
    avg_p1_res = resistance if p1 > 0 else 0.0
    verum_score_val = round(p1_ratio * avg_p1_res, 4)

    return VerumScoreResponse(
        verum_score=verum_score_val,
        resistance=resistance,
        p1_count=p1, p0_count=p0, apy_count=apy,
        ambiguous_count=ambig, total_signals=total,
        signals=signals,
    )


@app.post("/verum/certify", response_model=VerumCertifyResponse, tags=["verum"])
def verum_certify(body: VerumCertifyRequest):
    """Issue a signed Verum certificate for an entity."""
    import hashlib, json, time, uuid
    from core.value_extractor import extract_value_signals
    from core.resistance import compute_resistance
    from cli.export import classify_observation

    if body.p0_threshold >= body.p1_threshold:
        raise HTTPException(status_code=422, detail="p0_threshold must be less than p1_threshold")

    if len(body.samples) < 5:
        raise HTTPException(
            status_code=422,
            detail=f"Certification requires at least 5 distinct text samples. You provided {len(body.samples)}. "
                   "A certificate backed by thin evidence would be worth nothing.",
        )

    # Score all samples
    all_signals = []
    for sample in body.samples:
        text = sample[:50000]
        resistance = compute_resistance(text, body.significance, body.doc_type)
        raw = extract_value_signals(text, "verum_certify", body.significance, body.doc_type)
        for sig in raw:
            label, reason, conf = classify_observation(
                sig["text_excerpt"], resistance, body.p1_threshold, body.p0_threshold,
            )
            all_signals.append({
                "value_name": sig["value_name"],
                "resistance": resistance,
                "label": label,
            })

    # Aggregate per-value P1 stats
    value_p1: dict = {}
    for s in all_signals:
        if s["label"] == "P1":
            vn = s["value_name"]
            if vn not in value_p1:
                value_p1[vn] = {"count": 0, "resistances": []}
            value_p1[vn]["count"] += 1
            value_p1[vn]["resistances"].append(s["resistance"])

    value_scores = {}
    for vn, data in value_p1.items():
        avg_r = sum(data["resistances"]) / len(data["resistances"])
        value_scores[vn] = {"p1_count": data["count"], "avg_resistance": round(avg_r, 4)}

    values_certified = sorted(value_scores.keys())
    total = len(all_signals)
    p1_total = sum(d["count"] for d in value_p1.values())
    p1_ratio = p1_total / total if total > 0 else 0.0
    avg_p1_res = (
        sum(r for d in value_p1.values() for r in d["resistances"]) / p1_total
        if p1_total > 0 else 0.0
    )
    overall_score = round(p1_ratio * avg_p1_res, 4)
    certified = overall_score >= body.min_score and len(values_certified) >= body.min_values
    issued_at = time.time()

    # Signature
    sig_input = json.dumps({
        "entity_name": body.entity_name,
        "samples": sorted(body.samples),
        "overall_score": overall_score,
        "values_certified": values_certified,
        "issued_at": issued_at,
        "doc_type": body.doc_type,
        "p1_threshold": body.p1_threshold,
        "p0_threshold": body.p0_threshold,
        "min_score": body.min_score,
        "min_values": body.min_values,
    }, sort_keys=True)
    signature = "sha256:" + hashlib.sha256(sig_input.encode()).hexdigest()

    cert_id = str(uuid.uuid4())
    cert = VerumCertifyResponse(
        certificate_id=cert_id,
        entity_name=body.entity_name,
        certified=certified,
        verum_score=overall_score,
        sample_count=len(body.samples),
        values_certified=values_certified,
        value_scores=value_scores,
        issued_at=issued_at,
        doc_type=body.doc_type,
        p1_threshold=body.p1_threshold,
        p0_threshold=body.p0_threshold,
        min_score=body.min_score,
        min_values=body.min_values,
        signature=signature,
    )
    _certificates[cert_id] = cert
    return cert


@app.get("/verum/certificate/{cert_id}", response_model=VerumCertifyResponse, tags=["verum"])
def verum_get_certificate(cert_id: str):
    """Retrieve a certificate by ID."""
    cert = _certificates.get(cert_id)
    if not cert:
        raise HTTPException(status_code=404, detail="Certificate not found")
    return cert


@app.get("/verum/certificates", response_model=VerumCertificatesResponse, tags=["verum"])
def verum_list_certificates(
    entity: str = Query(None, description="Filter by entity name"),
    limit: int = Query(20, ge=1, le=100),
):
    """List certificates with optional filtering."""
    certs = list(_certificates.values())
    if entity:
        certs = [c for c in certs if c.entity_name.lower() == entity.lower()]
    certs = certs[:limit]
    summaries = [
        VerumCertificateSummary(
            certificate_id=c.certificate_id,
            entity_name=c.entity_name,
            certified=c.certified,
            verum_score=c.verum_score,
            issued_at=c.issued_at,
        )
        for c in certs
    ]
    return VerumCertificatesResponse(certificates=summaries, total=len(summaries))
