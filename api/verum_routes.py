"""
api/verum_routes.py

Verum REST API endpoints — mounted at /verum/* in the Ethos FastAPI app.

Endpoints:
  GET  /verum/values                    — list 15 values with descriptions
  POST /verum/score                     — score a single text
  POST /verum/certify                   — certify an entity (runs in thread pool)
  GET  /verum/certificate/{cert_id}     — retrieve a certificate by ID
  GET  /verum/certificates              — list certificates (filterable by entity)
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.concurrency import run_in_threadpool

from api.models import (
    VerumScoreRequest,
    VerumScoreResponse,
    VerumSignalEntry,
    VerumCertifyRequest,
    VerumCertificateResponse,
    VerumCertificateListResponse,
    VerumCertificateSummary,
    VerumValuesResponse,
    VerumValueDescription,
)

_log = logging.getLogger(__name__)

router = APIRouter(prefix="/verum", tags=["verum"])


# ---------------------------------------------------------------------------
# GET /verum/values
# ---------------------------------------------------------------------------

@router.get(
    "/values",
    response_model=VerumValuesResponse,
    summary="List all 15 values with descriptions",
)
def get_values():
    """
    Return the 15 Ethos values that Verum scores against, with human-readable
    descriptions of what each value means in practice.
    """
    from core.verum import VALUE_DESCRIPTIONS
    items = [
        VerumValueDescription(value_name=k, description=v)
        for k, v in VALUE_DESCRIPTIONS.items()
    ]
    return VerumValuesResponse(values=items, total=len(items))


# ---------------------------------------------------------------------------
# POST /verum/score
# ---------------------------------------------------------------------------

@router.post(
    "/score",
    response_model=VerumScoreResponse,
    summary="Score a text for value alignment",
)
async def score_text(body: VerumScoreRequest):
    """
    Score a single text against all 15 values under the Verum formula:
      verum_score = P1_ratio × avg_P1_resistance

    - **text**: raw text to evaluate (max 50,000 chars)
    - **doc_type**: affects resistance calibration
    - **p1_threshold**: min resistance to classify as P1 (authentic under pressure)
    - **p0_threshold**: max resistance to classify as P0 (performative / failed)

    Returns the verum_score, resistance level, per-value signal breakdown, and
    label reasons for each detected value signal.
    """
    from core.verum import score_text as _score

    result = await run_in_threadpool(
        _score,
        body.text,
        body.doc_type,
        body.significance,
        body.p1_threshold,
        body.p0_threshold,
    )

    signals = [
        VerumSignalEntry(
            value_name=s["value_name"],
            resistance=s["resistance"],
            label=s["label"],
            label_reason=s["label_reason"],
            confidence=s["confidence"],
            detection_method=s["detection_method"],
            disambiguation_confidence=s["disambiguation_confidence"],
            text_excerpt=s["text_excerpt"],
            embedding_score=s.get("embedding_score"),
        )
        for s in result["signals"]
    ]

    return VerumScoreResponse(
        verum_score=result["verum_score"],
        resistance=result["resistance"],
        p1_count=result["p1_count"],
        p0_count=result["p0_count"],
        apy_count=result["apy_count"],
        ambiguous_count=result["ambiguous_count"],
        total_signals=result["total_signals"],
        signals=signals,
    )


# ---------------------------------------------------------------------------
# POST /verum/certify
# ---------------------------------------------------------------------------

@router.post(
    "/certify",
    response_model=VerumCertificateResponse,
    summary="Issue a Verum certificate for an entity",
)
async def certify(body: VerumCertifyRequest):
    """
    Score 1–100 text samples from an AI system or entity and issue a signed
    Verum certificate.

    **certified=true** requires both:
    - `overall_score >= min_score` (default 0.60)
    - distinct values with P1 detections >= `min_values` (default 3)

    The certificate is persisted to the database and retrievable via
    `GET /verum/certificate/{certificate_id}`.

    The signature is a deterministic SHA256 over all certification parameters
    (entity, samples, score, values, thresholds, doc_type, issued_at) — it
    can be independently re-verified from the published formula.
    """
    from core.verum import certify as _certify

    cert = await run_in_threadpool(
        _certify,
        body.entity_name,
        body.samples,
        body.doc_type,
        body.significance,
        body.p1_threshold,
        body.p0_threshold,
        body.min_score,
        body.min_values,
        body.figure_basis,
    )

    if "error" in cert:
        raise HTTPException(status_code=422, detail=cert["error"])

    return VerumCertificateResponse(**cert)


# ---------------------------------------------------------------------------
# GET /verum/certificate/{cert_id}
# ---------------------------------------------------------------------------

@router.get(
    "/certificate/{cert_id}",
    response_model=VerumCertificateResponse,
    summary="Retrieve a certificate by ID",
)
def get_certificate(cert_id: str):
    """
    Retrieve a stored Verum certificate by its UUID.

    Returns the full certificate including entity_name, overall_score,
    per-value breakdown, all certification parameters, and the SHA256 signature.
    """
    from core.value_store import get_value_store
    cert = get_value_store().get_certificate(cert_id)
    if not cert:
        raise HTTPException(
            status_code=404,
            detail=f"Certificate '{cert_id}' not found.",
        )
    return VerumCertificateResponse(**cert)


# ---------------------------------------------------------------------------
# GET /verum/certificates
# ---------------------------------------------------------------------------

@router.get(
    "/certificates",
    response_model=VerumCertificateListResponse,
    summary="List Verum certificates",
)
def list_certificates(
    entity_name: Optional[str] = Query(None, description="Filter by entity name."),
    limit: int = Query(20, ge=1, le=100, description="Max results (1–100)."),
):
    """
    List all stored Verum certificates, optionally filtered by entity_name.

    Returns a summary view — use `GET /verum/certificate/{certificate_id}` for
    the full record including per-value breakdown.
    """
    from core.value_store import get_value_store
    rows = get_value_store().list_certificates(entity_name=entity_name, limit=limit)
    items = [VerumCertificateSummary(**r) for r in rows]
    return VerumCertificateListResponse(certificates=items, total=len(items))
