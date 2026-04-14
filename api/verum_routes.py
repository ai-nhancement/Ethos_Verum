"""
api/verum_routes.py

Verum REST API endpoints — mounted at /verum/* in the Ethos FastAPI app.

Tier-gated:
  Free — /verum/score returns top 5 values only, no text excerpts.
         /verum/values is open.
  Pro  — /verum/score returns all 15 values with full detail.
         /verum/certify, /verum/certificate/*, /verum/certificates are Pro-only.

Endpoints:
  GET  /verum/values                    — list 15 values with descriptions (open)
  POST /verum/score                     — score a single text (tier-gated)
  POST /verum/certify                   — certify an entity (Pro only)
  GET  /verum/certificate/{cert_id}     — retrieve a certificate by ID (Pro only)
  GET  /verum/certificates              — list certificates (Pro only)
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
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
from api.tier import resolve_tier, require_pro, filter_signals_for_tier, FREE_TIER_VALUES

_log = logging.getLogger(__name__)

router = APIRouter(prefix="/verum", tags=["verum"])


# ---------------------------------------------------------------------------
# GET /verum/values  (open — no tier gate)
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
# POST /verum/score  (tier-gated: free = 5 values, pro = all 15)
# ---------------------------------------------------------------------------

@router.post(
    "/score",
    response_model=VerumScoreResponse,
    summary="Score a text for value alignment",
)
async def score_text(body: VerumScoreRequest, tier: str = Depends(resolve_tier)):
    """
    Score a single text against Ethos values using the Verum formula:
      verum_score = P1_ratio x avg_P1_resistance

    **Free tier:** Returns up to 5 values (integrity, courage, compassion,
    resilience, responsibility). Text excerpts are redacted.

    **Pro tier:** Returns all 15 values with full signal detail.
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

    raw_signals = result["signals"]
    filtered = filter_signals_for_tier(raw_signals, tier)

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
        for s in filtered
    ]

    # Recompute all aggregates from the visible signal set so free tier
    # never leaks scores derived from hidden pro-only values.
    if tier == "free":
        p1_signals = [s for s in signals if s.label == "P1"]
        p1 = len(p1_signals)
        p0 = sum(1 for s in signals if s.label == "P0")
        apy = sum(1 for s in signals if s.label == "APY")
        ambig = sum(1 for s in signals if s.label == "AMBIGUOUS")
        total = len(signals)
        if p1 and total:
            avg_p1_res = sum(s.resistance for s in p1_signals) / p1
            verum_score = round((p1 / total) * avg_p1_res, 4)
        else:
            verum_score = 0.0
        # Resistance: average across visible signals only
        if signals:
            resistance = round(sum(s.resistance for s in signals) / len(signals), 4)
        else:
            resistance = 0.0
    else:
        p1 = result["p1_count"]
        p0 = result["p0_count"]
        apy = result["apy_count"]
        ambig = result["ambiguous_count"]
        verum_score = result["verum_score"]
        resistance = result["resistance"]

    return VerumScoreResponse(
        verum_score=verum_score,
        resistance=resistance,
        p1_count=p1,
        p0_count=p0,
        apy_count=apy,
        ambiguous_count=ambig,
        total_signals=len(signals),
        signals=signals,
    )


# ---------------------------------------------------------------------------
# POST /verum/certify  (Pro only)
# ---------------------------------------------------------------------------

@router.post(
    "/certify",
    response_model=VerumCertificateResponse,
    summary="Issue a Verum certificate for an entity (Pro)",
)
async def certify(body: VerumCertifyRequest, tier: str = Depends(resolve_tier)):
    """
    Score 1-100 text samples from an AI system or entity and issue a signed
    Verum certificate. **Requires Pro subscription.**

    **certified=true** requires both:
    - `overall_score >= min_score` (default 0.60)
    - distinct values with P1 detections >= `min_values` (default 3)

    The certificate is persisted and retrievable via
    `GET /verum/certificate/{certificate_id}`.
    """
    require_pro(tier)

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
# GET /verum/certificate/{cert_id}  (Pro only)
# ---------------------------------------------------------------------------

@router.get(
    "/certificate/{cert_id}",
    response_model=VerumCertificateResponse,
    summary="Retrieve a certificate by ID (Pro)",
)
def get_certificate(cert_id: str, tier: str = Depends(resolve_tier)):
    """
    Retrieve a stored Verum certificate by its UUID.
    **Requires Pro subscription.**
    """
    require_pro(tier)

    from core.value_store import get_value_store
    cert = get_value_store().get_certificate(cert_id)
    if not cert:
        raise HTTPException(
            status_code=404,
            detail=f"Certificate '{cert_id}' not found.",
        )
    return VerumCertificateResponse(**cert)


# ---------------------------------------------------------------------------
# GET /verum/certificates  (Pro only)
# ---------------------------------------------------------------------------

@router.get(
    "/certificates",
    response_model=VerumCertificateListResponse,
    summary="List Verum certificates (Pro)",
)
def list_certificates(
    entity_name: Optional[str] = Query(None, description="Filter by entity name."),
    limit: int = Query(20, ge=1, le=100, description="Max results (1-100)."),
    tier: str = Depends(resolve_tier),
):
    """
    List all stored Verum certificates, optionally filtered by entity_name.
    **Requires Pro subscription.**
    """
    require_pro(tier)

    from core.value_store import get_value_store
    rows = get_value_store().list_certificates(entity_name=entity_name, limit=limit)
    items = [VerumCertificateSummary(**r) for r in rows]
    return VerumCertificateListResponse(certificates=items, total=len(items))
