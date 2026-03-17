"""
core/pipeline.py

Shared ingestion + extraction logic for both the CLI and the REST API.

ingest_text()  — ingest raw text for a figure, return IngestResult.
figure_profile() — get value registry for a figure, return list of dicts.

All heavy lifting stays here; callers (cli/ingest.py, api/) just handle
I/O and presentation.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import List, Optional

_log = logging.getLogger(__name__)

_MAX_PASSAGE_CHARS   = 450
_DEFAULT_SIGNIFICANCE = 0.75  # fallback when doc_type is unrecognised
_FIGURE_NAME_RE      = re.compile(r'^[a-zA-Z0-9][a-zA-Z0-9_\-]{0,63}$')
_SENTENCE_SPLIT_RE   = re.compile(r'(?<=[.!?])\s+')


@dataclass
class IngestResult:
    figure_name:         str
    session_id:          str
    passages_ingested:   int
    observations_recorded: int
    source_lang:         str
    source_authenticity: float
    pub_year:            Optional[int]
    error:               Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None


def _segment(text: str, max_chars: int = _MAX_PASSAGE_CHARS) -> List[str]:
    sentences = _SENTENCE_SPLIT_RE.split(text.strip())
    passages: List[str] = []
    current = ""
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        if current and len(current) + 1 + len(sent) > max_chars:
            passages.append(current.strip())
            current = sent
        else:
            current = (current + " " + sent).strip() if current else sent
    if current.strip():
        passages.append(current.strip())
    return [p for p in passages if len(p) >= 30]


def ingest_text(
    figure_name: str,
    text: str,
    doc_type: str = "unknown",
    pub_year: Optional[int] = None,
    is_translation: Optional[bool] = None,
    significance: Optional[float] = None,
    run_extract: bool = True,
) -> IngestResult:
    """
    Ingest raw text for a historical figure.

    Handles language detection, archaic preprocessing, segmentation,
    DB insertion, and extraction. Returns an IngestResult.

    When significance is None (the default), it is auto-inferred from
    doc_type using DOC_TYPE_DEFAULT_SIGNIFICANCE — e.g., 'action' → 0.90,
    'speech' → 0.70. Pass an explicit value to override.

    Never raises.
    """
    # Auto-infer significance from doc_type when caller does not supply one
    if significance is None:
        from core.config import DOC_TYPE_DEFAULT_SIGNIFICANCE
        dt_key = (doc_type or "unknown").lower().strip()
        significance = DOC_TYPE_DEFAULT_SIGNIFICANCE.get(dt_key, _DEFAULT_SIGNIFICANCE)

    if not _FIGURE_NAME_RE.match(figure_name):
        return IngestResult(
            figure_name=figure_name, session_id="", passages_ingested=0,
            observations_recorded=0, source_lang="unknown",
            source_authenticity=1.0, pub_year=pub_year,
            error=f"Invalid figure name {figure_name!r}",
        )
    if not text or not text.strip():
        return IngestResult(
            figure_name=figure_name, session_id=f"figure:{figure_name.lower()}",
            passages_ingested=0, observations_recorded=0,
            source_lang="unknown", source_authenticity=1.0, pub_year=pub_year,
            error="Empty text",
        )

    try:
        from core.temporal_layer import (
            detect_language, source_authenticity as _auth, preprocess_archaic
        )
        from core.document_store import get_document_store
        from core.value_store import get_value_store
        from core.value_extractor import process_figure

        import calendar

        # Phase 3 — language detection + source authenticity
        lang_code, _ = detect_language(text)
        auth = _auth(lang_code, is_translation=is_translation)

        # Archaic preprocessing before segmentation
        if pub_year is not None and pub_year < 1850:
            text = preprocess_archaic(text)

        passages = _segment(text)
        if not passages:
            return IngestResult(
                figure_name=figure_name, session_id=f"figure:{figure_name.lower()}",
                passages_ingested=0, observations_recorded=0,
                source_lang=lang_code, source_authenticity=auth, pub_year=pub_year,
                error="No passages extracted from text",
            )

        session_id = f"figure:{figure_name.lower().strip()}"

        doc_store = get_document_store()
        val_store = get_value_store()

        existing_watermark = float(doc_store.get_watermark(session_id))
        if pub_year:
            base_ts = float(calendar.timegm((pub_year, 1, 1, 0, 0, 0, 0, 0, 0)))
        else:
            base_ts = time.time() - (365 * 24 * 3600)
        if existing_watermark >= base_ts:
            base_ts = existing_watermark + 1.0

        val_store.register_figure_source(
            session_id=session_id,
            figure_name=figure_name,
            document_type=doc_type,
            passage_count=len(passages),
        )

        for i, passage in enumerate(passages):
            doc_store.insert_passage(
                figure_name=figure_name,
                session_id=session_id,
                text=passage,
                doc_type=doc_type,
                significance=significance,
                ts=base_ts + i,
                source_lang=lang_code,
                source_authenticity=auth,
                pub_year=pub_year,
            )

        observations = 0
        if run_extract and passages:
            observations = process_figure(session_id)

        return IngestResult(
            figure_name=figure_name,
            session_id=session_id,
            passages_ingested=len(passages),
            observations_recorded=observations,
            source_lang=lang_code,
            source_authenticity=auth,
            pub_year=pub_year,
        )

    except Exception as exc:
        _log.exception("ingest_text failed for %r", figure_name)
        return IngestResult(
            figure_name=figure_name, session_id=f"figure:{figure_name.lower()}",
            passages_ingested=0, observations_recorded=0,
            source_lang="unknown", source_authenticity=1.0, pub_year=pub_year,
            error=str(exc),
        )


def figure_profile(figure_name: str, min_demonstrations: int = 1) -> List[dict]:
    """Return value registry rows for a figure, sorted by weight DESC."""
    try:
        from core.value_store import get_value_store
        session_id = f"figure:{figure_name.lower().strip()}"
        vs = get_value_store()
        rows = vs.get_registry(session_id=session_id, min_demonstrations=min_demonstrations)
        return [dict(r) for r in rows]
    except Exception:
        _log.warning("figure_profile failed for %r", figure_name, exc_info=True)
        return []


def universal_profile(min_demonstrations: int = 1) -> List[dict]:
    """Return cross-figure universal registry, sorted by weight DESC."""
    try:
        from core.value_store import get_value_store
        vs = get_value_store()
        rows = vs.get_universal_registry(min_demonstrations=min_demonstrations)
        return [dict(r) for r in rows]
    except Exception:
        _log.warning("universal_profile failed", exc_info=True)
        return []
