"""
api/models.py

Pydantic request and response models for the Ethos REST API.
"""

from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------

class IngestRequest(BaseModel):
    text:           str   = Field(..., description="Raw source text to ingest.")
    pronoun:        str   = Field(..., description="Figure's pronoun: he|she|they|i")
    doc_type:       str   = Field("unknown",
                                  description="Document type: journal|letter|speech|action|unknown")
    pub_year:       Optional[int]  = Field(None, description="Publication year (e.g. 1863).")
    is_translation: Optional[bool] = Field(None,
                                           description="True = known translation; "
                                                       "None = auto-detect from language.")
    significance:   float = Field(0.90, ge=0.0, le=1.0,
                                  description="Base significance score for all passages.")
    run_extract:    bool  = Field(True, description="Run extraction immediately after ingestion.")


class IngestResponse(BaseModel):
    figure:               str
    session_id:           str
    passages_ingested:    int
    observations_recorded: int
    source_lang:          str
    source_authenticity:  float
    pub_year:             Optional[int]
    ok:                   bool
    error:                Optional[str] = None


# ---------------------------------------------------------------------------
# Figure profile
# ---------------------------------------------------------------------------

class ValueRegistryEntry(BaseModel):
    value_name:       str
    demonstrations:   int
    avg_significance: float
    avg_resistance:   float
    consistency:      float
    weight:           float
    first_seen_ts:    Optional[float] = None
    last_seen_ts:     Optional[float] = None


class FigureProfileResponse(BaseModel):
    figure:    str
    session_id: str
    values:    List[ValueRegistryEntry]


class FigureListItem(BaseModel):
    figure_name:    str
    session_id:     str
    doc_type:       str
    passage_count:  int
    first_ingested: Optional[float] = None


class FigureListResponse(BaseModel):
    figures: List[FigureListItem]
    total:   int


# ---------------------------------------------------------------------------
# Universal profile
# ---------------------------------------------------------------------------

class UniversalValueEntry(BaseModel):
    value_name:          str
    total_demonstrations: int
    figure_count:        int
    avg_weight:          float


class UniversalProfileResponse(BaseModel):
    values:       List[UniversalValueEntry]
    figure_count: int


# ---------------------------------------------------------------------------
# Export / RIC
# ---------------------------------------------------------------------------

class ExportRequest(BaseModel):
    figure:           Optional[str]  = Field(None,
                                             description="Export only this figure (default: all).")
    p1_threshold:     float = Field(0.55, ge=0.0, le=1.0, description="Min resistance for P1 classification.")
    p0_threshold:     float = Field(0.35, ge=0.0, le=1.0, description="Max resistance for P0 classification.")
    min_observations: int   = Field(1, ge=1, description="Min observations per value to include.")
    min_consistency:  float = Field(0.0, ge=0.0, le=1.0, description="Min consistency score to include.")
    dry_run:          bool  = Field(False, description="Return stats only, write no files.")
    output_dir:       str   = Field("output/ric", description="Output directory path.")


class ExportResponse(BaseModel):
    ok:              bool
    figure:          Optional[str]
    p1_count:        int
    p0_count:        int
    apy_count:       int
    ambiguous_count: int
    total_count:     int
    output_dir:      Optional[str]
    files_written:   List[str]
    error:           Optional[str] = None


# ---------------------------------------------------------------------------
# Verum — scoring and certification
# ---------------------------------------------------------------------------

class VerumValueInfo(BaseModel):
    value_name:  str
    description: str


class VerumValuesResponse(BaseModel):
    values: List[VerumValueInfo]
    total:  int


class VerumSignal(BaseModel):
    value_name:               str
    resistance:               float
    label:                    str
    label_reason:             str
    confidence:               float
    detection_method:         str   = "keyword"
    disambiguation_confidence: float = 1.0
    text_excerpt:             str   = ""


class VerumScoreRequest(BaseModel):
    text:         str   = Field(..., description="Text to score (max 50K chars).")
    doc_type:     str   = Field("unknown", description="Document type.")
    significance: float = Field(0.90, ge=0.0, le=1.0)
    p1_threshold: float = Field(0.55, ge=0.0, le=1.0)
    p0_threshold: float = Field(0.35, ge=0.0, le=1.0)


class VerumScoreResponse(BaseModel):
    verum_score:     float
    resistance:      float
    p1_count:        int
    p0_count:        int
    apy_count:       int
    ambiguous_count: int
    total_signals:   int
    signals:         List[VerumSignal]


class VerumCertifyRequest(BaseModel):
    entity_name:  str          = Field(..., description="Entity being certified.")
    samples:      List[str]    = Field(..., description="5-100 text samples. Certification demands substantial evidence.", min_length=5, max_length=100)
    doc_type:     str          = Field("unknown")
    significance: float        = Field(0.90, ge=0.0, le=1.0)
    p1_threshold: float        = Field(0.55, ge=0.0, le=1.0)
    p0_threshold: float        = Field(0.35, ge=0.0, le=1.0)
    min_score:    float        = Field(0.60, ge=0.0, le=1.0)
    min_values:   int          = Field(3, ge=1)


class VerumValueScore(BaseModel):
    p1_count:       int
    avg_resistance: float


class VerumCertifyResponse(BaseModel):
    certificate_id:   str
    entity_name:      str
    certified:        bool
    verum_score:      float
    sample_count:     int
    values_certified: List[str]
    value_scores:     dict
    issued_at:        float
    doc_type:         str
    p1_threshold:     float
    p0_threshold:     float
    min_score:        float
    min_values:       int
    signature:        str


class VerumCertificateSummary(BaseModel):
    certificate_id: str
    entity_name:    str
    certified:      bool
    verum_score:    float
    issued_at:      float


class VerumCertificatesResponse(BaseModel):
    certificates: List[VerumCertificateSummary]
    total:        int
