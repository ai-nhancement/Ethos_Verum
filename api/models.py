"""
api/models.py

Pydantic request and response models for the Ethos REST API.
"""

from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field, model_validator


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
    figure:            Optional[str]  = Field(None,
                                              description="Export only this figure (default: all).")
    p1_threshold:      float = Field(0.55, ge=0.0, le=1.0, description="Min resistance for P1 classification.")
    p0_threshold:      float = Field(0.35, ge=0.0, le=1.0, description="Max resistance for P0 classification.")
    min_observations:  int   = Field(1, ge=1, description="Min observations per value to include.")
    min_consistency:   float = Field(0.0, ge=0.0, le=1.0, description="Min consistency score to include.")
    dry_run:           bool  = Field(False, description="Return stats only, write no files.")
    output_dir:        str   = Field("output/ric", description="Output directory path.")
    include_ambiguous: bool  = Field(True, description="Include AMBIGUOUS observations in per-figure files.")
    value_tension:     bool  = Field(False, description="Detect and export value tension events.")
    contrastive_pairs: bool  = Field(False, description="Export (pressure, failure) APY contrastive pairs.")

    @model_validator(mode="after")
    def p0_must_be_less_than_p1(self) -> "ExportRequest":
        if self.p0_threshold >= self.p1_threshold:
            raise ValueError(
                f"p0_threshold ({self.p0_threshold}) must be less than "
                f"p1_threshold ({self.p1_threshold})"
            )
        return self


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

class VerumScoreRequest(BaseModel):
    text:          str   = Field(..., max_length=50_000,
                                 description="Text to score (max 50,000 chars).")
    doc_type:      str   = Field("unknown",
                                 description="Document type: journal|letter|speech|action|unknown")
    significance:  float = Field(0.90, ge=0.0, le=1.0)
    p1_threshold:  float = Field(0.55, ge=0.0, le=1.0,
                                 description="Min resistance for P1 classification.")
    p0_threshold:  float = Field(0.35, ge=0.0, le=1.0,
                                 description="Max resistance for P0 classification.")

    @model_validator(mode="after")
    def p0_must_be_less_than_p1(self) -> "VerumScoreRequest":
        if self.p0_threshold >= self.p1_threshold:
            raise ValueError(
                f"p0_threshold ({self.p0_threshold}) must be less than "
                f"p1_threshold ({self.p1_threshold})"
            )
        return self


class VerumSignalEntry(BaseModel):
    value_name:               str
    resistance:               float
    label:                    str
    label_reason:             str
    confidence:               float
    detection_method:         str
    disambiguation_confidence: float
    text_excerpt:             str
    embedding_score:          Optional[float] = None


class VerumScoreResponse(BaseModel):
    verum_score:     float
    resistance:      float
    p1_count:        int
    p0_count:        int
    apy_count:       int
    ambiguous_count: int
    total_signals:   int
    signals:         List[VerumSignalEntry]


class VerumCertifyRequest(BaseModel):
    entity_name:   str         = Field(..., min_length=1, max_length=200,
                                        description="Name of the entity being certified.")
    samples:       List[str]   = Field(..., min_length=1, max_length=100,
                                        description="1–100 text samples to score.")
    doc_type:      str         = Field("unknown")
    significance:  float       = Field(0.90, ge=0.0, le=1.0)
    p1_threshold:  float       = Field(0.55, ge=0.0, le=1.0)
    p0_threshold:  float       = Field(0.35, ge=0.0, le=1.0)
    min_score:     float       = Field(0.60, ge=0.0, le=1.0,
                                        description="Min overall Verum score to certify.")
    min_values:    int         = Field(3, ge=1,
                                        description="Min distinct P1 values to certify.")
    figure_basis:  Optional[str] = Field(None,
                                          description="Figure name for benchmark comparison.")

    @model_validator(mode="after")
    def p0_must_be_less_than_p1(self) -> "VerumCertifyRequest":
        if self.p0_threshold >= self.p1_threshold:
            raise ValueError(
                f"p0_threshold ({self.p0_threshold}) must be less than "
                f"p1_threshold ({self.p1_threshold})"
            )
        return self


class VerumValueScore(BaseModel):
    p1_count:        int
    p0_count:        int
    apy_count:       int
    ambiguous_count: int
    avg_resistance:  float
    detection_rate:  float


class VerumCertificateResponse(BaseModel):
    certificate_id:    str
    entity_name:       str
    certified:         bool
    verum_score:       float
    sample_count:      int
    values_certified:  List[str]
    value_scores:      dict
    issued_at:         float
    doc_type:          str
    p1_threshold:      float
    p0_threshold:      float
    min_score:         float
    min_values:        int
    figure_basis:      Optional[str] = None
    figure_comparison: Optional[dict] = None
    signature:         str
    error:             Optional[str] = None


class VerumCertificateSummary(BaseModel):
    certificate_id:   str
    entity_name:      str
    certified:        bool
    verum_score:      float
    sample_count:     int
    values_certified: List[str]
    issued_at:        float
    figure_basis:     Optional[str] = None
    signature:        str


class VerumCertificateListResponse(BaseModel):
    certificates: List[VerumCertificateSummary]
    total:        int


class VerumValueDescription(BaseModel):
    value_name:  str
    description: str


class VerumValuesResponse(BaseModel):
    values: List[VerumValueDescription]
    total:  int
