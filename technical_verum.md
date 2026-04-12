# Verum — Technical Reference

> **System class:** Value Alignment Scoring and Certification Layer
> **Core claim:** Measuring value-consistent behavior under pressure — not stated values, but demonstrated ones.
> **Architecture:** Built on top of Ethos. No separate process. No separate database. Shares `values.db` directly.
> **Constitutional invariant:** No LLM calls. All scoring and certification is deterministic given fixed Ethos extraction weights.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Score Formula](#2-score-formula)
3. [Certification Logic](#3-certification-logic)
4. [Certificate Signature](#4-certificate-signature)
5. [Database Schema](#5-database-schema)
6. [API Reference](#6-api-reference)
   - Request/Response Models
   - Endpoint Specifications
7. [Module Reference](#7-module-reference)
   - `core/verum.py`
   - `api/verum_routes.py`
   - `api/models.py` (Verum types)
8. [Design Decisions](#8-design-decisions)
9. [Constitutional Invariants](#9-constitutional-invariants)

---

## 1. Architecture Overview

Verum is not a separate service. It is a scoring and certification layer that runs inside Ethos and reads from the same `values.db` that the extraction pipeline writes to.

```
Input text(s)
    │
    ▼
core/verum.score_text()
    ├── core.value_extractor.extract_value_signals()   ← keyword + layer matching
    ├── core.resistance.compute_resistance()            ← doc_type + significance + markers
    └── cli.export.classify_observation()              ← P1 / P0 / APY / AMBIGUOUS per signal
    │
    ▼
Aggregate across signals
    ├── P1_ratio         = P1_count / total_signals
    ├── avg_P1_resistance = mean(resistance for P1 signals)
    └── verum_score      = P1_ratio × avg_P1_resistance
    │
    ▼
core/verum.certify()   [optional: if score + value count meet thresholds]
    ├── aggregate score_text() across all samples
    ├── check score >= min_score AND len(values_certified) >= min_values
    ├── build certificate dict
    ├── sign with SHA-256 over all 10 parameters
    └── core.value_store.store_certificate()  → writes to values.db
```

**Shared database:** `data/values.db`. The `verum_certificates` table lives alongside `value_observations` and `value_registry`. The singleton `get_value_store()` is the same instance used by the extraction pipeline — no second connection, no separate file.

---

## 2. Score Formula

### The Formula

```
verum_score = P1_ratio × avg_P1_resistance
```

### Components

| Component | Definition | Range |
|-----------|-----------|-------|
| `P1_ratio` | Fraction of extracted value signals classified P1 (held under resistance ≥ `p1_threshold`) | 0.0–1.0 |
| `avg_P1_resistance` | Mean resistance score of P1 signals | 0.0–1.0 |
| `verum_score` | Product of the two | 0.0–1.0 |

### Why the product

`P1_ratio` alone could be inflated by a text full of comfortable assertions at low resistance. `avg_P1_resistance` alone could be driven by a single intense passage among many failures. The product requires both: *how consistently* the value holds AND *at what cost* each time it holds.

A score of 0.0 means either no P1 signals were found, or the P1 signals had zero resistance (comfort-stated values only). A score approaching 1.0 requires nearly all signals to be P1, and those P1 signals to have high resistance — values held consistently under real pressure.

### Score Breakdown (per-value)

`score_text()` returns a `signals` list with one entry per detected value:

```json
{
  "value_name": "integrity",
  "label": "P1",
  "resistance": 0.82,
  "confidence": 0.90,
  "excerpt": "...despite the threat to my position...",
  "source": "keyword+structural"
}
```

And a summary:

```json
{
  "verum_score": 0.71,
  "P1_count": 5,
  "P0_count": 1,
  "APY_count": 0,
  "total_signals": 6,
  "P1_ratio": 0.833,
  "avg_P1_resistance": 0.852,
  "values_detected": ["integrity", "courage", "commitment"]
}
```

### Default Thresholds

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `p1_threshold` | 0.55 | Minimum resistance for a signal to be classified P1 |
| `p0_threshold` | 0.35 | Maximum resistance for a signal to be classified P0 |
| `doc_type` | `"unknown"` | Affects resistance formula via document type bonus |

`p0_threshold` must be strictly less than `p1_threshold` — enforced at the API layer via `@model_validator`. Signals between the thresholds are AMBIGUOUS and do not contribute to `P1_ratio` or `avg_P1_resistance`.

---

## 3. Certification Logic

### `certify()` Process

1. Call `score_text()` on each sample independently.
2. Collect all signals across samples.
3. Compute `overall_score = P1_ratio × avg_P1_resistance` over the aggregated signal pool.
4. Collect `values_certified` = all unique value names found in P1 signals across all samples.
5. Check certification criteria:
   - `overall_score >= min_score` (default 0.40)
   - `len(values_certified) >= min_values` (default 2)
6. If criteria met: `certified = True`. Otherwise: `certified = False` (certificate still issued, but marked not certified).
7. Build and sign the certificate dict.
8. Persist to `values.db` via `store_certificate()`.
9. Return the full certificate.

### Certificate Structure

```json
{
  "certificate_id": "uuid4",
  "entity_name": "Lincoln",
  "certified": true,
  "verum_score": 0.73,
  "sample_count": 10,
  "values_certified": ["integrity", "courage", "commitment"],
  "issued_at": 1742515200.0,
  "figure_basis": "",
  "doc_type": "action",
  "p1_threshold": 0.55,
  "p0_threshold": 0.35,
  "min_score": 0.40,
  "min_values": 2,
  "signature": "sha256:a3f7...",
  "signal_detail": { ... }
}
```

### `figure_basis`

Optional. When set, `certify()` can compare the entity's score against a historical figure already ingested in Ethos. Used to benchmark an entity's behavioral evidence against a known reference (e.g. "scores above Gandhi's integrity baseline").

---

## 4. Certificate Signature

### Design Principle

The signature makes certificates tamper-evident. A certificate issued with lenient thresholds (`min_score=0.01`, `p1_threshold=0.10`) must produce a different signature from one issued with standard thresholds — so a consumer can verify not just the score, but *the conditions under which it was earned*.

### What Is Signed

All 10 certification parameters are included:

```python
sig_input = json.dumps({
    "entity_name":      entity_name,
    "samples":          sorted(samples),       # sorted for determinism
    "overall_score":    overall_score,
    "values_certified": sorted(values_certified),
    "issued_at":        issued_at,
    "doc_type":         doc_type,
    "p1_threshold":     p1_threshold,
    "p0_threshold":     p0_threshold,
    "min_score":        min_score,
    "min_values":       min_values,
}, sort_keys=True)

signature = "sha256:" + hashlib.sha256(sig_input.encode()).hexdigest()
```

### Verification

To verify a certificate independently:

1. Reconstruct `sig_input` from the certificate fields (using the same `json.dumps` with `sort_keys=True` and sorted samples/values).
2. SHA-256 hash the UTF-8 encoded string.
3. Compare `"sha256:" + digest` against the `signature` field.

### What It Does NOT Protect Against

The signature is a consistency seal, not a trust-chain proof. It proves the certificate fields are internally consistent — not that the underlying Ethos pipeline was run correctly, that the samples are genuine, or that the entity_name is accurate. It is tamper-evident, not cryptographically authoritative.

---

## 5. Database Schema

### `verum_certificates` table (in `data/values.db`)

```sql
CREATE TABLE verum_certificates (
    id              TEXT PRIMARY KEY,       -- certificate_id (UUID4)
    entity_name     TEXT NOT NULL,
    certified       INTEGER NOT NULL,       -- 1 = certified, 0 = not certified
    verum_score     REAL NOT NULL,
    sample_count    INTEGER NOT NULL,
    values_certified TEXT NOT NULL,         -- JSON array
    issued_at       REAL NOT NULL,          -- Unix timestamp
    figure_basis    TEXT NOT NULL DEFAULT '',
    signature       TEXT NOT NULL,          -- "sha256:<hex>"
    payload         TEXT NOT NULL           -- full certificate JSON
);

CREATE INDEX IF NOT EXISTS idx_verum_certs_entity ON verum_certificates(entity_name);
CREATE INDEX IF NOT EXISTS idx_verum_certs_issued ON verum_certificates(issued_at);
```

### Ownership

This table is created inside `ValueStore._init_db()` alongside the Ethos extraction tables. It is part of `values.db` — not a separate file. `store_certificate()`, `get_certificate()`, and `list_certificates()` are methods on `ValueStore`.

---

## 6. API Reference

### Request/Response Models

#### `VerumScoreRequest`

```python
class VerumScoreRequest(BaseModel):
    text: str                              # Text to score
    doc_type: str = "unknown"              # journal | letter | speech | action | unknown
    p1_threshold: float = 0.55            # Min resistance for P1
    p0_threshold: float = 0.35            # Max resistance for P0

    @model_validator                       # Enforces p0 < p1
```

#### `VerumScoreResponse`

```python
class VerumScoreResponse(BaseModel):
    verum_score: float                    # P1_ratio × avg_P1_resistance
    P1_count: int
    P0_count: int
    APY_count: int
    total_signals: int
    P1_ratio: float
    avg_P1_resistance: float
    values_detected: list[str]
    signals: list[VerumSignalEntry]       # Per-signal detail
```

#### `VerumSignalEntry`

```python
class VerumSignalEntry(BaseModel):
    value_name: str
    label: str                            # P1 | P0 | APY | AMBIGUOUS
    resistance: float
    confidence: float
    excerpt: str
    source: str                           # e.g. "keyword+structural"
```

#### `VerumCertifyRequest`

```python
class VerumCertifyRequest(BaseModel):
    entity_name: str
    samples: list[str]                    # 1–100 text samples
    doc_type: str = "unknown"
    p1_threshold: float = 0.55
    p0_threshold: float = 0.35
    min_score: float = 0.40               # Min verum_score to certify
    min_values: int = 2                   # Min distinct values required
    figure_basis: str = ""               # Optional: compare against historical figure

    @model_validator                      # Enforces p0 < p1
```

#### `VerumCertificateResponse`

```python
class VerumCertificateResponse(BaseModel):
    certificate_id: str
    entity_name: str
    certified: bool
    verum_score: float
    sample_count: int
    values_certified: list[str]
    issued_at: float
    figure_basis: str
    doc_type: str
    p1_threshold: float
    p0_threshold: float
    min_score: float
    min_values: int
    signature: str
    signal_detail: dict
```

#### `VerumCertificateListResponse`

```python
class VerumCertificateListResponse(BaseModel):
    certificates: list[VerumCertificateSummary]
    total: int
```

---

### Endpoint Specifications

#### `GET /verum`

Serves `static/verum.html` — the Verum product page frontend.

---

#### `GET /verum/values`

Returns all 15 value descriptions.

**Response:** `VerumValuesResponse`

```json
{
  "values": [
    {"name": "integrity", "description": "Honesty and consistency between stated beliefs and actions..."},
    ...
  ]
}
```

---

#### `POST /verum/score`

Score a single text sample.

**Request:** `VerumScoreRequest`

**Response:** `VerumScoreResponse`

**Notes:**
- Runs synchronously in a thread pool executor (non-blocking FastAPI event loop).
- Returns `verum_score: 0.0` with empty signals if no value keywords are detected.
- Does not persist anything to the database.

**Example:**

```bash
curl -X POST http://localhost:8000/verum/score \
  -H "Content-Type: application/json" \
  -d '{"text": "Despite the threat, I refused to sign.", "doc_type": "action"}'
```

---

#### `POST /verum/certify`

Issue a signed certificate for an entity.

**Request:** `VerumCertifyRequest`

**Response:** `VerumCertificateResponse`

**Notes:**
- Aggregates `score_text()` across all samples.
- Always issues a certificate record (even if `certified: false`).
- Persists to `verum_certificates` table in `values.db`.
- Runs synchronously in a thread pool executor.

**Example:**

```bash
curl -X POST http://localhost:8000/verum/certify \
  -H "Content-Type: application/json" \
  -d '{
    "entity_name": "Lincoln",
    "samples": ["passage 1", "passage 2", "passage 3"],
    "doc_type": "action",
    "min_score": 0.40,
    "min_values": 2
  }'
```

---

#### `GET /verum/certificate/{cert_id}`

Retrieve a previously issued certificate by its UUID.

**Response:** `VerumCertificateResponse`

**Errors:** `404` if certificate not found.

---

#### `GET /verum/certificates`

List all certificates, optionally filtered by entity name.

**Query params:**
- `entity` (optional) — filter by `entity_name`
- `limit` (optional, default 20) — max results

**Response:** `VerumCertificateListResponse`

---

## 7. Module Reference

### `core/verum.py`

**Imports from Ethos:**

```python
from core.value_extractor import extract_value_signals
from core.resistance import compute_resistance
from cli.export import classify_observation
from core.value_store import get_value_store
```

**Public functions:**

| Function | Signature | Purpose |
|----------|-----------|---------|
| `score_text` | `(text, doc_type, p1_threshold, p0_threshold) → dict` | Score one text sample |
| `certify` | `(entity_name, samples, doc_type, p1_threshold, p0_threshold, min_score, min_values, figure_basis) → dict` | Issue signed certificate |
| `_empty_score` | `() → dict` | Fail-open empty score result |
| `VALUE_DESCRIPTIONS` | `dict[str, str]` | 15 value descriptions for `/verum/values` |

**Never raises.** All public functions are wrapped in try/except and return empty/failed results on error.

---

### `api/verum_routes.py`

FastAPI `APIRouter` with `prefix="/verum"`. Mounted in `api/app.py` via:

```python
from api.verum_routes import router as verum_router
app.include_router(verum_router)
```

All compute endpoints use `asyncio.get_event_loop().run_in_executor(None, ...)` to keep scoring off the FastAPI event loop.

---

### `api/models.py` — Verum Types

All Verum Pydantic models are defined in `api/models.py` alongside the Ethos extraction models.

| Model | Used By |
|-------|---------|
| `VerumScoreRequest` | `POST /verum/score` |
| `VerumSignalEntry` | Inside `VerumScoreResponse` |
| `VerumScoreResponse` | `POST /verum/score` |
| `VerumCertifyRequest` | `POST /verum/certify` |
| `VerumValueScore` | Inside `VerumCertificateResponse.signal_detail` |
| `VerumCertificateResponse` | `POST /verum/certify`, `GET /verum/certificate/{id}` |
| `VerumCertificateSummary` | Inside `VerumCertificateListResponse` |
| `VerumCertificateListResponse` | `GET /verum/certificates` |
| `VerumValueDescription` | Inside `VerumValuesResponse` |
| `VerumValuesResponse` | `GET /verum/values` |

---

## 8. Design Decisions

### 2026-03-21 — Integrated architecture, not standalone

**Decision:** Verum lives inside Ethos (`core/verum.py`), not as a separate service or package.

**Rationale:** Verum's core function is to call `extract_value_signals()` and `compute_resistance()` — the same functions the Ethos extraction pipeline uses. Running these from a separate process would require either a separate database copy or an RPC layer to the Ethos API. A shared module in the same codebase avoids both problems and ensures Verum always reads from the same `values.db` as the pipeline that produced the underlying data.

### 2026-03-21 — All threshold parameters in signature

**Decision:** The certificate signature covers all 10 parameters including `p1_threshold`, `p0_threshold`, `min_score`, and `min_values`.

**Rationale:** The original standalone Verum only signed `entity_name`, `samples`, `score`, `values_certified`, and `issued_at`. A bad actor could reissue a certificate with `p1_threshold=0.10` (trivially satisfied) and produce the same signature as one issued with `p1_threshold=0.55`. Including the thresholds in the signature makes the certification conditions tamper-evident — changing any threshold changes the signature.

### 2026-03-21 — Certificates issued even when not certified

**Decision:** `certify()` always writes to `verum_certificates`, even when `certified: false`.

**Rationale:** A failed certification is still an auditable record. It shows that a certification attempt was made, at what thresholds, and what score was achieved. This makes the system auditable in both directions — not just for entities that pass.

### 2026-03-21 — `p0_threshold < p1_threshold` enforced at API layer

**Decision:** `@model_validator` on both `VerumScoreRequest` and `VerumCertifyRequest` raises `ValueError` if `p0_threshold >= p1_threshold`.

**Rationale:** If `p0 >= p1`, the classification bands collapse. Signals that should be P1 would be classified P0 and vice versa, making the score meaningless. This constraint is enforced at the Pydantic validation layer — before any computation runs — so it fails fast with a clear 422 validation error.

---

## 9. Constitutional Invariants

| Invariant | Rule |
|-----------|------|
| No LLM calls | `score_text()` and `certify()` are pure Ethos pipeline calls. No external API, no model inference beyond Ethos's own deterministic layers. |
| Deterministic scoring | Given fixed Ethos extraction weights, same input + same thresholds = same score every time. |
| Threshold transparency | All 5 certification threshold parameters are included in the certificate payload and the signature. A consumer can verify the exact conditions under which any certificate was issued. |
| Fail-open | All public functions in `core/verum.py` are wrapped in try/except. Errors return empty/failed results — they never raise to the caller. |
| Shared database | Verum uses the same `get_value_store()` singleton as Ethos. No second connection, no duplicate schema. |
| Certificates are immutable | `verum_certificates` is append-only. Certificates are never updated after issuance. A re-certification creates a new record with a new UUID. |
