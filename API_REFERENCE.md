# Ethos + Verum API Reference

Base URL: `http://localhost:8000`

Interactive docs: `http://localhost:8000/docs`

Start the server: `python -m api.server`

---

## Ethos Endpoints

### POST /figures/{name}/ingest

Ingest text for a historical figure.

**Path parameters:**
- `name` (string, required) — Figure identifier. Alphanumeric, underscores, hyphens. 1-64 chars.

**Request body:**

```json
{
  "text": "Full text to ingest",
  "pronoun": "he",
  "doc_type": "journal",
  "pub_year": 1869,
  "is_translation": false,
  "significance": 0.90,
  "run_extract": true,
  "doc_title": "Gandhi's Journal, Volume 1"
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `text` | string | required | Raw source text (UTF-8) |
| `pronoun` | string | required | `he`, `she`, `they`, or `i` |
| `doc_type` | string | `"unknown"` | `journal`, `letter`, `speech`, `action`, or `unknown` |
| `pub_year` | int | null | Publication year (pre-1400 texts get temporal discount) |
| `is_translation` | bool | null | `true` sets authenticity to 0.85. `null` auto-detects. |
| `significance` | float | auto | 0.0-1.0. Auto-derived from doc_type if omitted. |
| `run_extract` | bool | `true` | Run value extraction immediately after ingestion |
| `doc_title` | string | `""` | Document title for corpus tracking |

**Response (200):**

```json
{
  "figure": "gandhi",
  "session_id": "figure:gandhi",
  "passages_ingested": 47,
  "observations_recorded": 23,
  "source_lang": "en",
  "source_authenticity": 1.0,
  "pub_year": 1869,
  "ok": true,
  "error": null
}
```

---

### GET /figures

List all ingested figures.

**Response (200):**

```json
{
  "figures": [
    {
      "figure_name": "gandhi",
      "session_id": "figure:gandhi",
      "document_type": "journal",
      "passage_count": 47,
      "top_value": "integrity",
      "top_weight": 3.21
    }
  ],
  "total": 1
}
```

---

### GET /figures/{name}/profile

Value profile for a figure.

**Query parameters:**
- `min_demonstrations` (int, default 1) — Minimum observation count to include a value

**Response (200):**

```json
{
  "figure": "gandhi",
  "values": [
    {
      "value_name": "integrity",
      "demonstrations": 12,
      "avg_significance": 0.88,
      "avg_resistance": 0.76,
      "consistency": 0.82,
      "weight": 6.53
    }
  ]
}
```

---

### GET /figures/universal

Cross-figure aggregate value profile.

**Query parameters:**
- `min_demonstrations` (int, default 1)

**Response (200):** Same shape as figure profile, aggregated across all figures.

---

### POST /export/ric

Export labeled training data.

**Response (200):**

```json
{
  "total_records": 234,
  "p1_count": 145,
  "p0_count": 52,
  "apy_count": 18,
  "ambiguous_count": 19,
  "figures": ["gandhi", "lincoln", "seneca"],
  "output_dir": "output/ric/"
}
```

---

### GET /health

**Response (200):**

```json
{
  "status": "ok",
  "figures": 3,
  "observations": 234
}
```

---

## Verum Endpoints

### GET /verum/values

List all 15 values with descriptions.

**Response (200):**

```json
{
  "values": [
    {
      "value_name": "integrity",
      "description": "Honest and truthful — says what is real, refuses deception even at cost to self"
    }
  ],
  "total": 15
}
```

---

### POST /verum/score

Score a text for value alignment.

**Request body:**

```json
{
  "text": "Despite the threat, I refused to sign the false report.",
  "doc_type": "action",
  "significance": 0.90,
  "p1_threshold": 0.55,
  "p0_threshold": 0.35
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `text` | string | required | Text to score (max 50K chars) |
| `doc_type` | string | `"unknown"` | Document type for resistance calibration |
| `significance` | float | `0.90` | Base significance weight |
| `p1_threshold` | float | `0.55` | Min resistance for P1 classification |
| `p0_threshold` | float | `0.35` | Max resistance for P0 classification |

**Validation:** `p0_threshold` must be less than `p1_threshold`.

**Response (200):**

```json
{
  "verum_score": 0.82,
  "resistance": 0.91,
  "p1_count": 3,
  "p0_count": 0,
  "apy_count": 0,
  "ambiguous_count": 1,
  "total_signals": 4,
  "signals": [
    {
      "value_name": "integrity",
      "resistance": 0.91,
      "label": "P1",
      "label_reason": "high_resistance_hold_marker",
      "confidence": 0.90,
      "detection_method": "keyword+semantic",
      "disambiguation_confidence": 1.0,
      "text_excerpt": "...refused to sign the false report...",
      "embedding_score": 0.78
    }
  ]
}
```

---

### POST /verum/certify

Issue a signed Verum certificate for an entity.

**Request body:**

```json
{
  "entity_name": "Lincoln",
  "samples": [
    "I am naturally anti-slavery. If slavery is not wrong, nothing is wrong.",
    "Those who deny freedom to others deserve it not for themselves."
  ],
  "doc_type": "speech",
  "significance": 0.90,
  "p1_threshold": 0.55,
  "p0_threshold": 0.35,
  "min_score": 0.60,
  "min_values": 3,
  "figure_basis": null
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `entity_name` | string | required | Entity being certified |
| `samples` | list[string] | required | 1-100 text samples |
| `doc_type` | string | `"unknown"` | Document type |
| `significance` | float | `0.90` | Base significance |
| `p1_threshold` | float | `0.55` | P1 threshold |
| `p0_threshold` | float | `0.35` | P0 threshold |
| `min_score` | float | `0.60` | Min Verum score to certify |
| `min_values` | int | `3` | Min distinct P1 values to certify |
| `figure_basis` | string | null | Figure name for benchmark comparison |

**Response (200):**

```json
{
  "certificate_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "entity_name": "Lincoln",
  "certified": true,
  "verum_score": 0.71,
  "sample_count": 2,
  "values_certified": ["courage", "fairness", "integrity"],
  "value_scores": {
    "courage": { "p1_count": 2, "avg_resistance": 0.68 },
    "fairness": { "p1_count": 1, "avg_resistance": 0.72 },
    "integrity": { "p1_count": 3, "avg_resistance": 0.81 }
  },
  "issued_at": 1711535535,
  "doc_type": "speech",
  "p1_threshold": 0.55,
  "p0_threshold": 0.35,
  "min_score": 0.60,
  "min_values": 3,
  "figure_basis": null,
  "figure_comparison": null,
  "signature": "sha256:9f8e7d6c5b4a3210fedcba9876543210abcdef0123456789abcdef0123456789"
}
```

---

### GET /verum/certificate/{cert_id}

Retrieve a certificate by ID.

**Response (200):** Full certificate object (same shape as certify response).

**Response (404):** `{ "detail": "Certificate not found" }`

---

### GET /verum/certificates

List certificates.

**Query parameters:**
- `entity` (string, optional) — Filter by entity name
- `limit` (int, default 20) — Max results

**Response (200):**

```json
{
  "certificates": [
    {
      "certificate_id": "a1b2c3d4-...",
      "entity_name": "Lincoln",
      "certified": true,
      "verum_score": 0.71,
      "issued_at": 1711535535
    }
  ],
  "total": 1
}
```

---

## Certificate Verification

Certificates are independently verifiable without calling the API.

### Steps

1. Extract these fields from the certificate:
   - `entity_name`, `samples` (sorted), `overall_score`, `values_certified` (sorted), `issued_at`, `doc_type`, `p1_threshold`, `p0_threshold`, `min_score`, `min_values`

2. Build the signature input:
   ```python
   sig_input = json.dumps({
       "entity_name": entity_name,
       "samples": sorted(samples),
       "overall_score": overall_score,
       "values_certified": sorted(values_certified),
       "issued_at": issued_at,
       "doc_type": doc_type,
       "p1_threshold": p1_threshold,
       "p0_threshold": p0_threshold,
       "min_score": min_score,
       "min_values": min_values,
   }, sort_keys=True)
   ```

3. Hash it:
   ```python
   expected = "sha256:" + hashlib.sha256(sig_input.encode()).hexdigest()
   ```

4. Compare `expected` against the certificate's `signature` field. If they match, the certificate is authentic and has not been tampered with.

---

## Error Handling

All endpoints return standard HTTP error codes:

| Code | Meaning |
|------|---------|
| 200 | Success |
| 400 | Invalid request (bad JSON, threshold validation, etc.) |
| 404 | Resource not found |
| 500 | Internal error |

Error responses include a human-readable `detail` or `error` field.

---

## Rate Limits

None enforced by default. The server is designed for local or single-tenant deployment. For production use behind a reverse proxy, add rate limiting at the proxy layer.
