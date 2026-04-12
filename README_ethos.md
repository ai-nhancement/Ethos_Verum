# Ethos — Universal Value Extraction Pipeline

Ethos extracts behavioral evidence of human values from documented human conduct — historical or contemporary — and produces labeled training data for AI alignment research. The pipeline is source-agnostic: it processes any text associated with an identifiable person, from historical archives to personal blogs to conversational records.

**Core claim:** A value stated in comfort is weak signal. A value demonstrated at personal cost — under threat, under pressure, against interest — is strong signal. Ethos measures that cost.

**Novel contributions:**
- **Resistance score** — quantifies the evidential weight of holding a value in context (doc type × significance × text markers)
- **Three-class labeling** — P1 (held under resistance), P0 (failed), APY (Answer-Pressure Yield — abandoned under pressure)
- **Spectrum principle** — signal extracted from the full human spectrum, not just saints or villains

The keyword, lexicon, structural, and phrase layers are fully deterministic. The semantic (BGE embeddings) and zero-shot classifier layers are deterministic given fixed model weights; exact reproducibility requires pinning `sentence-transformers` and `transformers` package versions and using consistent GPU/CPU precision. An optional **comprehension panel** — three-model majority-vote verification via DigitalOcean Gradient API — can be enabled for post-extraction signal verification; it is off by default and the pipeline degrades gracefully without it.

---

## Quickstart

**Requirements:** Python 3.10+

```bash
pip install -r requirements.txt
```

### Ingest a figure

```bash
# Ingest from a UTF-8 text file
python -m cli.ingest --figure gandhi --file path/to/gandhi.txt --doc-type journal

# Preview segmentation without writing
python -m cli.ingest --figure lincoln --file path/to/lincoln.txt --doc-type speech --dry-run

# Historical document with translation penalty
python -m cli.ingest --figure seneca --file path/to/seneca.txt \
    --doc-type letter --pub-year 65 --translation
```

**`--doc-type`** options: `journal` · `letter` · `speech` · `action` · `unknown`

Higher-authenticity types (`action`, `journal`) produce higher base resistance scores.

### Export labeled training data

```bash
# Export all figures to output/ric/
python -m cli.export

# With quality filters
python -m cli.export --min-consistency 0.3 --min-observations 3

# Single figure
python -m cli.export --figure gandhi

# Dry run (stats only, no files)
python -m cli.export --dry-run
```

Output is JSONL with P1/P0/APY labels, resistance scores, source metadata, and value weights.

### Batch ingestion from a manifest

```bash
python -m cli.batch_ingest --manifest samples/manifest_example.json
```

See `samples/manifest_example.json` for the manifest format.

### Corpus statistics

```bash
# Human-readable report
python -m cli.corpus_stats

# JSON output
python -m cli.corpus_stats --json

# HuggingFace dataset card
python -m cli.dataset_card
```

### REST API

```bash
python -m api.server
# Server starts at http://localhost:8000
```

Key endpoints:

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/figures/{name}/ingest` | Ingest text for a figure |
| `GET` | `/figures` | List all ingested figures |
| `GET` | `/figures/{name}/profile` | Value profile for a figure |
| `GET` | `/figures/universal` | Cross-figure value aggregate |
| `POST` | `/export/ric` | Export labeled training data |
| `GET` | `/health` | Health check |
| `GET` | `/verum` | Verum product page |
| `GET` | `/verum/values` | List all 15 values with descriptions |
| `POST` | `/verum/score` | Score a text sample against the value corpus |
| `POST` | `/verum/certify` | Issue a signed certificate for an entity |
| `GET` | `/verum/certificate/{cert_id}` | Retrieve a certificate by ID |
| `GET` | `/verum/certificates` | List certificates (optional `entity` filter) |

Interactive docs at `http://localhost:8000/docs`.

---

## How it works

### 1. Ingestion

Text is segmented into passages (min 30 chars). Each passage is stored in `data/documents.db` with:
- `doc_type` — affects the resistance formula
- `source_lang` — auto-detected via langdetect
- `source_authenticity` — 1.0 (original) / 0.85 (known translation) / 0.70 (uncertain)
- `pub_year` — drives a temporal discount (pre-1400 texts discounted to 0.70×)

### 2. Extraction (multi-layer)

Each passage runs through up to seven independent layers:

| Layer | Component | Dependency |
|-------|-----------|------------|
| L1 — Keywords | `value_extractor.py` — 15 values, context disambiguation, APY pressure detection | stdlib only |
| L1b — Lexicons | `lexicon_layer.py` — MFD2.0 (2,041 entries) + MoralStrength (452 entries) | bundled |
| L1c — Phrase | `phrase_layer.py` — pronoun-aware subject/object resolution, agency detection | stdlib only |
| L2 — Semantic | `semantic_store.py` — BGE-large-en-v1.5 embeddings against 322 seed prototypes | optional |
| L3a — Structural | `structural_layer.py` — tiered adversity/agency/resistance/stakes regex | stdlib only |
| L3b — Zero-shot | DeBERTa zero-shot entailment against per-value hypotheses | optional |
| L3c — MFT | MoralFoundationsClassifier — 10 MFT labels → Ethos value mapping | optional |
| Panel — Verification | `comprehension_panel.py` — three-model majority vote (DigitalOcean Gradient API) | optional |

The pipeline degrades gracefully: if ML dependencies are absent, L1+L1b+L1c+L3a run on stdlib alone. The comprehension panel is off by default (`cfg.comprehension_panel_enabled = False`) and requires a `MODEL_ACCESS_KEY` environment variable.

### 3. Resistance scoring

```
resistance = doc_type_base + significance_component + text_marker_bonus
```

`action` documents (behavior recorded by observers) carry a higher base resistance than `speech` (stated in public). Structural adversity patterns (e.g. "refused to yield despite threat of execution") apply a text marker bonus.

### 4. Classification and export

Each observation is classified:
- **P1** — resistance ≥ p1_threshold (default 0.55) — value held under cost
- **P0** — resistance ≤ p0_threshold (default 0.35) — value failed
- **APY** — pressure detected in a preceding passage window, value then failed — yielded under pressure
- **AMBIGUOUS** — between thresholds

The registry weight formula:
```
weight = demonstrations × avg_significance × avg_resistance × consistency
```

where `consistency` encodes observation volume, resistance stability, temporal spread, and source diversity.

---

## 15 extracted values

integrity · courage · compassion · commitment · patience · responsibility · fairness · gratitude · curiosity · resilience · love · growth · independence · loyalty · humility

---

## Running tests

```bash
pytest tests/ -q
# 934 passed
```

---

## Project structure

```
core/           pipeline modules (extraction, storage, scoring)
cli/            command-line entry points
api/            FastAPI REST service
tests/          934 tests
data/           SQLite databases + bundled lexicons
samples/        example manifest and test figures
```

Key docs:
- `STATUS_ethos.md` — phase-by-phase implementation status and design decisions
- `technical_ethos.md` — technical reference (formulas, schema, layer wiring)
- `technical_verum.md` — Verum technical reference (score formula, cert signature, API, schema)
- `OPERATIONS.md` — operational guide (batch processing, export formats, API usage)
- `PAPER.md` — research paper

---

## Verum

Ethos ships with Verum — a value alignment scoring and certification layer built on top of the same extraction pipeline.

**Score formula:** `verum_score = P1_ratio × avg_P1_resistance`

A Verum score measures how consistently text demonstrates value-aligned behavior *under pressure* — not just in comfortable assertions. Certificates are deterministic SHA-256 signed and include all threshold parameters so they cannot be reissued with lenient settings to produce the same signature.

**Files:**
- `core/verum.py` — scoring and certification logic
- `api/verum_routes.py` — FastAPI routes
- `static/verum.html` — product page (served at `GET /verum`)
