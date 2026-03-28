<p align="center">
  <img src="ethos_logo.png" alt="Ethos" width="350">
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <img src="verum_logo.png" alt="Verum" width="350">
</p>

<h1 align="center">Ethos + Verum</h1>

<p align="center">
  <strong>Extract what integrity looks like. Certify what meets the standard.</strong>
</p>

<p align="center">
  <a href="#how-it-works"><img src="https://img.shields.io/badge/Ethos-extraction_pipeline-604020?style=for-the-badge" alt="Ethos"></a>
  <a href="VERUM.md"><img src="https://img.shields.io/badge/Verum-certification_layer-2ea44f?style=for-the-badge" alt="Verum"></a>
  <a href="API_REFERENCE.md"><img src="https://img.shields.io/badge/docs-API_Reference-604020?style=for-the-badge" alt="API Reference"></a>
  <a href="#the-15-values"><img src="https://img.shields.io/badge/values-15-8B6914?style=for-the-badge" alt="15 Values"></a>
  <a href="#testing"><img src="https://img.shields.io/badge/tests-934_passing-C0834D?style=for-the-badge" alt="934 Tests"></a>
</p>

---

**Ethos** extracts behavioral evidence of human values from historical documents and produces labeled training data for AI alignment research. **Verum** scores and certifies whether text demonstrates those values under pressure.

A value stated in comfort is weak signal. A value demonstrated at personal cost, under threat, under pressure, against interest, is strong signal. Ethos measures that cost. Verum certifies it.

---

## What Makes This Different

Most value-alignment work in AI is built on shaky ground:

- Human preference rankings capture what people *say* they value, not what they *do*
- Synthetic datasets are another model's opinion dressed up as ground truth
- RLHF measures preference, not principle
- Constitutional AI encodes rules from the top down, not evidence from the bottom up

Ethos goes to the source: documented human behavior under real conditions. The resistance score measures what nobody else is measuring systematically, the cost of holding a value when it would be easier not to.

When Ethos flags a value signal, the answer to "why?" traces back to a real person, a real moment, and a real cost.

---

## The 15 Values

| Value | What it means in practice |
|-------|--------------------------|
| **Integrity** | Honest and truthful. Says what is real, refuses deception even at cost to self. |
| **Courage** | Acts despite fear. Faces difficulty, speaks unpopular truths, accepts loss. |
| **Compassion** | Responds to suffering. Prioritizes others' wellbeing over personal convenience. |
| **Resilience** | Continues through adversity. Rebuilds after failure, does not stop when it is hard. |
| **Patience** | Waits without forcing. Allows things to unfold at the pace they require. |
| **Humility** | Acknowledges limitation. Defers when others know better, gives credit, admits error. |
| **Fairness** | Applies consistent standards. No favoritism, same measure for all parties. |
| **Loyalty** | Keeps faith with commitments and people. Does not abandon under pressure. |
| **Responsibility** | Owns outcomes. Accepts consequences, does not deflect, stays to the end. |
| **Growth** | Transforms through experience. Allows past errors to change present understanding. |
| **Independence** | Acts on own judgment. Does not require permission to do what conscience demands. |
| **Curiosity** | Pursues understanding. Follows questions past convenience. |
| **Commitment** | Sees things through. Stays when it costs something. |
| **Love** | Acts for others' wellbeing. Prioritizes the beloved over self. |
| **Gratitude** | Recognizes what was given. Carries the debt of others' generosity forward. |

---

## How It Works

### 1. Ingestion

Text is segmented into sentence-bounded passages. Each passage is stored with metadata:

- **Document type** (journal, letter, speech, action) affects resistance scoring. Private writing scores higher than public speech. Documented behavior scores highest.
- **Source authenticity** tracks whether the text is original (1.0), translated (0.85), or uncertain (0.70).
- **Publication year** drives a temporal discount for archaic texts.

### 2. Multi-Layer Extraction

Each passage runs through up to seven independent extraction layers:

| Layer | What it does | Requires |
|-------|-------------|----------|
| **L1 Keywords** | 15 value vocabularies with context disambiguation | Nothing (stdlib) |
| **L1b Lexicons** | MFD2.0 (2,041 entries) + MoralStrength (452 entries) | Bundled data |
| **L1c Phrase** | Pronoun-aware agency detection | Nothing (stdlib) |
| **L2 Semantic** | BGE-large-en-v1.5 embeddings against 322 prototypes | sentence-transformers |
| **L3a Structural** | Adversity, agency, resistance, and stakes patterns | Nothing (stdlib) |
| **L3b Zero-shot** | DeBERTa entailment against per-value hypotheses | transformers |
| **L3c MFT** | Moral Foundations Theory classification (10 labels) | transformers |

The pipeline degrades gracefully. If ML dependencies are absent, the keyword, lexicon, phrase, and structural layers run on stdlib alone. Each layer is independent. Agreement across layers strengthens confidence.

### 3. Resistance Scoring

```
resistance = base + significance_bonus + doc_type_bonus + text_tier_bonus
```

Text tier bonuses are awarded based on what the passage contains:

| Tier | Trigger | Bonus |
|------|---------|-------|
| **A: Mortal Stakes** | Death, execution, "I would rather die" | +0.62 |
| **B: High Adversity** | Imprisonment, exile, persecution, direct threats | +0.36 |
| **C: Standard Adversity** | "despite", "even though", "scared but", "at a cost" | +0.24 |

**Hold markers** ("nevertheless", "stood firm", "refused to yield") add +0.05 per tier.
**Failure markers** ("gave in", "surrendered", "backed down") suppress Tiers B and C.

### 4. Classification

Each observation receives a label:

| Label | Meaning | Threshold |
|-------|---------|-----------|
| **P1** | Value held under meaningful resistance | resistance >= 0.55 |
| **P0** | Value failed or corrupted | resistance <= 0.35 |
| **APY** | Answer-Pressure Yield: abandoned under external pressure | Pressure markers + failure |
| **AMBIGUOUS** | Between thresholds, insufficient evidence either way | 0.35 < resistance < 0.55 |

### 5. Registry Weight

The strength of a value in a figure's profile:

```
weight = demonstrations x avg_significance x avg_resistance x consistency
```

Consistency is a 4-component score:
- **Volume** (30%): saturates at 10 observations
- **Resistance stability** (30%): low variance in resistance scores
- **Temporal spread** (25%): observations spread over time
- **Source diversity** (15%): multiple document types

---

## Quick Start

**Requirements:** Python 3.10+

```bash
pip install -r requirements.txt
```

### Ingest a figure

```bash
python -m cli.ingest --figure gandhi --file gandhi_journal.txt \
    --doc-type journal --pronoun he

# With translation penalty and publication year
python -m cli.ingest --figure seneca --file seneca_letters.txt \
    --doc-type letter --pub-year 65 --translation --pronoun he

# Preview segmentation without writing
python -m cli.ingest --figure lincoln --file lincoln_speech.txt \
    --doc-type speech --pronoun he --dry-run
```

### Export labeled training data

```bash
# Export all figures
python -m cli.export

# With quality filters
python -m cli.export --min-consistency 0.3 --min-observations 3

# Single figure, dry run
python -m cli.export --figure gandhi --dry-run
```

Output is JSONL with P1/P0/APY labels, resistance scores, and source metadata.

### Batch ingestion

```bash
python -m cli.batch_ingest --manifest samples/manifest_example.json
```

### Corpus statistics

```bash
python -m cli.corpus_stats
python -m cli.corpus_stats --json
python -m cli.dataset_card    # HuggingFace format
```

### Start the API

```bash
python -m api.server
# http://localhost:8000
# Interactive docs at http://localhost:8000/docs
```

---

## API Endpoints

### Ethos

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/figures/{name}/ingest` | Ingest text for a figure |
| `GET` | `/figures` | List all ingested figures |
| `GET` | `/figures/{name}/profile` | Value profile for a figure |
| `GET` | `/figures/universal` | Cross-figure aggregate |
| `POST` | `/export/ric` | Export labeled training data |
| `GET` | `/health` | Health check |

### Verum (integrated)

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/verum/values` | List all 15 values with descriptions |
| `POST` | `/verum/score` | Score a text for value alignment |
| `POST` | `/verum/certify` | Issue a signed certificate |
| `GET` | `/verum/certificate/{cert_id}` | Retrieve a certificate |
| `GET` | `/verum/certificates` | List certificates |

---

## Training Data Format

Each exported record is a JSONL line:

```json
{
  "figure": "gandhi",
  "value_name": "integrity",
  "text_excerpt": "...was immense, but I still believed that truth mattered more...",
  "document_type": "journal",
  "significance": 0.90,
  "resistance": 1.0,
  "label": "P1",
  "label_reason": "high_resistance_hold_marker",
  "training_weight": 1.26,
  "confidence": 0.90,
  "hold_markers": ["still", "despite"],
  "failure_markers": [],
  "disambiguation_confidence": 1.0,
  "observation_consistency": 0.85
}
```

---

## Design Principles

1. **Behavioral evidence over hypotheticals.** Extract from documented history, not surveys or preferences.
2. **Cost-weighted signal.** High resistance = high informational value. A value demonstrated under threat is worth more than a value stated in comfort.
3. **Document authenticity calibration.** Private writing > public speech. Observed behavior > stated belief.
4. **Deterministic reproducibility.** No randomness. Same input + same weights = same output.
5. **Multi-layer convergence.** Agreement from independent extraction layers = strong evidence.
6. **Spectrum principle.** Signal extracted from the full human range, not just heroes or villains.
7. **No LLM calls at base layer.** The optional comprehension panel is off by default.
8. **Append-only observations.** The historical record is immutable once written.

---

## Project Structure

```
core/           Extraction pipeline, storage, scoring (22 modules)
cli/            Command-line tools (ingest, export, batch, stats)
api/            FastAPI REST service + Verum endpoints
tests/          934 tests
data/           SQLite databases + bundled lexicons
samples/        Example manifests and test figures
static/         Verum product page
IP/             Research notes
```

---

## Testing

```bash
pytest tests/ -q
# 934 passed
```

---

## Documentation

| Document | Description |
|----------|-------------|
| `technical_ethos.md` | Full technical reference: formulas, schema, layer wiring |
| `technical_verum.md` | Verum technical reference: score formula, certificate signature, API |
| `OPERATIONS.md` | Operational guide: batch processing, export, API usage |
| `STATUS_ethos.md` | Phase-by-phase implementation status |
| `PAPER.md` | Research paper |

---

## About

Ethos and Verum were built by [ai-nhancement](https://github.com/ai-nhancement) as part of the AiMe project ecosystem.

| Product | Role |
|---------|------|
| **[AiMe](https://github.com/ai-nhancement/AiMe-public)** | How AI relates to a person |
| **Ethos** | What integrity looks like, extracted from the human record |
| **Verum** | Whether AI output meets that standard |

Ethos is the foundation. Verum is what you build on top of it. See the [full Verum documentation](VERUM.md) for the scoring formula, certification flow, and API.

> *"A value stated in comfort is weak signal. A value demonstrated at personal cost is strong signal."*
