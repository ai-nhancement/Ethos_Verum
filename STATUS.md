# Ethos — Project Status

> **System class:** Universal Value Extraction Pipeline (UVEP)
> **Core claim:** Human values are behavioral patterns — extractable from documented history using deterministic computation, verifiable through accumulation.
> **Destination:** A universal, empirically-derived ethics corpus built from the full human spectrum — not hypotheticals, not hagiography, not monsters alone.
>
> *"We didn't invent values. We extracted them from people who lived them — and people who didn't."*

**Last Updated:** 2026-03-04 — Phase 0 complete. Full standalone pipeline operational: ingest → extract → resistance score → P1/P0/APY classify → JSONL export. Decoupled from AiMe. Zero external dependencies.
**Version:** Living document — update whenever a phase completes or design changes.

---

## 0. The Claim

Most value-alignment datasets are built from the poles: celebrated virtuous figures and documented villains. That produces models that recognize virtue performance and villainy — not the moral complexity of actual human behavior.

**The gap:** the middle ground. JFK. MLK. Malcolm X. Churchill. Nixon. Oppenheimer. Figures of real consequence with asymmetric value profiles — high in some dimensions, failing in others, under real pressure across all of them.

Ethos fills that gap.

The pipeline works from **documented behavior under real conditions** — not hypotheticals, not preference rankings, not moral philosophy. A value stated in comfort is weak signal. A value demonstrated at cost — under threat, under pressure, against interest — is strong signal.

The resistance score measures that cost. The document type calibrates the authenticity of the evidence. The P1/P0/APY classification labels whether the value held.

---

## Status Key

| Symbol | Label | Meaning |
|--------|-------|---------|
| ✅ | Complete | Shipped, tested, operational |
| 🔄 | In Progress | Actively being built |
| ⏳ | Planned | Designed, not started |
| 💡 | Concept | Direction set, design open |

---

## Phase 0 — Foundation ✅ (2026-03-04)

**Goal:** Standalone pipeline. Ingest any figure → extract value signals → score resistance → export labeled training data. No AiMe dependency. No external dependencies.

| Component | Status | Notes |
|-----------|--------|-------|
| `core/config.py` | ✅ | Config dataclass, no file dependency |
| `core/document_store.py` | ✅ | `documents.db` — passage storage, watermarks |
| `core/value_store.py` | ✅ | `values.db` — observations, registry, figure_sources |
| `core/resistance.py` | ✅ | doc_type + significance + text markers formula |
| `core/value_extractor.py` | ✅ | Keyword extraction, watermark-gated, 15 values |
| `cli/ingest.py` | ✅ | Segment, store, extract, print profile |
| `cli/export.py` | ✅ | P1/P0/APY classify, JSONL export, report |
| Git repo | ✅ | Initialized, `.gitignore` in place |
| `technical.md` | ✅ | Full architecture reference |
| `STATUS.md` | ✅ | This document |

**What works today:**
```bash
# Ingest any figure from any UTF-8 text file
python -m cli.ingest --figure gandhi --file samples/gandhi.txt --doc-type journal

# Export P1/P0/APY training data for all ingested figures
python -m cli.export

# Preview segmentation without writing anything
python -m cli.ingest --figure lincoln --file samples/lincoln.txt --doc-type speech --dry-run
```

**Verified pipeline output** (Phase 0 smoke test — 6-passage mixed corpus):
- 8 P1 (held under resistance)
- 4 P0 (failed — humility correctly flagged self-confessed failures)
- 2:1 positive/negative ratio on a deliberately balanced sample

---

## Phase 1 — Semantic Extraction ⏳

**Goal:** Replace keyword vocabulary with embedding-based clustering. Capture values from passages where the exact keywords don't appear but the semantic meaning is present.

**Why this matters:**
- Keyword matching misses paraphrase, historical diction, translated text
- Embedding clustering can generalize across writing styles and eras
- Phase 0 keyword baseline validates the pipeline before the switch

| Component | Status | Notes |
|-----------|--------|-------|
| Passage embedding | ⏳ | `sentence-transformers`, BGE-base or similar |
| Value cluster vectors | ⏳ | 15 seed vectors from VALUE_VOCAB centroids |
| FAISS/Qdrant integration | ⏳ | ANN search against value cluster prototypes |
| Hybrid scoring | ⏳ | Keyword + semantic blend, configurable weight |
| Backward compatibility | ⏳ | Phase 0 observations remain valid |

---

## Phase 2 — API Layer ⏳

**Goal:** REST API wrapping the pipeline. Enables integration with other tools, web UI, and batch processing.

| Endpoint | Status | Notes |
|----------|--------|-------|
| `POST /figures/{name}/ingest` | ⏳ | Accepts text payload + doc_type |
| `GET /figures/{name}/profile` | ⏳ | Value registry as JSON |
| `GET /figures/universal` | ⏳ | Cross-figure aggregate |
| `GET /export/ric` | ⏳ | Trigger export, return report |
| `GET /figures` | ⏳ | List all ingested figures |

**Technology:** FastAPI. Same pattern as AiMe's `api/` modules.

---

## Phase 3 — Web Dashboard ⏳

**Goal:** Browser UI for corpus management, profile exploration, and training set building.

| Feature | Status | Notes |
|---------|--------|-------|
| Figure browser | ⏳ | Profile cards — value radar chart per figure |
| Corpus upload | ⏳ | Drag-and-drop text file + doc_type selector |
| Universal registry view | ⏳ | Cross-figure value heatmap |
| Training set builder | ⏳ | Filter by figure / value / doc_type / label |
| Observation inspector | ⏳ | Raw passage view with resistance + label |

---

## Phase 4 — Corpus Scale ⏳

**Goal:** Support large-scale corpus processing and HuggingFace-compatible dataset publishing.

| Feature | Status | Notes |
|---------|--------|-------|
| Batch ingestion CLI | ⏳ | `cli/batch_ingest.py` — process a directory |
| Multi-file figure support | ⏳ | Multiple doc_types per figure, additive |
| Corpus statistics | ⏳ | Value distribution, resistance distribution, vocabulary coverage |
| Dataset card generation | ⏳ | HuggingFace-compatible metadata |
| `datasets` library export | ⏳ | Compatible format for `load_dataset()` |

---

## Phase 5 — SRL Integration 💡

**Goal:** Port AiMe's Self-Reflection Layer to Ethos as an optional module — enabling behavioral trait compilation and the RIC gate for evaluating AI model outputs.

| Feature | Status | Notes |
|---------|--------|-------|
| `srl/claim_extractor.py` | 💡 | Assertion level, hedge detection, source impersonation risk |
| `srl/ric_gate.py` | 💡 | Ground coherence + claim coherence scoring |
| `srl/trait_compiler.py` | 💡 | Behavioral trait derivation from observation history |
| `srl/inflection_engine.py` | 💡 | Scope A/B behavioral guidance signals |

**Use case:** Evaluate AI model outputs against the Ethos value corpus. Detect when a model's claims conflict with documented historical behavior, or when its reasoning patterns match known failure modes.

---

## Design Decisions Log

### 2026-03-04 — Standalone architecture
**Decision:** Full decoupling from AiMe. No shared databases. No shared config. Own `documents.db`, own `values.db`. Resistance formula simplified to historical-only path (doc_type bonus replaces live portrait/SRL bonuses).
**Rationale:** AiMe is a single-user cognitive system. Ethos is a corpus pipeline. Different audiences, different deployment models, different operational requirements.

### 2026-03-04 — No LLM in extraction stack
**Decision:** All classification is deterministic keyword regex + SQLite arithmetic. No model call anywhere.
**Rationale:** Deterministic → reproducible. Reproducible → auditable. The moment you introduce a model, the pipeline becomes non-deterministic and non-auditable. For a training data generation tool, that is unacceptable.

### 2026-03-04 — Any figure, no pre-labeling
**Decision:** No figure is labeled positive or negative at ingestion time. Classification emerges from the data.
**Rationale:** The spectrum principle. Pre-labeling figures as "good" or "bad" defeats the purpose — it injects human reputation bias into what should be a behavioral measurement system.

### 2026-03-04 — Document type as first-class signal
**Decision:** Document type set at ingestion, propagates through resistance scoring and export weighting. Cannot be overridden post-ingestion without re-ingesting.
**Rationale:** Authenticity of evidence is a structural property, not a metadata tag. It must affect scoring at every stage.
