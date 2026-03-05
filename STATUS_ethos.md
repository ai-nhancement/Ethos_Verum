# Ethos — Project Status

> **System class:** Universal Value Extraction Pipeline (UVEP)
> **Core claim:** Human values are behavioral patterns — extractable from documented history using deterministic computation, verifiable through accumulation.
> **Destination:** A universal, empirically-derived ethics corpus built from the full human spectrum — not hypotheticals, not hagiography, not monsters alone.
>
> *"We didn't invent values. We extracted them from people who lived them — and people who didn't."*

**Last Updated:** 2026-03-05 — Phase 0 complete and operational. PAPER.md §7 expanded with five concrete technical gap analyses: cross-passage APY detection (§7.6), keyword context disambiguation (§7.7), value co-occurrence and tension modeling (§7.8), observation consistency scoring (§7.9), SRL claim-level validation integration (§7.10). Pipeline and docs reviewed against AiMe SCAL/SRL implementations for robustness parity.
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

## Phase 1 — Multilingual Ingestion ⏳

**Goal:** Score value signals in a passage's original language before translation loss corrupts the evidence. Extend the vocabulary with historical dialect synonyms across six language families.

**Why this matters:**
- Gandhi's private letters were in Gujarati. Translating before scoring destroys register and diction cues the resistance formula relies on.
- Historical English uses different vocabulary than modern English (`fortitude` → courage, `verity` → integrity).
- A pipeline that only works in modern English cannot build a universal corpus.

| Component | Status | Notes |
|-----------|--------|-------|
| Multilingual embedding model | ⏳ | LaBSE or mE5-large — 109-language sentence vectors |
| Parallel keyword lists | ⏳ | Greek / Latin / Arabic / Chinese / French / German value synonyms |
| Historical dialect expansion | ⏳ | `VALUE_VOCAB` extended with pre-20th-century English synonyms |
| Native-language ingestion path | ⏳ | `--lang` flag in `cli/ingest.py` selects embedding model |
| Dual-language agreement scoring | ⏳ | Score in original + translated; use max or agreement-weighted blend |

---

## Phase 2 — Hybrid Detection + Agreement Confidence ⏳

**Goal:** Replace keyword-only detection with a hybrid metric combining keyword signal and embedding signal. The agreement between them becomes a first-class quality signal.

**Design:**

```
keyword_signal   = keyword match score (0.0–1.0, current Phase 0 output)
embedding_signal = cosine similarity to value cluster centroid (0.0–1.0)

hybrid_score         = α × keyword_signal + (1 − α) × embedding_signal
agreement_confidence = 1.0 − |keyword_signal − embedding_signal|
```

`agreement_confidence` is high when both methods agree, low when they diverge. Low-confidence observations are flagged for human review rather than discarded.

| Component | Status | Notes |
|-----------|--------|-------|
| Passage embedding store | ⏳ | 15 value cluster centroids from VALUE_VOCAB seed passages |
| FAISS/Qdrant ANN search | ⏳ | Nearest value cluster per passage |
| `hybrid_score` field | ⏳ | Added to observation + export JSONL |
| `agreement_confidence` field | ⏳ | Added to observation + export JSONL |
| α tuning | ⏳ | Configurable in `core/config.py` — default α=0.5 |
| Backward compatibility | ⏳ | Phase 0 keyword observations remain valid |

---

## Phase 3 — Temporal Value Arcs ⏳

**Goal:** Track how a figure's values shift across time — life stages, decades, the period of peak influence — rather than collapsing all their writing into a single flat profile.

**Why this matters:**
- MLK's early writings express different priorities than his final years.
- Churchill's documented values during wartime differ from peacetime.
- Averaging across a life erases the arc. The arc is the evidence.

| Component | Status | Notes |
|-----------|--------|-------|
| Decade sub-sessions | ⏳ | `--era 1960s` flag → `session_id = figure:mlk:1960s` |
| `value_trajectory()` query | ⏳ | Returns ordered list of `(era, value_name, weight)` tuples |
| Peak period parameter | ⏳ | `--peak-era` flag — boosts significance weighting for that era |
| Temporal export mode | ⏳ | `cli/export.py --by-era` — separate JSONL per era per figure |
| Cross-era consistency | ⏳ | Add `temporal_consistency` column to registry — how stable across eras |

---

## Phase 4 — Corpus Balance Tool ⏳

**Goal:** Prevent corpus bias toward famous virtuous figures. The pipeline should produce training data that mirrors the distribution of documented human behavior — not the distribution of celebrated biography.

**Target ratio:** 1 unambiguously positive figure : 4 middle-ground / complex figures. Negative figures are included as needed to populate P0/APY labels.

**No pre-labeling invariant preserved:** the balance tool works on post-extraction profile shape (P1 rate, avg resistance), not on human-assigned labels at ingestion.

| Component | Status | Notes |
|-----------|--------|-------|
| `cli/balance.py` | ⏳ | Analyze corpus composition by P1-rate distribution |
| Corpus report | ⏳ | Shows figure type breakdown, value coverage, P1/P0/APY ratios |
| Inverse-frequency export weighting | ⏳ | Over-represented value types down-weighted in export |
| Balance target config | ⏳ | `corpus.target_spectrum_ratio` in `core/config.py` |
| Recommendation output | ⏳ | Suggests which figure types are missing from corpus |

---

## Phase 5 — API Layer ⏳

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

## Phase 6 — Web Dashboard ⏳

**Goal:** Browser UI for corpus management, profile exploration, and training set builder.

| Feature | Status | Notes |
|---------|--------|-------|
| Figure browser | ⏳ | Profile cards — value radar chart per figure |
| Temporal arc view | ⏳ | Per-figure value trajectory across eras |
| Corpus upload | ⏳ | Drag-and-drop text file + doc_type selector |
| Universal registry view | ⏳ | Cross-figure value heatmap |
| Training set builder | ⏳ | Filter by figure / value / doc_type / label / era |
| Observation inspector | ⏳ | Raw passage view with resistance + hybrid score + agreement confidence |

---

## Phase 7 — Corpus Scale + HuggingFace ⏳

**Goal:** Large-scale corpus processing and HuggingFace-compatible dataset publishing.

| Feature | Status | Notes |
|---------|--------|-------|
| Batch ingestion CLI | ⏳ | `cli/batch_ingest.py` — process a directory |
| Multi-file figure support | ⏳ | Multiple doc_types per figure, additive |
| Corpus statistics | ⏳ | Value distribution, resistance distribution, vocabulary coverage |
| Dataset card generation | ⏳ | HuggingFace-compatible metadata |
| `datasets` library export | ⏳ | Compatible format for `load_dataset()` |

---

## Phase 8a — Cross-Passage APY Detection ⏳

**Goal:** Detect APY events that span multiple passages — pressure in one passage, failure in a later one. Currently these are classified P0 + AMBIGUOUS; the most informative APY label is silently lost.

**Why this matters:**
- Historical APY is structurally cross-passage. Pressure and response rarely co-occur in 450 characters.
- Lincoln documents political pressure in a journal entry; the corresponding reversal is in a speech weeks later. The current pipeline sees two unrelated events.
- Deferred APY (held initially, failed later) is a distinct behavioral signal from immediate yield.

| Component | Status | Notes |
|-----------|--------|-------|
| `apy_context` table in `values.db` | ⏳ | `(session_id, record_id, ts, markers)` — rolling buffer per figure |
| Context window lookup in `classify_observation()` | ⏳ | Check context buffer before failure-marker branch |
| `pressure_source_id` field in export JSONL | ⏳ | Links failure record back to its pressure source passage |
| `deferred_apy_lag` field | ⏳ | Passage count or time between pressure and failure |
| Context window config | ⏳ | N=5 undated / 72h dated — configurable in `core/config.py` |

---

## Phase 8b — Keyword Context Disambiguation ⏳

**Goal:** Add a second-pass disambiguation filter to reduce false positives from context-free keyword matching. Words like `afraid`, `fair`, `patient`, `love`, `promise` fire on correct keywords in wrong semantic contexts.

**Why this matters:**
- "I was afraid of being late" fires `courage`. Zero courage content.
- "My patient recovered" fires `patience`. Word used as noun.
- "This promises to be interesting" fires `commitment`. No commitment expressed.
- These inflate P1 counts and lower average resistance scores across the corpus.

| Component | Status | Notes |
|-----------|--------|-------|
| Grammatical role filter | ⏳ | Regex disqualifying patterns per value (nominal/idiomatic uses) |
| First-person grounding check | ⏳ | Agent presence within token window of keyword |
| Action-evidence list per value | ⏳ | Short action word list per value — presence boosts, absence flags |
| `disambiguation_confidence` field | ⏳ | Added to `value_observations` + export JSONL [0.0, 1.0] |
| `--min-disambiguation` export flag | ⏳ | Filter low-confidence observations without re-ingesting |

---

## Phase 8c — Value Co-occurrence and Interaction Modeling ⏳

**Goal:** Model relationships between values — co-occurrence patterns, value tension events (where one value was held by sacrificing another). Values are not independent dimensions; their interactions are behavioral signal.

**Why this matters:**
- Courage + integrity co-occur in resistance-to-authority passages — coupled examples are higher-information training records.
- Independence vs. loyalty, fairness vs. compassion — documented tension events reveal value hierarchy, not just value presence.
- Value tension events (one value held, one failed) are the highest-value training records in the corpus.

| Component | Status | Notes |
|-----------|--------|-------|
| Co-occurrence matrix at export | ⏳ | Per-pair (P1+P1 / P1+P0 / P0+P0) counts per figure and cross-figure |
| `value_tension` table in `values.db` | ⏳ | `(session_id, record_id, ts, value_held, value_failed, resistance, text_excerpt)` |
| Tension pair list config | ⏳ | 5 default pairs in `core/config.py` (independence/loyalty, fairness/compassion, etc.) |
| `--value-tension` export flag | ⏳ | Outputs `ric_value_tensions.jsonl` with highest training weights |
| Co-occurrence report in `ric_historical_report.json` | ⏳ | 105-pair interaction matrix |

---

## Phase 8d — Observation Consistency Scoring ⏳

**Goal:** Add a consistency score to the value registry that captures the *distribution* of resistance evidence across observations, not just the cumulative weight. 3 observations in one speech ≠ 45 observations across 30 years.

**Formula:**
```
consistency = min(1.0,
    0.30 × min(1.0, n / 10)              ← observation volume (saturates at 10)
  + 0.30 × (1.0 − σ_r / 0.40)           ← resistance stability (low variance = high consistency)
  + 0.25 × min(1.0, span_s / 31536000)  ← temporal spread (saturates at 1 year)
  + 0.15 × min(1.0, doc_types / 3)      ← source diversity (saturates at 3 types)
)
```

| Component | Status | Notes |
|-----------|--------|-------|
| `consistency` column in `value_registry` | ⏳ | Computed at `upsert_registry()` call time |
| `observation_consistency` field in export JSONL | ⏳ | Propagated from registry |
| `--min-consistency` export flag | ⏳ | Excludes low-consistency observations from training set |
| Divergence detection | ⏳ | Flags figures with high weight but low consistency (persona/behavior divergence signal) |

---

## Phase 8 — SRL Integration 💡

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

### 2026-03-04 — Original-language scoring (Phase 1)
**Decision:** Value detection will target source-language text using multilingual embeddings (LaBSE/mE5), not only modern English translations.
**Rationale:** Translation degrades diction, register, and emotional precision — all of which the resistance formula depends on. A Gandhi journal entry in Gujarati carries richer signal than an English translation of it. The pipeline must honor that.

### 2026-03-04 — Hybrid detection with agreement confidence (Phase 2)
**Decision:** Phase 2 will run both keyword matching and embedding cosine similarity, then report `hybrid_score = α×keyword + (1−α)×embedding` and `agreement_confidence = 1 − |keyword_signal − embedding_signal|`.
**Rationale:** Neither method alone is sufficient. Keywords miss paraphrase; embeddings miss precision. Agreement confidence turns the disagreement between methods into a signal — low-confidence observations get flagged for review rather than silently included.

### 2026-03-04 — Temporal sub-sessions (Phase 3)
**Decision:** Support `session_id = figure:mlk:1960s` sub-sessions for era-partitioned value extraction.
**Rationale:** Collapsing a person's entire life into one profile erases value evolution. The arc of change across life stages is itself a training signal for models that need to understand moral development, not just moral state.

### 2026-03-04 — Corpus balance ratio (Phase 4)
**Decision:** Target 1:4 unambiguous positive-to-middle-ground figure ratio. Balance tool operates on post-extraction P1-rate, not on pre-ingestion human labels.
**Rationale:** Corpus composition bias toward celebrated virtuous figures produces training data that reflects reputation, not behavioral reality. The balance tool corrects this structurally. No figure is labeled at ingestion — that invariant is preserved.
