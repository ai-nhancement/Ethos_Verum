# Ethos — Project Status

> **System class:** Universal Value Extraction Pipeline (UVEP)
> **Core claim:** Human values are behavioral patterns — extractable from documented history using deterministic computation, verifiable through accumulation.
> **Destination:** A universal, empirically-derived ethics corpus built from the full human spectrum — not hypotheticals, not hagiography, not monsters alone.
>
> *"We didn't invent values. We extracted them from people who lived them — and people who didn't."*

**Last Updated:** 2026-03-16 (Phase 6 complete)
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

**Goal:** Standalone pipeline. Ingest any figure → extract value signals → score resistance → export labeled training data. No AiMe dependency. No external dependencies beyond Python stdlib.

| Component | Status | Notes |
|-----------|--------|-------|
| `core/config.py` | ✅ | Config dataclass + VALUE_TENSION_PAIRS + `is_tension_pair()` |
| `core/document_store.py` | ✅ | `documents.db` — passage storage, watermarks |
| `core/value_store.py` | ✅ | `values.db` — observations, registry, tension, APY context, figure_sources |
| `core/resistance.py` | ✅ | doc_type + significance + text markers formula |
| `core/apy_patterns.py` | ✅ | Shared APY pressure regex — single source of truth |
| `core/value_extractor.py` | ✅ | Keyword extraction, disambiguation, APY context writing, watermark-gated |
| `cli/ingest.py` | ✅ | Segment, store, extract, print profile |
| `cli/export.py` | ✅ | P1/P0/APY classify, JSONL export, report, co-occurrence, consistency flag |

**What works today:**
```bash
# Ingest any figure from any UTF-8 text file
python -m cli.ingest --figure gandhi --file samples/gandhi.txt --doc-type journal

# Export P1/P0/APY training data for all ingested figures
python -m cli.export

# Preview segmentation without writing anything
python -m cli.ingest --figure lincoln --file samples/lincoln.txt --doc-type speech --dry-run

# Export with consistency filter
python -m cli.export --min-consistency 0.3
```

**Verified pipeline output** (Phase 0 smoke test — 6-passage mixed corpus):
- 8 P1 (held under resistance)
- 4 P0 (failed — humility correctly flagged self-confessed failures)
- 2:1 positive/negative ratio on a deliberately balanced sample

---

## Quality Extensions ✅ (complete, 213 tests passing)

These four extensions were built on top of the Phase 0 base pipeline. All are integrated into the live extraction and export paths. All tests pass.

### QX-A — Keyword Context Disambiguation ✅

**Problem:** Simple substring matching produces false positives ("my patient" → patience, "love music" → love, "fair trade" → fairness).

**Solution:** `_check_signal()` in `value_extractor.py` applies two filters before accepting a keyword hit:
1. Per-value `_DISQUALIFIERS` regex — drops hits where the surrounding context window matches a known false-positive pattern (11 values covered)
2. `_REQUIRES_FIRST_PERSON` proximity check — 9 values require a first-person pronoun within ±80 chars; `action` doc_type bypasses this (biographical third-person is valid)

**Output:** Each observation carries a `disambiguation_confidence` score (1.0 / 0.7 / 0.6) reflecting attribution strength.

**Tests:** `tests/test_disambiguation.py` — 517 lines

---

### QX-B — Value Co-occurrence and Tension Modeling ✅

**Problem:** Values don't appear in isolation. Passages often show two values simultaneously — sometimes reinforcing, sometimes in tension (courage vs. patience, independence vs. loyalty).

**Solution:**
- `VALUE_TENSION_PAIRS` in `config.py` — 5 researcher-configurable tension pairs derived from Schwartz (1992) and Rokeach (1973)
- `is_tension_pair(v1, v2)` — O(1) lookup via frozenset
- `value_tension` table in `values.db` — records when two tension-pair values appear in the same passage
- `record_tension()` / `get_tensions()` in `value_store.py`
- `cli/export.py` emits `ric_value_tensions.jsonl` when `--value-tension` flag is set; co-occurrence stats in report JSON

**Tests:** `tests/test_cooccurrence.py` — 526 lines

---

### QX-C — Cross-Passage APY Detection ✅

**Problem:** APY (Answer-Pressure Yield) events sometimes span multiple passages — pressure appears in passage N, failure appears in passage N+2. Single-passage classification misses these.

**Solution:**
- `apy_context` table in `values.db` — rolling window of passages containing pressure markers (default window: 5 passages)
- `write_apy_context()` in `value_store.py` — written during ingestion whenever `APY_PRESSURE_RE` matches
- `cli/export.py` queries the `apy_context` window during classification — if pressure was detected within the previous N passages, the current passage gets a deferred APY upgrade if failure markers are present
- Output fields: `pressure_source_id`, `deferred_apy_lag`

**Tests:** `tests/test_cross_passage_apy.py` — 462 lines

---

### QX-D — Observation Consistency Scoring ✅

**Problem:** A figure who demonstrates `integrity` once under extreme pressure is different from one who demonstrates it a hundred times in comfort. The registry weight needs to encode both.

**Solution:** 4-component `_compute_consistency()` in `value_store.py`:

| Component | Weight | What it measures |
|-----------|--------|-----------------|
| Observation volume | 0.30 | `min(1, n / 10)` — saturates at 10 observations |
| Resistance stability | 0.30 | `max(0, 1 − σ_r / 0.40)` — low variance across observations = high stability |
| Temporal spread | 0.25 | `min(1, span / 1yr)` — value demonstrated across time, not a single document |
| Source diversity | 0.15 | `min(1, distinct_doc_types / 3)` — demonstrated in multiple doc types |

Consistency feeds directly into the registry weight formula:
```
weight = demonstrations × avg_significance × avg_resistance × consistency
```

`cli/export.py` supports `--min-consistency` filter for downstream training set quality control.

**Tests:** `tests/test_consistency.py` — 593 lines

---

## Phase 1 — Semantic Extraction ✅ (2026-03-16)

**Goal:** Add embedding-based prototype matching alongside keyword detection. Capture values from passages where exact keywords don't appear but the semantic meaning is present.

| Component | Status | Notes |
|-----------|--------|-------|
| `core/embedder.py` | ✅ | `BAAI/bge-large-en-v1.5` (1024d) singleton, lazy-load, fail-open |
| `core/value_seeds.py` | ✅ | 322 seed sentences across 15 values — modern + archaic register |
| `core/semantic_store.py` | ✅ | Qdrant collection `ethos_value_prototypes`, `build_prototypes()` + `query_points()` |
| `cli/build_prototypes.py` | ✅ | CLI to embed seeds and load all 15 prototypes — 15/15 stored |
| Hybrid scoring | ✅ | Keyword + semantic merge in `_run_extraction()`: agreement boosts confidence; semantic-only detections added as new signals |
| Archaic seed coverage | ✅ | All 15 values include archaic vocabulary (probity, fortitude, valour, etc.) |
| Backward compatibility | ✅ | Semantic layer is additive — pipeline degrades to keyword-only if model unavailable |

**Sanity check results (8 test passages):**
- 6/8 top-1 prototype correct at threshold=0.0
- 2 misses are real prototype boundary cases (probity/independence, fortitude/compassion) — both correctly caught by keyword layer
- Typical cosine similarity scores: 0.69–0.83 for correct matches
- Config threshold `semantic_threshold=0.45` calibrated for semantic-only detection (keyword layer gates primary extraction)

**Signal merge logic:**
- Both layers fire on same value → keyword signal kept, `disambiguation_confidence` boosted by `semantic_score × 0.20`, source tagged `keyword+semantic`
- Only semantic fires → added as new signal with source `semantic`
- Only keyword fires → kept as-is with source `keyword`

---

## Phase 2 — Structural / Independent Classifiers ✅ (2026-03-16)

**Goal:** Third and fourth independent signal layers that validate value detections without using Ethos vocabulary.

**Why this matters:** When Layer 1 (keyword), Layer 2 (embedding), and Layer 3 (independent classifiers) all agree on a value signal, the resistance score has convergent evidence from models that had no knowledge of Ethos vocabulary. This directly answers the construct validity question.

| Component | Status | Notes |
|-----------|--------|-------|
| Structural adversity patterns | ✅ | `core/structural_layer.py` — pure regex, zero deps. 4 pattern classes: adversity context, first-person agency, resistance-to-failure, stakes vocabulary. Returns structural_score in [0.0, 1.0]. |
| Zero-shot value hypotheses | ✅ | `MoritzLaurer/deberta-v3-large-zeroshot-v2.0` — 15 per-value hypothesis strings. Entailment probability as confidence. Lazy-load, fail-open. |
| Two-threshold detection | ✅ | `zeroshot_threshold=0.35` for agreement boost (L1/L2 detected it); `zeroshot_standalone_threshold=0.70` for zero-shot-only detections. Prevents false positives. |
| Signal pooling | ✅ | Structural score → resistance/confidence boost on all existing signals. Zero-shot agreement → per-value confidence boost. New zero-shot signals appended to merged list. Source tags: `keyword+semantic+structural+zeroshot`. |
| MFD2.0 lexicon | ✅ | 2,041 entries. `data/lexicons/mfd2.txt`. Bundled — no network dep at runtime. |
| MoralStrength lexicon | ✅ | 452 entries across 5 foundation TSVs. `data/lexicons/moralstrength_*.tsv`. Continuous 1–10 moral strength scores. |
| Lexicon layer (Layer 1b) | ✅ | `core/lexicon_layer.py`. Loaded at import. MFT→Ethos mapping. MoralStrength boosts confidence up to +0.15 when it agrees with MFD2.0. Virtue signals merged into main pipeline; vice signals flagged for future P0 detection. 27 tests. |
| MFT classifier (Layer 3c) | ✅ | `MMADS/MoralFoundationsClassifier` — fine-tuned RoBERTa, 10 MFT labels (5 foundations × virtue/vice). `core/mft_classifier.py`. Lazy-load singleton, fail-open. `mft_enabled`, `mft_min_virtue_score=0.60`, `mft_standalone_threshold=0.80` in config. |
| MFT→Ethos mapping | ✅ | `MFT_VIRTUE_TO_ETHOS`: care→{compassion,love,gratitude}, fairness→{fairness,responsibility,integrity}, loyalty→{loyalty,courage,commitment,resilience}, authority→{responsibility,humility}, sanctity→{integrity,commitment}. authority.vice above threshold → independence hint (discounted 0.70×). |
| MFT integration | ✅ | Layer 3c in `_run_extraction()`: virtue boosts (+`mft_agreement_boost=0.10`) applied when MFT agrees with L1/L2; standalone signals created when `score ≥ mft_standalone_threshold`. Vice flags noted but never written as P1 observations. Source tag: `+mft`. |

**Note on spaCy:** Incompatible with Python 3.14 (pydantic v1 `ConfigError`). Structural patterns implemented with pure regex instead — more portable, no dependency on model downloads for structural detection.

**Phase 2 Critical Review — Hardening (post-completion):**

Five categories of defects found and fixed during independent critical review:

1. **Regex dead code and ambiguity** (`structural_layer.py`):
   - `_ADVERSITY_RE`: `despite` appeared twice — once as a bare word and once in a `despite\s+(?:the\s+)?(?:threat|...)` clause. Removed duplicate.
   - `_STAKES_RE`: `imprisonment` appeared twice in the same alternation. Removed duplicate.
   - `_AGENCY_RE`: `chose?` matched `"chos"` (not a word) — `?` was applied to `e`. Fixed to `chose`. `stood\s+firm` was dead code because `stood` appeared earlier in the same alternation and subsumes it. Removed redundant alternative.

2. **Thread safety** (`lexicon_layer.py`):
   - `_ensure_loaded()` had no lock. Two concurrent threads could both pass `if _LOADED: return` before either set `_LOADED = True`, causing both lexicons to be loaded twice. Fixed with standard double-checked locking (`threading.Lock()`).

3. **HuggingFace truncation** (`mft_classifier.py`):
   - `clf(text[:512])` truncates at 512 *characters* ≈ 128 tokens. For passages of several hundred words this silently discards 75%+ of context. Fixed: `clf(text, truncation=True, max_length=512)` — tokenizer-level truncation at 512 tokens.

4. **Resistance computation in signal loop** (`value_extractor.py`):
   - `compute_resistance()` depends only on `text`, `significance`, `doc_type` — all passage-scoped. It was called once per signal inside the loop (N calls for N signals in the same passage). Fixed: single call above the loop, result guards the entire write block.

5. **MFT mapping consistency** (`lexicon_layer.py` and `mft_classifier.py`):
   - Both modules map MFT foundations to Ethos values independently. After review they were inconsistent: `authority` mapped to `[responsibility, commitment]` in lexicon vs `[humility, responsibility, patience]` in classifier. Standardized to `[responsibility, humility]` in both. `patience` and `commitment` are not MFT authority-foundation values.

6. **Signal schema consistency** (`value_extractor.py`):
   - MFT standalone signal dicts were missing `text_excerpt` (required by `record_observation()`) and included spurious `label` and `record_id` fields. Fixed to match the standard 5-field schema: `value_name`, `text_excerpt`, `significance`, `disambiguation_confidence`, `source`.

7. **`multi_label` constructor/call contradiction** (`structural_layer.py`):
   - `_get_zeroshot_pipeline()` constructed the pipeline with `multi_label=False`, but `zeroshot_scores()` called it with `multi_label=True`. For 15 independent value hypotheses, `multi_label=True` (independent sigmoid) is correct — `multi_label=False` applies softmax and divides probability mass across all 15 labels. Fixed constructor to match.

**New test coverage added during review:**
- `tests/test_structural_layer.py` — 98 tests (structural layer had NO dedicated tests before review). Every regex clause, all 5 score levels, fail-open, VALUE_HYPOTHESES completeness invariant.

**Verified results (test suite):**
- Structural patterns: adversity, agency, resistance, stakes correctly detected
- Zero-shot DeBERTa: integrity=0.993, courage=0.995, responsibility=1.000 on strong passages
- Mundane passages correctly produce no detections at standalone threshold
- MFT classifier: LABEL_4 (loyalty.virtue) → courage/loyalty confirmed at 0.9999; LABEL_0 (care.virtue) → compassion; vice flags fire correctly; neutral text produces no boosts at high threshold
- **553 tests passing** (247 base + 36 MFT + 98 structural + 172 prior suite)

---

## Phase 3 — Translation and Temporal Handling ✅ (2026-03-16)

**Goal:** Calibrate scores for documents that are not in their original language or original era.

| Feature | Status | Notes |
|---------|--------|-------|
| Language detection at ingest | ✅ | `langdetect` (seeded for determinism) — auto-detects language of source text at ingest |
| Translation penalty | ✅ | `source_authenticity` multiplier: original=1.0 / known translation=0.85 / uncertain=0.70. Stored per-passage in `documents.db`. Applied to all signal confidence scores at extraction time. |
| Archaic preprocessing | ✅ | `preprocess_archaic()` — 26 deterministic replacement rules (thou/thee/thy/hath/doth/art/'tis/'twas/shouldst/etc. → modern equivalents). Applied to text before embedding. Also applied to source text before segmentation when `pub_year < 1850`. |
| `pub_year` temporal discount | ✅ | 5-tier multiplier: pre-1400=0.70 / 1400-1600=0.80 / 1600-1700=0.88 / 1700-1800=0.93 / 1800-1850=0.97 / 1850+=1.0. Applied to all signal confidences alongside source_authenticity. |
| `--translation` CLI flag | ✅ | `python -m cli.ingest ... --translation` declares the document is a known translation. |
| `documents.db` Phase 3 columns | ✅ | `source_lang TEXT`, `source_authenticity REAL`, `pub_year INTEGER`. Migration-safe (existing DBs upgraded automatically). |
| Translation fidelity scoring | ⏳ | `LaBSE` — when original + translation both available (future) |
| Non-English zero-shot | ⏳ | `mDeBERTa-v3-base-mnli-xnli` — apply value hypotheses without translating (future) |

**Verified smoke test:**
```
Seneca (Latin, known translation, 65 AD):  confidence 0.90 → 0.535  (×0.85 ×0.70)
Lincoln (English original, 1863):          confidence 0.90 → 0.900  (×1.0  ×1.0)
Archaic: "Thou hast spoken truly. Doth thy conscience not trouble thee?"
Modern:  "You have spoken truly. Does your conscience not trouble you?"
```

**CLI usage:**
```bash
python -m cli.ingest --figure seneca --file samples/seneca.txt --doc-type letter --pub-year 65 --translation
python -m cli.ingest --figure lincoln --file samples/lincoln.txt --doc-type speech --pub-year 1863
```

**Tests:** `tests/test_temporal_layer.py` — 46 tests passing

---

## Phase 4 — REST API ✅ (2026-03-16)

**Goal:** FastAPI service wrapping the pipeline for integration with other tools, web UI, and batch processing.

| Component | Status | Notes |
|-----------|--------|-------|
| `core/pipeline.py` | ✅ | Shared ingest/query logic — `ingest_text()`, `figure_profile()`, `universal_profile()`. Used by both CLI and API. |
| `api/models.py` | ✅ | Pydantic request/response models: `IngestRequest/Response`, `FigureProfileResponse`, `FigureListResponse`, `UniversalProfileResponse`, `ExportRequest/Response` |
| `api/app.py` | ✅ | FastAPI app. 5 endpoints + health. Route order correct (`/universal` before `/{name}/profile` to prevent shadowing). |
| `api/server.py` | ✅ | Uvicorn development entrypoint: `python -m api.server` |
| `POST /figures/{name}/ingest` | ✅ | Ingest text payload, returns session stats. Delegates to `core.pipeline.ingest_text()`. |
| `GET /figures` | ✅ | List all ingested figures with passage counts and metadata. |
| `GET /figures/universal` | ✅ | Cross-figure value aggregate. Query param: `min_demonstrations`. |
| `GET /figures/{name}/profile` | ✅ | Per-figure value registry sorted by weight. Query param: `min_demonstrations`. Returns 404 for unknown figures. |
| `POST /export/ric` | ✅ | Triggers `cli.export` subprocess. Returns P1/P0/APY counts. Supports `dry_run`. |
| `GET /health` | ✅ | Liveness check. |

**Usage:**
```bash
# Start development server (hot-reload)
python -m api.server

# Or with uvicorn directly
uvicorn api.app:app --port 8000 --reload

# Interactive docs at:
# http://127.0.0.1:8000/docs
```

**Example API calls:**
```bash
# Ingest
curl -X POST http://localhost:8000/figures/gandhi/ingest \
  -H "Content-Type: application/json" \
  -d '{"text": "He walked toward the sea...", "doc_type": "journal"}'

# Profile
curl http://localhost:8000/figures/gandhi/profile

# Universal registry
curl http://localhost:8000/figures/universal?min_demonstrations=3

# Export (dry run)
curl -X POST http://localhost:8000/export/ric \
  -H "Content-Type: application/json" \
  -d '{"dry_run": true}'
```

**Tests:** `tests/test_api.py` — 42 tests passing (27 mocked contract + 11 real pipeline integration + 4 routing + validation).

---

## Phase 6 — Corpus Scale ✅ (2026-03-16)

**Goal:** Make the pipeline usable at real corpus scale — batch ingestion from a manifest file, multi-file per figure, corpus-level statistics, and HuggingFace dataset card generation.

| Component | Status | Notes |
|-----------|--------|-------|
| `core/value_store.py` — multi-file fix | ✅ | `register_figure_source` UPSERT now accumulates `passage_count` across multiple ingests. Mixed `doc_type` detected: second ingest with different doc_type sets `document_type='mixed'`. |
| `core/corpus.py` | ✅ | Corpus-level stats queries: `get_overview()`, `get_figure_summaries()`, `get_value_distribution()`, `get_resistance_distribution()`, `get_cross_figure_values()`, `get_full_report()`. Used by CLI and future API/dashboard. |
| `cli/batch_ingest.py` | ✅ | Manifest-driven batch ingestion. All files per figure ingested with `run_extract=False`; extraction runs once per figure at end. Continues on per-file errors. Reports failure count and exits non-zero if any source failed. |
| `cli/corpus_stats.py` | ✅ | Human-readable and JSON corpus report. Coverage rate, value distribution, resistance distribution histogram, cross-figure values table. |
| `cli/dataset_card.py` | ✅ | Auto-generates a HuggingFace-compatible dataset card (YAML frontmatter + markdown) from live corpus stats. |
| `samples/manifest_example.json` | ✅ | Example manifest with 3 figures (gandhi, lincoln, seneca), multiple doc_types, pub_years, and translation flags. |

**Usage:**
```bash
# Batch ingest a full corpus from a manifest
python -m cli.batch_ingest --manifest corpus/manifest.json

# Preview segmentation without writing
python -m cli.batch_ingest --manifest corpus/manifest.json --dry-run

# Ingest a single figure from a manifest
python -m cli.batch_ingest --manifest corpus/manifest.json --figure gandhi

# Print corpus statistics
python -m cli.corpus_stats

# Export as JSON
python -m cli.corpus_stats --format json --output report.json

# Generate HuggingFace dataset card
python -m cli.dataset_card --output DATASET_CARD.md
```

**Multi-file example:**
```json
{
  "corpus_name": "historical_figures_v1",
  "default_significance": 0.90,
  "figures": [
    {
      "name": "gandhi",
      "sources": [
        {"file": "samples/autobiography.txt", "doc_type": "journal", "pub_year": 1927},
        {"file": "samples/speeches.txt",      "doc_type": "speech",  "pub_year": 1930},
        {"file": "samples/letters.txt",       "doc_type": "letter"}
      ]
    }
  ]
}
```

**Key fix:** Before this phase, ingesting a second file for the same figure silently overwrote `passage_count` with the new file's count, losing all prior ingests. The fix: `passage_count = figure_sources.passage_count + excluded.passage_count` in the UPSERT.

**Tests:**
- `tests/test_batch_ingest.py` — 30 tests: accumulation correctness, manifest loading, per-figure ingestion, multi-source extraction once, partial failure recovery, dry-run isolation, mixed doc_type detection, passage count accuracy
- `tests/test_corpus_stats.py` — 27 tests: overview stats, figure summaries, value distribution ordering, resistance histogram invariants, cross-figure filtering, dataset card structure

---

## Phase 5 — Web Dashboard ⏳

| Feature | Status |
|---------|--------|
| Figure browser — profile cards, value radar charts | ⏳ |
| Corpus upload UI | ⏳ |
| Universal registry visualization — cross-figure value heatmap | ⏳ |
| Training set builder — filter by figure / value / doc_type / label | ⏳ |

---

## Phase 6 — Corpus Scale ✅  (see above)

---

## Phase 7 — SRL Integration 💡

**Goal:** Port AiMe's Self-Reflection Layer to Ethos — enabling behavioral trait compilation and the RIC gate for evaluating AI model outputs.

**Use case:** Evaluate AI model outputs against the Ethos value corpus. Detect when a model's behavioral patterns match known failure modes.

---

## Design Decisions Log

### 2026-03-04 — Standalone architecture
**Decision:** Full decoupling from AiMe. No shared databases. No shared config.
**Rationale:** AiMe is a single-user cognitive system. Ethos is a corpus pipeline. Different audiences, different deployment models.

### 2026-03-04 — No LLM in extraction stack
**Decision:** All classification is deterministic keyword regex + SQLite arithmetic. No model call anywhere in Phase 0.
**Rationale:** Deterministic → reproducible. Reproducible → auditable. For a training data generation tool, non-determinism is unacceptable.

### 2026-03-04 — Any figure, no pre-labeling
**Decision:** No figure is labeled positive or negative at ingestion time. Classification emerges from the data.
**Rationale:** The spectrum principle. Pre-labeling injects reputation bias into what should be a behavioral measurement system.

### 2026-03-04 — Document type as first-class signal
**Decision:** Document type set at ingestion, propagates through resistance scoring and export weighting. Cannot be overridden post-ingestion without re-ingesting.
**Rationale:** Authenticity of evidence is a structural property, not a metadata tag.

### 2026-03-16 — Multi-layer extraction architecture
**Decision:** Keyword layer (Phase 0 + QX) is Layer 1. Semantic embedding (BGE-large + Qdrant) is Layer 2. Structural + independent classifiers (spaCy + MFT classifier + zero-shot DeBERTa) is Layer 3. Cross-passage coherence is Layer 4.
**Rationale:** Single-layer keyword detection is brittle. Multi-layer convergent evidence makes the resistance score defensible. When independent models agree, the signal is real. This directly answers the construct validity question.

### 2026-03-16 — Two-threshold zero-shot detection
**Decision:** Zero-shot uses `zeroshot_threshold=0.35` for agreement boost (where L1/L2 already detected a value) and `zeroshot_standalone_threshold=0.70` for independent detections.
**Rationale:** At threshold=0.35, "The meeting was fair" correctly scores 0.98 on fairness — the model is right, but that's not a strong behavioral signal. The higher standalone threshold requires the model to be very confident before adding a new detection without keyword/semantic support. Agreement boosts have no such restriction because L1/L2 already provide the primary detection.

### 2026-03-16 — Structural patterns over spaCy
**Decision:** Pure regex structural patterns instead of spaCy dependency parsing.
**Rationale:** spaCy pydantic v1 is incompatible with Python 3.14. More importantly, the structural signals Ethos needs (adversity markers, agency phrases, resistance vocabulary) are well-captured by regex patterns — dependency parsing would add complexity without corresponding accuracy gain for this specific detection task.

### 2026-03-16 — BGE-large over BGE-base
**Decision:** `BAAI/bge-large-en-v1.5` (1024d) for Ethos semantic layer.
**Rationale:** Ethos has no existing vectors — clean slate, no migration cost. AiMe remains on BGE-base; upgrading AiMe is a separate project.

### 2026-03-16 — Resistance computation is passage-scoped
**Decision:** `compute_resistance(text, significance, doc_type)` is called once per passage, not once per signal. The result guards the entire write block via `if resistance >= cfg.min_resistance_threshold:`.
**Rationale:** All three arguments (`text`, `significance`, `doc_type`) are fixed for a given passage — every signal extracted from the same passage has identical resistance. Calling it N times per passage is semantically redundant and computationally wasteful. The guard structure also makes the threshold logic explicit at the call site rather than buried inside an iterating loop.

### 2026-03-16 — MFT lexicon mappings must stay synchronized
**Decision:** `core/mft_classifier.py::MFT_VIRTUE_TO_ETHOS` and `core/lexicon_layer.py::_MFT_TO_ETHOS` must be kept in sync. Both map the same 5 MFT foundations to Ethos values. The classifier mapping may be slightly broader (it also maps `integrity` for fairness), but the core foundation→value assignments must agree.
**Rationale:** If the two independent layers produce different Ethos value sets for the same MFT foundation, they will never agree, defeating the purpose of convergent evidence. Divergence will appear as systematic disagreement between L1b and L3c signals on the same passage, corrupting the confidence boost logic.
