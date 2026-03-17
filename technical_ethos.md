# Ethos — Technical Reference

> **System class:** Universal Value Extraction Pipeline (UVEP)
> **Core claim:** Human values are not pre-installed rules. They are behavioral patterns — observable in text, extractable with deterministic computation, and verifiable through accumulation across figures, eras, and cultures.
> **Core invariant:** No LLM call anywhere in the extraction stack. All classification is deterministic Python.
> **Architecture:** Standalone pipeline. Ingest any figure → extract value signals → score resistance → export labeled training data.

---

> *"We didn't invent values. We extracted them from people who lived them — and people who didn't."*
>
> Ethos is not an annotation tool and not a sentiment classifier. It is a behavioral extraction engine — a pipeline that works from documented human behavior under real conditions, not from hypotheticals, preference rankings, or moral philosophy. It produces empirically-derived, resistance-weighted training data for value-aligned AI development.
>
> The central insight: a value stated in comfort is weak signal. A value demonstrated at real cost — under threat, under pressure, against interest — is strong signal. The resistance score is the measurement of that cost.
>
> The central gap this fills: most ethics datasets train on the poles. Ethos works the full spectrum — saints, monsters, and the complex majority in between. That middle ground is where human behavior actually lives, and where the most useful training signal lies.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Pipeline — End to End](#2-pipeline--end-to-end)
3. [Core Modules](#3-core-modules)
   - DocumentStore · ValueStore · Config · Resistance · ValueExtractor
4. [Value Vocabulary](#4-value-vocabulary)
5. [Resistance Scoring](#5-resistance-scoring)
6. [RIC Classification](#6-ric-classification)
   - P1 · P0 · APY · AMBIGUOUS
7. [Document Types & Authenticity Weighting](#7-document-types--authenticity-weighting)
8. [CLI Reference](#8-cli-reference)
9. [Key Algorithms](#9-key-algorithms)
   - Weight Formula · Consistency · Resistance Formula
10. [Design Principles](#10-design-principles)
11. [Module Index](#11-module-index)
12. [The Spectrum Principle](#12-the-spectrum-principle)
13. [Constitutional Invariants](#13-constitutional-invariants)
14. [Roadmap](#14-roadmap)

---

## 1. Architecture Overview

Ethos is a pure data pipeline. No user interface at runtime. No LLM calls. No external services required. It runs entirely on SQLite and the Python standard library.

Two databases:

| Database | Purpose |
|----------|---------|
| `data/documents.db` | Ingested passages — the source corpus |
| `data/values.db` | Extracted signals, value registry, figure metadata |

The pipeline is designed around a single invariant: **extraction is reproducible**. Given the same source text and the same thresholds, the output is identical every time. No randomness, no model variance.

### Component Map

```
cli/ingest.py          — user-facing entry point: segment + store + extract
    ↓
core/document_store.py — passage storage (documents.db)
    ↓
core/value_extractor.py — keyword scan + resistance scoring
    ├── core/resistance.py    — resistance formula (doc_type + significance + text markers)
    └── core/value_store.py   — observation + registry persistence (values.db)
    ↓
cli/export.py          — read value_observations → P1/P0/APY classification → JSONL
```

---

## 2. Pipeline — End to End

```
Source Text File (UTF-8)
    │
    ▼
cli/ingest.py
    ├── segment_text()           sentence-aware, ≤450 chars per passage, ≥30 chars min
    ├── document_store.insert_passage()   stores passages to documents.db
    ├── value_store.register_figure_source()   registers metadata to values.db
    ├── document_store.set_watermark(session_id, -1T)   reset to sentinel
    │
    ▼
core/value_extractor.process_figure(session_id)
    ├── document_store.get_watermark(session_id)
    ├── document_store.get_passages_since(session_id, watermark)
    │
    ├── for each passage:
    │       extract_value_signals()        keyword match against VALUE_VOCAB (15 values)
    │       resistance.compute_resistance()   doc_type + significance + text markers
    │       value_store.record_observation()  append to value_observations
    │       value_store.upsert_registry(session_id)   per-figure registry
    │       value_store.upsert_registry('')             cross-figure aggregate
    │
    └── document_store.set_watermark(session_id, latest_ts)
    │
    ▼
cli/export.py
    ├── _read_figure_observations()   JOIN value_observations + figure_sources
    ├── classify_observation()        P1 / P0 / APY / AMBIGUOUS per observation
    ├── build_training_records()      attach markers, weights, confidence
    │
    └── write JSONL:
            ric_figure_<name>.jsonl
            ric_historical_positive.jsonl
            ric_historical_negative.jsonl
            ric_historical_report.json
```

**Session ID namespace:** all historical figure sessions use `figure:<name>` (e.g. `figure:gandhi`, `figure:nixon`). This is the only architectural distinction between figures — the extraction code is identical for all.

---

## 3. Core Modules

### `core/config.py`

Simple `Config` dataclass with default thresholds. No file dependency at runtime.

```python
@dataclass
class Config:
    min_significance_threshold: float = 0.10
    min_resistance_threshold:   float = 0.20
    enabled:                    bool  = True
    apy_context_window_n:       int   = 5
```

`get_config()` returns a module-level singleton. Override at startup by modifying the singleton directly.

`core/config.py` also defines:
- `VALUE_TENSION_PAIRS` — list of 5 `(value_a, value_b)` tuples; default pairs derived from Schwartz (1992) / Rokeach (1973)
- `is_tension_pair(v1, v2) → bool` — symmetric lookup against `_TENSION_PAIR_SET`

---

### `core/document_store.py`

SQLite singleton at `data/documents.db`. Stores ingested passages and processing watermarks.

**Schema:**

```sql
CREATE TABLE passages (
    id           TEXT PRIMARY KEY,
    figure_name  TEXT NOT NULL,
    session_id   TEXT NOT NULL,
    text         TEXT NOT NULL,
    doc_type     TEXT NOT NULL,
    significance REAL NOT NULL DEFAULT 0.90,
    ts           REAL NOT NULL,
    ingested_at  REAL NOT NULL
);
CREATE TABLE watermarks (
    session_id          TEXT PRIMARY KEY,
    last_processed_ts   REAL NOT NULL
);
```

**Key methods:**

| Method | Purpose |
|--------|---------|
| `insert_passage(figure_name, session_id, text, doc_type, significance, ts)` | Append a passage |
| `get_passages_since(session_id, since_ts)` | Watermark-gated passage read for extractor |
| `get_watermark(session_id)` | Returns `-1T` sentinel if never set |
| `set_watermark(session_id, ts)` | Upsert watermark after processing |
| `list_figures()` | Grouped summary of all ingested figures |

**Watermark sentinel:** `-1_000_000_000_000.0` (predates all of human history). Set on ingestion so the extractor picks up all passages regardless of publication year, including texts with negative Unix timestamps (pre-1970 works).

---

### `core/value_store.py`

SQLite singleton at `data/values.db`. Stores value observations, registry, watermarks, and figure source metadata.

**Schema:**

| Table | Purpose |
|-------|---------|
| `value_observations` | Append-only. One row per matched value signal per passage. Includes `disambiguation_confidence` field. |
| `value_registry` | UPSERT. PK `(session_id, value_name)`. Running averages of demonstrations, significance, resistance, consistency, weight. `session_id=''` is the cross-figure aggregate. |
| `value_watermarks` | Per-session `last_processed_ts`. |
| `figure_sources` | Figure metadata: name, doc_type, passage_count, ingested_at. |
| `value_tension` | Value tension events: `(id, session_id, record_id, ts, value_held, value_failed, resistance, text_excerpt, created_at)`. Written by `_detect_tensions()` in `cli/export.py`. |
| `apy_context` | Rolling APY pressure buffer: `(id, session_id, record_id, ts, passage_idx, markers_json, window_n, created_at)`. Used by cross-passage APY detection. Pruned to N most recent per session. |

**Key methods:**

| Method | Purpose |
|--------|---------|
| `record_observation(...)` | Append a value signal observation |
| `upsert_registry(session_id, value_name, significance, resistance, ts)` | Update running weight + recompute consistency |
| `get_registry(session_id, min_demonstrations)` | Sorted-by-weight value profile |
| `get_universal_registry(min_demonstrations)` | Cross-figure aggregate |
| `register_figure_source(session_id, figure_name, document_type, passage_count)` | Figure metadata |
| `get_figures_list()` | All figures with registry stats joined |
| `get_stats()` | Summary: total observations, top values, figure count |
| `record_tension(session_id, record_id, ts, value_held, value_failed, resistance, text_excerpt)` | Write a tension event |
| `get_tensions(session_id)` | All tension events for a figure |
| `write_apy_context(session_id, record_id, ts, passage_idx, markers, window_n)` | Write pressure markers to rolling buffer |
| `get_apy_context(session_id, window_n)` | Read N most recent pressure entries for context lookup |
| `prune_apy_context(session_id, window_n)` | Delete entries beyond window |

---

### `core/resistance.py`

Pure function. No I/O. No state.

`compute_resistance(text, significance, doc_type) → float [0.0, 1.0]`

See [Section 5](#5-resistance-scoring) for full formula.

---

### `core/value_extractor.py`

The extraction loop.

**Public API:**

```python
process_figure(session_id: str) -> int
```

Returns number of observations recorded. Never raises.

**Internal:** `_run_extraction(session_id)` — watermark-gated loop over passages, calls `extract_value_signals()` + `compute_resistance()`, writes to `value_store`.

**Vocabulary access:**

```python
from core.value_extractor import VALUE_VOCAB
# Dict[str, List[str]] — 15 values → keyword lists
```

---

## 4. Value Vocabulary

15 named values, each with a keyword trigger list. Matching is case-insensitive substring. One match per value per passage (first matching keyword wins).

| Value | Example Keywords |
|-------|-----------------|
| `integrity` | "honest", "truth", "genuine", "transparent", "won't lie" |
| `courage` | "afraid", "brave", "risk", "hard to say", "facing my fear" |
| `compassion` | "care about", "worry about", "sad for", "heart goes out" |
| `commitment` | "promise", "committed", "dedicated", "won't give up" |
| `patience` | "patient", "waiting", "take time", "eventually" |
| `responsibility` | "my fault", "responsible for", "accountable", "on me" |
| `fairness` | "fair", "equal", "justice", "unfair", "not right" |
| `gratitude` | "grateful", "thankful", "appreciate", "means a lot" |
| `curiosity` | "wondering", "curious", "fascinated", "trying to understand" |
| `resilience` | "keep going", "bounce back", "despite", "won't quit" |
| `love` | "love", "care deeply", "cherish", "means the world" |
| `growth` | "better at", "improve", "learning", "working on myself" |
| `independence` | "on my own", "my choice", "self-reliant", "figure it out myself" |
| `loyalty` | "stand by", "loyal", "won't leave", "through thick and thin" |
| `humility` | "I was wrong", "my mistake", "I learned", "I need to admit" |

**Phase 1** adds semantic embedding (BGE-large + Qdrant) alongside keyword detection — passages where the value is present but no keyword fires are caught by the embedding layer.

### 4.1 VIA Character Strengths Mapping

The 15 Ethos values map to the VIA Classification of Character Strengths (Peterson & Seligman, 2004). This anchors Ethos values in a peer-reviewed psychological taxonomy and facilitates cross-study comparability.

| Ethos Value | VIA Strength | VIA Virtue Category |
|-------------|-------------|---------------------|
| `integrity` | Honesty (authenticity, genuineness) | Wisdom |
| `courage` | Bravery (valor, not shrinking from threat) | Courage |
| `compassion` | Kindness (generosity, nurturance, care) | Humanity |
| `commitment` | Perseverance (industry, diligence) | Courage |
| `patience` | Self-regulation (self-control, discipline) | Temperance |
| `responsibility` | Prudence (discretion, caution) | Temperance |
| `fairness` | Fairness (justice, equity) | Justice |
| `gratitude` | Gratitude (thankfulness, appreciation) | Transcendence |
| `curiosity` | Curiosity (interest, novelty-seeking) | Wisdom |
| `resilience` | Perseverance + Bravery (composite) | Courage |
| `love` | Love (capacity to give and receive love) | Humanity |
| `growth` | Love of Learning (mastering new skills) | Wisdom |
| `independence` | Self-regulation + Perspective (composite) | Wisdom / Temperance |
| `loyalty` | Teamwork (citizenship, social responsibility) | Justice |
| `humility` | Humility/Modesty | Temperance |

**Notes:**
- `resilience` is a composite of Perseverance and Bravery — VIA does not separate the two as cleanly as Ethos does.
- `independence` maps partially to Self-regulation (directing one's own conduct) and Perspective (wise counsel); neither is a perfect fit, reflecting a genuine gap in VIA's individual-level taxonomy.
- VIA focuses on virtues as trait-level strengths; Ethos measures demonstrated evidence under real conditions (cost-bearing), which is orthogonal to VIA's self-report framing.

**Reference:** Peterson, C., & Seligman, M. E. P. (2004). *Character Strengths and Virtues: A Handbook and Classification*. Oxford University Press / American Psychological Association.

---

## 5. Resistance Scoring

`compute_resistance(text, significance, doc_type)` estimates the cost of holding the demonstrated value in this passage. Range: `[0.0, 1.0]`.

**Formula (additive):**

| Signal | Contribution | Source |
|--------|-------------|--------|
| Base | +0.25 | Always |
| Significance bonus | +(significance × 0.40), capped 0.30 | Passage significance score |
| Document type bonus | see table below | Authenticity of source |
| Text markers | +0.20 | Adversity language detected in passage |

**Adversity markers** (`_RESISTANCE_RE`):
`even though`, `despite`, `but I still`, `but still`, `hard to`, `difficult to`, `I have to`, `I need to`, `I won't give up`, `I won't stop`, `I still believe`, `not easy`, `scared but`, `afraid but`, `at a cost`, `might lose`, `risk losing`

**Document type bonuses:**

| Type | Bonus | Rationale |
|------|-------|-----------|
| `action` | +0.40 | Documented real-world behavior — highest stakes |
| `journal` | +0.35 | Private writing — no audience pressure |
| `letter` | +0.30 | Directed correspondence — lower performance pressure |
| `speech` | +0.10 | Public address — highest performance pressure, lowest authenticity |
| `unknown` | +0.20 | Default |

**Rationale for the bonuses:** a value stated in a private journal cannot be performance — the audience is zero. A value stated in a public speech may be theatre. Resistance scores calibrate accordingly. The most valuable passages are `action` type: documented decisions under real stakes, not words but deeds.

---

## 6. RIC Classification

RIC = Relational Integrity Coefficient. Applied in `cli/export.py` during training data generation.

Each `value_observation` is classified into one of four labels:

| Label | Meaning | Signal |
|-------|---------|--------|
| **P1** | Value demonstrated under meaningful resistance — held under pressure | High resistance + hold markers, or APY pressure present but value maintained |
| **P0** | Value failed or corrupted — yielded, rationalized, compromised | Failure markers present, or low resistance with no hold markers |
| **APY** | Answer-Pressure Yield — external pressure detected + value failed | Pressure markers + failure markers together |
| **AMBIGUOUS** | Insufficient signal to classify | Middle resistance, no clear markers |

### Classification Logic (priority order)

```
1. APY pressure detected?
   → YES + failure markers  → APY (confidence 0.95)
   → YES + no failure       → P1  (confidence 0.95, APY-resistance)

2. Failure markers present?
   → P0 (confidence 0.85)

3. resistance >= p1_threshold AND hold markers?
   → P1 (confidence 0.90)

4. resistance >= p1_threshold alone?
   → P1 (confidence 0.75)

5. resistance < p0_threshold?
   → P0 (confidence 0.55)

6. Otherwise → AMBIGUOUS (confidence 0.40)
```

**Default thresholds:** `p1_threshold=0.55`, `p0_threshold=0.35`. Both are CLI-configurable.

### Marker Patterns

**Failure markers (→ P0):**
`gave in`, `gave up`, `yielded`, `backed down`, `compromised my`, `failed to`, `i lied`, `i deceived`, `i was weak`, `i rationalized`, `i caved`, `i relented`, `i pretended`, `i fabricated`

**Hold markers (→ P1):**
`despite`, `even though`, `nevertheless`, `stood firm`, `refused to give`, `persevered`, `maintained`, `held to`, `stayed true`, `did not yield`, `would not surrender`, `pushed through`, `carried on`

**APY pressure markers:**
`under pressure`, `when pressed`, `when threatened`, `they demanded`, `forced to`, `compelled to`, `to avoid punishment`, `to save myself`, `or face consequences`

---

## 7. Document Types & Authenticity Weighting

Document type is set at ingestion time and propagates through the entire pipeline — affecting resistance scores during extraction and training weights during export.

| Type | Resistance Bonus | Export Weight | When to Use |
|------|-----------------|--------------|-------------|
| `action` | +0.40 | 1.5× | Documented decisions, behaviors, acts — not words |
| `journal` | +0.35 | 1.4× | Diaries, private notebooks, unpublished letters to self |
| `letter` | +0.30 | 1.2× | Personal correspondence — directed, not performed |
| `speech` | +0.10 | 0.8× | Addresses, interviews, public statements |
| `unknown` | +0.20 | 1.0× | Default when source type is unclear |

**Guidance:** When a single source file mixes types (e.g. a biography that includes quoted letters AND reported actions), split into separate ingestion runs with the appropriate `--doc-type` per section. The pipeline supports multiple ingestion passes for the same figure — each appends to the same `session_id`.

---

## 8. CLI Reference

### `python -m cli.ingest`

```
python -m cli.ingest \
    --figure <name>          Figure identifier (e.g. gandhi, lincoln, nixon)
    --file <path>            Path to source text file (UTF-8)
    --doc-type <type>        journal | letter | speech | action | unknown
    [--pub-year <year>]      Publication year — for timestamp ordering
    [--significance <float>] Significance score for all passages (default 0.90)
    [--dry-run]              Preview segmentation only — no DB writes
    [--no-extract]           Skip immediate value extraction after ingestion
```

**What it does:**
1. Reads source file, segments into sentence-bounded passages (~450 chars)
2. Registers figure in `values.db`
3. Stores all passages in `documents.db` with timestamps spread 1 second apart
4. Resets watermark to `-1T` sentinel
5. Runs `process_figure()` immediately and prints the value profile

**Multiple ingestion passes** for the same figure are supported and additive:
```bash
python -m cli.ingest --figure aurelius --file meditations.txt   --doc-type journal --pub-year 180
python -m cli.ingest --figure aurelius --file letters_fronto.txt --doc-type letter  --pub-year 145
```

### `python -m cli.export`

```
python -m cli.export \
    [--figure <name>]              Export only this figure
    [--p1-threshold <float>]       Min resistance for P1 (default 0.55)
    [--p0-threshold <float>]       Max resistance for P0 (default 0.35)
    [--min-observations <int>]     Min observations per value per figure (default 1)
    [--output-dir <path>]          Output directory (default output/ric/)
    [--dry-run]                    Print stats only — no files written
    [--no-ambiguous]               Exclude AMBIGUOUS from per-figure files
    [--db <path>]                  Path to values.db (default data/values.db)
    [--value-tension]              Also write ric_value_tensions.jsonl (1.5× training weight)
    [--min-disambiguation <float>] Exclude observations below this disambiguation_confidence
    [--min-consistency <float>]    Exclude observations below this observation_consistency
```

**Output files:**
- `ric_historical_positive.jsonl` — all P1 examples across all figures
- `ric_historical_negative.jsonl` — all P0 + APY examples
- `ric_figure_<name>.jsonl` — per-figure, all labels (optionally excluding AMBIGUOUS)
- `ric_historical_report.json` — summary: counts by label / figure / value, training weights, balance ratio

---

## 9. Key Algorithms

### Weight Formula

```
weight = demonstrations × avg_significance × avg_resistance × consistency
```

Where:
- `demonstrations` — number of passages where this value was detected for this figure
- `avg_significance` — running mean of passage significance scores
- `avg_resistance` — running mean of resistance scores for this value in this figure
- `consistency` — 4-component score: volume + resistance stability + temporal spread + source diversity (see below)
  - Returns `0.5` when fewer than 2 observations (undefined with single observation)

**What weight captures:** *how often* × *how much each instance mattered* × *how costly it was* × *how stable the pattern is*.

A figure who demonstrates `integrity` once under extreme pressure scores differently from one who demonstrates it a hundred times in comfort. Weight encodes both quantity and quality.

### Consistency (Phase 8d — 4-component formula)

```
n         = observation count for (session_id, value_name)
σ_r       = stddev(resistance_scores)
span_s    = max(ts) − min(ts)          ← temporal span in seconds
doc_types = count(distinct doc_types for this (session_id, value_name))

consistency = min(1.0,
    0.30 × min(1.0, n / 10)             ← volume (saturates at 10 observations)
  + 0.30 × (1.0 − σ_r / 0.40)          ← resistance stability (low variance = high score)
  + 0.25 × min(1.0, span_s / 31536000) ← temporal spread (saturates at 1 year)
  + 0.15 × min(1.0, doc_types / 3)     ← source diversity (saturates at 3 doc types)
)
```

Returns `0.5` when fewer than 2 observations (undefined — single observation, no spread).

Consistency rewards figures who demonstrate a value *stably* across time, across source types, and across contexts — not just the single highest-resistance moment. Figures with high weight but low consistency are flagged as potential persona/behavior divergence candidates.

### Resistance Formula (summary)

```
resistance = clip(base + sig_bonus + doc_type_bonus + text_bonus, 0.0, 1.0)

base          = 0.25
sig_bonus     = min(significance × 0.40, 0.30)
doc_type_bonus = action:0.40 / journal:0.35 / letter:0.30 / speech:0.10 / unknown:0.20
text_bonus    = 0.20  (if adversity phrase pattern matches)
```

Maximum achievable: `0.25 + 0.30 + 0.40 + 0.20 = 1.15 → clipped to 1.0`
Minimum: `0.25` (base only, speech type, no markers, zero significance)

### Training Weight (export)

```
training_weight = doc_type_weight × significance
```

Where `doc_type_weight`: `action=1.5`, `journal=1.4`, `letter=1.2`, `speech=0.8`, `unknown=1.0`.

This makes action-sourced examples count 1.5× as much in downstream training — reflecting that documented deeds are harder evidence than documented words.

---

## 10. Design Principles

### No LLM in the extraction stack

All classification is keyword regex + SQLite arithmetic. This makes extraction:
- Deterministic — same input, same output every time
- Auditable — every classification decision is traceable to a specific rule
- Fast — runs in milliseconds, no API calls, no rate limits
- Independent — no model dependency, no API key required

### The spectrum must be complete

Training data built only from virtuous figures produces a model that recognizes virtue *performance*, not virtue. Training data built only from villains teaches the model to recognize monsters, not moral drift.

The most useful signal lives in the middle: figures with asymmetric value profiles, documented moments of failure alongside documented moments of extraordinary courage. That is where human moral complexity actually lives.

### Document type is a first-class signal

Not all text evidence is equal. A private diary entry about fear is a stronger integrity signal than a public speech about courage. The pipeline encodes this distinction explicitly and consistently — not as a post-hoc annotation but as a structural property of every ingestion.

### Watermarks prevent reprocessing

Every extraction pass advances the watermark. Re-ingesting the same figure after adding new source material only processes the new passages. Existing observations are never overwritten — the `value_observations` table is strictly append-only.

### Multiple passes are additive

A figure can be ingested from multiple source files, multiple document types, across multiple sessions. Each pass appends to the same `figure:<name>` session and the registry accumulates correctly. The per-figure profile emerges from the full corpus, not any single document.

---

## 11. Module Index

| File | Role | Key Exports |
|------|------|-------------|
| `core/config.py` | Config dataclass + tension pairs | `get_config()`, `Config`, `VALUE_TENSION_PAIRS`, `is_tension_pair(v1, v2)` |
| `core/document_store.py` | Passage storage | `get_document_store()`, `DocumentStore` |
| `core/value_store.py` | Value persistence | `get_value_store()`, `ValueStore` |
| `core/resistance.py` | Resistance scoring | `compute_resistance(text, significance, doc_type)` |
| `core/value_extractor.py` | Extraction loop | `process_figure(session_id)`, `extract_value_signals(...)`, `VALUE_VOCAB`, `_DISQUALIFIERS` |
| `cli/ingest.py` | Ingestion CLI | `ingest(...)`, `segment_text(...)` |
| `cli/export.py` | Export CLI | `export(...)`, `classify_observation(...)`, `build_training_records(...)`, `_compute_cooccurrence(...)`, `_detect_tensions(...)` |

---

## 12. The Spectrum Principle

The design insight that most ethics datasets miss — and the reason Ethos produces more useful training data:

**The poles are the wrong target.**

- **Far positive** (Gandhi, Lincoln, Mandela) — clean P1 examples, but the corpus is already filtered toward the heroic. Models trained only here learn to recognize virtue *performance*, not virtue. They learn to pattern-match on reputation, not action in context.

- **Far negative** (historical monsters) — clean P0 examples, but the failure is so total and so documented that models learn to recognize monsters, not moral drift. That is not where real harm happens.

**The middle ground is where the signal lives.**

JFK, MLK, Malcolm X, Churchill, Nixon, Oppenheimer — figures of genuine consequence with:

1. **Asymmetric value profiles** — high courage, low integrity; or extraordinary public commitment with documented private failures. Neither saint nor villain. Realistic.

2. **Domain-specific resistance** — MLK's `courage` under life-threatening pressure in civil rights work scores very differently from his personal domain. The profile is asymmetric within the same figure.

3. **Value evolution over time** — Malcolm X's transformation from radical to post-Mecca universalism is a dataset in itself. Values can be learned and revised under evidence. That is the most important training signal about human moral development.

4. **APY in abundance** — the moment of hesitation, the political compromise, the expedient choice. Pressure-yield dynamics are richest in complex middle-ground figures because they have both the pressure and the partial yield documented.

5. **Prevents shortcut learning** — a mixed-spectrum corpus forces the model to evaluate the *action in context*, not the person's historical reputation.

**The pipeline design reflects this:** no figure is pre-labeled at ingestion time. The resistance scores and marker patterns determine the classification. The same extraction code processes Lincoln and Nixon identically — the data decides.

---

## 13. Constitutional Invariants

| Invariant | Rule |
|-----------|------|
| No LLM call | Extraction is pure Python keyword/regex + SQLite arithmetic. No model dependency. |
| Deterministic output | Same input + same thresholds = identical output. No randomness. |
| Append-only observations | `value_observations` is never updated or deleted. Only appended. |
| Watermark continuity | Processing never repeats. Watermarks advance monotonically. |
| Fail-open | `process_figure()` never raises. All write failures log at WARNING; read failures at DEBUG — both with `exc_info=True`. |
| No pre-labeling | No figure is labeled positive or negative at ingestion. Classification emerges from the data. |
| Doc-type transparency | Document type is recorded at ingestion and flows through to every output field. Authenticity weighting is explicit, not implicit. |

---

## 14. Roadmap

### Phase 0 — Foundation ✅ (2026-03-04)
- [x] `core/document_store.py` — standalone passage storage
- [x] `core/value_store.py` — value observations + registry + figure_sources
- [x] `core/config.py` — threshold config
- [x] `core/resistance.py` — resistance scoring (doc_type formula)
- [x] `core/value_extractor.py` — keyword extraction loop, watermark-gated
- [x] `cli/ingest.py` — full ingestion pipeline with dry-run
- [x] `cli/export.py` — P1/P0/APY classification + JSONL export

### Quality Extensions (QX-A through QX-D) ✅ (2026-03-05)
- [x] **QX-A Keyword Disambiguation** — `_DISQUALIFIERS` per-value regex + `_FIRST_PERSON_RE` proximity check + `disambiguation_confidence` field
- [x] **QX-B Co-occurrence + Tension** — `VALUE_TENSION_PAIRS`, `value_tension` table, `ric_value_tensions.jsonl` export
- [x] **QX-C Cross-Passage APY** — `apy_context` table, `pressure_source_id` + `deferred_apy_lag` fields
- [x] **QX-D Consistency Scoring** — 4-component `_compute_consistency()`: volume + resistance stability + temporal spread + source diversity
- [x] Production hardening: structured logging, figure name guard, 50 MB file limit, thread-safe DB connections

### Phase 1 — Semantic Extraction ✅ (2026-03-16)
- [x] `core/embedder.py` — `BAAI/bge-large-en-v1.5` (1024d) singleton, lazy-load, fail-open
- [x] `core/value_seeds.py` — 322 seed sentences across 15 values (modern + archaic register)
- [x] `core/semantic_store.py` — Qdrant collection `ethos_value_prototypes`, `build_prototypes()` + `query_points()`
- [x] `cli/build_prototypes.py` — embed seeds and load all 15 prototypes
- [x] Hybrid scoring: agreement → `keyword+semantic` source tag + confidence boost; semantic-only → new signal with source `semantic`

### Phase 2 — Structural / Independent Classifiers ✅ (2026-03-16, hardened 2026-03-16)
- [x] `core/structural_layer.py` — pure regex adversity/agency/resistance/stakes patterns; `structural_score` in [0.0, 1.0]
- [x] Zero-shot DeBERTa (`MoritzLaurer/deberta-v3-large-zeroshot-v2.0`) — 15 per-value hypotheses, two-threshold detection
- [x] `core/lexicon_layer.py` (Layer 1b) — MFD2.0 (2,041 entries) + MoralStrength (452 entries); thread-safe double-checked locking
- [x] `core/mft_classifier.py` (Layer 3c) — `MMADS/MoralFoundationsClassifier` (RoBERTa, 10 MFT labels); tokenizer-level truncation
- [x] Phase 2 hardening: 7 defects fixed (duplicate regex, dead code, thread safety, truncation, resistance loop, schema consistency, multi_label contradiction)
- [x] 553 tests passing

### Phase 3 — Translation and Temporal Handling ✅ (2026-03-16)
- [x] `langdetect` language detection at ingest (seeded for determinism)
- [x] `source_authenticity` multiplier: original=1.0 / known translation=0.85 / uncertain=0.70
- [x] `preprocess_archaic()` — 26 deterministic replacement rules (thou/thee/thy/hath/etc.)
- [x] `pub_year` 6-tier temporal discount: pre-1400=0.70 through 1850+=1.0
- [x] `--translation` and `--pub-year` CLI flags; `source_lang`, `source_authenticity`, `pub_year` columns in `documents.db`

### Phase 4 — REST API ✅ (2026-03-16)
- [x] `core/pipeline.py` — `ingest_text()`, `figure_profile()`, `universal_profile()` (shared by CLI + API)
- [x] `api/models.py` — Pydantic request/response models
- [x] `api/app.py` — FastAPI, 5 endpoints + health
- [x] `api/server.py` — Uvicorn dev entrypoint
- [x] Endpoints: `POST /figures/{name}/ingest`, `GET /figures`, `GET /figures/universal`, `GET /figures/{name}/profile`, `POST /export/ric`, `GET /health`

### Phase 5 — Web Dashboard ⏳
- [ ] Figure browser — profile cards, value radar charts
- [ ] Corpus upload UI
- [ ] Universal registry visualization — cross-figure value heatmap
- [ ] Training set builder — filter by figure / value / doc_type / label

### Phase 6 — Corpus Scale ✅ (2026-03-16)
- [x] `core/corpus.py` — corpus-level stats: overview, figure summaries, value/resistance distributions, cross-figure table
- [x] `cli/batch_ingest.py` — manifest-driven batch ingestion, multi-file per figure, per-file error recovery
- [x] `cli/corpus_stats.py` — human-readable + JSON corpus report
- [x] `cli/dataset_card.py` — HuggingFace-compatible dataset card auto-generation
- [x] `core/value_store.py` multi-file fix: `passage_count` accumulates across ingests; mixed `doc_type` auto-detected

### Phase 7 — SRL Integration 💡
- [ ] Port `modules/srl/` from AiMe: claim_extractor, ric_gate, trait_compiler
- [ ] Behavioral trait profiles derived from extracted value registry
- [ ] AI integrity layer: applies RIC gate to model outputs during evaluation
