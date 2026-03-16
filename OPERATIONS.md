# Ethos — Operations Guide

> How to use the pipeline, what to feed it, and what it produces.

This document is the practical companion to `technical.md`. It covers the full workflow end to end with concrete examples, field-by-field output explanations, and guidance for getting quality results from different kinds of source material.

---

## Table of Contents

1. [Requirements](#1-requirements)
2. [Quick Start](#2-quick-start)
3. [Input: What to Feed the Pipeline](#3-input-what-to-feed-the-pipeline)
   - Text file format
   - What content works well
   - What to avoid
   - Document type selection guide
4. [Ingestion — `cli/ingest.py`](#4-ingestion--cliingestpy)
   - All flags
   - How segmentation works
   - What happens during a run
   - Multiple ingestion passes
   - Dry-run mode
5. [Understanding the Value Profile](#5-understanding-the-value-profile)
   - Registry fields explained
   - Reading the weight score
   - Consistency
6. [Export — `cli/export.py`](#6-export--cliexportpy)
   - All flags
   - Classification logic
   - Threshold tuning
7. [Output Files — Field Reference](#7-output-files--field-reference)
   - JSONL training record
   - Report JSON
8. [Workflow Patterns](#8-workflow-patterns)
   - Building a single figure corpus
   - Building a multi-figure corpus
   - Negative figures
   - Mixed-signal figures (the middle ground)
9. [Tips for Quality Results](#9-tips-for-quality-results)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. Requirements

- **Python 3.10+** — no virtual environment needed
- **Standard library only** — no pip installs required for Phase 0
- **Disk:** ~1MB per 1,000 passages (SQLite)
- **OS:** Windows, macOS, Linux

Run everything from the `C:\Ethos` root:

```bash
cd C:\Ethos
python -m cli.ingest --help
python -m cli.export --help
```

---

## 2. Quick Start

```bash
# 1. Ingest a figure
python -m cli.ingest \
    --figure gandhi \
    --file samples/gandhi.txt \
    --doc-type journal

# 2. Export training data
python -m cli.export

# That's it. Output is in output/ric/
```

The ingest step extracts value signals immediately and prints the value profile. The export step reads all ingested figures and writes labeled JSONL.

---

## 3. Input: What to Feed the Pipeline

### Text File Format

- **UTF-8 plain text** (`.txt`)
- Any line endings (LF, CRLF)
- No minimum or maximum file size
- Paragraphs, run-on text, and formatted transcripts all work — the segmenter is sentence-aware

The pipeline does **not** accept:
- PDF (convert to text first)
- HTML (strip tags first)
- Word documents (export as plain text)

### What Content Works Well

The pipeline detects values from **first-person language** and **behavioral description**. The best source material reads like the figure is speaking or acting, not being described abstractly.

**Excellent sources (highest signal):**

| Source type | `--doc-type` | Why it works |
|-------------|-------------|-------------|
| Diary / notebook entries | `journal` | First-person, private, no performance |
| Personal letters | `letter` | Directed, intimate, honest under pressure |
| Documented decisions and actions | `action` | Behavior under stakes, highest authenticity |
| Court testimony | `action` | Sworn statement, adversarial context |
| Memoir passages (candid sections) | `journal` | Retrospective first-person, often high candor |

**Good sources (useful signal):**

| Source type | `--doc-type` | Notes |
|-------------|-------------|-------|
| Speeches | `speech` | Gets lower resistance bonus — public performance |
| Interviews | `speech` | Candid moments within scripted context |
| Published essays | `letter` | Considered writing, directed at audience |
| Autobiography | `journal` or `letter` | Depends on the candor level of the specific work |

**Weaker sources (use with intent):**

| Source type | Notes |
|-------------|-------|
| Biographies (third-person narrative) | Pipeline reads first-person markers — third-person descriptions of behavior produce fewer keyword matches. Still useful for `action` segments. |
| Press releases, official statements | Highest performance pressure, most scripted. Use `speech`. |
| Fiction attributed to a figure | Do not use — pipeline treats text as genuine behavioral evidence. |

### What to Avoid

- **Third-person biography prose** for the entire source — "Gandhi believed in..." triggers fewer signals than Gandhi's own words. Better: extract direct quotes and diary passages from the biography and ingest those separately.
- **Wikipedia articles** — summary prose, no first-person voice
- **Interview transcripts with heavy editing** — use raw transcripts where available
- **Fiction** — the pipeline treats all text as behavioral evidence from the figure

### Document Type Selection Guide

This is the most important decision at ingestion time. It cannot be changed after ingestion without re-ingesting.

```
Is the text behavior (decision, act) documented by a third party?
    → action  (+0.40 resistance bonus, 1.5× training weight)

Is the text private writing with no intended audience?
    → journal (+0.35 resistance bonus, 1.4× training weight)

Is the text a letter to a specific person?
    → letter  (+0.30 resistance bonus, 1.2× training weight)

Is the text a public address or recorded statement?
    → speech  (+0.10 resistance bonus, 0.8× training weight)

Not sure?
    → unknown (+0.20 resistance bonus, 1.0× training weight)
```

**When to use `action` for text documents:** any document that records what someone *did* rather than what they *said* — court records, contemporaneous accounts of decisions, documented choices under pressure. The text doesn't have to be first-person; it can be a witness account. What matters is that it records behavior, not words.

---

## 4. Ingestion — `cli/ingest.py`

### All Flags

```
python -m cli.ingest [flags]

Required:
  --figure NAME      Figure identifier. 1–64 chars. Alphanumeric, underscore, hyphen only.
                     e.g. gandhi, jfk, malcolm_x, marcus-aurelius
                     Becomes the session_id: figure:<name>
                     Invalid names are rejected before any DB write.
  --file PATH        Path to source text file (UTF-8). Maximum 50 MB.
  --doc-type TYPE    journal | letter | speech | action | unknown

Optional:
  --pub-year YEAR    Integer. Sets the base timestamp for all passages.
                     e.g. --pub-year 1927 sets timestamps to Jan 1, 1927 00:00:00 UTC.
                     Without this, timestamps are set to 1 year ago from today.
                     Ordering is preserved: passages spread 1 second apart.

  --significance N   Float 0.0–1.0. Significance score for all passages.
                     Default: 0.90 (treat historical text as high-significance input)
                     Lower this (e.g. 0.5) for less authoritative sources.

  --dry-run          Preview segmentation only. No database writes. Prints the first
                     8 passages and total count. Segmentation info is emitted via
                     logging — configure logging to see figure/session/doc_type headers.

  --no-extract       Insert passages but skip immediate value extraction.
                     Extraction still runs on the next ingest or manual call.
```

### How Segmentation Works

The segmenter splits text into sentence-bounded passages of up to ~450 characters. It never cuts mid-sentence.

**Algorithm:**
1. Split the full text on sentence-ending punctuation (`. ! ?` followed by whitespace)
2. Accumulate sentences into a passage until adding the next would exceed 450 chars
3. When limit is reached, close the current passage and start a new one
4. Filter: discard any passage shorter than 30 characters

**Result:** Each passage is a coherent thought — usually 2–5 sentences — that can be read independently and scored for value signals. Passages are stored with timestamps 1 second apart to preserve their original order.

**Dry-run example:**
```bash
python -m cli.ingest --figure lincoln --file samples/lincoln.txt --doc-type speech --dry-run

  [00] Four score and seven years ago our fathers brought forth on this continent...
  [01] Now we are engaged in a great civil war, testing whether that nation, or any...
  [02] We are met on a great battle-field of that war. We have come to dedicate...
  ...and 44 more passages
```

Figure/session/doc_type summary lines are emitted via `logging.INFO` — visible when logging is configured (e.g. `python -m cli.ingest ... --dry-run 2>&1` or set `LOG_LEVEL=INFO`).

### What Happens During a Run

```
1. Read and segment source file
   → segment_text() splits into sentence-bounded passages ≤450 chars

2. Register figure metadata in values.db
   → figure_sources table: figure_name, document_type, passage_count, ingested_at

3. Reset watermark to -1T sentinel in documents.db
   → ensures the extractor processes ALL passages, including those from any era

4. Insert each passage into documents.db
   → passages table: id, figure_name, session_id, text, doc_type, significance, ts

5. Run value extraction immediately
   → value_extractor.process_figure('figure:<name>')
   → keyword scan against 15 values, compute resistance, write observations

6. Print value profile
   → value registry sorted by weight DESC
   → universal registry (all figures combined)
```

### Multiple Ingestion Passes

The same figure can be ingested from multiple source files. Each pass appends to the same session — the registry accumulates correctly.

```bash
# Marcus Aurelius from multiple sources
python -m cli.ingest --figure aurelius --file meditations.txt     --doc-type journal --pub-year 180
python -m cli.ingest --figure aurelius --file letters_fronto.txt  --doc-type letter  --pub-year 145
python -m cli.ingest --figure aurelius --file historia_augusta.txt --doc-type action  --pub-year 300
```

Each run resets the watermark to `-1T`, so re-ingesting the same file will re-process all passages. To avoid double-counting: only re-ingest a file if you've cleared the database or if you want to add it again intentionally.

### Dry-Run Mode

Use dry-run to check segmentation quality before writing anything:

```bash
python -m cli.ingest --figure jfk --file samples/jfk_inaugural.txt --doc-type speech --dry-run
```

Look for:
- Passages that are clearly mid-sentence (increase max_chars or fix punctuation in source)
- Passages that are too short (<50 chars) — these may not carry enough signal
- Passages that span unrelated topics — these produce mixed signals

---

## 5. Understanding the Value Profile

After ingestion, the pipeline prints the value profile. Here is how to read it.

### Profile Output

```
[extract] figure:test_figure — 8 values observed (12 observations recorded):
  courage               demos=2  weight=1.8000  resistance=1.000
  resilience            demos=2  weight=1.8000  resistance=1.000
  humility              demos=2  weight=1.8000  resistance=1.000
  loyalty               demos=2  weight=1.6236  resistance=0.950
  integrity             demos=1  weight=0.4500  resistance=1.000
  commitment            demos=1  weight=0.4500  resistance=1.000
  responsibility        demos=1  weight=0.4500  resistance=1.000
  curiosity             demos=1  weight=0.4500  resistance=1.000
```

### Registry Fields Explained

| Field | What it means |
|-------|--------------|
| `value_name` | One of 15 canonical value names |
| `demonstrations` | Number of passages where this value was detected |
| `avg_significance` | Mean significance score across those passages (default 0.90 unless overridden) |
| `avg_resistance` | Mean resistance score across those passages |
| `consistency` | How stable the resistance pattern is: `1 − (std_dev / mean)`. `1.0` = perfectly consistent; `0.5` = single sample (undefined); lower = erratic |
| `weight` | `demonstrations × avg_significance × avg_resistance × consistency` |
| `first_seen_ts` | Timestamp of the earliest matching passage |
| `last_seen_ts` | Timestamp of the most recent matching passage |

### Reading the Weight Score

Weight encodes **how often × how much × how costly × how stable** the value appeared. It is the primary ranking signal.

```
weight = 1.8   → 2 passages, high significance (0.9), full resistance (1.0), consistent
weight = 1.62  → 2 passages, high significance (0.9), high resistance (0.95), slightly inconsistent
weight = 0.45  → 1 passage, high significance (0.9), full resistance (1.0), consistency undefined (0.5)
```

**What low weight means:** A value with weight 0.45 is not weak — it was observed once at maximum resistance. With more source material, that weight will compound. A value with weight 0.45 from 5 speeches may matter less than one with weight 0.45 from a private journal under duress.

**What consistency means:** A figure who demonstrates `integrity` at resistance 0.8 in every passage has consistency ~1.0. A figure who demonstrates `integrity` at 0.3 in some passages and 1.0 in others has lower consistency — the value is erratic, demonstrated only under extreme pressure but not as a general pattern. Both are informative.

---

## 6. Export — `cli/export.py`

### All Flags

```
python -m cli.export [flags]

Optional:
  --figure NAME        Export only this one figure. Default: all ingested figures.
  --p1-threshold N     Float 0.0–1.0. Min resistance to classify as P1 (held).
                       Default: 0.55
  --p0-threshold N     Float 0.0–1.0. Max resistance for P0 when no hold markers.
                       Default: 0.35
  --min-observations N Int. Min observations for a (figure, value) pair to be included.
                       Default: 1 (include everything)
  --output-dir PATH    Output directory. Default: output/ric/
  --dry-run            Print classification stats only. No files written.
  --no-ambiguous       Exclude AMBIGUOUS observations from per-figure files.
                       Still included in the aggregate files by default.
  --db PATH            Path to values.db. Default: data/values.db

  --value-tension      Also write ric_value_tensions.jsonl — tension events where
                       one value was held and a paired value failed in the same
                       passage. These records carry 1.5× training weight.

  --min-disambiguation Float 0.0–1.0. Exclude observations below this disambiguation
                       confidence score. Useful for filtering likely false-positive
                       keyword matches without re-ingesting. Default: 0.0 (include all).

  --min-consistency    Float 0.0–1.0. Exclude observations whose value-registry entry
                       has consistency below this threshold. Useful for high-quality
                       filtered exports. Default: 0.0 (include all).
```

### Classification Logic

Each value observation is classified into one of four labels:

```
P1  — Value held under meaningful resistance
P0  — Value failed or corrupted
APY — Answer-Pressure Yield (pressure present; value failed)
AMBIGUOUS — Insufficient signal to classify
```

The classifier checks conditions in priority order:

```
1. APY pressure markers detected?
   ├── YES + failure markers  → APY  (confidence 0.95)
   └── YES + no failure       → P1   (confidence 0.95, highest confidence P1)

2. Failure markers present?
   └── P0 (confidence 0.85)

3. resistance >= 0.55 AND hold markers?
   └── P1 (confidence 0.90)

4. resistance >= 0.55 alone?
   └── P1 (confidence 0.75)

5. resistance < 0.35?
   └── P0 (confidence 0.55)

6. Otherwise → AMBIGUOUS (confidence 0.40)
```

### Threshold Tuning

**`--p1-threshold`** (default 0.55):
- **Raise** (e.g. 0.70): stricter P1 — only count value demonstrations under very high cost. Fewer P1 records, higher quality signal.
- **Lower** (e.g. 0.40): more P1 records — includes demonstrations under moderate resistance. More quantity, some dilution.

**`--p0-threshold`** (default 0.35):
- **Raise** (e.g. 0.50): more P0 records — captures failures that occurred under moderate resistance (someone who failed despite having some resolve).
- **Lower** (e.g. 0.20): stricter P0 — only captures clear, low-resistance failures.

**When to adjust thresholds:**
- For a corpus of private journals (all `journal` doc_type): resistance scores will be high. P1 threshold can be raised.
- For a corpus of speeches (all `speech` doc_type): resistance scores will be lower. Lower the P1 threshold or expect more AMBIGUOUS.
- For figures with documented coercion: APY will handle most cases; thresholds matter less.

**`--min-observations`**:
- Useful for filtering noise. With `--min-observations 3`, a value must appear in at least 3 passages before it's included. Good for large corpora where single-passage matches may be accidental keyword hits.

---

## 7. Output Files — Field Reference

### JSONL Training Records

Each line in the `.jsonl` files is a JSON object. Here is a complete positive (P1) example:

```json
{
  "id": "f938b123-d019-48d8-aef1-a1eebbee139e",
  "source_obs_id": "05aabcc6-86d5-4e2b-91c6-8c4234980f2a",
  "figure": "test_figure",
  "session_id": "figure:test_figure",
  "record_id": "94ee119e-1c24-47f7-ace7-a6f2df32b5e5",
  "ts": 1741125113.93,
  "value_name": "integrity",
  "text_excerpt": "...was immense, but I still believed that truth mattered more than safety. Despite the threats and the dema...",
  "document_type": "journal",
  "significance": 0.9,
  "resistance": 1.0,
  "label": "P1",
  "label_reason": "high_resistance_hold_marker",
  "fail_mode": "",
  "training_weight": 1.26,
  "confidence": 0.9,
  "pressure_markers": [],
  "failure_markers": [],
  "hold_markers": ["still", "despite"]
}
```

**Field definitions:**

| Field | Type | Description |
|-------|------|-------------|
| `id` | UUID | Unique ID for this training record |
| `source_obs_id` | UUID | ID of the original `value_observations` row this was derived from |
| `figure` | string | Figure name (e.g. `"gandhi"`) |
| `session_id` | string | Always `"figure:<name>"` |
| `record_id` | UUID | ID of the source passage in `documents.db` |
| `ts` | float | Unix timestamp of the source passage (derived from `--pub-year` if set) |
| `value_name` | string | One of 15 canonical value names |
| `text_excerpt` | string | Up to 200 chars of the passage around the matched keyword. May have `...` at start/end. |
| `document_type` | string | `journal` / `letter` / `speech` / `action` / `unknown` |
| `significance` | float | Passage significance score (0.0–1.0, set at ingestion) |
| `resistance` | float | Computed resistance score (0.0–1.0). See [Section 5 of technical.md] |
| `label` | string | `"P1"` / `"P0"` / `"APY"` / `"AMBIGUOUS"` |
| `label_reason` | string | Machine-readable reason for the label (see below) |
| `fail_mode` | string | `""` for P1, `"APY"` for APY, `"UA"` or `"SO"` for P0 |
| `training_weight` | float | `doc_type_weight × significance`. Use for weighted loss in training. |
| `confidence` | float | Classifier confidence in the label (0.40–0.95) |
| `pressure_markers` | list[str] | APY pressure phrases found in `text_excerpt` |
| `failure_markers` | list[str] | Failure phrases found in `text_excerpt` |
| `hold_markers` | list[str] | Hold phrases found in `text_excerpt` |
| `disambiguation_confidence` | float | 0.0–1.0. Confidence that keyword match is a genuine value demonstration (not incidental/idiomatic use). 1.0 = all three disambiguation checks passed. 0.0 = grammatical role disqualifier fired. |
| `observation_consistency` | float | 0.0–1.0. 4-component consistency score from the value registry: volume + resistance stability + temporal spread + source diversity. |
| `pressure_source_id` | string | UUID of the source passage containing the APY pressure markers. Present on cross-passage APY records only (when `label="APY"` and pressure came from a different passage). Empty string otherwise. |
| `deferred_apy_lag_s` | float | Seconds between the pressure passage and this failure passage. Present on cross-passage APY records only. |
| `deferred_apy_lag_n` | int | Passage count between the pressure passage and this failure passage. Present on cross-passage APY records only. |

**`label_reason` values:**

| Reason | Label | Meaning |
|--------|-------|---------|
| `apy_resistance_held_under_pressure` | P1 | Pressure markers present but value maintained |
| `high_resistance_hold_marker` | P1 | resistance ≥ 0.55 + hold phrase detected |
| `high_resistance_held` | P1 | resistance ≥ 0.55, no hold marker |
| `pressure_detected_value_failed` | APY | Pressure markers + failure markers both present |
| `failure_markers_present` | P0 | Explicit failure language in excerpt |
| `low_resistance_no_hold` | P0 | resistance < 0.35, no hold markers |
| `insufficient_signal` | AMBIGUOUS | Middle resistance, no clear markers |

**`fail_mode` values (P0 only):**

| Value | Meaning |
|-------|---------|
| `""` | Not a failure (P1 or APY) |
| `"APY"` | Answer-Pressure Yield failure mode |
| `"SO"` | Scope Overreach — explicit failure markers present |
| `"UA"` | Unanchored — no failure markers, just low resistance |

**`training_weight`:**

The weight to apply during model training. Higher weight = more authoritative evidence.

| doc_type | base weight | × significance 0.9 |
|----------|-------------|-------------------|
| `action` | 1.5 | **1.35** |
| `journal` | 1.4 | **1.26** |
| `letter` | 1.2 | **1.08** |
| `speech` | 0.8 | **0.72** |
| `unknown` | 1.0 | **0.90** |

---

Here is a negative (P0) example:

```json
{
  "id": "5b725935-b21f-4187-85ef-2588b2301df4",
  "source_obs_id": "bfc69433-85b0-431b-981f-5c8ea80fc293",
  "figure": "test_figure",
  "session_id": "figure:test_figure",
  "record_id": "94ee119e-1c24-47f7-ace7-a6f2df32b5e5",
  "ts": 1741125113.93,
  "value_name": "humility",
  "text_excerpt": "...not surrender what I knew to be right. I was wrong to have hesitated earlier.",
  "document_type": "journal",
  "significance": 0.9,
  "resistance": 1.0,
  "label": "P0",
  "label_reason": "failure_markers_present",
  "fail_mode": "SO",
  "training_weight": 1.26,
  "confidence": 0.85,
  "pressure_markers": [],
  "failure_markers": ["i was wrong to"],
  "hold_markers": []
}
```

Note: `resistance` is 1.0 here because the passage as a whole is high-resistance (private journal under great pressure). The P0 label is driven by the **failure marker** (`"i was wrong to"`) which indicates the value of `humility` was *acknowledged as having been absent*, not demonstrated. This is correct — it is an admission of a past failure, not a current demonstration.

---

### Report JSON

`ric_historical_report.json` is written after every export run.

```json
{
  "generated_at": "2026-03-04T21:51:56Z",
  "db_path": "C:\\Ethos\\data\\values.db",
  "p1_threshold": 0.55,
  "p0_threshold": 0.35,
  "min_observations": 1,
  "total_observations": 12,
  "total_classified": 12,
  "by_label": {
    "P1": 8,
    "P0": 4,
    "APY": 0,
    "AMBIGUOUS": 0
  },
  "total_weight_positive": 10.08,
  "total_weight_negative": 5.04,
  "by_figure": {
    "gandhi": { "P1": 45, "P0": 3, "APY": 2 },
    "nixon":  { "P1": 12, "P0": 28, "APY": 7 }
  },
  "by_value": {
    "integrity": { "P1": 18, "P0": 9, "APY": 3 },
    "courage":   { "P1": 14, "P0": 6, "APY": 1 }
  },
  "output_files": {
    "positive": "C:\\Ethos\\output\\ric\\ric_historical_positive.jsonl",
    "negative": "C:\\Ethos\\output\\ric\\ric_historical_negative.jsonl",
    "per_figure": {
      "gandhi": "C:\\Ethos\\output\\ric\\ric_figure_gandhi.jsonl",
      "nixon":  "C:\\Ethos\\output\\ric\\ric_figure_nixon.jsonl"
    }
  }
}
```

**Key report fields:**

| Field | What to check |
|-------|--------------|
| `by_label` | Overall P1/P0/APY/AMBIGUOUS balance across all figures |
| `total_weight_positive / negative` | Effective training weight balance (not just count) |
| `by_figure` | Which figures are contributing which labels |
| `by_value` | Which values have the most labeled examples |

**Reading the balance:**

```
P1=8  P0=4  ratio=2.00:1
```

A 2:1 positive/negative ratio is common when most source material is from a positively-regarded figure. For a balanced training dataset, target closer to 1:1 by adding more figures with documented failures — or mixing in speeches (lower resistance, more AMBIGUOUS/P0) alongside journals (higher resistance, more P1).

---

## 8. Workflow Patterns

### Building a Single Figure Corpus

```bash
# Best practice: ingest from multiple source types for the same figure

# Private writings first (highest authenticity)
python -m cli.ingest \
    --figure mlk \
    --file sources/mlk_letters_birmingham.txt \
    --doc-type letter \
    --pub-year 1963

# Documented actions (biographer's account of specific decisions)
python -m cli.ingest \
    --figure mlk \
    --file sources/mlk_montgomery_decisions.txt \
    --doc-type action \
    --pub-year 1956

# Speeches (lower resistance bonus — public performance pressure)
python -m cli.ingest \
    --figure mlk \
    --file sources/mlk_i_have_a_dream.txt \
    --doc-type speech \
    --pub-year 1963

# Export just this figure
python -m cli.export --figure mlk
```

### Building a Multi-Figure Corpus

```bash
# Positive figures
python -m cli.ingest --figure gandhi   --file sources/gandhi_autobiography.txt  --doc-type journal --pub-year 1927
python -m cli.ingest --figure aurelius --file sources/meditations.txt            --doc-type journal --pub-year 180
python -m cli.ingest --figure tubman   --file sources/tubman_interviews.txt      --doc-type letter  --pub-year 1886

# Complex middle-ground figures
python -m cli.ingest --figure jfk     --file sources/jfk_personal_letters.txt  --doc-type letter  --pub-year 1960
python -m cli.ingest --figure jfk     --file sources/jfk_cuban_crisis.txt      --doc-type action  --pub-year 1962
python -m cli.ingest --figure malcolm --file sources/malcolm_autobiography.txt  --doc-type journal --pub-year 1964
python -m cli.ingest --figure nixon   --file sources/nixon_transcripts.txt      --doc-type speech  --pub-year 1973

# Negative figures
python -m cli.ingest --figure macbeth_historical --file sources/inverness_accounts.txt --doc-type action --pub-year 1040

# Export everything — all figures, all labels
python -m cli.export

# Check the balance
python -m cli.export --dry-run
```

### Negative Figures

Negative figures produce P0 and APY labels that are essential for balanced training data. The pipeline processes them identically to positive figures — no special handling required.

**What to source for negative figures:**
- Trial transcripts (direct testimony — use `action`)
- Private correspondence showing rationalization (use `letter`)
- Known decision records showing the moment of failure (use `action`)
- Public statements that contradict private behavior (use `speech` — the contradiction creates APY signal)

**What the pipeline produces from negative figures:**
- P0 observations where failure markers are explicit (`"gave in"`, `"i rationalized"`, `"i caved"`)
- APY observations where pressure + failure coincide
- Occasional P1 observations — almost all complex historical figures demonstrate *some* values in some contexts. Nixon showed `courage` in specific documented moments. That is not a problem; it is accurate.

**Key insight:** a figure does not need to be entirely negative to contribute negative training data. A figure who is 70% P1 and 30% P0 across their documented behavior contributes both — and the mixed profile is more realistic and more useful than a pure negative would be.

### Mixed-Signal Figures (The Middle Ground)

These are the highest-value figures for the corpus. Their profiles are asymmetric — strong in some values, weak or failed in others.

```bash
# JFK example: multiple source types, multiple domains
# The contrast between his documented private courage (PT-109 letters)
# and his political compromises (speech transcripts) will produce
# a genuinely mixed profile — which is the point

python -m cli.ingest --figure jfk --file sources/jfk_pt109_letters.txt     --doc-type letter --pub-year 1943
python -m cli.ingest --figure jfk --file sources/jfk_cuban_missiles.txt    --doc-type action --pub-year 1962
python -m cli.ingest --figure jfk --file sources/jfk_press_conference.txt  --doc-type speech --pub-year 1961
python -m cli.ingest --figure jfk --file sources/jfk_private_notes.txt     --doc-type journal --pub-year 1962

python -m cli.export --figure jfk
```

Expected result: high P1 on `courage` (PT-109 letters, crisis decisions), mixed on `integrity` (political speeches vs private notes), variable on `commitment` depending on which domain.

The asymmetric profile is not noise — it is the most honest measurement of a complex human being.

---

## 9. Tips for Quality Results

**1. Match doc_type to the actual source, not the figure's reputation.**
A Gandhi speech gets `speech` — not `journal` because Gandhi was virtuous. The pipeline scores authenticity of evidence, not reputation of the person.

**2. Pre-process source files before ingestion.**
- Remove footnotes, page numbers, editor annotations
- Remove section headers if they interrupt the narrative flow
- Keep quotes and direct speech — they carry the strongest signals

**3. Use `--dry-run` to check segmentation before committing.**
If passages are too short (under 50 chars), the source may have a lot of sentence fragments. If passages are too long, the source has few sentence-ending punctuation marks (common in transcripts). Manually add periods to the source file if needed.

**4. For translated texts, check keyword coverage.**
The value vocabulary is built around English keywords. Translated texts (e.g. Meditations, translated from Greek; Confessions, from Latin) work reasonably well because good translators preserve the emotional language. But some values — particularly `compassion` and `fairness` — may be under-represented in formal translations. Supplement with modern translation versions where available.

**5. Lower `--significance` for less authoritative sources.**
The default 0.90 treats every passage as high-signal. For a source you're less confident about (heavily edited memoir, attributed but unverified letters), use `--significance 0.60` or `0.70`. This reduces the observations' weight in the registry without excluding them.

**6. Use `--min-observations 3` for large corpora.**
When processing hundreds of passages, single-keyword hits can produce accidental value matches. `--min-observations 3` requires a value to appear in at least 3 separate passages before it's included in the export — a reasonable bar for claiming the value was a genuine behavioral pattern rather than a one-off word choice.

**7. Build toward 1:1 balance across the corpus.**
The `ric_historical_report.json` shows your P1:P0 ratio. Aim for 1:1 to 2:1. If you're at 5:1, add more figures with documented failures. If you're at 1:5, add more positive-sourced material or lower the P1 threshold.

**8. Use `--no-ambiguous` for clean training sets.**
AMBIGUOUS observations (middle resistance, no clear markers) are informative for corpus analysis but noisy for training. Use `--no-ambiguous` when generating final training exports.

---

## 10. Troubleshooting

**"ingest returns 0 / no output"** (check logging for ERROR lines)
- Figure name contains invalid characters. Only `a-z A-Z 0-9 _ -` are allowed, 1–64 chars. Names with spaces, slashes, or dots are rejected before any DB write. Use `malcolm_x` not `Malcolm X`.
- File not found: confirm the path is correct and accessible.
- File exceeds 50 MB: split the source file into multiple smaller files and ingest each separately.

**"No passages extracted from file"**
- Check the file is valid UTF-8: `python -c "open('file.txt', encoding='utf-8').read()"`
- Check the file has sentence-ending punctuation. Transcripts without periods won't segment well.
- Check the file isn't empty.

**"0 values observed after extraction"**
- Run `--dry-run` and read the passages. Do they contain first-person language?
- Check `--significance` — if set very high (e.g. 1.0), passages at the default 0.90 will be filtered. The significance threshold is `min_significance_threshold = 0.10` by default, so this is rarely the issue.
- The text may not contain keywords from the vocabulary. Try searching a sample passage against the `VALUE_VOCAB` in `core/value_extractor.py`.

**"0 positive examples in export"**
- Check your `--p1-threshold`. If set above the resistance values in your corpus, nothing qualifies.
- Run `--dry-run` to see the actual resistance distribution before setting thresholds.
- For a `speech`-only corpus, resistance scores will be lower (max from speech is 0.25 + 0.30 + 0.10 + 0.20 = 0.85). Lower p1-threshold to 0.45 for speech-dominant corpora.

**"All observations are AMBIGUOUS"**
- Resistance scores are falling in the 0.35–0.55 gap. This means: journal/letter doc_type without adversity markers in the text, mid-significance passages.
- Either: add source material with more explicit adversity language, or adjust thresholds.

**"Same passage produces the same observation twice"**
- The figure was ingested twice from the same file. The pipeline does not deduplicate on text content — it relies on the watermark system. To reset: delete `data/documents.db` and `data/values.db` and re-ingest.

**Figures list shows wrong passage_count**
- `passage_count` in `figure_sources` is set at ingestion time. If you did multiple passes, only the last run's count is stored (it's an UPSERT). The actual passage count in `documents.db` may be higher. Use `document_store.count_passages(session_id)` to get the true count.

**Export produces no files even though ingest showed observations**
- Check `--db` path is pointing to the right `values.db`. By default it resolves to `data/values.db` relative to the project root.
- Confirm the figure session_id matches: `figure:gandhi` not `figure:Gandhi` (lowercase).
