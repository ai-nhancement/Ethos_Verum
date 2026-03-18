# Ethos — Multi-Layer Extraction Research
**Date:** 2026-03-15 | **Status:** Ready for architecture planning

---

## Recommended Stack — 4-Layer Architecture

### Layer 1 — Lexical (current, keep)
- VALUE_VOCAB keyword matching (15 values)
- **Add:** MFD2.0 lexicon (~3,500 tokens, virtue/vice polarity per MFT foundation) — free, static file, zero runtime cost
- **Add:** MoralStrength lexicon (~1,000 words with continuous moral valence scores)
- Map MFT foundations to Ethos values via bridge table (care→compassion, fairness→justice, etc.)

### Layer 2 — Semantic Embedding (Qdrant)
- **Upgrade:** `BAAI/bge-large-en-v1.5` (1024d, MTEB 64.23) over current BGE-base (+1.04 retrieval gain)
- Build 15 value prototype vectors: 20-40 seed sentences per value, averaged + normalized
  - Seeds must include archaic vocabulary (probity, fortitude, valour, magnanimity, rectitude)
  - Include both modern and archaic-register seeds to bridge historical text
- Store prototypes in Qdrant; cosine similarity at query time = deterministic confidence score

### Layer 3 — Structural / Syntactic
Primary: **spaCy `en_core_web_trf`** (RoBERTa-base backbone)
- First-person agency detection: `nsubj` DEP + `{i, we, my, our}` + verb head
- Adversity clause detection: `advcl` DEP + adversity marker lexicon
- Negation patterns: `neg` DEP (captures "never yielded", "was not afraid")
- Direct object patterns: `dobj` chains for action-directed-at-patient signals

Supplement with **Stanza** for historical text morphology (handles archaic verb endings better than spaCy's web-trained models)

Secondary: **`vblagoje/bert-english-uncased-finetuned-srl`** (HuggingFace BERT-SRL)
- Semantic Role Labeling: ARG0 (agent) + V (action) + ARGM-ADV (adversative circumstance)
- Detects "agent performs value-consistent action under constraint" regardless of voice/clause order
- Run only on passages where Layer 1/2 fire a candidate signal (not on every passage — too slow)

Auxiliary: **`MMADS/MoralFoundationsClassifier`** (RoBERTa-base, 10 MFT labels)
- Trained on 60M+ sentences from political/formal text (letters, speeches, essays)
- F1 0.9957 overall; strong on fairness_virtue (0.9715), loyalty_virtue (0.9795)
- Weakness: loyalty_vice F1 only 0.1008 — don't use for vice detection on loyalty
- Output: 10 MFT foundation scores → bridge to Ethos value confidence

**`MoritzLaurer/deberta-v3-large-zeroshot-v2.0`** (zero-shot classifier)
- 36% better than bart-large-mnli on zero-shot tasks
- Apply custom value hypotheses per passage: "This passage demonstrates courage in the face of adversity."
- Fully deterministic (fixed inference, no sampling), MIT license
- Most flexible tool in the stack — catches nuanced signals that lexicons and prototypes miss

### Layer 4 — Contextual / Cross-Passage
No off-the-shelf tool exists. Design architecturally:

1. **Signal persistence tracking:** Value signal fires in ≥3 consecutive passage windows → confirmed arc (not noise)
2. **Signal escalation tracking:** Rising score across windows = stronger signal than single high-score passage
3. **Coherence weighting:** `coherence_factor = mean(cosine_sim(passage_i, passage_j))` across window — thematically coherent passages amplify each other
4. **Discourse connective detection:** spaCy detects "therefore", "consequently", "despite this", "nevertheless" — logical continuation markers strengthen adjacent signals
5. **REBEL relation extraction** (`Babelscape/rebel-large`) for entity-relation triplets across passages: builds local event graph — `Lincoln → refused → compromise` → `Lincoln → maintained → position`

---

## Signal Pooling

Each layer produces confidence `[0.0, 1.0]` per value per passage.
- Missing/errored layers: `-1.0` skip sentinel (NOT 0.0 — same pattern as AiMe `_blend3`)
- Final observation confidence = weighted blend across available layers
- Configurable weights per layer in config.py

---

## Translation Penalty

**Language detection:** `fasttext lid.176.ftz` (176 languages, returns confidence score, fastest, deterministic)
- At ingest time — routes non-English to multilingual pipeline
- Confidence score usable as continuous quality gate

**Translation fidelity scoring:** `sentence-transformers/LaBSE` (109 languages)
- If original + translation both available: `cosine_sim(LaBSE(original), LaBSE(translation))` = fidelity score
- Fidelity < 0.75 → flag as potentially low-quality translation
- Apply as `source_authenticity` multiplier on pooled confidence before resistance runs
  - Original language: 1.0
  - Known translation, high fidelity: 0.85
  - Uncertain/secondary translation: 0.70

**Non-English text (no translation):** `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli`
- Apply value hypotheses directly to source language — avoids introducing translation artifacts
- 100+ languages, XNLI avg 0.808 (English 0.883)

---

## Temporal Context

**The problem:** BGE-large trained on modern web text. Pre-1900 text has:
- Archaic vocabulary (fortitude, probity, magnanimity, dissimulation)
- Spelling variants (honour/honor, hath/has, thee/you)
- Inverted syntax, Latinized constructions

**Solutions:**
1. **Preprocessing normalization:** Deterministic rule set — `hath→has`, `doth→does`, `thee/thou→you`, `-est` endings. CLTK library (`pip install cltk`) has historical normalization utilities.
2. **Archaic seeds in prototypes:** Integrity seeds include "probity", "honour", "rectitude". Courage seeds include "fortitude", "valour", "intrepid". Prototypes averaged across modern + archaic register.
3. **Low similarity + high MFT confidence = archaic expression of value.** Don't penalize this — it's a detection path, not a failure mode.
4. **`pub_year` field (already in schema):** Apply mild embedding confidence discount for documents before ~1850; flag for archaic-register processing path.

**Note:** `bigscience-historical-texts/bert-base-blbooks-cased` exists (trained on 19th-century British Library books) but has no published embedding benchmarks. Test before committing.

---

## Production-Readiness Summary

| Component | Ready? |
|-----------|--------|
| `BAAI/bge-large-en-v1.5` | ✅ 6M+ downloads/month, MIT |
| `MMADS/MoralFoundationsClassifier` | ✅ (loyalty_vice weak) |
| `MoritzLaurer/deberta-v3-large-zeroshot-v2.0` | ✅ MIT, 36% over BART baseline |
| `sentence-transformers/LaBSE` | ✅ Standard multilingual tool |
| `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli` | ✅ MIT, 145K downloads/month |
| spaCy `en_core_web_trf` | ✅ |
| Stanza | ✅ |
| `vblagoje/bert-english-uncased-finetuned-srl` | ✅ Stable |
| `fasttext lid.176.ftz` | ✅ |
| MFD2.0 lexicon | ✅ Static file |
| MoralStrength lexicon | ✅ Static file |
| `vjosap/moralBERT-*` | ⚠️ Experimental, low validation |
| AllenNLP SRL (full library) | ⚠️ Maintenance mode, Python 3.8/3.9 only |
| Narrative arc detection tools | ❌ None exist — design architecturally |

---

## What This Solves for the Paper

The reviewer's core construct validity critique was: "you built the moral theory into the score and then reported that the score found what you built in."

Multi-layer extraction answers this directly:
- Layer 1 (lexical) = researcher vocabulary → expected to find what was designed
- Layer 2 (embedding similarity) = unsupervised prototype clustering → finds by meaning, not by word choice
- Layer 3 (MFT classifier + zero-shot) = independently trained models with no Ethos vocabulary → convergent validation
- Layer 4 (cross-passage coherence) = structural signal, not content-dependent

When all four layers agree, the resistance score is sitting on convergent evidence from independent detection methods. That's the answer to the construct validity problem.
