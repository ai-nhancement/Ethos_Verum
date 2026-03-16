# Ethos: A Resistance-Weighted Behavioral Corpus for Value-Aligned AI Training

**Abstract**

The alignment of artificial intelligence systems with human values remains one of the central unsolved problems in AI safety research. Existing approaches draw on one of two sources: hypothetical moral scenarios designed by researchers, or human preference rankings collected at scale. Both inherit a fundamental limitation — they capture declared values, not demonstrated ones. We introduce **Ethos**, a behavioral extraction pipeline that derives value signal from documented historical behavior: journals, letters, speeches, and recorded actions of historical figures across the full human spectrum, from the canonically virtuous to the historically destructive. Ethos introduces two novel contributions: (1) a **resistance score** that quantifies the cost of holding a demonstrated value in context — a signal absent from all existing value alignment datasets — and (2) a **document authenticity weighting** scheme that calibrates evidence quality based on the performance pressure present at the time of writing. The pipeline produces labeled training data in three classes: P1 (value held under meaningful resistance), P0 (value failed or corrupted), and APY (Answer-Pressure Yield — value abandoned under identified external pressure). We argue that the most important signal in a value-alignment corpus lies not at the poles — celebrated saints or documented villains — but in the complex middle ground where asymmetric value profiles, domain-specific courage, and documented moments of both extraordinary integrity and ordinary failure reside. Ethos is the first pipeline designed to extract that signal systematically.

---

## 1. Introduction

When we ask whether an AI system is "aligned with human values," we are implicitly assuming we know what human values look like in practice. The evidence suggests we are less certain than we believe.

Most datasets used to train value-aligned models fall into one of two categories. The first category consists of **hypothetical moral scenarios**: carefully constructed vignettes asking annotators to judge the morality of an action, assign blame, or rank outcomes (Hendrycks et al., 2021; Emmons & McCullough, 2003). These datasets are rigorous by design, but they measure declared moral preference — what annotators believe they would do, or what they believe is right, in a situation they have never faced. Decades of social psychology research documents the gap between moral intention and moral behavior under pressure (Milgram, 1963; Haidt, 2001). A dataset of intentions is not a dataset of behavior.

The second category consists of **human preference rankings**: pairs of model outputs labeled by annotators as better or worse along axes like helpfulness and harmlessness (Ouyang et al., 2022). These datasets are large-scale and have produced demonstrably improved models. But preference rankings are scalar and momentary. They capture which of two outputs an annotator preferred at one point in time — not the underlying value that drove the preference, and certainly not whether the annotator would have maintained that preference under adversarial pressure, personal cost, or sustained challenge.

The result is that current value-alignment training data, whatever its scale and quality, addresses the question: *what do people say is right?* The question it does not address — the harder and more important question — is: *what do people actually do when holding a value costs them something?*

This paper introduces **Ethos**, a pipeline designed to address the second question.

Ethos extracts value signal from historical text — the documented words and actions of real people in real situations with real stakes. It scores each observation not only on which value was demonstrated, but on how much it cost to hold that value: the **resistance score**. It calibrates the authenticity of evidence based on the performance conditions under which it was produced: the **document type** signal. And it classifies extracted observations into three labels — P1 (held), P0 (failed), APY (yielded under pressure) — that together capture the full dynamic of values under pressure, not just their presence.

Two research gaps motivate this work directly. First, no existing NLP dataset extracts human values from historical text. The historical record contains thousands of documented instances of humans demonstrating, failing, and struggling with specific values under conditions ranging from mortal threat to political coercion to intimate betrayal — and none of this behavioral evidence is currently represented in alignment training data. Second, no existing alignment dataset quantifies what we call resistance: the cost of holding a value. Resistance is arguably the most important signal in value data, because a value that costs nothing to hold is weak signal — it may be preference, habit, or social performance rather than genuine commitment.

Our contributions are:

1. A **behavioral extraction pipeline** (Ethos) that generates value signal from historical text using deterministic keyword and marker analysis — no LLM required.
2. A **resistance scoring framework** that estimates the cost of demonstrating a value in context, incorporating document authenticity, significance, and adversity language markers.
3. A **three-class labeling scheme** (P1/P0/APY) that captures held values, failed values, and pressure-yield dynamics.
4. An argument for **the spectrum principle**: that the most valuable training signal for value alignment lies not at the behavioral poles (ideal virtue vs. obvious villainy) but in the complex middle ground where asymmetric value profiles, domain-specific courage, and documented moral failure coexist in the same person.

---

## 2. Background and Related Work

### 2.1 Value Theory in Psychology

Human value research has produced several taxonomies relevant to AI alignment work.

**Schwartz's Theory of Basic Human Values** (Schwartz, 1992) proposes ten universal values organized around two bipolar dimensions: Openness to Change vs. Conservation, and Self-Enhancement vs. Self-Transcendence. The theory has been empirically validated across more than 80 countries using the Schwartz Values Survey. Crucially for our purposes, Schwartz operationalizes values as motivational goals that serve as guiding principles — abstract ideals that express themselves in behavioral patterns over time, not momentary preferences.

**Peterson and Seligman's Character Strengths and Virtues** (2004) taxonomizes 24 character strengths organized under six virtues: wisdom, courage, humanity, justice, temperance, and transcendence. Their framework is explicitly behavioral — strengths are defined as positive traits reflected in thoughts, feelings, and behaviors — and was designed to be cross-culturally and historically grounded. The VIA-IS (Values in Action Inventory of Strengths) measures these strengths through self-report, but the theoretical framework explicitly positions character strengths as observable behavioral tendencies.

**Haidt's Moral Foundations Theory** (Haidt & Joseph, 2004; Haidt, 2012) proposes that human moral reasoning is organized around five to six innate foundations: Care/Harm, Fairness/Reciprocity, Loyalty/Betrayal, Authority/Subversion, Sanctity/Degradation, and Liberty/Oppression. Moral Foundations Theory has been applied in political psychology and NLP analysis of moral language, notably in the Moral Foundations Twitter Corpus (Hoover et al., 2020).

These frameworks share a common assumption that Ethos inherits: values are observable in behavior, not only in self-report. Where they rely on survey instruments and annotated datasets, Ethos relies on the historical record — which offers something surveys cannot: behavioral evidence under conditions of genuine cost.

### 2.2 Value Alignment Datasets

**The ETHICS Dataset** (Hendrycks et al., 2021) spans five moral domains — commonsense morality, deontology, justice, utilitarianism, and virtue ethics — with approximately 130,000 examples asking models to predict human moral judgments on constructed scenarios. The dataset is carefully designed and has been widely used. Its primary limitation for our purposes is that all examples are hypothetical: they capture moral cognition about imagined situations, not behavioral evidence from real ones.

**InstructGPT and RLHF** (Ouyang et al., 2022) represent the dominant practical paradigm for value alignment: collect human preference rankings on model outputs, train a reward model, optimize with reinforcement learning. This approach has produced measurably improved model behavior. Its limitations as a value-learning framework are structural: (a) preferences are scalar and contextless — the annotator is not under pressure, (b) preferences may reflect aesthetic or stylistic bias rather than value judgment, (c) the feedback loop is between model outputs and annotator preferences, not between model behavior and demonstrated human values.

**Constitutional AI** (Bai et al., 2022) operationalizes values as explicit principles — a "constitution" of rules used for self-critique and revision during training. This approach is transparent and scalable, but values are pre-specified by researchers rather than derived from behavioral evidence. The question of whether the specified principles correspond to values as actually demonstrated by humans in practice is separate from the question of whether models follow them.

### 2.3 Historical Text and NLP

The extraction of moral or value content from historical text is a relatively underexplored area of NLP. Studies have applied sentiment analysis and moral lexicons to literary corpora (Klinger et al., 2021), and moral language analysis has been applied to social media text (Hoover et al., 2020) and news corpora (Fulgoni et al., 2016). However, no published dataset is designed specifically to extract value demonstrations from historical figures' documented behavior and to score those demonstrations for resistance — the cost incurred in holding the demonstrated value.

This represents a distinct research gap. The historical record is not a sociological dataset; it is behavioral evidence accumulated over centuries, imperfectly preserved but real. Diaries were written under actual conditions. Letters were sent under actual circumstances. Actions were taken with actual consequences. The gap between hypothetical moral scenarios and this evidence is the gap between what people say and what people do.

### 2.4 The Cost of Ethical Behavior

The concept of "resistance" — the cost of holding a value — is under-theorized in the AI alignment literature. Kenton et al. (2021) discuss the "alignment tax" at the system level: the performance or capability costs of constrained AI behavior. But at the dataset level, the question of what it cost a human actor to demonstrate a value has not been operationalized.

This is a significant gap. A value held at zero cost is weak behavioral evidence — it may be habit, social expectation, or self-interest alignment rather than genuine commitment. The most informative examples of human values in the historical record are precisely those where holding the value came at real cost: reputational damage, physical danger, financial loss, political consequence. Milgram's (1963) obedience experiments demonstrated exactly this: absent cost, people comply with authority far more readily than they believe they would. The inverse — maintaining values when authority demands otherwise — is the rare and informative case.

Ethos operationalizes this insight as the resistance score.

---

## 3. The Ethos Pipeline

### 3.1 Overview

Ethos is a three-stage pipeline:

1. **Ingestion:** A source text is segmented into sentence-bounded passages and stored with metadata (figure identity, document type, publication year, significance score).
2. **Extraction:** Each passage is scanned against a 15-value vocabulary. Matching passages receive a resistance score. Observations are stored in an append-only ledger.
3. **Export:** Observations are classified (P1/P0/APY) using resistance scores and text marker analysis. Training records are written as JSONL with full provenance metadata.

The pipeline is deterministic throughout. No LLM call occurs at any stage. Given identical input and identical thresholds, output is identical. This is a design requirement: a training data generation tool must be reproducible and auditable.

### 3.2 Segmentation

Source text is segmented into sentence-bounded passages of up to 450 characters using a simple sentence-boundary split followed by an accumulation pass. Passages shorter than 30 characters are discarded. The result is typically 2–5 sentence passages that can be read and scored independently.

Passage length is constrained to 450 characters as a practical balance: long enough to capture context (a single sentence is insufficient for value signal detection), short enough that value signal remains attributable to a specific thought rather than a paragraph averaging across several.

### 3.3 The Value Vocabulary

Phase 1 of Ethos uses a keyword vocabulary of 15 named values, each with a set of keyword triggers matched case-insensitively as substrings:

| Value | Representative Keywords |
|-------|------------------------|
| integrity | honest, truth, genuine, transparent, won't lie |
| courage | afraid, brave, risk, facing my fear, terrified |
| compassion | care about, worry about, sad for, heart goes out |
| commitment | promise, committed, dedicated, won't give up |
| patience | patient, waiting, take time, eventually |
| responsibility | my fault, responsible for, accountable, on me |
| fairness | fair, equal, justice, unfair, not right |
| gratitude | grateful, thankful, appreciate, means a lot |
| curiosity | wondering, curious, fascinated, want to know |
| resilience | keep going, bounce back, despite, won't quit |
| love | love, care deeply, cherish, means the world |
| growth | improve, learning, growing, working on myself |
| independence | on my own, my choice, self-reliant |
| loyalty | stand by, loyal, won't leave, through thick and thin |
| humility | I was wrong, my mistake, I need to admit |

The 15-value vocabulary draws from Peterson and Seligman's Character Strengths taxonomy (2004) and Schwartz's universal values framework (1992), selecting values with clear keyword signatures in first-person and behavioral description text. The vocabulary is a Phase 1 approximation; Phase 2 replaces keyword matching with embedding-based clustering over learned value prototype vectors.

At most one observation is recorded per value per passage (the first matching keyword). A passage may produce observations for multiple distinct values.

### 3.4 Resistance Scoring

The resistance score is the core novel contribution of Ethos. It estimates the cost of holding the demonstrated value in the specific passage context.

```
resistance = clip( base + sig_bonus + doc_type_bonus + text_bonus, 0.0, 1.0 )

base          = 0.25                          (always present)
sig_bonus     = min(significance × 0.40, 0.30)
doc_type_bonus = f(document_type)             (see Table 1)
text_bonus    = 0.20                          (if adversity markers detected)
```

**Table 1. Document type bonuses**

| Document Type | Bonus | Rationale |
|---------------|-------|-----------|
| action | +0.40 | Documented behavior under stakes — not words, deeds |
| journal | +0.35 | Private writing — no audience, no performance pressure |
| letter | +0.30 | Directed correspondence — lower performance pressure |
| speech | +0.10 | Public address — highest performance pressure |
| unknown | +0.20 | Default |

The document type bonus encodes a principled intuition from performance theory: the degree to which text is shaped by audience expectation. A private diary entry cannot be written for an audience that does not exist; its content reflects genuine inner state rather than social presentation. A public speech is partially or wholly constructed for effect. These differences in performance pressure correspond to differences in the authenticity of the value signal — and resistance scores adjust accordingly.

The significance score (range 0.0–1.0, set at ingestion) represents the researcher's assessment of source authority. A published, verified diary entry from a primary historical source receives 0.90; a dubious attribution might receive 0.50. Significance propagates through both the resistance score and the final training weight.

The text bonus fires when adversity language markers are detected in the passage. These markers — *even though*, *despite*, *but I still*, *hard to*, *at a cost*, *risk losing*, and related phrases — indicate that the value was held in the face of identified difficulty. Their presence increases the resistance score regardless of document type.

Maximum achievable resistance: 1.0 (clipped from 0.25 + 0.30 + 0.40 + 0.20 = 1.15).
Minimum: 0.25 (base only, speech type, no adversity markers, zero significance).

### 3.5 The Value Registry

For each figure (identified by session_id `figure:<name>`), Ethos maintains a value registry: an aggregate profile across all ingested passages. The primary measure is **weight**:

```
weight = demonstrations × avg_significance × avg_resistance × consistency
```

Where consistency is `1 − (std_dev / mean)` of resistance scores across all observations for a (figure, value) pair, clamped to [0.0, 1.0]. Consistency rewards figures who demonstrate a value stably across diverse contexts, not only in the single highest-stakes moment.

Weight encodes: *how often* × *how much each instance mattered* × *how costly it was* × *how stable the pattern is*.

The registry accumulates across multiple ingestion passes for the same figure. A figure's profile is not fixed at first ingestion — it deepens as more source material is added, which mirrors how real knowledge of a historical figure accumulates.

### 3.6 RIC Classification

During export, each value observation is classified into one of four labels using the Relational Integrity Coefficient (RIC) framework:

**P1 — Value held under meaningful resistance.** The value was demonstrated at real cost. Classified by: high resistance score (≥ 0.55), hold markers in the text (*despite*, *stood firm*, *refused to give*), or APY pressure context with no yield.

**P0 — Value failed or corrupted.** The value was claimed but not upheld, or was abandoned under pressure. Classified by: explicit failure markers (*gave in*, *yielded*, *I lied*, *I rationalized*), or low resistance score (< 0.35) with no hold markers.

**APY — Answer-Pressure Yield.** The most informative negative class: external pressure was explicitly present AND the value failed. Classified by: pressure markers (*under pressure*, *when threatened*, *they demanded*, *forced to*) combined with failure markers. APY captures the moment of capitulation under identified coercion — structurally distinct from simple failure.

**AMBIGUOUS.** Insufficient signal to classify: middle resistance range (0.35–0.55) with no clear markers in either direction. These observations are retained in the database and per-figure files but excluded from aggregate training sets when using `--no-ambiguous`.

Classification priority: APY context is checked first (most specific), then failure markers, then resistance thresholds. Confidence scores (0.40–0.95) reflect the strength of each classification pathway.

---

## 4. The Spectrum Principle

The most consequential design decision in Ethos is not technical — it is the decision about *which figures to ingest*.

### 4.1 The Problem With Poles

Consider the two most common approaches to sourcing historical value data:

**Canonically virtuous figures** (Gandhi, Lincoln, Mandela) produce predominantly P1 observations at high resistance. These are valuable training examples. But a corpus built only from such figures has a structural bias: these figures are documented *because* they were virtuous. The historical record is already filtered toward the heroic, and the corpus inherits that filter. A model trained on this data learns to recognize virtue performance — the markers of recognized virtue as they appear in already-celebrated figures — not the underlying behavioral pattern that constitutes virtue in context.

**Historically negative figures** (genocidaires, corporate criminals at documented moments of fraud) produce predominantly P0 and APY observations. These are also valuable. But the most extreme negative figures present a similar problem in reverse: their failures are so total and so well-documented that a model trained on these examples learns to recognize obvious villainy, not moral drift. The moral failures that cause actual harm in AI systems are not the obvious cases — they are the subtle, incremental rationalizations, the small compromises that compound, the APY dynamics where values erode under sustained pressure.

Both poles fail to train for what they claim to train for. They teach pattern matching on reputation, not judgment in context.

### 4.2 The Middle Ground

The figures of greatest value to an alignment corpus are neither saints nor monsters. They are figures like JFK, MLK, Malcolm X, Churchill, Nixon, Oppenheimer — people of genuine historical consequence with documented asymmetric value profiles.

**Asymmetric profiles** are the norm, not the exception. MLK demonstrated extraordinary `courage` and `commitment` under life-threatening pressure in civil rights contexts. His personal life contains documented failures of `integrity` and `loyalty`. These are not contradictions to be explained away — they are the most realistic picture available of how values actually operate in a human being: domain-specific, context-sensitive, not uniformly distributed.

**Domain-specificity** is itself a training signal. A person who demonstrates `courage` in public activism but not in personal relationships has a different behavioral profile than one whose courage is consistent across domains. Both are informationally richer than a figure whose courage score is uniformly high (saint) or uniformly absent (villain). The asymmetry is the data.

**Value evolution over time** is only visible in the middle ground. Malcolm X's transformation — from the uncompromising separatism of his early activism to the universalism of his post-Mecca period — documents a human being revising deeply held values under accumulated evidence. That process of revision, visible across years of documented writing and speech, is arguably the most important behavioral signal for alignment: values can be learned, revised, and improved. A corpus that only includes figures with stable, unambiguous value profiles cannot represent this.

**APY dynamics** are richest in middle-ground figures. The moment of hesitation before a political compromise, the private letter acknowledging doubt before a public statement of certainty, the documented decision that traded a stated value for a practical gain — these are the APY patterns that appear in abundance in complex figures and almost nowhere in pure exemplars of virtue or villainy.

### 4.3 Implementation

The spectrum principle is implemented architecturally in Ethos through the no-pre-labeling invariant: **no figure is labeled positive or negative at ingestion time**. Classification emerges entirely from the resistance scores and marker patterns in the source text. The pipeline processes Gandhi and Nixon identically. Their profiles emerge from the data.

This design prevents researcher bias from contaminating the corpus at the source — a problem that afflicts any dataset where figures are pre-selected for a label and then data is sourced to confirm it.

---

## 5. Applications and Industry Benefit

### 5.1 Training Data for Value-Aligned Models

The most direct application of Ethos output is as training data for language models undergoing value alignment.

Current RLHF pipelines train on preference rankings that do not distinguish between a model output that is preferred because it reflects genuine value alignment and one that is preferred because it is stylistically pleasing, culturally familiar, or agrees with the annotator's prior beliefs. The Ethos corpus provides a different signal: what does value-aligned behavior look like when holding the value cost something? When the pressure to yield was present but the value held? That is the P1 class.

Equally important: what does value failure look like from the inside? Not as an external judgment, but as a first-person account of rationalization, capitulation, and post-hoc justification? That is the P0 and APY class. A model trained on both classes can distinguish between genuine value demonstration and performance — a distinction that scalar preference rankings cannot provide.

### 5.2 Evaluation and Red-Teaming

Beyond training, Ethos output supports evaluation of existing models. A corpus of historically documented value failures — APY sequences where external pressure was applied and values eroded — serves as a red-teaming dataset: can a model identify the value failure in a passage? Can it recognize the APY dynamic? Can it distinguish between a figure holding a value under pressure and a figure performing the holding of a value without the pressure being real?

These are evaluations that current benchmarks do not support because they lack the APY category and the resistance dimension.

### 5.3 Alignment Research Infrastructure

The Ethos pipeline is designed to be extended. The resistance score is a first approximation of behavioral cost — it can be refined as understanding develops. The value vocabulary is a Phase 1 keyword approximation — it will be replaced with embedding-based clustering in Phase 2. The document type weighting is a principled heuristic — it can be validated empirically against independent assessments of source authenticity.

What Ethos provides as infrastructure is a framework: the conceptual architecture for thinking about value data in terms of *cost*, *authenticity*, and *dynamics under pressure*, and a reproducible pipeline for generating such data from the historical record at scale.

### 5.4 Dataset Scale and Diversity

The historical record is not small. The Library of Congress alone holds more than 170 million items. The Project Gutenberg corpus contains more than 60,000 public domain works. Published diaries, letters, court transcripts, and biographical accounts from thousands of documented historical figures across every major culture and historical period are available in digital form.

Ethos is designed to process any UTF-8 text. A researcher can ingest Marcus Aurelius's *Meditations*, Frederick Douglass's autobiographical writings, Hannah Arendt's letters to Karl Jaspers, Richard Nixon's Oval Office transcripts, and Simone de Beauvoir's journals — each with the appropriate document type — and produce a corpus spanning two millennia, multiple continents, and radically different historical contexts, all scored on the same resistance framework.

The temporal and cultural breadth of the historical record is itself a form of diversity that annotator-sourced datasets cannot replicate: the values demonstrated under pressure in 16th-century Japan and 20th-century Mississippi are expressed in different language, under different social structures, against different forms of resistance — but the value vocabulary extracts the common signal across all of them.

---

## 6. Limitations

### 6.1 Keyword Vocabulary Coverage

The Phase 1 keyword vocabulary captures value signal well for direct, first-person English expression. It systematically under-detects value demonstrations that are:

- **Indirect:** "I gave him everything I had" signals commitment without using any keyword from the commitment list
- **Translated:** Texts originally in Greek, Arabic, Japanese, or other languages may express values in translated English that avoids standard keyword forms
- **Period-specific:** Historical English uses different vocabulary; "I will not flinch" does not contain any current integrity keyword but clearly expresses it

Phase 2 addresses this through embedding-based clustering, which generalizes across paraphrase, translation, and historical diction. Phase 1 data serves as a validation set for Phase 2 development.

### 6.2 Resistance Score Precision

The resistance formula is a principled heuristic, not a validated psychological measure. The document type bonuses (journal +0.35, speech +0.10) are motivated by performance theory but have not been empirically calibrated against independent assessments of behavioral authenticity. The significance score is researcher-assigned and therefore subject to researcher judgment.

These limitations are partially mitigated by the reproducibility of the system: given an agreed significance assignment and document type classification, all downstream scores are deterministic and auditable. But the calibration of the input parameters remains a source of variance.

### 6.3 Historical Corpus Bias

The historical record is not a representative sample of human behavior. It systematically overrepresents:
- Figures with high social status (those whose writings were preserved)
- Western and literate cultures (those with written records)
- Figures in public life (those about whom accounts exist)
- Positive reputation figures (those whose writings were collected and preserved)

Ethos partially addresses the last bias through the spectrum principle — deliberate inclusion of figures with negative reputations — but cannot address the underlying preservation bias in what survives as historical record.

### 6.4 No Temporal Value Dynamics

The current pipeline treats each passage independently. It does not model how a figure's values changed over time, even when multiple ingestion passes cover different periods of a figure's life. Value trajectory — the arc from early Malcolm X to late Malcolm X — is visible in the registry as an accumulation of observations but is not explicitly modeled as a temporal sequence.

---

## 7. Proposed Extensions

The limitations above are not design dead-ends — each has a concrete architectural resolution. We describe these extensions here as a development roadmap, ordered by impact on corpus quality.

### 7.1 Original-Language Scoring

The most significant limitation of Phase 1 is that it scores translated text rather than original-language text. This means extraction results depend on a translator's word choices, not on the figure's own. Two English translations of the same Greek passage from Marcus Aurelius can produce entirely different keyword matches. The extraction is measuring the translator, not Aurelius.

The architectural solution is multilingual embedding scoring. Modern multilingual models — LaBSE (Feng et al., 2022), multilingual-E5, and mBERT — map semantically equivalent sentences from different languages to neighboring points in a shared embedding space. The Greek "δικαιοσύνη" (justice/fairness) and the English "fairness" land near each other without any translation step. Value prototype vectors, once built from English seed examples, generalize to Greek, Arabic, Latin, Japanese, and Chinese automatically in the shared space.

This enables a fundamentally different ingestion mode: **native-language ingestion**. The original text is stored alongside detected language metadata, and scoring runs against the multilingual embedding space directly. Translator bias is structurally eliminated.

A secondary benefit: both the original text and an available translation can be scored independently. Agreement between scores in the two languages increases confidence in the observation. Disagreement between scores is itself a signal — it may indicate a translation that significantly altered semantic content, which is useful both for corpus quality control and for translation analysis.

For the most common classical source languages — Ancient Greek, Latin, Classical Arabic, Classical Chinese — parallel keyword vocabularies in the original language provide an additional high-precision first pass before the embedding path, following the hybrid detection design described in Section 7.3.

### 7.2 Temporal Value Arc and Life-Stage Modeling

Human values are not static across a lifetime. Research in developmental psychology establishes that value priorities shift systematically with age: younger adults weight achievement, stimulation, and autonomy more heavily; older adults tend toward benevolence and security (Schwartz & Rubel, 2005). A corpus that treats a figure's 25-year-old writing and 65-year-old writing as equivalent observations from the same value profile loses this structure entirely.

More concretely: Malcolm X in 1952 and Malcolm X in 1964 are empirically different value profiles. The transformation is documented, dateable, and arguably the most important behavioral data about him — it shows that deeply held values can be revised under accumulated evidence and experience. That trajectory is invisible in a flat lifetime aggregate.

The proposed extension preserves the full lifetime profile while adding temporal sub-sessions derived from the `pub_year` field already stored at ingestion:

```
figure:malcolm_x              ← lifetime aggregate (always maintained)
figure:malcolm_x:1950s        ← decade sub-session
figure:malcolm_x:1960s        ← decade sub-session
```

The registry runs on all three sessions simultaneously. The lifetime profile captures the full behavioral picture. The decade profiles expose the arc. A new `value_trajectory()` query returns the sequence of decade-registry snapshots ordered chronologically, enabling direct visualization of how weight, resistance, and consistency evolve across a figure's life.

A **peak influence period** parameter marks specific year ranges as the figure's most scrutinized era — the period when public attention was highest and therefore when behavioral evidence carries the most interpretive weight. Passages from this period receive a significance multiplier, increasing their contribution to both the registry and the exported training records.

This extension also supports a developmental calibration claim that is difficult to make from current alignment datasets: the relationship between a figure's age, their accumulated experience, and the stability of their value profile. Young figures under pressure demonstrate value patterns that are more volatile; experienced figures show higher consistency. These developmental signatures are currently absent from alignment training data entirely.

### 7.3 Hybrid Detection: Keyword and Embedding with Agreement Confidence

Keyword matching and embedding similarity have complementary failure modes:

| | Keyword Detection | Embedding Detection |
|---|---|---|
| Strength | High precision — exact vocabulary matches are reliable | High recall — catches paraphrase, historical diction, translation |
| Failure mode | Misses synonyms, historical variants, indirect expression | May match semantically adjacent concepts that are not the target value |

A hybrid detection architecture uses both and computes an **agreement confidence** score:

```
keyword_signal  ∈ {0, 1}     (match or no match)
embedding_signal ∈ [0, 1]     (cosine similarity to value prototype vector)

hybrid_score = (α × keyword_signal) + (1 − α) × embedding_signal
agreement_confidence = 1.0 − |keyword_signal − embedding_signal|
```

Where α is a tunable keyword weight (default 0.5). Agreement confidence approaches 1.0 when both methods agree (both detect or both do not detect) and approaches 0.0 when they diverge. Divergence cases are the most informative:

- **Keyword hit, low embedding similarity:** The keyword appeared in a context where it does not carry its value meaning (e.g., "I was afraid of running out of time" matches `courage` keywords but the embedding scores low on the courage prototype). The hybrid score attenuates; agreement confidence is low.
- **No keyword hit, high embedding similarity:** A value was expressed in paraphrase, historical synonym, or translated diction that the keyword vocabulary does not cover. The hybrid score preserves the signal; agreement confidence is low but the observation is flagged as a Phase 2 catch — a case where embedding generalization outperformed keyword matching.

High-agreement observations (agreement_confidence ≥ 0.8) become the highest-quality training examples in the corpus, validated by two independent detection methods. Low-agreement observations require human review before promotion to training status.

This dual-method architecture also provides a continuous mechanism for improving the keyword vocabulary: embedding-only detections that are subsequently validated become candidates for addition to the Phase 1 keyword lists, closing the recall gap iteratively.

### 7.4 Corpus Composition Balancing

The historical record over-represents positively-regarded figures for structural reasons: their writings were preserved, collected, and digitized because they were celebrated. Left uncorrected, a corpus built from the historical record inherits this bias and produces a skewed P1:P0:APY ratio regardless of ingestion strategy.

We propose a two-level balancing framework:

**Figure-level composition guideline:** For every canonically virtuous figure (Gandhi, Lincoln, Mandela), the corpus targets four or more middle-ground figures — figures whose profiles are asymmetric, documented on both value demonstrations and value failures, and representative of the complex majority of historical actors rather than the exceptional minority. This 1:4 ratio is a recommended composition target, not a hard gate. The `cli/balance.py` reporting tool surfaces current ratio metrics and recommends additional ingestion to approach the target.

**Observation-level export weighting:** Within the exported training set, observations are reweighted by inverse label frequency to achieve a configurable target balance. A corpus currently at P1:P0 = 70:30 can be exported at 50:50 effective training weight without discarding any observations — the P0 and APY observations receive higher training weights to compensate for their underrepresentation in the raw corpus.

The balance correction at export time decouples corpus composition (what was ingested) from training data composition (what balance the downstream model sees). Researchers can experiment with different balance targets without re-ingesting, and can report both the raw corpus composition and the effective training composition transparently.

### 7.5 Vocabulary Extension for Temporal Dialect Coverage

The Phase 1 keyword vocabulary is tuned for contemporary English. Historical texts — even those originally written in English — use vocabulary that has shifted significantly. Key examples:

- `fortitude`, `valour`, `steadfastness` → courage
- `verity`, `candour`, `probity` → integrity
- `clemency`, `forbearance`, `benevolence` → compassion
- `fealty`, `fidelity`, `constancy` → loyalty
- `contrition`, `penitence` → humility

Systematic expansion of the keyword vocabulary with period-specific and register-specific synonyms increases recall across historical texts without degrading precision, since these terms carry unambiguous value semantics in their historical contexts. A vocabulary maintenance tool tracks which additions were prompted by embedding-only detections (Section 7.3), creating a continuous improvement loop between the two detection paths.

For non-English original-language texts, parallel keyword vocabularies in Ancient Greek, Latin, Classical Arabic, Classical Chinese, and Early Modern French and German cover the most common sources in the historical record and provide high-precision original-language matching before the multilingual embedding path.

### 7.6 Cross-Passage APY Detection

The most significant structural limitation of the Phase 0 classification system is architectural, not merely a matter of threshold calibration: APY detection is constrained to a single passage window of 450 characters or fewer.

In documented historical behavior, pressure and response rarely co-occur in a single sentence. The pressure event — a threat, a demand, an ultimatum — typically appears in one passage. The behavioral response appears in a subsequent passage, sometimes weeks or years later. The current `classify_observation()` function requires both an APY pressure marker and a failure marker to appear in the same text excerpt. When they are separated, the APY event is invisible to the pipeline: the pressure passage is classified AMBIGUOUS, the failure passage is classified P0 rather than APY, and the most informative label — that this specific value failed specifically because of external pressure — is lost entirely.

This is not an edge case. Historical APY is structurally cross-passage. Consider: Lincoln documents political pressure in a journal entry; the corresponding policy reversal is documented in a speech weeks later. Nixon documents the fear of exposure in private notes; the corresponding deception occurs in a press conference. The single-passage window catches neither.

**This extension is implemented.** The pipeline maintains a short-term **APY context window** per figure: a rolling buffer of the N most recent passages (configurable via `apy_context_window_n` in `core/config.py`; default N=5). When a passage contains APY pressure markers, those markers are written to the figure's context window with a timestamp. When a subsequent passage contains failure markers, the classifier checks whether the context window holds any pressure signal within a configurable age threshold (default: 72 hours for dated materials, 5 passages for undated). If pressure is found, the failure passage is labeled APY rather than P0, and the source pressure passage is referenced by `record_id` in the training record's `pressure_source_id` field.

The data model addition is an `apy_context` table in `values.db` with `(session_id, record_id, ts, passage_idx, markers_json, window_n)`, pruned on each extraction run. The classification logic in `cli/export.py` performs a context lookup before the failure-marker branch.

Cross-passage APY detection also enables a new label subclass: **deferred APY** — cases where pressure is applied and the figure resists in the immediate passage but fails in a later one. This temporal gap between pressure and failure is itself informative: figures who hold longer before yielding under sustained pressure demonstrate a meaningfully different value stability profile than those who yield immediately. The `deferred_apy_lag_s` (seconds) and `deferred_apy_lag_n` (passage count) fields are exported in each cross-passage APY training record.

### 7.7 Keyword Context Disambiguation

The Phase 0 keyword vocabulary is a substring match: if the keyword appears anywhere in the passage, the value is detected. This produces a systematic class of false positives that are not recoverable from the resistance formula alone.

The clearest examples arise from value-word uses that are incidental rather than intentional:

- `afraid` appears in the `courage` keyword list. "I was afraid of being late to the meeting" fires a courage detection. No courage is demonstrated; the passage is about scheduling anxiety.
- `fair` appears in the `fairness` list. "It is fair to say that the weather was poor" fires a fairness detection with no moral content.
- `patient` appears in the `patience` list. "My patient recovered from surgery" fires as a patience demonstration. The word is used as a noun, not a virtue.
- `love` appears in the `love` list. "I would love a cup of tea" fires as a love demonstration. The word is used idiomatically.
- `promise` appears in the `commitment` list. "This promises to be an interesting question" fires a commitment detection with no commitment expressed.

These false positives are not rare corner cases — they are the normal behavior of substring matching against a vocabulary that includes short, polysemous English words. They inflate P1 counts, lower average resistance scores (because the extracted excerpt rarely contains adversity language), and introduce noise into every training record that cites the affected value.

The proposed extension adds a **context disambiguation filter** — a second-pass check applied to any passage that fires on a keyword shorter than a configurable character threshold (default: 12 characters, targeting words like `fair`, `just`, `love`, `brave`, `vow`). The filter checks three conditions in order:

1. **Grammatical role signal:** Does the keyword appear in a pattern consistent with nominal (non-virtue) use? A short list of disqualifying patterns per value catches the most common false-positive forms: `my [value-noun] recovered`, `a [value-noun] of`, `it [value-verb]s to be`, `would [value-verb] [article]`. These are regex-based, deterministic, and require no external model.

2. **First-person grounding:** Is the value expression first-person? A value demonstration requires an agent — the figure must be the one demonstrating the value, not describing someone else's, or using the word abstractly. First-person grounding (`I`, `my`, `me`, `we`, `our`) within a configurable token window of the keyword is a lightweight positive signal. Its absence is not disqualifying — third-person constructions can document demonstrated values — but its presence increases confidence.

3. **Value-relevant action context:** Does the surrounding sentence contain at least one word from a short action-evidence list associated with the value? For `courage`, this includes `stood`, `refused`, `despite`, `pressed`, `faced`. For `fairness`, this includes `judged`, `decided`, `treated`, `owed`, `denied`. Passages with no action-evidence words score lower on an `action_grounding` confidence field added to the observation record.

**This extension is implemented.** The filter does not discard low-confidence observations — it flags them. The `disambiguation_confidence` field in `value_observations` and in the export JSONL ranges from 0.0 to 1.0, where 1.0 means all three conditions passed and 0.0 means the grammatical-role filter fired a probable false positive. The `--min-disambiguation` export flag applies a confidence threshold at export time without re-ingesting. The filter is implemented in `core/value_extractor.py` via `_DISQUALIFIERS` (per-value regex patterns) and `_FIRST_PERSON_RE` (proximity check).

### 7.8 Value Co-occurrence and Interaction Modeling

The Phase 0 pipeline treats the 15 named values as independent dimensions. This reflects a practical decision — it simplifies the extraction loop and the registry schema — but it loses a significant class of behavioral evidence: the relationships between values.

Human moral lives are not independent univariate signals. They are structured by recurring co-occurrences and documented tensions. Courage and integrity co-occur frequently in documented resistance to authority: the same passage that expresses one often expresses the other. Independence and loyalty are frequently in tension: figures who value both leave records of the conflict explicitly. Humility and growth co-occur almost universally in genuine self-reflection passages. Responsibility and courage co-occur in crisis leadership texts. These relationships are not accidents of vocabulary overlap — they reflect real structural features of how human values are expressed and lived.

Modeling co-occurrence at the observation level adds two capabilities:

**Value pair co-occurrence corpus:** The export pipeline gains a `co_occurrence` report: for every ordered pair of values (V₁, V₂), the count of passages in which both were detected, and the rate at which their co-occurrence is P1 (both held), mixed P1/P0 (one held, one failed), or P0 (both failed). A co-occurrence matrix across the 105 unique value pairs produces a behavioral interaction map for each figure and for the cross-figure aggregate.

The behavioral interaction map has direct applications in training data construction: a model that sees examples of integrity-under-pressure as always independent from courage-under-pressure may learn a flatter representation than a model that sees coupled observations where both values were simultaneously tested and both held. The coupled examples are higher-information training records.

**Value tension detection:** When two values that are structurally in tension (independence vs. loyalty, fairness vs. compassion) are both detected in the same passage and one is labeled P1 while the other is labeled P0 or contains failure markers, this is a **value tension event** — a documented case where a figure explicitly prioritized one value over another under pressure. These events are among the most valuable behavioral evidence in the corpus: they reveal value hierarchy, not just value presence.

**This extension is implemented.** A `value_tension` table stores these events: `(session_id, record_id, ts, value_held, value_failed, resistance, text_excerpt)`. The export pipeline includes a `--value-tension` flag that outputs a dedicated `ric_value_tensions.jsonl` file. These records carry 1.5× training weight — the highest in the corpus — because they document not just that a value was held, but that holding it required sacrificing another valued outcome. The co-occurrence matrix (105 pairs) is computed by `_compute_cooccurrence()` in `cli/export.py` and included in `ric_historical_report.json`.

The tension pair list is researcher-configurable in `core/config.py`. The default pairs, derived from established value theory research (Schwartz, 1992; Rokeach, 1973):

| Value A | Value B | Tension Type |
|---------|---------|-------------|
| Independence | Loyalty | Autonomy vs. belonging |
| Fairness | Compassion | Impartiality vs. mercy |
| Courage | Patience | Action vs. deliberation |
| Responsibility | Humility | Ownership vs. deference |
| Commitment | Growth | Persistence vs. revision |

### 7.9 Observation Stability and Consistency Scoring

The current value registry collapses all observations of a given value for a figure into a single weight scalar, computed by the `upsert_registry()` formula. This scalar encodes the accumulated evidence that a value is present, but it discards the *distribution* of the evidence — which is itself a meaningful quality signal.

Consider two figures, both with a registry weight of 0.68 for `integrity`:

- **Figure A:** 45 integrity observations, resistance scores ranging from 0.41 to 0.92, spread across journals, letters, and speeches, across 30 years of documented life.
- **Figure B:** 3 integrity observations, all from the same speech, resistance scores 0.55, 0.58, 0.56.

These are not equivalent evidence bases. Figure A's profile represents a behavioral pattern — integrity demonstrated repeatedly, across contexts, across time, at varying levels of cost. Figure B's profile represents three keyword matches in a single public address. The training data derived from each should be weighted differently, and a model trained on both without that distinction will fail to learn what a stable, cross-context value demonstration looks like.

The proposed extension adds a **consistency score** to the value registry, computed at upsert time:

```
n        = observation count for (session_id, value_name)
r̄        = mean(resistance_scores)
σ_r      = stddev(resistance_scores)
span_s   = max(ts) − min(ts)           (temporal span in seconds)
doc_types = count(distinct document_types observed)

consistency = min(1.0,
    0.30 × min(1.0, n / 10)            ← observation volume (saturates at 10)
  + 0.30 × (1.0 − σ_r / 0.40)         ← resistance stability (low variance = high consistency)
  + 0.25 × min(1.0, span_s / 31536000) ← temporal spread (saturates at 1 year)
  + 0.15 × min(1.0, doc_types / 3)     ← source diversity (saturates at 3 types)
)
```

**This extension is implemented.** The consistency score is stored in `value_registry` as a `consistency` column, computed by `_compute_consistency()` on every `upsert_registry()` call (at both INSERT and UPDATE time). It is propagated to export JSONL as `observation_consistency`. The `--min-consistency` export flag applies a threshold at export time without discarding observations from the database.

Consistency scoring also surfaces a behavioral analysis capability: figures whose value profiles have high mean consistency are making stable, cross-context behavioral claims. Figures with low consistency but high total weight are making many claims in a narrow context — potentially the signature of a figure whose public persona and private behavior diverge. That divergence pattern, when confirmed by cross-document comparison, is itself high-value training data for models that need to understand the difference between value performance and value embodiment.

### 7.10 SRL Integration: Claim-Level Behavioral Validation

Phase 8 of the Ethos roadmap identifies SRL integration as a conceptual future direction. The architecture is now sufficiently developed to specify it concretely. The integration connects in two directions: Ethos corpus data validates SRL observations; the SRL claim-extraction methodology strengthens Ethos observation quality.

**Direction 1: Corpus-grounded claim validation**

AiMe's `claim_extractor.py` (in `modules/srl/`) decomposes model responses into atomic behavioral claims — assertions about values, facts, or identities. The `ric_gate.py` scores each claim for groundedness (G) against known evidence and calibration (C) against uncertainty signals. Currently, the "known evidence" is AiMe's own ledger: prior user-acknowledged facts and verified information.

Ethos provides an alternative evidence base: documented behavioral patterns from historical figures. When a model claims "I will always be honest with you regardless of consequences," the corpus can surface resistance-scored examples of human figures who made equivalent commitments — and examples of figures who made equivalent commitments and then failed under documented conditions. The claim is not validated against preference data or human ratings, but against the behavioral record of what that commitment actually looked like when tested.

The integration is a lookup API at Ethos's REST layer (Phase 5): `GET /corpus/relevant?claim=<text>&value_name=<value>&label=<P1|P0|APY>`. The claim text is embedded (or matched by keyword) against the value observation corpus. The top matching observations are returned with their labels, resistance scores, figure names, and document types. The RIC gate in AiMe uses these as evidence atoms: matching P1 examples increase groundedness confidence; matching P0/APY examples decrease it.

This connects the two systems without coupling them: Ethos remains a standalone corpus pipeline; AiMe's SRL remains a live behavioral scoring system. The API is the only integration surface.

**Direction 2: Claim-level observation quality scoring**

The SRL methodology itself provides a quality criterion that Ethos currently lacks: *assertion level*. AiMe's `claim_extractor.py` scores assertions on a spectrum from direct first-person behavioral claim ("I will not lie") to hedged statement ("I try to be honest when possible") to reported speech ("He said he valued honesty") to abstract assertion ("Honesty is important"). First-person, direct behavioral claims are the highest-quality value signal; abstract assertions are the lowest.

The Phase 0 keyword vocabulary does not distinguish between these levels. "I will not lie" and "Honesty is important in any democracy" both fire on `integrity` keywords and receive the same treatment. The former is a first-person behavioral commitment; the latter is an editorial sentiment with no behavioral content.

Porting the assertion-level classifier from AiMe's `claim_extractor.py` to a standalone `core/claim_level.py` in Ethos requires extracting the assertion-level detection logic, which is deterministic regex-based rather than LLM-based, preserving Ethos's no-LLM-in-extraction invariant. The extracted module runs as a post-processing step on `extract_value_signals()` output, adding an `assertion_level` field (1–4, where 1 = direct behavioral, 4 = abstract) to each observation. The export pipeline accepts `--max-assertion-level` to exclude abstract assertions from training data.

The combination of assertion level with the existing resistance score produces a more precise behavioral quality metric:

| Assertion Level | High Resistance (≥0.55) | Low Resistance (<0.55) |
|----------------|------------------------|----------------------|
| 1 — Direct behavioral | **Highest quality P1/P0 signal** | Claimed but uncosted |
| 2 — Hedged behavioral | Demonstrated with qualification | Probable AMBIGUOUS |
| 3 — Reported speech | Third-party attribution, needs corroboration | Weak signal |
| 4 — Abstract assertion | No behavioral content | Noise |

Level 1 + high resistance is the gold standard for training data. Level 4 + any resistance is noise that the current pipeline promotes to AMBIGUOUS or P1. The assertion level filter converts that noise into a properly labeled low-confidence category.

---

## 8. Conclusion

The central problem with current value-alignment datasets is not their size or their methodology — it is their source material. Hypothetical scenarios capture declared values. Preference rankings capture momentary aesthetic judgment. Neither captures what human values actually look like when they are tested: when holding them costs something, when external pressure is applied, when the easy choice is abandonment and the hard choice is maintenance.

Ethos addresses this gap by turning to the one source of evidence that captures values under real conditions: documented human history. The historical record contains thousands of first-person accounts of values demonstrated, failed, and abandoned under circumstances ranging from mortal threat to political convenience to intimate betrayal. That evidence has never been systematically extracted, scored for resistance, and labeled for value alignment training purposes.

We have described a pipeline that does this extraction: deterministic, auditable, reproducible, requiring no external model calls, and designed around two novel contributions — the resistance score and the spectrum principle — that distinguish Ethos from all existing approaches.

The resistance score encodes the insight that a value held at cost is strong signal and a value held at no cost is weak signal. The spectrum principle encodes the insight that the most useful training data lives not in the celebrated virtuous or the documented villains but in the complex majority of historical figures — people who demonstrated courage in some domains and failure in others, whose value profiles were asymmetric, whose moral lives looked like human moral lives actually look.

The historical record is not a perfect source. It is biased, incomplete, mediated by the choices of those who preserved it, and limited by the vocabulary of its time. But it contains something that no annotation task can manufacture: real stakes. Real consequences. Real people choosing to hold or abandon their values when it mattered.

That is the signal alignment research needs most. Ethos is a framework for extracting it.

---

## References

Bai, Y., Kadavath, S., Kundu, S., Askell, A., et al. (2022). Constitutional AI: Harmlessness from AI feedback. *arXiv preprint arXiv:2212.08073*.

Bommasani, R., Hudson, D. A., Adeli, E., Altman, R., et al. (2021). On the opportunities and risks of foundation models. *arXiv preprint arXiv:2108.07258*.

Emmons, R. A., & McCullough, M. E. (2003). Counting blessings versus burdens: An experimental investigation of gratitude and subjective well-being in daily life. *Journal of Personality and Social Psychology, 84*(2), 377–389.

Fulgoni, D., Carpenter, J., Ungar, L., & Preoțiuc-Pietro, D. (2016). An empirical exploration of moral foundations theory in partisan news sources. In *Proceedings of LREC 2016*.

Haidt, J. (2001). The emotional dog and its rational tail: A social intuitionist approach to moral judgment. *Psychological Review, 108*(4), 814–834.

Haidt, J. (2012). *The Righteous Mind: Why Good People Are Divided by Politics and Religion*. Pantheon Books.

Haidt, J., & Joseph, C. (2004). Intuitive ethics: How innately prepared intuitions generate culturally variable virtues. *Daedalus, 133*(4), 55–66.

Hendrycks, D., Burns, C., Basart, S., Critch, A., Li, J., Song, D., & Steinhardt, J. (2021). Aligning AI with shared human values. In *Proceedings of ICLR 2021*. *arXiv preprint arXiv:2008.02275*.

Hoover, J., Johnson, K., Boghrati, R., Graham, J., & Dehghani, M. (2020). Moral framing and charitable donation: Integrating exploratory social media analyses and confirmatory experimentation. *Collabra: Psychology, 4*(1).

Kenton, Z., Everitt, T., Weidinger, L., Gabriel, I., Garfinkel, B., & Irving, G. (2021). Alignment of language agents. *arXiv preprint arXiv:2103.14659*.

Klinger, R., de Clercq, O., Mohammad, S., & Balahur, A. (2021). IEST: WASSA-2018 implicit emotions shared task. In *Proceedings of EMNLP Workshop*.

Milgram, S. (1963). Behavioral study of obedience. *Journal of Abnormal and Social Psychology, 67*(4), 371–378.

Ouyang, L., Wu, J., Jiang, X., Almeida, D., et al. (2022). Training language models to follow instructions with human feedback. In *Proceedings of NeurIPS 2022*. *arXiv preprint arXiv:2203.02155*.

Peterson, C., & Seligman, M. E. P. (2004). *Character Strengths and Virtues: A Handbook and Classification*. Oxford University Press.

Schwartz, S. H. (1992). Universals in the content and structure of values: Theoretical advances and empirical tests in 20 countries. *Advances in Experimental Social Psychology, 25*, 1–65.
