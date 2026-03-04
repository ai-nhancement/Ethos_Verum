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

Future work should model value trajectories explicitly, tracking how resistance patterns and value demonstrations shift across decades of a figure's documented life.

---

## 7. Conclusion

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
