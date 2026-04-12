# Ethos: A Resistance-Weighted Behavioral Corpus Proposal for Value-Aligned AI Training

**[Author: John V. - AI-nhancement]**

---

## Abstract

The alignment of artificial intelligence systems with human values remains one of the central unsolved problems in AI safety research. Existing approaches draw primarily on hypothetical moral scenarios designed by researchers or human preference rankings collected at scale. Both are useful, but both emphasize declared judgment more than demonstrated behavior. We present **Ethos**, a behavioral extraction pipeline intended to derive value-relevant signal from documented historical behavior: journals, letters, speeches, and recorded actions of historical figures across the human spectrum, from widely admired figures to deeply compromised ones. Ethos introduces four main ideas: (1) a **resistance score** that operationalizes the cost of holding a demonstrated value in context; (2) a **document authenticity weighting** scheme that calibrates evidence quality based on likely performance pressure at the time of writing; (3) a **two-axis value model** (resistance x polarity) that separates demonstration strength from constructive or destructive direction; and (4) an optional **comprehension panel**, a three-model majority-vote verification pass that can independently confirm, relabel, or discard extracted signals after deterministic extraction has completed. The pipeline produces labeled training records in three principal classes: P1 (value held under meaningful resistance), P0 (value failed or corrupted), and APY (Answer-Pressure Yield - value abandoned under identified external pressure). We argue that a potentially useful source of alignment signal lies not only at the poles of celebrated virtue or obvious villainy, but also in the middle ground where asymmetric value profiles, domain-specific courage, and documented moments of both integrity and failure coexist. In a preliminary 12-passage Gandhi sample, a full-pipeline run produced 46 observations across 12 values; 38/46 were retained by the verification panel, and the panel reduced apparent false-positive P0 signals in that pilot setting. Cross-figure pilot testing across Gandhi, Lincoln, and Marcus Aurelius identified five values present in all three figures across widely separated historical contexts. We present Ethos as a reproducible pipeline and research direction for extracting behaviorally grounded value signals from historical text, not as a completed solution to value alignment.

---

## 1. Introduction

When we ask whether an AI system is "aligned with human values," we are implicitly assuming we know what human values look like in practice. The evidence suggests we are less certain than we often assume.

Most datasets used to train value-aligned models fall into one of two categories. The first category consists of **hypothetical moral scenarios**: carefully constructed vignettes asking annotators to judge the morality of an action, assign blame, or rank outcomes (Hendrycks et al., 2021). These datasets are useful and methodologically clear, but they measure declared moral judgment in imagined situations. Decades of social psychology research document the gap between moral intention and moral behavior under pressure (Milgram, 1963; Haidt, 2001). A dataset of intentions is not the same thing as a dataset of behavior.

The second category consists of **human preference rankings**: pairs of model outputs labeled by annotators as better or worse along axes such as helpfulness and harmlessness (Ouyang et al., 2022). These datasets are large-scale and have produced measurably improved model behavior. But preference rankings are scalar and momentary. They capture which of two outputs an annotator preferred at one point in time, not the underlying value that drove the preference, and not whether that preference would hold under adversarial pressure, personal cost, or sustained challenge.

The result is that current value-alignment training data, whatever its scale and quality, mostly addresses the question: *what do people say is right?* The question it addresses less directly is: *what do people actually do when holding a value costs them something?*

This paper introduces **Ethos**, a pipeline intended to investigate that second question.

Ethos extracts value signal from historical text - the documented words and actions of real people in real situations with real stakes. It scores each observation on how much it may have cost to hold that value: the **resistance score**. It calibrates the authenticity of evidence based on the performance conditions under which it was produced: the **document type** signal. And it classifies extracted observations into three labels - P1 (held), P0 (failed), and APY (yielded under pressure) - that together aim to capture more of the dynamics of values under pressure than simple positive or negative labels alone.

Two research gaps motivate this work. First, to our knowledge, no existing NLP dataset is built specifically to extract human value evidence from historical text as a behavioral corpus for alignment work. The historical record contains many documented instances of humans demonstrating, failing, and struggling with specific values under conditions ranging from mortal threat to political coercion to intimate betrayal, yet this type of evidence is only weakly represented in current alignment data. Second, current alignment datasets do not typically quantify **resistance**: the cost of holding a value in context. In Ethos, resistance is treated as an operational variable rather than a validated psychological measurement, but we argue it may capture an important distinction between low-cost value expression and higher-cost value maintenance.

Our contributions are:

1. A **behavioral extraction pipeline** (Ethos) that generates value signal from historical text using multi-layer extraction - keyword, lexicon, phrase, structural, semantic, and classifier layers - with no LLM required for base operation.
2. A **resistance scoring framework** that estimates the cost of demonstrating a value in context, incorporating document authenticity, significance, and adversity language markers.
3. A **three-class labeling scheme** (P1/P0/APY) that captures held values, failed values, and pressure-yield dynamics.
4. A **two-axis value model** (resistance x polarity) that distinguishes the strength of value demonstration from its constructive or destructive direction.
5. A **pronoun-aware phrase layer** that disambiguates first-person agency from third-person description and passive framing.
6. An optional **comprehension panel** - three-model majority-vote verification - that fires post-extraction to filter false positives and refine signal labels in pilot runs.
7. An argument for the **spectrum principle**: that a particularly useful source of training signal for value alignment may lie not at the behavioral poles alone, but in the complex middle ground where asymmetric value profiles, domain-specific courage, and documented moral failure coexist in the same person.

---

## 2. Background and Related Work

### 2.1 Value Theory in Psychology

Human value research has produced several taxonomies relevant to AI alignment work.

**Schwartz's Theory of Basic Human Values** (Schwartz, 1992) proposes ten universal values organized around two bipolar dimensions: Openness to Change vs. Conservation, and Self-Enhancement vs. Self-Transcendence. Crucially, Schwartz operationalizes values as motivational goals that express themselves in behavioral patterns over time, not merely momentary preference.

**Peterson and Seligman's Character Strengths and Virtues** (2004) taxonomizes 24 character strengths organized under six virtues: wisdom, courage, humanity, justice, temperance, and transcendence. Their framework is explicitly behavioral and was designed to be cross-culturally and historically grounded.

**Haidt's Moral Foundations Theory** (Haidt and Joseph, 2004; Haidt, 2012) proposes that human moral reasoning is organized around several recurring foundations, including Care/Harm, Fairness/Reciprocity, Loyalty/Betrayal, Authority/Subversion, Sanctity/Degradation, and Liberty/Oppression. This framework has been used in NLP analysis of moral language, including the Moral Foundations Twitter Corpus (Hoover et al., 2020).

These frameworks share a common assumption that Ethos inherits: values are observable in behavior, not only in self-report. Where they often rely on survey instruments or annotated modern datasets, Ethos turns to the historical record as a source of behavior under cost.

### 2.2 Value Alignment Datasets

**The ETHICS Dataset** (Hendrycks et al., 2021) spans five moral domains with approximately 130,000 examples asking models to predict human moral judgments on constructed scenarios. Its primary limitation for our purposes is that its examples are hypothetical. They capture moral cognition about imagined situations, not behavioral evidence from real ones.

**InstructGPT and RLHF** (Ouyang et al., 2022) represent the dominant practical paradigm: collect human preference rankings on model outputs, train a reward model, then optimize against that reward signal. This approach has produced measurably improved model behavior. Its limitations as a value-learning framework are structural: preferences are scalar and context-thin, the annotator is not under pressure, and the feedback loop is between model outputs and annotator preferences rather than between model behavior and demonstrated human values.

**Constitutional AI** (Bai et al., 2022) operationalizes values as explicit principles used for self-critique and revision during training. This approach is transparent and scalable, but values are pre-specified by researchers rather than derived from behavioral evidence.

### 2.3 Historical Text and NLP

The extraction of moral or value content from historical text is relatively underexplored in NLP. Studies have applied sentiment analysis and moral lexicons to literary corpora (Klinger et al., 2021), and moral language analysis has been applied to social media text (Hoover et al., 2020) and news corpora (Fulgoni et al., 2016). However, we are not aware of prior work that frames historical text explicitly as a corpus for extracting behavioral value demonstrations and failures for alignment-oriented use.

### 2.4 The Cost of Ethical Behavior

The concept of "resistance" - the cost of holding a value - is under-theorized in the AI alignment literature. Kenton et al. (2021) discuss the "alignment tax" at the system level: the performance costs of constrained AI behavior. But at the dataset level, the question of what it cost a human actor to demonstrate a value has not, to our knowledge, been operationalized in this way.

A value held at zero cost is weak behavioral evidence. It may reflect habit, social expectation, or alignment with self-interest rather than genuine commitment. The most informative examples of human values in the historical record may be those where holding the value came at real cost: reputational damage, physical danger, financial loss, or political consequence. Ethos operationalizes this intuition as the resistance score.

---

## 3. The Ethos Pipeline

### 3.1 Overview

Ethos is a three-stage pipeline:

1. **Ingestion:** A source text is segmented into sentence-bounded passages and stored with metadata such as figure identity, document type, publication year, and significance score.
2. **Extraction:** Each passage is scanned against a 15-value vocabulary and additional supporting layers. Matching passages receive a resistance score. Observations are stored in an append-only ledger.
3. **Export:** Observations are classified (P1/P0/APY) using resistance scores, polarity cues, and text marker analysis. Training records are written as JSONL with provenance metadata.

The base pipeline is intended to be reproducible. The keyword, lexicon, phrase, structural, semantic, and classifier layers produce the same output given the same inputs, thresholds, software environment, and model weights. This is a design requirement: a training data generation tool should be auditable.

An optional fourth stage - **verification** - may be applied after extraction. The comprehension panel queries three LLMs in parallel with binary questions and applies majority-vote verdicts to filter, relabel, or split extracted signals. This stage is disabled by default. When enabled, it introduces bounded non-determinism in post-processing, while preserving the original extraction outputs.

### 3.2 Segmentation

Source text is segmented into sentence-bounded passages of up to 450 characters. Passages shorter than 30 characters are discarded. The passage length constraint balances two requirements: long enough to capture context, short enough that value signal remains attributable to a specific thought rather than averaged across a longer paragraph.

### 3.3 The Value Vocabulary

Ethos uses a keyword vocabulary of 15 named values, each with a set of keyword triggers matched case-insensitively:

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

The vocabulary draws from Peterson and Seligman's Character Strengths taxonomy (2004) and Schwartz's universal values framework (1992), selecting values with relatively clear signatures in first-person and behavioral description text. This vocabulary is still a design choice, not a discovered universal ontology, and should be understood as a practical starting schema.

### 3.4 Resistance Scoring

The resistance score is the core proposal of Ethos. It estimates the cost of holding the demonstrated value in the specific passage context.

```text
resistance = clip(base + sig_bonus + doc_type_bonus + text_bonus, 0.0, 1.0)

base           = 0.25
sig_bonus      = min(significance x 0.40, 0.30)
doc_type_bonus = f(document_type)
text_bonus     = 0.20  if adversity markers are detected
```

**Table 1. Document type bonuses**

| Document Type | Bonus | Rationale |
|---------------|-------|-----------|
| action | +0.40 | Documented behavior - not words, deeds |
| journal | +0.35 | Private writing - lower performance pressure |
| letter | +0.30 | Directed correspondence - lower performance pressure |
| speech | +0.10 | Public address - highest performance pressure |
| unknown | +0.20 | Default |

The document type bonus encodes a simple intuition from performance theory: not all textual evidence is equally informative. A private journal entry is typically less performative than a public speech. The resistance score should be understood as an operational heuristic, not as a validated psychological instrument.

### 3.5 The Value Registry

For each figure, Ethos maintains a value registry: an aggregate profile across all ingested passages. The primary measure is **weight**:

```text
weight = demonstrations x avg_significance x avg_resistance x consistency
```

Here consistency rewards figures who demonstrate a value stably across contexts rather than only in one high-stakes moment. Weight is intended to encode frequency, evidential quality, cost, and stability together.

### 3.6 Classification

During export, each value observation is classified into one of four labels:

- **P1** - Value held under meaningful resistance.
- **P0** - Value failed or corrupted.
- **APY** - Answer-Pressure Yield: external pressure is explicitly present and the value fails.
- **AMBIGUOUS** - Insufficient signal to classify.

The APY class is important because it attempts to distinguish simple failure from failure under identified pressure. This distinction is central to the Ethos framing even if its eventual utility must be evaluated empirically.

### 3.7 Phrase Layer and Polarity Scoring

Ethos includes a pronoun-aware phrase layer that distinguishes first-person agency from third-person description and passive framing. It also applies a polarity layer so that value observations can be represented on two axes: strength and direction. This matters because high-resistance language may still point toward destructive conduct rather than constructive demonstration.

### 3.8 Comprehension Panel

The optional comprehension panel queries three models after extraction with binary questions such as "does the figure hold this value?" and "does the figure violate this value?" Majority-vote agreement is then used to retain, discard, or relabel signals. This layer is not part of the deterministic base path and is disabled by default.

### 3.9 Extended Pipeline Features

Beyond the base extraction path, Ethos currently includes several quality extensions:

1. Cross-passage APY detection.
2. Keyword context disambiguation.
3. Value co-occurrence and tension modeling.
4. Observation consistency scoring.
5. Source chain tracking.
6. Corpus quality gating.

These extensions are intended to reduce brittleness in a problem that is unlikely to be solved by any single detector.

---

## 4. The Spectrum Principle

The strongest training signal may not lie at the behavioral poles alone.

### 4.1 The Problem With Poles

**Canonically virtuous figures** such as Gandhi, Lincoln, or Mandela produce many high-resistance P1 observations. These are useful examples. But a corpus built only from such figures risks inheriting the historical record's own reputational filtering: these figures are remembered in part because they were judged exemplary.

**Historically negative figures** produce many P0 and APY observations. But the most extreme negative figures present the reverse problem: their failures are so total and so heavily documented that a model trained only on those examples may learn to recognize obvious villainy rather than moral drift.

Both poles can encourage shortcut learning on reputation rather than action in context.

### 4.2 The Middle Ground

The figures of greatest interest to an alignment corpus may be neither saints nor monsters, but people with documented asymmetric value profiles: JFK, MLK, Malcolm X, Churchill, Nixon, Oppenheimer, and many others.

Such figures matter for at least three reasons:

1. **Asymmetric profiles are normal.** People show courage in one domain and failure in another.
2. **Value evolution becomes visible.** Historical figures can revise their commitments over time.
3. **APY dynamics become legible.** Hesitation, compromise, pressure, and partial yield are often clearest in the middle ground.

### 4.3 Implementation

The spectrum principle is implemented through the **no-pre-labeling invariant**: no figure is labeled positive or negative at ingestion time. The pipeline processes Gandhi and Nixon identically. Their profiles emerge from the text and scoring framework rather than from prior reputation labels.

---

## 5. Preliminary Results

We report two sets of preliminary results. The first is a multi-figure pilot exploring cross-figure convergence. The second is a single-figure deep test of the full pipeline including the comprehension panel.

### 5.1 Multi-Figure Pilot

Extraction across three historical figures - Gandhi (journal passages, 1927), Lincoln (letters, mixed period), and Marcus Aurelius (*Meditations*, around 180 AD) - served as a pilot test of the pipeline and a first check on cross-figure value convergence.

**Pilot corpus summary:**

| Figure | Document Type | Passages | Values Detected | Mean Resistance |
|--------|--------------|----------|----------------|-----------------|
| Gandhi | journal | ~14 | 15 | 0.89 |
| Lincoln | letter | ~10 | 12 | 0.76 |
| Marcus Aurelius | journal | ~13 | 11 | 0.82 |

**Cross-figure value convergence:** Five values were detected in all three figures: `commitment`, `resilience`, `fairness`, `courage`, and `humility`. This result is encouraging, but it should be treated as pilot evidence rather than definitive proof that the 15-value vocabulary captures culture- or period-robust behavioral signals.

### 5.2 Full-Pipeline Verification Test (Gandhi)

A full-pipeline extraction - including all deterministic layers and the comprehension panel - was run on a 12-passage Gandhi sample covering nonviolent resistance, imprisonment, South Africa racial hierarchy positions, brahmacharya experiments, Noakhali riot courage, and public self-criticism.

**Results:**

- **46 observations** extracted across **12 distinct values**
- **38/46** panel-retained observations
- **4 P0 signals** remaining after panel filtering, down from approximately 11 pre-panel in this pilot setting
- P0 examples included South Africa racial hierarchy acceptance (fairness) and brahmacharya-related integrity and responsibility concerns
- P1 examples included nonviolent resistance to the Rowlatt Act, courage in Noakhali, and public acknowledgment of personal failures

**Sample source chains:** `keyword+semantic+zeroshot+panel`, `keyword+structural+panel`, `keyword+lexicon+mft+panel`

These results suggest that the multi-layer architecture may improve signal precision, especially when multiple layers agree and the optional panel retains the signal. They do not by themselves validate Ethos as a training corpus.

**Panel performance note:** At the current base-tier DigitalOcean models (`openai-gpt-oss-120b`, `deepseek-r1-distill-llama-70b`, `openai-gpt-oss-20b`), verification ran at approximately 34 seconds per passage.

These results are preliminary. They are intended to demonstrate pipeline functionality and panel behavior rather than to validate the corpus as a training resource. Validation at scale - with broader figure coverage, independent annotation of sample observations, ablations, and downstream model evaluation - remains future work.

---

## 6. Potential Applications

### 6.1 Training Data for Value-Aligned Models

The most direct application of Ethos output is as a candidate source of training data for language models undergoing value alignment.

Current RLHF pipelines train on preference rankings that do not distinguish between a model output preferred because it reflects genuine value alignment and one preferred because it is stylistically pleasing or culturally familiar. Ethos proposes a different signal: what does value-aligned behavior look like when holding the value cost something? When pressure to yield was present but the value held?

Equally important is the negative side: what does value failure look like from the inside, not as an external judgment, but as a first-person account of rationalization, capitulation, or post-hoc justification? In principle, a model exposed to both P1 and P0/APY classes may become better at distinguishing between genuine value demonstration and value performance. Whether this leads to better downstream behavior remains an empirical question.

### 6.2 Evaluation and Red-Teaming

Beyond training, Ethos output could support evaluation of existing models. A corpus of historically documented APY sequences - where external pressure is present and values erode - could serve as a red-teaming dataset: can a model identify the value failure in a passage? Can it recognize the APY dynamic? Can it distinguish between a figure holding a value under pressure and a figure merely performing conviction?

These evaluations are not well supported by current benchmarks, which generally lack both an APY category and an explicit resistance dimension.

### 6.3 Scale and Diversity

The historical record is extensive. Published diaries, letters, court transcripts, autobiographies, and biographical accounts from many cultures and periods are available in digital form. Ethos is designed to process any UTF-8 text.

If the pipeline holds up under broader validation, a researcher could ingest Marcus Aurelius's *Meditations*, Frederick Douglass's autobiographical writings, Hannah Arendt's letters, Richard Nixon's transcripts, and Simone de Beauvoir's journals - each with the appropriate document type - and build a corpus spanning centuries and contexts. That kind of temporal and cultural breadth differs from annotator-sourced datasets, even if it brings its own biases.

---

## 7. Limitations

### 7.1 Keyword Vocabulary Coverage

The keyword vocabulary captures value signal best for relatively direct English expression. It under-detects demonstrations that are indirect, translated, or period-specific. Later layers mitigate this, but they do not remove the problem.

### 7.2 Resistance Score Calibration

The resistance formula is a principled heuristic, not a validated psychological measure. The document type bonuses are motivated by performance theory but have not been empirically calibrated against independent assessments of behavioral authenticity. The significance score is researcher-assigned and subject to researcher judgment.

### 7.3 Historical Corpus Bias

The historical record systematically overrepresents figures with high social status, Western and literate cultures, figures in public life, and figures whose writings were preserved. Ethos can partially respond through deliberate corpus design, but it cannot eliminate preservation bias.

### 7.4 Static Value Profiles

The current pipeline treats each passage independently. Value trajectory - for example the arc from early Malcolm X to later Malcolm X - is visible in accumulation but is not yet explicitly modeled as a temporal sequence.

### 7.5 Validation Burden

The more expressive the pipeline becomes, the greater the validation burden becomes. Ethos ultimately requires inter-annotator comparison, ablation testing, error analysis, and downstream evaluation to justify strong claims about usefulness in model alignment. At present, the evidence is preliminary.

---

## 8. Future Work

### 8.1 Original-Language Scoring

The most significant limitation is that scoring currently runs mainly on translated text rather than the figure's original language. Multilingual embedding models could support native-language ingestion and reduce translator bias.

### 8.2 Temporal Value Arcs

Decade-partitioned sub-sessions such as `figure:malcolm_x:1960s` could expose value trajectory - how a figure's weights, resistance means, and consistency scores evolve across life stages - while maintaining the lifetime aggregate profile.

### 8.3 Hybrid Detection with Agreement Confidence

Replacing binary keyword matching with a hybrid score combining keyword signal and embedding cosine similarity could convert disagreement between detection methods into a quality signal. High-agreement observations would become stronger training candidates; low-agreement observations would be flagged for review.

### 8.4 Corpus Composition Balancing

A corpus balance tool reporting current P1:P0:APY ratios and figure-type composition could help prevent the corpus from inheriting the historical record's own bias toward celebrated figures.

### 8.5 Companion Evaluation Layer

If Ethos matures into a larger corpus, one possible extension is a companion evaluation layer: a service that compares AI model outputs against patterns in the historical behavioral record. At present, this should be understood as a future application concept rather than a validated product direction.

---

## 9. Conclusion

The central problem with current value-alignment datasets may be less their size or methodology than their source material. Hypothetical scenarios capture declared values. Preference rankings capture momentary preference and judgment. Neither directly captures what human values look like when they are tested: when holding them costs something, when external pressure is applied, and when the easy choice is abandonment rather than maintenance.

Ethos is an attempt to address this gap by turning to one source of evidence that does capture values under real conditions: documented human history. The historical record contains many first-person and behaviorally described accounts of values demonstrated, failed, and abandoned under circumstances ranging from mortal threat to political convenience to intimate betrayal. We argue that this evidence is underused in alignment-oriented dataset construction.

We have described a pipeline that performs this extraction: deterministic in its core layers, auditable through source chain tracking, and organized around four main ideas - the resistance score, the spectrum principle, the two-axis polarity model, and the optional comprehension panel. We have also described quality extensions - cross-passage APY detection, keyword context disambiguation, value co-occurrence and tension modeling, observation consistency scoring, and pronoun-aware phrase analysis - that are meant to reduce brittleness in a difficult task.

The APY class deserves particular emphasis. A value held at cost may be strong signal. A value abandoned under identified external pressure may be an especially informative negative signal. The dynamics of that failure - the gap between pressure and capitulation, the rationalizations that accompany it, and the difference between immediate and delayed yield - are behavioral patterns that current alignment datasets rarely represent directly.

The resistance score encodes the hypothesis that a value held at cost is stronger signal than a value held at no cost. The spectrum principle encodes the hypothesis that some of the most useful training data may live not in the celebrated virtuous or the documented villains alone, but in the complex majority of historical figures whose value profiles are asymmetric and whose moral lives look more like actual human moral lives.

The historical record is not a perfect source. It is biased, incomplete, and mediated by preservation. But it contains something that no annotation task can fully manufacture: real stakes, real consequences, and real people choosing to hold or abandon their values when it mattered.

Whether this is ultimately the signal alignment research needs most remains an empirical question. Ethos is offered as a framework for investigating that possibility.

---

## References

Bai, Y., Kadavath, S., Kundu, S., Askell, A., et al. (2022). Constitutional AI: Harmlessness from AI feedback. *arXiv preprint arXiv:2212.08073*.

Feng, F., Yang, Y., Cer, D., Arivazhagan, N., & Wang, W. (2022). Language-agnostic BERT sentence embedding. In *Proceedings of ACL 2022*.

Fulgoni, D., Carpenter, J., Ungar, L., & Preotiuc-Pietro, D. (2016). An empirical exploration of moral foundations theory in partisan news sources. In *Proceedings of LREC 2016*.

Haidt, J. (2001). The emotional dog and its rational tail: A social intuitionist approach to moral judgment. *Psychological Review, 108*(4), 814-834.

Haidt, J. (2012). *The Righteous Mind: Why Good People Are Divided by Politics and Religion*. Pantheon Books.

Haidt, J., & Joseph, C. (2004). Intuitive ethics: How innately prepared intuitions generate culturally variable virtues. *Daedalus, 133*(4), 55-66.

Hendrycks, D., Burns, C., Basart, S., Critch, A., Li, J., Song, D., & Steinhardt, J. (2021). Aligning AI with shared human values. In *Proceedings of ICLR 2021*.

Hoover, J., Johnson, K., Boghrati, R., Graham, J., & Dehghani, M. (2020). Moral framing and charitable donation: Integrating exploratory social media analyses and confirmatory experimentation. *Collabra: Psychology, 4*(1).

Kenton, Z., Everitt, T., Weidinger, L., Gabriel, I., Garfinkel, B., & Irving, G. (2021). Alignment of language agents. *arXiv preprint arXiv:2103.14659*.

Klinger, R., de Clercq, O., Mohammad, S., & Balahur, A. (2021). IEST: WASSA-2018 implicit emotions shared task. In *Proceedings of EMNLP Workshop*.

Milgram, S. (1963). Behavioral study of obedience. *Journal of Abnormal and Social Psychology, 67*(4), 371-378.

Ouyang, L., Wu, J., Jiang, X., Almeida, D., et al. (2022). Training language models to follow instructions with human feedback. In *Proceedings of NeurIPS 2022*.

Peterson, C., & Seligman, M. E. P. (2004). *Character Strengths and Virtues: A Handbook and Classification*. Oxford University Press.

Schwartz, S. H. (1992). Universals in the content and structure of values: Theoretical advances and empirical tests in 20 countries. *Advances in Experimental Social Psychology, 25*, 1-65.

Schwartz, S. H., & Rubel, T. (2005). Sex differences in value priorities: Cross-cultural and multimethod studies. *Journal of Personality and Social Psychology, 89*(6), 1010-1028.
