# Ethos: A Behavioral Dataset Compiler for Value-Aligned AI

**[Author: John V. - AI-nhancement]**

---

## Ethos in 60 Seconds

Ethos is a pipeline for extracting value-relevant training data from documented human behavior rather than from hypotheticals alone.

It does four things:

1. It reads historical material such as journals, letters, speeches, and documented actions.
2. It extracts value signals such as courage, integrity, fairness, or failure of those values.
3. It scores each signal by **resistance**: how costly it appears to have been to hold that value in context.
4. It optionally sends the extracted signal through a **comprehension panel**: three independent models voting on whether the signal should be kept, relabeled, or discarded.

The core idea is simple: a value stated in comfort is weak evidence. A value held under pressure is stronger evidence. A value abandoned under pressure may be especially informative negative evidence.

Ethos is not presented here as a solved theory of morality or a finished alignment solution. It is a research direction and a reproducible pipeline for building behavior-grounded value data.

---

## Abstract

The alignment of artificial intelligence systems with human values remains one of the central unsolved problems in AI safety. Most existing alignment data is built from hypothetical scenarios or human preference rankings over model outputs. These approaches capture what people say is right, but only partially capture what people do when values are tested under real conditions.

We present Ethos, a behavioral extraction pipeline that builds value-relevant training data from documented human behavior: journals, letters, speeches, and recorded actions across the human spectrum. Ethos is based on a simple premise: values in comfort are weak evidence. Values under pressure are stronger. Values that fail under pressure may be the most informative of all.

The system introduces four linked ideas: a resistance score that operationalizes the cost of maintaining a value in context; document authenticity weighting that distinguishes private writing from public performance; a two-axis model separating value strength from value direction; and an optional comprehension panel, where three independent models vote on whether extracted signals should be retained, relabeled, or discarded.

Ethos produces three principal label classes: P1 (value held), P0 (value failed), and APY (Answer-Pressure Yield, where a value is abandoned under identifiable pressure). Rather than focusing only on idealized exemplars or obvious failures, Ethos targets the middle range of human behavior - where commitment, compromise, failure, and revision coexist under real stakes.

We present Ethos as a behavioral dataset compiler for AI alignment: a method for constructing training and evaluation data grounded in observed human conduct rather than hypothetical judgment.

---

## 1. Introduction

Most alignment datasets answer some version of this question:

*What do people say is right?*

That question matters. But it is not sufficient.

A second question is at least as important - and far less studied:

*What do people actually do when holding a value costs them something?*

There is a gap between stated values and demonstrated behavior. People endorse principles in comfort that they do not always maintain under pressure. Social psychology has documented this gap for decades. Yet most alignment data remains closer to stated judgment than to lived behavior.

A person may endorse honesty in principle, but choose differently when honesty risks prison, reputation, or power.

Ethos is an attempt to build data for that missing layer.

Instead of starting with rules, hypothetical scenarios, or preference rankings, Ethos starts with documented history: diaries, private letters, public speeches, autobiographical records, and observed actions. It asks whether these materials can be transformed into a structured dataset of value-relevant behavior under real conditions.

The core claim is simple:

*values under pressure are more informative than values in comfort.*

If that claim holds, then current alignment methods may be missing a critical category of signal. Models trained primarily on preferences and rules may learn to reproduce acceptable outputs, but not necessarily to model what value maintenance looks like when incentives push in the opposite direction.

Ethos is designed to explore whether a behavior-grounded dataset can help fill that gap.

---

## 2. Why Ethos Matters

Ethos matters because it shifts the source material of alignment work.

Most current pipelines rely on one of three sources:

1. Human judgments about hypothetical moral scenarios.
2. Human preferences over model outputs.
3. Researcher-authored constitutions or principles.

Those are all valuable. But they are still one step removed from lived behavior.

Ethos asks a different question: what if alignment data were built from documented human conduct under real conditions?

That shift matters for three reasons.

### 2.1 It moves from declared values to demonstrated values

A person can say they value honesty, courage, or fairness. That tells us something. But it tells us less than seeing whether they maintained those values when it became costly.

Ethos is built around that difference.

### 2.2 It treats pressure as signal, not noise

Most datasets flatten away the very thing that may matter most: adversity, coercion, danger, temptation, self-interest, and fear. Ethos treats those as part of the evidence. The point is not simply whether a value appears. The point is whether it survives contact with resistance.

### 2.3 It creates a path toward behavior-grounded alignment data

If successful, Ethos would not just be another ethics dataset. It would be a way of compiling behavioral evidence for alignment work: training examples, red-team cases, and possibly evaluation sets built from real-world human value maintenance and value failure.

That is why the simplest shorthand for Ethos is:

**a behavioral dataset compiler for AI alignment**

Given this shift in source material, the next question is how such behavior can be represented and measured.

---

## 3. Core Idea

Ethos is built around three intuitions.

### 3.1 Values are more informative when they cost something

A value held at no cost may reflect habit, convenience, or social performance. A value held under threat, sacrifice, or sustained pressure is stronger evidence of commitment.

Ethos operationalizes that intuition through the **resistance score**.

### 3.2 Failure under pressure is not the same as failure in general

A person can fail a value casually, repeatedly, or under direct coercion. Those are not the same pattern. Ethos treats **Answer-Pressure Yield (APY)** as a distinct negative class: the value is present, pressure is present, and the person yields.

That distinction matters because many real failures are not simple negations. They are erosions under stress.

### 3.3 The middle ground is where much of the signal lives

The most useful historical figures for this kind of corpus may not be the clean exemplars or the obvious monsters. They may be the mixed cases: people with asymmetric profiles, domain-specific courage, public principle and private failure, or visible value revision over time.

Ethos calls this the **spectrum principle**.

---

## 4. How the Pipeline Works

At a high level, Ethos has four stages.

### 4.1 Ingestion

Source material is segmented into short, sentence-bounded passages and stored with metadata such as:

- figure identity
- document type
- publication year
- significance score

### 4.2 Extraction

Each passage is scanned for value-relevant signal using multiple layers:

- keyword matching
- lexicon support
- phrase and agency analysis
- structural cues such as adversity and stakes
- semantic similarity
- independent classifier support

The reason for multiple layers is straightforward: no single detector is reliable enough for this task on its own.

### 4.3 Scoring

Each extracted observation receives a **resistance score** based on:

- a base value
- source significance
- document type
- adversity markers in the text

The score is not meant to be a final scientific measurement of moral cost. It is an explicit heuristic for ranking how strong the behavioral evidence appears to be.

### 4.4 Verification

After extraction, Ethos can optionally run a **comprehension panel**.

This is one of the most important parts of the system. Three independent models are asked simple binary questions such as:

- does the figure hold this value?
- does the figure violate this value?

The models then vote. Majority agreement determines whether the signal is retained, relabeled, split, or discarded.

This matters because it turns verification into a separate layer rather than burying it inside extraction. Rather than relying on a single model's interpretation, Ethos tests whether a value signal survives disagreement across independent models.

In other words, Ethos does not only extract signals. It also creates a path for validating them across model disagreement.

---

## 5. Main Concepts

### 5.1 Resistance

**Resistance** is Ethos's term for the apparent cost of maintaining a value in context.

The central idea is straightforward: not all value expressions carry the same evidential weight.

Saying "honesty matters" in a low-stakes setting is weak evidence.

Remaining honest when it risks punishment, loss of status, or personal harm is stronger evidence.

Abandoning honesty under that same pressure may be the most informative negative signal.

Examples of resistance include:

- risk of punishment or imprisonment
- reputational damage or public backlash
- physical danger or threat
- conflict with personal interest or group loyalty

Ethos does not treat these as noise to be removed. It treats them as part of the signal itself.

The resistance score is not a final or validated psychological measure. It is an explicit heuristic for ranking how costly a value appears to be in context. Its purpose is to distinguish between:

- values expressed when nothing is at stake
- values maintained when something real is at risk

That distinction is central to the project.

### 5.2 P1, P0, and APY

Ethos uses three main label classes:

- **P1**: the value is held
- **P0**: the value fails or is corrupted
- **APY**: the value fails under identified pressure

This is meant to capture more than simple positive versus negative sentiment. It captures value maintenance, value failure, and value yield under pressure as related but distinct patterns.

### 5.3 Document Type

Not all sources carry the same evidential weight.

Ethos distinguishes between:

- documented actions
- journals
- letters
- speeches

The intuition is simple: a private journal entry usually involves less performance pressure than a public speech. Document type therefore affects the resistance score and export weighting.

### 5.4 Spectrum Principle

Ethos does not pre-label figures as morally positive or morally negative. The same pipeline processes Gandhi and Nixon, Lincoln and Churchill, Marcus Aurelius and Malcolm X. Profiles are supposed to emerge from evidence rather than reputation labels.

This is a strong design choice for the project. It reduces shortcut learning based on status and pushes the system toward context-sensitive judgment.

---

## 6. Preliminary Results

The current results should be understood as pilot evidence, not final validation.

### 6.1 Multi-Figure Pilot

Pilot extraction across Gandhi, Lincoln, and Marcus Aurelius suggests that the pipeline can detect recurring values across widely separated historical contexts.

Five values appeared in all three pilot figures:

- commitment
- resilience
- fairness
- courage
- humility

This is encouraging, though not yet enough to make strong claims about universality or cross-cultural robustness.

### 6.2 Gandhi Full-Pipeline Test

In a 12-passage Gandhi sample, a full-pipeline run produced:

- 46 observations across 12 values
- 38 of 46 retained by the comprehension panel
- a reduction in apparent false-positive P0 signals after panel verification

These results suggest that the layered approach is technically workable and that the comprehension panel may improve precision. They do not yet show that Ethos improves downstream alignment performance.

That next step remains open.

---

## 7. What Is New Here

Ethos is not simply another moral text classifier. It represents a shift in what counts as alignment data.

Most existing approaches focus on judgment: what people say is right, or how they evaluate model outputs. Ethos shifts the focus toward behavior: what people actually did when values were tested.

Its novelty lies in the combination of:

**Behavior-first source material**  
Not hypothetical scenarios, but documented human conduct under real conditions.

**Resistance-weighted evidence**  
Values are not treated as binary labels, but as signals whose strength depends on the cost of maintaining them.

**Pressure-aware failure modeling (APY)**  
Failures under pressure are treated as a distinct and informative class, rather than collapsed into generic negatives.

**No pre-labeling of figures**  
Moral profiles emerge from observed behavior rather than reputation or category assignment.

**Independent verification through model disagreement**  
The comprehension panel introduces a separate validation layer, where signals must survive scrutiny across multiple models.

Taken together, these changes shift alignment data from:

- static labels -> contextual behavior
- stated preference -> observed evidence
- hypothetical scenarios -> historical reality

Ethos is best understood not as a classifier, but as infrastructure for compiling behavior-grounded alignment data.

---

## 8. Limitations

Ethos has important limitations.

### 8.1 The value schema is still a schema

Even if grounded in established psychology frameworks, the 15-value vocabulary is still a chosen structure. It is a practical starting point, not a discovered universal map of morality.

### 8.2 Resistance is a heuristic

The resistance score is useful as an operational variable, but it is not yet a validated measure. It needs calibration, testing, and external comparison.

### 8.3 Historical evidence is biased

The historical record overrepresents literate, high-status, preserved figures and underrepresents countless others. Ethos can mitigate some of this through corpus design, but it cannot remove preservation bias.

### 8.4 Validation remains the real test

To justify stronger claims, Ethos still needs:

- inter-annotator comparison
- ablation studies
- systematic error analysis
- downstream model evaluation

At present, the evidence supports feasibility, not final efficacy.

---

## 9. Future Work

The next steps are clear.

### 9.1 Broader Corpus Testing

The pipeline should be run across many more figures, cultures, eras, and source types.

### 9.2 Original-Language Support

Native-language processing could reduce translation bias and improve cross-cultural coverage.

### 9.3 Temporal Value Arcs

Ethos should eventually model how a figure's value profile changes over time, not only how values appear in isolated passages.

### 9.4 Downstream Alignment Evaluation

The most important unresolved question is whether Ethos-built data improves model behavior, evaluation quality, or red-team coverage in practice.

That is the test that ultimately matters.

---

## 10. Conclusion

Current alignment datasets are strong at capturing stated judgments and preferences. They are weaker at capturing what values look like when they are tested.

Ethos is an attempt to build that missing layer.

It does so by treating documented human behavior as a potential source of alignment data, by treating resistance as a meaningful variable, by separating failure under pressure from failure in general, and by validating extracted signals through independent model voting.

The larger promise of Ethos is not that it solves alignment. It is that it may change the kind of data we use to approach it.

If alignment needs more than rules, more than preferences, and more than hypotheticals, then it may also need datasets built from real human conduct under real stakes.

If alignment depends on understanding how values behave under pressure, then datasets that exclude that condition may be systematically incomplete.

Ethos treats history as training data for values.

That is the possibility Ethos is trying to test.

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
