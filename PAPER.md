# Ethos: A Behavioral Dataset Compiler for Value-Aligned AI

**Author:** John Canady Jr.
**Affiliation:** AI nHancement, Independent Research
**Contact:** john@ai-nhancement.com
**Date:** April 2026
**Code:** https://github.com/ai-nhancement/Ethos

---

## Abstract

Preference-based and constitution-based alignment methods have improved the behavior of language models, but they draw most of their supervision from stated judgments: what people say is right, which outputs they prefer, or which principles researchers decide a model should follow. Those sources matter. What they capture less well is how values are maintained, compromised, or abandoned when real costs are attached to them. This paper introduces Ethos, a behavioral dataset compiler designed to extract value-relevant evidence from documented human conduct rather than from hypothetical scenarios alone. Ethos ingests any text associated with an identifiable person -- historical archives, contemporary journals, personal blogs, interview transcripts, social media posts, or conversational records -- and applies a uniform extraction pipeline: detecting candidate value signals, assigning a resistance score intended to approximate the cost of maintaining a value in context, and optionally verifying candidate labels through a three-model comprehension panel. The system distinguishes among three principal label classes: P1 (value held), P0 (value failed), and APY (Answer-Pressure Yield), where a value is abandoned under identifiable pressure. Ethos also separates value direction from value strength, allowing a passage to be coded not only for whether a value was enacted or violated, but for how strongly the surrounding context supports that interpretation. The pipeline is source-agnostic: the same extraction, scoring, and verification mechanisms apply whether the input is a letter written by Abraham Lincoln in 1863 or a journal entry written by a warehouse worker in 2025. Pilot results across three historical figures and a live contemporary conversational corpus demonstrate that the pipeline operates across both archival and modern sources. Ethos is not presented as a theory of morality or a finished alignment solution. It is proposed as infrastructure for compiling behavior-grounded value data that may complement preference-based, constitutional, and evaluation-focused alignment methods.

---

## 1. Introduction

The alignment problem is often framed in terms of judgment. What outputs do humans prefer? Which responses seem helpful, harmless, or fair? What principles should a model follow when competing values collide? Contemporary alignment pipelines have made substantial progress by collecting these forms of supervision, whether through human preference data, rule-like constitutions, or more explicit attempts to encode broadly shared values (Ouyang et al., 2022; Bai et al., 2022; Hendrycks et al., 2021; Kenton et al., 2021).

Yet there is an important category of evidence these approaches tend to leave underused: documented behavior under pressure.

The point is not that hypothetical judgments are useless. They are indispensable. The point is narrower: a value endorsed in the abstract is not the same thing as a value maintained when maintaining it becomes costly. A person can affirm honesty, fairness, courage, or loyalty in a low-stakes setting and fail each of them when reputation, power, fear, punishment, or group pressure enters the scene. Social psychology has long emphasized that stated norms and actual conduct can diverge, sometimes sharply, when incentives and authority are introduced (Milgram, 1963; Haidt, 2001). Alignment datasets, by contrast, remain richer in declared preferences than in evidence of values under load.

This paper presents Ethos, a pipeline built around that missing layer. Ethos treats documented human conduct -- from any era, any social position, any medium -- as a source of structured, value-relevant evidence. Rather than beginning with moral dilemmas or preference rankings, it begins with any text associated with an identifiable person: historical archives, contemporary journals, personal blogs, interview transcripts, social media confessions, workplace incident reports, or conversational records. It then asks a different question: What values appear in this record, and what happened to those values when the surrounding context imposed real resistance?

The motivating claim is simple: values expressed in comfort are weaker evidence than values maintained under pressure. By the same reasoning, values abandoned under pressure may be especially informative negative evidence. This claim holds regardless of whether the person is famous or anonymous, historical or contemporary, extraordinary or ordinary. A warehouse worker who writes "I hate this job but I won't quit because my kids need stability" is demonstrating commitment under resistance as clearly as any historical figure. Ethos operationalizes this claim by extracting value signals from documented behavior, weighting those signals by contextual resistance, and optionally passing them through a separate verification layer in which three independent models vote on whether a signal should be retained, relabeled, or discarded.

The contribution of the paper is methodological rather than grandiose. Ethos does not claim to solve alignment, discover a universal moral ontology, or transform documented behavior into a ready-made ethical oracle. It contributes four things. First, it formulates behavior-grounded value extraction as a dataset compilation problem applicable to any documented human conduct. Second, it introduces resistance as an explicit heuristic for weighting value evidence by apparent contextual cost. Third, it defines APY as a pressure-aware failure class distinct from generic negative labels. Fourth, it separates extraction from verification through an optional comprehension panel, making disagreement visible instead of hiding it inside a single model's first pass.

What follows is a methods paper: an argument for a kind of data, a concrete compilation pipeline, and pilot evidence that the approach is technically workable.

---

## 2. Related Work

Ethos sits at the intersection of AI alignment, moral psychology, and computational text analysis.

In alignment research, much recent progress has come from learning from human preferences and instructions. Reinforcement learning from human feedback and related methods train models against judgments over outputs rather than against world-grounded demonstrations of value maintenance (Ouyang et al., 2022). Constitutional approaches replace at least part of that supervision with authored principles and self-critique procedures (Bai et al., 2022). Other work has tried to define shared human values more directly or to frame alignment in terms of agent behavior and goal stability (Hendrycks et al., 2021; Kenton et al., 2021). These approaches have been productive, but most still treat alignment data as something closer to judgment than conduct.

Moral psychology offers a different lens. Work on values, moral intuitions, and character strengths has argued that moral life cannot be reduced to explicit reasoning alone and that value structures vary across domains and cultures (Schwartz, 1992; Haidt, 2001; Haidt & Joseph, 2004; Peterson & Seligman, 2004). That literature is relevant to Ethos not because it settles which values are correct, but because it provides structured vocabularies for talking about recurring value dimensions. At the same time, experiments in social psychology underscore that professed commitments can weaken under authority, conformity, or threat (Milgram, 1963). Ethos takes that instability seriously and treats pressure not as nuisance context but as part of the signal.

There is also relevant work in NLP on detecting moral framing, affect, and implicit signals in text (Fulgoni et al., 2016; Hoover et al., 2020; Klinger et al., 2021). Ethos borrows from this tradition in using lexicons, semantic matching, and classification support. Its departure lies in the target of extraction. The goal is not merely to classify whether a text mentions a moral concept, but to compile records of value maintenance, value failure, and value yield under pressure, with provenance and contextual weighting attached.

For that reason, Ethos is best understood as complementary to existing alignment data sources. It is not proposed as a replacement for human preference data or constitutional supervision. It is an attempt to supply a kind of evidence those approaches typically underrepresent.

---

## 3. Problem Formulation

Ethos treats value-relevant behavior as a structured extraction task.

Let a corpus D = {d1, d2, ..., dn} consist of historical or documentary sources associated with identifiable figures, events, or episodes. Each document is segmented into short, sentence-bounded passages p, each passage carrying metadata such as figure identity, source type, date, and significance. Let V = {v1, v2, ..., vk} denote a fixed but revisable value vocabulary, informed by existing value frameworks rather than assumed to be universal (Schwartz, 1992; Peterson & Seligman, 2004).

For each candidate passage-value pair (p, v), Ethos seeks to produce a record of the form:

> e = < figure, passage, v, y, r, m, q >

where *y* is the label, *r* is the resistance score, *m* is the provenance metadata, and *q* is the verification state.

A central design decision is that Ethos separates direction from strength. Direction concerns what happened to the value in the passage: was it upheld, violated, or surrendered under pressure? Strength concerns how much evidential weight should be assigned to that observation. This matters because a value can be mentioned weakly, performed strategically, maintained at substantial cost, or violated under coercive conditions. Treating all such cases as identical labels would flatten the very distinctions Ethos is designed to preserve.

Ethos uses three main directional labels:

- **P1:** the value is held or enacted
- **P0:** the value fails, is corrupted, or is contradicted
- **APY:** the value is abandoned under identifiable pressure

The distinction between P0 and APY requires operational precision. Both are negative labels, and a skeptical reader may ask whether APY is merely a rhetorically renamed failure class. The answer is that the distinction is machine-verifiable rather than interpretive. A P0 classification requires only that the passage contradicts or violates a value. An APY classification requires the conjunction of two independently detectable conditions: (1) the passage contains identifiable pressure markers -- adversity language, coercion indicators, or contextual stakes as detected by the resistance scoring system (Section 4.3) -- and (2) the value fails in the presence of those markers. If condition (1) is absent, the label is P0 regardless of the severity of the failure. If condition (1) is present but the value holds, the label is P1 with elevated resistance. APY therefore captures a specific structural pattern: value failure co-occurring with detectable pressure. This matters for alignment applications because many consequential failures in AI systems are not random errors but structured collapses under competing incentives -- the system yields because something pushed it to yield. A label that encodes that structure is more informative than one that does not.

---

## 4. The Ethos Pipeline

### 4.1 Corpus Ingestion and Provenance

Ethos begins with a corpus of documented human conduct. The pipeline is source-agnostic: it accepts any text associated with an identifiable person, segmented into short, sentence-bounded passages. Source categories include but are not limited to:

- **Historical archives:** journals, letters, speeches, autobiographical passages, documented actions
- **Contemporary personal writing:** blogs, personal essays, diary entries, social media posts
- **Conversational records:** interview transcripts, therapy or counseling records, AI interaction logs
- **Institutional documentation:** workplace incident reports, court testimony, oral histories
- **Community writing:** support group forums, anonymous confessions, advice columns

Each source is stored with provenance metadata. At minimum, the metadata include figure identity (which may be pseudonymous), document type, date, and a source significance score. The figure identity need not be a famous person -- it is simply a persistent key that links multiple passages to the same individual, enabling the pipeline to build a value profile across documents.

This provenance layer is not administrative bookkeeping. It is part of the method. A value signal without a traceable origin is of limited scientific use. Ethos therefore treats every extracted observation as a record tied to a specific passage and source context.

### 4.2 Multi-Layer Signal Extraction

Candidate signals are identified through a seven-layer extraction procedure rather than through a single detector. Each layer uses an independent detection method, and agreement across layers strengthens confidence. The layers are:

**Layer 1: Keyword vocabulary (L1).** Fifteen value vocabularies with 25-35 keyword triggers each, spanning modern assertions, historical and formal diction, action-phrase patterns, and explicit failure markers. Each keyword match includes negation detection: a window preceding the keyword is scanned for negation words ("not," "never," "without," "failed to") and negating prefixes ("dis-," "un-," "in-"), and the signal is suppressed when negation is detected. Failure markers (e.g., "fled," "betrayed," "gave in") are exempt from negation suppression -- "never fled" is positive evidence that the value held. This layer requires no external dependencies.

**Layer 1b: Moral lexicons (L1b).** Two bundled lexicons augment vocabulary detection. The Moral Foundations Dictionary 2.0 (MFD2.0; Frimer et al., 2019) provides approximately 2,100 words annotated across ten Moral Foundations Theory categories. MoralStrength (Araque et al., 2019) provides approximately 450 lemmas with continuous moral valence scores. Both are mapped to Ethos values through a bridge table (e.g., MFT "care" maps to Ethos "compassion"). These are static files with zero runtime cost.

**Layer 1c: Phrase composition (L1c).** A pronoun-aware agency detector that disambiguates compositional phrases involving vice words. "Committed cruelty" and "resisted cruelty" have opposite polarities; "they committed cruelty against him" attributes the vice to a different agent. The phrase layer resolves subject and object using the figure's stored pronoun, preventing false attribution. This layer requires no external dependencies.

**Layer 2: Semantic embedding (L2).** BGE-large-en-v1.5 (1024-dimensional) embeddings encode each passage and compare it against 322 prototype vectors (approximately 20-40 seed sentences per value, averaged and normalized). Prototypes include both modern and archaic-register seeds to bridge historical text. The layer is direction-aware: it queries both hold prototypes (value demonstrated) and failure prototypes (value violated), and suppresses signals where the failure score exceeds the hold score. This prevents violation passages from contributing positive evidence. Requires the sentence-transformers library; fails open if unavailable.

**Layer 3a: Structural patterns (L3a).** A pure-regex layer that detects the context in which values appear: adversity markers, agency indicators, resistance phrases, and stakes language. Each pattern class is scored on a three-tier intensity scale -- mild/concessive (0.30), real cost/active pressure (0.65), existential/irreversible (1.00). The structural score is the mean across four classes, producing a continuous [0.0, 1.0] signal that feeds directly into resistance scoring. This layer requires no external dependencies.

**Layer 3b: Zero-shot entailment (L3b).** DeBERTa-v3-large-zeroshot-v2.0 (Laurer et al.) evaluates custom value hypotheses per passage (e.g., "This passage demonstrates courage in the face of adversity"). Fully deterministic given fixed model weights. This is the most flexible layer in the stack, catching nuanced signals that lexicons and prototypes miss. Requires the transformers library; fails open if unavailable.

**Layer 3c: Moral Foundations classifier (L3c).** A fine-tuned RoBERTa model (MMADS/MoralFoundationsClassifier) trained on over 60 million sentences from political and formal text, producing ten MFT foundation scores per passage. These are mapped to Ethos value confidence through the same bridge table used by the lexicon layer. Known weakness: loyalty_vice detection (F1 = 0.10) is unreliable and excluded from vice scoring. Requires the transformers library; fails open if unavailable.

The pipeline degrades gracefully. If ML dependencies are absent, the keyword, lexicon, phrase, and structural layers run on standard library alone. Each layer produces a confidence score in [0.0, 1.0] per value per passage, and the final observation confidence is a weighted blend across all available layers. Missing or errored layers contribute a skip sentinel rather than zero, ensuring that their absence does not penalize the remaining signals.

The use of multiple independent layers addresses the construct validity concern that a single extraction method would simply find what was designed into it. Layer 1 (keywords) uses researcher-defined vocabulary and is expected to find what was built in. Layer 2 (embedding similarity) finds by meaning, not by word choice. Layers 3b and 3c (zero-shot classifier and MFT classifier) are independently trained models with no Ethos vocabulary. When all layers agree on a signal, the observation rests on convergent evidence from independent detection methods rather than on a single researcher's vocabulary choices.

### 4.3 Resistance as Contextual Cost

The most distinctive feature of Ethos is the resistance score, a heuristic intended to capture the apparent cost of maintaining a value in context. The underlying intuition is straightforward. A value stated when nothing is at stake is weak evidence. A value maintained when honesty risks prison, fairness costs power, or loyalty conflicts with safety is stronger evidence.

Resistance is computed as an additive score with five components, each contributing a bounded increment to a total clipped to [0.0, 1.0]:

- **Base value (0.25).** Every observation receives a floor score reflecting the minimum evidential weight of any detected value signal.
- **Significance bonus (0.00-0.30).** Derived from the passage's independently computed significance score (a composite of affective intensity, contextual novelty, resolution signals, and concern resonance), scaled by 0.40 and capped at 0.30. Higher-significance passages -- those with emotional weight or novelty -- receive proportionally higher resistance credit.
- **Contextual pressure bonus (0.15).** For live conversational data, this fires when the user has at least one active unresolved concern in their relational state. For historical corpora, this slot is replaced by a document-type bonus (see Section 4.4).
- **Correction bonus (0.15).** For live data, this fires when the system was corrected by the user within the preceding five minutes, indicating a context in which the user is actively challenging the system's outputs. For historical corpora, this slot is also replaced by document-type weighting.
- **Adversity text markers (0.20).** A regex-based detector scans for phrases indicating that a value is being held at cost: "even though," "despite," "but I still," "hard to," "scared but," "afraid but," "at a cost," "risk losing," and similar constructions. This component applies identically to historical texts and live conversational data.

The operational formula for historical figure corpora is therefore:

> R(p) = 0.25 + min(sig x 0.40, 0.30) + doc_type_bonus + adversity_markers

And for live conversational data:

> R(p) = 0.25 + min(sig x 0.40, 0.30) + concern_bonus + correction_bonus + adversity_markers

The choice of these components reflects a specific design judgment: resistance should be computable from information already available in the pipeline (significance scores, document metadata, textual content) without requiring a separate annotation pass or a language model call. Each component is independently auditable -- a researcher can inspect which components contributed to any given resistance score and disagree with the weighting without needing to understand the entire system.

The score is heuristic, not psychometric. It is not meant to be a final scientific measure of "moral cost." Its purpose is more modest and more operational: to make the weighting function explicit, auditable, and improvable. In practical terms, the score helps distinguish between two cases that many datasets flatten together: a value performed in low-friction conditions and a value maintained when real pressure pushes the other way.

A natural concern is sensitivity to weighting choices. Would different component weights produce substantially different datasets? Section 8 addresses this directly, but the short answer is that the current weights are a practical starting point chosen to ensure that no single component dominates the score. The base value (0.25) ensures a floor. The significance cap (0.30) prevents high-affect passages from overwhelming the score. The text marker bonus (0.20) is binary and supplements rather than replaces the continuous components. Whether these specific weights are optimal is an empirical question that requires calibration against human annotation -- work that remains to be done.

### 4.4 Source-Type Weighting

Ethos treats document type as evidentially relevant. The key variable is performative pressure: how much audience awareness shapes the writing. A private journal entry, a letter to a close associate, a public speech, and an externally documented action do not place the author under the same performative conditions. Source type therefore enters the resistance scoring formula as a bonus that reflects this distinction.

The current weighting for the document-type bonus in the resistance formula:

- **Documented action (+0.40):** Behavioral evidence with no words to analyze for self-presentation -- the highest-weight category because it reflects what someone did, not what they said.
- **Journal or private writing (+0.35):** No external audience, minimal performance incentive. When Marcus Aurelius admits failure in his private notebook, or when a modern diarist writes about caving to pressure at work, the absence of audience makes the candor more evidentially weighty.
- **Letter or directed communication (+0.30):** Some social stakes, but directed to a specific recipient rather than performed for a crowd.
- **Blog post or social media (+0.20):** Public but often informal; audience awareness is present but variable.
- **Speech or public address (+0.10):** Maximum performance incentive. The lowest-weight category because public speech optimizes for audience reception.

This should not be mistaken for a simple truth hierarchy. Private writing can be candid, but it can also be self-exculpatory. Public speech can be strategic, but it can also be costly and revealing. Social media posts can be performative, but they can also be raw and unfiltered in ways that formal writing never is. Ethos uses source type as a weighting factor because it changes the interpretation of a passage; it does not assume that one document class is inherently truthful in all cases.

### 4.5 Label Taxonomy: P1, P0, and APY

After extraction and scoring, candidate observations are labeled according to the three-way taxonomy defined earlier.

- **P1** identifies value maintenance or enactment.
- **P0** identifies value failure without a clear pressure-yield structure.
- **APY** identifies value abandonment under identifiable pressure.

This taxonomy allows Ethos to represent more than polarity. It captures whether the value was present, what direction the behavior took, and whether pressure played an explanatory role in the failure. That last distinction is especially important for alignment applications, where understanding how systems fail under competing incentives may be as important as understanding which outputs they prefer in stable conditions.

### 4.6 Verification Through a Comprehension Panel

Ethos includes an optional verification layer: the comprehension panel. Rather than letting the extracting model decide its own correctness, the system can submit each candidate signal to three independent models, each asked deliberately simple questions such as whether the passage indicates that a figure upheld or violated a particular value.

The panel then votes. Majority agreement determines whether the signal is retained, relabeled, split into separate candidates, or discarded.

This architecture matters because it separates interpretation from verification. A first-pass extractor is allowed to propose candidates; it is not allowed to treat those proposals as settled. Disagreement becomes visible, and low-agreement examples can be filtered rather than silently absorbed into the dataset. Ethos therefore uses model disagreement as a signal of uncertainty, not as a nuisance to be hidden.

A legitimate concern is that model agreement may reflect correlated bias rather than genuine validation. If three language models share similar training data and similar blind spots, unanimous agreement on a value signal may indicate shared error rather than independent confirmation. The panel is therefore not designed to establish ground truth. It is designed to serve a more limited function: catching obvious extraction errors where the passage text does not support the proposed value label. When the panel disagrees -- when one model says the passage demonstrates courage and another says it does not -- that disagreement is itself informative, because it identifies cases where the textual evidence is ambiguous enough to warrant exclusion from a training dataset that claims to be behavior-grounded.

The panel's value is as a noise filter, not as an oracle. It reduces false positives from the extraction layer at the cost of possibly filtering some true positives where model disagreement reflects genuine interpretive difficulty. For the purpose of compiling a high-precision behavioral dataset, that tradeoff is acceptable. Cases that survive panel agreement are not proven correct, but they are at least not trivially wrong -- a meaningful distinction when the alternative is no verification at all.

---

## 5. Design Principles

Beyond its mechanics, Ethos is defined by several design commitments.

### 5.1 Behavior First

Ethos begins from documented conduct rather than hypothetical preference elicitation. Its core wager is that alignment work may be missing a meaningful category of evidence by focusing so heavily on what people say should happen and too little on what they actually do when incentives bite.

### 5.2 Pressure Is Part of the Signal

Most data pipelines treat adversity, coercion, temptation, and self-interest as confounds or background context. Ethos treats them as central. A value unsupported by resistance may reflect convention, image management, or circumstance. A value sustained through resistance is stronger evidence of commitment. A value abandoned under resistance may be stronger evidence still, albeit in the negative direction.

### 5.3 The Middle Range Is Often the Most Informative

Ethos does not depend on a gallery of saints and villains. In fact, such extremes may be less useful than mixed cases. Much of the signal relevant to alignment lives in the morally uneven record: public courage paired with private failure, fairness in one domain and opportunism in another, revision over time, compromise under pressure, and inconsistent but interpretable conduct. A system trained only on paragons and monsters learns caricature. Ethos is designed for the more difficult middle.

This is also why the pipeline's scope extends beyond historical figures of recognized moral significance. Everyone has values -- positive, negative, mixed, evolving. A single mother working two jobs who writes about the compromises she makes to keep her family stable is demonstrating values under resistance as meaningfully as any statesman. A teenager posting about peer pressure on social media is producing APY-relevant data. A worker describing why they stayed silent when they saw something wrong is documenting a value failure under pressure with the same structural pattern that Ethos detects in historical archives. The behavioral signal is in the conduct, not in the fame of the person exhibiting it. A dataset built exclusively from historically prominent figures would inherit a systematic bias toward elite, literate, and culturally dominant perspectives. Mixing historical and contemporary sources -- famous and anonymous, powerful and ordinary -- produces a more representative picture of how values actually behave across the full range of human experience.

### 5.4 No Pre-Labeling of Figures

The pipeline does not begin by classifying people as morally positive or negative. It processes a figure's record and allows the profile to emerge from the evidence. This matters methodologically because pre-labeling invites shortcut learning. If a model knows in advance that a figure is meant to stand for virtue or corruption, it may learn reputation rather than behavior.

---

## 6. Pilot Results

The current results are preliminary. They establish feasibility and describe the operational characteristics of the pipeline across both historical and live data. They do not establish downstream efficacy.

### 6.1 Multi-Figure Pilot

A pilot run processed 37 passages across three historical figures: Gandhi (14 passages, journal source), Lincoln (10 passages, letter source), and Marcus Aurelius (13 passages, journal source). The pipeline produced 93 value observations across the three corpora: 46 from Gandhi, 19 from Lincoln, and 28 from Aurelius.

Nine of the fifteen vocabulary values appeared across all three figures: commitment (12 observations), resilience (10), fairness (9), courage (9), humility (8), patience (5), gratitude (5), curiosity (3), and compassion (3). An additional five values appeared in two of three figures: integrity (12 observations across Gandhi and Aurelius), love (6), responsibility (5), loyalty (2), and growth (2). One value (independence) appeared only in the Gandhi corpus.

This overlap should not be overread. It does not establish universality, cultural invariance, or a settled ontology of virtues. But it does suggest that the extraction procedure is capable of recovering overlapping value structures from texts that differ in era, genre, and political context -- and that the overlapping values (commitment, resilience, fairness, courage) are the ones with the highest observation counts, indicating that the pipeline's sensitivity is highest for values that recur across diverse behavioral records.

### 6.2 Gandhi Full-Pipeline Test

The Gandhi corpus produced 46 candidate observations across 14 passages spanning 12 distinct values. After comprehension-panel review, 38 of 46 observations were retained -- an 82.6% retention rate. Panel review reduced the number of apparently spurious P0 assignments, suggesting that the verification layer improves precision on negative labels specifically.

The three most frequently detected values in the Gandhi corpus were integrity (9 observations), resilience (6), and courage (6). Mean resistance scores were high across all detected values (range: 0.90-1.00), consistent with the source material: Gandhi's journals document values maintained under conditions of political persecution, imprisonment, and social opposition -- precisely the high-resistance contexts Ethos is designed to capture.

### 6.3 Resistance Score Distribution

Across all 93 historical observations, resistance scores clustered in the elevated (0.5-0.7) and high (0.7-1.0) ranges, with no observations below 0.30. This concentration reflects two properties of the pilot corpora: the source material was selected for documented behavioral content (not casual writing), and the document-type bonus for journals and letters (0.35 and 0.30 respectively) elevates the floor. A broader corpus including speeches (document-type bonus 0.10) and mixed-stakes material would be expected to produce a wider resistance distribution.

### 6.4 Live Operational Data

In addition to the historical pilot, Ethos has been running continuously on live conversational data from a single-user persistent AI system since deployment. As of the time of writing, the live pipeline has produced 957 value observations across 15 values. The most frequently detected value in live data is fairness (448 observations), followed by love (108), integrity (107), growth (69), and gratitude (66). Resistance scores in live data show a wider distribution than historical data, with the majority falling in the 0.5-0.7 range (761 observations) and a substantial high-resistance cluster (280 observations above 0.7).

The live data is presented as evidence that the pipeline operates at scale on naturalistic conversational input, not as evidence for any specific alignment claim. It demonstrates that the extraction and scoring mechanisms are robust enough to run continuously without human intervention and that the value vocabulary produces non-degenerate distributions across sustained real-world use.

### 6.5 Caution

These results do not show that Ethos improves aligned model behavior. They do not establish inter-annotator reliability, and they do not yet demonstrate robustness across cultures, translations, or source types. What they do show is that the pipeline is technically coherent: signals can be extracted, scored, and filtered in a way that produces structured outputs rather than a loose pile of moralized interpretations -- and that it operates reliably across both curated historical corpora and uncurated live conversational data.

---

## 7. Discussion

### 7.1 What Kind of Data Ethos Produces

Ethos is not best described as a moral text classifier. It is better understood as dataset infrastructure. Its output is not a single judgment about a passage but a provenance-rich record of behavior-relevant evidence: who, where, when, which value, what direction, under what apparent resistance, and with what degree of verification.

That distinction matters for alignment. A compiler produces training examples, evaluation items, and red-team cases that can be reused across model families and experimental settings. The ambition of Ethos is therefore not merely to classify text more cleverly. It is to expand what counts as alignment data.

### 7.2 Why Pressure Matters for Alignment

A large share of alignment work implicitly assumes that the interesting question is what response humans endorse when asked to judge one. That is a valuable question, but it is not the only one. Systems deployed in the world confront conflicting incentives, partial information, adversarial prompts, and role pressures. If alignment is partly about how a model behaves when one objective pulls against another, then data about values under pressure may be unusually relevant.

APY is especially important in this respect. Many harmful failures are not random. They are structured collapses under pressure: yielding to incentives, protecting power, masking uncertainty, or prioritizing convenience over principle. A label that distinguishes pressure-yield from generic failure may therefore have value not only for training but for evaluation and red-teaming.

### 7.3 Source Diversity as a Design Requirement

Historical material is attractive because it contains precisely the mixture alignment research often lacks: ideals, compromise, revision, rationalization, endurance, and collapse under real stakes. It is also attractive because it provides narrative context, not just isolated choices. Values rarely appear as abstract toggles. They appear in episodes.

But a dataset built exclusively from historical figures would be biased in at least three ways. First, it would overrepresent elite perspectives -- the people whose behavior was documented and preserved were disproportionately powerful, literate, and culturally dominant. Second, it would overrepresent high-stakes moral drama -- historical records survive precisely because they document exceptional circumstances, not ordinary ones. Third, it would underrepresent the quiet, undramatic forms of value maintenance and failure that characterize most human moral life: the daily compromises, the small acts of integrity that no one records, the values that erode not under persecution but under fatigue, financial pressure, or social inconvenience.

Contemporary sources correct all three biases. A personal blog written by a home health aide contains value signals under resistance that no historical archive captures. An anonymous forum post about workplace ethics represents a moral context that history rarely preserves. Conversational AI interaction logs -- where a user expresses values in the course of sustained, unperformed dialogue -- may be among the least performative text sources available, because the user is not writing for an audience.

The point of Ethos is not to train systems to imitate any particular person, historical or contemporary. It is to give researchers a corpus in which values appear under load across the full breadth of human experience. The pipeline is source-agnostic by design because the behavioral signal it extracts -- value held, value failed, value abandoned under pressure -- is the same structural pattern regardless of who produces it.

---

## 8. Limitations

Ethos has clear limitations, and any serious evaluation of the method has to begin there.

First, the value vocabulary is still a chosen schema. The current fifteen-value vocabulary is informed by established work on values and character strengths (Schwartz, 1992; Peterson & Seligman, 2004), but it is a practical scaffold rather than a discovered universal map of morality. A different schema would yield a different dataset. This is a feature of the architecture, not a defect of the implementation -- Ethos is designed to accept alternative vocabularies -- but it means that any downstream results are conditional on the vocabulary used to produce them.

Second, resistance is a heuristic with unvalidated weights. The five-component additive formula (Section 4.3) makes the weighting explicit and auditable, but explicitness does not substitute for validation. The specific weights (base 0.25, significance cap 0.30, context bonuses 0.15 each, text markers 0.20) were chosen to prevent single-component dominance and to ensure that no observation scores below 0.25 or above 1.0. But whether these weights produce resistance rankings that agree with human judgments of contextual cost is an open empirical question. The most important next step for resistance validation is a sensitivity analysis: holding the corpus fixed, varying the component weights across a systematic grid, and measuring how much the resulting dataset changes in composition and ranking. If the dataset is robust to moderate weight perturbations, the specific values matter less. If small weight changes produce large dataset shifts, the current weights require calibration against human annotation before they can be treated as more than a reasonable starting point.

Third, corpus diversity is both the solution and the challenge. A dataset built only from historical archives inherits preservation bias -- favoring the literate, the powerful, and the culturally dominant. Contemporary sources correct this bias but introduce others: social media posts are performative in different ways than speeches, anonymous forum posts lack verifiable provenance, and conversational records raise privacy considerations that historical archives do not. The pipeline is source-agnostic by design, but any particular compiled dataset will reflect the biases of its input corpus. Researchers using Ethos-compiled data should report corpus composition (historical vs. contemporary, elite vs. non-elite, source-type distribution) alongside their results.

Fourth, source type is informative but ambiguous. A private journal may be less performative than a public speech, yet also more self-justifying. A documented action may be behaviorally rich, yet mediated through later interpretation. A social media post may be raw and unfiltered, or carefully curated for audience effect. Weighting source types helps, but it does not dissolve the underlying epistemic uncertainty.

Fifth, the pilot corpus is small and source-narrow. Three historical figures from roughly overlapping intellectual traditions (Indian independence, American governance, Roman Stoicism) do not constitute a diverse sample. The nine-value overlap observed across all three figures (Section 6.1) may reflect genuine cross-cultural recurrence, or it may reflect the limited cultural range of the pilot. The live conversational data (Section 6.4) demonstrates that the pipeline operates on contemporary sources, but a systematic comparison of value distributions across historical and contemporary corpora -- and across elite and non-elite figures -- has not yet been conducted.

Sixth, the compiler can fail in specific, predictable ways. The keyword-based extraction layer will miss values expressed through action descriptions without value-laden vocabulary ("he returned the money" demonstrates integrity without using the word). It will produce false positives when value vocabulary appears in non-value contexts ("I love pizza" is not a demonstration of the value of love). And the resistance score will overweight passages with adversity language that is rhetorical rather than experiential ("despite all odds" in a graduation speech). These failure modes are systematic rather than random, which means they introduce directional bias into the dataset rather than noise. A future version of Ethos should include explicit failure-mode auditing: sampling observations from each failure category and reporting error rates alongside the compiled dataset.

Seventh, the comprehension panel is not ground truth. Agreement among three models is useful as a filter, but it can still reflect shared blind spots or shared training-set biases. Model consensus should be treated as a verification signal, not as final adjudication.

Finally, the central practical question remains unanswered: does Ethos-built data improve downstream alignment outcomes? At present, the evidence supports feasibility, not efficacy.

---

## 9. Future Work

Several next steps follow naturally from the current design.

The first is systematic corpus diversification. The pipeline should be run across contemporary sources -- personal blogs, forum posts, interview transcripts, anonymized conversational records -- alongside historical archives. The most important empirical question is whether value distributions and resistance patterns differ systematically between historical and contemporary corpora, between elite and non-elite figures, and between public and private writing. If they do, those differences are themselves informative about how documentation context shapes behavioral evidence.

The second is original-language support. Translation can flatten moral nuance and distort the signals most relevant to value extraction. Native-language processing would improve both cultural coverage and interpretive fidelity, and would be especially important for contemporary non-English sources where translation introduces unnecessary information loss.

The third is temporal modeling. Values are not static traits. People revise, harden, fracture, and rationalize over time. Ethos would be substantially more useful if it could represent value trajectories rather than isolated passage-level observations. This is particularly feasible with contemporary sources, where a single person's blog or journal may span years of documented value evolution in a way that fragmentary historical records rarely permit.

The fourth is stronger validation. That includes human-annotator comparison, ablation studies, systematic error analysis, and clearer reporting on where the panel helps and where it fails.

The fifth, and most important, is downstream evaluation. Ethos becomes truly consequential only if the data it compiles improves model behavior, red-team coverage, calibration, or alignment evaluation in practice. A critical test is whether models trained on Ethos-compiled data that includes both historical and contemporary sources -- mixing elite and ordinary figures, famous and anonymous, positive and negative value profiles -- produce more robust alignment than models trained on any single source category alone.

---

## 10. Conclusion

Current alignment datasets are strong at capturing stated judgments and preferences. They are weaker at capturing what values look like when those values are tested.

Ethos is an attempt to build that missing layer.

It does so by treating documented human conduct -- from any era, any social position, any medium -- as a source of alignment-relevant evidence, by treating resistance as a meaningful variable rather than background noise, by distinguishing pressure-yield from ordinary failure, and by separating extraction from verification through an independent panel. The pipeline is deliberately source-agnostic: the same extraction, scoring, and verification mechanisms that process a letter written by Abraham Lincoln also process a blog post written by a nurse, a forum confession written by a teenager, or a conversational record from a persistent AI system. Values under pressure are values under pressure, regardless of who is holding them.

Its larger claim is modest but important: alignment may require not only principles, preferences, and hypothetical cases, but also datasets built from how values actually behave under real stakes -- across the full range of human experience, not only among the historically prominent.

That is the premise Ethos is designed to test.

---

## Code Availability

The complete Ethos pipeline -- including all seven extraction layers, resistance scoring, the comprehension panel, CLI tools, REST API, and 934 passing tests -- is open-source and available at https://github.com/ai-nhancement/Ethos. The repository includes sample corpora, batch ingestion tools, and export utilities for producing labeled JSONL training data. Verum, the companion certification layer, is included in the same repository.

---

## References

Araque, O., Gatti, L., & Kalimeri, K. (2019). MoralStrength: Exploiting a moral lexicon and embedding similarity for moral foundations prediction. *Knowledge-Based Systems, 191*, 105184.

Bai, Y., Kadavath, S., Kundu, S., Askell, A., et al. (2022). Constitutional AI: Harmlessness from AI feedback. *arXiv preprint arXiv:2212.08073*.

Frimer, J. A., Boghrati, R., Haidt, J., Graham, J., & Dehghani, M. (2019). Moral Foundations Dictionary 2.0. Unpublished manuscript.

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
