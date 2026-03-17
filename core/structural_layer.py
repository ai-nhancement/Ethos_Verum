"""
core/structural_layer.py

Layer 3 signal extraction for Ethos.

Two independent sub-layers:

  A. Structural patterns (pure regex, zero deps)
     Detect the context in which values appear — agency, adversity, and
     resistance markers. Each of four pattern classes (adversity, agency,
     resistance, stakes) is scored on a three-tier intensity scale:

       T1 = 0.30  mild / concessive  ("despite", "continued")
       T2 = 0.65  real cost / active pressure  ("at great risk", "refused")
       T3 = 1.00  existential / irreversible  ("risking everything", "would not betray")

     structural_score = mean of the four class scores → continuous [0.0, 1.0].
     "despite" and "risking everything" no longer score identically.

  B. Zero-shot entailment (HuggingFace DeBERTa, lazy-load, fail-open)
     Model: MoritzLaurer/deberta-v3-large-zeroshot-v2.0
     Tests each passage against a per-value hypothesis string.
     Entailment probability → confidence score in [0.0, 1.0].
     Returns [] if model unavailable or inference fails.

Constitutional invariants:
  No LLM call.
  No network call at import time.
  All methods return safe defaults on error (fail-open).
  Sub-layer B degrades to [] if model unavailable; sub-layer A is always on.
"""

from __future__ import annotations

import logging
import re
import threading
from typing import Dict, List, Optional, Tuple

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tier constants
# ---------------------------------------------------------------------------

_T1: float = 0.30   # mild / concessive signal
_T2: float = 0.65   # real cost / active pressure
_T3: float = 1.00   # existential / irreversible / moral-betrayal stakes

# ---------------------------------------------------------------------------
# Sub-layer A — Structural patterns (tiered)
# ---------------------------------------------------------------------------
#
# Each class has three regexes (T1, T2, T3).
# _class_score() checks T3 first and returns the highest tier that matches.
# structural_score() = mean of the four class scores.
#
# Class scores are value-agnostic — they characterise the CONTEXT, not the
# value.  The caller uses the aggregate score to boost resistance scoring
# and disambiguation confidence.

# --- ADVERSITY CLASS -------------------------------------------------------
# T1: concessive conjunctions — something is being acknowledged but not at
#     measurable personal cost.
_ADV_T1 = re.compile(
    r"\b(?:in\s+spite\s+of|although|even\s+though|notwithstanding|despite)\b",
    re.IGNORECASE,
)

# T2: real cost and active external pressure — cost is acknowledged and
#     somewhat specific, or the opposition is named.
_ADV_T2 = re.compile(
    r"\b(?:"
    r"at\s+(?:great\s+)?(?:personal\s+)?(?:risk|cost|price|peril)|"
    r"knowing\s+(?:the\s+)?(?:cost|risk|danger|consequences?)|"
    r"at\s+the\s+expense\s+of|"
    r"against\s+(?:all\s+)?(?:opposition|resistance|advice|orders|the\s+odds)|"
    r"under\s+(?:pressure|threat|fire|attack|duress|siege|interrogation)|"
    r"in\s+the\s+face\s+of"
    r")\b",
    re.IGNORECASE,
)

# T3: existential and irreversible cost — life, career, freedom named
#     explicitly as what was put at risk or sacrificed.
_ADV_T3 = re.compile(
    r"\b(?:"
    r"risking\s+(?:everything|my\s+life|his\s+life|her\s+life|their\s+lives|"
    r"my\s+career|my\s+reputation|his\s+career|her\s+career)|"
    r"sacrific(?:ing|ed)\s+(?:my|his|her|their|everything)|"
    r"gave\s+(?:up\s+)?(?:everything|my\s+life|his\s+life|her\s+life)|"
    r"laid\s+down\s+(?:his|her|their|my)\s+life"
    r")\b",
    re.IGNORECASE,
)

# --- AGENCY CLASS ----------------------------------------------------------
# Requires first-person subject (I|we).  Third-person biographical text is
# handled separately by doc_type (action) weighting.

# T1: passive continuation — the subject kept going without a named choice.
_AGC_T1 = re.compile(
    r"\b(?:I|we)\s+(?:still\s+|yet\s+|nevertheless\s+|nonetheless\s+)?"
    r"(?:continued|remained?|stayed?|kept\s+going|marched\s+on)\b",
    re.IGNORECASE,
)

# T2: active resistance verbs — the subject made an explicit choice to hold.
_AGC_T2 = re.compile(
    r"\b(?:I|we)\s+(?:still\s+|yet\s+|nevertheless\s+|nonetheless\s+)?"
    r"(?:refused?|chose|stood|held|pressed\s+on|carried\s+on|"
    r"persisted?|rose|rebuilt?|recovered?|endured?|persevered?)\b",
    re.IGNORECASE,
)

# T3: explicit refusal of a named failure act — the subject is described as
#     refusing to do something morally or situationally specific.
_AGC_T3 = re.compile(
    r"\b(?:I|we)\s+(?:still\s+|yet\s+|nevertheless\s+|nonetheless\s+)?"
    r"(?:did\s+not|would\s+not|"
    r"could\s+not\s+(?:abandon|betray|yield|flee|lie|deceive))\b",
    re.IGNORECASE,
)

# --- RESISTANCE CLASS ------------------------------------------------------
# T1: did-not / never constructions — the value held but without an active
#     named refusal.
_RES_T1 = re.compile(
    r"\b(?:"
    r"did\s+not\s+(?:yield|capitulate|abandon|flee|retreat|falter|waver)|"
    r"never\s+(?:yielded|capitulated|fled|retreated|faltered|wavered)"
    r")\b",
    re.IGNORECASE,
)

# T2: explicit refused/would-not for tactical/positional failure.
_RES_T2 = re.compile(
    r"\b(?:"
    r"refused?\s+to\s+(?:yield|capitulate|surrender|give\s+up|give\s+in|"
    r"back\s+down|stand\s+aside|flee|retreat)|"
    r"would\s+not\s+(?:yield|capitulate|surrender|give\s+up|give\s+in|"
    r"back\s+down|stand\s+aside|be\s+silenced)|"
    r"never\s+(?:surrendered|capitulated)"
    r")\b",
    re.IGNORECASE,
)

# T3: refused to betray / abandon / deceive — moral and relational stakes.
#     These involve harm to others or a direct violation of a value, not just
#     personal tactical retreat.
_RES_T3 = re.compile(
    r"\b(?:"
    r"refused?\s+to\s+(?:abandon|betray|lie|deceive)|"
    r"would\s+not\s+(?:abandon|betray|lie|deceive)|"
    r"did\s+not\s+(?:betray|abandon)|"
    r"never\s+(?:betrayed|abandoned)|"
    r"could\s+not\s+(?:abandon|betray|leave|forsake)\s+(?:them|him|her|the)"
    r")\b",
    re.IGNORECASE,
)

# --- STAKES CLASS ----------------------------------------------------------
# T1: abstract stakes — something of personal value is noted as being at
#     risk, but without specificity about the threat.
_STK_T1 = re.compile(
    r"\b(?:"
    r"(?:my|his|her|their|our)\s+(?:life|career|freedom|reputation|safety|"
    r"livelihood|position|future|everything)\s+(?:was\s+at\s+stake|"
    r"depended\s+on\s+it|hung\s+in\s+the\s+balance)"
    r")\b",
    re.IGNORECASE,
)

# T2: named external agent applying pressure — someone else is explicitly
#     issuing a threat or demand.
_STK_T2 = re.compile(
    r"\b(?:"
    r"(?:threatened|warned|ordered)\s+(?:me|him|her|them|us)\s+to"
    r")\b",
    re.IGNORECASE,
)

# T3: physical violence, death, imprisonment, or execution named — the cost
#     of resistance is irreversible or lethal.
_STK_T3 = re.compile(
    r"\b(?:"
    r"(?:threatened|faced|risked)\s+(?:death|imprisonment|exile|ruin|"
    r"execution|persecution)|"
    r"(?:threatened|warned)\s+(?:me|him|her|them|us)\s+with\s+(?:death|"
    r"imprisonment|exile|ruin|execution|persecution)|"
    r"at\s+(?:gunpoint|knifepoint|sword(?:point)?|threat\s+of\s+death)"
    r")\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Class-score helper
# ---------------------------------------------------------------------------

def _class_score(
    text: str,
    t1: re.Pattern,
    t2: re.Pattern,
    t3: re.Pattern,
) -> float:
    """
    Return the highest tier score that matches *text* for one adversity class.
    Checks T3 first; returns 0.0 if nothing matches.
    """
    if t3.search(text):
        return _T3
    if t2.search(text):
        return _T2
    if t1.search(text):
        return _T1
    return 0.0


def structural_score(text: str) -> float:
    """
    Compute a structural adversity score for a passage.

    Returns a continuous value in [0.0, 1.0] — the mean of four independent
    class scores:

        adversity  context (T1: despite | T2: at great cost | T3: risking everything)
        agency     first-person choice (T1: continued | T2: refused | T3: would not betray)
        resistance refusal of failure (T1: did not yield | T2: refused to surrender | T3: refused to betray)
        stakes     counterfactual cost (T1: life at stake | T2: threatened to | T3: at gunpoint)

    Each class returns its highest-tier match:
        T1 = 0.30   mild / concessive
        T2 = 0.65   real cost / active pressure
        T3 = 1.00   existential / irreversible

    Notable calibration points:
        All four classes absent              → 0.0
        One class fires at T1 only          → 0.075
        One class fires at T3 only          → 0.25
        All four classes fire at T1         → 0.30
        All four classes fire at T2         → 0.65
        All four classes fire at T3         → 1.00

    Higher score = stronger structural evidence that the passage describes
    a value being demonstrated under real adversity.  This is value-agnostic.
    """
    try:
        adv = _class_score(text, _ADV_T1, _ADV_T2, _ADV_T3)
        agc = _class_score(text, _AGC_T1, _AGC_T2, _AGC_T3)
        res = _class_score(text, _RES_T1, _RES_T2, _RES_T3)
        stk = _class_score(text, _STK_T1, _STK_T2, _STK_T3)
        return round((adv + agc + res + stk) / 4.0, 4)
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Sub-layer B — Zero-shot DeBERTa
# ---------------------------------------------------------------------------

_ZEROSHOT_MODEL = "MoritzLaurer/deberta-v3-large-zeroshot-v2.0"

# Per-value hypothesis strings.
# Formulated as entailment hypotheses: the model tests whether the passage
# ENTAILS this claim. These are deliberately independent of Ethos vocabulary.
VALUE_HYPOTHESES: Dict[str, str] = {
    "integrity":      "The person acts with honesty and refuses to deceive despite pressure.",
    "courage":        "The person faces danger or fear and does not retreat.",
    "compassion":     "The person genuinely cares about another's suffering and responds.",
    "commitment":     "The person keeps a promise or obligation despite difficulty.",
    "patience":       "The person endures delay or frustration without acting impulsively.",
    "responsibility": "The person accepts blame or ownership for consequences of their actions.",
    "fairness":       "The person treats others equitably and resists bias or favoritism.",
    "gratitude":      "The person acknowledges a debt of appreciation for what they have received.",
    "curiosity":      "The person is driven to understand and investigate despite uncertainty.",
    "resilience":     "The person recovers from adversity and continues despite hardship.",
    "love":           "The person demonstrates deep, lasting care or devotion toward another.",
    "growth":         "The person changes their beliefs or behavior based on learning.",
    "independence":   "The person makes their own choice despite external pressure to conform.",
    "loyalty":        "The person remains faithful to another when doing so is costly.",
    "humility":       "The person acknowledges their own error or limitation without defensiveness.",
}

_ZS_LOCK:      threading.Lock          = threading.Lock()
_ZS_PIPELINE:  Optional[object]        = None
_ZS_AVAILABLE: Optional[bool]          = None


def _get_zeroshot_pipeline() -> Optional[object]:
    global _ZS_PIPELINE, _ZS_AVAILABLE
    if _ZS_AVAILABLE is False:
        return None
    if _ZS_PIPELINE is not None:
        return _ZS_PIPELINE
    with _ZS_LOCK:
        if _ZS_PIPELINE is not None:
            return _ZS_PIPELINE
        try:
            from transformers import pipeline  # type: ignore
            _log.info("Loading zero-shot classifier: %s (this may take a minute)", _ZEROSHOT_MODEL)
            pipe = pipeline(
                "zero-shot-classification",
                model=_ZEROSHOT_MODEL,
                device=-1,          # CPU
                multi_label=True,   # independent sigmoid per label (not softmax)
            )
            _ZS_PIPELINE  = pipe
            _ZS_AVAILABLE = True
            _log.info("Zero-shot classifier loaded.")
        except ImportError:
            _log.warning("transformers not installed — zero-shot layer disabled")
            _ZS_AVAILABLE = False
        except Exception:
            _log.warning("Zero-shot classifier load failed — sub-layer B disabled",
                         exc_info=True)
            _ZS_AVAILABLE = False
    return _ZS_PIPELINE


def zeroshot_scores(
    text: str,
    values: List[str],
    threshold: float = 0.35,
) -> List[Tuple[str, float]]:
    """
    Score a passage against zero-shot hypotheses for the given values.

    Args:
        text:      The passage text.
        values:    Value names to test (subset of VALUE_HYPOTHESES keys).
        threshold: Minimum entailment probability to include.

    Returns:
        [(value_name, entailment_probability), ...] sorted by score DESC.
        Empty list on error or if model unavailable.
    """
    pipe = _get_zeroshot_pipeline()
    if pipe is None:
        return []
    if not text or not values:
        return []

    # Build candidate labels — the hypothesis strings for requested values
    candidate_labels = [VALUE_HYPOTHESES[v] for v in values if v in VALUE_HYPOTHESES]
    label_to_value   = {VALUE_HYPOTHESES[v]: v for v in values if v in VALUE_HYPOTHESES}

    if not candidate_labels:
        return []

    try:
        # Truncate very long passages to avoid OOM on CPU
        passage = text[:1024] if len(text) > 1024 else text

        result = pipe(passage, candidate_labels, multi_label=True)
        scores: List[Tuple[str, float]] = []

        for label, score in zip(result["labels"], result["scores"]):
            value_name = label_to_value.get(label)
            if value_name and score >= threshold:
                scores.append((value_name, round(float(score), 4)))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores

    except Exception:
        _log.debug("zeroshot_scores failed", exc_info=True)
        return []


# ---------------------------------------------------------------------------
# Combined Layer 3 signal builder
# ---------------------------------------------------------------------------

def layer3_signals(
    text: str,
    significance: float,
    doc_type: str,
    candidate_values: List[str],
    zeroshot_threshold: float = 0.35,
    zeroshot_standalone_threshold: float = 0.70,
    zeroshot_enabled: bool = True,
) -> Tuple[float, List[Dict], Dict[str, float]]:
    """
    Run Layer 3 on a passage and return:
      (structural_score, new_zeroshot_signals, zs_agreement_scores)

    structural_score:
        Continuous float in [0.0, 1.0] — mean of four tiered class scores.
        Caller uses this to boost resistance/confidence on all L1/L2 signals.

    new_zeroshot_signals:
        Signal dicts for values NOT in candidate_values that exceeded
        zeroshot_standalone_threshold.  These are new standalone detections.
        source = "zeroshot"

    zs_agreement_scores:
        {value_name: entailment_probability} for values that ARE in
        candidate_values and exceeded zeroshot_threshold.  Caller uses
        these to apply agreement confidence boosts.

    candidate_values:
        Values already detected by Layers 1 and 2.
    """
    struct_score = structural_score(text)

    new_sigs:    List[Dict]       = []
    agreement:   Dict[str, float] = {}

    if not zeroshot_enabled:
        return struct_score, new_sigs, agreement

    # Single zero-shot call at the lower threshold to catch both agreement
    # cases and potential standalone detections.
    all_values = list(VALUE_HYPOTHESES.keys())
    zs_hits    = zeroshot_scores(text, all_values, threshold=zeroshot_threshold)

    candidate_set = set(candidate_values)
    excerpt = text[:150] if len(text) <= 150 else text[:147] + "..."

    for value_name, score in zs_hits:
        if value_name in candidate_set:
            # Agreement with L1/L2 detection — record for caller's boost
            agreement[value_name] = score
        else:
            # Standalone detection — requires higher confidence bar
            if score >= zeroshot_standalone_threshold:
                new_sigs.append({
                    "value_name":               value_name,
                    "text_excerpt":             excerpt,
                    "significance":             significance,
                    "disambiguation_confidence": round(score, 4),
                    "source":                   "zeroshot",
                })

    return struct_score, new_sigs, agreement


# ---------------------------------------------------------------------------
# Availability check
# ---------------------------------------------------------------------------

def is_zeroshot_available() -> bool:
    """True if the zero-shot model is loaded and ready."""
    return _get_zeroshot_pipeline() is not None
