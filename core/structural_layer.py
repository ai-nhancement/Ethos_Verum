"""
core/structural_layer.py

Layer 3 signal extraction for Ethos.

Two independent sub-layers:

  A. Structural patterns (pure regex, zero deps)
     Detect the context in which values appear — agency, adversity, and
     resistance markers. A passage where value keywords appear alongside
     adversity vocabulary ("despite", "at great cost", "refused to yield")
     is stronger evidence than the same keywords in neutral context.

     Returns a structural_score in [0.0, 1.0] for each value candidate
     the passage is tested against.

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
# Sub-layer A — Structural patterns
# ---------------------------------------------------------------------------

# Adversity context markers — close presence of these near a value keyword
# suggests the value is being demonstrated under real pressure.
_ADVERSITY_RE = re.compile(
    r"\b(?:"
    # Concessive conjunctions / prepositions
    r"in\s+spite\s+of|although|even\s+though|notwithstanding|despite|"
    # Cost and risk vocabulary
    r"at\s+(?:great\s+)?(?:personal\s+)?(?:risk|cost|price|peril)|"
    r"risking\s+(?:everything|my\s+life|his\s+life|her\s+life|their\s+lives|"
    r"my\s+career|my\s+reputation|his\s+career|her\s+career)|"
    r"knowing\s+(?:the\s+)?(?:cost|risk|danger|consequences?)|"
    r"at\s+the\s+expense\s+of|"
    # Opposition and pressure
    r"against\s+(?:all\s+)?(?:opposition|resistance|advice|orders|the\s+odds)|"
    r"under\s+(?:pressure|threat|fire|attack|duress|siege|interrogation)|"
    r"in\s+the\s+face\s+of|"
    # Sacrifice vocabulary
    r"sacrific(?:ing|ed)\s+(?:my|his|her|their|everything)|"
    r"gave\s+(?:up\s+)?(?:everything|my\s+life|his\s+life|her\s+life)|"
    r"laid\s+down\s+(?:his|her|their|my)\s+life"
    r")\b",
    re.IGNORECASE,
)

# Agency markers — first-person + active verb in adversity context
# Pattern: (I|we) [optionally: still/yet] [action_verb]
_AGENCY_RE = re.compile(
    r"\b(?:I|we)\s+(?:still\s+|yet\s+|nevertheless\s+|nonetheless\s+)?"
    r"(?:refused?|chose|stood|held|pressed\s+on|"
    r"carried\s+on|kept\s+going|continued|persisted?|remained?|stayed?|"
    r"marched\s+on|rose|rebuilt?|recovered?|endured?|persevered?|"
    r"did\s+not|would\s+not|could\s+not\s+(?:abandon|betray|yield|flee|lie|deceive))\b",
    re.IGNORECASE,
)

# Resistance-to-failure markers — explicit refusal of value-failure actions
_RESISTANCE_RE = re.compile(
    r"\b(?:"
    r"refused?\s+to\s+(?:yield|capitulate|surrender|abandon|betray|lie|deceive|"
    r"flee|retreat|give\s+up|give\s+in|back\s+down|stand\s+aside)|"
    r"would\s+not\s+(?:yield|capitulate|surrender|abandon|betray|lie|deceive|"
    r"flee|retreat|give\s+up|give\s+in|back\s+down|stand\s+aside|be\s+silenced)|"
    r"did\s+not\s+(?:yield|capitulate|abandon|betray|flee|retreat|falter|waver)|"
    r"never\s+(?:yielded|capitulated|abandoned|betrayed|fled|retreated|faltered|wavered)|"
    r"could\s+not\s+(?:abandon|betray|leave|forsake)\s+(?:them|him|her|the)"
    r")\b",
    re.IGNORECASE,
)

# Counterfactual pressure markers — explicit description of what was at stake
_STAKES_RE = re.compile(
    r"\b(?:"
    r"(?:my|his|her|their|our)\s+(?:life|career|freedom|reputation|safety|"
    r"livelihood|position|future|everything)\s+(?:was\s+at\s+stake|"
    r"depended\s+on\s+it|hung\s+in\s+the\s+balance)|"
    r"(?:threatened|warned|ordered)\s+(?:me|him|her|them|us)\s+to|"
    r"(?:threatened|faced|risked)\s+(?:death|imprisonment|exile|ruin|"
    r"execution|persecution)|"
    r"at\s+(?:gunpoint|knifepoint|sword(?:point)?|threat\s+of\s+death)"
    r")\b",
    re.IGNORECASE,
)


def structural_score(text: str) -> float:
    """
    Compute a structural adversity score for a passage.

    Returns a value in [0.0, 1.0]:
      0.0  — no structural signals detected
      0.3  — one signal type present
      0.5  — two signal types present
      0.8  — three signal types present
      1.0  — all four signal types present (max)

    Higher score = stronger structural evidence the passage describes
    a value being demonstrated under real adversity.

    This is value-agnostic — it characterizes the CONTEXT, not the value.
    Used to boost resistance scoring and disambiguation confidence.
    """
    try:
        has_adversity  = bool(_ADVERSITY_RE.search(text))
        has_agency     = bool(_AGENCY_RE.search(text))
        has_resistance = bool(_RESISTANCE_RE.search(text))
        has_stakes     = bool(_STAKES_RE.search(text))

        hits = sum([has_adversity, has_agency, has_resistance, has_stakes])
        score_map = {0: 0.0, 1: 0.3, 2: 0.5, 3: 0.8, 4: 1.0}
        return score_map[hits]
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
        float in [0.0, 1.0] — value-agnostic adversity context strength.
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
