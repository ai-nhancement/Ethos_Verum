"""
core/polarity_layer.py

Value polarity detection — determines whether a value signal represents
a constructive (+1), destructive (-1), or ambiguous (0) expression.

Resistance and polarity are independent axes:
  - Resistance measures the cost/pressure under which a value was held.
  - Polarity measures the moral direction of the value expression.

A dictator refusing to stop atrocities under military defeat:
  → Resistance: HIGH (real cost, real pressure)
  → Polarity:   NEGATIVE (value directed toward harm)

A civil rights leader refusing to denounce protesters at gunpoint:
  → Resistance: HIGH (same measurement)
  → Polarity:   POSITIVE (value directed toward dignity/rights)

Detection architecture (three tiers, applied in order):

  Tier 1 — Target lexicon proximity (deterministic):
    Find positive or negative target words within a window around the
    value keyword match position. Check for inversion verbs ("fight
    against", "resist") that reverse the direction. Fast and transparent.

  Tier 2 — MFT lexicon vice/virtue signals (deterministic):
    When the passage already fired a moral foundation vice signal
    (harm, betrayal, subversion, degradation, oppression, cheating)
    for the same value, push polarity toward -1.
    Vice signal confidence is weighted against the Tier 1 result.

  Tier 3 — Zero-shot classifier (model-based, optional):
    When polarity is still ambiguous after Tiers 1 and 2, ask the
    zero-shot classifier whether the value is applied destructively.
    Hypothesis: "The person applies {value} in a harmful or destructive way."
    Only fires when zeroshot_enabled=True (core.config) and the
    sentence-transformers + DeBERTa pipeline is available.

Output: (polarity: int, confidence: float)
  polarity:   +1 constructive, -1 destructive, 0 ambiguous
  confidence: [0.0, 1.0] — how certain the detection is

Never raises. Returns (0, 0.0) on any error.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Target word lexicons
# ---------------------------------------------------------------------------
# These are stem/substring matches (lowercase) applied within a window around
# the value keyword. Order within each set does not matter.

# Words/stems indicating the value is directed toward constructive ends
POSITIVE_TARGET_WORDS: frozenset = frozenset({
    # Rights, dignity, freedom
    "right", "rights", "freedom", "liberty", "dignity", "equal", "equality",
    "justice", "just cause", "fair", "fairness", "truth", "honest",
    # Vulnerable people as objects of protection
    "oppressed", "vulnerable", "innocent", "victim", "victims", "the poor",
    "the weak", "the suffering", "the displaced", "refugee", "civilian",
    "prisoner", "enslaved", "marginalized",
    # Constructive social goals
    "peace", "humanity", "welfare", "well-being", "wellbeing", "conscience",
    "transparency", "accountability", "democracy", "human dignity",
    "reconciliation", "healing", "mercy", "compassion", "grace",
    # Knowledge and understanding
    "knowledge", "truth", "science", "evidence", "understanding", "reason",
    # Life and protection
    "life", "survival", "protection", "safety of",
})

# Words/stems indicating the value is directed toward destructive ends
NEGATIVE_TARGET_WORDS: frozenset = frozenset({
    # Systems of harm
    "tyranny", "tyrant", "oppression", "persecution", "discrimination",
    "cruelty", "atrocit", "massacre", "genocide", "purge", "extermination",
    "enslavement", "subjugation", "domination", "conquest",
    # Harmful actions
    "exploitation", "torture", "suppression", "dehumanization", "dehumaniz",
    "violence against", "brutality", "terror", "ethnic cleansing",
    # Dishonesty and manipulation
    "lies", "deceit", "deception", "propaganda", "false", "fabricat",
    "corruption", "fraud",
    # Harmful ideologies/regimes (qualified)
    "oppressive regime", "authoritarian", "totalitarian", "apartheid",
    "inquisition", "pogrom",
    # Harm directed at people
    "hatred", "hatred of", "contempt for", "hatred toward",
})

# Inversion patterns: when these appear BETWEEN the value keyword position
# and a target word, the polarity relationship is reversed.
# "He fought against tyranny" → inversion + negative target → +1 (positive)
# "He resisted justice" → inversion + positive target → -1 (destructive)
_INVERSION_RE = re.compile(
    r"\b(?:"
    r"fight(?:s|ing)?\s+against|fought\s+against|"
    r"resist(?:s|ed|ing)?|"
    r"oppos(?:e|es|ed|ing)|"
    r"stand(?:s|ing)?\s+(?:against|up\s+to)|stood\s+(?:against|up\s+to)|"
    r"defying|defied|defi(?:es?)|"
    r"refus(?:ed|es|ing)\s+to\s+(?:serve|enable|allow|support|advance|uphold)|"
    r"against\b"
    r")",
    re.IGNORECASE,
)

# Window size (characters) around the value match to search for target words
_TARGET_WINDOW = 120

# Tier 1 confidence levels
_CONF_DIRECT        = 0.80  # target word found, no inversion
_CONF_INVERTED      = 0.68  # target word found with inversion (more ambiguous)
_CONF_VICE_ONLY     = 0.60  # no target word, only MFT vice signal
_CONF_TIER3_BASE    = 0.55  # zero-shot tier 3 base confidence


# ---------------------------------------------------------------------------
# Tier 1: target lexicon proximity
# ---------------------------------------------------------------------------

def _tier1_target(
    text: str,
    match_idx: int,
) -> Tuple[int, float]:
    """
    Scan the window around match_idx for positive/negative target words.
    Returns (polarity, confidence).
    """
    start = max(0, match_idx - _TARGET_WINDOW)
    end   = min(len(text), match_idx + _TARGET_WINDOW)
    window = text[start:end].lower()

    pos_hits: List[str] = [w for w in POSITIVE_TARGET_WORDS if w in window]
    neg_hits: List[str] = [w for w in NEGATIVE_TARGET_WORDS if w in window]

    if not pos_hits and not neg_hits:
        return 0, 0.0

    # Check for inversion pattern in the same window
    inverted = bool(_INVERSION_RE.search(window))

    if pos_hits and not neg_hits:
        polarity = -1 if inverted else +1
        return polarity, _CONF_INVERTED if inverted else _CONF_DIRECT

    if neg_hits and not pos_hits:
        polarity = +1 if inverted else -1
        return polarity, _CONF_INVERTED if inverted else _CONF_DIRECT

    # Both present — competing signals. Inversion complicates further.
    # Resolve by count: majority wins, confidence reduced.
    if len(pos_hits) > len(neg_hits):
        polarity = -1 if inverted else +1
    elif len(neg_hits) > len(pos_hits):
        polarity = +1 if inverted else -1
    else:
        return 0, 0.0  # tied — genuinely ambiguous

    return polarity, round(_CONF_DIRECT * 0.75, 4)  # reduced confidence for contested


# ---------------------------------------------------------------------------
# Tier 2: MFT lexicon vice/virtue integration
# ---------------------------------------------------------------------------

def _tier2_lexicon(
    lexicon_vice_score: float,
    lexicon_virtue_score: float,
    tier1_polarity: int,
    tier1_conf: float,
) -> Tuple[int, float]:
    """
    Incorporate MFT vice/virtue lexicon signals.

    If Tier 1 found no clear signal (polarity == 0) and a vice signal is
    present, move toward -1 with _CONF_VICE_ONLY confidence.

    If Tier 1 already set a polarity, use the vice/virtue signal to
    confirm or discount:
      - Vice confirms -1 or contradicts +1
      - Virtue confirms +1 or contradicts -1
    """
    if lexicon_vice_score <= 0.0 and lexicon_virtue_score <= 0.0:
        return tier1_polarity, tier1_conf

    if tier1_polarity == 0:
        # Tier 1 found nothing — use lexicon alone
        if lexicon_vice_score > lexicon_virtue_score and lexicon_vice_score >= 0.40:
            return -1, min(lexicon_vice_score * 0.75, _CONF_VICE_ONLY)
        if lexicon_virtue_score > lexicon_vice_score and lexicon_virtue_score >= 0.40:
            return +1, min(lexicon_virtue_score * 0.75, _CONF_VICE_ONLY)
        return 0, 0.0

    # Tier 1 has a result — adjust confidence
    if tier1_polarity == -1 and lexicon_vice_score > 0.0:
        # Confirmation: vice signal agrees with negative polarity
        boosted = min(1.0, tier1_conf + lexicon_vice_score * 0.15)
        return -1, round(boosted, 4)

    if tier1_polarity == +1 and lexicon_virtue_score > 0.0:
        # Confirmation: virtue signal agrees with positive polarity
        boosted = min(1.0, tier1_conf + lexicon_virtue_score * 0.15)
        return +1, round(boosted, 4)

    if tier1_polarity == +1 and lexicon_vice_score > tier1_conf:
        # Contradiction: strong vice signal overrides weak positive tier1
        return -1, round(lexicon_vice_score * 0.65, 4)

    if tier1_polarity == -1 and lexicon_virtue_score > tier1_conf:
        # Contradiction: strong virtue signal overrides weak negative tier1
        return +1, round(lexicon_virtue_score * 0.65, 4)

    return tier1_polarity, tier1_conf


# ---------------------------------------------------------------------------
# Tier 3: zero-shot classifier (optional)
# ---------------------------------------------------------------------------

def _tier3_zeroshot(
    text: str,
    value_name: str,
    cfg,
) -> Tuple[int, float]:
    """
    Ask the zero-shot DeBERTa classifier whether the value is applied
    destructively. Only fires when cfg.zeroshot_enabled is True.

    Hypothesis: "The person applies {value} in a harmful or destructive way."
    """
    try:
        if not getattr(cfg, "zeroshot_enabled", False):
            return 0, 0.0
        if not getattr(cfg, "polarity_zeroshot_enabled", True):
            return 0, 0.0

        from core.zero_shot_layer import zero_shot_entailment, is_available
        if not is_available():
            return 0, 0.0

        hypothesis = f"The person applies {value_name} in a harmful or destructive way."
        score = zero_shot_entailment(text, hypothesis)
        threshold = getattr(cfg, "polarity_zeroshot_threshold", 0.55)
        if score >= threshold:
            return -1, round(score * _CONF_TIER3_BASE / threshold, 4)

        # If entailment is very low, weakly affirm positive
        if score < 0.25:
            return +1, round((0.25 - score) * 0.5, 4)

        return 0, 0.0
    except Exception:
        return 0, 0.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_polarity(
    text: str,
    match_idx: int,
    value_name: str,
    lexicon_vice_score: float = 0.0,
    lexicon_virtue_score: float = 0.0,
    cfg=None,
) -> Tuple[int, float]:
    """
    Determine value polarity for a single extracted signal.

    Args:
        text:                 Full passage text.
        match_idx:            Character index of the value keyword match.
        value_name:           The Ethos value name (e.g., "loyalty").
        lexicon_vice_score:   MFD2.0/MoralStrength vice confidence for this passage/value.
        lexicon_virtue_score: MFD2.0/MoralStrength virtue confidence.
        cfg:                  Config object (optional — needed for Tier 3).

    Returns:
        (polarity, confidence):
          polarity   — +1 constructive, -1 destructive, 0 ambiguous
          confidence — [0.0, 1.0]

    Never raises. Returns (0, 0.0) on any error.
    """
    try:
        # Tier 1: target word proximity
        p1, c1 = _tier1_target(text, match_idx)

        # Tier 2: MFT lexicon integration
        p2, c2 = _tier2_lexicon(lexicon_vice_score, lexicon_virtue_score, p1, c1)

        # Tier 3: zero-shot, only if still ambiguous
        if p2 == 0 and cfg is not None:
            p3, c3 = _tier3_zeroshot(text, value_name, cfg)
            if p3 != 0:
                return p3, c3

        return p2, c2

    except Exception:
        return 0, 0.0


def polarity_label(polarity: int) -> str:
    """Human-readable polarity label."""
    return {1: "constructive", -1: "destructive", 0: "ambiguous"}.get(polarity, "ambiguous")
