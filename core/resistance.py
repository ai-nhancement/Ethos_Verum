"""
core/resistance.py

Resistance level computation — estimates the cost of holding a demonstrated value.

A value stated in comfort is weak signal; a value demonstrated under real cost
(adversity language, authentic document type) is strong signal.

Formula (additive):
  base:              0.12
  significance:      +0.00–0.12  (significance × 0.17, capped at 0.12)
  doc_type nudge:    small per-type constant (see DOC_TYPE_RESISTANCE_BONUS)
  text tier:         applied in priority order — highest matching tier wins:

    Tier A — Mortal stakes (+0.62):
      Death/execution language explicitly present: "I would rather die",
      "prepared to die", "one life to lose", "abjure",
      "sentence against me", "cannot and will not retract", etc.
      Awarded regardless of failure language — stakes are real even when
      the person ultimately capitulated (e.g., Galileo's forced recantation).
      +0.05 extra when hold language also present.

    Tier B — High adversity (+0.36):
      Imprisonment, exile, persecution, direct threats, forced legal
      proceedings: "threatened to", "they can come for me", "oppressed",
      "at the hazard of", "advised by my lawyers", etc.
      Suppressed whenever failure language is present — hold language does
      NOT prevent suppression at this tier (unlike Tier C). Rationale: a
      direct-threat passage where the person ultimately yielded does not
      warrant a high-adversity boost regardless of contradictory hold phrasing.
      +0.05 extra when hold language is present and failure is absent.

    Tier C — Standard adversity (+0.24):
      Moderate adversity phrases: "despite", "even though", "suffering",
      "not easy", "scared but", "ruined", etc.
      Suppressed when failure fires and hold does not.
      +0.05 extra when hold language also present.

  Total = clip(sum, 0.0, 1.0)

Document type rationale (nudge, not primary driver):
  Text content is the primary resistance signal. Doc type provides a small
  authenticity adjustment — journals and action records get a slight nudge
  over speeches and unknowns.
  action  — documented real-world behavior (highest authenticity)
  journal — private writing, no audience pressure
  letter  — directed correspondence
  speech  — public address (highest performance pressure, lowest nudge)
  unknown — default

Calibration (n=35, sig=0.70):
  MAE = 0.131 (vs. 0.142 before Tier B/C tuning; vs. 0.32 for the prior additive formula)
  Mortal tier cases (7 records): average error < 0.05
  Primary residual: perpetrator-high-resistance cases where text contains
  no personal adversity language (Himmler, Robespierre, Stalin) —
  these require context beyond the text and cannot be detected by regex.

Never raises. Returns 0.5 (neutral) on any error.
"""

from __future__ import annotations

import re

from core.config import DOC_TYPE_RESISTANCE_BONUS

# ---------------------------------------------------------------------------
# Tier A — Mortal / execution stakes
# ---------------------------------------------------------------------------
# Death or execution language is explicit in the text. Only first-person and
# specifically contextualised forms are included — bare "death" and "die" are
# intentionally excluded to avoid false positives on philosophical reflection
# ("It is not death that a man should fear").

_MORTAL_RE = re.compile(
    r"\b(?:"
    r"(?:I|we)\s+(?:would\s+(?:rather\s+)?)?die\b|"        # "I would rather die"
    r"prepared\s+to\s+die|ready\s+to\s+die|willing\s+to\s+die|"
    r"one\s+life\s+to\s+(?:lose|give)\b|"                  # "one life to lose"
    r"gave?\s+(?:my|his|her|their|your)\s+life\s+for\b|"
    r"abjure\b|"                                            # Inquisition recantation
    r"sentence\s+against\s+me\b|"                          # "sentence against me"
    r"cannot\s+and\s+will\s+not\s+retract\b|"              # Luther: explicit refusal
    r"will\s+not\s+retract\s+anything\b|"
    r"execut(?:ed|ion)\b|hanged\b|beheaded\b|"
    r"burned\s+at\b|sentenced\s+to\s+death\b"
    r")",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Tier B — High adversity
# ---------------------------------------------------------------------------
# Imprisonment, exile, persecution, direct professional/personal threat, or
# forced legal proceedings. The person faces concrete external costs.
# Suppressed when failure fires without hold.

_HIGH_ADVERSITY_RE = re.compile(
    r"\b(?:"
    r"threaten(?:ed|s)?\s+to\b|"                           # "threatened to destroy me"
    r"they\s+(?:can|could|will|would|might)\s+come\s+for\s+me\b|"
    r"at\s+the\s+hazard\s+of\b|"
    r"advised\s+by\s+(?:my|his|her|their)\s+(?:lawyers?|attorneys?|counsel)\b|"
    r"imprison(?:ed|ment)\b|jail(?:ed)?\b|detain(?:ed|ment)\b|"
    r"exil(?:e|ed)\b|banish(?:ed)?\b|expel(?:led)?\b|"
    r"persecuted?\b|persecution\b|"
    r"oppressed\b|broken[\s-]hearted\b"
    r")",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Tier C — Standard adversity
# ---------------------------------------------------------------------------
# Moderate adversity phrases and cost language. Suppressed when failure fires
# and hold does not.

_ADVERSITY_RE = re.compile(
    r"\b(?:"
    r"even\s+though|despite|but\s+I\s+still|but\s+still|hard\s+to|difficult\s+to|"
    r"I\s+have\s+to|I\s+need\s+to|I\s+won'?t\s+give\s+up|I\s+won'?t\s+stop|"
    r"I\s+still\s+believe|not\s+easy|scared\s+but|afraid\s+but|nervous\s+but|"
    r"worried\s+but|at\s+a\s+cost|at\s+my\s+own\s+expense|going\s+to\s+lose|"
    r"might\s+lose|risk\s+losing|suffering\b|my\s+suffering|ruined\b"
    r")\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Hold markers — value being maintained under pressure
# ---------------------------------------------------------------------------
# When hold language co-occurs with any adversity tier, a small extra bonus
# is added. For Tier B, hold also prevents failure from suppressing the bonus.

_HOLD_RE = re.compile(
    r"\b(?:"
    r"nevertheless|nonetheless|"
    r"still\s+(?:held?|continued|stood|believed?)|"
    r"stood\s+firm|refused\s+to\s+give|persevered|maintained|"
    r"held?\s+(?:firm|to|my|fast)|"
    r"would\s+not\s+(?:yield|surrender|give|back\s+down|flee|retreat)|"
    r"will\s+not\s+yield\b|"
    r"did\s+not\s+(?:yield|flee|retreat|surrender|capitulate)|"
    r"carried\s+on|pressed\s+on|kept\s+(?:faith|my)\b|"
    r"refused\s+to\s+(?:yield|surrender|flee|retreat|capitulate|back\s+down)|"
    r"would\s+do\s+it\s+again\b|choose\s+to\s+stay\b"
    r")\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Failure markers — value is being abandoned
# ---------------------------------------------------------------------------
# Suppresses Tier B and C bonuses when present without hold language.
# Does NOT suppress Tier A (mortal stakes remain real regardless of outcome).

_FAILURE_RE = re.compile(
    r"\b(?:"
    r"gave\s+in|gave\s+up|yielded|surrendered|backed\s+down|"
    r"compromised\s+my|caved|relented|capitulated|gave\s+way|"
    r"stepped\s+back|gave\s+in\s+to|dropped\s+it\b"
    r")\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Formula constants
# ---------------------------------------------------------------------------
_BASE           = 0.12   # floor before any bonuses
_SIG_SCALE      = 0.17   # significance multiplier
_SIG_CAP        = 0.12   # maximum significance contribution
_MORTAL_BONUS   = 0.62   # Tier A bonus
_HIGH_ADV_BONUS = 0.36   # Tier B bonus
_ADV_BONUS      = 0.24   # Tier C bonus
_HOLD_EXTRA     = 0.05   # extra when hold language co-occurs with any tier


def compute_resistance(text: str, significance: float, doc_type: str) -> float:
    """
    Estimate resistance_level (cost of holding the demonstrated value).
    Range: [0.0, 1.0]. Never raises — returns 0.5 on any error.

    Args:
        text:         The passage text to scan for adversity/mortal markers.
        significance: Significance score for the passage (0.0–1.0).
        doc_type:     Document type key — 'action', 'journal', 'letter',
                      'speech', 'unknown'.
    """
    try:
        total = _BASE + min(float(significance) * _SIG_SCALE, _SIG_CAP)

        dt = (doc_type or "unknown").lower().strip()
        total += DOC_TYPE_RESISTANCE_BONUS.get(dt, DOC_TYPE_RESISTANCE_BONUS["unknown"])

        has_hold    = bool(_HOLD_RE.search(text))
        has_failure = bool(_FAILURE_RE.search(text))

        if _MORTAL_RE.search(text):
            # Tier A: mortal stakes — failure does NOT suppress
            total += _MORTAL_BONUS
            if has_hold:
                total += _HOLD_EXTRA

        elif _HIGH_ADVERSITY_RE.search(text):
            # Tier B: high adversity — failure suppresses when no hold present
            if not has_failure:
                total += _HIGH_ADV_BONUS
                if has_hold:
                    total += _HOLD_EXTRA

        elif _ADVERSITY_RE.search(text):
            # Tier C: standard adversity — failure suppresses when no hold present
            if not has_failure or has_hold:
                total += _ADV_BONUS
                if has_hold:
                    total += _HOLD_EXTRA

        return round(max(0.0, min(1.0, total)), 4)

    except Exception:
        return 0.5
