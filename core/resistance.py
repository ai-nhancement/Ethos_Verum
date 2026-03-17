"""
core/resistance.py

Resistance level computation — estimates the cost of holding a demonstrated value.

A value stated in comfort is weak signal; a value demonstrated under real cost
(adversity language, authentic document type) is strong signal.

Formula (additive):
  base:              0.25
  significance:      +0.00–0.30  (significance * 0.40, capped at 0.30)
  doc_type bonus:    imported from core.config.DOC_TYPE_RESISTANCE_BONUS
  text markers:      +0.20  if adversity phrases detected AND the passage
                     does not contain failure-without-hold evidence.
                     (Prevents "gave in despite my principles" from receiving
                     a resistance boost — adversity language must co-occur with
                     holding language, or absence of failure language.)
  Total = clip(sum, 0.0, 1.0)

Document type rationale:
  action  — documented real-world behavior under stakes (highest authenticity)
  journal — private writing, no audience pressure
  letter  — directed correspondence
  speech  — public address (highest performance pressure, lowest authenticity bonus)
  unknown — default

Never raises. Returns 0.5 (neutral) on any error.
"""

from __future__ import annotations

import re

from core.config import DOC_TYPE_RESISTANCE_BONUS

# Adversity context markers — passage describes a costly situation.
_ADVERSITY_RE = re.compile(
    r"\b(?:"
    r"even\s+though|despite|but\s+I\s+still|but\s+still|hard\s+to|difficult\s+to|"
    r"I\s+have\s+to|I\s+need\s+to|I\s+won'?t\s+give\s+up|I\s+won'?t\s+stop|"
    r"I\s+still\s+believe|not\s+easy|scared\s+but|afraid\s+but|nervous\s+but|"
    r"worried\s+but|at\s+a\s+cost|at\s+my\s+own\s+expense|going\s+to\s+lose|"
    r"might\s+lose|risk\s+losing"
    r")\b",
    re.IGNORECASE,
)

# Hold markers — passage describes the value being maintained.
# Text bonus is awarded when adversity AND hold are both present,
# or when adversity is present and no failure markers are found.
_HOLD_RE = re.compile(
    r"\b(?:"
    r"despite|even\s+though|nevertheless|nonetheless|"
    r"still\s+(?:held?|continued|stood|believed?)|"
    r"stood\s+firm|refused\s+to\s+give|persevered|maintained|"
    r"held?\s+(?:firm|to|my|fast)|"
    r"would\s+not\s+(?:yield|surrender|give|back\s+down|flee|retreat)|"
    r"did\s+not\s+(?:yield|flee|retreat|surrender|capitulate)|"
    r"carried\s+on|pressed\s+on|kept\s+(?:going|faith|my)|"
    r"refused\s+to\s+(?:yield|surrender|flee|retreat|capitulate|back\s+down)"
    r")\b",
    re.IGNORECASE,
)

# Failure markers — passage describes the value being abandoned.
# If these appear WITHOUT hold markers, the text bonus is withheld
# to prevent boosting resistance on passages where the value failed.
_FAILURE_RE = re.compile(
    r"\b(?:"
    r"gave\s+in|gave\s+up|yielded|surrendered|backed\s+down|"
    r"compromised\s+my|caved|relented|capitulated|gave\s+way|"
    r"stepped\s+back|gave\s+in\s+to"
    r")\b",
    re.IGNORECASE,
)

_BASE = 0.25
_SIG_CAP = 0.30
_TEXT_BONUS = 0.20


def compute_resistance(text: str, significance: float, doc_type: str) -> float:
    """
    Estimate resistance_level (cost of holding the demonstrated value).
    Range: [0.0, 1.0]. Never raises — returns 0.5 on any error.

    Args:
        text:        The passage text to scan for adversity markers.
        significance: Significance score for the passage (0.0–1.0).
        doc_type:    Document type key — 'action', 'journal', 'letter', 'speech', 'unknown'.
    """
    try:
        total = _BASE

        sig_bonus = min(float(significance) * 0.40, _SIG_CAP)
        total += sig_bonus

        dt = (doc_type or "unknown").lower().strip()
        total += DOC_TYPE_RESISTANCE_BONUS.get(dt, DOC_TYPE_RESISTANCE_BONUS["unknown"])

        if _ADVERSITY_RE.search(text):
            # Only award the text bonus when the passage is not describing failure
            # without hold evidence. Prevents adversity language in failure passages
            # ("gave in despite my principles") from inflating resistance.
            has_failure = bool(_FAILURE_RE.search(text))
            has_hold    = bool(_HOLD_RE.search(text))
            if not has_failure or has_hold:
                total += _TEXT_BONUS

        return round(max(0.0, min(1.0, total)), 4)

    except Exception:
        return 0.5
