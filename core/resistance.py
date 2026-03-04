"""
core/resistance.py

Resistance level computation — estimates the cost of holding a demonstrated value.

A value stated in comfort is weak signal; a value demonstrated under real cost
(adversity language, authentic document type) is strong signal.

Formula (additive):
  base:              0.25
  significance:      +0.00–0.30  (significance * 0.40, capped at 0.30)
  doc_type bonus:    action +0.40 / journal +0.35 / letter +0.30 / speech +0.10 / unknown +0.20
  text markers:      +0.20  if adversity phrases detected in text
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

_RESISTANCE_RE = re.compile(
    r"\b("
    r"even though|despite|but I still|but still|hard to|difficult to|"
    r"I have to|I need to|I won't give up|I won't stop|I still believe|"
    r"not easy|scared but|afraid but|nervous but|worried but|"
    r"at a cost|at my own expense|going to lose|might lose|risk losing"
    r")\b",
    re.IGNORECASE,
)

_BASE = 0.25
_SIG_CAP = 0.30
_TEXT_BONUS = 0.20

_DOCUMENT_TYPE_BONUS = {
    "action":  0.40,
    "journal": 0.35,
    "letter":  0.30,
    "speech":  0.10,
    "unknown": 0.20,
}


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
        total += _DOCUMENT_TYPE_BONUS.get(dt, _DOCUMENT_TYPE_BONUS["unknown"])

        if _RESISTANCE_RE.search(text):
            total += _TEXT_BONUS

        return round(max(0.0, min(1.0, total)), 4)

    except Exception:
        return 0.5
