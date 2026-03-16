"""
core/mft_classifier.py

Layer 3c — Moral Foundations Theory (MFT) classifier.

Uses MMADS/MoralFoundationsClassifier (fine-tuned RoBERTa, 10 output labels)
to detect which moral foundations are active in a passage, then maps that to
Ethos value boosts and vice flags.

Label mapping (from model config.json):
    LABEL_0: care.virtue       LABEL_1: care.vice
    LABEL_2: fairness.virtue   LABEL_3: fairness.vice
    LABEL_4: loyalty.virtue    LABEL_5: loyalty.vice
    LABEL_6: authority.virtue  LABEL_7: authority.vice
    LABEL_8: sanctity.virtue   LABEL_9: sanctity.vice

Constitutional invariants:
  No LLM call anywhere in this module.
  Fail-open: any import or inference error returns empty results.
  Model loaded lazily and cached as a singleton.
"""

from __future__ import annotations

import logging
import threading
from typing import Dict, List

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Label → (foundation, polarity)
# ---------------------------------------------------------------------------

LABEL_MAP: Dict[str, tuple[str, str]] = {
    "LABEL_0": ("care",      "virtue"),
    "LABEL_1": ("care",      "vice"),
    "LABEL_2": ("fairness",  "virtue"),
    "LABEL_3": ("fairness",  "vice"),
    "LABEL_4": ("loyalty",   "virtue"),
    "LABEL_5": ("loyalty",   "vice"),
    "LABEL_6": ("authority", "virtue"),
    "LABEL_7": ("authority", "vice"),
    "LABEL_8": ("sanctity",  "virtue"),
    "LABEL_9": ("sanctity",  "vice"),
}

# ---------------------------------------------------------------------------
# MFT foundation → Ethos values boosted when foundation virtue fires
# ---------------------------------------------------------------------------

MFT_VIRTUE_TO_ETHOS: Dict[str, List[str]] = {
    "care":      ["compassion", "love", "gratitude"],
    "fairness":  ["fairness", "responsibility", "integrity"],
    "loyalty":   ["loyalty", "courage", "commitment", "resilience"],
    "authority": ["responsibility", "humility"],
    "sanctity":  ["integrity", "commitment"],
}

# authority.vice (defiance/subversion) can signal independence in the right context.
# Discounted because it is indirect evidence — the model fires on defiance language,
# not on an explicit independence demonstration.
_AUTHORITY_VICE_DISCOUNT: float = 0.70
MFT_AUTHORITY_VICE_HINT: List[str] = ["independence"]

# ---------------------------------------------------------------------------
# Singleton classifier
# ---------------------------------------------------------------------------

_clf      = None
_clf_lock = threading.Lock()
_clf_unavailable = False   # set True after first failed load (avoids retry cost)


def _get_classifier():
    """Lazy-load singleton.  Returns None if model is unavailable."""
    global _clf, _clf_unavailable
    if _clf is not None:
        return _clf
    if _clf_unavailable:
        return None
    with _clf_lock:
        if _clf is not None:
            return _clf
        if _clf_unavailable:
            return None
        try:
            from transformers import pipeline as hf_pipeline
            _clf = hf_pipeline(
                "text-classification",
                model="MMADS/MoralFoundationsClassifier",
                top_k=None,
                device=-1,   # CPU; GPU would be device=0
            )
            _log.info("MFT classifier loaded (MMADS/MoralFoundationsClassifier)")
        except Exception as exc:
            _log.warning("MFT classifier unavailable: %s", exc)
            _clf_unavailable = True
    return _clf


def is_available() -> bool:
    """Return True if the classifier can be loaded."""
    return _get_classifier() is not None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def mft_signals(
    text: str,
    min_virtue_score: float = 0.60,
    min_vice_score:   float = 0.85,
    *,
    truncation: bool = True,
    max_length: int  = 512,
) -> Dict:
    """
    Run the MFT classifier on *text* and return:
        {
            "boosted_values": [
                {"value_name": str, "mft_foundation": str, "score": float}
            ],
            "vice_flags": [
                {"foundation": str, "score": float}
            ],
        }

    boosted_values: Ethos values that should receive a confidence boost,
        derived from firing virtue foundations.
    vice_flags: foundations whose vice label fired above min_vice_score —
        these are hints that the passage may contain a P0/APY event.

    Fail-open: returns {"boosted_values": [], "vice_flags": []} on any error.
    """
    empty: Dict = {"boosted_values": [], "vice_flags": []}
    clf = _get_classifier()
    if clf is None:
        return empty
    if not text or not text.strip():
        return empty

    try:
        # Let the tokenizer handle truncation at the token level (not chars).
        # Passing truncation=True avoids the over-aggressive text[:512] char
        # slice that was discarding 75%+ of context for typical passages.
        raw = clf(text, truncation=truncation, max_length=max_length)
        scores = raw[0]                # list of {"label": ..., "score": ...}
    except Exception as exc:
        _log.debug("MFT inference failed: %s", exc)
        return empty

    boosted: List[Dict] = []
    vice_flags: List[Dict] = []

    for item in scores:
        label_id = item["label"]
        score    = float(item["score"])
        if label_id not in LABEL_MAP:
            continue

        foundation, polarity = LABEL_MAP[label_id]

        if polarity == "virtue" and score >= min_virtue_score:
            ethos_values = MFT_VIRTUE_TO_ETHOS.get(foundation, [])
            for vname in ethos_values:
                boosted.append({
                    "value_name":     vname,
                    "mft_foundation": foundation,
                    "score":          round(score, 4),
                })

        elif polarity == "vice":
            if foundation == "authority" and score >= min_virtue_score:
                # High-confidence authority.vice can signal independence
                # (defiance of authority ≈ asserting one's own judgment).
                for vname in MFT_AUTHORITY_VICE_HINT:
                    boosted.append({
                        "value_name":     vname,
                        "mft_foundation": "authority_defiance",
                        "score":          round(score * _AUTHORITY_VICE_DISCOUNT, 4),
                    })
            if score >= min_vice_score:
                vice_flags.append({
                    "foundation": foundation,
                    "score":      round(score, 4),
                })

    return {"boosted_values": boosted, "vice_flags": vice_flags}
