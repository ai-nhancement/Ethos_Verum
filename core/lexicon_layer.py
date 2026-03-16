"""
core/lexicon_layer.py

Layer 1b — Moral lexicon augmentation.

Two bundled lexicons loaded at import time from data/lexicons/:

  MFD2.0 (Moral Foundations Dictionary 2.0, Frimer et al. 2019)
    2,104 words annotated with 10 MFT categories (5 foundations × virtue/vice).
    Format: LIWC .dic — word → category code.

  MoralStrength (Araque et al. 2019)
    ~454 lemmas (across 5 TSV files) with crowd-sourced moral strength scores
    on a 1–10 continuous scale per foundation.
    1 = strongly associated with the vice side of that foundation.
    10 = strongly associated with the virtue side.

Both lexicons use Moral Foundations Theory (MFT) categories:
  care / fairness / loyalty / authority / sanctity (purity)

These map to Ethos values via _MFT_TO_ETHOS.
Values not covered by MFT (courage, patience, curiosity, resilience,
growth, independence, gratitude) are not scored by this layer —
they remain fully covered by Layers 1 and 2.

Signal schema (same as all other layers):
  {value_name, text_excerpt, significance, disambiguation_confidence, source}
  source = "lexicon" or "lexicon+ms" (when MoralStrength agrees)

Constitutional invariants:
  No network call. Lexicons loaded from local files only.
  Fail-open — missing files produce empty dicts, layer returns [].
  No LLM call.
"""

from __future__ import annotations

import logging
import os
import re
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_log = logging.getLogger(__name__)

_LEXICON_DIR = Path(__file__).resolve().parent.parent / "data" / "lexicons"

# ---------------------------------------------------------------------------
# MFT foundation → Ethos value mapping
# ---------------------------------------------------------------------------
# virtue polarity: mapped values receive a positive signal
# vice polarity:   same mapped values get a failure indicator
#                  (currently used to flag potential P0; included with
#                   lower confidence so resistance scoring can gate them)

_MFT_TO_ETHOS: Dict[str, List[str]] = {
    "care":      ["compassion", "love", "gratitude"],
    "fairness":  ["fairness", "responsibility"],
    "loyalty":   ["loyalty", "commitment", "resilience"],
    "authority": ["responsibility", "humility"],
    "sanctity":  ["integrity", "commitment"],
}

# Base confidence for a lexicon-only signal.
# Lower than keyword (0.7–1.0) because a lexicon match lacks the
# contextual specificity of the keyword vocabulary.
_LEXICON_BASE_CONF   = 0.55
_MS_MAX_BOOST        = 0.15  # max additional boost from MoralStrength agreement
_MS_VIRTUE_THRESHOLD = 5.5   # MoralStrength scores > this are virtue-leaning


# ---------------------------------------------------------------------------
# MFD2.0 loader
# ---------------------------------------------------------------------------

def _load_mfd2() -> Dict[str, Tuple[str, str]]:
    """
    Parse MFD2.0 LIWC .dic format.
    Returns {word: (foundation, polarity)} e.g. {"compassion": ("care", "virtue")}.
    """
    path = _LEXICON_DIR / "mfd2.txt"
    if not path.exists():
        _log.warning("MFD2.0 not found at %s — lexicon layer partially disabled", path)
        return {}

    # Category number → (foundation, polarity)
    cat_map: Dict[str, Tuple[str, str]] = {}
    result:  Dict[str, Tuple[str, str]] = {}

    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            content = f.read()

        segments = content.split("%")
        if len(segments) < 3:
            _log.warning("MFD2.0 format unexpected — skipping")
            return {}

        # Segment 1: category definitions ("1\tcare.virtue\n2\tcare.vice\n...")
        for line in segments[1].strip().splitlines():
            parts = line.strip().split("\t")
            if len(parts) == 2:
                num, label = parts
                # label format: "care.virtue" or "sanctity.vice"
                if "." in label:
                    foundation, polarity = label.rsplit(".", 1)
                    # normalize "purity" → "sanctity" (used interchangeably in MFT)
                    if foundation == "purity":
                        foundation = "sanctity"
                    cat_map[num.strip()] = (foundation, polarity)

        # Segment 2+: word entries ("word\t1\n")
        for line in segments[2].strip().splitlines():
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                word = parts[0].lower().strip()
                cat_num = parts[1].strip()
                if word and cat_num in cat_map:
                    result[word] = cat_map[cat_num]

        _log.info("MFD2.0 loaded: %d entries", len(result))
        return result

    except Exception:
        _log.warning("MFD2.0 load failed", exc_info=True)
        return {}


# ---------------------------------------------------------------------------
# MoralStrength loader
# ---------------------------------------------------------------------------

def _load_moralstrength() -> Dict[str, Dict[str, float]]:
    """
    Load MoralStrength TSV files.
    Returns {word: {foundation: score}} where score ∈ [1.0, 10.0].
    A score > 5.5 is virtue-leaning; < 5.5 is vice-leaning.
    """
    foundations = ["care", "fairness", "loyalty", "authority", "purity"]
    result: Dict[str, Dict[str, float]] = {}

    for foundation in foundations:
        path = _LEXICON_DIR / f"moralstrength_{foundation}.tsv"
        if not path.exists():
            _log.debug("MoralStrength %s.tsv not found — skipping", foundation)
            continue
        try:
            with open(path, encoding="utf-8") as f:
                lines = f.readlines()
            for line in lines[1:]:  # skip header
                parts = line.strip().split("\t")
                if len(parts) == 2:
                    word = parts[0].lower().strip()
                    try:
                        score = float(parts[1])
                    except ValueError:
                        continue
                    # normalize "purity" → "sanctity"
                    norm_foundation = "sanctity" if foundation == "purity" else foundation
                    if word not in result:
                        result[word] = {}
                    result[word][norm_foundation] = round(score, 4)
        except Exception:
            _log.warning("MoralStrength %s.tsv load failed", foundation, exc_info=True)

    _log.info("MoralStrength loaded: %d entries", len(result))
    return result


# ---------------------------------------------------------------------------
# Module-level singletons — loaded once at import
# ---------------------------------------------------------------------------

_MFD2:          Dict[str, Tuple[str, str]]       = {}
_MORALSTRENGTH: Dict[str, Dict[str, float]]      = {}
_LOADED    = False
_LOAD_LOCK = threading.Lock()


def _ensure_loaded() -> None:
    global _LOADED
    if _LOADED:
        return
    with _LOAD_LOCK:
        if _LOADED:
            return
        # Use .update() so that `from lexicon_layer import _MFD2` references
        # stay valid — reassignment would create a new object and break them.
        _MFD2.update(_load_mfd2())
        _MORALSTRENGTH.update(_load_moralstrength())
        _LOADED = True


# ---------------------------------------------------------------------------
# Token extraction helper
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"\b[a-z][a-z\-']{1,}\b")


def _tokenize(text: str) -> List[str]:
    """Extract lowercase alphabetic tokens from text."""
    return _TOKEN_RE.findall(text.lower())


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def lexicon_signals(
    text: str,
    significance: float,
    doc_type: str = "unknown",
) -> List[Dict]:
    """
    Scan text for MFD2.0 and MoralStrength matches.

    Returns signal dicts in the standard Ethos schema:
      {value_name, text_excerpt, significance, disambiguation_confidence, source}

    One signal per (value_name, polarity) pair per passage.
    Virtue signals: source="lexicon" or "lexicon+ms"
    Vice signals: source="lexicon_vice" (lower confidence, failure indicators)

    Returns [] if lexicons not available or no matches found.
    """
    if not text:
        return []

    _ensure_loaded()
    if not _MFD2 and not _MORALSTRENGTH:
        return []

    tokens = _tokenize(text)
    if not tokens:
        return []

    excerpt = text[:150] if len(text) <= 150 else text[:147] + "..."

    # Accumulate best signal per (ethos_value, polarity)
    # key: (value_name, polarity) → (best_conf, source_tag)
    best: Dict[Tuple[str, str], Tuple[float, str]] = {}

    for token in tokens:
        # MFD2.0 lookup
        mfd_entry = _MFD2.get(token)
        if mfd_entry:
            foundation, polarity = mfd_entry
            ethos_values = _MFT_TO_ETHOS.get(foundation, [])
            for vname in ethos_values:
                conf = _LEXICON_BASE_CONF
                src  = "lexicon"

                # MoralStrength boost: if token has a score for this foundation
                ms_entry = _MORALSTRENGTH.get(token, {})
                ms_score = ms_entry.get(foundation)
                if ms_score is not None:
                    if polarity == "virtue" and ms_score > _MS_VIRTUE_THRESHOLD:
                        boost = min(_MS_MAX_BOOST,
                                    (ms_score - _MS_VIRTUE_THRESHOLD) / (10.0 - _MS_VIRTUE_THRESHOLD)
                                    * _MS_MAX_BOOST)
                        conf = round(conf + boost, 4)
                        src  = "lexicon+ms"
                    elif polarity == "vice" and ms_score < _MS_VIRTUE_THRESHOLD:
                        # Vice match confirmed by MoralStrength
                        src = "lexicon+ms"

                key = (vname, polarity)
                existing_conf, _ = best.get(key, (0.0, ""))
                if conf > existing_conf:
                    best[key] = (conf, src)

        # MoralStrength-only matches (words in MS but not in MFD2.0)
        elif token in _MORALSTRENGTH:
            ms_entry = _MORALSTRENGTH[token]
            for foundation, ms_score in ms_entry.items():
                ethos_values = _MFT_TO_ETHOS.get(foundation, [])
                polarity = "virtue" if ms_score > _MS_VIRTUE_THRESHOLD else "vice"
                conf = round(
                    _LEXICON_BASE_CONF * (abs(ms_score - 5.5) / 4.5),
                    4,
                )
                if conf < 0.20:
                    continue  # too close to neutral — skip
                src = "lexicon+ms"
                for vname in ethos_values:
                    key = (vname, polarity)
                    existing_conf, _ = best.get(key, (0.0, ""))
                    if conf > existing_conf:
                        best[key] = (conf, src)

    # Build output signal list
    results: List[Dict] = []
    for (vname, polarity), (conf, src) in best.items():
        results.append({
            "value_name":               vname,
            "text_excerpt":             excerpt,
            "significance":             significance,
            "disambiguation_confidence": conf,
            "source":                   src if polarity == "virtue" else src + "_vice",
            "lexicon_polarity":         polarity,  # "virtue" or "vice"
        })

    return results


def mfd2_size() -> int:
    """Return number of entries in the loaded MFD2.0 lexicon."""
    _ensure_loaded()
    return len(_MFD2)


def moralstrength_size() -> int:
    """Return number of entries in the loaded MoralStrength lexicon."""
    _ensure_loaded()
    return len(_MORALSTRENGTH)


def is_available() -> bool:
    """True if at least one lexicon loaded successfully."""
    _ensure_loaded()
    return bool(_MFD2 or _MORALSTRENGTH)
