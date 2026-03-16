"""
core/temporal_layer.py

Phase 3 — Translation and Temporal Handling.

Three independent calibration functions:

  1. Language detection + source_authenticity multiplier
     Detects the language of a document at ingest time.
     If the text is not in its original language (translation),
     the resistance/confidence scores are discounted because the
     evidential signal may be weakened by translation choices.

     source_authenticity multipliers:
       original (detected English + not flagged as translation): 1.00
       known_translation (caller asserts it):                    0.85
       uncertain (non-English detected, no flag):                0.70

  2. Archaic preprocessing
     Deterministic normalization of Early Modern English (c.1400–1800)
     to modern equivalents before embedding. Preserves meaning, improves
     embedding alignment with modern training corpora.
     Run before encode() calls in the semantic layer.

  3. pub_year temporal discount
     Mild discount on embedding confidence for pre-1850 documents.
     Embedding models are trained on modern text; archaic register and
     vocabulary shift reduce the reliability of cosine similarity scores
     even after preprocessing.

Constitutional invariants:
  No LLM call.
  Fail-open — all functions return safe defaults on error.
  langdetect seeded for deterministic output.
"""

from __future__ import annotations

import logging
import re
from typing import Optional, Tuple

_log = logging.getLogger(__name__)

# Seed langdetect for deterministic output
try:
    from langdetect import DetectorFactory  # type: ignore
    DetectorFactory.seed = 0
    _LANGDETECT_OK = True
except ImportError:
    _LANGDETECT_OK = False
    _log.warning("langdetect not installed — language detection disabled. "
                 "Install with: pip install langdetect")


# ---------------------------------------------------------------------------
# Source authenticity multipliers
# ---------------------------------------------------------------------------

AUTHENTICITY_ORIGINAL     = 1.00
AUTHENTICITY_TRANSLATION  = 0.85
AUTHENTICITY_UNCERTAIN    = 0.70


def detect_language(text: str) -> Tuple[str, float]:
    """
    Detect the language of a text sample.

    Args:
        text: The text to analyze (uses first 2,000 chars for speed).

    Returns:
        (lang_code, confidence) where lang_code is an ISO 639-1 code
        (e.g. 'en', 'fr', 'de') and confidence is in [0.0, 1.0].
        Returns ('unknown', 0.0) on failure.
    """
    if not _LANGDETECT_OK or not text:
        return "unknown", 0.0
    try:
        from langdetect import detect_langs  # type: ignore
        sample = text[:2000]
        results = detect_langs(sample)
        if results:
            top = results[0]
            return top.lang, round(float(top.prob), 4)
        return "unknown", 0.0
    except Exception:
        _log.debug("detect_language failed", exc_info=True)
        return "unknown", 0.0


def source_authenticity(
    lang_code: str,
    is_translation: Optional[bool] = None,
) -> float:
    """
    Compute the source_authenticity multiplier for a document.

    Args:
        lang_code:      ISO 639-1 language code from detect_language().
        is_translation: True  = caller asserts this is a translation
                        False = caller asserts this is original text
                        None  = unknown; infer from lang_code

    Returns:
        Float multiplier in {1.0, 0.85, 0.70}.
    """
    if is_translation is True:
        return AUTHENTICITY_TRANSLATION
    if is_translation is False:
        return AUTHENTICITY_ORIGINAL
    # Infer: if detected as English, assume original (most Ethos corpora are English)
    if lang_code == "en":
        return AUTHENTICITY_ORIGINAL
    if lang_code == "unknown":
        return AUTHENTICITY_UNCERTAIN
    # Non-English — likely a translated historical source
    return AUTHENTICITY_UNCERTAIN


# ---------------------------------------------------------------------------
# Archaic preprocessing
# ---------------------------------------------------------------------------

# Ordered replacement table: (pattern, replacement)
# Applied left-to-right; order matters for overlapping rules.
# Patterns use word boundaries to avoid replacing substrings.
_ARCHAIC_RULES: list[Tuple[re.Pattern, str]] = [
    # Contractions first (before pronouns, to avoid 'tis → it is → confusion)
    # Note: \b before ' doesn't work (' is \W); use (?<!\w) instead.
    (re.compile(r"(?<!\w)'tis\b",    re.IGNORECASE), "it is"),
    (re.compile(r"(?<!\w)'twas\b",   re.IGNORECASE), "it was"),
    (re.compile(r"(?<!\w)'twere\b",  re.IGNORECASE), "it were"),
    (re.compile(r"(?<!\w)'twill\b",  re.IGNORECASE), "it will"),

    # Thou/thee/thy/thine/ye
    (re.compile(r"\bthou\b",    re.IGNORECASE), "you"),
    (re.compile(r"\bthee\b",    re.IGNORECASE), "you"),
    (re.compile(r"\bthy\b",     re.IGNORECASE), "your"),
    (re.compile(r"\bthine\b",   re.IGNORECASE), "your"),
    (re.compile(r"\bye\b",      re.IGNORECASE), "you"),

    # Verb forms: third-person singular present
    (re.compile(r"\bhath\b",    re.IGNORECASE), "has"),
    (re.compile(r"\bdoth\b",    re.IGNORECASE), "does"),
    (re.compile(r"\bsayeth\b",  re.IGNORECASE), "says"),
    (re.compile(r"\bgoeth\b",   re.IGNORECASE), "goes"),
    (re.compile(r"\bcometh\b",  re.IGNORECASE), "comes"),
    (re.compile(r"\bknoweth\b", re.IGNORECASE), "knows"),
    (re.compile(r"\bseeth\b",   re.IGNORECASE), "sees"),
    (re.compile(r"\btaketh\b",  re.IGNORECASE), "takes"),
    (re.compile(r"\bmaketh\b",  re.IGNORECASE), "makes"),
    (re.compile(r"\bgiveth\b",  re.IGNORECASE), "gives"),
    (re.compile(r"\bliveth\b",  re.IGNORECASE), "lives"),

    # Second-person singular present: art/wast/wert
    (re.compile(r"\bart\b",     re.IGNORECASE), "are"),
    (re.compile(r"\bwast\b",    re.IGNORECASE), "were"),
    (re.compile(r"\bwert\b",    re.IGNORECASE), "were"),

    # Modal past forms: shouldst/wouldst/couldst/mightst/mustst
    (re.compile(r"\bshouldst\b",  re.IGNORECASE), "should"),
    (re.compile(r"\bwouldst\b",   re.IGNORECASE), "would"),
    (re.compile(r"\bcouldst\b",   re.IGNORECASE), "could"),
    (re.compile(r"\bmightst\b",   re.IGNORECASE), "might"),

    # Dost / didst
    (re.compile(r"\bdost\b",    re.IGNORECASE), "do"),
    (re.compile(r"\bdidst\b",   re.IGNORECASE), "did"),

    # Canst / wilt (archaic will/can)
    (re.compile(r"\bcanst\b",   re.IGNORECASE), "can"),
    (re.compile(r"\bwilt\b",    re.IGNORECASE), "will"),

    # Hast (have)
    (re.compile(r"\bhast\b",    re.IGNORECASE), "have"),

    # Nay / yea
    (re.compile(r"\bnay\b",     re.IGNORECASE), "no"),
    (re.compile(r"\byea\b",     re.IGNORECASE), "yes"),

    # Wherefore / henceforth / hitherto / thereof / therein / thereof
    # Leave these: they're rare and their removal could distort meaning.
    # The embedding model handles them adequately as compound words.
]


def preprocess_archaic(text: str) -> str:
    """
    Normalize archaic/Early Modern English to modern equivalents.

    Designed for documents from c.1400–1800. Safe to run on modern text —
    the patterns are specific enough to avoid false matches.

    Returns the normalized text, preserving all punctuation and structure.
    Never raises — returns original text on error.
    """
    if not text:
        return text
    try:
        for pattern, replacement in _ARCHAIC_RULES:
            text = pattern.sub(
                lambda m: _match_case(replacement, m.group(0)),
                text,
            )
        return text
    except Exception:
        _log.debug("preprocess_archaic failed", exc_info=True)
        return text


def _match_case(replacement: str, original: str) -> str:
    """Preserve capitalization of original when applying replacement."""
    if original.isupper():
        return replacement.upper()
    if original[0].isupper():
        return replacement[0].upper() + replacement[1:]
    return replacement


# ---------------------------------------------------------------------------
# pub_year embedding confidence discount
# ---------------------------------------------------------------------------

# Thresholds for temporal discount
_YEAR_THRESHOLDS = [
    (1400, 0.70),   # Pre-1400: heavy discount — very archaic, low embedding reliability
    (1600, 0.80),   # 1400–1600: significant archaic register
    (1700, 0.88),   # 1600–1700: Early Modern English
    (1800, 0.93),   # 1700–1800: 18th century, mostly modernizable
    (1850, 0.97),   # 1800–1850: minor discount
]
_YEAR_MODERN = 1.00   # 1850+: no discount


def pub_year_discount(pub_year: Optional[int]) -> float:
    """
    Return a multiplier in (0, 1] for embedding confidence based on pub_year.

    More recent documents get multiplier 1.0.
    Older documents get a mild discount reflecting that embedding models
    are trained on modern text and archaic vocabulary/syntax reduces
    cosine similarity reliability.

    Returns 1.0 if pub_year is None (no information → no penalty).
    """
    if pub_year is None:
        return 1.0
    for threshold, multiplier in _YEAR_THRESHOLDS:
        if pub_year < threshold:
            return multiplier
    return _YEAR_MODERN


# ---------------------------------------------------------------------------
# Combined: calibrate a signal's confidence for temporal + translation effects
# ---------------------------------------------------------------------------

def calibrate_confidence(
    confidence: float,
    source_auth: float,
    year_discount: float,
) -> float:
    """
    Apply source_authenticity and pub_year_discount to a signal confidence.

    Both multipliers apply independently:
      calibrated = confidence × source_auth × year_discount

    Result is clipped to [0.0, 1.0].
    """
    return round(max(0.0, min(1.0, confidence * source_auth * year_discount)), 4)
