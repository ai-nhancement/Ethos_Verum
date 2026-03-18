"""
core/phrase_layer.py

Two-pass phrase composition detection for vice-word tokens.

Motivation
----------
Single-token lexicon lookup cannot distinguish:

  "committed cruelty"  — destructive  (agent perpetrating vice)
  "resisted cruelty"   — constructive (agent opposing vice)
  "ordered persecution" — destructive
  "condemned persecution" — constructive

This layer fills the gap by detecting (compositional_verb + vice_word)
pairs and emitting signals with pre-determined polarity. Virtue words are
handled well enough by the existing polarity layer (which already uses
inversion verbs) and are not addressed here.

Algorithm
---------
Pass 1: Tokenize text; find all tokens that appear in the MFD2.0 vice lexicon.
Pass 2: For each vice token, scan the _PHRASE_PROX_CHARS-wide pre-window
        (text immediately preceding the token) for a compositional verb.
        Classify:
          destructive_verb + vice_word → polarity_hint = -1
          opposition_verb  + vice_word → polarity_hint = +1
        Tokens with no qualifying verb in-window fall through and are NOT
        consumed — the caller's single-word lexicon path handles them.

Returns
-------
(signals, consumed_tokens)
  signals          — list of phrase-level signal dicts; each includes a
                     'polarity_hint' key (int: -1 or +1) in addition to
                     the standard Ethos signal schema.
  consumed_tokens  — set of lowercase token strings consumed by phrase
                     matches. Caller may use this to suppress redundant
                     single-word scoring for those tokens.

Constitutional invariants
-------------------------
  No network call. Imports lexicon singletons already loaded by lexicon_layer.
  Fail-open — returns ([], set()) on any error.
  No LLM call.
"""

from __future__ import annotations

import re
from typing import Dict, List, Set, Tuple

# ---------------------------------------------------------------------------
# Proximity window
# ---------------------------------------------------------------------------
# Scan this many characters *before* the vice word for a compositional verb.
# 60 chars catches most "committed ... cruelty" constructions including
# short prepositional phrases: "committed acts of cruelty" (30 chars before).
_PHRASE_PROX_CHARS = 60

# Base confidence for a phrase-level signal.
# Slightly above _LEXICON_BASE_CONF (0.55) because we have compositional
# evidence, but below keyword confidence (0.70) since we still lack
# entity-level disambiguation.
_PHRASE_BASE_CONF = 0.58

# ---------------------------------------------------------------------------
# Compositional verb patterns
# ---------------------------------------------------------------------------

# Destructive agency verbs — the subject perpetrated / directed the vice.
# Covers explicit execution verbs but not generic carriers (used, showed)
# to avoid false positives on "used against tyranny" or "showed courage".
_DESTRUCTIVE_RE = re.compile(
    r"\b(?:"
    r"commit(?:ted|s|ting)?|"
    r"inflict(?:ed|s|ing)?|"
    r"perpetrat(?:ed|es|ing)?|"
    r"orchestrat(?:ed|es|ing)?|"
    r"authoriz(?:ed|es|ing)?|authoris(?:ed|es|ing)?|"
    r"sanction(?:ed|s|ing)?|"
    r"order(?:ed|s|ing)?|"
    r"organiz(?:ed|es|ing)?|organis(?:ed|es|ing)?|"
    r"practic(?:ed|es|ing)?|"
    r"exercis(?:ed|es|ing)?|"
    r"unleash(?:ed|es|ing)?|"
    r"employ(?:ed|s|ing)?|"
    r"propagat(?:ed|es|ing)?"
    r")\b",
    re.IGNORECASE,
)

# Opposition verbs — the subject resisted / opposed the vice.
# Explicit compound patterns (fought against, stood against) are included;
# bare "against" alone is NOT included (same rationale as polarity_layer).
_OPPOSITION_RE = re.compile(
    r"\b(?:"
    r"resist(?:ed|s|ing)?|"
    r"oppos(?:ed|es|ing)?|"
    r"confront(?:ed|s|ing)?|"
    r"challeng(?:ed|es|ing)?|"
    r"condemn(?:ed|s|ing)?|"
    r"denounc(?:ed|es|ing)?|"
    r"reject(?:ed|s|ing)?|"
    r"combat(?:t?(?:ed|ing)|s)?|"
    r"expos(?:ed|es|ing)?|"
    r"fought\s+against|fighting\s+against|"
    r"stood\s+against|standing\s+against|"
    r"stood\s+up\s+to|stands?\s+up\s+to|standing\s+up\s+to|"
    r"spoke\s+against|speaking\s+against|"
    r"speak(?:s)?\s+against|"
    r"campaigned\s+against|campaign(?:s|ing)?\s+against|"
    r"worked\s+against|work(?:s|ing)?\s+against|"
    r"protest(?:ed|s|ing)?(?:\s+against)?"
    r")\b",
    re.IGNORECASE,
)

# Token regex: same alphabet as lexicon_layer._TOKEN_RE
_TOKEN_RE = re.compile(r"\b[a-zA-Z][a-zA-Z\-']{1,}\b")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def phrase_signals(
    text: str,
    significance: float,
    doc_type: str = "unknown",
) -> Tuple[List[Dict], Set[str]]:
    """
    Detect compositional (verb + vice-word) phrases in text.

    Args:
        text:         Passage text to scan.
        significance: Significance score (0.0–1.0) for the passage.
        doc_type:     Document type key (informational only; not used in
                      phrase scoring currently).

    Returns:
        (signals, consumed_tokens) — see module docstring for schema.
        Never raises; returns ([], set()) on any error.
    """
    try:
        from core.lexicon_layer import _ensure_loaded, _MFD2, _MFT_TO_ETHOS  # type: ignore
        _ensure_loaded()
        if not _MFD2:
            return [], set()

        if not text:
            return [], set()

        excerpt = text[:150] if len(text) <= 150 else text[:147] + "..."
        signals: List[Dict] = []
        consumed: Set[str] = set()

        for m in _TOKEN_RE.finditer(text):
            token_lower = m.group(0).lower()

            mfd_entry = _MFD2.get(token_lower)
            if not mfd_entry:
                continue

            foundation, polarity = mfd_entry
            if polarity != "vice":
                # Virtue tokens fall through to existing lexicon_layer path
                continue

            ethos_values = _MFT_TO_ETHOS.get(foundation, [])
            if not ethos_values:
                continue

            start = m.start()
            pre_window = text[max(0, start - _PHRASE_PROX_CHARS) : start]

            if _DESTRUCTIVE_RE.search(pre_window):
                polarity_hint = -1
            elif _OPPOSITION_RE.search(pre_window):
                polarity_hint = 1
            else:
                # No qualifying verb — fall through, do not consume
                continue

            consumed.add(token_lower)
            for vname in ethos_values:
                signals.append({
                    "value_name":                vname,
                    "text_excerpt":              excerpt,
                    "significance":              significance,
                    "disambiguation_confidence": _PHRASE_BASE_CONF,
                    "source":                    "phrase_layer",
                    "polarity_hint":             polarity_hint,
                    "match_idx":                 start,
                })

        # Deduplicate: keep one signal per (value_name, polarity_hint) pair.
        # When both verb types fire for the same vice word (unlikely but possible),
        # the destructive signal is kept — that indicates a richer context.
        best: Dict[Tuple, Dict] = {}
        for sig in signals:
            key = (sig["value_name"], sig["polarity_hint"])
            if key not in best:
                best[key] = sig

        return list(best.values()), consumed

    except Exception:
        return [], set()
