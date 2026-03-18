"""
core/phrase_layer.py

Two-pass phrase composition detection for vice-word tokens.

Motivation
----------
Single-token lexicon lookup cannot distinguish:

  "committed cruelty"           — destructive (figure perpetrated vice)
  "they committed cruelty against him" — destructive for THEM, not the figure
  "resisted cruelty"            — constructive (figure opposed vice)
  "ordered persecution"         — destructive
  "condemned persecution"       — constructive

This layer fills the gap by detecting (compositional_verb + vice_word) pairs
and emitting signals with pre-determined polarity.  A subject/object
resolution step uses the figure's stored pronoun to determine whether the
figure is the agent or the recipient of the act, preventing false attribution.

Virtue words are handled well enough by the existing polarity layer (which
already uses inversion verbs) and are not addressed here.

Algorithm
---------
Pass 1: Tokenize text; find all tokens that appear in the MFD2.0 vice lexicon.
Pass 2: For each vice token, scan the _PHRASE_PROX_CHARS-wide pre-window
        (text immediately preceding the token) for a compositional verb.
        Classify verb type:
          destructive — committed, inflicted, perpetrated, ordered, …
          opposition  — resisted, opposed, condemned, fought against, …
        Then resolve agency:
          For destructive verbs:
            • Figure-subject pronoun or name before verb → figure is agent → -1
            • Figure-object pronoun after vice word (with preposition) → recipient → suppress
            • Non-figure pronoun before verb → external agent → suppress
            • No pronoun context → biographical default → emit -1
          For opposition verbs:
            • Always emit +1 — opposition is constructive regardless of subject
        Tokens with no qualifying verb in-window fall through (not consumed).

Subject pronoun sets by pronoun value
--------------------------------------
  "he"   → subject: {he, his}         object: {him, his}
  "she"  → subject: {she, her, hers}  object: {her, hers}
  "they" → subject: {they, their}     object: {them, their}
  "i"    → subject: {i, we, my, our}  object: {me, us}
  "unknown" → skip resolution (legacy / backward-compatible behavior)

Returns
-------
(signals, consumed_tokens)
  signals          — list of phrase-level signal dicts; each includes a
                     'polarity_hint' key (int: -1 or +1) in addition to
                     the standard Ethos signal schema.
  consumed_tokens  — set of lowercase token strings consumed by phrase
                     matches.

Constitutional invariants
-------------------------
  No network call. Imports lexicon singletons already loaded by lexicon_layer.
  Fail-open — returns ([], set()) on any error.
  No LLM call.
"""

from __future__ import annotations

import re
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# Proximity windows
# ---------------------------------------------------------------------------
# Scan this many chars *before* the vice word for a compositional verb.
_PHRASE_PROX_CHARS = 60

# Scan this many chars *before* the verb within the pre-window for a subject.
_SUBJECT_PROX_CHARS = 30

# Scan this many chars *after* the vice word for an object preposition.
_OBJECT_PROX_CHARS = 40

# Base confidence for a phrase-level signal.
_PHRASE_BASE_CONF = 0.58

# ---------------------------------------------------------------------------
# Valid pronouns
# ---------------------------------------------------------------------------
VALID_PRONOUNS: FrozenSet[str] = frozenset({"he", "she", "they", "i"})

# ---------------------------------------------------------------------------
# Pronoun → subject / object word sets
# ---------------------------------------------------------------------------
_PRONOUN_SETS: Dict[str, Dict[str, FrozenSet[str]]] = {
    "he":   {
        "subject": frozenset({"he", "his"}),
        "object":  frozenset({"him", "his"}),
    },
    "she":  {
        "subject": frozenset({"she", "her", "hers"}),
        "object":  frozenset({"her", "hers"}),
    },
    "they": {
        "subject": frozenset({"they", "their"}),
        "object":  frozenset({"them", "their"}),
    },
    "i":    {
        "subject": frozenset({"i", "we", "my", "our"}),
        "object":  frozenset({"me", "us"}),
    },
}

# All common third-person subject pronouns (used to detect external agents).
_ALL_SUBJECT_PRONOUNS: FrozenSet[str] = frozenset(
    {"he", "she", "they", "it", "those", "others", "one", "we", "i"}
)

# Prepositions that mark the recipient of an action.
_RECIPIENT_PREP_RE = re.compile(
    r"\b(?:against|toward|towards|upon|at)\s+(\w+)\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Compositional verb patterns
# ---------------------------------------------------------------------------

# Destructive agency verbs — the subject perpetrated / directed the vice.
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
# Agency resolution
# ---------------------------------------------------------------------------

def _resolve_agency(
    pre_window: str,
    post_window: str,
    verb_match,
    figure_name_lower: str,
    pronoun: str,
) -> str:
    """
    Determine the figure's role relative to a destructive verb match.

    Returns one of:
      "agent"          — figure is the subject performing the act
      "recipient"      — figure is the object receiving the act
      "external_agent" — an explicit non-figure pronoun is the subject
      "emit_anyway"    — no pronoun context; biographical default (emit)
    """
    if pronoun == "unknown" or pronoun not in _PRONOUN_SETS:
        return "emit_anyway"

    sets = _PRONOUN_SETS[pronoun]
    fig_subjects: FrozenSet[str] = sets["subject"]
    fig_objects:  FrozenSet[str] = sets["object"]

    verb_start = verb_match.start()
    near = pre_window[max(0, verb_start - _SUBJECT_PROX_CHARS) : verb_start].lower()

    # Is the figure explicitly the subject (pronoun or name before verb)?
    for subj in fig_subjects:
        if re.search(r"\b" + re.escape(subj) + r"\b", near):
            return "agent"
    if figure_name_lower and re.search(
        r"\b" + re.escape(figure_name_lower) + r"\b", near
    ):
        return "agent"

    # Is the figure explicitly the recipient (object pronoun after preposition)?
    m = _RECIPIENT_PREP_RE.search(post_window)
    if m and m.group(1).lower() in fig_objects:
        return "recipient"

    # Is an explicit non-figure pronoun the subject?
    external = _ALL_SUBJECT_PRONOUNS - fig_subjects
    for ext in external:
        if re.search(r"\b" + re.escape(ext) + r"\b", near):
            return "external_agent"

    return "emit_anyway"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def phrase_signals(
    text: str,
    significance: float,
    doc_type: str = "unknown",
    figure_name: str = "",
    pronoun: str = "unknown",
) -> Tuple[List[Dict], Set[str]]:
    """
    Detect compositional (verb + vice-word) phrases in text.

    Args:
        text:         Passage text to scan.
        significance: Significance score (0.0–1.0) for the passage.
        doc_type:     Document type key (informational; not used in scoring).
        figure_name:  Name of the historical figure being ingested.
                      Used for name-based subject detection.
        pronoun:      Figure's pronoun — one of "he", "she", "they", "i",
                      or "unknown" to skip subject resolution (legacy mode).

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

        figure_name_lower = figure_name.lower() if figure_name else ""
        excerpt = text[:150] if len(text) <= 150 else text[:147] + "..."
        signals: List[Dict] = []
        consumed: Set[str] = set()

        for m in _TOKEN_RE.finditer(text):
            token_lower = m.group(0).lower()

            mfd_entry = _MFD2.get(token_lower)
            if not mfd_entry:
                continue

            foundation, lex_polarity = mfd_entry
            if lex_polarity != "vice":
                continue

            ethos_values = _MFT_TO_ETHOS.get(foundation, [])
            if not ethos_values:
                continue

            start = m.start()
            end   = m.end()
            pre_window  = text[max(0, start - _PHRASE_PROX_CHARS) : start]
            post_window = text[end : end + _OBJECT_PROX_CHARS]

            dest_m = _DESTRUCTIVE_RE.search(pre_window)
            if dest_m:
                agency = _resolve_agency(
                    pre_window, post_window, dest_m,
                    figure_name_lower, pronoun,
                )
                if agency in ("recipient", "external_agent"):
                    continue  # figure is not the perpetrator — suppress
                polarity_hint = -1

            elif _OPPOSITION_RE.search(pre_window):
                # Opposition is always constructive — no subject check needed.
                polarity_hint = 1

            else:
                # No qualifying verb — fall through, do not consume.
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

        # Deduplicate: one signal per (value_name, polarity_hint).
        best: Dict[Tuple, Dict] = {}
        for sig in signals:
            key = (sig["value_name"], sig["polarity_hint"])
            if key not in best:
                best[key] = sig

        return list(best.values()), consumed

    except Exception:
        return [], set()
