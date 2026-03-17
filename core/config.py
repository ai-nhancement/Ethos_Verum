from __future__ import annotations
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Document type authenticity — single source of truth for both modules.
#
# resistance_bonus  : added to base resistance score during extraction
#                     (core/resistance.py)
# training_weight   : multiplied against significance when building training
#                     records (cli/export.py)
# default_significance : used when the caller does not supply a significance
#                     score at ingest time (cli/ingest.py, api)
#
# Relationship: training_weight = 0.8 + resistance_bonus
# This preserves the authenticity hierarchy in both contexts and keeps
# speech at a slight penalty (0.90) relative to unknown (1.00).
# ---------------------------------------------------------------------------

DOC_TYPE_RESISTANCE_BONUS: dict = {
    "action":  0.40,   # documented real-world behavior — highest authenticity
    "journal": 0.35,   # private writing, no audience pressure
    "letter":  0.30,   # directed correspondence
    "speech":  0.10,   # public address — highest performance pressure
    "unknown": 0.20,   # default
}

DOC_TYPE_TRAINING_WEIGHT: dict = {
    "action":  1.20,   # 0.80 + 0.40
    "journal": 1.15,   # 0.80 + 0.35
    "letter":  1.10,   # 0.80 + 0.30
    "speech":  0.90,   # 0.80 + 0.10  (slight penalty — performance context)
    "unknown": 1.00,   # 0.80 + 0.20
}

DOC_TYPE_DEFAULT_SIGNIFICANCE: dict = {
    "action":  0.90,   # observed behavior under stakes
    "journal": 0.85,   # private writing
    "letter":  0.80,   # directed correspondence
    "speech":  0.70,   # public address — performance pressure
    "unknown": 0.75,   # default
}


@dataclass
class Config:
    min_significance_threshold: float = 0.10
    min_resistance_threshold: float = 0.20
    enabled: bool = True
    apy_context_window_n: int = 5

    # Semantic layer (Phase 1)
    semantic_enabled: bool = True          # False = keyword-only mode
    semantic_threshold: float = 0.45      # Min cosine similarity to fire a signal
    semantic_top_k: int = 3               # Number of prototypes to retrieve per passage
    # Blend weights for keyword + semantic confidence scores.
    # keyword_weight + semantic_weight need not sum to 1.0 — they are applied
    # independently and the max of the two is used as the final detection confidence.
    # Set semantic_weight = 0.0 to disable blending (semantic as gating only).
    keyword_weight: float = 1.0
    semantic_weight: float = 0.8

    # Layer 1b — Moral lexicons (Phase 2)
    lexicon_enabled: bool = True
    lexicon_standalone_min_conf: float = 0.55  # min conf for lexicon-only new signals

    # Layer 3 — Structural + zero-shot classifiers (Phase 2)
    layer3_enabled: bool = True
    structural_resistance_boost: float = 0.15   # max boost to resistance from structural score
    zeroshot_enabled: bool = True               # False = structural only (faster)
    zeroshot_threshold: float = 0.35            # min entailment probability (agreement mode)
    zeroshot_standalone_threshold: float = 0.70 # higher bar when zero-shot fires alone
    # Confidence boost applied when zero-shot agrees with L1/L2 on same value
    zeroshot_agreement_boost: float = 0.15

    # Layer 3c — MFT classifier (Phase 2)
    mft_enabled: bool = True
    mft_min_virtue_score: float = 0.60     # min score for virtue foundation to fire
    mft_min_vice_score: float = 0.85       # min score for vice flag to be recorded
    mft_agreement_boost: float = 0.10      # confidence boost when MFT agrees with L1/L2
    mft_standalone_threshold: float = 0.80 # min score to create a new standalone signal
    mft_standalone_weight: float = 0.50    # weight applied to score for standalone confidence


_default: Config | None = None


def get_config() -> Config:
    global _default
    if _default is None:
        _default = Config()
    return _default


# ---------------------------------------------------------------------------
# Value tension pairs (researcher-configurable)
# From Schwartz (1992) and Rokeach (1973) — default 5 pairs per §7.8 spec.
# Each tuple is (value_a, value_b); tension is symmetric.
# ---------------------------------------------------------------------------

VALUE_TENSION_PAIRS: list[tuple[str, str]] = [
    ("independence", "loyalty"),      # Autonomy vs. belonging
    ("fairness",     "compassion"),   # Impartiality vs. mercy
    ("courage",      "patience"),     # Action vs. deliberation
    ("responsibility", "humility"),   # Ownership vs. deference
    ("commitment",   "growth"),       # Persistence vs. revision
]

_TENSION_PAIR_SET: frozenset = frozenset(
    frozenset(p) for p in VALUE_TENSION_PAIRS
)


def is_tension_pair(v1: str, v2: str) -> bool:
    return frozenset({v1, v2}) in _TENSION_PAIR_SET
