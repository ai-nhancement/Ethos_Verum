from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class Config:
    min_significance_threshold: float = 0.10
    min_resistance_threshold: float = 0.20
    enabled: bool = True


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
