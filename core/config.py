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
