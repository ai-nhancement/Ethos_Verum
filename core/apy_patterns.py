"""
core/apy_patterns.py

Shared APY pressure regex — single source of truth.
Imported by both core/value_extractor.py (ingestion) and cli/export.py (classification).
"""
import re

APY_PRESSURE_RE = re.compile(
    r'\b(under pressure|when pressed|when threatened|when they demanded|'
    r'to avoid punishment|to save myself|to protect my position|'
    r'they insisted|they demanded|forced to|compelled to|'
    r'or face consequences|or be punished|or lose everything|'
    r'i told them what they wanted to hear|said what was expected)\b',
    re.IGNORECASE,
)
