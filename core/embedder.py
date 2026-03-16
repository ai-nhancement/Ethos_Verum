"""
core/embedder.py

BGE-large embedding singleton for Ethos semantic extraction.

Model: BAAI/bge-large-en-v1.5 (1024d)
  - Chosen over BGE-base for +1.04 retrieval gain on MTEB benchmarks
  - Ethos has no existing vectors — clean slate, no migration cost
  - AiMe remains on BGE-base; this is an independent model load

Constitutional invariants:
  No LLM call. Pure embedding inference — deterministic, no sampling.
  Fail-open — returns None on any error so extraction degrades gracefully.
  Lazy load — model not loaded until first encode() call.
"""

from __future__ import annotations

import logging
import threading
from typing import List, Optional

import numpy as np

_log = logging.getLogger(__name__)

_MODEL_NAME = "BAAI/bge-large-en-v1.5"
_EMBED_DIM  = 1024

_LOCK:     threading.Lock            = threading.Lock()
_MODEL:    Optional[object]          = None   # SentenceTransformer instance
_AVAILABLE: Optional[bool]           = None   # None = not yet checked


def _load_model() -> Optional[object]:
    """Load BGE-large once. Returns None if sentence-transformers unavailable."""
    global _MODEL, _AVAILABLE
    if _AVAILABLE is False:
        return None
    if _MODEL is not None:
        return _MODEL
    with _LOCK:
        if _MODEL is not None:
            return _MODEL
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
            _log.info("Loading %s — first call may take a moment...", _MODEL_NAME)
            _MODEL = SentenceTransformer(_MODEL_NAME)
            _AVAILABLE = True
            _log.info("BGE-large loaded (%dd)", _EMBED_DIM)
        except ImportError:
            _log.warning(
                "sentence-transformers not installed — semantic layer disabled. "
                "Install with: pip install sentence-transformers"
            )
            _AVAILABLE = False
        except Exception:
            _log.warning("Failed to load %s — semantic layer disabled", _MODEL_NAME,
                         exc_info=True)
            _AVAILABLE = False
    return _MODEL


def encode(text: str) -> Optional[np.ndarray]:
    """
    Embed a single text string. Returns a 1024-dim float32 numpy array,
    L2-normalized. Returns None if the model is unavailable or on any error.
    """
    model = _load_model()
    if model is None:
        return None
    try:
        vec = model.encode(text, normalize_embeddings=True, show_progress_bar=False)
        return np.array(vec, dtype=np.float32)
    except Exception:
        _log.debug("encode() failed", exc_info=True)
        return None


def encode_batch(texts: List[str], batch_size: int = 32) -> List[Optional[np.ndarray]]:
    """
    Embed a list of texts in batches. Returns a list of normalized float32 arrays.
    Individual failures return None in their slot.
    """
    model = _load_model()
    if model is None:
        return [None] * len(texts)
    try:
        vecs = model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=batch_size,
            show_progress_bar=False,
        )
        return [np.array(v, dtype=np.float32) for v in vecs]
    except Exception:
        _log.debug("encode_batch() failed", exc_info=True)
        return [None] * len(texts)


def is_available() -> bool:
    """Return True if the embedding model is loaded and ready."""
    return _load_model() is not None


def embed_dim() -> int:
    return _EMBED_DIM
