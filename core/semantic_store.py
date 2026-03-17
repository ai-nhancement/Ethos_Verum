"""
core/semantic_store.py

Qdrant-backed prototype store for Ethos semantic value extraction.

Collections:
  ethos_value_prototypes   — success/hold prototypes (one centroid per value)
  ethos_failure_prototypes — failure/violation prototypes (one centroid per value)

  Point ID mapping: value_name → stable integer (index in VALUE_NAMES list)
  Payload: {"value_name": str, "seed_count": int}

Cosine similarity is used for all queries (vectors are pre-normalized,
so cosine = dot product, which Qdrant handles natively).

Usage:
  from core.semantic_store import get_semantic_store
  store = get_semantic_store()

  # Query hold prototypes (value being demonstrated):
  results = store.query_passage(embedding, top_k=5)
  # → [(value_name, score), ...] sorted by score DESC

  # Query failure prototypes (value being violated):
  failures = store.query_failure_passage(embedding, top_k=5)
  # → [(value_name, score), ...] sorted by score DESC

  # Build prototypes from seed sentences (run once or when seeds change):
  store.build_prototypes(seed_dict)         # {value_name: [sentence, ...]}
  store.build_failure_prototypes(seed_dict) # {value_name: [sentence, ...]}

Constitutional invariants:
  Fail-open — all methods return safe defaults on error.
  No LLM call. Pure vector arithmetic.
  Qdrant on localhost:6333 (same instance as AiMe, separate collection).
"""

from __future__ import annotations

import logging
import threading
from typing import Dict, List, Optional, Tuple

import numpy as np

_log = logging.getLogger(__name__)

_COLLECTION         = "ethos_value_prototypes"
_FAILURE_COLLECTION = "ethos_failure_prototypes"
_QDRANT_HOST        = "localhost"
_QDRANT_PORT        = 6333
_EMBED_DIM          = 1024

# Stable integer IDs for the 15 values (index = point ID in Qdrant)
VALUE_NAMES: List[str] = [
    "integrity", "courage", "compassion", "commitment", "patience",
    "responsibility", "fairness", "gratitude", "curiosity", "resilience",
    "love", "growth", "independence", "loyalty", "humility",
]
_VALUE_TO_ID: Dict[str, int] = {v: i for i, v in enumerate(VALUE_NAMES)}

_LOCK:      threading.Lock         = threading.Lock()
_CLIENT:    Optional[object]       = None
_AVAILABLE: Optional[bool]         = None


# ---------------------------------------------------------------------------
# Qdrant client access
# ---------------------------------------------------------------------------

def _get_client() -> Optional[object]:
    global _CLIENT, _AVAILABLE
    if _AVAILABLE is False:
        return None
    if _CLIENT is not None:
        return _CLIENT
    with _LOCK:
        if _CLIENT is not None:
            return _CLIENT
        try:
            from qdrant_client import QdrantClient               # type: ignore
            from qdrant_client.models import Distance, VectorParams  # type: ignore

            client = QdrantClient(host=_QDRANT_HOST, port=_QDRANT_PORT, timeout=5.0)

            # Ensure both collections exist
            existing = {c.name for c in client.get_collections().collections}
            for cname in (_COLLECTION, _FAILURE_COLLECTION):
                if cname not in existing:
                    client.create_collection(
                        collection_name=cname,
                        vectors_config=VectorParams(size=_EMBED_DIM, distance=Distance.COSINE),
                    )
                    _log.info("Created Qdrant collection: %s", cname)

            _CLIENT = client
            _AVAILABLE = True
            _log.info("Semantic store connected — collection: %s", _COLLECTION)

        except ImportError:
            _log.warning(
                "qdrant-client not installed — semantic layer disabled. "
                "Install with: pip install qdrant-client"
            )
            _AVAILABLE = False
        except Exception:
            _log.warning("Failed to connect to Qdrant — semantic layer disabled",
                         exc_info=True)
            _AVAILABLE = False

    return _CLIENT


# ---------------------------------------------------------------------------
# SemanticStore
# ---------------------------------------------------------------------------

class SemanticStore:
    """
    Manages value prototype vectors in Qdrant.
    Obtain via get_semantic_store() singleton factory.
    """

    def build_prototypes(
        self,
        seed_dict: Dict[str, List[str]],
    ) -> Dict[str, bool]:
        """
        Embed seed sentences for each value, average them into a prototype
        vector, and upsert into Qdrant.

        Args:
            seed_dict: {value_name: [sentence, sentence, ...]}

        Returns:
            {value_name: True/False} — whether each prototype was stored.
        """
        from core.embedder import encode_batch  # import here to stay lazy
        from qdrant_client.models import PointStruct  # type: ignore

        client = _get_client()
        if client is None:
            return {v: False for v in seed_dict}

        results: Dict[str, bool] = {}

        for value_name, sentences in seed_dict.items():
            if value_name not in _VALUE_TO_ID:
                _log.warning("Unknown value name: %s — skipping", value_name)
                results[value_name] = False
                continue
            if not sentences:
                _log.warning("No seed sentences for %s — skipping", value_name)
                results[value_name] = False
                continue

            try:
                vecs = encode_batch(sentences)
                valid = [v for v in vecs if v is not None]
                if not valid:
                    _log.warning("All embeddings failed for %s", value_name)
                    results[value_name] = False
                    continue

                # Average + re-normalize to unit vector
                centroid = np.mean(np.stack(valid), axis=0)
                norm = np.linalg.norm(centroid)
                if norm > 0:
                    centroid = centroid / norm

                point_id = _VALUE_TO_ID[value_name]
                client.upsert(
                    collection_name=_COLLECTION,
                    points=[
                        PointStruct(
                            id=point_id,
                            vector=centroid.tolist(),
                            payload={
                                "value_name": value_name,
                                "seed_count": len(valid),
                            },
                        )
                    ],
                )
                _log.info(
                    "Prototype stored: %s (id=%d, seeds=%d)",
                    value_name, point_id, len(valid),
                )
                results[value_name] = True

            except Exception:
                _log.warning("build_prototypes failed for %s", value_name, exc_info=True)
                results[value_name] = False

        return results

    def build_failure_prototypes(
        self,
        seed_dict: Dict[str, List[str]],
    ) -> Dict[str, bool]:
        """
        Embed failure seed sentences for each value, average them into a
        prototype vector, and upsert into the failure collection.

        These prototypes represent the value being *violated* — a passage
        that scores high here is a signal that the value was failed, not held.

        Args:
            seed_dict: {value_name: [sentence, sentence, ...]}

        Returns:
            {value_name: True/False} — whether each prototype was stored.
        """
        from core.embedder import encode_batch  # import here to stay lazy
        from qdrant_client.models import PointStruct  # type: ignore

        client = _get_client()
        if client is None:
            return {v: False for v in seed_dict}

        results: Dict[str, bool] = {}

        for value_name, sentences in seed_dict.items():
            if value_name not in _VALUE_TO_ID:
                _log.warning("Unknown value name: %s — skipping", value_name)
                results[value_name] = False
                continue
            if not sentences:
                _log.warning("No failure seed sentences for %s — skipping", value_name)
                results[value_name] = False
                continue

            try:
                vecs = encode_batch(sentences)
                valid = [v for v in vecs if v is not None]
                if not valid:
                    _log.warning("All failure embeddings failed for %s", value_name)
                    results[value_name] = False
                    continue

                centroid = np.mean(np.stack(valid), axis=0)
                norm = np.linalg.norm(centroid)
                if norm > 0:
                    centroid = centroid / norm

                point_id = _VALUE_TO_ID[value_name]
                client.upsert(
                    collection_name=_FAILURE_COLLECTION,
                    points=[
                        PointStruct(
                            id=point_id,
                            vector=centroid.tolist(),
                            payload={
                                "value_name": value_name,
                                "seed_count": len(valid),
                            },
                        )
                    ],
                )
                _log.info(
                    "Failure prototype stored: %s (id=%d, seeds=%d)",
                    value_name, point_id, len(valid),
                )
                results[value_name] = True

            except Exception:
                _log.warning(
                    "build_failure_prototypes failed for %s", value_name, exc_info=True
                )
                results[value_name] = False

        return results

    def query_passage(
        self,
        embedding: np.ndarray,
        top_k: int = 5,
        score_threshold: float = 0.0,
    ) -> List[Tuple[str, float]]:
        """
        Find the top-k value prototypes closest to the given passage embedding.

        Args:
            embedding:       L2-normalized 1024-dim float32 array.
            top_k:           Number of results to return.
            score_threshold: Minimum cosine similarity to include.

        Returns:
            [(value_name, score), ...] sorted by score DESC.
            Empty list on error or if semantic store unavailable.
        """
        client = _get_client()
        if client is None:
            return []

        try:
            from qdrant_client.models import QueryRequest  # type: ignore
            hits = client.query_points(
                collection_name=_COLLECTION,
                query=embedding.tolist(),
                limit=top_k,
                score_threshold=score_threshold if score_threshold > 0.0 else None,
                with_payload=True,
            ).points
            return [
                (h.payload.get("value_name", ""), float(h.score))
                for h in hits
                if h.payload.get("value_name")
            ]
        except Exception:
            _log.debug("query_passage failed", exc_info=True)
            return []

    def query_failure_passage(
        self,
        embedding: np.ndarray,
        top_k: int = 5,
        score_threshold: float = 0.0,
    ) -> List[Tuple[str, float]]:
        """
        Find the top-k failure prototypes closest to the given passage embedding.

        A high score here means the passage resembles a *violation* of that value.
        Callers should compare hold scores vs. failure scores to determine label
        direction (P1 if hold >> failure; P0 if failure >> hold).

        Args:
            embedding:       L2-normalized 1024-dim float32 array.
            top_k:           Number of results to return.
            score_threshold: Minimum cosine similarity to include.

        Returns:
            [(value_name, score), ...] sorted by score DESC.
            Empty list on error or if semantic store unavailable.
        """
        client = _get_client()
        if client is None:
            return []

        try:
            hits = client.query_points(
                collection_name=_FAILURE_COLLECTION,
                query=embedding.tolist(),
                limit=top_k,
                score_threshold=score_threshold if score_threshold > 0.0 else None,
                with_payload=True,
            ).points
            return [
                (h.payload.get("value_name", ""), float(h.score))
                for h in hits
                if h.payload.get("value_name")
            ]
        except Exception:
            _log.debug("query_failure_passage failed", exc_info=True)
            return []

    def prototype_count(self) -> int:
        """Return how many hold-prototype vectors are currently stored."""
        client = _get_client()
        if client is None:
            return 0
        try:
            info = client.get_collection(_COLLECTION)
            return int(info.points_count or 0)
        except Exception:
            return 0

    def failure_prototype_count(self) -> int:
        """Return how many failure-prototype vectors are currently stored."""
        client = _get_client()
        if client is None:
            return 0
        try:
            info = client.get_collection(_FAILURE_COLLECTION)
            return int(info.points_count or 0)
        except Exception:
            return 0

    def is_ready(self) -> bool:
        """True if Qdrant is connected and hold prototypes are built."""
        return _get_client() is not None and self.prototype_count() == len(VALUE_NAMES)

    def failure_prototypes_ready(self) -> bool:
        """True if Qdrant is connected and failure prototypes are built."""
        return _get_client() is not None and self.failure_prototype_count() == len(VALUE_NAMES)


# ---------------------------------------------------------------------------
# Singleton factory
# ---------------------------------------------------------------------------

_STORE_LOCK: threading.Lock           = threading.Lock()
_STORE:      Optional[SemanticStore]  = None


def get_semantic_store() -> SemanticStore:
    global _STORE
    if _STORE is None:
        with _STORE_LOCK:
            if _STORE is None:
                _STORE = SemanticStore()
    return _STORE


def is_available() -> bool:
    """True if both Qdrant and the embedding model are available."""
    from core.embedder import is_available as emb_ok
    return emb_ok() and _get_client() is not None
