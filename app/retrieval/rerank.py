"""
app/retrieval/rerank.py — Optional cross-encoder rerank (P1, behind RERANK_ENABLED flag).

Used in the ablation: "with rerank vs without rerank" for the benchmark report.
Only loaded if RERANK_ENABLED=true in config.
"""

from __future__ import annotations

from typing import Optional

from app.config import get_settings
from app.schemas import RetrievalResult


class CrossEncoderReranker:
    """
    Cross-encoder reranker using a small model (ms-marco-MiniLM-L-6-v2).
    Scores (query, passage) pairs directly for more accurate relevance scoring.
    """

    def __init__(self) -> None:
        cfg = get_settings()
        self._model_name = cfg.rerank_model
        self._model = None

    def _load_model(self) -> None:
        if self._model is not None:
            return
        from sentence_transformers import CrossEncoder
        self._model = CrossEncoder(self._model_name)

    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int = 4,
    ) -> list[RetrievalResult]:
        """Rerank results using the cross-encoder; return top_k."""
        if not results:
            return results
        self._load_model()
        pairs = [(query, r.chunk.text) for r in results]
        scores = self._model.predict(pairs)
        ranked = sorted(zip(scores, results), key=lambda x: x[0], reverse=True)
        reranked = []
        for score, result in ranked[:top_k]:
            result.rrf_score = float(score)  # override with rerank score
            reranked.append(result)
        return reranked


_reranker: Optional[CrossEncoderReranker] = None


def get_reranker() -> Optional[CrossEncoderReranker]:
    """Return reranker only if RERANK_ENABLED=true; None otherwise."""
    global _reranker
    cfg = get_settings()
    if not cfg.rerank_enabled:
        return None
    if _reranker is None:
        _reranker = CrossEncoderReranker()
    return _reranker
