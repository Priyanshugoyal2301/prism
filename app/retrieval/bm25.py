"""
retrieval/bm25.py — BM25 sparse retrieval index.

Uses rank_bm25 (BM25Okapi variant). Fully CPU-bound and LangChain-free.
"""

from __future__ import annotations

import re
from typing import Optional

from rank_bm25 import BM25Okapi

from app.config import get_settings
from app.schemas import Chunk, RetrievalResult


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokeniser. No external dependencies."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return [t for t in text.split() if len(t) > 1]


class BM25Index:
    """
    BM25 retrieval index over corpus chunks.
    Build once; query many times.
    """

    def __init__(self) -> None:
        cfg = get_settings()
        self._k1 = cfg.bm25_k1
        self._b = cfg.bm25_b
        self._index: Optional[BM25Okapi] = None
        self._chunks: list[Chunk] = []

    def build(self, chunks: list[Chunk]) -> None:
        """Build the BM25 index from a list of chunks."""
        self._chunks = chunks
        corpus_tokens = [_tokenize(c.text) for c in chunks]
        self._index = BM25Okapi(corpus_tokens, k1=self._k1, b=self._b)

    def query(self, text: str, top_k: int = 20) -> list[RetrievalResult]:
        """Return top-k chunks ranked by BM25 score."""
        if self._index is None:
            raise RuntimeError("BM25Index.build() has not been called.")
        query_tokens = _tokenize(text)
        if not query_tokens:
            return []

        scores = self._index.get_scores(query_tokens)
        # Get top-k indices, sorted descending
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results: list[RetrievalResult] = []
        for idx in top_indices:
            if scores[idx] <= 0:
                continue
            results.append(RetrievalResult(
                chunk=self._chunks[idx],
                bm25_score=float(scores[idx]),
                dense_score=0.0,
                rrf_score=0.0,
            ))
        return results
