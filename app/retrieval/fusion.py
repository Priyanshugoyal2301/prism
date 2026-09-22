"""
retrieval/fusion.py — Reciprocal Rank Fusion (RRF) + deduplication.

RRF formula: score(d) = Σ 1 / (k + rank(d)) for each retrieval list.
Default k=60 (well-validated in literature; Cormack et al. 2009).

Also provides the provisional result cache (R4 spike):
  (session_id, normalised_query) → FusedResults
"""

from __future__ import annotations

import hashlib
import re
import time
from collections import defaultdict
from typing import Optional

from app.config import get_settings
from app.schemas import Chunk, FusedResults, RetrievalResult


def _normalise_query(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace for cache keying."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def rrf_fuse(
    result_lists: list[list[RetrievalResult]],
    k: int = 60,
    cosine_dedup_threshold: float = 0.95,
) -> list[RetrievalResult]:
    """
    Fuse multiple ranked lists with Reciprocal Rank Fusion.
    Deduplicates near-identical chunks by chunk_id first, then by text similarity.

    Args:
        result_lists: One ranked list per retrieval method / sub-query.
        k: RRF constant (default 60).
        cosine_dedup_threshold: Chunks with same doc+section are considered duplicates.

    Returns:
        Merged, deduplicated list sorted by RRF score descending.
    """
    scores: dict[str, float] = defaultdict(float)
    chunk_map: dict[str, RetrievalResult] = {}

    for result_list in result_lists:
        for rank, result in enumerate(result_list, start=1):
            cid = result.chunk.chunk_id
            scores[cid] += 1.0 / (k + rank)
            if cid not in chunk_map:
                chunk_map[cid] = result

    # Sort by RRF score
    fused: list[RetrievalResult] = []
    for cid, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        r = chunk_map[cid]
        fused.append(RetrievalResult(
            chunk=r.chunk,
            bm25_score=r.bm25_score,
            dense_score=r.dense_score,
            rrf_score=score,
            sub_query_id=r.sub_query_id,
        ))

    # Deduplicate same doc+section (keep highest RRF only)
    seen_doc_section: set[str] = set()
    deduped: list[RetrievalResult] = []
    for r in fused:
        key = f"{r.chunk.doc_id}::{r.chunk.section}"
        if key not in seen_doc_section:
            seen_doc_section.add(key)
            deduped.append(r)

    return deduped


class HybridRetriever:
    """
    Runs BM25 + dense retrieval per sub-query and fuses with RRF.
    Includes the speculative retrieval cache (R4 spike).
    """

    def __init__(self) -> None:
        cfg = get_settings()
        self._rrf_k = cfg.rrf_k
        self._top_k = cfg.top_k_retrieval
        self._top_k_rerank = cfg.top_k_rerank
        # Provisional cache: (session_id, normalised_query) → FusedResults
        self._cache: dict[str, FusedResults] = {}

    def _cache_key(self, session_id: str, query: str) -> str:
        norm = _normalise_query(query)
        return hashlib.md5(f"{session_id}::{norm}".encode()).hexdigest()

    def query(
        self,
        session_id: str,
        sub_query_id: str,
        query: str,
        bm25_index,
        dense_index,
        force_cache_miss: bool = False,
    ) -> FusedResults:
        """
        Run hybrid retrieval for a single sub-query.
        Returns cached result if available (R4 speculative cache).
        """
        cache_key = self._cache_key(session_id, query)
        if not force_cache_miss and cache_key in self._cache:
            cached = self._cache[cache_key]
            # Update sub_query_id in cached results
            return FusedResults(
                sub_query_id=sub_query_id,
                query=query,
                hits=cached.hits,
                retrieval_latency_ms=cached.retrieval_latency_ms,
                cache_hit=True,
            )

        t0 = time.perf_counter()
        sparse_results = bm25_index.query(query, top_k=self._top_k)
        dense_results = dense_index.query(query, top_k=self._top_k)

        # Tag results with sub_query_id
        for r in sparse_results:
            r.sub_query_id = sub_query_id
        for r in dense_results:
            r.sub_query_id = sub_query_id

        fused = rrf_fuse(
            [sparse_results, dense_results],
            k=self._rrf_k,
        )
        # Keep top_k_rerank for synthesis
        fused = fused[:self._top_k_rerank]

        latency_ms = (time.perf_counter() - t0) * 1000
        result = FusedResults(
            sub_query_id=sub_query_id,
            query=query,
            hits=fused,
            retrieval_latency_ms=round(latency_ms, 2),
            cache_hit=False,
        )
        self._cache[cache_key] = result
        return result

    def clear_session_cache(self, session_id: str) -> None:
        """Remove all cached results for a session (called on session reset)."""
        keys_to_remove = [k for k in self._cache if k.startswith(session_id)]
        for k in keys_to_remove:
            del self._cache[k]

    @property
    def cache_size(self) -> int:
        return len(self._cache)


# ── Module-level singleton ──────────────────────────────────────────────────
_retriever: Optional[HybridRetriever] = None


def get_retriever() -> HybridRetriever:
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever()
    return _retriever
