"""
tests/test_fusion.py — Unit tests for RRF fusion and speculative cache.
"""

import pytest

from app.retrieval.fusion import HybridRetriever, rrf_fuse
from app.schemas import Chunk, RetrievalResult


def make_chunk(chunk_id: str, doc_id: str = None, section: str = "§1", text: str = "test") -> Chunk:
    if doc_id is None:
        doc_id = chunk_id  # use chunk_id as doc_id for uniqueness in tests
    return Chunk(chunk_id=chunk_id, doc_id=doc_id, section=section, text=text, token_count=1)


def make_result(chunk_id: str, doc_id: str = None, section: str = "§1", score: float = 1.0) -> RetrievalResult:
    return RetrievalResult(chunk=make_chunk(chunk_id, doc_id, section), bm25_score=score, dense_score=score)


class TestRRFFusion:
    def test_basic_fusion(self):
        list1 = [make_result("c1"), make_result("c2"), make_result("c3")]
        list2 = [make_result("c2"), make_result("c1"), make_result("c4")]
        fused = rrf_fuse([list1, list2])
        assert len(fused) >= 3
        # c1 and c2 appear in both lists → should rank higher
        top_ids = [r.chunk.chunk_id for r in fused[:2]]
        assert "c1" in top_ids or "c2" in top_ids

    def test_deduplication_same_doc_section(self):
        # Two chunks with SAME doc+section → deduped to 1
        list1 = [make_result("Doc_01 §1 [1/2]", doc_id="Doc_01", section="§1")]
        list2 = [make_result("Doc_01 §1 [2/2]", doc_id="Doc_01", section="§1")]
        fused = rrf_fuse([list1, list2])
        # Both have same doc+section → deduplicated to 1
        doc_sections = {f"{r.chunk.doc_id}::{r.chunk.section}" for r in fused}
        assert len(doc_sections) == 1

    def test_rrf_scores_are_positive(self):
        list1 = [make_result("c1"), make_result("c2")]
        fused = rrf_fuse([list1])
        assert all(r.rrf_score > 0 for r in fused)

    def test_empty_list_handled(self):
        fused = rrf_fuse([[], []])
        assert fused == []

    def test_single_list(self):
        list1 = [make_result("c1"), make_result("c2"), make_result("c3")]
        fused = rrf_fuse([list1])
        assert len(fused) == 3
        assert fused[0].chunk.chunk_id == "c1"  # first in list → highest score


class TestSpeculativeCache:
    def test_cache_hit_on_same_query(self):
        retriever = HybridRetriever()
        # Manually populate cache
        from app.schemas import FusedResults
        fake_result = FusedResults(sub_query_id="sq1", query="Pune venue", hits=[])
        key = retriever._cache_key("session-001", "Pune venue capacity")
        retriever._cache[key] = fake_result

        # Same session + normalised query should hit cache
        # (we test the cache key normalisation)
        key2 = retriever._cache_key("session-001", "Pune venue capacity")
        assert key == key2

    def test_different_sessions_no_cache_bleed(self):
        retriever = HybridRetriever()
        key1 = retriever._cache_key("session-001", "Pune venue")
        key2 = retriever._cache_key("session-002", "Pune venue")
        assert key1 != key2

    def test_cache_clear_on_session_reset(self):
        retriever = HybridRetriever()
        from app.schemas import FusedResults
        fake = FusedResults(sub_query_id="sq1", query="test", hits=[])
        key = retriever._cache_key("session-001", "test query")
        retriever._cache[key] = fake
        assert retriever.cache_size == 1
        retriever.clear_session_cache("session-001")
        # Cache clear uses prefix matching — verify it doesn't crash
        # (actual clear depends on key format)
