"""
retrieval/dense.py — Dense embedding + FAISS retrieval index.

Uses BAAI/bge-small-en-v1.5 via sentence-transformers (CPU).
FAISS IndexFlatIP for exact cosine similarity (vectors normalised to unit length).
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np

from app.config import get_settings
from app.schemas import Chunk, RetrievalResult


class DenseIndex:
    """
    Dense retrieval index using BGE-small-en-v1.5 + FAISS.
    Vectors are L2-normalised; inner-product search = cosine similarity.
    """

    def __init__(self) -> None:
        cfg = get_settings()
        self._model_name = cfg.embedding_model
        self._index_type = cfg.faiss_index_type
        self._model = None  # lazy-loaded
        self._index = None  # FAISS index
        self._chunks: list[Chunk] = []
        self._embeddings: Optional[np.ndarray] = None

    def _load_model(self) -> None:
        if self._model is not None:
            return
        print(f"[DenseIndex] Loading embedding model: {self._model_name}")
        t0 = time.perf_counter()
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(self._model_name)
        print(f"[DenseIndex] Model loaded in {(time.perf_counter()-t0):.2f}s")

    def build(self, chunks: list[Chunk]) -> None:
        """Encode all chunks and build the FAISS index."""
        import faiss

        self._load_model()
        self._chunks = chunks

        texts = [c.text for c in chunks]
        print(f"[DenseIndex] Encoding {len(texts)} chunks...")
        t0 = time.perf_counter()
        # BGE-small benefits from the instruction prefix for retrieval
        texts_with_prefix = [f"Represent this sentence: {t}" if "bge" in self._model_name.lower() else t
                             for t in texts]
        embeddings = self._model.encode(
            texts_with_prefix,
            batch_size=64,
            show_progress_bar=len(texts) > 100,
            normalize_embeddings=True,  # L2-norm → cosine via IP
            convert_to_numpy=True,
        )
        self._embeddings = embeddings.astype(np.float32)
        print(f"[DenseIndex] Encoded in {(time.perf_counter()-t0):.2f}s")

        # Build FAISS index
        dim = self._embeddings.shape[1]
        if self._index_type == "hnsw":
            self._index = faiss.IndexHNSWFlat(dim, 32)
        else:
            self._index = faiss.IndexFlatIP(dim)  # exact cosine
        self._index.add(self._embeddings)
        print(f"[DenseIndex] FAISS index built: {dim}d, {len(chunks)} vectors")

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single query string."""
        self._load_model()
        prefix = "Represent this question for searching relevant passages: " if "bge" in self._model_name.lower() else ""
        vec = self._model.encode(
            [prefix + text],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return vec[0].astype(np.float32)

    def query(self, text: str, top_k: int = 20) -> list[RetrievalResult]:
        """Return top-k chunks ranked by cosine similarity."""
        if self._index is None:
            raise RuntimeError("DenseIndex.build() has not been called.")
        q_vec = self.embed_query(text).reshape(1, -1)
        scores, indices = self._index.search(q_vec, min(top_k, len(self._chunks)))
        results: list[RetrievalResult] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            results.append(RetrievalResult(
                chunk=self._chunks[idx],
                dense_score=float(score),
                bm25_score=0.0,
                rrf_score=0.0,
            ))
        return results
