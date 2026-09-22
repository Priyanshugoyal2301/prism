"""
config.py — Centralised configuration loaded from environment variables.
All tunable thresholds live here so judges can change them without touching code.
"""

from __future__ import annotations

import os
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings
from pydantic_settings import SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── App ──────────────────────────────────────────────────────────────────
    app_version: str = "0.1.0"
    log_level: str = "INFO"
    log_text: bool = False  # if True, log full transcript text (local debug only)

    # ── LLM ──────────────────────────────────────────────────────────────────
    llm_api_key: str = ""
    llm_provider: str = "openai"   # openai | gemini
    llm_base_url: str = "https://api.openai.com/v1"
    llm_controller_model: str = "gpt-4o-mini"
    llm_synthesizer_model: str = "gpt-4o"
    llm_timeout_s: float = 30.0
    llm_max_retries: int = 2

    @property
    def rules_only_mode(self) -> bool:
        """True when no LLM key is provided — entire pipeline runs rule-based."""
        return not self.llm_api_key.strip()

    # ── Retrieval ─────────────────────────────────────────────────────────────
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    faiss_index_type: str = "flat"
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    rrf_k: int = 60
    top_k_retrieval: int = 20
    top_k_rerank: int = 4
    rerank_enabled: bool = False
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Chunking
    chunk_size_tokens: int = 300
    chunk_overlap_tokens: int = 40

    # ── Controller ────────────────────────────────────────────────────────────
    controller_debounce_s: float = 0.4
    controller_similarity_threshold: float = 0.92
    controller_min_tokens: int = 8
    controller_drift_threshold: float = 0.15  # R2: embedding-drift signal
    controller_use_llm_fallback: bool = False

    # ── Decomposer ────────────────────────────────────────────────────────────
    decomposer_max_subqueries: int = 4
    decomposer_orthogonality_threshold: float = 0.85  # R5: merge similar sub-queries

    # ── Session ───────────────────────────────────────────────────────────────
    session_ttl_s: float = 1800.0
    session_max_concurrent: int = 100

    # ── Grounding ─────────────────────────────────────────────────────────────
    grounding_cosine_threshold: float = 0.35   # R6: claim-support scoring
    grounding_lexical_threshold: float = 0.20

    # ── Security ──────────────────────────────────────────────────────────────
    auth_mode: str = "open"
    app_api_token: str = ""
    debug_admin: bool = False
    admin_token: str = ""
    cors_origins: str = "*"
    max_chunk_length: int = 2000
    rate_limit_chunks_per_sec: int = 20
    rate_limit_sessions_per_min: int = 30

    # ── Telemetry ─────────────────────────────────────────────────────────────
    telemetry_log_path: str = "logs/telemetry.jsonl"

    # ── Data paths ────────────────────────────────────────────────────────────
    corpus_path: str = "data/corpus"
    index_cache_path: str = ".index_cache"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()

