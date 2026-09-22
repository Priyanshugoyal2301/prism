"""
synthesis/session.py — Per-session state model.

The session is the source of truth for:
- The current utterance buffer
- All answer versions (versioned claims → citations)
- The evidence pool (chunk_id → Chunk)
- Sub-query registry
- Event history for telemetry

Research spike R3: claim graph — claims link to sub_query_id AND evidence_chunk_ids,
enabling surgical delta patching.
"""

from __future__ import annotations

import time
import uuid
from collections import defaultdict
from threading import RLock
from typing import Optional

from app.config import get_settings
from app.schemas import (
    AnswerDiff,
    AnswerDiffOp,
    AnswerVersion,
    Chunk,
    Claim,
    PatchOp,
    SessionState,
    SubQuery,
)


class Session:
    """
    In-memory session. One instance per active session_id.
    Thread-safe via RLock.
    """

    def __init__(self, session_id: str) -> None:
        cfg = get_settings()
        self.session_id = session_id
        self.created_at = time.time()
        self.ttl_s = cfg.session_ttl_s
        self._lock = RLock()

        # Utterance buffer (raw concatenated text)
        self.utterance_buffer: str = ""
        self.buffer_start_t: float = 0.0

        # Answer state
        self.answer_version: int = 0
        self.answer_versions: list[AnswerVersion] = []
        self.current_claims: list[Claim] = []

        # Sub-queries
        self.sub_queries: dict[str, SubQuery] = {}  # sub_query_id → SubQuery

        # Evidence pool — all chunks retrieved in this session
        self.evidence_pool: dict[str, Chunk] = {}  # chunk_id → Chunk

        # Event history (lightweight, for debug)
        self.history: list[dict] = []

        # Timestamps
        self.last_retrieval_t: float = 0.0
        self.last_answer_t: float = 0.0

    # ── Buffer management ──────────────────────────────────────────────────

    def append_chunk(self, text: str, t_start_s: float) -> None:
        with self._lock:
            if not self.utterance_buffer:
                self.buffer_start_t = t_start_s
            self.utterance_buffer = (self.utterance_buffer + " " + text).strip()

    def clear_buffer(self) -> None:
        with self._lock:
            self.utterance_buffer = ""
            self.buffer_start_t = 0.0

    # ── Evidence pool ──────────────────────────────────────────────────────

    def add_evidence(self, chunks: list[Chunk]) -> None:
        with self._lock:
            for c in chunks:
                self.evidence_pool[c.chunk_id] = c

    def chunk_in_pool(self, chunk_id: str) -> bool:
        return chunk_id in self.evidence_pool

    def get_chunk(self, chunk_id: str) -> Optional[Chunk]:
        return self.evidence_pool.get(chunk_id)

    # ── Sub-query registry ─────────────────────────────────────────────────

    def register_sub_queries(self, sub_queries: list[SubQuery]) -> None:
        with self._lock:
            for sq in sub_queries:
                self.sub_queries[sq.sub_query_id] = sq

    # ── Answer versioning ──────────────────────────────────────────────────

    def commit_answer(self, claims: list[Claim], uncertainty: str = "") -> AnswerVersion:
        """
        Commit a new answer version. Returns the new AnswerVersion.
        R3: claims carry sub_query_id and citations, enabling delta patching.
        """
        with self._lock:
            self.answer_version += 1
            all_citations = list({
                cit.chunk_id
                for claim in claims
                for cit in claim.citations
                if cit.verified
            })
            version = AnswerVersion(
                version=self.answer_version,
                claims=claims,
                citations=all_citations,
                uncertainty=uncertainty,
                t_created_s=time.time(),
            )
            self.answer_versions.append(version)
            self.current_claims = claims
            self.last_answer_t = time.time()
            return version

    def get_current_version(self) -> Optional[AnswerVersion]:
        if not self.answer_versions:
            return None
        return self.answer_versions[-1]

    # ── R3: Claim graph delta patching ────────────────────────────────────

    def build_claim_graph(self) -> dict[str, set[str]]:
        """
        R3: Build a mapping sub_query_id → set of claim_ids.
        Used to identify which claims are affected by a late constraint.
        """
        graph: dict[str, set[str]] = defaultdict(set)
        for claim in self.current_claims:
            graph[claim.sub_query_id].add(claim.claim_id)
        return graph

    def patch_claims(
        self,
        affected_sub_query_ids: list[str],
        new_claims: list[Claim],
        new_uncertainty: str = "",
    ) -> tuple[AnswerVersion, AnswerDiff]:
        """
        R3: Apply a delta patch to the current answer.

        For each claim:
        - RETAIN: claim's sub_query_id is not in affected_sub_query_ids
        - UPDATE/ADD: comes from new_claims
        - (Removed claims are dropped silently)

        Returns (new_version, diff).
        """
        with self._lock:
            claim_graph = self.build_claim_graph()
            affected_claim_ids: set[str] = set()
            for sq_id in affected_sub_query_ids:
                affected_claim_ids.update(claim_graph.get(sq_id, set()))

            # Build patched claim list
            ops: list[AnswerDiffOp] = []
            patched: list[Claim] = []

            # 1. Retain unaffected claims
            for claim in self.current_claims:
                if claim.claim_id not in affected_claim_ids:
                    claim.patch_op = PatchOp.RETAIN
                    patched.append(claim)
                    ops.append(AnswerDiffOp(
                        op=PatchOp.RETAIN,
                        claim_id=claim.claim_id,
                        text_after=claim.text,
                    ))

            # 2. Apply new/updated claims
            new_claim_ids = {c.claim_id for c in new_claims}
            for new_claim in new_claims:
                # Check if this is an update to an existing claim
                existing_match = next(
                    (c for c in self.current_claims
                     if c.claim_id in affected_claim_ids
                     and _claims_similar(c.text, new_claim.text)),
                    None,
                )
                if existing_match:
                    new_claim.patch_op = PatchOp.UPDATE
                    patched.append(new_claim)
                    ops.append(AnswerDiffOp(
                        op=PatchOp.UPDATE,
                        claim_id=new_claim.claim_id,
                        text_before=existing_match.text,
                        text_after=new_claim.text,
                        citations_added=[c.chunk_id for c in new_claim.citations],
                    ))
                else:
                    new_claim.patch_op = PatchOp.ADD
                    patched.append(new_claim)
                    ops.append(AnswerDiffOp(
                        op=PatchOp.ADD,
                        claim_id=new_claim.claim_id,
                        text_after=new_claim.text,
                        citations_added=[c.chunk_id for c in new_claim.citations],
                    ))

            # Commit patched answer
            retained_count = sum(1 for o in ops if o.op == PatchOp.RETAIN)
            updated_count = sum(1 for o in ops if o.op == PatchOp.UPDATE)
            added_count = sum(1 for o in ops if o.op == PatchOp.ADD)

            prev_version = self.answer_version
            new_version = self.commit_answer(patched, new_uncertainty)

            diff = AnswerDiff(
                version_from=prev_version,
                version_to=new_version.version,
                ops=ops,
                summary=(
                    f"Retained {retained_count} claims · "
                    f"Updated {updated_count} · "
                    f"Added {added_count} · "
                    f"Retrieval only for delta"
                ),
            )
            return new_version, diff

    # ── Serialisation ──────────────────────────────────────────────────────

    def to_state(self) -> SessionState:
        with self._lock:
            return SessionState(
                session_id=self.session_id,
                created_at=self.created_at,
                ttl_s=self.ttl_s,
                utterance_buffer=self.utterance_buffer,
                answer_version=self.answer_version,
                answer_versions=list(self.answer_versions),
                sub_queries=dict(self.sub_queries),
                evidence_pool=dict(self.evidence_pool),
                history=list(self.history),
            )

    @property
    def is_expired(self) -> bool:
        return (time.time() - self.created_at) > self.ttl_s


def _claims_similar(text1: str, text2: str, threshold: float = 0.5) -> bool:
    """Lightweight Jaccard similarity to detect updated vs new claims."""
    w1 = set(text1.lower().split())
    w2 = set(text2.lower().split())
    if not w1 or not w2:
        return False
    return len(w1 & w2) / len(w1 | w2) >= threshold


# ─────────────────────────────────────────────────────────────────────────────
# Session Store
# ─────────────────────────────────────────────────────────────────────────────

class SessionStore:
    """
    In-memory session store with TTL eviction.
    Session-bound and ephemeral: no cross-session access, no persistence.
    """

    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}
        self._lock = RLock()
        cfg = get_settings()
        self._max_sessions = cfg.session_max_concurrent

    def get_or_create(self, session_id: str) -> Session:
        with self._lock:
            self._evict_expired()
            if session_id not in self._sessions:
                if len(self._sessions) >= self._max_sessions:
                    raise RuntimeError(
                        f"Maximum concurrent sessions ({self._max_sessions}) reached."
                    )
                self._sessions[session_id] = Session(session_id)
            return self._sessions[session_id]

    def get(self, session_id: str) -> Optional[Session]:
        return self._sessions.get(session_id)

    def reset(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)

    def _evict_expired(self) -> None:
        expired = [sid for sid, s in self._sessions.items() if s.is_expired]
        for sid in expired:
            del self._sessions[sid]

    @property
    def active_count(self) -> int:
        return len(self._sessions)


# ── Module-level singleton ──────────────────────────────────────────────────
_session_store: Optional[SessionStore] = None


def get_session_store() -> SessionStore:
    global _session_store
    if _session_store is None:
        _session_store = SessionStore()
    return _session_store
