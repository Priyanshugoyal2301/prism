"""
controller.py — Retrieval Controller: decides WAIT / RETRIEVE / NO_RETRIEVAL.

Two-tier architecture:
1. Rules fast-path (< 5 ms, no LLM): handles most cases deterministically.
2. Optional LLM classifier fallback for genuinely ambiguous cases (P1, gated by env flag).

Research spike R2: embedding-drift signal — measures cosine distance between
consecutive partial-utterance embeddings; fires RETRIEVE when drift is low
(semantic settledness) AND ≥1 named entity is detected.

All decisions are logged with reason codes for telemetry and ablation analysis.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from app.config import get_settings
from app.schemas import (
    ChunkEvent,
    ControllerDecision,
    ControllerOutput,
    ReasonCode,
    TriggerType,
)
from app.telemetry import get_telemetry, TelemetryEventType


# ─────────────────────────────────────────────────────────────────────────────
# Keyword lexicons for the rules fast-path
# ─────────────────────────────────────────────────────────────────────────────

PRESENTATION_VERBS: set[str] = {
    "repeat", "rephrase", "restate", "summarize", "summarise",
    "shorten", "bullet", "bullets", "translate", "format",
    "reformat", "list", "simpler", "briefer", "shorter",
    "expand", "elaborate", "explain that", "say that again",
}

INTENT_VERBS: set[str] = {
    "need", "want", "find", "show", "tell", "what", "how",
    "when", "where", "which", "who", "can", "is", "are",
    "does", "do", "cost", "price", "policy", "option",
    "available", "book", "cancel", "reimburse", "refund",
    "catering", "venue", "capacity", "allow", "permit",
}

# Named entity patterns (lightweight, no spaCy dependency)
PLACE_PATTERNS = [
    r"\b(?:pune|mumbai|delhi|bangalore|hyderabad|chennai|kolkata|india|dubai|london|new york|paris)\b",
    r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b",  # Capitalised words as potential places
]
NUMBER_PATTERNS = [
    r"\b\d+\s*(?:people|person|persons|attendees|guests|pax)\b",
    r"\b\d+\s*(?:days?|nights?|hours?|minutes?)\b",
    r"\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\b",
    r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
]
CONJUNCTION_PATTERNS = [
    r"\band\b", r"\balso\b", r"\bplus\b", r"\badditionally\b",
    r"\bmoreover\b", r"\bfurthermore\b", r"\bbesides\b",
]


def _detect_entities(text: str) -> dict[str, list[str]]:
    """Extract entities without spaCy — regex-based, fast."""
    text_lower = text.lower()
    entities: dict[str, list[str]] = {"place": [], "number": [], "conjunction": []}
    for p in PLACE_PATTERNS:
        matches = re.findall(p, text_lower)
        entities["place"].extend(matches)
    for p in NUMBER_PATTERNS:
        matches = re.findall(p, text_lower)
        entities["number"].extend(matches)
    for p in CONJUNCTION_PATTERNS:
        if re.search(p, text_lower):
            entities["conjunction"].append(p)
    return entities


def _has_intent_verb(text: str) -> bool:
    text_lower = text.lower()
    return any(v in text_lower for v in INTENT_VERBS)


def _is_presentation_request(text: str, has_prior_answer: bool) -> bool:
    """True if this chunk is a reformatting request with no new retrieval needed."""
    if not has_prior_answer:
        return False
    text_lower = text.lower()
    # Must contain a presentation verb
    if not any(v in text_lower for v in PRESENTATION_VERBS):
        return False
    # Must NOT introduce new entities (that would signal a new query)
    entities = _detect_entities(text)
    significant_entities = entities["place"] + entities["number"]
    return len(significant_entities) == 0


def _count_tokens(text: str) -> int:
    return len(text.split())


# ─────────────────────────────────────────────────────────────────────────────
# Per-session controller state
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ControllerSessionState:
    session_id: str
    last_decision: ControllerDecision = ControllerDecision.WAIT
    last_retrieval_t: float = 0.0
    last_query_text: str = ""
    last_query_embedding: Optional[np.ndarray] = None
    prev_buffer_embedding: Optional[np.ndarray] = None  # R2: drift signal
    retrieval_count: int = 0
    chunk_count: int = 0


# ─────────────────────────────────────────────────────────────────────────────
# Retrieval Controller
# ─────────────────────────────────────────────────────────────────────────────

class RetrievalController:
    """
    Decides WAIT / RETRIEVE / NO_RETRIEVAL for each incoming transcript chunk.

    Decision flow:
    1. Validate and debounce.
    2. Check for presentation-only request → NO_RETRIEVAL.
    3. Check entity/intent stability (rules fast-path) → RETRIEVE(provisional).
    4. Check for conjunction / multi-intent cue → RETRIEVE(multi_intent).
    5. [R2] Check embedding-drift signal (if dense index is ready) → RETRIEVE.
    6. LLM fallback for genuinely ambiguous cases (if enabled).
    7. Default → WAIT.
    """

    def __init__(self) -> None:
        cfg = get_settings()
        self._debounce_s = cfg.controller_debounce_s
        self._similarity_threshold = cfg.controller_similarity_threshold
        self._min_tokens = cfg.controller_min_tokens
        self._drift_threshold = cfg.controller_drift_threshold
        self._use_llm = cfg.controller_use_llm_fallback and not cfg.rules_only_mode
        self._session_states: dict[str, ControllerSessionState] = {}

    def _get_state(self, session_id: str) -> ControllerSessionState:
        if session_id not in self._session_states:
            self._session_states[session_id] = ControllerSessionState(session_id=session_id)
        return self._session_states[session_id]

    def reset_session(self, session_id: str) -> None:
        self._session_states.pop(session_id, None)

    def decide(
        self,
        chunk: ChunkEvent,
        buffer: str,
        has_prior_answer: bool,
        dense_index=None,
    ) -> ControllerOutput:
        """
        Make a retrieval decision for the current chunk + accumulated buffer.

        Args:
            chunk: The incoming transcript chunk.
            buffer: Full utterance buffer so far (all chunks concatenated).
            has_prior_answer: True if the session has a prior answer (enables NO_RETRIEVAL).
            dense_index: Optional dense index for embedding-drift computation (R2).

        Returns:
            ControllerOutput with decision, trigger, reason, confidence.
        """
        t_wall = time.time()
        state = self._get_state(chunk.session_id)
        state.chunk_count += 1
        cfg = get_settings()

        # ── 1. Presentation check FIRST (handles short reformatting requests) ─
        # Must check before min_tokens because reformatting requests are often short
        if has_prior_answer and _is_presentation_request(buffer, has_prior_answer):
            return self._make_output(
                chunk, ControllerDecision.NO_RETRIEVAL,
                TriggerType.PRESENTATION_RESTRUCTURE,
                ReasonCode.PRESENTATION_RESTRUCTURE, 0.95, t_wall,
                entity_count=0,
            )

        # ── 2. Minimum token guard ─────────────────────────────────────────
        token_count = _count_tokens(buffer)
        if token_count < self._min_tokens:
            return self._make_output(
                chunk, ControllerDecision.WAIT, None,
                ReasonCode.INTENT_INCOMPLETE, 0.9, t_wall,
                entity_count=0,
            )

        # ── 2. Debounce guard ────────────────────────────────────────────
        elapsed_since_last = t_wall - state.last_retrieval_t
        if elapsed_since_last < self._debounce_s and state.retrieval_count > 0:
            return self._make_output(
                chunk, ControllerDecision.WAIT, None,
                ReasonCode.DEBOUNCE, 1.0, t_wall,
                entity_count=0,
            )

        # ── 3. Presentation-only detector ─────────────────────────────────
        if _is_presentation_request(buffer, has_prior_answer):
            return self._make_output(
                chunk, ControllerDecision.NO_RETRIEVAL,
                TriggerType.PRESENTATION_RESTRUCTURE,
                ReasonCode.PRESENTATION_RESTRUCTURE, 0.95, t_wall,
                entity_count=0,
            )

        # ── 4. Entity detection ───────────────────────────────────────────
        entities = _detect_entities(buffer)
        entity_count = len(entities["place"]) + len(entities["number"])
        has_intent = _has_intent_verb(buffer)

        # ── 5. Similarity guard (avoid re-retrieving for near-identical query) ──
        query_text = buffer.strip()
        if state.last_query_text and self._query_similarity(query_text, state.last_query_text) > self._similarity_threshold:
            return self._make_output(
                chunk, ControllerDecision.WAIT, None,
                ReasonCode.SIMILARITY_GUARD, 1.0, t_wall,
                entity_count=entity_count,
            )

        # ── 6. Multi-intent detection (conjunction cue) ───────────────────
        has_multi_intent_cue = bool(entities["conjunction"]) and entity_count >= 2 and has_intent
        if has_multi_intent_cue and state.retrieval_count > 0:
            # A conjunction after we've already fired provisional → multi_intent
            state.last_retrieval_t = t_wall
            state.retrieval_count += 1
            state.last_query_text = query_text
            return self._make_output(
                chunk, ControllerDecision.RETRIEVE,
                TriggerType.MULTI_INTENT,
                ReasonCode.MULTI_INTENT, 0.88, t_wall,
                entity_count=entity_count,
            )

        # ── 7. Provisional stable: entity + intent + enough tokens ─────────
        # Require min_tokens (already passed step 1), entity, and intent verb
        if entity_count >= 1 and has_intent:
            state.last_retrieval_t = t_wall
            state.retrieval_count += 1
            state.last_query_text = query_text
            return self._make_output(
                chunk, ControllerDecision.RETRIEVE,
                TriggerType.PROVISIONAL,
                ReasonCode.PROVISIONAL_STABLE, 0.82, t_wall,
                entity_count=entity_count,
            )

        # ── 8. R2: Embedding-drift signal ─────────────────────────────────
        if dense_index is not None:
            drift_score = self._compute_drift(buffer, state, dense_index)
            if drift_score is not None and drift_score < self._drift_threshold and has_intent:
                state.last_retrieval_t = t_wall
                state.retrieval_count += 1
                state.last_query_text = query_text
                return self._make_output(
                    chunk, ControllerDecision.RETRIEVE,
                    TriggerType.PROVISIONAL,
                    ReasonCode.PROVISIONAL_STABLE, 0.75, t_wall,
                    entity_count=entity_count,
                    embedding_drift=float(drift_score),
                )

        # ── 9. Default: WAIT ──────────────────────────────────────────────
        if entity_count == 0:
            reason = ReasonCode.NO_ENTITY
        else:
            reason = ReasonCode.INTENT_INCOMPLETE

        return self._make_output(
            chunk, ControllerDecision.WAIT, None,
            reason, 0.7, t_wall,
            entity_count=entity_count,
        )

    def decide_late_constraint(
        self,
        chunk: ChunkEvent,
        buffer: str,
    ) -> ControllerOutput:
        """
        Special decision for utterances that arrive after a prior answer —
        checks if this is a late constraint (refine) or a new topic.
        """
        t_wall = time.time()
        state = self._get_state(chunk.session_id)
        entities = _detect_entities(buffer)
        entity_count = len(entities["place"]) + len(entities["number"])
        has_intent = _has_intent_verb(buffer)

        if _is_presentation_request(buffer, has_prior_answer=True):
            return self._make_output(
                chunk, ControllerDecision.NO_RETRIEVAL,
                TriggerType.PRESENTATION_RESTRUCTURE,
                ReasonCode.PRESENTATION_RESTRUCTURE, 0.95, t_wall,
                entity_count=entity_count,
            )

        # If it introduces new context (entities) related to prior query → late_constraint
        if entity_count >= 1 or has_intent:
            state.last_retrieval_t = t_wall
            state.retrieval_count += 1
            state.last_query_text = buffer.strip()
            return self._make_output(
                chunk, ControllerDecision.RETRIEVE,
                TriggerType.LATE_CONSTRAINT,
                ReasonCode.LATE_CONSTRAINT, 0.85, t_wall,
                entity_count=entity_count,
            )

        return self._make_output(
            chunk, ControllerDecision.WAIT, None,
            ReasonCode.INTENT_INCOMPLETE, 0.7, t_wall,
            entity_count=entity_count,
        )

    def _compute_drift(
        self,
        buffer: str,
        state: ControllerSessionState,
        dense_index,
    ) -> Optional[float]:
        """
        R2: Compute embedding drift between previous and current buffer.
        Returns cosine *distance* (1 - similarity). Low value = semantic settledness.
        """
        try:
            curr_emb = dense_index.embed_query(buffer)
            if state.prev_buffer_embedding is not None:
                # Cosine distance = 1 - dot(a, b) for normalised vectors
                drift = float(1.0 - np.dot(curr_emb, state.prev_buffer_embedding))
            else:
                drift = None
            state.prev_buffer_embedding = curr_emb
            return drift
        except Exception:
            return None

    def _query_similarity(self, q1: str, q2: str) -> float:
        """
        Lightweight lexical similarity between two query strings.
        Jaccard on word sets — fast, no embedding needed.
        """
        w1 = set(q1.lower().split())
        w2 = set(q2.lower().split())
        if not w1 or not w2:
            return 0.0
        return len(w1 & w2) / len(w1 | w2)

    def _make_output(
        self,
        chunk: ChunkEvent,
        decision: ControllerDecision,
        trigger: Optional[TriggerType],
        reason: ReasonCode,
        confidence: float,
        t_wall: float,
        entity_count: int = 0,
        embedding_drift: Optional[float] = None,
    ) -> ControllerOutput:
        return ControllerOutput(
            session_id=chunk.session_id,
            chunk_id=chunk.chunk_id,
            decision=decision,
            trigger=trigger,
            reason=reason,
            confidence=confidence,
            t_wall_s=t_wall,
            entity_count=entity_count,
            embedding_drift=embedding_drift,
        )


# ── Module-level singleton ──────────────────────────────────────────────────
_controller: Optional[RetrievalController] = None


def get_controller() -> RetrievalController:
    global _controller
    if _controller is None:
        _controller = RetrievalController()
    return _controller
