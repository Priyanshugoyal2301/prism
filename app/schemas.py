"""
schemas.py — All Pydantic v2 models for the Streaming Live RAG Engine.
These models define the full API contract and internal data structures.
TODO(SCHEMA): Align field names with the released test-suite schema when available.
"""

from __future__ import annotations

import uuid
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


# ─────────────────────────────────────────────────────────────────────────────
# Enumerations
# ─────────────────────────────────────────────────────────────────────────────

class ControllerDecision(str, Enum):
    WAIT = "WAIT"
    RETRIEVE = "RETRIEVE"
    NO_RETRIEVAL = "NO_RETRIEVAL"


class TriggerType(str, Enum):
    PROVISIONAL = "provisional"
    MULTI_INTENT = "multi_intent"
    LATE_CONSTRAINT = "late_constraint"
    PRESENTATION_RESTRUCTURE = "presentation_restructure"


class ReasonCode(str, Enum):
    INTENT_INCOMPLETE = "intent_incomplete"
    PROVISIONAL_STABLE = "provisional_stable"
    MULTI_INTENT = "multi_intent"
    PRESENTATION_RESTRUCTURE = "presentation_restructure"
    LATE_CONSTRAINT = "late_constraint"
    DEBOUNCE = "debounce"
    SIMILARITY_GUARD = "similarity_guard"
    NO_ENTITY = "no_entity"


class PatchOp(str, Enum):
    ADD = "add"
    UPDATE = "update"
    RETAIN = "retain"


class TelemetryEventType(str, Enum):
    CHUNK_RECEIVED = "chunk_received"
    CONTROLLER_DECISION = "controller_decision"
    RETRIEVAL_STARTED = "retrieval_started"
    RETRIEVAL_DONE = "retrieval_done"
    DECOMPOSITION = "decomposition"
    FUSION_DONE = "fusion_done"
    SYNTHESIS_FIRST_TOKEN = "synthesis_first_token"
    SYNTHESIS_DONE = "synthesis_done"
    CITATION_CHECK = "citation_check"
    ANSWER_VERSION_BUMP = "answer_version_bump"
    SUPPRESSED_RETRIEVAL = "suppressed_retrieval"
    CACHE_HIT = "cache_hit"
    ERROR = "error"


# ─────────────────────────────────────────────────────────────────────────────
# Ingestion — Transcript Stream Events
# ─────────────────────────────────────────────────────────────────────────────

class ChunkEvent(BaseModel):
    """A single transcript chunk from the stream."""
    session_id: str = Field(..., description="UUID4 identifying the session")
    chunk_id: str = Field(..., description="Monotonic chunk identifier within the session")
    text: str = Field(..., max_length=2000, description="Raw transcript text of this chunk")
    t_start_s: float = Field(..., ge=0.0, description="Timestamp (seconds) when this chunk was spoken")
    is_final: bool = Field(default=False, description="True when the utterance has ended")

    @field_validator("session_id")
    @classmethod
    def validate_session_id(cls, v: str) -> str:
        try:
            uuid.UUID(v)
        except ValueError:
            raise ValueError("session_id must be a valid UUID4")
        return v

    @field_validator("text")
    @classmethod
    def text_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Chunk text must not be empty")
        return v


class UtteranceEndEvent(BaseModel):
    """Signals the end of the current utterance."""
    session_id: str
    t_end_s: float = Field(..., ge=0.0)


class ResetEvent(BaseModel):
    """Resets the session state."""
    session_id: str


# ─────────────────────────────────────────────────────────────────────────────
# Corpus & Retrieval
# ─────────────────────────────────────────────────────────────────────────────

class Chunk(BaseModel):
    """A single retrievable document chunk with stable citation ID."""
    chunk_id: str = Field(..., description="Stable ID: 'Doc_12 §2.3' style")
    doc_id: str
    section: str
    text: str
    token_count: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalResult(BaseModel):
    """Single chunk with its retrieval scores."""
    chunk: Chunk
    bm25_score: float = 0.0
    dense_score: float = 0.0
    rrf_score: float = 0.0
    sub_query_id: str = ""


class FusedResults(BaseModel):
    """RRF-fused and deduplicated retrieval results for one or more sub-queries."""
    sub_query_id: str
    query: str
    hits: list[RetrievalResult]
    retrieval_latency_ms: float = 0.0
    cache_hit: bool = False


# ─────────────────────────────────────────────────────────────────────────────
# Controller
# ─────────────────────────────────────────────────────────────────────────────

class ControllerOutput(BaseModel):
    """Output of the retrieval controller for one chunk."""
    session_id: str
    chunk_id: str
    decision: ControllerDecision
    trigger: Optional[TriggerType] = None
    reason: ReasonCode
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    t_wall_s: float = 0.0
    # Research spike: embedding drift signal
    embedding_drift: Optional[float] = None
    entity_count: int = 0


# ─────────────────────────────────────────────────────────────────────────────
# Decomposition
# ─────────────────────────────────────────────────────────────────────────────

class SubQuery(BaseModel):
    """One orthogonal sub-query derived from an utterance."""
    sub_query_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    text: str
    carried_context: dict[str, str] = Field(
        default_factory=dict,
        description="Context entities carried into this sub-query, e.g. {'city': 'Pune', 'capacity': '30'}"
    )
    status: str = "queued"  # queued | running | done | no_evidence


class DecompositionOutput(BaseModel):
    """Output of the multi-intent decomposer."""
    session_id: str
    sub_queries: list[SubQuery]
    is_refinement: bool = False
    affected_sub_query_ids: list[str] = Field(
        default_factory=list,
        description="Sub-query IDs affected by a late constraint (refinement mode)"
    )
    # Research spike metrics
    orthogonality_scores: list[float] = Field(
        default_factory=list,
        description="Pairwise cosine similarities between sub-queries (post-dedupe)"
    )
    decomposer_mode: str = "llm"  # llm | rules | rules_fallback


# ─────────────────────────────────────────────────────────────────────────────
# Claims & Citations
# ─────────────────────────────────────────────────────────────────────────────

class Citation(BaseModel):
    """A single verified citation."""
    chunk_id: str  # e.g. "Doc_12 §2"
    doc_id: str
    section: str
    support_score: float = Field(
        default=1.0, ge=0.0, le=1.0,
        description="Semantic support score between claim text and cited chunk"
    )
    verified: bool = True


class Claim(BaseModel):
    """One factual claim in the answer, linked to citations and a sub-query."""
    claim_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    text: str
    sub_query_id: str = ""
    citations: list[Citation] = Field(default_factory=list)
    is_uncertain: bool = False
    patch_op: Optional[PatchOp] = None  # Set during delta patching


# ─────────────────────────────────────────────────────────────────────────────
# Session State
# ─────────────────────────────────────────────────────────────────────────────

class AnswerVersion(BaseModel):
    """One versioned snapshot of the answer."""
    version: int
    claims: list[Claim]
    citations: list[str]  # flat deduplicated list of chunk_ids
    uncertainty: str = ""
    t_created_s: float = 0.0


class AnswerDiffOp(BaseModel):
    """One operation in an answer diff."""
    op: PatchOp
    claim_id: str
    text_before: Optional[str] = None
    text_after: Optional[str] = None
    citations_added: list[str] = Field(default_factory=list)


class AnswerDiff(BaseModel):
    """Diff between two answer versions."""
    version_from: int
    version_to: int
    ops: list[AnswerDiffOp]
    summary: str = ""


class SessionState(BaseModel):
    """Full ephemeral session state."""
    session_id: str
    created_at: float
    ttl_s: float
    utterance_buffer: str = ""
    answer_version: int = 0
    answer_versions: list[AnswerVersion] = Field(default_factory=list)
    sub_queries: dict[str, SubQuery] = Field(default_factory=dict)
    evidence_pool: dict[str, Chunk] = Field(default_factory=dict)  # chunk_id -> Chunk
    history: list[dict] = Field(default_factory=list)
    last_retrieval_query: Optional[str] = None
    last_retrieval_t: float = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Turn Record — The main output contract (FR-8)
# TODO(SCHEMA): Align exact field names with released test-suite schema
# ─────────────────────────────────────────────────────────────────────────────

class RetrievalEvent(BaseModel):
    """A single retrieval event within a turn."""
    timestamp_s: float
    query: str
    trigger: str
    sub_query_id: str = ""
    n_hits: int = 0
    latency_ms: float = 0.0
    cache_hit: bool = False


class TurnRecord(BaseModel):
    """
    Structured output for one complete turn.
    This is the primary output contract checked by the evaluation suite.
    TODO(SCHEMA): Verify field names against released test-suite schema.
    """
    session_id: str
    turn_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    retrieval_events: list[RetrievalEvent] = Field(default_factory=list)
    sub_queries: list[str] = Field(default_factory=list)
    answer: str = ""
    answer_version: int = 1
    citations: list[str] = Field(default_factory=list)  # ["Doc_12 §2", ...]
    uncertainty: str = ""
    answer_diff: Optional[AnswerDiff] = None
    # Extended telemetry fields
    ttft_ms: float = 0.0
    end_to_end_ms: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    controller_decisions: list[ControllerOutput] = Field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Telemetry Events
# ─────────────────────────────────────────────────────────────────────────────

class TelemetryEvent(BaseModel):
    """Base class for all telemetry events emitted to JSONL."""
    event_type: TelemetryEventType
    ts: float  # Unix timestamp
    session_id: str
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:12])
    payload: dict[str, Any] = Field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# API Request/Response models
# ─────────────────────────────────────────────────────────────────────────────

class ReplayRequest(BaseModel):
    """Request to replay a scenario from a file or inline chunks."""
    scenario_name: Optional[str] = None  # name of bundled scenario
    chunks: Optional[list[ChunkEvent]] = None  # or inline chunks
    speed_multiplier: float = Field(default=1.0, gt=0.0, le=100.0)
    session_id: Optional[str] = None  # if None, one is generated


class ReplayResponse(BaseModel):
    """Full event log from a replay run."""
    session_id: str
    turn_record: TurnRecord
    telemetry_events: list[TelemetryEvent]
    duration_ms: float


class HealthResponse(BaseModel):
    status: str = "ok"
    rules_only_mode: bool = False
    corpus_loaded: bool = False
    index_chunks: int = 0
    version: str = "0.1.0"


class MetricsResponse(BaseModel):
    """Aggregate metrics snapshot."""
    total_turns: int = 0
    total_retrievals: int = 0
    suppressed_retrievals: int = 0
    avg_ttft_ms: float = 0.0
    avg_e2e_ms: float = 0.0
    avg_sub_queries: float = 0.0
    cache_hit_rate: float = 0.0
    citation_violation_count: int = 0
    avg_cost_usd: float = 0.0
    g2_early_retrieval_rate: float = 0.0
    g3_multi_intent_rate: float = 0.0
    g4_citation_support_rate: float = 0.0
    g6_trace_coverage: float = 1.0


# ─────────────────────────────────────────────────────────────────────────────
# WebSocket message envelopes (server → client)
# ─────────────────────────────────────────────────────────────────────────────

class WSMessage(BaseModel):
    """Generic WebSocket message envelope."""
    type: str
    session_id: str
    payload: dict[str, Any] = Field(default_factory=dict)
