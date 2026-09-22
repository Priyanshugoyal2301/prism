"""
decomposer.py — Multi-intent utterance decomposer.

Converts the current utterance buffer into 1..N orthogonal sub-queries,
carrying shared context (city, capacity, date) into each sub-query.

Two modes:
1. LLM mode: structured JSON output with retry.
2. Rule-based fallback: conjunction/comma split + context resolution (works with no LLM key).

Research spike R5: orthogonality score — pairwise cosine similarity between
sub-query embeddings; pairs above threshold are merged to prevent over-fragmentation.
"""

from __future__ import annotations

import json
import re
import uuid
from typing import Optional

import numpy as np

from app.config import get_settings
from app.llm import LLMClient, RulesOnlyModeError
from app.schemas import DecompositionOutput, SubQuery
from app.telemetry import get_telemetry, TelemetryEventType


# ─────────────────────────────────────────────────────────────────────────────
# LLM decomposer prompt
# ─────────────────────────────────────────────────────────────────────────────

DECOMPOSER_SYSTEM_PROMPT = """You are a query decomposition assistant for a RAG system.
Your job: given a user utterance, produce 1-4 independent sub-queries suitable for document retrieval.

Rules:
1. Each sub-query must be self-contained and include all relevant context (location, quantity, date, etc.).
2. Sub-queries must be ORTHOGONAL — they must not ask for the same information.
3. A simple single-topic utterance must yield exactly 1 sub-query.
4. Do NOT invent information not present in the utterance.
5. For refinement: identify which sub-queries are affected by the new constraint.
6. Output ONLY valid JSON matching this schema:

{
  "sub_queries": [
    {"text": "...", "carried_context": {"city": "Pune", "capacity": "30"}}
  ],
  "is_refinement": false,
  "affected_sub_query_ids": []
}"""

DECOMPOSER_USER_TEMPLATE = """Utterance: {utterance}

Prior sub-queries (if refinement): {prior_sub_queries}

Decompose this utterance into orthogonal sub-queries. Carry all context into each sub-query."""


# ─────────────────────────────────────────────────────────────────────────────
# Rule-based fallback decomposer
# ─────────────────────────────────────────────────────────────────────────────

CONJUNCTIONS = re.compile(
    r"\b(?:and|also|plus|additionally|furthermore|moreover|besides|as well as)\b",
    re.IGNORECASE,
)

CONTEXT_PATTERNS = {
    "city": re.compile(
        r"\b(pune|mumbai|delhi|bangalore|hyderabad|chennai|kolkata|india|dubai|london|new york|paris)\b",
        re.IGNORECASE,
    ),
    "capacity": re.compile(
        r"\b(\d+)\s*(?:people|persons?|attendees|guests|pax)\b", re.IGNORECASE
    ),
    "date": re.compile(
        r"\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}|\b\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?\b",
        re.IGNORECASE,
    ),
}


def _extract_context(text: str) -> dict[str, str]:
    """Extract shared context entities from the full utterance."""
    ctx: dict[str, str] = {}
    for key, pat in CONTEXT_PATTERNS.items():
        m = pat.search(text)
        if m:
            ctx[key] = m.group(0)
    return ctx


def _rule_based_split(utterance: str, max_queries: int) -> list[str]:
    """
    Split utterance on conjunction boundaries.
    Falls back to the full utterance as a single query if no conjunction found.
    """
    parts = CONJUNCTIONS.split(utterance)
    parts = [p.strip() for p in parts if p.strip() and len(p.split()) >= 3]
    if len(parts) <= 1:
        return [utterance.strip()]
    return parts[:max_queries]


def _inject_context(text: str, ctx: dict[str, str]) -> str:
    """Inject carried context into a sub-query text if not already present."""
    result = text
    for key, val in ctx.items():
        if val.lower() not in text.lower():
            result = f"{result} (context: {key}={val})"
    return result


def rule_based_decompose(
    utterance: str,
    prior_sub_queries: Optional[list[str]] = None,
    max_queries: int = 4,
) -> DecompositionOutput:
    """
    Rule-based decomposition: conjunction split + context injection.
    Used as fallback when no LLM key is present, or LLM call fails.
    """
    ctx = _extract_context(utterance)
    raw_parts = _rule_based_split(utterance, max_queries)

    sub_queries: list[SubQuery] = []
    for part in raw_parts:
        text_with_ctx = _inject_context(part, ctx)
        sub_queries.append(SubQuery(
            sub_query_id=str(uuid.uuid4())[:8],
            text=text_with_ctx,
            carried_context=ctx,
        ))

    # Detect refinement: short utterance that references a prior answer
    is_refinement = False
    affected: list[str] = []
    if prior_sub_queries and len(utterance.split()) <= 20:
        # Simple heuristic: if it's short and there are prior queries, assume refinement
        is_refinement = True
        affected = []  # Rule-based can't identify specific affected queries

    return DecompositionOutput(
        session_id="",  # set by caller
        sub_queries=sub_queries,
        is_refinement=is_refinement,
        affected_sub_query_ids=affected,
        decomposer_mode="rules",
    )


# ─────────────────────────────────────────────────────────────────────────────
# R5: Orthogonality guard
# ─────────────────────────────────────────────────────────────────────────────

def _compute_orthogonality(
    sub_queries: list[SubQuery],
    dense_index,
) -> tuple[list[SubQuery], list[float]]:
    """
    R5 spike: Compute pairwise cosine similarity between sub-query embeddings.
    Merge pairs above the orthogonality threshold.
    Returns (deduped_sub_queries, pairwise_similarity_scores).
    """
    if len(sub_queries) <= 1 or dense_index is None:
        return sub_queries, []

    cfg = get_settings()
    threshold = cfg.decomposer_orthogonality_threshold

    try:
        embeddings = np.array([
            dense_index.embed_query(sq.text) for sq in sub_queries
        ])  # shape: (N, dim)

        # Pairwise cosine (vectors already normalised)
        sim_matrix = embeddings @ embeddings.T
        scores: list[float] = []
        merge_pairs: set[int] = set()

        for i in range(len(sub_queries)):
            for j in range(i + 1, len(sub_queries)):
                sim = float(sim_matrix[i, j])
                scores.append(sim)
                if sim > threshold:
                    # Merge j into i (keep i, discard j)
                    merge_pairs.add(j)

        deduped = [sq for idx, sq in enumerate(sub_queries) if idx not in merge_pairs]
        return deduped, scores

    except Exception:
        return sub_queries, []


# ─────────────────────────────────────────────────────────────────────────────
# Main Decomposer
# ─────────────────────────────────────────────────────────────────────────────

class MultiIntentDecomposer:
    """
    Multi-intent decomposer with LLM primary path and rule-based fallback.
    Includes R5 orthogonality guard to prevent over-fragmentation.
    """

    def __init__(self, llm_client: LLMClient) -> None:
        self._llm = llm_client
        cfg = get_settings()
        self._max_queries = cfg.decomposer_max_subqueries

    async def decompose(
        self,
        session_id: str,
        utterance: str,
        prior_sub_queries: Optional[list[str]] = None,
        dense_index=None,
    ) -> DecompositionOutput:
        """
        Decompose the utterance into sub-queries.
        Falls back to rule-based if LLM is unavailable or returns invalid output.
        """
        tel = get_telemetry()

        result = await self._llm_decompose(session_id, utterance, prior_sub_queries)
        if result is None:
            result = rule_based_decompose(utterance, prior_sub_queries, self._max_queries)

        result.session_id = session_id

        # Cap sub-queries
        result.sub_queries = result.sub_queries[:self._max_queries]

        # R5: orthogonality guard
        deduped, orth_scores = _compute_orthogonality(result.sub_queries, dense_index)
        if len(deduped) < len(result.sub_queries):
            result.sub_queries = deduped
        result.orthogonality_scores = orth_scores

        await tel.emit(
            TelemetryEventType.DECOMPOSITION,
            session_id,
            sub_query_count=len(result.sub_queries),
            sub_queries=[sq.text for sq in result.sub_queries],
            is_refinement=result.is_refinement,
            mode=result.decomposer_mode,
            orthogonality_scores=orth_scores,
        )

        return result

    async def _llm_decompose(
        self,
        session_id: str,
        utterance: str,
        prior_sub_queries: Optional[list[str]],
    ) -> Optional[DecompositionOutput]:
        """Attempt LLM decomposition; return None on any failure."""
        cfg = get_settings()
        if cfg.rules_only_mode:
            return None

        try:
            messages = [
                {"role": "system", "content": DECOMPOSER_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": DECOMPOSER_USER_TEMPLATE.format(
                        utterance=utterance,
                        prior_sub_queries=json.dumps(prior_sub_queries or []),
                    ),
                },
            ]
            data = await self._llm.complete_json(
                session_id=session_id,
                messages=messages,
                model="fast",
                max_tokens=512,
            )

            raw_queries = data.get("sub_queries", [])
            if not isinstance(raw_queries, list) or not raw_queries:
                return None

            sub_queries: list[SubQuery] = []
            for item in raw_queries[:self._max_queries]:
                if isinstance(item, dict):
                    text = item.get("text", "")
                    ctx = item.get("carried_context", {})
                elif isinstance(item, str):
                    text = item
                    ctx = _extract_context(utterance)
                else:
                    continue
                if text:
                    sub_queries.append(SubQuery(
                        sub_query_id=str(uuid.uuid4())[:8],
                        text=text,
                        carried_context=ctx,
                    ))

            if not sub_queries:
                return None

            return DecompositionOutput(
                session_id=session_id,
                sub_queries=sub_queries,
                is_refinement=bool(data.get("is_refinement", False)),
                affected_sub_query_ids=data.get("affected_sub_query_ids", []),
                decomposer_mode="llm",
            )

        except RulesOnlyModeError:
            return None
        except Exception as e:
            # Log and fall through to rule-based
            tel = get_telemetry()
            await tel.emit(
                TelemetryEventType.ERROR,
                session_id,
                stage="decomposer",
                error=str(e),
                fallback="rules",
            )
            return None
