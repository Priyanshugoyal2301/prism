"""
synthesis/verifier.py — Citation verifier and grounding checker.

Gate G4: >=85% citation support, ZERO hallucinated Doc IDs.

Two-level verification:
1. ID existence check: chunk_id must be in the session's evidence pool.
2. Semantic support check (R6 spike): cosine similarity between claim text
   embedding and cited chunk text embedding must exceed a threshold.

Claims that fail verification are either:
- Removed (if no evidence supports them at all)
- Flagged as uncertain (if evidence is partial)

This is the security gate: a prompt-injected fake Doc_999 always fails step 1.
"""

from __future__ import annotations

import re
from typing import Optional

import numpy as np

from app.config import get_settings
from app.schemas import Chunk, Citation, Claim
from app.telemetry import get_telemetry, TelemetryEventType


# ─────────────────────────────────────────────────────────────────────────────
# Lexical support score (fast, no model needed)
# ─────────────────────────────────────────────────────────────────────────────

def _lexical_overlap(text1: str, text2: str) -> float:
    """Normalised word-level Jaccard between claim and chunk text."""
    w1 = set(re.findall(r"\b\w+\b", text1.lower()))
    w2 = set(re.findall(r"\b\w+\b", text2.lower()))
    if not w1 or not w2:
        return 0.0
    return len(w1 & w2) / len(w1 | w2)


def _embedding_support(
    claim_text: str,
    chunk_text: str,
    dense_index=None,
) -> Optional[float]:
    """
    R6 spike: cosine similarity between claim and chunk embeddings.
    Returns None if the dense index is not available.
    """
    if dense_index is None:
        return None
    try:
        c_emb = dense_index.embed_query(claim_text)
        k_emb = dense_index.embed_query(chunk_text)
        return float(np.dot(c_emb, k_emb))
    except Exception:
        return None


def compute_support_score(
    claim_text: str,
    chunk: Chunk,
    dense_index=None,
) -> float:
    """
    Combined support score: max(lexical, embedding) if embedding is available,
    otherwise lexical only.
    """
    lexical = _lexical_overlap(claim_text, chunk.text)
    emb = _embedding_support(claim_text, chunk.text, dense_index)
    if emb is not None:
        # Weighted combination — embedding is more reliable
        return 0.3 * lexical + 0.7 * max(emb, 0.0)
    return lexical


# ─────────────────────────────────────────────────────────────────────────────
# Citation verifier
# ─────────────────────────────────────────────────────────────────────────────

class CitationVerifier:
    """
    Verifies claims and their citations against the session evidence pool.
    Modifies citations in-place: marks them verified=False if they fail.
    Removes or flags claims that have no verified citations.
    """

    def __init__(self) -> None:
        cfg = get_settings()
        self._cosine_threshold = cfg.grounding_cosine_threshold
        self._lexical_threshold = cfg.grounding_lexical_threshold

    async def verify_claims(
        self,
        session_id: str,
        claims: list[Claim],
        evidence_pool: dict[str, Chunk],
        dense_index=None,
    ) -> tuple[list[Claim], list[Claim], str]:
        """
        Verify all claims.

        Returns:
            (verified_claims, removed_claims, uncertainty_text)

        verified_claims: claims with at least one verified citation (may have is_uncertain=True
                         if support is partial).
        removed_claims: claims with NO valid citations — stripped from the answer.
        uncertainty_text: human-readable summary of what could not be verified.
        """
        tel = get_telemetry()
        verified: list[Claim] = []
        removed: list[Claim] = []
        uncertain_topics: list[str] = []

        for claim in claims:
            verified_citations, violations = await self._verify_claim(
                claim, evidence_pool, dense_index
            )

            # Log each citation check
            for cit_id, ok, score, reason in violations:
                await tel.emit(
                    TelemetryEventType.CITATION_CHECK,
                    session_id,
                    claim_id=claim.claim_id,
                    chunk_id=cit_id,
                    verified=ok,
                    support_score=score,
                    reason=reason,
                )

            if verified_citations:
                claim.citations = verified_citations
                # If some citations were removed, flag as partial
                if len(verified_citations) < len(claim.citations):
                    claim.is_uncertain = True
                verified.append(claim)
            else:
                # No valid citations — remove or flag
                removed.append(claim)
                # Build uncertainty text
                short_claim = claim.text[:80] + ("..." if len(claim.text) > 80 else "")
                uncertain_topics.append(short_claim)

        uncertainty = ""
        if uncertain_topics:
            if len(uncertain_topics) == 1:
                uncertainty = (
                    f"The following could not be verified from the retrieved corpus: "
                    f"{uncertain_topics[0]}"
                )
            else:
                uncertainty = (
                    f"The following {len(uncertain_topics)} claims could not be verified "
                    f"from the retrieved corpus: " + "; ".join(uncertain_topics[:3])
                )

        return verified, removed, uncertainty

    async def _verify_claim(
        self,
        claim: Claim,
        evidence_pool: dict[str, Chunk],
        dense_index=None,
    ) -> tuple[list[Citation], list[tuple[str, bool, float, str]]]:
        """
        Verify all citations in a single claim.
        Returns (valid_citations, [(chunk_id, ok, score, reason)])
        """
        valid: list[Citation] = []
        violations: list[tuple[str, bool, float, str]] = []

        for citation in claim.citations:
            chunk_id = citation.chunk_id

            # Step 1: ID existence check (hard gate — no hallucinated IDs)
            chunk = evidence_pool.get(chunk_id)
            if chunk is None:
                citation.verified = False
                violations.append((chunk_id, False, 0.0, "id_not_in_pool"))
                continue

            # Step 2: Semantic support check (R6)
            score = compute_support_score(claim.text, chunk, dense_index)
            threshold = self._cosine_threshold if dense_index else self._lexical_threshold

            if score >= threshold:
                citation.verified = True
                citation.support_score = score
                valid.append(citation)
                violations.append((chunk_id, True, score, "supported"))
            else:
                citation.verified = False
                citation.support_score = score
                violations.append((chunk_id, False, score, "below_threshold"))

        return valid, violations
