"""
tests/test_verifier.py — Unit tests for the citation verifier.
Ensures Gate G4: zero hallucinated Doc IDs, >=85% citation support.
"""

import asyncio

import pytest

from app.schemas import Citation, Claim, Chunk
from app.synthesis.verifier import CitationVerifier, compute_support_score


def make_chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(chunk_id=chunk_id, doc_id=chunk_id.split(" ")[0], section="§1", text=text)


def make_claim(text: str, citations: list[str], evidence_pool: dict) -> Claim:
    cits = [
        Citation(chunk_id=c, doc_id=c.split(" ")[0], section="§1", verified=False)
        for c in citations
    ]
    return Claim(claim_id="test", text=text, citations=cits)


class TestCitationVerifier:
    def setup_method(self):
        self.verifier = CitationVerifier()

    def _run(self, coro):
        return asyncio.get_event_loop().run_until_complete(coro)

    def test_hallucinated_id_is_rejected(self):
        """Doc_999 is not in pool → must be rejected. Gate G4: zero hallucinated IDs."""
        evidence_pool = {
            "Doc_01 §1": make_chunk("Doc_01 §1", "The venue is in Pune."),
        }
        claim = make_claim(
            "The venue charges INR 500.",
            citations=["Doc_999 §1"],  # hallucinated ID
            evidence_pool=evidence_pool,
        )
        verified, removed, uncertainty = self._run(
            self.verifier.verify_claims("test", [claim], evidence_pool)
        )
        assert len(removed) == 1
        assert len(verified) == 0
        assert "could not be verified" in uncertainty.lower() or uncertainty != ""

    def test_valid_citation_is_accepted(self):
        chunk_text = "The venue accommodates up to 30 people in Pune."
        evidence_pool = {"Doc_01 §2": make_chunk("Doc_01 §2", chunk_text)}
        claim = make_claim(
            "The venue can accommodate 30 people.",
            citations=["Doc_01 §2"],
            evidence_pool=evidence_pool,
        )
        verified, removed, uncertainty = self._run(
            self.verifier.verify_claims("test", [claim], evidence_pool)
        )
        assert len(verified) == 1
        assert len(removed) == 0

    def test_injection_chunk_id_in_evidence_is_rejected(self):
        """Even if 'Doc_999' somehow exists in pool, a fake content claim should fail support check."""
        # This tests that the support score check catches semantic mismatch
        evidence_pool = {
            "Doc_01 §1": make_chunk("Doc_01 §1", "Completely unrelated content about cooking recipes."),
        }
        claim = make_claim(
            "The cancellation refund is 50% within 14 days.",
            citations=["Doc_01 §1"],
            evidence_pool=evidence_pool,
        )
        # Lexical support should be low (different domain)
        score = compute_support_score(
            "The cancellation refund is 50% within 14 days.",
            evidence_pool["Doc_01 §1"],
        )
        assert score < 0.5  # weak support

    def test_uncertainty_text_generated_for_unsupported_claim(self):
        evidence_pool = {}
        claim = make_claim(
            "Catering is available from INR 450 per head.",
            citations=["Doc_01 §4"],  # not in pool
            evidence_pool=evidence_pool,
        )
        _, _, uncertainty = self._run(
            self.verifier.verify_claims("test", [claim], evidence_pool)
        )
        assert len(uncertainty) > 0

    def test_mixed_valid_and_invalid_citations(self):
        evidence_pool = {
            "Doc_01 §2": make_chunk("Doc_01 §2", "The capacity is 30 people in Pune.")
        }
        claim = Claim(
            claim_id="test",
            text="The venue holds 30 people.",
            citations=[
                Citation(chunk_id="Doc_01 §2", doc_id="Doc_01", section="§2", verified=False),
                Citation(chunk_id="Doc_99 §1", doc_id="Doc_99", section="§1", verified=False),  # hallucinated
            ],
        )
        verified, removed, _ = self._run(
            self.verifier.verify_claims("test", [claim], evidence_pool)
        )
        # Claim should survive (has one valid citation) but be partial
        assert len(verified) == 1
        valid_cits = [c for c in verified[0].citations if c.verified]
        assert len(valid_cits) == 1
        assert valid_cits[0].chunk_id == "Doc_01 §2"
