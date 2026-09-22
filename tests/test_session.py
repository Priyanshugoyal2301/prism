"""
tests/test_session.py — Unit tests for session state model, claim graph, and delta patching.
Tests the R3 spike: claim-level answer graph for surgical delta patching.
"""

import pytest

from app.schemas import Citation, Claim, Chunk, PatchOp
from app.synthesis.session import Session, SessionStore


def make_claim(claim_id: str, text: str, sub_query_id: str = "sq1", citations: list = None) -> Claim:
    if citations is None:
        citations = [Citation(chunk_id="Doc_01 §1", doc_id="Doc_01", section="§1", verified=True)]
    return Claim(claim_id=claim_id, text=text, sub_query_id=sub_query_id, citations=citations)


class TestSession:
    def setup_method(self):
        self.session = Session("test-session-0000-0000-0000-000000000001")

    def test_append_chunk_builds_buffer(self):
        self.session.append_chunk("I need a venue", 0.0)
        self.session.append_chunk("in Pune", 0.8)
        assert "I need a venue" in self.session.utterance_buffer
        assert "in Pune" in self.session.utterance_buffer

    def test_clear_buffer(self):
        self.session.append_chunk("some text", 0.0)
        self.session.clear_buffer()
        assert self.session.utterance_buffer == ""

    def test_answer_versioning(self):
        claims = [make_claim("c1", "The venue is in Pune.")]
        v1 = self.session.commit_answer(claims)
        assert v1.version == 1
        assert self.session.answer_version == 1

        claims2 = [make_claim("c1", "The venue is in Pune."), make_claim("c2", "Capacity is 30.")]
        v2 = self.session.commit_answer(claims2)
        assert v2.version == 2
        assert self.session.answer_version == 2

    def test_evidence_pool_isolation(self):
        chunk = Chunk(chunk_id="Doc_01 §1", doc_id="Doc_01", section="§1", text="test")
        self.session.add_evidence([chunk])
        assert self.session.chunk_in_pool("Doc_01 §1")
        assert not self.session.chunk_in_pool("Doc_99 §1")

    def test_claim_graph_maps_subquery_to_claims(self):
        claims = [
            make_claim("c1", "Venue in Pune.", sub_query_id="sq1"),
            make_claim("c2", "Capacity 30.", sub_query_id="sq1"),
            make_claim("c3", "Cancellation policy.", sub_query_id="sq2"),
        ]
        self.session.commit_answer(claims)
        graph = self.session.build_claim_graph()
        assert "c1" in graph["sq1"]
        assert "c2" in graph["sq1"]
        assert "c3" in graph["sq2"]

    def test_delta_patch_retains_unaffected_claims(self):
        # Initial answer: sq1 and sq2 claims
        c1 = make_claim("c1", "Domestic reimbursement: INR 750/day.", sub_query_id="sq1")
        c2 = make_claim("c2", "Travel must be pre-approved.", sub_query_id="sq2")
        self.session.commit_answer([c1, c2])

        # Delta: only sq1 is affected (late detail: international trip)
        c1_new = make_claim("c1_updated", "International reimbursement: USD 75/day.", sub_query_id="sq1")
        new_version, diff = self.session.patch_claims(
            affected_sub_query_ids=["sq1"],
            new_claims=[c1_new],
        )

        # sq2 claim should be retained
        retained_ids = {c.claim_id for c in new_version.claims if c.patch_op == PatchOp.RETAIN}
        assert "c2" in retained_ids

        # sq1 claim should be added/updated
        new_ids = {c.claim_id for c in new_version.claims if c.patch_op in (PatchOp.ADD, PatchOp.UPDATE)}
        assert "c1_updated" in new_ids

    def test_delta_patch_bumps_version(self):
        c1 = make_claim("c1", "Some claim.", sub_query_id="sq1")
        self.session.commit_answer([c1])
        assert self.session.answer_version == 1

        c2 = make_claim("c2", "New claim.", sub_query_id="sq1")
        new_version, diff = self.session.patch_claims(["sq1"], [c2])
        assert self.session.answer_version == 2
        assert diff.version_from == 1
        assert diff.version_to == 2

    def test_diff_summary_is_human_readable(self):
        c1 = make_claim("c1", "Claim 1.", sub_query_id="sq1")
        c2 = make_claim("c2", "Claim 2.", sub_query_id="sq2")
        self.session.commit_answer([c1, c2])

        c3 = make_claim("c3", "Updated claim.", sub_query_id="sq1")
        _, diff = self.session.patch_claims(["sq1"], [c3])
        assert "Retained" in diff.summary
        assert "Added" in diff.summary or "Updated" in diff.summary


class TestSessionStore:
    def test_session_isolation(self):
        store = SessionStore()
        sid1 = "00000000-0000-0000-0000-000000000001"
        sid2 = "00000000-0000-0000-0000-000000000002"
        s1 = store.get_or_create(sid1)
        s2 = store.get_or_create(sid2)
        s1.append_chunk("Pune venue", 0.0)
        assert "Pune venue" in s1.utterance_buffer
        assert s2.utterance_buffer == ""

    def test_reset_clears_session(self):
        store = SessionStore()
        sid = "00000000-0000-0000-0000-000000000001"
        s = store.get_or_create(sid)
        s.append_chunk("some text", 0.0)
        store.reset(sid)
        assert store.get(sid) is None

    def test_new_session_after_reset(self):
        store = SessionStore()
        sid = "00000000-0000-0000-0000-000000000001"
        s1 = store.get_or_create(sid)
        s1.append_chunk("data", 0.0)
        store.reset(sid)
        s2 = store.get_or_create(sid)
        assert s2.utterance_buffer == ""  # Fresh session
