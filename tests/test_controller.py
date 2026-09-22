"""
tests/test_controller.py — Unit tests for the retrieval controller.

Tests cover:
- WAIT on insufficient tokens
- RETRIEVE(provisional) on entity + intent stability
- RETRIEVE(multi_intent) on conjunction detection
- NO_RETRIEVAL on presentation request with prior answer
- Debounce guard
- Similarity guard
"""

import time
from unittest.mock import MagicMock

import pytest

from app.controller import RetrievalController
from app.schemas import ChunkEvent, ControllerDecision, ReasonCode, TriggerType


TEST_SESSION_ID = "a1b2c3d4-e5f6-4a7b-8c9d-0e1f2a3b4c5d"


def make_chunk(text: str, t: float = 0.0, is_final: bool = False, session_id: str = TEST_SESSION_ID) -> ChunkEvent:
    return ChunkEvent(
        session_id=session_id,
        chunk_id=f"c_{int(t*10)}",
        text=text,
        t_start_s=t,
        is_final=is_final,
    )


class TestControllerRulesPath:
    def setup_method(self):
        self.ctrl = RetrievalController()

    def test_wait_on_too_few_tokens(self):
        chunk = make_chunk("I need")
        result = self.ctrl.decide(chunk, "I need", has_prior_answer=False)
        assert result.decision == ControllerDecision.WAIT
        assert result.reason == ReasonCode.INTENT_INCOMPLETE

    def test_retrieve_provisional_on_entity_and_intent(self):
        buffer = "I need to plan a customer workshop in Pune for 30 people"
        chunk = make_chunk("Pune for 30 people", t=0.8)
        result = self.ctrl.decide(chunk, buffer, has_prior_answer=False)
        assert result.decision == ControllerDecision.RETRIEVE
        assert result.trigger == TriggerType.PROVISIONAL
        assert result.reason == ReasonCode.PROVISIONAL_STABLE

    def test_no_retrieval_on_presentation_request(self):
        ctrl = RetrievalController()
        # Now send a presentation request with has_prior_answer=True
        buffer2 = "Please repeat that in two bullet points"
        chunk2 = make_chunk("repeat in two bullets", t=10.0)
        result = ctrl.decide(chunk2, buffer2, has_prior_answer=True)
        assert result.decision == ControllerDecision.NO_RETRIEVAL
        assert result.trigger == TriggerType.PRESENTATION_RESTRUCTURE

    def test_debounce_prevents_rapid_retrieval(self):
        ctrl = RetrievalController()
        buffer = "I need to plan a workshop in Pune for 30 people"
        chunk1 = make_chunk("Pune for 30 people", t=0.0)
        r1 = ctrl.decide(chunk1, buffer, has_prior_answer=False)
        assert r1.decision == ControllerDecision.RETRIEVE

        # Immediately another chunk — should be debounced
        chunk2 = make_chunk("and some catering", t=0.1)
        r2 = ctrl.decide(chunk2, buffer + " and some catering", has_prior_answer=False)
        assert r2.decision == ControllerDecision.WAIT
        assert r2.reason == ReasonCode.DEBOUNCE

    def test_multi_intent_on_conjunction_after_provisional(self):
        ctrl = RetrievalController()
        # First: provisional retrieval
        buffer1 = "I need a venue in Pune for 30 people"
        chunk1 = make_chunk("Pune for 30 people", t=0.0)
        ctrl.decide(chunk1, buffer1, has_prior_answer=False)

        # Wait for debounce (manipulate state directly)
        state = ctrl._get_state(TEST_SESSION_ID)
        state.last_retrieval_t = time.time() - 2.0  # simulate 2s elapsed

        # Now: conjunction + new entity
        buffer2 = "I need a venue in Pune for 30 people and also the cancellation policy"
        chunk2 = make_chunk("and also the cancellation policy", t=2.0)
        r2 = ctrl.decide(chunk2, buffer2, has_prior_answer=False)
        assert r2.decision == ControllerDecision.RETRIEVE
        assert r2.trigger == TriggerType.MULTI_INTENT

    def test_no_retrieval_on_formatting_without_new_entities(self):
        # A presentation request with enough tokens and has_prior_answer=True
        buffer = "Please summarize that in one paragraph for me now"
        chunk = make_chunk("summarize that in one paragraph", t=0.0)
        result = self.ctrl.decide(chunk, buffer, has_prior_answer=True)
        assert result.decision == ControllerDecision.NO_RETRIEVAL

    def test_wait_on_no_entities(self):
        buffer = "I am thinking about the options available"
        chunk = make_chunk("I am thinking about the options available", t=0.0)
        result = self.ctrl.decide(chunk, buffer, has_prior_answer=False)
        # No entities → WAIT
        assert result.decision == ControllerDecision.WAIT

    def test_late_constraint_detected(self):
        ctrl = RetrievalController()
        buffer = "Actually the trip was international and booked after travel"
        chunk = make_chunk("international booking", t=5.0, is_final=True)
        result = ctrl.decide_late_constraint(chunk, buffer)
        assert result.decision == ControllerDecision.RETRIEVE
        assert result.trigger == TriggerType.LATE_CONSTRAINT

    def test_session_isolation(self):
        """Two sessions must not share controller state."""
        ctrl = RetrievalController()
        sid1 = "a1b2c3d4-e5f6-4a7b-8c9d-000000000001"
        sid2 = "a1b2c3d4-e5f6-4a7b-8c9d-000000000002"

        buffer = "I need to plan a workshop in Pune for 30 people"
        chunk1 = ChunkEvent(session_id=sid1, chunk_id="c1", text="Pune for 30 people", t_start_s=0.0)
        ctrl.decide(chunk1, buffer, has_prior_answer=False)

        # Session 2 state should be independent (chunk_count is the reliable counter)
        state1 = ctrl._get_state(sid1)
        state2 = ctrl._get_state(sid2)
        assert state1.chunk_count > 0
        assert state2.chunk_count == 0
