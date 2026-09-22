"""
gateway.py — Stream Gateway: WebSocket live streaming + replay endpoint.

Handles:
- WS /ws/stream/{session_id}: live chunk-by-chunk ingestion
- POST /replay: scripted scenario replay at 1x or accelerated speed

The gateway orchestrates the full pipeline:
chunk → controller → decomposer → retrieval → synthesis → verifier → session
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from pathlib import Path
from typing import Any, Optional

from fastapi import WebSocket, WebSocketDisconnect

from app.config import get_settings
from app.controller import get_controller
from app.decomposer import MultiIntentDecomposer
from app.llm import get_llm_client
from app.retrieval.fusion import get_retriever
from app.retrieval.index import get_index_manager
from app.schemas import (
    AnswerDiff,
    ChunkEvent,
    ControllerDecision,
    FusedResults,
    ReplayRequest,
    ReplayResponse,
    RetrievalEvent,
    SubQuery,
    TelemetryEvent,
    TelemetryEventType,
    TriggerType,
    TurnRecord,
)
from app.synthesis.session import Session, get_session_store
from app.synthesis.synth import GroundedSynthesiser
from app.synthesis.verifier import CitationVerifier
from app.telemetry import get_telemetry


class PipelineOrchestrator:
    """
    Orchestrates the full streaming RAG pipeline for one session.
    Called by both the WebSocket handler and the replay runner.
    """

    def __init__(self) -> None:
        self._controller = get_controller()
        self._retriever = get_retriever()
        self._verifier = CitationVerifier()
        llm = get_llm_client()
        self._decomposer = MultiIntentDecomposer(llm)
        self._synth = GroundedSynthesiser(llm)
        self._tel = get_telemetry()
        self._index = get_index_manager()

    async def process_chunk(
        self,
        chunk: ChunkEvent,
        session: Session,
        ws_send=None,  # optional: async callable to push WS events
    ) -> Optional[TurnRecord]:
        """
        Process a single transcript chunk through the pipeline.
        Returns a TurnRecord only when synthesis completes (is_final=True or multi-intent).
        """
        cfg = get_settings()
        t_chunk_received = time.time()

        # Append to session buffer
        session.append_chunk(chunk.text, chunk.t_start_s)

        await self._tel.emit(
            TelemetryEventType.CHUNK_RECEIVED,
            chunk.session_id,
            chunk_id=chunk.chunk_id,
            text=chunk.text if cfg.log_text else f"[{len(chunk.text)} chars]",
            t_start_s=chunk.t_start_s,
            buffer_len=len(session.utterance_buffer.split()),
        )

        # Get dense index (for R2 embedding drift, optional)
        dense_index = self._index.dense if self._index.is_loaded else None

        # Controller decision
        has_prior_answer = session.answer_version > 0
        if has_prior_answer and chunk.is_final:
            ctrl = self._controller.decide_late_constraint(
                chunk, session.utterance_buffer
            )
        else:
            ctrl = self._controller.decide(
                chunk, session.utterance_buffer,
                has_prior_answer=has_prior_answer,
                dense_index=dense_index,
            )

        await self._tel.emit(
            TelemetryEventType.CONTROLLER_DECISION,
            chunk.session_id,
            decision=ctrl.decision.value,
            trigger=ctrl.trigger.value if ctrl.trigger else None,
            reason=ctrl.reason.value,
            confidence=ctrl.confidence,
            entity_count=ctrl.entity_count,
            embedding_drift=ctrl.embedding_drift,
        )

        if ws_send:
            await ws_send("controller_decision", {
                "decision": ctrl.decision.value,
                "trigger": ctrl.trigger.value if ctrl.trigger else None,
                "reason": ctrl.reason.value,
                "confidence": ctrl.confidence,
                "t": t_chunk_received,
            })

        # ── NO_RETRIEVAL: reformat from session ───────────────────────────
        if ctrl.decision == ControllerDecision.NO_RETRIEVAL:
            await self._tel.emit(
                TelemetryEventType.SUPPRESSED_RETRIEVAL,
                chunk.session_id,
                reason=ctrl.reason.value,
                buffer=session.utterance_buffer[:100],
            )
            if ws_send:
                await ws_send("suppressed_retrieval", {
                    "reason": ctrl.reason.value,
                })
            if has_prior_answer and chunk.is_final:
                # Reformat mode
                reformatted = await self._synth.reformat(
                    session_id=chunk.session_id,
                    current_claims=session.current_claims,
                    format_request=session.utterance_buffer,
                )
                current_version = session.get_current_version()
                return TurnRecord(
                    session_id=chunk.session_id,
                    retrieval_events=[],
                    sub_queries=[session.utterance_buffer],
                    answer=reformatted,
                    answer_version=session.answer_version,
                    citations=current_version.citations if current_version else [],
                    uncertainty="",
                )
            return None

        # ── WAIT: just buffer ─────────────────────────────────────────────
        if ctrl.decision == ControllerDecision.WAIT and not chunk.is_final:
            return None

        # ── RETRIEVE: run the pipeline ────────────────────────────────────
        if ctrl.decision == ControllerDecision.RETRIEVE or chunk.is_final:
            return await self._run_retrieval_pipeline(
                chunk=chunk,
                session=session,
                ctrl_trigger=ctrl.trigger,
                t_chunk_received=t_chunk_received,
                ws_send=ws_send,
            )

        return None

    async def _run_retrieval_pipeline(
        self,
        chunk: ChunkEvent,
        session: Session,
        ctrl_trigger: Optional[TriggerType],
        t_chunk_received: float,
        ws_send=None,
    ) -> TurnRecord:
        """Full retrieval + synthesis pipeline for one turn."""
        cfg = get_settings()
        t0_pipeline = time.perf_counter()
        retrieval_events: list[RetrievalEvent] = []
        dense_index = self._index.dense if self._index.is_loaded else None

        # 1. Decompose
        prior_sq_texts = [sq.text for sq in session.sub_queries.values()]
        decomp = await self._decomposer.decompose(
            session_id=chunk.session_id,
            utterance=session.utterance_buffer,
            prior_sub_queries=prior_sq_texts if session.answer_version > 0 else None,
            dense_index=dense_index,
        )
        session.register_sub_queries(decomp.sub_queries)

        if ws_send:
            await ws_send("decomposition", {
                "sub_queries": [{"id": sq.sub_query_id, "text": sq.text} for sq in decomp.sub_queries],
                "is_refinement": decomp.is_refinement,
            })

        # 2. Determine affected sub-queries (for refinement)
        if decomp.is_refinement and session.answer_version > 0:
            affected_sq_ids = decomp.affected_sub_query_ids or [sq.sub_query_id for sq in decomp.sub_queries]
            sub_queries_to_retrieve = [sq for sq in decomp.sub_queries if sq.sub_query_id in affected_sq_ids]
        else:
            sub_queries_to_retrieve = decomp.sub_queries

        # 3. Parallel retrieval (asyncio.gather)
        if self._index.is_loaded:
            retrieval_tasks = [
                self._retrieve_one(
                    session_id=chunk.session_id,
                    sub_query=sq,
                    t_chunk_received=t_chunk_received,
                )
                for sq in sub_queries_to_retrieve
            ]
            fused_results: list[FusedResults] = await asyncio.gather(*retrieval_tasks)

            # Emit retrieval events for timeline
            for fr in fused_results:
                retrieval_events.append(RetrievalEvent(
                    timestamp_s=t_chunk_received,
                    query=fr.query,
                    trigger=ctrl_trigger.value if ctrl_trigger else "final",
                    sub_query_id=fr.sub_query_id,
                    n_hits=len(fr.hits),
                    latency_ms=fr.retrieval_latency_ms,
                    cache_hit=fr.cache_hit,
                ))
            # Add retrieved chunks to session evidence pool
            for fr in fused_results:
                session.add_evidence([hit.chunk for hit in fr.hits])
        else:
            # No index — will use uncertainty flag
            fused_results = []

        await self._tel.emit(
            TelemetryEventType.FUSION_DONE,
            chunk.session_id,
            sub_query_count=len(sub_queries_to_retrieve),
            total_hits=sum(len(fr.hits) for fr in fused_results),
            cache_hits=sum(1 for fr in fused_results if fr.cache_hit),
        )

        # 4. Synthesis
        t_synth_start = time.perf_counter()

        if decomp.is_refinement and session.answer_version > 0:
            # Delta synthesis (G5)
            affected_claims = [
                c for c in session.current_claims
                if c.sub_query_id in (decomp.affected_sub_query_ids or [])
            ]
            new_claims, uncertainty = await self._synth.synthesise_delta(
                session_id=chunk.session_id,
                affected_sub_queries=sub_queries_to_retrieve,
                existing_claims=affected_claims,
                new_constraint=session.utterance_buffer,
                new_fused_results=fused_results,
            )
            # Verify new claims
            verified_new, removed_new, _ = await self._verifier.verify_claims(
                session_id=chunk.session_id,
                claims=new_claims,
                evidence_pool=session.evidence_pool,
                dense_index=dense_index,
            )
            # Patch session (R3: claim graph)
            new_version, diff = session.patch_claims(
                affected_sub_query_ids=decomp.affected_sub_query_ids or [sq.sub_query_id for sq in sub_queries_to_retrieve],
                new_claims=verified_new,
                new_uncertainty=uncertainty,
            )
            answer_diff = diff
        else:
            # Initial synthesis
            all_claims, uncertainty, ttft_ms = await self._synth.synthesise(
                session_id=chunk.session_id,
                sub_queries=decomp.sub_queries,
                fused_results=fused_results,
            )
            # Verify
            verified, removed, unc_from_verifier = await self._verifier.verify_claims(
                session_id=chunk.session_id,
                claims=all_claims,
                evidence_pool=session.evidence_pool,
                dense_index=dense_index,
            )
            final_uncertainty = " ".join(filter(None, [uncertainty, unc_from_verifier]))
            # Commit answer version
            new_version = session.commit_answer(verified, final_uncertainty)
            answer_diff = None
            uncertainty = final_uncertainty

        # 5. Build TurnRecord
        current_version = session.get_current_version()
        answer_text = self._synth.claims_to_answer_text(current_version.claims) if current_version else ""

        e2e_ms = (time.perf_counter() - t0_pipeline) * 1000
        await self._tel.emit(
            TelemetryEventType.ANSWER_VERSION_BUMP,
            chunk.session_id,
            answer_version=session.answer_version,
            e2e_ms=round(e2e_ms, 2),
            claim_count=len(current_version.claims) if current_version else 0,
        )

        if ws_send:
            await ws_send("answer_complete", {
                "version": session.answer_version,
                "answer": answer_text,
                "claims": [c.model_dump() for c in (current_version.claims if current_version else [])],
                "citations": current_version.citations if current_version else [],
                "uncertainty": current_version.uncertainty if current_version else "",
            })
            if answer_diff:
                await ws_send("answer_diff", answer_diff.model_dump())

        # Clear buffer after successful synthesis
        session.clear_buffer()

        return TurnRecord(
            session_id=chunk.session_id,
            retrieval_events=retrieval_events,
            sub_queries=[sq.text for sq in decomp.sub_queries],
            answer=answer_text,
            answer_version=session.answer_version,
            citations=current_version.citations if current_version else [],
            uncertainty=current_version.uncertainty if current_version else "",
            answer_diff=answer_diff,
            end_to_end_ms=round(e2e_ms, 2),
        )

    async def _retrieve_one(
        self,
        session_id: str,
        sub_query: SubQuery,
        t_chunk_received: float,
    ) -> FusedResults:
        """Retrieve evidence for one sub-query (runs in asyncio.gather)."""
        await self._tel.emit(
            TelemetryEventType.RETRIEVAL_STARTED,
            session_id,
            sub_query_id=sub_query.sub_query_id,
            query=sub_query.text,
            t=t_chunk_received,
        )
        sub_query.status = "running"
        result = self._retriever.query(
            session_id=session_id,
            sub_query_id=sub_query.sub_query_id,
            query=sub_query.text,
            bm25_index=self._index.bm25,
            dense_index=self._index.dense,
        )
        sub_query.status = "done" if result.hits else "no_evidence"

        await self._tel.emit(
            TelemetryEventType.RETRIEVAL_DONE,
            session_id,
            sub_query_id=sub_query.sub_query_id,
            n_hits=len(result.hits),
            latency_ms=result.retrieval_latency_ms,
            cache_hit=result.cache_hit,
        )
        return result


# ─────────────────────────────────────────────────────────────────────────────
# WebSocket handler
# ─────────────────────────────────────────────────────────────────────────────

async def websocket_handler(ws: WebSocket, session_id: str) -> None:
    """Handle a live WebSocket streaming session."""
    await ws.accept()
    store = get_session_store()
    tel = get_telemetry()
    orchestrator = PipelineOrchestrator()

    try:
        session = store.get_or_create(session_id)
    except RuntimeError as e:
        await ws.close(code=1008, reason=str(e))
        return

    async def ws_send(event_type: str, payload: dict) -> None:
        try:
            await ws.send_json({"type": event_type, "session_id": session_id, "payload": payload})
        except Exception:
            pass

    chunk_counter = 0
    try:
        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await ws_send("error", {"code": "invalid_json", "message": "Expected JSON"})
                continue

            msg_type = msg.get("type", "chunk")

            if msg_type == "reset":
                store.reset(session_id)
                session = store.get_or_create(session_id)
                get_controller().reset_session(session_id)
                get_retriever().clear_session_cache(session_id)
                await ws_send("reset_ack", {})
                continue

            if msg_type == "utterance_end":
                # Force final processing of whatever is buffered
                if session.utterance_buffer:
                    chunk_counter += 1
                    dummy_chunk = ChunkEvent(
                        session_id=session_id,
                        chunk_id=f"end_{chunk_counter}",
                        text=" ",
                        t_start_s=time.time() - session.buffer_start_t,
                        is_final=True,
                    )
                    await orchestrator.process_chunk(dummy_chunk, session, ws_send)
                continue

            # Regular chunk
            chunk_counter += 1
            try:
                chunk = ChunkEvent(
                    session_id=session_id,
                    chunk_id=msg.get("chunk_id", f"c{chunk_counter}"),
                    text=msg.get("text", ""),
                    t_start_s=float(msg.get("t_start_s", time.time() - session.buffer_start_t)),
                    is_final=bool(msg.get("is_final", False)),
                )
            except Exception as e:
                await ws_send("error", {"code": "validation_error", "message": str(e)})
                continue

            turn_record = await orchestrator.process_chunk(chunk, session, ws_send)
            if turn_record:
                await ws_send("turn_record", turn_record.model_dump())

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await ws_send("error", {"code": "internal_error", "message": str(e)})
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# Replay runner
# ─────────────────────────────────────────────────────────────────────────────

async def run_replay(request: ReplayRequest) -> ReplayResponse:
    """
    Run a scripted scenario replay.
    Loads chunks from a bundled scenario or inline chunks, plays them at speed.
    Returns the full event log.
    """
    from app.schemas import TelemetryEvent

    t_start = time.perf_counter()
    session_id = request.session_id or str(uuid.uuid4())

    store = get_session_store()
    store.reset(session_id)
    session = store.get_or_create(session_id)
    get_controller().reset_session(session_id)
    get_retriever().clear_session_cache(session_id)

    orchestrator = PipelineOrchestrator()
    tel = get_telemetry()

    # Load scenario chunks
    chunks: list[ChunkEvent] = []
    if request.scenario_name:
        chunks = _load_scenario(request.scenario_name, session_id)
    elif request.chunks:
        chunks = request.chunks

    if not chunks:
        raise ValueError("No chunks provided and scenario not found.")

    # Simulate timing
    t0_stream = chunks[0].t_start_s if chunks else 0.0
    last_turn_record: Optional[TurnRecord] = None

    for chunk in chunks:
        # Simulate real-time delay (scaled by speed multiplier)
        if len(chunks) > 1 and request.speed_multiplier < 100:
            delay = (chunk.t_start_s - t0_stream) / request.speed_multiplier
            if delay > 0:
                await asyncio.sleep(min(delay, 5.0))

        turn_record = await orchestrator.process_chunk(chunk, session)
        if turn_record:
            last_turn_record = turn_record

    # Build final turn record if none produced (e.g., all WAIT)
    if last_turn_record is None:
        if session.utterance_buffer:
            final_chunk = ChunkEvent(
                session_id=session_id,
                chunk_id="replay_final",
                text=" ",
                t_start_s=chunks[-1].t_start_s if chunks else 0.0,
                is_final=True,
            )
            last_turn_record = await orchestrator.process_chunk(final_chunk, session)

    if last_turn_record is None:
        last_turn_record = TurnRecord(
            session_id=session_id,
            answer="No answer generated.",
            uncertainty="Pipeline did not produce a response.",
        )

    telemetry_events = tel.get_session_events(session_id)
    duration_ms = (time.perf_counter() - t_start) * 1000

    return ReplayResponse(
        session_id=session_id,
        turn_record=last_turn_record,
        telemetry_events=telemetry_events,
        duration_ms=round(duration_ms, 2),
    )


def _load_scenario(scenario_name: str, session_id: str) -> list[ChunkEvent]:
    """Load a bundled scenario from bench/scenarios/."""
    scenario_dir = Path("bench/scenarios")
    # Try exact name first, then with .jsonl extension
    candidates = [
        scenario_dir / scenario_name,
        scenario_dir / f"{scenario_name}.jsonl",
        scenario_dir / f"{scenario_name}.json",
    ]
    for path in candidates:
        if path.exists():
            chunks: list[ChunkEvent] = []
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    try:
                        data = json.loads(line)
                        chunks.append(ChunkEvent(
                            session_id=session_id,
                            chunk_id=data.get("chunk_id", f"c{len(chunks)+1}"),
                            text=data.get("text", ""),
                            t_start_s=float(data.get("t_start_s", 0.0)),
                            is_final=bool(data.get("is_final", False)),
                        ))
                    except Exception as e:
                        print(f"[Replay] Warning: skipped line in {path}: {e}")
            return chunks
    raise FileNotFoundError(f"Scenario '{scenario_name}' not found in {scenario_dir}")
