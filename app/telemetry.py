"""
telemetry.py — Structured JSONL telemetry for every pipeline stage.
Satisfies Gate G6: 100% trace coverage.

Every stage calls emit(event_type, session_id, **payload).
Events are written to JSONL and kept in-memory for the /metrics endpoint.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Optional

from app.config import get_settings
from app.schemas import TelemetryEvent, TelemetryEventType


class TelemetryWriter:
    """
    Thread-safe, async-compatible telemetry emitter.
    Writes structured JSONL events and maintains in-memory aggregates.
    """

    def __init__(self) -> None:
        cfg = get_settings()
        self._log_path = Path(cfg.telemetry_log_path)
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_file = open(self._log_path, "a", encoding="utf-8", buffering=1)  # line-buffered
        self._lock = asyncio.Lock()

        # In-memory aggregates (per session and global)
        self._session_events: dict[str, list[TelemetryEvent]] = defaultdict(list)
        self._global_counters: dict[str, float] = defaultdict(float)
        # Sliding window for rate metrics (last 1000 events)
        self._recent: deque[TelemetryEvent] = deque(maxlen=1000)

    async def emit(
        self,
        event_type: TelemetryEventType,
        session_id: str,
        **payload: Any,
    ) -> TelemetryEvent:
        """Emit a telemetry event. Non-blocking for the caller."""
        event = TelemetryEvent(
            event_type=event_type,
            ts=time.time(),
            session_id=session_id,
            payload=dict(payload),
        )
        async with self._lock:
            line = event.model_dump_json()
            self._log_file.write(line + "\n")
            self._session_events[session_id].append(event)
            self._recent.append(event)
            self._update_counters(event)
        return event

    def emit_sync(
        self,
        event_type: TelemetryEventType,
        session_id: str,
        **payload: Any,
    ) -> TelemetryEvent:
        """Synchronous emit for use outside async context."""
        event = TelemetryEvent(
            event_type=event_type,
            ts=time.time(),
            session_id=session_id,
            payload=dict(payload),
        )
        line = event.model_dump_json()
        self._log_file.write(line + "\n")
        self._session_events[session_id].append(event)
        self._recent.append(event)
        self._update_counters(event)
        return event

    def _update_counters(self, event: TelemetryEvent) -> None:
        t = event.event_type
        self._global_counters["total_events"] += 1
        self._global_counters[f"event_{t.value}"] += 1

        p = event.payload
        if t == TelemetryEventType.RETRIEVAL_STARTED:
            self._global_counters["total_retrievals"] += 1
        elif t == TelemetryEventType.SUPPRESSED_RETRIEVAL:
            self._global_counters["suppressed_retrievals"] += 1
        elif t == TelemetryEventType.SYNTHESIS_DONE:
            self._global_counters["total_turns"] += 1
            if "ttft_ms" in p:
                self._global_counters["sum_ttft_ms"] += p["ttft_ms"]
            if "e2e_ms" in p:
                self._global_counters["sum_e2e_ms"] += p["e2e_ms"]
            if "cost_usd" in p:
                self._global_counters["sum_cost_usd"] += p["cost_usd"]
        elif t == TelemetryEventType.CACHE_HIT:
            self._global_counters["cache_hits"] += 1
        elif t == TelemetryEventType.CITATION_CHECK:
            if not p.get("verified", True):
                self._global_counters["citation_violations"] += 1

    def get_session_events(self, session_id: str) -> list[TelemetryEvent]:
        return list(self._session_events.get(session_id, []))

    def get_aggregates(self) -> dict[str, Any]:
        c = self._global_counters
        turns = max(c["total_turns"], 1)
        retrievals = max(c["total_retrievals"] + c["suppressed_retrievals"], 1)
        return {
            "total_turns": int(c["total_turns"]),
            "total_retrievals": int(c["total_retrievals"]),
            "suppressed_retrievals": int(c["suppressed_retrievals"]),
            "avg_ttft_ms": c["sum_ttft_ms"] / turns,
            "avg_e2e_ms": c["sum_e2e_ms"] / turns,
            "avg_cost_usd": c["sum_cost_usd"] / turns,
            "cache_hit_rate": c["cache_hits"] / max(c["total_retrievals"], 1),
            "citation_violation_count": int(c["citation_violations"]),
        }

    def clear_session(self, session_id: str) -> None:
        self._session_events.pop(session_id, None)

    def close(self) -> None:
        self._log_file.close()


# ── Module-level singleton ──────────────────────────────────────────────────
_telemetry: Optional[TelemetryWriter] = None


def get_telemetry() -> TelemetryWriter:
    global _telemetry
    if _telemetry is None:
        _telemetry = TelemetryWriter()
    return _telemetry


# ── Convenience context manager for timing pipeline stages ─────────────────

class StageTimer:
    """
    Async context manager that emits a RETRIEVAL_DONE (or custom) event
    with the elapsed time on exit.

    Usage:
        async with StageTimer(tel, TelemetryEventType.RETRIEVAL_DONE, session_id,
                              sub_query_id=sq_id) as t:
            results = await do_retrieval(...)
        # After the block, t.elapsed_ms is available
    """

    def __init__(
        self,
        writer: TelemetryWriter,
        event_type: TelemetryEventType,
        session_id: str,
        **extra_payload: Any,
    ) -> None:
        self._writer = writer
        self._event_type = event_type
        self._session_id = session_id
        self._extra = extra_payload
        self.elapsed_ms: float = 0.0
        self._start: float = 0.0

    async def __aenter__(self) -> "StageTimer":
        self._start = time.perf_counter()
        return self

    async def __aexit__(self, *_: Any) -> None:
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000
        await self._writer.emit(
            self._event_type,
            self._session_id,
            elapsed_ms=round(self.elapsed_ms, 2),
            **self._extra,
        )
