# Telemetry Schema — Streaming Live RAG Engine
**Gate G6: 100% Trace Coverage**

All pipeline events are written as newline-delimited JSON (JSONL) to `logs/telemetry.jsonl`. Every event shares a common envelope and carries stage-specific payload.

---

## Common Envelope

```json
{
  "event_type": "<string>",
  "session_id": "<uuid4>",
  "ts": 1727023200.123,
  "payload": { ... }
}
```

| Field | Type | Description |
|-------|------|-------------|
| `event_type` | string | One of the event types below |
| `session_id` | UUID4 | Session that generated this event |
| `ts` | float | Unix timestamp (seconds, wall clock) |
| `payload` | object | Stage-specific fields (see below) |

---

## Event Types

### `chunk_received`
Emitted when a transcript chunk arrives at the gateway.

```json
{
  "event_type": "chunk_received",
  "session_id": "...",
  "ts": 1727023200.010,
  "payload": {
    "chunk_id": "c2",
    "text": "in Pune for 30 people",
    "t_start_s": 0.8,
    "buffer_tokens": 12
  }
}
```

---

### `controller_decision`
Emitted after every `RetrievalController.decide()` call.

```json
{
  "event_type": "controller_decision",
  "session_id": "...",
  "ts": 1727023200.013,
  "payload": {
    "decision": "RETRIEVE",
    "trigger": "provisional",
    "reason": "provisional_stable",
    "confidence": 0.82,
    "entity_count": 2,
    "latency_ms": 2.1
  }
}
```

| `decision` | Meaning |
|------------|---------|
| `RETRIEVE` | Retrieval should fire (provisional or multi-intent) |
| `WAIT` | Not enough signal yet |
| `NO_RETRIEVAL` | Presentation restructure or suppression |

---

### `suppressed_retrieval`
Emitted when a chunk triggers NO_RETRIEVAL (suppression or presentation).

```json
{
  "event_type": "suppressed_retrieval",
  "session_id": "...",
  "ts": 1727023200.020,
  "payload": {
    "reason": "presentation_restructure",
    "chunk_id": "c4"
  }
}
```

---

### `decomposition`
Emitted after the decomposer produces sub-queries.

```json
{
  "event_type": "decomposition",
  "session_id": "...",
  "ts": 1727023200.030,
  "payload": {
    "sub_query_count": 2,
    "sub_queries": ["the cancellation policy", "the catering options"],
    "method": "rules",
    "is_refinement": false,
    "orthogonality_score": 0.12,
    "latency_ms": 8.4
  }
}
```

---

### `retrieval_started`
Emitted when retrieval begins for a sub-query.

```json
{
  "event_type": "retrieval_started",
  "session_id": "...",
  "ts": 1727023200.040,
  "payload": {
    "sub_query_id": "sq1",
    "query": "the cancellation policy",
    "trigger": "provisional",
    "t": 1727023200.040
  }
}
```

---

### `retrieval_complete`
Emitted when retrieval finishes for a sub-query.

```json
{
  "event_type": "retrieval_complete",
  "session_id": "...",
  "ts": 1727023200.058,
  "payload": {
    "sub_query_id": "sq1",
    "n_hits": 4,
    "top_doc_ids": ["Doc_03 §3", "Doc_03 §4"],
    "cache_hit": false,
    "bm25_ms": 1.2,
    "dense_ms": 8.4,
    "fusion_ms": 0.9,
    "total_ms": 10.5
  }
}
```

---

### `synthesis_first_token`
Emitted when the first synthesis token is streamed (TTFT).

```json
{
  "event_type": "synthesis_first_token",
  "session_id": "...",
  "ts": 1727023200.090,
  "payload": {
    "ttft_ms": 32.0,
    "answer_version": 1,
    "n_claims": 3,
    "method": "extractive"
  }
}
```

---

### `citation_verified`
Emitted per citation after verification.

```json
{
  "event_type": "citation_verified",
  "session_id": "...",
  "ts": 1727023200.110,
  "payload": {
    "citation": "Doc_03 §3",
    "verified": true,
    "support_score": 0.72,
    "claim_id": "c1"
  }
}
```

---

### `citation_rejected`
Emitted when a citation fails ID-existence or semantic-support check.

```json
{
  "event_type": "citation_rejected",
  "session_id": "...",
  "ts": 1727023200.112,
  "payload": {
    "citation": "Doc_99 §1",
    "reason": "id_not_in_evidence",
    "claim_id": "c2"
  }
}
```

---

### `answer_version_bump`
Emitted when a new answer version is finalised.

```json
{
  "event_type": "answer_version_bump",
  "session_id": "...",
  "ts": 1727023200.220,
  "payload": {
    "answer_version": 2,
    "delta_claims": 1,
    "retained_claims": 2,
    "e2e_ms": 215.0,
    "ttft_ms": 32.0
  }
}
```

---

### `delta_patch`
Emitted with the diff ops when answer v(N+1) patches v(N).

```json
{
  "event_type": "delta_patch",
  "session_id": "...",
  "ts": 1727023200.215,
  "payload": {
    "version_from": 1,
    "version_to": 2,
    "ops": [
      {"op": "RETAIN", "claim_id": "c1", "text_before": "..."},
      {"op": "UPDATE", "claim_id": "c2", "text_before": "...", "text_after": "..."},
      {"op": "ADD",    "claim_id": "c3", "text_after": "..."}
    ],
    "summary": "1 updated, 1 added, 1 retained"
  }
}
```

---

## Coverage Guarantee (Gate G6)

Every turn MUST emit at minimum:
1. `chunk_received` × (number of chunks)
2. `controller_decision` × (number of chunks)
3. `decomposition` × 1 (if RETRIEVE)
4. `retrieval_started` + `retrieval_complete` × (number of sub-queries)
5. `synthesis_first_token` × 1
6. `answer_version_bump` × 1

If any of these are missing, the G6 check fails in `bench/replay.py`.

---

## Querying Telemetry

```python
import json
from pathlib import Path

events = [json.loads(l) for l in Path("logs/telemetry.jsonl").read_text().splitlines() if l]

# Get all retrieval latencies
retrieval_ms = [
    e["payload"]["total_ms"]
    for e in events
    if e["event_type"] == "retrieval_complete"
]
print(f"Avg retrieval latency: {sum(retrieval_ms)/len(retrieval_ms):.1f}ms")

# Check G6 coverage per session
from collections import defaultdict
by_session = defaultdict(list)
for e in events:
    by_session[e["session_id"]].append(e["event_type"])

for sid, types in by_session.items():
    has_all = all(t in types for t in ["chunk_received","controller_decision","answer_version_bump"])
    print(f"{sid[:8]}: {'OK' if has_all else 'MISSING EVENTS'}")
```

---

## REST Metrics Endpoint

`GET /metrics` returns aggregate statistics over all sessions:

```json
{
  "total_turns": 3,
  "total_retrievals": 5,
  "suppressed_retrievals": 1,
  "avg_ttft_ms": 32,
  "avg_e2e_ms": 185,
  "cache_hit_rate": 0.4,
  "session_count": 3
}
```
