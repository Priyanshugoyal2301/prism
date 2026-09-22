# Technical Architecture — Streaming Live RAG Engine
**Draft v1 · Theme 4 · Samsung PRISM GenAI Hackathon 3.0**

## 1. Design Principles
1. **Simple and cheap beats heavy orchestration.** One Python service, one process, no agent framework.
2. **Event-driven core.** Everything is a typed event flowing through a small pipeline.
3. **Deterministic first, LLM second.** Rules handle the easy cases; an LLM is used only when needed (controller ambiguity, decomposition, synthesis).
4. **Grounded by construction.** Synthesis can only see retrieved chunks; a verifier checks every citation.
5. **State is ephemeral and per-session.**

## 2. High-Level Architecture

```
 Transcript chunks (WS / replay file)
            │
            ▼
 ┌────────────────────────┐
 │ Stream Gateway         │  FastAPI: /ws/stream, /replay, /health
 └───────────┬────────────┘
             ▼
 ┌────────────────────────┐   WAIT ─────────► (buffer more)
 │ [1] Retrieval          │   NO_RETRIEVAL ─► Session Synthesizer (reformat only)
 │     Controller         │
 └───────────┬────────────┘
             │ RETRIEVE (provisional | multi_intent | late_constraint)
             ▼
 ┌────────────────────────┐
 │ [2] Multi-Intent       │  1..N orthogonal sub-queries + carried context
 │     Decomposer         │
 └───────────┬────────────┘
             ▼  (asyncio.gather, one task per sub-query)
 ┌────────────────────────┐
 │ [3] Retrieval & Fusion │  BM25 + dense → RRF → dedupe → (optional rerank)
 └───────────┬────────────┘
             ▼
 ┌────────────────────────┐        ┌───────────────────────┐
 │ [4] Session-Aware      │◄──────►│ Session Store         │
 │     Synthesizer        │        │ (in-memory, TTL)      │
 │  • initial / delta     │        │ answer versions,      │
 │  • citation verifier   │        │ claims→citations,     │
 │  • uncertainty flag    │        │ evidence pool         │
 └───────────┬────────────┘        └───────────────────────┘
             ▼
 Streamed answer + citations + structured record
             │
             └──► [5] Telemetry (JSONL + /metrics) — every stage emits events
```

## 3. Technology Choices
| Concern | Choice | Rationale |
|---|---|---|
| Language / runtime | Python 3.11 | Ecosystem, async |
| API | **FastAPI** + `uvicorn`, WebSocket + SSE | Native async, typed models |
| Schemas | **Pydantic v2** | Contract enforcement and validation |
| Sparse retrieval | `rank_bm25` (or Whoosh/Tantivy if corpus is large) | Cheap, strong on exact terms/IDs |
| Dense retrieval | `sentence-transformers` small embedding model (e.g., BGE-small / MiniLM) + **FAISS** (CPU) or NumPy | Runs on CPU, low cost |
| Fusion | **Reciprocal Rank Fusion** (k≈60) | Parameter-light, robust |
| Rerank (P1) | Small cross-encoder, behind a flag | Enables the required ablation |
| LLM | Provider-agnostic wrapper (`LLMClient`) reading key from env; small/fast model for controller + decomposer, stronger model for synthesis | Cost per turn; swap without code change |
| Session store | In-process dict + TTL (Redis optional, off by default) | Ephemeral by requirement |
| Telemetry | Structured JSON logs (`structlog`) + in-memory metrics | G6 |
| Packaging | Docker + `docker compose up` | G1 |
| Frontend | Static HTML/JS (or small Vite/React) served by FastAPI | One container |

## 4. Component Design

### 4.1 Stream Gateway
- **Input event:** `{session_id, chunk_id, text, t_start_s, is_final}`.
- **Modes:** `WS /ws/stream/{session_id}` (live) and `POST /replay` (accepts a scenario file, plays it at 1× or accelerated, and returns the full event log).
- Assigns monotonic `event_id`, records arrival `t_wall` and `t_stream`.

### 4.2 Retrieval Controller
Two-tier decision:
1. **Rules fast-path (no LLM):**
   - Presentation verbs (`repeat`, `shorten`, `bullet`, `translate`, `summarize that`) + existing answer → **NO_RETRIEVAL**.
   - Entity/intent stability: detect named entities (place, quantity, date) and intent verbs via regex/spaCy-light or keyword lexicon. A stable partial across ≥ *k* chunks or ≥ *n* tokens → **RETRIEVE(provisional)**.
   - Sentence-boundary or conjunction cue ("and", "also", "plus") → check for new sub-intent → **RETRIEVE(multi_intent)**.
2. **LLM/classifier fallback (P1):** only for ambiguous cases; small model returns `{decision, reason}` JSON.

Debounce: minimum interval between retrievals, and skip if the query embedding is within a similarity threshold of the last dispatched query.

**Output:** `{decision: WAIT|RETRIEVE|NO_RETRIEVAL, trigger, reason, confidence}`.

### 4.3 Multi-Intent Decomposer
- Prompt returns strict JSON: `{"sub_queries": [...], "is_refinement": bool, "affected_subqueries": [...]}`.
- Rule-based fallback: split on conjunction/comma boundaries, resolve shared context (city, size), dedupe.
- Guardrails: max N sub-queries (default 4); embedding-similarity dedupe to prevent over-fragmentation; a single-intent utterance must stay a single query.

### 4.4 Retrieval & Fusion
```
for each sub_query (parallel):
    sparse = bm25.top_k(q, k=20)
    dense  = faiss.top_k(embed(q), k=20)
    fused  = RRF(sparse, dense)
merged = RRF across sub-queries (keeps per-sub-query provenance)
deduped = drop near-duplicate chunks (same doc/section or cosine > 0.95)
final  = optional rerank → top-K per sub-query (K≈3–4)
```
- **Indexing at startup only** from the provided corpus (no precomputed answers, no external sources).
- Chunking: section-aware (respect headings), ~200–400 tokens, overlap ~40; each chunk carries `doc_id`, `section`, `text`.
- **Provisional cache:** `(session_id, normalized_query) → results`, reused when the same sub-query reappears at final decomposition.

### 4.5 Session-Aware Synthesizer
**Session object**
```python
Session:
  id, created_at, ttl
  utterance_buffer: str
  answer_version: int
  answer_claims: list[Claim]        # {claim_id, text, citations[], sub_query_id}
  sub_queries: dict[id -> SubQuery] # text, status, evidence_ids
  evidence_pool: dict[chunk_id -> Chunk]
  history: list[Event]
```
**Modes**
- **Initial:** compose one answer over all sub-queries; each claim carries citation IDs from the evidence pool.
- **Delta (refine):**
  1. Classify the late detail → maps to *existing topic* vs *new topic*.
  2. Retrieve only for the delta sub-queries.
  3. Ask the LLM to output **patch operations** (`add`, `update`, `retain`) against the claim list, not a rewrite.
  4. Apply patch → `answer_version += 1`; keep prior citations; append delta citations.
- **Reformat (no retrieval):** transform `answer_claims` into bullets/short/other language; reuse existing citations; add none.

**Grounding & uncertainty**
- Citation verifier: (1) ID exists in the evidence pool; (2) lexical/embedding entailment score between claim and chunk above threshold; failing claims are removed or rewritten as uncertainty.
- Missing evidence for a sub-query → `uncertainty` entry ("X could not be verified from the retrieved corpus") or a targeted clarifying question.

### 4.6 Telemetry
Event types (each with `ts`, `session_id`, `event_id`):
`chunk_received`, `controller_decision`, `retrieval_started`, `retrieval_done`, `decomposition`, `fusion_done`, `synthesis_first_token`, `synthesis_done`, `citation_check`, `answer_version_bump`, `suppressed_retrieval`, `error`.
Metrics: time-to-first-retrieval, time-to-first-token, end-to-end latency, tokens in/out, est. cost per turn, cache hits, false-trigger count.

## 5. API Surface
| Endpoint | Purpose |
|---|---|
| `GET /health` | `{"status":"ok"}` |
| `WS /ws/stream/{session_id}` | Live chunk in / events out |
| `POST /replay` | Run a scripted stream, return event log + final record |
| `POST /session/{id}/reset` | Drop ephemeral state |
| `GET /session/{id}/state` | Current answer version, claims, evidence (debug) |
| `GET /metrics` | Aggregated telemetry snapshot |
| `GET /` | Demo console |

Structured turn record (matches PRD FR-8): `retrieval_events`, `sub_queries`, `answer`, `answer_version`, `citations`, `uncertainty`. ⚠️ Align field names with the released test-suite schema.

## 6. Event Timeline Walk-Through (from the guide's Example 1)
| t | Chunk | Component behaviour |
|---|---|---|
| 0.0 | "I need to plan a customer workshop in…" | Controller: WAIT (unstable) |
| 0.8 | "…Pune for 30 people, and I need…" | RETRIEVE(provisional): "Pune workshop venue capacity 30"; log `retrieval_started` |
| 1.6 | "…the cancellation policy and the catering options." | Decompose to 3 sub-queries; parallel retrieval; RRF |
| 2.1 | [Utterance End] | Synthesize a unified cited answer, flag unverified aspects |

## 7. Performance & Cost Strategy
- Rules fast-path avoids an LLM call for most controller decisions.
- Parallel sub-query retrieval; provisional results reused.
- Stream synthesis tokens as soon as evidence is ready → low TTFT.
- Small model for controller/decomposer; single synthesis call per turn.
- Embeddings computed once at start-up; cached to disk in the image build (derived from the corpus only).
- Log token counts to estimate cost per turn.

## 8. Evaluation Harness (in-repo)
- `bench/replay.py`: play a scenario file with timestamps, record events.
- `bench/metrics.py`: computes G2–G6 (early retrieval ratio, sub-intent accuracy, citation support, version continuity, trace coverage) plus recall@k, groundedness, TTFT, cost/turn.
- **Baseline:** classic wait-for-end, single-query, dense-only RAG for comparison.
- **Ablations (≥2):** hybrid vs dense-only; rule-based vs model-based controller; (optional) with/without rerank.
- **Failure analysis (≥3):** curated edge cases with root-cause notes.

## 9. Repository Layout
```
/
├─ app/
│  ├─ main.py               # FastAPI wiring
│  ├─ gateway.py            # WS/replay
│  ├─ controller.py
│  ├─ decomposer.py
│  ├─ retrieval/{index.py,bm25.py,dense.py,fusion.py,rerank.py}
│  ├─ synthesis/{session.py,synth.py,verifier.py,delta.py}
│  ├─ telemetry.py
│  ├─ llm.py                # provider-agnostic client
│  └─ schemas.py            # Pydantic models
├─ frontend/                # demo console
├─ bench/                   # replay + metrics + ablations
├─ data/                    # corpus (provided) — no answers/queries
├─ docs/{architecture_brief.md,benchmark_report.md,telemetry_schema.md}
├─ Dockerfile, docker-compose.yml, requirements.lock, .env.example
└─ README.md
```

## 10. Deployment
- `docker compose up` builds the image, indexes the corpus, and serves API + console on one port.
- Config via env (`LLM_API_KEY`, `LLM_MODEL_*`, thresholds). `.env.example` committed; real `.env` ignored.
- Offline fallback: if no LLM key is present, run in **rules-only mode** (rule controller/decomposer + extractive synthesis) so G1 still completes.

## 11. Trade-offs & Decisions Log
| Decision | Alternative | Why |
|---|---|---|
| In-memory session store | Redis | Ephemeral requirement, fewer moving parts |
| Rules-first controller | Always LLM | Latency, cost, determinism |
| RRF fusion | Learned fusion | No training data; robust |
| Patch-based delta | Full regeneration | Preserves state, cuts tokens, satisfies G5 |
| Single service | Microservices | Architectural parsimony |

## 12. Failure Modes
| Failure | Handling |
|---|---|
| LLM timeout | Fall back to rules/extractive answer; log `degraded` |
| Empty retrieval | Uncertainty flag / clarification |
| Malformed LLM JSON | Retry once, then rule fallback |
| Duplicate/rapid chunks | Debounce + idempotent `chunk_id` |
| Corrupt session | Reset session, emit error event |
