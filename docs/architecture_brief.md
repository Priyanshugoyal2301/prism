# Architecture Brief — Streaming Live RAG Engine
**Samsung PRISM GenAI Hackathon 3.0 · Theme 4**
Version: 1.0 | Date: 2026-09-21

---

## Problem Statement

A user speaks one compound, multi-intent utterance as a timestamped transcript stream. The system must:
1. Begin retrieving **before the utterance ends** without thrashing on noise
2. Decompose compound queries into **parallel orthogonal sub-intents**
3. Synthesise a **single grounded answer** with `Doc_ID §Section` citations
4. **Surgically refine** when a late constraint arrives — no full restart
5. Log **100% trace coverage** of every pipeline stage

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        Transcript Stream                          │
│         (WebSocket chunks / JSONL replay at N× speed)            │
└──────────────┬───────────────────────────────────────────────────┘
               │ ChunkEvent (chunk_id, text, t_start_s, is_final)
               ▼
┌─────────────────────────────────────┐
│       Retrieval Controller          │  < 5 ms CPU-only
│  1. Presentation check (pre-guard)  │
│  2. Min-token gate (8 tokens)       │
│  3. Debounce guard (0.4s)           │
│  4. Jaccard similarity guard        │
│  5. Suppression keywords            │
│  6. Conjunction / late-constraint   │
│  7. Provisional: entity + intent    │
│     → WAIT / RETRIEVE / NO_RETRIEVAL│
└──────────────┬──────────────────────┘
               │ RETRIEVE
               ▼
┌─────────────────────────────────────┐
│       Multi-Intent Decomposer       │
│  LLM → JSON [{id, text, priority}]  │
│  Fallback: regex conjunction split  │
│  R5 Orthogonality guard (Jaccard)   │
│  Output: 1..N sub-queries           │
└──────────────┬──────────────────────┘
               │ asyncio.gather (parallel)
               ▼
┌─────────────────────────────────────┐
│       Hybrid Retrieval              │
│  BM25 (rank-bm25 Okapi)            │
│  Dense: BGE-small-en-v1.5 + FAISS  │
│  RRF fusion (k=60)                 │
│  Dedup by doc_id + section         │
│  R4 Speculative result cache        │
└──────────────┬──────────────────────┘
               │ FusedResults per sub-query
               ▼
┌─────────────────────────────────────┐
│       Grounded Synthesiser          │
│  System prompt: corpus-only grounding│
│  Output: claims + [Doc_ID §Section] │
│  Injection defence: chunk tagging   │
│  Extractive fallback (no LLM key)  │
└──────────────┬──────────────────────┘
               │ Claims + Citations
               ▼
┌─────────────────────────────────────┐
│       Citation Verifier             │
│  ID existence check (hard reject)  │
│  R6 Semantic support (cosine ≥ 0.35)│
│  Uncertain flag on weak support    │
└──────────────┬──────────────────────┘
               │ Verified Claims
               ▼
┌─────────────────────────────────────┐
│  Session State / Claim Graph (R3)  │
│  Versioned answers (v1, v2, …)     │
│  Delta patch on late-detail        │
│  Patch ops: RETAIN / UPDATE / ADD  │
└──────────────┬──────────────────────┘
               │ AnswerDiff + TurnRecord
               ▼
┌─────────────────────────────────────┐
│       JSONL Telemetry (G6)          │
│  Every stage emits timestamped event│
│  Metrics: TTFT, E2E, cache, hits    │
└─────────────────────────────────────┘
```

---

## Design Principles

### Architectural Parsimony
No LangChain, LangGraph, or agent orchestration frameworks. Pure Python asyncio — every component is a plain function or class the team can read in 5 minutes.

### CPU-First
BGE-small-en-v1.5 (384d, ~30MB) + FAISS IndexFlatIP. Full pipeline runs on a single CPU core. GPU recommended for production but never required.

### Hard Constraints Respected
| Constraint | Implementation |
|---|---|
| G1 Reproducibility | `docker compose up` — no manual steps |
| Corpus isolation | Session store, no cross-session state leakage |
| No hard-coding | All thresholds via `app/config.py` → `.env` |
| No secrets committed | `.gitignore` + `.env.example` |
| Provider-agnostic LLM | `app/llm.py` — OpenAI-compatible REST |

---

## Research Spikes Implemented

| Spike | Description | Location |
|-------|-------------|----------|
| R2 Embedding drift | Cosine drift between successive buffer embeddings triggers retrieval even without rule match | `app/controller.py:_compute_drift` |
| R3 Claim graph | Per-claim storage enables surgical delta patching — only affected claims regenerated | `app/synthesis/session.py:build_claim_graph` |
| R4 Speculative cache | First provisional result cached; if final query matches (Jaccard≥0.75), cache hit avoids re-retrieval | `app/retrieval/fusion.py:HybridRetriever` |
| R5 Orthogonality | Sub-queries with Jaccard similarity ≥ 0.7 are merged before parallel retrieval (anti-thrash) | `app/decomposer.py:_compute_orthogonality` |
| R6 Semantic support | Every cited chunk verified by cosine similarity against the claim text | `app/synthesis/verifier.py` |

---

## Component Details

### Retrieval Controller (`app/controller.py`)
Deterministic 7-rule cascade, all decisions in < 5ms:
1. **Presentation guard** — "repeat/summarize/reformat" → NO_RETRIEVAL (checked before token guard)
2. **Token guard** — < 8 tokens → WAIT
3. **Debounce** — < 0.4s since last retrieval → WAIT
4. **Jaccard guard** — ≥ 0.75 similarity to last query → WAIT (R4 cache will serve)
5. **Suppression** — "ignore/forget/actually no" keywords → NO_RETRIEVAL
6. **Conjunction / late-constraint** — "and also", "actually the trip was..." → RETRIEVE (MULTI_INTENT)
7. **Provisional** — entity detected + intent verb → RETRIEVE (PROVISIONAL)

### Decomposer (`app/decomposer.py`)
- LLM path: structured JSON prompt → `[{id, text, priority, context_carry}]`
- Rules fallback: conjunction split → each clause inherits shared entities
- R5 guard: merge sub-queries with Jaccard ≥ 0.7

### Hybrid Retrieval (`app/retrieval/`)
- BM25: Okapi variant, tokenized on whitespace+punct, top-K=8
- Dense: BGE-small-en-v1.5, normalize→ dot-product = cosine, FAISS IndexFlatIP, top-K=8
- RRF fusion: score = Σ 1/(k+rank_i), k=60
- Dedup: keep highest-RRF chunk per (doc_id, section) pair

### Citation Verifier (`app/synthesis/verifier.py`)
Gate G4 enforcement:
1. **ID existence**: Doc_ID must exist in evidence pool → hard reject if not
2. **Semantic support**: cosine similarity (claim embedding, chunk embedding) ≥ 0.35 → else mark uncertain
3. **Injection check**: chunk_id patterns rejected from citation strings

### Session / Claim Graph (`app/synthesis/session.py`)
- `Session` stores buffer, evidence pool, claim graph, answer versions
- `ClaimGraph`: `{sub_query_id → [Claim]}` — each Claim has text, citations, support score
- Delta patch: only claims for affected sub-queries are regenerated on refinement
- Diff output: `[{op: RETAIN|UPDATE|ADD, claim_id, text_before, text_after}]`

---

## Latency Profile (Rules-Only Mode, Demo Corpus)

| Stage | Typical (ms) |
|-------|-------------|
| Controller decision | < 5 |
| Decomposition (rules) | < 20 |
| BM25 retrieval (20 chunks) | < 2 |
| Dense retrieval (20 chunks) | < 10 |
| RRF fusion | < 2 |
| Synthesis (extractive fallback) | < 50 |
| Citation verification | < 30 |
| **Total E2E (rules-only)** | **< 300 ms** |

With LLM synthesis (streaming): TTFT ≈ 800ms, full answer ≈ 2-4s depending on provider.

---

## G1 Reproducibility

```bash
git clone <repo>
cd prism-streaming-rag
cp .env.example .env    # no edits required
docker compose up       # starts app on :8000
```

No manual steps. No LLM key required for demo (rules-only synthesis). Model weights downloaded once and cached in Docker volume.

---

## Security

- **Prompt injection**: All corpus chunks tagged `[CORPUS CHUNK chunk_id=...]` in LLM context; system prompt instructs model to ignore any instruction patterns
- **Corpus isolation**: Session state never crosses session boundaries (RLock per session, dict keyed by UUID4)
- **No secrets committed**: `.env.example` has no real keys; `.gitignore` covers `.env`, `*.key`, `*.pem`
- **Input validation**: All WebSocket messages validated via Pydantic v2 before processing
