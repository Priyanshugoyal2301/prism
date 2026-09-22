# Presentation Deck — Streaming Live RAG Engine
## Samsung PRISM GenAI Hackathon 3.0 · Theme 4

---

## Slide 1 — Title

**Streaming Live RAG: Retrieval Before You Finish Speaking**

*Team:* Priyanshu Goyal
*Track:* Theme 4 — Streaming Live RAG
*Hackathon:* Samsung PRISM GenAI Hackathon 3.0

---

## Slide 2 — The Problem

### Users speak in compound, messy streams

> *"I need to plan a customer workshop in Pune for 30 people — I also need the cancellation policy and catering options, and actually the trip is international."*

**Challenges:**
- Query arrives as a **stream of chunks**, not a complete sentence
- Contains **2–4 orthogonal sub-intents** mixed together
- **Late constraints** arrive after retrieval has already started
- A naive system waits for the full utterance → **wasted seconds**
- A greedy system retrieves on every chunk → **retrieval thrash**

---

## Slide 3 — Our Solution: 5 Innovations

| Innovation | What it solves |
|------------|---------------|
| **Streaming Controller** | Fires retrieval before utterance ends — no thrash |
| **Multi-Intent Decomposer** | Parallel sub-queries → single merged answer |
| **Claim Graph (R3)** | Surgical delta patch on late constraint — no restart |
| **Speculative Cache (R4)** | Provisional result reused if final query matches |
| **Citation Verifier (R6)** | Zero hallucinated Doc_IDs — semantic support check |

---

## Slide 4 — Architecture

```
Transcript chunks (WebSocket / JSONL replay)
         │
         ▼
  ┌─────────────────┐
  │ Retrieval        │  < 5ms — 7-rule cascade
  │ Controller       │  WAIT / RETRIEVE / NO_RETRIEVAL
  └────────┬────────┘
           │ RETRIEVE
           ▼
  ┌─────────────────┐
  │ Multi-Intent     │  LLM + rules fallback
  │ Decomposer       │  R5 orthogonality guard
  └────────┬────────┘
           │ 1..N sub-queries (parallel)
           ▼
  ┌─────────────────┐
  │ Hybrid Retrieval │  BM25 + BGE-small-en-v1.5
  │ BM25 + Dense     │  FAISS + RRF fusion (k=60)
  │ + RRF            │  R4 speculative cache
  └────────┬────────┘
           │
           ▼
  ┌─────────────────┐
  │ Grounded         │  Corpus-only prompt
  │ Synthesiser      │  [Doc_ID §Section] citations
  └────────┬────────┘
           │
           ▼
  ┌─────────────────┐
  │ Citation         │  ID existence check
  │ Verifier (R6)    │  Cosine support ≥ 0.35
  └────────┬────────┘
           │
           ▼
  ┌─────────────────┐
  │ Claim Graph (R3) │  Versioned answers
  │ Session State    │  Delta patch on refinement
  └─────────────────┘
           │
           ▼
  JSONL Telemetry (G6 — 100% coverage)
```

---

## Slide 5 — Key Innovation: Streaming Controller

### 7-Rule Cascade, < 5ms, CPU-Only

```
1. Presentation guard  → "repeat/summarize" → NO_RETRIEVAL (pre-guards)
2. Min-token gate      → < 8 tokens → WAIT
3. Debounce guard      → < 0.4s since last → WAIT
4. Jaccard guard       → ≥ 0.75 similarity → WAIT (cache serves)
5. Suppression kws     → "ignore/forget" → NO_RETRIEVAL
6. Conjunction detect  → "and also/actually the..." → RETRIEVE (MULTI_INTENT)
7. Provisional         → entity + intent verb → RETRIEVE (PROVISIONAL)
```

**Result:** Retrieval fires **before** the utterance ends on compound queries, with < 5% false-positive rate.

Also implements **R2 embedding drift**: if cosine similarity of successive buffer embeddings drops below threshold, retrieval fires even without a rule match.

---

## Slide 6 — Key Innovation: Claim Graph + Delta Patch (R3)

### No full restart on late constraints

```
Initial answer v1:
  [c1] Venue: Novotel Pune, INR 5500/night          [Doc_04 §2]  ← RETAIN
  [c2] Cancellation: 48h notice required             [Doc_03 §3]  ← RETAIN
  [c3] Catering: lunch packages from INR 800/head    [Doc_03 §4]  ← UPDATE

Late constraint: "actually it's international travel"

Delta answer v2 (patch ops only):
  c1 RETAIN  (venue unchanged)
  c2 RETAIN  (cancellation unchanged)
  c3 UPDATE  "International catering: customs clearance required [Doc_05 §2]"
```

**Only the affected claim is regenerated.** E2E latency for refinement: < 100ms additional.

---

## Slide 7 — Gate Results

| Gate | Requirement | Our Score | Details |
|------|-------------|-----------|---------|
| **G1** | `docker compose up` works | ✅ Pass | No manual steps; no LLM key required |
| **G2** | Early retrieval | ✅ **3/3 (100%)** | Provisional fire pre-utterance-end |
| **G3** | Multi-intent | ✅ **2/2 eligible (100%)** | 2 sub-queries on compound utterances |
| **G4** | Grounded citations | ✅ **3/3 (100%)** | 0 hallucinated Doc_IDs |
| **G5** | Refinement | ✅ **2/2 eligible (100%)** | Delta patch v1→v2 |
| **G6** | Telemetry | ✅ **3/3 (100%)** | 100% trace coverage |

### Unit Tests: 33/33 pass

---

## Slide 8 — Performance

### Latency (Rules-Only Mode, Demo Corpus)

| Stage | Latency |
|-------|---------|
| Controller decision | < 5ms |
| BM25 retrieval (20 chunks) | < 2ms |
| Dense retrieval (20 chunks) | < 10ms |
| RRF fusion | < 2ms |
| Extractive synthesis | < 50ms |
| Citation verification | < 30ms |
| **Total E2E** | **< 300ms** |

### Latency Waterfall (Scenario 01)
```
t=0ms    Chunk 1 → WAIT (too few tokens)
t=43ms   Chunk 2 → RETRIEVE PROVISIONAL (entity + intent)
t=58ms   BM25 + Dense done (parallel)
t=90ms   First token (TTFT=32ms from synthesis start)
t=130ms  Chunk 3 (final) → RETRIEVE MULTI_INTENT
t=185ms  Delta claim patch
t=215ms  Answer v2 complete
```

### Ablation: Retrieval Strategy
| Strategy | Citation Rate | E2E (ms) |
|----------|--------------|---------|
| BM25 only | ~70% | ~80ms |
| Dense only | ~75% | ~150ms |
| **Hybrid BM25+Dense+RRF** | **~90%** | **~130ms** |

---

## Slide 9 — Reproducibility & Deployment

### One Command to Run

```bash
git clone <repo>
cp .env.example .env    # no edits needed
docker compose up       # http://localhost:8000
```

### Tech Stack (no agent frameworks)
- **FastAPI** + **asyncio** — event-driven pipeline
- **BGE-small-en-v1.5** — 384d dense retrieval, CPU-only
- **FAISS IndexFlatIP** — exact cosine search
- **rank-bm25** — Okapi BM25 sparse retrieval
- **Pydantic v2** — strict schema validation
- **structlog** — structured JSONL telemetry

### Security
- Corpus chunks tagged; prompt injection neutralised
- Session state isolated by UUID4 (no cross-session bleed)
- No secrets committed; `.env.example` has placeholder keys

---

## Slide 10 — Demo & Future Work

### Demo Scenarios
1. **Compound query** — Workshop in Pune: venue + cancellation + catering (3 sub-intents)
2. **Late detail** — Reimbursement policy → international + post-travel refinement
3. **Suppression** — Answer given, user asks "repeat in bullet points" → NO_RETRIEVAL, reformat only

### Live at `http://localhost:8000`
- WebSocket live streaming
- Scenario replay picker
- Pipeline timeline (per-stage events)
- Claim-level answer with citations
- Version switcher + diff view

### Future Work
- Cross-encoder reranker (already scaffolded, `RERANK_ENABLED=true`)
- Beam-search speculative decoding integration
- Multi-corpus routing (per-topic index shards)
- Confidence calibration with temperature sweep
