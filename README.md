# Streaming Live RAG Engine
**Samsung PRISM GenAI Hackathon 3.0 · Theme 4**

An event-driven RAG system that starts retrieving *before* the user finishes speaking, decomposes compound queries into parallel sub-queries, synthesises one grounded cited answer, and refines it surgically when a late detail arrives — without restarting.

---

## Quick Start (One Command)

```bash
# 1. Clone and enter the repo
git clone <repo-url> && cd prism-streaming-rag

# 2. Copy environment template (LLM key is optional — runs in rules-only mode without it)
cp .env.example .env
# Edit .env and set LLM_API_KEY=<your-key> for LLM synthesis (optional)

# 3. Add your corpus (or use the demo corpus)
# Real corpus: place files in data/corpus/  (JSONL/JSON/Markdown/TXT)
# Demo corpus is already at data/corpus/demo_corpus.json

# 4. Start everything
docker compose up

# App is live at http://localhost:8000
# API docs at http://localhost:8000/api/docs
```

**No manual steps. No LLM key required for G1 reproducibility.**

---

## Architecture at a Glance

```
Transcript chunks (WS / replay)
         │
         ▼
  Retrieval Controller          WAIT / NO_RETRIEVAL / RETRIEVE
  (rules-first, <5ms)           (entity+intent stability + drift signal)
         │ RETRIEVE
         ▼
  Multi-Intent Decomposer       1..N orthogonal sub-queries (LLM or rules)
         │
         ▼ (asyncio.gather — parallel)
  Hybrid Retrieval              BM25 + BGE-small-en-v1.5 + FAISS → RRF
         │
         ▼
  Grounded Synthesiser          Claims + [Doc_ID §Section] citations
  + Citation Verifier           ID existence + semantic support check
         │
         ▼
  Session / Claim Graph         Versioned answer, delta patching on refinement
         │
         ▼
  JSONL Telemetry               100% trace coverage (G6)
```

---

## Gates & Performance

| Gate | Target | Status |
|------|--------|--------|
| G1 Reproducibility | `docker compose up` works offline | ✅ |
| G2 Early retrieval | ≥80% of eligible queries start before utterance end | Measured via `bench/replay.py` |
| G3 Multi-intent | ≥70% of compound utterances → ≥2 correct sub-intents | Measured via `bench/replay.py` |
| G4 Grounding | ≥85% citation support; 0 hallucinated Doc IDs | Citation verifier enforces |
| G5 Refinement | Delta-only update, no full restart | Claim graph patch ops |
| G6 Telemetry | 100% trace coverage | All stages instrumented |

---

## Running the Benchmark

```bash
# Run all bundled scenarios
python bench/replay.py --all

# Run a specific scenario
python bench/replay.py --scenario scenario_01_workshop_pune

# Run with verbose telemetry output
python bench/replay.py --scenario scenario_02_late_detail_trip --verbose
```

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Configuration

All settings are in `.env`. Key options:

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_API_KEY` | *(empty)* | Leave blank for rules-only mode (G1) |
| `LLM_BASE_URL` | OpenAI | Any OpenAI-compatible endpoint (Gemini, Mistral, Ollama) |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Dense retrieval model |
| `RERANK_ENABLED` | `false` | Enable cross-encoder rerank (P1 ablation) |
| `CONTROLLER_DEBOUNCE_S` | `0.4` | Minimum seconds between retrievals |
| `CORPUS_PATH` | `data/corpus` | Path to corpus directory |
| `DEBUG_ADMIN` | `false` | Enable `/session/{id}/state` debug endpoint |

---

## Corpus Format

The loader accepts files in `data/corpus/`. Supported formats:

**JSONL / JSON** (recommended):
```json
{
  "doc_id": "Doc_01",
  "title": "Corporate Event Venue Guide",
  "sections": [
    {"id": "§1", "heading": "Overview", "body": "..."},
    {"id": "§2", "heading": "Capacity", "body": "..."}
  ]
}
```

**Markdown** (.md): `##` headings become section boundaries.
**TXT**: Treated as a single-section document.

> **TODO(SCHEMA):** Adapt the loader once the official corpus format is confirmed. The adapter is in `app/retrieval/index.py:CorpusLoader`.

---

## API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check + index status |
| `/ws/stream/{session_id}` | WS | Live chunk streaming |
| `/replay` | POST | Run a scripted scenario |
| `/session/{id}/reset` | POST | Clear session state |
| `/session/{id}/state` | GET | Debug: session state (requires `DEBUG_ADMIN=true`) |
| `/metrics` | GET | Aggregate telemetry |
| `/scenarios` | GET | List bundled scenarios |
| `/` | GET | Demo console |
| `/api/docs` | GET | OpenAPI documentation |

---

## WebSocket Protocol

**Client → Server:**
```json
{"type": "chunk", "chunk_id": "c3", "text": "...", "t_start_s": 1.6}
{"type": "utterance_end"}
{"type": "reset"}
```

**Server → Client:**
```json
{"type": "controller_decision", "payload": {"decision": "RETRIEVE", "trigger": "provisional", "reason": "provisional_stable"}}
{"type": "decomposition", "payload": {"sub_queries": [...]}}
{"type": "retrieval_started", "payload": {"sub_query_id": "sq1", "query": "..."}}
{"type": "answer_complete", "payload": {"version": 1, "answer": "...", "citations": ["Doc_01 §2"]}}
{"type": "answer_diff", "payload": {"version_from": 1, "version_to": 2, "ops": [...]}}
```

---

## Repository Layout

```
app/
  main.py              FastAPI app wiring + startup
  gateway.py           Pipeline orchestrator + WS/replay
  controller.py        Retrieval controller (rules + drift signal)
  decomposer.py        Multi-intent decomposer
  llm.py               Provider-agnostic LLM client
  telemetry.py         JSONL telemetry writer
  config.py            Settings from environment
  schemas.py           All Pydantic v2 models
  retrieval/
    index.py           Corpus loader + chunker + IndexManager
    bm25.py            BM25 sparse retrieval
    dense.py           Dense (BGE-small) + FAISS
    fusion.py          RRF + speculative cache
    rerank.py          Cross-encoder rerank (P1, behind flag)
  synthesis/
    session.py         Session state + claim graph + delta patching
    synth.py           Grounded synthesiser + reformat
    verifier.py        Citation verifier (ID + semantic support)
bench/
  replay.py            CLI replay + G2-G6 metrics
  generate_streams.py  Synthetic stream generator (dev/eval set)
  scenarios/           Bundled JSONL scenario files
data/corpus/           Corpus files (not committed)
docs/                  Architecture brief, benchmark report, telemetry schema
frontend/              Demo console (static HTML/JS)
tests/                 Unit tests
```

---

## Submission

Tag: `PRISM_GENAI_HACKATHON_Y2026`

```bash
git tag -a PRISM_GENAI_HACKATHON_Y2026 -m "PRISM Gen AI Hackathon Y2026 Final Submission"
git push origin PRISM_GENAI_HACKATHON_Y2026
```

Deadline: **25 Sep 2026, 11:59 PM IST**
