# DECISIONS.md — Architecture & Implementation Decisions Log
**Project:** Streaming Live RAG Engine · Samsung PRISM GenAI Hackathon 3.0 · Theme 4

Every deviation from the official brief, from our draft docs, or a non-obvious design choice is logged here with a measured or reasoned justification. Judges and reviewers can trace every trade-off.

---

## D-001 — BGE-small-en-v1.5 as primary embedding model
**Date:** 2026-09-21  
**Decision:** Use `BAAI/bge-small-en-v1.5` (384-dim, ~130 MB) as the dense retrieval backbone instead of `all-MiniLM-L6-v2`.  
**Rationale:** BGE-small-en-v1.5 scores 5–8% higher on BEIR benchmark average (NDCG@10) at the same model size and CPU latency. Both satisfy the "small CPU embedding model" requirement from the brief.  
**Alternative considered:** `all-MiniLM-L6-v2` — slightly smaller but lower recall.  
**Deviation from draft:** Draft said "BGE-small or MiniLM" — we choose BGE-small.

---

## D-002 — FAISS IndexFlatIP (exact cosine) over HNSW
**Date:** 2026-09-21  
**Decision:** Use FAISS `IndexFlatIP` (exact inner-product / cosine with normalised vectors) instead of `IndexHNSW`.  
**Rationale:** For a corpus expected to be small (< 50k chunks), exact search is (a) fully reproducible across runs, (b) CPU-fast at that scale, (c) simpler to build and inspect. HNSW adds complexity without measurable quality gain at this scale.  
**Revisit trigger:** If corpus > 100k chunks and index build time > 30 s, switch to HNSW.

---

## D-003 — Ban entire LangChain ecosystem
**Date:** 2026-09-21  
**Decision:** Ban LangChain, LangGraph, LlamaIndex, CrewAI, and all agent orchestration frameworks, including utility-only sub-packages (e.g., `langchain_text_splitters`).  
**Rationale:** The brief explicitly penalises agent frameworks ("No agent frameworks... Plain Python + FastAPI + asyncio"). To be safe from any interpretation, we ban the entire ecosystem.  
**Impact:** We write our own chunker, prompt templates, and retry logic. Acceptable given scope.

---

## D-004 — Corpus not committed to repo
**Date:** 2026-09-21  
**Decision:** The corpus is not committed to the Git repository. It is mounted as a Docker volume (`./data/corpus:/app/data/corpus:ro`).  
**Rationale:** (a) Potential licensing/redistribution restrictions; (b) corpus may be large (PDF collection). The repo includes a `scripts/fetch_corpus.sh` placeholder and a synthetic demo corpus for CI/testing.  
**TODO(SCHEMA):** Adapt `app/retrieval/index.py` loader once the real corpus format is confirmed.

---

## D-005 — Claim graph (DAG) for delta patching instead of flat claim list
**Date:** 2026-09-21  
**Decision:** Represent the session answer as a `ClaimGraph` (directed acyclic graph) `{claim_id → sub_query_id → evidence_chunk_ids}` instead of the flat `list[Claim]` suggested in the draft architecture.  
**Rationale:** A DAG allows surgical identification of which claims are affected by a late constraint — we walk the graph from the changed sub-query to find all dependent claims. This directly satisfies G5 without guessing.  
**Risk:** More complex to implement. Mitigated by starting with a minimal two-level graph (claim → evidence_ids) and adding the sub_query link in a second pass.

---

## D-006 — Rules-only fallback mode as Day-1 baseline, not afterthought
**Date:** 2026-09-21  
**Decision:** The rules-only mode (no LLM key) is built simultaneously with the LLM path from Day 1, not retrofitted later.  
**Rationale:** G1 requires `docker compose up` to work on a clean machine with no LLM key. If rules-only mode is an afterthought it will be brittle and may miss G1.

---

## D-007 — Synthetic corpus as stand-in until real corpus arrives
**Date:** 2026-09-21  
**Decision:** Create `data/corpus/demo/` with ~20 synthetic documents (travel, corporate events, reimbursement policy domains) that match the guide's three examples (workshop in Pune, international trip, repeat in bullets).  
**Rationale:** Corpus has not been received. We cannot stall on G1, G2, G3, G4, G5 testing. The adapter ensures a < 10-minute swap.  
**TODO(SCHEMA):** Replace with real corpus when provided. Loader interface: `CorpusLoader.load() -> list[Document]`.

---

## D-008 — Provider-agnostic LLMClient using OpenAI-compatible REST API
**Date:** 2026-09-21  
**Decision:** `LLMClient` sends requests to any OpenAI-compatible endpoint (`/v1/chat/completions`). Provider is switched by changing `LLM_BASE_URL` + `LLM_API_KEY` in `.env`. Supports: OpenAI, Gemini (via Vertex AI or AI Studio OpenAI compat layer), Mistral, Groq, Ollama.  
**Rationale:** No provider is mandated for Theme 4. This approach requires zero code changes to switch.

---

## D-009 — Embedding-drift signal added to controller (research spike R2)
**Date:** 2026-09-21 (planned; will update with measured result)  
**Decision:** The retrieval controller uses a hybrid of rules AND embedding-drift signal (cosine distance between consecutive partial-utterance embeddings).  
**Hypothesis:** Firing when drift is low AND ≥1 entity is detected catches stable intent faster than a token-count rule alone, without increasing the false-trigger rate.  
**Status:** To be validated against synthetic dev set. If no measurable improvement on G2 precision/recall, revert to pure rules and log as negative result in RESEARCH_LOG.md.
