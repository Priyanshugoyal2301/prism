# PRD — Streaming Live RAG Engine
**Samsung PRISM GenAI Hackathon 3.0 · Theme 4 · Draft v1**

> **Assumptions to verify:** The corpus and the held-out test suite were announced for release on **16 Sep 2026** (FAQ Q14). Anything below that depends on corpus format or test-suite format is marked ⚠️ and should be confirmed once you have them. Final submission closes **25 Sep 2026, 11:59 PM**.

---

## 1. Summary
An event-driven RAG engine that consumes a **timestamped transcript stream**, starts retrieving **before the user finishes speaking**, splits one natural utterance into **several sub-queries**, fuses the evidence into **one grounded, cited answer**, and **refines that answer in place** when a late detail arrives, instead of restarting.

## 2. Problem
| Pain in standard RAG | Consequence |
|---|---|
| Waits for the full utterance | Multi-second dead air |
| One utterance hides several needs (capacity + cancellation + catering) | Single-query retrieval misses parts |
| Late constraint ("actually, it was international") | Context is discarded or the pipeline restarts |
| Users never phrase queries for a retriever | Poor recall |

## 3. Goals & Non-Goals
**Goals**
1. Start retrieval speculatively on a stable partial intent.
2. Decompose compound utterances into ≥2 sub-intents and retrieve in parallel.
3. Refine answers via a delta update (versioned), not a full re-run.
4. Every factual claim cites `[Doc_ID §Section]`; emit an explicit uncertainty flag when the corpus is insufficient.
5. Suppress retrieval for presentation-only turns ("repeat in two bullets").
6. Emit structured telemetry for every decision.
7. One-command reproducible run (`docker compose up`).

**Non-Goals**
- Speech recognition / synthesis (voice input is simulated from transcripts).
- Cross-session user profiles or persistent memory.
- Web search or any knowledge outside the provided corpus.
- Multi-agent orchestration frameworks (the brief penalises unjustified complexity).
- Wake-word detection; heavy UI polish.

## 4. Users & Personas
| Persona | Need |
|---|---|
| **Evaluator / Judge** | Run the container, replay held-out streams, inspect gates G1–G6 and telemetry |
| **End user (simulated caller)** | Speak naturally, get a correct, cited answer quickly, add details mid-flow |
| **Developer (team)** | Tune thresholds, run ablations, read traces |

## 5. Functional Requirements

### FR-1 Stream Ingestion
- Accept transcript chunks `{session_id, chunk_id, text, t_start_s, is_final}`. Support both live WebSocket and offline **replay** from a file at real-time or accelerated speed.
- Maintain a per-session rolling buffer of the current utterance.

### FR-2 Retrieval Controller (Wait / Retrieve / Suppress)
- After each chunk, output one decision: **WAIT**, **RETRIEVE** (provisional or multi-intent), or **NO_RETRIEVAL**.
- **RETRIEVE** when stable entities/intent are detected (e.g., city + capacity) or a new sub-intent appears.
- **NO_RETRIEVAL** for format/style/summarise-previous requests; log `retrieval_required:false` with a reason code.
- Debounce so it does not fire on every token (pitfall #1).
- Log reason codes: `intent_incomplete`, `provisional_stable`, `multi_intent`, `presentation_restructure`, `late_constraint`.

### FR-3 Multi-Intent Decomposition
- Convert the utterance so far into 1..N **orthogonal** sub-queries; avoid near-duplicates (pitfall #5).
- Keep carried-over context (e.g., "Pune", "30 people") in each sub-query.
- Simple, single-intent utterances must yield exactly 1 query.

### FR-4 Retrieval, Fusion & Rerank
- Hybrid **dense + sparse (BM25)** per sub-query, run in parallel.
- Merge with **Reciprocal Rank Fusion**, deduplicate chunks, optional lightweight rerank.
- Provisional results are cached per session so later sub-queries reuse them.

### FR-5 Session-Aware Synthesis
- Generate one answer covering all sub-intents, streamed token by token.
- Only use retrieved chunks; attach `[Doc_ID §Section]` citations.
- **Grounding check:** each cited ID must exist and the cited chunk must support the claim; drop or flag failures.
- If a sub-intent has no evidence → emit an `uncertainty` entry (or a targeted clarifying question).

### FR-6 Incremental Refinement (Answer Delta)
- Store per session: answer version, claims→citations map, sub-queries, evidence pool.
- On a late detail: classify as **refine** vs **new topic**; query **only the delta**; mutate only affected claims; keep prior citations; bump `answer_version`.
- Never clear session state or re-run the full search on a refine.

### FR-7 Observability
- Structured JSONL events with timestamps, retrieval triggers, source mappings, answer version transitions, and token/cost estimates. 100% trace coverage.

### FR-8 Output Contract
Per turn, emit the structured record:
```json
{
  "retrieval_events": [{"timestamp_s": 0.8, "query": "...", "trigger": "provisional"}],
  "sub_queries": ["..."],
  "answer": "...",
  "answer_version": 1,
  "citations": ["Doc_12 §2"],
  "uncertainty": "..."
}
```
⚠️ Confirm exact field names against the test-suite schema when released.

### FR-9 Demo Console (see Frontend Spec)
Lightweight UI to type or replay a stream and watch controller decisions, sub-queries, versioned answer, citations and telemetry live.

## 6. Success Metrics (mapped to the brief's gates)
| Gate | Target | How we'll measure |
|---|---|---|
| G1 Reproducibility | Pass | Fresh machine → `docker compose up` → replay suite completes unattended |
| G2 Early retrieval | ≥ 80% of eligible queries begin retrieval before utterance end; low false-trigger rate on no-retrieval cases | Replay harness compares first `retrieval_started` to `[Utterance End]` |
| G3 Multi-intent | ≥ 70% of compound utterances yield ≥2 correct sub-intents | Labelled compound set |
| G4 Grounding | ≥ 85% citation support; **0** hallucinated Doc IDs | Automated citation verifier |
| G5 Refinement | State continuity; no full restart | Version lineage check; assert no full-corpus rerun |
| G6 Telemetry | 100% trace coverage | Schema validator on every turn |

Also report (per the theme card): **retrieval recall, answer groundedness, time-to-first-token, cost per turn**.

## 7. Scope & Prioritisation
| Priority | Scope |
|---|---|
| **P0 (must ship)** | Replay ingestion, controller, decomposer, hybrid retrieval + RRF, cited synthesis, uncertainty flag, session refine, telemetry, Docker one-command, README |
| **P1** | Demo console, cross-encoder rerank ablation, benchmark report with 3 failure analyses + 2 ablations, architecture brief (≤ 6 pages) |
| **P2** | Live WebSocket streaming UI polish, cost dashboard, clarifying-question generation |

## 8. Deliverables (from Theme 4 guide + hackathon rules)
- Public/shared GitHub repo, README with reproducible setup, Docker files, pinned lockfiles, env templates
- Release tag **`PRISM_GENAI_HACKATHON_Y2026`** on the final commit (everything referenced must be inside it)
- Architecture brief (≤ 6 pages), benchmarking report (baseline comparison, ≥3 edge-case failures, ≥2 ablations), telemetry schema
- Demo video ≤ 5 min: early retrieval, multi-intent, late-detail refinement, query suppression, citation traceability, telemetry
- PPT named `CollegeName_TeamName` using the provided 12-slide template

## 8b. Hard Constraints (non-negotiable)
Corpus isolation · no hard-coding or precomputed answers · every claim cited or flagged uncertain · session-bound ephemeral memory · justified architectural complexity.

## 9. Risks & Mitigations
| Risk | Mitigation |
|---|---|
| Premature retrieval thrash | Debounce + stability check; measure false-trigger rate |
| Over-fragmented sub-queries | Cap N, dedupe by embedding similarity before dispatch |
| Late constraint wipes context | Versioned session store; delta-only retrieval |
| Fabricated citations | Post-generation verifier strips unsupported claims |
| LLM latency/cost | Rule-based controller fast-path; small model for decomposition; cache |
| Only ~5 days left | Strict P0 cut line; build harness first |
| Unknown corpus/test format | Adapter layer for loaders; confirm 16 Sep release |

## 10. Timeline (target)
| Day | Milestone |
|---|---|
| 20–21 Sep | Corpus ingest + hybrid baseline + replay harness + schemas |
| 22 Sep | Controller + decomposer + RRF |
| 23 Sep | Session refinement + grounding verifier + telemetry |
| 24 Sep | Benchmarks/ablations, Docker, README, console |
| 25 Sep | Freeze, demo video, PPT, tag release, submit (before 11:59 PM) |

## 11. Open Questions
1. Corpus format/size and exact test-suite schema (due 16 Sep).
2. Is a specific LLM/API expected for scoring? (Theme 2 FAQ gives bonus for Gemini/Mistral; nothing stated for Theme 4.) Ask prism@samsung.com.
3. Will judges call the app over HTTP or run an offline replay CLI? Support both.
4. Compute limits for Theme 4 (Theme 5 FAQ mentions a 48 GB GPU; not confirmed here). Design for CPU-friendly models.
