# RESEARCH_LOG.md — Research Spikes & Ablation Results
**Project:** Streaming Live RAG Engine · Samsung PRISM GenAI Hackathon 3.0 · Theme 4

Each spike records: hypothesis, what was built, how it was measured, the result (including negative), and whether it was adopted. Negative results are explicitly valuable — they become our ablation material for judges.

---

## Template
```
### R-XXX — [Name]
**Date:** YYYY-MM-DD
**Hypothesis:** [What we expected]
**Method:** [What we built and how we measured]
**Result:** WIN / LOSS / NEUTRAL — [numbers]
**Adopted:** YES / NO — [reason]
**References:** [papers, repos read]
```

---

## R-001 — Synthetic Stream Generator (Dev/Eval Set)
**Date:** TBD  
**Hypothesis:** A seeded generator that produces compound / late-detail / suppression / disfluent / adversarial streams from the corpus with auto-derived ground-truth labels will give us a reproducible dev/eval set independent of the organiser's held-out data.  
**Method:** Build `bench/generate_streams.py`. Seed with `--seed 42`. Generate ≥30 streams across: compound (2+ sub-intents), late-detail (constraint arrives after first chunk), suppression (reformatting request), no-answer (query not in corpus), disfluent (um/actually/wait), adversarial (injection in transcript or corpus chunk).  
**Result:** TBD  
**Adopted:** TBD  
**References:** The guide's three examples (multi-intent workshop, late-detail trip, repeat-in-bullets); general synthetic data generation practice.

---

## R-002 — Embedding-Drift Trigger vs Pure Rules
**Date:** TBD  
**Hypothesis:** Measuring cosine distance between consecutive partial-utterance embeddings and firing provisional retrieval when drift < threshold AND ≥1 named entity is detected will improve G2 precision/recall over a pure token-count/keyword rule.  
**Method:** Implement both controllers; run both on the synthetic dev set; compare: (a) false-trigger rate on no-retrieval cases, (b) head-start in seconds on eligible queries, (c) G2 score (≥80% eligible queries start before utterance end).  
**Result:** TBD  
**Adopted:** TBD — adopt hybrid if it improves G2 without degrading false-trigger rate.  
**References:** Semantic stability ideas from streaming NLP literature; embedding-drift as a change-point detector.

---

## R-003 — Claim-Level Answer Graph for Delta Patching
**Date:** TBD  
**Hypothesis:** Representing the answer as a DAG {claim → sub_query → evidence_chunks} allows surgical identification and patching of only the claims affected by a late constraint, preserving all unaffected citations and claims. This should directly satisfy G5 and reduce re-generation tokens by >50% vs even a targeted re-synthesis.  
**Method:** Implement `ClaimGraph` with patch operations (add / update / retain). Run the "international trip" late-detail scenario; measure: (a) which claims were patched vs retained, (b) tokens consumed for the delta vs a full re-synthesis, (c) whether prior citations survive in the patched answer.  
**Result:** TBD  
**Adopted:** TBD  
**References:** Draft architecture doc §4.5; G5 requirement.

---

## R-004 — Speculative Retrieval Cache
**Date:** TBD  
**Hypothesis:** Caching the provisional BM25-only search result keyed by `(session_id, normalised_query)` and reusing it at full decomposition will reduce TTFT when the provisional and final sub-queries overlap.  
**Method:** Track cache hit rate on synthetic dev set; measure TTFT with and without cache.  
**Result:** TBD  
**Adopted:** TBD  
**References:** Draft architecture §4.4 "Provisional cache".

---

## R-005 — Orthogonality Score for Decomposition Guard
**Date:** TBD  
**Hypothesis:** Computing pairwise cosine similarity of sub-query embeddings and merging pairs > 0.85 will prevent over-fragmentation without requiring an LLM call.  
**Method:** Run decomposer on compound utterances; measure: (a) average pairwise similarity before/after guard, (b) G3 score with/without guard, (c) cases where the guard incorrectly merged orthogonal queries.  
**Result:** TBD  
**Adopted:** TBD  

---

## R-006 — Claim-Support Scoring Beyond ID Existence
**Date:** TBD  
**Hypothesis:** Checking cosine similarity between claim text embedding and cited chunk embedding (with a threshold ~0.35) will catch hallucinated-but-plausible claims that pass the ID-existence check, improving G4 precision.  
**Method:** Inject known-unsupported claims into a test scenario; measure false-accept and false-reject rates with and without the embedding support check.  
**Result:** TBD  
**Adopted:** TBD  
**References:** NLI-based citation checking literature; Cross-encoder approach considered but deferred due to CPU cost.

---

## R-007 — Section-Aware Chunking vs Naive Fixed-Size
**Date:** TBD  
**Hypothesis:** Section-aware chunking (respecting document headings, keeping stable `Doc_ID §Section` IDs) will produce better recall@5 than naive 300-token fixed-size chunks with 40-token overlap.  
**Method:** Index the same corpus with both strategies; run the same queries; compare recall@5 on the synthetic dev set.  
**Result:** TBD  
**Adopted:** Section-aware is the default (needed for stable citation IDs regardless); this ablation quantifies the recall benefit.  

---

## R-008 — Rules-First Latency Waterfall
**Date:** TBD  
**Hypothesis:** The rules-only controller path costs < 5 ms per chunk vs > 200 ms for an LLM-classifier call. Documenting this waterfall justifies the rules-first design in the architecture brief.  
**Method:** Profile with `time.perf_counter` around each stage; plot/table the waterfall: chunk_received → controller_decision → retrieval_started → retrieval_done → synthesis_first_token → synthesis_done.  
**Result:** TBD  
**Adopted:** Yes (documentation; the rules-first design is adopted regardless).  
