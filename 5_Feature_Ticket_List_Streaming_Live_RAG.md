# Feature Ticket List — Streaming Live RAG
**Draft v1 · Theme 4 · Samsung PRISM GenAI Hackathon 3.0**

**Deadline:** 25 Sep 2026, 11:59 PM (submit earlier; aim for a 24 Sep freeze).
**Priority:** P0 = must ship · P1 = should ship · P2 = nice to have.
**Size:** S ≈ ≤2 h · M ≈ half day · L ≈ full day.
Owners are left blank; assign by team size (1–4 members).

---

## Epic 0 — Setup & Foundations
| ID | Ticket | Pri | Size | Depends | Acceptance criteria |
|---|---|:-:|:-:|---|---|
| T-001 | Create GitHub repo, folder layout, `.gitignore`, `.env.example`, pinned deps | P0 | S | — | Repo public/shared; layout matches architecture doc; no secrets |
| T-002 | Read corpus + test-suite (released 16 Sep); document format & size | P0 | S | — | Loader spec written; open questions logged to prism@samsung.com if unclear |
| T-003 | Define Pydantic schemas: chunk event, controller decision, sub-query, turn record, telemetry events | P0 | M | T-002 | Schemas match the guide's structured output; unit tests validate examples |
| T-004 | Base FastAPI app with `GET /health` → `{"status":"ok"}` | P0 | S | T-001 | Returns 200 with expected JSON |
| T-005 | Provider-agnostic `LLMClient` (env key, timeout, retry, JSON mode, token counting) | P0 | M | T-001 | Swappable model via env; missing key → rules-only mode flag |

## Epic 1 — Corpus Indexing & Baseline Retrieval
| ID | Ticket | Pri | Size | Depends | Acceptance criteria |
|---|---|:-:|:-:|---|---|
| T-010 | Corpus loader + section-aware chunker with `doc_id` and `section` | P0 | M | T-002 | Every chunk has stable ID (`Doc_12 §2` style); tests on sample docs |
| T-011 | BM25 index | P0 | S | T-010 | Top-k query returns ranked chunks |
| T-012 | Dense embeddings + FAISS/NumPy index (CPU) | P0 | M | T-010 | Index built at start-up; top-k works |
| T-013 | Hybrid scoring + RRF fusion + dedupe | P0 | M | T-011, T-012 | Fusion unit-tested; near-duplicates removed |
| T-014 | **Baseline pipeline** (wait-for-end, single query, dense-only, one LLM answer) | P0 | M | T-013, T-005 | Runs on the same scenarios; used for comparison in the report |
| T-015 | Optional cross-encoder rerank behind a flag | P1 | M | T-013 | Toggle via env; used in ablation |

## Epic 2 — Streaming Ingestion & Replay Harness
| ID | Ticket | Pri | Size | Depends | Acceptance criteria |
|---|---|:-:|:-:|---|---|
| T-020 | Chunk ingestion endpoint (`WS /ws/stream/{id}`) + input validation | P0 | M | T-003, T-004 | Accepts `{chunk_id,text,t_start_s}`; rejects malformed input |
| T-021 | Replay runner (CLI + `POST /replay`) that plays timestamped scenarios at 1×/accelerated | P0 | M | T-020 | Deterministic event log output |
| T-022 | Per-session utterance buffer + session store with TTL | P0 | M | T-020 | Sessions isolated; TTL eviction test |
| T-023 | Convert the three guide examples into bundled scenario files | P0 | S | T-021 | Scenarios replay end-to-end |

## Epic 3 — Retrieval Controller
| ID | Ticket | Pri | Size | Depends | Acceptance criteria |
|---|---|:-:|:-:|---|---|
| T-030 | Rules-based controller: WAIT / RETRIEVE / NO_RETRIEVAL with reason codes | P0 | L | T-022 | On Example 1: WAIT @0.0, provisional @0.8, multi-intent @1.6 |
| T-031 | Entity/intent stability detector (place, quantity, date, intent verbs) | P0 | M | T-030 | Fires only when stable; unit tests on partial phrases |
| T-032 | Debounce + similarity guard against duplicate dispatches | P0 | S | T-030 | No repeat retrieval for near-identical queries |
| T-033 | Presentation-only detector (repeat / shorten / bullets / translate) | P0 | S | T-030 | Example 3 → `retrieval_required:false`, reason `presentation_restructure` |
| T-034 | LLM classifier fallback for ambiguous cases | P1 | M | T-005, T-030 | Used only when rules confidence low; logged |
| T-035 | Tune thresholds against labelled streams; report false-trigger rate | P1 | M | T-030, T-052 | G2 metric computed |

## Epic 4 — Multi-Intent Decomposition & Parallel Retrieval
| ID | Ticket | Pri | Size | Depends | Acceptance criteria |
|---|---|:-:|:-:|---|---|
| T-040 | Decomposer prompt + strict JSON output + parser with retry | P0 | M | T-005 | Example 1 → 3 orthogonal sub-queries with "Pune/30" context carried |
| T-041 | Rule-based decomposition fallback (conjunction/comma split + context resolution) | P0 | M | T-040 | Works with no LLM key |
| T-042 | Over-fragmentation guard (cap N, embedding-similarity dedupe) | P0 | S | T-040 | Simple question stays 1 query |
| T-043 | Parallel sub-query retrieval (`asyncio.gather`) + per-sub-query provenance | P0 | M | T-013, T-040 | Latency ≈ slowest sub-query, not sum |
| T-044 | Provisional-retrieval cache reused at final decomposition | P1 | S | T-043 | Cache hit logged; no duplicate search |

## Epic 5 — Synthesis, Grounding & Uncertainty
| ID | Ticket | Pri | Size | Depends | Acceptance criteria |
|---|---|:-:|:-:|---|---|
| T-050 | Grounded synthesizer: claims-with-citations JSON, streamed output | P0 | L | T-043, T-005 | Every claim has ≥1 `[Doc §Sec]` or is marked uncertain |
| T-051 | Citation verifier (ID exists in pool + support check) | P0 | M | T-050 | Fabricated ID such as `Doc_999` is always removed; logged as violation |
| T-052 | Uncertainty flag / targeted clarification when evidence is missing | P0 | S | T-050 | Example 1 emits "catering for Venue A unverified" style entry |
| T-053 | Extractive/rules-only synthesis fallback | P1 | M | T-050 | Answers without an LLM key (G1 safety net) |
| T-054 | Reformat mode (no retrieval; bullets/short from session claims) | P0 | S | T-033, T-050 | Example 3 passes; no new citations |

## Epic 6 — Session Refinement (Answer Delta)
| ID | Ticket | Pri | Size | Depends | Acceptance criteria |
|---|---|:-:|:-:|---|---|
| T-060 | Session state model: versions, claims→citations, sub-queries, evidence pool | P0 | M | T-022, T-050 | Serializable; `answer_version` increments |
| T-061 | Late-detail classifier: refine existing topic vs new topic | P0 | M | T-060 | Example 2 recognised as refinement |
| T-062 | Delta retrieval (only affected sub-queries) | P0 | M | T-061, T-043 | No full-corpus re-run; asserted in tests |
| T-063 | Patch-based answer update (add / update / retain) with citation preservation | P0 | L | T-062, T-051 | Example 2 → v2 keeps base-policy claim + new exceptions with delta citations |
| T-064 | Answer diff generator for UI | P1 | S | T-063 | Ops list `{retained, updated, added}` |

## Epic 7 — Telemetry & Observability
| ID | Ticket | Pri | Size | Depends | Acceptance criteria |
|---|---|:-:|:-:|---|---|
| T-070 | Telemetry event schema + JSONL writer (timestamps, decisions, sources, versions, tokens) | P0 | M | T-003 | 100% of turns have complete traces (G6) |
| T-071 | Instrument all stages (controller, retrieval, fusion, synthesis, verifier) | P0 | M | T-070 | Event coverage test |
| T-072 | Cost-per-turn + TTFT calculation | P0 | S | T-071 | Values appear in `/metrics` and per-turn record |
| T-073 | `GET /metrics` aggregate endpoint | P1 | S | T-071 | JSON snapshot |

## Epic 8 — Benchmarking & Evaluation
| ID | Ticket | Pri | Size | Depends | Acceptance criteria |
|---|---|:-:|:-:|---|---|
| T-080 | Build a small labelled dev set (compound, late-detail, suppression, no-answer cases) from the corpus; **do not commit organiser-held-out data** | P0 | M | T-002 | ≥ 30 streams across categories |
| T-081 | Metrics script for G2–G6 + recall@k, groundedness, TTFT, cost/turn | P0 | M | T-080, T-071 | One command prints a table |
| T-082 | Ablation 1: hybrid vs dense-only | P0 | S | T-081 | Result table with commentary |
| T-083 | Ablation 2: rule-based vs model-based controller | P0 | S | T-034, T-081 | Result table with commentary |
| T-084 | Edge-case failure analysis (≥ 3 cases with root cause + mitigation) | P0 | M | T-081 | Included in the benchmark report |
| T-085 | Baseline vs streaming comparison | P0 | S | T-014, T-081 | Table for the deck |

## Epic 9 — Security & Compliance
| ID | Ticket | Pri | Size | Depends | Acceptance criteria |
|---|---|:-:|:-:|---|---|
| T-090 | Prompt-injection hardening (delimited chunks, no tools, JSON-only outputs) | P0 | S | T-050 | Injection tests pass |
| T-091 | Input limits, rate limiting, timeouts, TTL eviction | P1 | S | T-020 | Oversized/flood tests rejected |
| T-092 | Secret scan + `pip-audit` before tagging | P0 | S | T-001 | Clean reports |
| T-093 | Confirm no hard-coded prompts/answers and no external knowledge calls | P0 | S | all | Checklist signed off |
| T-094 | Optional token auth mode + admin-route gating | P2 | S | T-004 | Off by default; works when enabled |

## Epic 10 — Frontend Demo Console
| ID | Ticket | Pri | Size | Depends | Acceptance criteria |
|---|---|:-:|:-:|---|---|
| T-100 | Static console shell + WebSocket client + status pill | P1 | M | T-020 | Connects, shows session ID |
| T-101 | Scenario picker + play/pause/step/speed | P1 | M | T-023, T-100 | Guide scenarios play in UI |
| T-102 | Pipeline timeline with head-start badge | P1 | L | T-100, T-071 | Shows WAIT/RETRIEVE/NO_RETRIEVAL/Synthesize at correct times |
| T-103 | Answer panel with citation chips + evidence drawer | P1 | M | T-050 | Click chip → exact chunk |
| T-104 | Version switcher + diff view | P1 | M | T-064 | v1/v2 diff visible |
| T-105 | Telemetry panel + log download | P1 | M | T-073 | KPIs match backend |
| T-106 | Uncertainty banner + suppression notice | P1 | S | T-052, T-054 | Both states render |
| T-107 | Dark mode, responsive stack, keyboard shortcuts, a11y pass | P2 | M | T-100 | Meets spec §10 |

## Epic 11 — Packaging & Reproducibility (Gate G1)
| ID | Ticket | Pri | Size | Depends | Acceptance criteria |
|---|---|:-:|:-:|---|---|
| T-110 | Dockerfile (non-root, pinned base, build-time index) | P0 | M | T-013 | Image builds offline of corpus |
| T-111 | `docker-compose.yml` single-command startup + healthcheck | P0 | S | T-110 | `docker compose up` → `/health` OK |
| T-112 | Automated replay on clean machine/VM test | P0 | M | T-111, T-021 | Suite completes with no manual steps |
| T-113 | README with reproducible setup, config table, troubleshooting | P0 | M | T-111 | A stranger can run it in < 10 min |
| T-114 | Pinned lockfile + model revision pinning | P0 | S | T-001 | Deterministic install |

## Epic 12 — Submission Deliverables
| ID | Ticket | Pri | Size | Depends | Acceptance criteria |
|---|---|:-:|:-:|---|---|
| T-120 | Architecture brief (≤ 6 pages): rationale, trigger logic, decomposition, provenance, trade-offs, failure mitigations | P0 | M | T-030…T-063 | ≤ 6 pages |
| T-121 | Benchmark & evaluation report | P0 | M | T-082…T-085 | Includes baseline comparison, ≥3 failures, ≥2 ablations |
| T-122 | Telemetry & observability schema doc | P0 | S | T-070 | Matches emitted logs |
| T-123 | Record demo video (≤ 5 min) following storyboard | P0 | M | T-102 or CLI | Shows early retrieval, multi-intent, refine, suppression, citations, telemetry |
| T-124 | Fill `CollegeName_TeamName` PPT (12 slides incl. checklist) | P0 | M | T-121 | Filename follows convention; all slides complete |
| T-125 | Commit PPT, video link/file, docs into the repo | P0 | S | T-120…T-124 | Everything referenced exists in the tagged commit |
| T-126 | Create annotated tag `PRISM_GENAI_HACKATHON_Y2026` on final commit and push | P0 | S | T-125 | `git tag -a PRISM_GENAI_HACKATHON_Y2026 -m "PRISM Gen AI Hackathon Y2026 Final Submission"` then `git push origin PRISM_GENAI_HACKATHON_Y2026` |
| T-127 | Submit via Google Form (one submission per team) | P0 | S | T-126 | Confirmation received **before 11:59 PM, 25 Sep** |
| T-128 | Confirm team registration was completed by 16 Sep (final submission link is only shared with registered teams) | P0 | S | — | **Do first.** Contact prism@samsung.com immediately if not |

---

## Suggested Sprint Plan (5 working days)
| Day | Focus | Tickets |
|---|---|---|
| **Sun 20 Sep** | Foundations, corpus, baseline, harness | T-128, T-001–T-005, T-010–T-014, T-020–T-023 |
| **Mon 21 Sep** | Controller + decomposer + parallel retrieval | T-030–T-033, T-040–T-043 |
| **Tue 22 Sep** | Synthesis, grounding, uncertainty, telemetry | T-050–T-054, T-070–T-072, T-090 |
| **Wed 23 Sep** | Refinement + benchmarks + Docker | T-060–T-063, T-080–T-081, T-110–T-114 |
| **Thu 24 Sep** | Ablations, console (if time), docs, video, PPT, **feature freeze** | T-082–T-085, T-100–T-106, T-120–T-125 |
| **Fri 25 Sep** | Clean-machine test, secret scan, tag, submit (target before evening) | T-092, T-093, T-112, T-126, T-127 |

## Cut Line (if time runs short)
1. **Never cut:** T-128, hybrid retrieval, controller, decomposition, cited synthesis + verifier, session refinement, telemetry, Docker/README, tag + submission.
2. **Cut first:** T-107, T-094, T-015, T-034/T-083 (keep a documented rules-only comparison instead), T-044.
3. **Console fallback:** if the UI slips, present the replay CLI output and telemetry JSON in the demo video; the brief scores on functionality, not UI polish.

## Definition of Done (per ticket)
Code merged to `main` · tests or replay scenario covering the acceptance criteria · telemetry emitted where relevant · README/docs updated if behaviour or config changed · no secrets committed.
