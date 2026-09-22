# Frontend Specification — Streaming RAG Demo Console
**Draft v1 · Theme 4 · Samsung PRISM GenAI Hackathon 3.0**

## 1. Purpose
A single-page **demo and inspection console** that makes the invisible parts of streaming RAG *visible*: when retrieval starts, how an utterance is decomposed, how the answer evolves across versions, and which citations back each claim. It serves the demo video, the live jury demo, and your own debugging. It is **not** a consumer product; polish is secondary to clarity (UI polish is explicitly not what is judged, but *Presentation & documentation* is 10% and "Does the prototype actually work?" is 30%).

## 2. Design Principles
1. **Show the pipeline, not just the answer.** Every controller decision is visible.
2. **Time is a first-class axis.** A timeline shows chunks, decisions and the moment retrieval fires relative to utterance end.
3. **Diffs over dumps.** Answer refinement is shown as a version diff.
4. **Zero-setup.** Served by the backend at `/`; no separate frontend server for judges.
5. **Works offline** (no CDN dependencies in the final build).

## 3. Tech Stack
- **Option A (recommended, fastest):** static `index.html` + vanilla JS/ES modules + small CSS, served by FastAPI.
- **Option B:** Vite + React + TypeScript, built to `/frontend/dist` and served statically.
- Transport: WebSocket for live stream, `fetch` for replay/state. Fallback to SSE if WS is unavailable.
- Diff rendering: lightweight word-level diff (`diff-match-patch` or hand-rolled).
- Fonts/icons bundled locally.

## 4. Screen Layout (desktop, single page)

```
┌───────────────────────────────────────────────────────────────────────┐
│ Header: Streaming Live RAG · session id · status ● · [Reset session]   │
├───────────────┬───────────────────────────────┬───────────────────────┤
│ A. INPUT      │ B. LIVE PIPELINE TIMELINE     │ D. TELEMETRY          │
│  • Scenario   │  chunks ▸ controller ▸ queries │  • TTFT, latency      │
│    picker     │  ▸ retrieval ▸ synthesis       │  • tokens / cost      │
│  • Typed/mic  │                               │  • counters           │
│    simulator  ├───────────────────────────────┤  • event log (JSONL)  │
│  • Speed ctrl │ C. ANSWER PANEL               │                       │
│  • Play/Pause │  Answer vN  [v1|v2|Diff]      │                       │
│               │  claims + [Doc §Sec] chips    │                       │
│               │  ⚠ uncertainty banner         │                       │
│               │  Evidence drawer              │                       │
└───────────────┴───────────────────────────────┴───────────────────────┘
```
Responsive: below 1000 px, panels stack (Input → Timeline → Answer → Telemetry).

## 5. Components

### A. Input Panel
| Element | Behaviour |
|---|---|
| **Scenario picker** | Dropdown of bundled demo scenarios (the three from the guide: multi-intent, late detail, suppression) + "Custom" |
| **Transcript composer** | Text box; "Send chunk" appends a chunk with the current stream time; "End utterance" sends `[Utterance End]` |
| **Auto-play** | Feeds a scenario's timestamped chunks at 0.5×/1×/2×/4× speed; Pause/Step |
| **Mic simulator (P2)** | Browser SpeechRecognition → chunks (optional; may be unsupported) |
| **New turn / Reset** | Starts a new utterance in the same session / clears the session |

### B. Live Pipeline Timeline
- Horizontal time axis (seconds). Rows: **Transcript chunks**, **Controller decision**, **Retrieval events**, **Synthesis**.
- Chips coloured by decision: WAIT (grey), RETRIEVE provisional (blue), RETRIEVE multi-intent (purple), NO_RETRIEVAL (amber), Synthesize (green).
- A vertical marker at **Utterance End**; the gap between the first retrieval chip and that marker is shown as **"Head start: 1.3 s"**, which directly demonstrates gate G2.
- Hover a chip → tooltip: reason code, query text, latency, confidence.
- Sub-queries appear as parallel lanes under the decomposition chip, each with status (queued / running / done / no evidence).

### C. Answer Panel
- **Header:** `Answer v2` badge, version switcher (v1 / v2 / **Diff**).
- **Body:** streamed text, each claim followed by citation chips `[Doc_12 §2]`.
  - Click chip → opens **Evidence drawer** with the chunk text (highlighted matching span).
  - Unverified/removed claims are shown in a collapsed "Removed by verifier" section (useful for the demo).
- **Diff view:** additions (green), updates (yellow), retained (normal), with a caption such as "Retained 3 claims · Updated 1 · Added 1 · Retrieval only for delta".
- **Uncertainty banner:** ⚠ "Catering policies for Venue A could not be verified from the retrieved corpus."
- **Suppression state:** for presentation-only turns, show a "No retrieval needed — reformatted from session context" notice; citation chips are reused, none added.

### D. Telemetry Panel
- KPI tiles: **Time to first retrieval**, **Time to first token**, **End-to-end latency**, **Tokens in/out**, **Est. cost / turn**, **Cache hits**.
- Counters: retrievals fired, suppressed, sub-queries, answer versions.
- Scrollable JSONL event log with filter by type; **Download log** button.
- Link/button "Copy structured record" (the FR-8 JSON).

### E. Global Elements
- Status pill: Connected / Replaying / Degraded (rules-only mode) / Error.
- Toast for errors (e.g., LLM fallback used).
- "About this run" modal listing config (models, thresholds, corpus name/size).

## 6. Key User Flows

**Flow 1 — Incremental multi-intent**
1. Pick scenario "Workshop in Pune"; press Play.
2. Timeline: WAIT at 0.0 s → provisional retrieve at 0.8 s → decompose into three lanes at 1.6 s.
3. At utterance end, the answer streams with citations; head-start badge is shown.

**Flow 2 — Late detail refinement**
1. Ask for the reimbursement policy; see Answer v1.
2. Send "The trip was international and the booking was made after travel."
3. UI shows targeted delta retrieval only; Answer v2 with a diff; prior citations preserved.

**Flow 3 — Query suppression**
1. Send "Please repeat your last answer in two bullets."
2. Controller chip = NO_RETRIEVAL (`presentation_restructure`); bullets appear; telemetry shows zero retrieval calls.

**Flow 4 — Insufficient evidence**
1. Ask about something absent from the corpus.
2. Answer shows only supported claims plus the uncertainty banner; no fabricated citations.

## 7. Backend Contract Used by the UI

**WebSocket** `/ws/stream/{session_id}`
- Client → server: `{"type":"chunk","chunk_id":"c3","text":"…","t_start_s":1.6}` · `{"type":"utterance_end"}` · `{"type":"reset"}`
- Server → client (events):
  - `controller_decision` `{decision, trigger, reason, confidence, t}`
  - `retrieval_started` / `retrieval_done` `{sub_query_id, query, n_hits, latency_ms}`
  - `decomposition` `{sub_queries[]}`
  - `answer_token` `{text}` · `answer_complete` `{version, claims[], citations[], uncertainty}`
  - `answer_diff` `{version_from, version_to, ops[]}`
  - `telemetry` `{ttft_ms, tokens_in, tokens_out, cost_usd}`
  - `error` `{code, message}`

**REST:** `POST /replay`, `GET /session/{id}/state`, `GET /metrics`, `GET /health`.

## 8. State Model (client)
```
session: { id, status, version_current }
timeline: [ { t, kind, payload } ]
subQueries: { id -> { text, status, hits[] } }
answers: [ { version, claims[], citations[], uncertainty, diff } ]
telemetry: { kpis, events[] }
ui: { selectedVersion, drawerChunkId, speed, playing }
```
Rules: events are append-only; the answer view derives from `answers`; reset clears everything.

## 9. Visual Style
- Light theme default with dark-mode toggle; neutral background, high-contrast text.
- Palette (semantic): blue = retrieval, purple = multi-intent, green = synthesis/added, amber = suppression/uncertainty, red = error.
- Typography: system sans-serif; monospace for queries, IDs and logs.
- Spacing: 8-pt grid; cards with subtle borders.

## 10. Accessibility & Quality
- Keyboard operable (Space play/pause, → step, R reset).
- Colour is never the only signal (icons + labels on chips).
- ARIA live region for streamed answer; focus management for the evidence drawer.
- Contrast ≥ WCAG AA.
- Handles empty, loading and error states for every panel.

## 11. Acceptance Criteria
- [ ] Console loads from `docker compose up` at `/` with no external network calls.
- [ ] The three guide scenarios replay end to end and display: head-start metric, decomposition lanes, answer versions with diff, suppression notice.
- [ ] Every citation chip opens the exact chunk text.
- [ ] Uncertainty banner appears when a sub-query has no supported evidence.
- [ ] Telemetry tiles and downloadable JSONL match backend logs.
- [ ] Usable at 1280×720 (video-recording friendly) and on tablet width.
- [ ] Reconnect handling: if WS drops, UI shows Degraded and offers Retry.

## 12. Out of Scope
Speech quality, wake-word, animation polish, authentication UI, multi-user dashboards.

## 13. Demo-Video Storyboard (≤ 5 min)
| Time | Show |
|---|---|
| 0:00–0:30 | Problem + one-command startup |
| 0:30–1:30 | Flow 1: early retrieval + decomposition (timeline head-start) |
| 1:30–2:30 | Flow 2: late-detail refinement with diff |
| 2:30–3:15 | Flow 3: query suppression |
| 3:15–3:45 | Flow 4: uncertainty + citation drawer |
| 3:45–4:30 | Telemetry + benchmark results (G2–G6, baseline vs ours) |
| 4:30–5:00 | Limitations + next steps |
