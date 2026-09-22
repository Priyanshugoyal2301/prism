"""
main.py — FastAPI application entrypoint.

Wiring:
- Startup: load corpus, build index
- Routes: /health, /ws/stream/{session_id}, /replay, /session/{id}/state,
          /session/{id}/reset, /metrics, / (demo console)
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from app.config import get_settings
from app.gateway import run_replay, websocket_handler
from app.retrieval.index import get_index_manager
from app.schemas import (
    HealthResponse,
    MetricsResponse,
    ReplayRequest,
    ReplayResponse,
)
from app.synthesis.session import get_session_store
from app.telemetry import get_telemetry


# ─────────────────────────────────────────────────────────────────────────────
# Startup / shutdown
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = get_settings()
    index = get_index_manager()

    print(f"[Startup] Streaming Live RAG v{cfg.app_version}")
    print(f"[Startup] Rules-only mode: {cfg.rules_only_mode}")
    print(f"[Startup] Corpus path: {cfg.corpus_path}")

    try:
        index.build(cfg.corpus_path)
        print(f"[Startup] Index ready: {len(index.chunks)} chunks")
    except FileNotFoundError as e:
        print(f"[Startup] WARNING: {e}")
        print("[Startup] Running WITHOUT corpus index. Only rules-only mode will work.")
    except Exception as e:
        print(f"[Startup] ERROR building index: {e}")

    # Ensure log directory exists
    Path("logs").mkdir(exist_ok=True)

    yield  # Application runs here

    print("[Shutdown] Closing telemetry writer...")
    get_telemetry().close()
    print("[Shutdown] Done.")


# ─────────────────────────────────────────────────────────────────────────────
# Application
# ─────────────────────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    cfg = get_settings()

    app = FastAPI(
        title="Streaming Live RAG Engine",
        description="Samsung PRISM GenAI Hackathon 3.0 — Theme 4",
        version=cfg.app_version,
        lifespan=lifespan,
        docs_url="/api/docs",
        redoc_url="/api/redoc",
    )

    # CORS
    origins = [o.strip() for o in cfg.cors_origins.split(",")]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Health ────────────────────────────────────────────────────────────

    @app.get("/health", response_model=HealthResponse, tags=["System"])
    async def health() -> HealthResponse:
        index = get_index_manager()
        return HealthResponse(
            status="ok",
            rules_only_mode=cfg.rules_only_mode,
            corpus_loaded=index.is_loaded,
            index_chunks=len(index.chunks),
            version=cfg.app_version,
        )

    # ── WebSocket streaming ───────────────────────────────────────────────

    @app.websocket("/ws/stream/{session_id}")
    async def ws_stream(websocket: WebSocket, session_id: str):
        await websocket_handler(websocket, session_id)

    # ── Replay ────────────────────────────────────────────────────────────

    @app.post("/replay", response_model=ReplayResponse, tags=["Pipeline"])
    async def replay(request: ReplayRequest) -> ReplayResponse:
        try:
            return await run_replay(request)
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Replay error: {e}")

    # ── Session management ────────────────────────────────────────────────

    @app.get("/session/{session_id}/state", tags=["Session"])
    async def session_state(session_id: str):
        if not cfg.debug_admin:
            raise HTTPException(status_code=403, detail="Debug admin mode not enabled.")
        store = get_session_store()
        session = store.get(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found.")
        return session.to_state().model_dump()

    @app.post("/session/{session_id}/reset", tags=["Session"])
    async def session_reset(session_id: str):
        from app.controller import get_controller
        from app.retrieval.fusion import get_retriever

        store = get_session_store()
        store.reset(session_id)
        get_controller().reset_session(session_id)
        get_retriever().clear_session_cache(session_id)
        get_telemetry().clear_session(session_id)
        return {"status": "reset", "session_id": session_id}

    # ── Metrics ───────────────────────────────────────────────────────────

    @app.get("/metrics", response_model=MetricsResponse, tags=["Observability"])
    async def metrics():
        agg = get_telemetry().get_aggregates()
        return MetricsResponse(
            total_turns=agg["total_turns"],
            total_retrievals=agg["total_retrievals"],
            suppressed_retrievals=agg["suppressed_retrievals"],
            avg_ttft_ms=round(agg["avg_ttft_ms"], 2),
            avg_e2e_ms=round(agg["avg_e2e_ms"], 2),
            avg_cost_usd=round(agg["avg_cost_usd"], 6),
            cache_hit_rate=round(agg["cache_hit_rate"], 3),
            citation_violation_count=agg["citation_violation_count"],
        )

    # ── Demo console ──────────────────────────────────────────────────────

    frontend_path = Path("frontend/index.html")

    @app.get("/", response_class=HTMLResponse, tags=["UI"])
    async def demo_console():
        if frontend_path.exists():
            return HTMLResponse(content=frontend_path.read_text(encoding="utf-8"))
        return HTMLResponse(content=_minimal_console_html(), status_code=200)

    # ── Scenario list (for UI picker) ─────────────────────────────────────

    @app.get("/scenarios", tags=["Pipeline"])
    async def list_scenarios():
        scenario_dir = Path("bench/scenarios")
        if not scenario_dir.exists():
            return {"scenarios": []}
        scenarios = [
            f.stem for f in scenario_dir.iterdir()
            if f.suffix in (".jsonl", ".json")
        ]
        return {"scenarios": sorted(scenarios)}

    return app


def _minimal_console_html() -> str:
    """Fallback HTML if frontend/index.html is not built yet."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Streaming Live RAG — Demo Console</title>
  <style>
    body { font-family: system-ui; max-width: 900px; margin: 40px auto; padding: 20px; background: #0f0f1a; color: #e0e0e0; }
    h1 { color: #7c8dff; } code { background: #1a1a2e; padding: 2px 6px; border-radius: 4px; }
    .status { padding: 10px; border-radius: 8px; background: #1a1a2e; margin: 10px 0; }
  </style>
</head>
<body>
  <h1>🚀 Streaming Live RAG Engine</h1>
  <p>Samsung PRISM GenAI Hackathon 3.0 · Theme 4</p>
  <div class="status">
    <strong>API is running.</strong> Frontend demo console is building...
    <br>Use <code>POST /replay</code> with a scenario name to test the pipeline.
    <br>Open <code>/api/docs</code> for the interactive API documentation.
  </div>
  <script>
    fetch('/health').then(r=>r.json()).then(d=>{
      document.querySelector('.status').innerHTML +=
        '<br><br>Status: ' + JSON.stringify(d, null, 2).replace(/\n/g,'<br>').replace(/ /g,'&nbsp;');
    });
  </script>
</body>
</html>"""


app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False)
