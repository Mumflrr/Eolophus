"""
api/server.py — FastAPI entrypoint for the Eolophus pipeline.

Exposes all pipeline functionality over HTTP so the web UI, macOS SwiftUI
app, and the chess Bluetooth bridge can all talk to the same localhost
server.

This module only wires things together: app creation, middleware,
lifespan, static file mounting, and router registration. All actual
endpoint logic lives in api/routers/*, and shared helpers live in the
other api/*.py modules (paths, state, schemas, json_utils, attachments,
status, searxng, db_helpers, routing_config) — see each module's
docstring for what it owns and why.

Run:
    uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload

Dependencies (add to requirements.txt):
    fastapi uvicorn[standard] sse-starlette psutil
"""

from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from api.paths import STATIC_DIR
from api.routers import chat, chess, clarify, config, health, lessons, models, pipelines, runs, stream
from api.searxng import lifespan

log = logging.getLogger(__name__)

app = FastAPI(title="Eolophus Pipeline API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

for r in (runs, chat, clarify, pipelines, models, config, lessons, chess, stream, health):
    app.include_router(r.router)

# ── Static / UI ───────────────────────────────────────────────────────────────

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def serve_ui():
    index = STATIC_DIR / "index.html"
    if not index.exists():
        return {"error": "UI not built. Place index.html in api/static/"}
    return FileResponse(str(index))


# ── Startup ───────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def on_startup():
    from storage.db import initialise
    initialise()
    log.info("Eolophus API server started. UI: http://localhost:8000")
