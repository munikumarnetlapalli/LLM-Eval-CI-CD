"""FastAPI main application — thin routes, logic in services."""
from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path

from backend.app.core.config import get_settings
from backend.app.api import evaluations, experiments, comparisons, health

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    yield
    # Shutdown


app = FastAPI(
    title="LLM Eval CI/CD API",
    description="Automated LLM + RAG Evaluation, Regression Detection, and CI/CD Quality Gates",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow dashboard and local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start = time.monotonic()
    response = await call_next(request)
    process_time = (time.monotonic() - start) * 1000
    response.headers["X-Process-Time-Ms"] = f"{process_time:.2f}"
    return response


# ─── Routers ──────────────────────────────────────────────────────────────────

app.include_router(health.router,       prefix="/api/v1",               tags=["Health"])
app.include_router(evaluations.router,  prefix="/api/v1/evaluations",   tags=["Evaluations"])
app.include_router(experiments.router,  prefix="/api/v1/experiments",   tags=["Experiments"])
app.include_router(comparisons.router,  prefix="/api/v1",               tags=["Comparisons & Gates"])

# ─── Static files (dashboard) ─────────────────────────────────────────────────

frontend_dir = Path(__file__).parent.parent.parent / "frontend"
if (frontend_dir / "static").exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dir / "static")), name="static")

# ─── Global error handler ─────────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Never leak stack traces to clients."""
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "type": type(exc).__name__},
    )
