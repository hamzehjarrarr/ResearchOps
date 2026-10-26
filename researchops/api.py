"""FastAPI backend: starts research runs in the background and serves results."""

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from researchops import store
from researchops.config import MODEL, OLLAMA_BASE_URL
from researchops.graph import run_research

pool = ThreadPoolExecutor(max_workers=2)


class ResearchRequest(BaseModel):
    question: str = Field(min_length=12, max_length=2000)


@asynccontextmanager
async def lifespan(app: FastAPI):
    with store.connection():
        pass
    yield
    pool.shutdown(wait=False, cancel_futures=True)


app = FastAPI(title="ResearchOps AI", version="1.1.0", lifespan=lifespan)


def process(run_id: str, question: str):
    try:
        result = run_research(
            question,
            on_progress=lambda events: store.progress(run_id, events),
        )
        store.finish(run_id, result)
    except Exception as exc:
        # The stored message stays generic so it never contains secrets;
        # the full error goes to the server log.
        store.fail(
            run_id,
            f"{type(exc).__name__}: research failed. "
            "Check the API terminal for details.",
        )
        logging.exception("Research run %s failed", run_id)


def ollama_status():
    """Check that Ollama is reachable and the configured model is pulled."""
    try:
        response = httpx.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        response.raise_for_status()
        names = {item.get("name", "") for item in response.json().get("models", [])}
        wanted = MODEL if ":" in MODEL else f"{MODEL}:latest"
        return True, wanted in names
    except Exception:
        return False, False


@app.get("/health")
def health():
    reachable, model_present = ollama_status()
    return {
        "status": "ok",
        "ollama_reachable": reachable,
        "ollama_model": MODEL,
        "ollama_model_pulled": model_present,
        "tavily_configured": bool(os.getenv("TAVILY_API_KEY")),
    }


@app.post("/research", status_code=202)
def research(request: ResearchRequest):
    run_id = store.start(request.question)
    pool.submit(process, run_id, request.question)
    return {"id": run_id, "status": "running"}


@app.get("/runs")
def runs():
    return store.recent()


@app.get("/runs/{run_id}")
def get_run(run_id: str):
    item = store.get(run_id)
    if not item:
        raise HTTPException(404, "Run not found")
    return item