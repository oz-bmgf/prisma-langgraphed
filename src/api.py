"""FastAPI application exposing the NQPR pipeline as an async HTTP API.

Endpoints:
  POST   /runs                       — start a run (202, background task)
  GET    /runs/{thread_id}           — run status snapshot
  POST   /runs/{thread_id}/resume    — send Command(resume=value) to interrupted run
  GET    /runs/{thread_id}/report    — analyst report (text/markdown)
  GET    /runs/{thread_id}/stream    — SSE stream of node-completion events
  DELETE /runs/{thread_id}           — cancel a run (204)

Shared state (initialised in lifespan):
  app.state.graph          — compiled LangGraph workflow
  app.state.tasks          — dict[thread_id, asyncio.Task]
  app.state.queues         — dict[thread_id, asyncio.Queue[str]]  SSE events
  app.state.resume_queues  — dict[thread_id, asyncio.Queue[Any]]  interrupt values
"""
from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse, Response
from langgraph.types import Command
from pydantic import BaseModel
from sse_starlette import ServerSentEvent
from sse_starlette.sse import EventSourceResponse

from observability.tracing import init_tracing, shutdown as tracing_shutdown
from src.backends.factory import build_search_backend
from src.config import DEFAULT_RESEARCH_MODEL, DEFAULT_SYNTHESIS_MODEL
from src.core.checkpointer import build_checkpointer
from src.graph.workflow import compile_graph, create_initial_state, make_thread_id

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class RunRequest(BaseModel):
    program: str
    run_name: str
    collection_name: str
    base_dir: str
    ingested_dir: str
    research_model: str = DEFAULT_RESEARCH_MODEL
    synthesis_model: str = DEFAULT_SYNTHESIS_MODEL
    focus: Optional[str] = None
    focus_bows: Optional[list[str]] = None
    aux_collections: Optional[list[str]] = None


class RunCreated(BaseModel):
    thread_id: str
    status: str = "started"


class RunStatus(BaseModel):
    thread_id: str
    status: str  # "running" | "interrupted" | "done" | "error"
    next_nodes: list[str]
    values_summary: dict[str, Any]
    tool_trace_summary: dict = {}


class ResumeRequest(BaseModel):
    value: Any


# ---------------------------------------------------------------------------
# Lifespan — graph initialisation
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialise OTEL before any LLM calls — registers all three exporters.
    init_tracing("nqpr-api")
    async with build_checkpointer() as checkpointer:
        app.state.graph = compile_graph(checkpointer)
        # asyncio-APPROVED-4: Task/Queue restricted to api.py SSE infrastructure
        app.state.tasks: dict[str, asyncio.Task] = {}
        # asyncio-APPROVED-4: Task/Queue restricted to api.py SSE infrastructure
        app.state.queues: dict[str, asyncio.Queue] = {}
        # asyncio-APPROVED-4: Task/Queue restricted to api.py SSE infrastructure
        app.state.resume_queues: dict[str, asyncio.Queue] = {}
        yield
    # Flush all pending spans to every backend before the process exits.
    tracing_shutdown()


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="NQPR Pipeline API", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(thread_id: str, ingested_dir: str, collection_name: str) -> dict:
    try:
        backend = build_search_backend(ingested_dir, collection_name)
    except Exception:
        logger.warning("Could not build search backend — search will be unavailable")
        backend = None
    config: dict = {"configurable": {"thread_id": thread_id}}
    if backend is not None:
        config["configurable"]["search_backend"] = backend
    return config


async def _run_background(
    app_state: Any,
    thread_id: str,
    initial_state: dict,
    config: dict,
) -> None:
    # asyncio-APPROVED-4: Task/Queue restricted to api.py SSE infrastructure
    queue: asyncio.Queue = app_state.queues[thread_id]
    # asyncio-APPROVED-4: Task/Queue restricted to api.py SSE infrastructure
    resume_queue: asyncio.Queue = app_state.resume_queues[thread_id]
    graph = app_state.graph

    payload: Any = initial_state
    try:
        while True:
            async for event in graph.astream(payload, config, stream_mode="updates"):
                for node_name in event:
                    await queue.put(json.dumps({"event": "node_complete", "node": node_name}))

            state = await graph.aget_state(config)
            if not state.next:
                await queue.put(json.dumps({"event": "done", "data": "pipeline complete"}))
                break

            interrupted_at = state.next[0]
            await queue.put(json.dumps({"event": "interrupt", "node": interrupted_at}))
            value = await resume_queue.get()
            payload = Command(resume=value)

    # asyncio-APPROVED-4: re-raise after SSE shutdown signal
    except asyncio.CancelledError:
        try:
            queue.put_nowait(json.dumps({"event": "cancelled", "data": "run cancelled"}))
        except Exception:
            pass
        raise   # ← MUST re-raise CancelledError
    except Exception as exc:
        logger.exception("Background run failed: thread_id=%s", thread_id)
        await queue.put(json.dumps({"event": "error", "data": str(exc)}))
    finally:
        app_state.tasks.pop(thread_id, None)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.post("/runs", response_model=RunCreated, status_code=202)
async def start_run(req: RunRequest) -> RunCreated:
    thread_id = make_thread_id(req.program, req.run_name)
    if thread_id in app.state.tasks:
        raise HTTPException(status_code=409, detail=f"Run {thread_id!r} already in progress")

    initial_state = create_initial_state(
        program=req.program,
        run_name=req.run_name,
        collection_name=req.collection_name,
        base_dir=req.base_dir,
        ingested_dir=req.ingested_dir,
        research_model=req.research_model,
        synthesis_model=req.synthesis_model,
        focus=req.focus,
        focus_bows=req.focus_bows,
        aux_collections=req.aux_collections,
    )
    config = _make_config(thread_id, req.ingested_dir, req.collection_name)

    # asyncio-APPROVED-4: Task/Queue restricted to api.py SSE infrastructure
    app.state.queues[thread_id] = asyncio.Queue()
    # asyncio-APPROVED-4: Task/Queue restricted to api.py SSE infrastructure
    app.state.resume_queues[thread_id] = asyncio.Queue()
    # asyncio-APPROVED-4: Task/Queue restricted to api.py SSE infrastructure
    task = asyncio.create_task(
        _run_background(app.state, thread_id, initial_state, config)
    )
    app.state.tasks[thread_id] = task

    return RunCreated(thread_id=thread_id)


@app.get("/runs/{thread_id}", response_model=RunStatus)
async def get_run_status(thread_id: str) -> RunStatus:
    config = {"configurable": {"thread_id": thread_id}}
    state = await app.state.graph.aget_state(config)
    if not state.values:
        raise HTTPException(status_code=404, detail=f"Run {thread_id!r} not found")

    if thread_id in app.state.tasks:
        status = "interrupted" if state.next else "running"
    else:
        status = "done"

    summary = {
        k: v
        for k, v in state.values.items()
        if not isinstance(v, (list, dict)) or len(str(v)) < 500
    }
    from src.core.tool_tracing import summarise_traces
    trace_summary = summarise_traces(state.values)
    return RunStatus(
        thread_id=thread_id,
        status=status,
        next_nodes=list(state.next),
        values_summary=summary,
        tool_trace_summary=trace_summary,
    )


@app.post("/runs/{thread_id}/resume", status_code=200)
async def resume_run(thread_id: str, req: ResumeRequest) -> dict:
    if thread_id not in app.state.resume_queues:
        raise HTTPException(
            status_code=404, detail=f"Run {thread_id!r} not found or not interrupted"
        )
    await app.state.resume_queues[thread_id].put(req.value)
    return {"status": "resumed", "thread_id": thread_id}


@app.get("/runs/{thread_id}/report")
async def get_report(thread_id: str) -> PlainTextResponse:
    config = {"configurable": {"thread_id": thread_id}}
    state = await app.state.graph.aget_state(config)
    if not state.values:
        raise HTTPException(status_code=404, detail=f"Run {thread_id!r} not found")
    report = state.values.get("analyst_report")
    if not report:
        raise HTTPException(status_code=404, detail="Report not yet available")
    return PlainTextResponse(content=report, media_type="text/markdown")


@app.get("/runs/{thread_id}/stream")
async def stream_run(thread_id: str) -> EventSourceResponse:
    if thread_id not in app.state.queues:
        raise HTTPException(status_code=404, detail=f"Run {thread_id!r} not found")

    # asyncio-APPROVED-4: Task/Queue restricted to api.py SSE infrastructure
    queue: asyncio.Queue = app.state.queues[thread_id]

    async def event_generator() -> AsyncGenerator[ServerSentEvent, None]:
        while True:
            try:
                # asyncio-APPROVED-4: wait_for on SSE queue.get() for heartbeat timeout
                msg = await asyncio.wait_for(queue.get(), timeout=30.0)
                yield ServerSentEvent(data=msg)
                parsed = json.loads(msg)
                if parsed.get("event") in {"done", "error", "cancelled"}:
                    break
            # asyncio-APPROVED-4: wait_for on SSE queue.get() for heartbeat timeout
            except asyncio.TimeoutError:
                yield ServerSentEvent(data=json.dumps({"event": "heartbeat"}))

    return EventSourceResponse(event_generator())


@app.get("/runs/{thread_id}/traces")
async def get_run_traces(thread_id: str) -> dict:
    config = {"configurable": {"thread_id": thread_id}}
    state = await app.state.graph.aget_state(config)
    if not state.values:
        raise HTTPException(status_code=404, detail=f"Run {thread_id!r} not found")
    v = state.values
    trace_fields = [
        "asta_traces", "slr_traces", "lbd_traces", "deep_web_traces",
        "edison_traces", "web_search_traces", "compute_traces",
        "collection_search_traces", "investigation_traces",
    ]
    return {
        "thread_id": thread_id,
        **{field: v.get(field, []) for field in trace_fields},
    }


@app.delete("/runs/{thread_id}", status_code=204)
async def cancel_run(thread_id: str) -> Response:
    task = app.state.tasks.pop(thread_id, None)
    if task is not None:
        task.cancel()
    app.state.queues.pop(thread_id, None)
    app.state.resume_queues.pop(thread_id, None)
    return Response(status_code=204)
