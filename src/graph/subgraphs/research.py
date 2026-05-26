"""Research dispatch subgraph — 6 nodes (Stage 4).

fan_out_research_tasks  ← pure router, one Send() per research_plan item
slr_worker              ← handles task_type="slr"
lbd_worker              ← handles task_type="lbd"
deep_web_worker         ← handles task_type="deep_web"
edison_worker           ← handles task_type="edison", rewrites query inline
aggregate_research_results ← no-op reducer; research_results already accumulated by workers

Note: research_plan items carry key "type" (ARCHITECTURE.md §2); fan_out maps it
to task_type in the Send() payload.
"""
from __future__ import annotations

import logging
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from src.graph.subgraphs.deep_web import deep_web_graph
from src.graph.subgraphs.edison import edison_graph
from src.graph.subgraphs.lbd import lbd_graph
from src.graph.subgraphs.slr import slr_graph
from src.graph.state import ResearchDispatchState, ResearchTaskState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# fan_out_research_tasks — pure router
# ---------------------------------------------------------------------------

_TASK_TYPE_TO_NODE: dict[str, str] = {
    "slr": "slr_worker",
    "lbd": "lbd_worker",
    "deep_web": "deep_web_worker",
    "edison": "edison_worker",
}


async def fan_out_research_tasks(state: ResearchDispatchState) -> list[Send] | str:
    already_done = {r.get("task_id") for r in state.get("research_results", [])}
    sends: list[Send] = []

    for task in state.get("research_plan", []):
        # ARCHITECTURE.md §2: research_plan items use key "type"
        task_type = task.get("task_type") or task.get("type", "")
        node_name = _TASK_TYPE_TO_NODE.get(task_type, "deep_web_worker")
        task_id = task.get("id", "")
        if task_id in already_done:
            continue
        sends.append(Send(node_name, {
            "task_id": task_id,
            "task_type": task_type,
            "query": task.get("query", ""),
            "linked_scope": task.get("linked_scope", ""),
            "priority": task.get("priority", ""),
            "result": None,
        }))

    return sends or "aggregate_research_results"


# ---------------------------------------------------------------------------
# slr_worker
# ---------------------------------------------------------------------------


async def slr_worker(state: ResearchTaskState, config: RunnableConfig | None = None) -> dict:
    import time
    from datetime import datetime, timezone

    task_id = state["task_id"]
    linked_scope = state.get("linked_scope", "")

    start = time.monotonic()
    started_at = datetime.now(timezone.utc).isoformat()
    try:
        state_out = await slr_graph.ainvoke(
            {"task_id": task_id, "query": state["query"], "context": "", "top_k": 20},
            config=config,
        )
        inner = state_out.get("result") or {}
        result_dict: dict = {
            "task_id": task_id,
            "task_type": "slr",
            "channel": "slr",
            "linked_scope": linked_scope,
            "query": state["query"],
            "thesis": inner.get("thesis") or "",
            "results": inner.get("papers") or [],
            "success": inner.get("success", True),
            "error_message": inner.get("error_message"),
            "status": "ok",
        }
        success = True
        error_message = None
    except Exception as exc:
        logger.error("slr_worker failed %s: %s", task_id, exc)
        result_dict = {"task_id": task_id, "task_type": "slr", "error": str(exc), "status": "error"}
        success = False
        error_message = str(exc)

    duration_ms = int((time.monotonic() - start) * 1000)
    # Extract metadata from result for trace
    result_items = result_dict.get("results") or result_dict.get("documents") or []
    source_urls = [r.get("url", r.get("source", "")) for r in result_items[:5]] if isinstance(result_items, list) else []
    trace = {
        "tool_name": "slr_worker",
        "called_at": started_at,
        "duration_ms": duration_ms,
        "success": success,
        "error_message": error_message,
        "query": state["query"],
        "task_id": task_id,
        "linked_scope": linked_scope,
        "result_count": len(result_items) if isinstance(result_items, list) else (0 if not success else 1),
        "top_source_urls": source_urls,
        "agent_model": "",
    }

    return {"research_results": [result_dict], "slr_traces": [trace]}


# ---------------------------------------------------------------------------
# lbd_worker
# ---------------------------------------------------------------------------


async def lbd_worker(state: ResearchTaskState, config: RunnableConfig | None = None) -> dict:
    import time
    from datetime import datetime, timezone

    task_id = state["task_id"]
    linked_scope = state.get("linked_scope", "")

    start = time.monotonic()
    started_at = datetime.now(timezone.utc).isoformat()
    try:
        state_out = await lbd_graph.ainvoke(
            {"task_id": task_id, "query": state["query"], "context": ""},
            config=config,
        )
        inner = state_out.get("result") or {}
        result_dict: dict = {
            "task_id": task_id,
            "task_type": "lbd",
            "channel": "lbd",
            "linked_scope": linked_scope,
            "query": state["query"],
            "thesis": inner.get("thesis") or "",
            "concepts": inner.get("concepts") or [],
            "results": inner.get("papers") or [],
            "success": inner.get("success", True),
            "error_message": inner.get("error_message"),
            "status": "ok",
        }
        success = True
        error_message = None
    except Exception as exc:
        logger.error("lbd_worker failed %s: %s", task_id, exc)
        result_dict = {"task_id": task_id, "task_type": "lbd", "error": str(exc), "status": "error"}
        success = False
        error_message = str(exc)

    duration_ms = int((time.monotonic() - start) * 1000)
    result_items = result_dict.get("results") or result_dict.get("documents") or []
    trace = {
        "tool_name": "lbd_worker",
        "called_at": started_at,
        "duration_ms": duration_ms,
        "success": success,
        "error_message": error_message,
        "query": state["query"],
        "task_id": task_id,
        "linked_scope": linked_scope,
        "result_count": len(result_items) if isinstance(result_items, list) else (0 if not success else 1),
        "concepts_discovered": (result_dict.get("concepts") or [])[:10],
    }

    return {"research_results": [result_dict], "lbd_traces": [trace]}


# ---------------------------------------------------------------------------
# deep_web_worker
# ---------------------------------------------------------------------------


async def deep_web_worker(state: ResearchTaskState, config: RunnableConfig | None = None) -> dict:
    import re
    import time
    from datetime import datetime, timezone

    task_id = state["task_id"]
    linked_scope = state.get("linked_scope", "")

    start = time.monotonic()
    started_at = datetime.now(timezone.utc).isoformat()
    try:
        state_out = await deep_web_graph.ainvoke(
            {"task_id": task_id, "question": state["query"], "context": ""},
            config=config,
        )
        inner = state_out.get("result") or {}
        _answer = inner.get("result") or inner.get("answer") or inner.get("content") or ""
        result_dict: dict = {
            "task_id": task_id,
            "task_type": "deep_web",
            "channel": "deep_web",
            "linked_scope": linked_scope,
            "query": state["query"],
            "thesis": _answer,
            "result": _answer,
            "content": _answer,
            "sources": inner.get("sources") or [],
            "model_used": inner.get("model_used") or "",
            "search_rounds": inner.get("search_rounds") or 0,
            "success": inner.get("success", True),
            "error_message": inner.get("error_message"),
            "status": "ok",
        }
        success = True
        error_message = None
    except Exception as exc:
        logger.error("deep_web_worker failed %s: %s", task_id, exc)
        result_dict = {"task_id": task_id, "task_type": "deep_web", "error": str(exc), "status": "error"}
        success = False
        error_message = str(exc)

    duration_ms = int((time.monotonic() - start) * 1000)
    result_text = str(result_dict.get("result") or result_dict.get("content") or "")
    sources_cited = re.findall(r'https?://[^\s\]\)]+', result_text)[:5]
    trace = {
        "tool_name": "deep_web_worker",
        "called_at": started_at,
        "duration_ms": duration_ms,
        "success": success,
        "error_message": error_message,
        "query": state["query"],
        "task_id": task_id,
        "model_used": "",
        "result_summary_chars": len(result_text),
        "sources_cited": sources_cited,
    }

    return {"research_results": [result_dict], "deep_web_traces": [trace]}


# ---------------------------------------------------------------------------
# edison_worker  (runs edison_query_rewriter inline before dispatch)
# ---------------------------------------------------------------------------


async def edison_worker(state: ResearchTaskState, config: RunnableConfig | None = None) -> dict:
    import time
    from datetime import datetime, timezone

    task_id = state["task_id"]
    linked_scope = state.get("linked_scope", "")

    original_query = state["query"]
    start = time.monotonic()
    started_at = datetime.now(timezone.utc).isoformat()
    try:
        state_out = await edison_graph.ainvoke(
            {"task_id": task_id, "original_query": original_query, "context": "", "skip_rewrite": False},
            config=config,
        )
        inner = state_out.get("result") or {}
        rewritten_query = state_out.get("rewritten_query") or original_query
        _papers = inner.get("papers") or []
        _thesis_parts = []
        for p in _papers[:10]:
            title = p.get("title", "")
            abstract = (p.get("abstract") or "")[:300].strip()
            if title:
                _thesis_parts.append(f"{title}. {abstract}" if abstract else title)
        _thesis = " | ".join(_thesis_parts)
        result_dict: dict = {
            "task_id": task_id,
            "task_type": "edison",
            "channel": "edison",
            "linked_scope": linked_scope,
            "query": rewritten_query,
            "original_query": original_query,
            "rewritten_query": rewritten_query,
            "thesis": _thesis,
            "papers": _papers,
            "success": inner.get("success", True),
            "error_message": inner.get("error_message"),
            "status": inner.get("status") or "ok",
        }
        success = True
        error_message = None
    except Exception as exc:
        logger.error("edison_worker failed %s: %s", task_id, exc)
        rewritten_query = original_query
        result_dict = {
            "task_id": task_id,
            "task_type": "edison",
            "rewritten_query": rewritten_query,
            "error": str(exc),
            "status": "error",
        }
        success = False
        error_message = str(exc)

    duration_ms = int((time.monotonic() - start) * 1000)
    result_items = result_dict.get("papers") or result_dict.get("results") or []
    top_paper_ids = [str(r.get("paperId", r.get("id", ""))) for r in result_items[:5]] if isinstance(result_items, list) else []
    trace = {
        "tool_name": "edison_worker",
        "called_at": started_at,
        "duration_ms": duration_ms,
        "success": success,
        "error_message": error_message,
        "original_query": original_query,
        "rewritten_query": rewritten_query,
        "task_id": task_id,
        "result_count": len(result_items) if isinstance(result_items, list) else (0 if not success else 1),
        "top_paper_ids": top_paper_ids,
    }

    return {"research_results": [result_dict], "edison_traces": [trace]}


# ---------------------------------------------------------------------------
# aggregate_research_results — reducer
# ---------------------------------------------------------------------------


async def aggregate_research_results(state: ResearchDispatchState) -> dict:
    research_results = state.get("research_results") or []
    dispatch_results = [r for r in research_results if r.get("task_type") != "edison"]
    edison_results = [r for r in research_results if r.get("task_type") == "edison"]
    return {
        "dispatch_results": dispatch_results,
        "edison_results": edison_results,
    }


# ---------------------------------------------------------------------------
# Graph compilation
# ---------------------------------------------------------------------------

_builder = StateGraph(ResearchDispatchState)

_builder.add_node("slr_worker", slr_worker)
_builder.add_node("lbd_worker", lbd_worker)
_builder.add_node("deep_web_worker", deep_web_worker)
_builder.add_node("edison_worker", edison_worker)
_builder.add_node("aggregate_research_results", aggregate_research_results)

# fan_out_research_tasks is a conditional edge router (returns list[Send] or fallback str),
# not a node — matches causal.py dispatch pattern
_builder.add_conditional_edges(
    START,
    fan_out_research_tasks,
    {
        "slr_worker": "slr_worker",
        "lbd_worker": "lbd_worker",
        "deep_web_worker": "deep_web_worker",
        "edison_worker": "edison_worker",
        "aggregate_research_results": "aggregate_research_results",
    },
)
_builder.add_edge("slr_worker", "aggregate_research_results")
_builder.add_edge("lbd_worker", "aggregate_research_results")
_builder.add_edge("deep_web_worker", "aggregate_research_results")
_builder.add_edge("edison_worker", "aggregate_research_results")
_builder.add_edge("aggregate_research_results", END)

research_graph = _builder.compile()
