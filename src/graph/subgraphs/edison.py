"""Edison literature platform subgraph — 3 nodes.

# Step 1 — Audit findings vs src/graph/agents/edison_graph.py
# ──────────────────────────────────────────────────────────────────────────────
# | Pattern found                                     | File:line             | Fix applied                                             |
# |---------------------------------------------------|-----------------------|---------------------------------------------------------|
# | Missing config=config on acall_llm                | edison_graph.py:47    | config threaded in edison_rewrite_query                 |
# | Silent except in edison_rewrite_query: falls back | edison_graph.py:60-62 | error surfaced to "errors" field (fallback kept)        |
# |   to original query, error not surfaced           |                       |                                                         |
# | "__start__" string literal instead of START const | edison_graph.py:154   | Use START from langgraph.graph                          |
# | Implementation in graph/agents/, not subgraphs/   | edison_graph.py       | moved to src/graph/subgraphs/edison.py                  |
# ──────────────────────────────────────────────────────────────────────────────
#
# Note — what is unique to Edison (cannot be templated from SLR):
#   No fan-out. Edison is a single-tool agent: optional LLM query rewrite →
#   one Edison platform search → finalise. No parallelism is appropriate here
#   because the search is a single call whose query depends on the rewrite step.
#   The conditional entry (skip_rewrite) is the only branching point.

# Edison Subgraph topology:
#
#   START
#     │
#     │ [conditional via route_rewrite_entry (synchronous)]
#     │   skip_rewrite=True  → "edison_search"
#     │   skip_rewrite=False → "edison_rewrite_query"
#     ▼
#   edison_rewrite_query (optional)  ← LLM rewrites query for Edison platform
#     │
#     ▼
#   edison_search                    ← calls Edison literature search tool
#     │
#     ▼
#   edison_finalise                  ← assembles result dict
#     │
#     ▼
#    END
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from src.config import DEFAULT_FAST_MODEL, EDISON_TIMEOUT_SECONDS
from src.core.llm_utils import acall_llm
from src.graph.agents.tools.edison_tools import search_edison
from src.graph.state import EdisonAgentState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Routing: route_rewrite_entry — synchronous conditional edge (not a node)
# ---------------------------------------------------------------------------


def route_rewrite_entry(state: EdisonAgentState) -> str:
    """Skip rewriting if flagged, otherwise rewrite query first."""
    if state.get("skip_rewrite"):
        return "edison_search"
    return "edison_rewrite_query"


# ---------------------------------------------------------------------------
# Node: edison_rewrite_query — LLM query rewriter
# ---------------------------------------------------------------------------


async def edison_rewrite_query(state: EdisonAgentState, config: RunnableConfig) -> dict:
    """LLM: rewrite query for Edison literature platform."""
    if state.get("rewritten_query") is not None:
        return {}

    from src.prompts.research_prompts import EDISON_REWRITE_SYSTEM, EDISON_REWRITE_TEMPLATE

    model = (config.get("configurable") or {}).get("research_model", DEFAULT_FAST_MODEL)
    original = state["original_query"]
    try:
        raw = await acall_llm(
            EDISON_REWRITE_TEMPLATE.format(
                query=original,
                context=state.get("context") or "",
            ),
            EDISON_REWRITE_SYSTEM,
            model=model,
            config=config,
        )
        rewritten = raw.strip() if isinstance(raw, str) else original
        if rewritten.lower().startswith("rewrite") or len(rewritten) < 5:
            rewritten = original
    except Exception as exc:
        logger.warning("edison_rewrite_query failed: %s — using original", exc)
        return {
            "rewritten_query": original,
            "errors": [f"edison_rewrite_query: {exc}"],
        }

    return {"rewritten_query": rewritten}


# ---------------------------------------------------------------------------
# Node: edison_search — calls Edison literature search tool
# ---------------------------------------------------------------------------


async def edison_search(state: EdisonAgentState, config: RunnableConfig) -> dict:
    """Call Edison search with rewritten (or original) query."""
    if state.get("papers") is not None:
        return {}

    start = time.monotonic()
    called_at = datetime.now(timezone.utc).isoformat()

    query = state.get("rewritten_query") or state["original_query"]
    try:
        papers = await search_edison.ainvoke({
            "query": query,
            "top_k": 10,
            "timeout": EDISON_TIMEOUT_SECONDS,
        })
        duration_ms = int((time.monotonic() - start) * 1000)
        trace = {
            "tool_name": "search_edison",
            "called_at": called_at,
            "duration_ms": duration_ms,
            "success": True,
            "result_count": len(papers) if papers else 0,
            "query_used": query,
        }
        return {
            "papers": papers or [],
            "result_count": len(papers) if papers else 0,
            "tool_traces": [trace],
        }
    except Exception as exc:
        duration_ms = int((time.monotonic() - start) * 1000)
        return {
            "papers": [],
            "result_count": 0,
            "tool_traces": [{
                "tool_name": "search_edison",
                "called_at": called_at,
                "duration_ms": duration_ms,
                "success": False,
                "error_message": str(exc),
            }],
            "errors": [f"edison_search: {exc}"],
        }


# ---------------------------------------------------------------------------
# Node: edison_finalise — assembles EdisonQueryResult
# ---------------------------------------------------------------------------


async def edison_finalise(state: EdisonAgentState, config: RunnableConfig) -> dict:
    """Assemble final result dict."""
    errors = state.get("errors") or []
    papers = state.get("papers") or []

    result = {
        "task_id": state["task_id"],
        "original_query": state["original_query"],
        "rewritten_query": state.get("rewritten_query"),
        "papers": papers,
        "result_count": state.get("result_count") or len(papers),
        "status": "ok" if papers else ("no_evidence" if not errors else "error"),
        "success": not errors,
        "error_message": "; ".join(errors) if errors else None,
    }
    return {"result": result, "success": result["success"], "error_message": result["error_message"]}


# ---------------------------------------------------------------------------
# Graph compilation
# ---------------------------------------------------------------------------


def build_edison_graph():
    builder = StateGraph(EdisonAgentState)
    builder.add_node("edison_rewrite_query", edison_rewrite_query)
    builder.add_node("edison_search", edison_search)
    builder.add_node("edison_finalise", edison_finalise)

    builder.add_conditional_edges(
        START,
        route_rewrite_entry,
        {
            "edison_rewrite_query": "edison_rewrite_query",
            "edison_search": "edison_search",
        },
    )
    builder.add_edge("edison_rewrite_query", "edison_search")
    builder.add_edge("edison_search", "edison_finalise")
    builder.add_edge("edison_finalise", END)

    return builder.compile()


edison_graph = build_edison_graph()
