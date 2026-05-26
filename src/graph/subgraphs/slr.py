"""SLR (Systematic Literature Review) subgraph — 6 nodes.

# Step 1 — Audit findings vs src/graph/agents/slr_graph.py
# ──────────────────────────────────────────────────────────────────────────────
# | Pattern found                                   | File:line          | Fix applied                                          |
# |-------------------------------------------------|--------------------|------------------------------------------------------|
# | Missing config=config on acall_llm              | slr_graph.py:74-81 | config threaded through slr_expand_queries           |
# | Missing config=config on acall_llm              | slr_graph.py:213   | config threaded through slr_synthesise               |
# | except Exception: queries=[] — silent discard   | slr_graph.py:88-89 | errors surfaced into state "errors" field            |
# | Redundant dedup in both collect and synthesise  | slr_graph.py:155,  | slr_collect_papers writes merged_papers; synthesise  |
# |                                                 | 191-197            | reads it directly (no re-dedup)                      |
# | slr_finalise re-concatenates raw source lists   | slr_graph.py:233   | reads merged_papers from state (already deduped)     |
# | slr_plan_sources always emits 2 Sends — no      | slr_graph.py:34-49 | checks openalex_results / asta_results in state;     |
# | idempotency on LangGraph checkpoint resume      |                    | skips already-fetched sources                        |
# | Implementation in graph/agents/ not subgraphs/  | slr_graph.py       | moved to src/graph/subgraphs/slr.py                  |
# ──────────────────────────────────────────────────────────────────────────────

# SLR Subgraph topology:
#
#   START
#     │
#     ▼
#   slr_start ───────────────────────────────────────────────────────┐
#     │ [unconditional edge]                [conditional edge via     │
#     │                                      slr_plan_sources]        │
#     ▼                                                              ▼
#   slr_expand_queries                        slr_fetch_source (×0-2: openalex, asta)
#     │                                                              │
#     └──────────────────────────┬───────────────────────────────────┘
#                                │  all active branches join here
#                                ▼
#                       slr_collect_papers  ←── dedup; writes merged_papers, source_count,
#                                │               search_strategy to state
#                                ▼
#                       slr_synthesise      ←── LLM over merged_papers; uses expanded_queries
#                                │
#                                ▼
#                       slr_finalise        ←── assembles result dict from state
#                                │
#                                ▼
#                               END
#
# Idempotency: slr_plan_sources checks openalex_results / asta_results in state
# and skips already-fetched sources (enables clean resume from LangGraph checkpoints).
# slr_expand_queries and slr_synthesise each check their output field and return {}
# if already populated.
"""
from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime, timezone
from typing import Optional

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from src.config import DEFAULT_RESEARCH_MODEL, OPENALEX_MAX_RESULTS
from src.core.llm_utils import acall_llm
from src.graph.agents.tools.search_tools import search_asta, search_openalex
from src.graph.state import SLRAgentState, SLRFetchState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Node: slr_start — trivial entry enabling parallel fan-out from one hub
# ---------------------------------------------------------------------------


async def slr_start(state: SLRAgentState, config: RunnableConfig) -> dict:
    """Trivial entry node — hub that fans out to slr_expand_queries and
    slr_plan_sources simultaneously from a single source node."""
    return {}


# ---------------------------------------------------------------------------
# Conditional edge: slr_plan_sources — idempotent router, returns list[Send]
# ---------------------------------------------------------------------------


async def slr_plan_sources(state: SLRAgentState) -> list[Send] | str:
    """Fan-out router: emit one Send per source that hasn't been fetched yet.

    Idempotency: checks openalex_results and asta_results in state. Non-empty
    means the source was already fetched on a previous run — skip it.
    Falls through to slr_collect_papers when all sources are already done.
    """
    sends: list[Send] = []

    if not state.get("openalex_results"):
        sends.append(Send("slr_fetch_source", SLRFetchState(
            source="openalex",
            query=state["query"],
            top_k=state.get("top_k") or OPENALEX_MAX_RESULTS,
            result=None,
        )))

    if not state.get("asta_results"):
        sends.append(Send("slr_fetch_source", SLRFetchState(
            source="asta",
            query=state["query"],
            top_k=state.get("top_k") or OPENALEX_MAX_RESULTS,
            result=None,
        )))

    return sends or "slr_collect_papers"


# ---------------------------------------------------------------------------
# Node: slr_expand_queries — LLM query expansion, runs parallel with fetch
# ---------------------------------------------------------------------------


async def slr_expand_queries(state: SLRAgentState, config: RunnableConfig) -> dict:
    """LLM generates 2-3 alternative query phrasings for better recall.
    Runs in parallel with the slr_fetch_source fan-out.
    Results stored in expanded_queries and picked up by slr_synthesise.
    """
    if state.get("expanded_queries") is not None:
        return {}

    from src.prompts.research_prompts import (
        SLR_QUERY_EXPANSION_SYSTEM,
        SLR_QUERY_EXPANSION_USER,
    )

    model = (config.get("configurable") or {}).get("research_model", DEFAULT_RESEARCH_MODEL)
    try:
        raw = await acall_llm(
            SLR_QUERY_EXPANSION_USER.format(
                query=state["query"],
                context=state.get("context") or "",
            ),
            SLR_QUERY_EXPANSION_SYSTEM,
            model=model,
            config=config,
        )
        m = re.search(r'\{.*\}', raw, re.DOTALL)
        queries: list[str] = json.loads(m.group()).get("queries", []) if m else []
    except Exception as exc:
        logger.warning("slr_expand_queries failed: %s", exc)
        queries = []
        return {
            "expanded_queries": queries,
            "errors": [f"slr_expand_queries: {exc}"],
        }

    return {"expanded_queries": queries[:3]}


# ---------------------------------------------------------------------------
# Node: slr_fetch_source — worker, HTTP only, no LLM
# ---------------------------------------------------------------------------


async def slr_fetch_source(state: SLRFetchState, config: RunnableConfig) -> dict:
    """Worker: fetch papers from one source. HTTP only, no LLM.
    Returns to the openalex_results or asta_results reducer field.
    """
    if state.get("result") is not None:
        field = "openalex_results" if state["source"] == "openalex" else "asta_results"
        return {field: state["result"]}

    start = time.monotonic()
    called_at = datetime.now(timezone.utc).isoformat()
    source = state["source"]

    try:
        if source == "openalex":
            papers = await search_openalex.ainvoke({"query": state["query"], "top_k": state["top_k"]})
            field = "openalex_results"
        else:
            papers = await search_asta.ainvoke({"query": state["query"], "top_k": state["top_k"]})
            field = "asta_results"

        duration_ms = int((time.monotonic() - start) * 1000)
        trace = {
            "tool_name": f"search_{source}",
            "called_at": called_at,
            "duration_ms": duration_ms,
            "success": True,
            "result_count": len(papers) if papers else 0,
        }
        return {field: papers or [], "tool_traces": [trace]}

    except Exception as exc:
        duration_ms = int((time.monotonic() - start) * 1000)
        return {
            "errors": [f"slr_fetch_{source}: {exc}"],
            "tool_traces": [{
                "tool_name": f"search_{source}",
                "called_at": called_at,
                "duration_ms": duration_ms,
                "success": False,
                "error_message": str(exc),
            }],
        }


# ---------------------------------------------------------------------------
# Node: slr_collect_papers — deduplicate once; write merged_papers to state
# ---------------------------------------------------------------------------


async def slr_collect_papers(state: SLRAgentState, config: RunnableConfig) -> dict:
    """Merge and deduplicate results from both sources into merged_papers.

    This is the single deduplication point. slr_synthesise and slr_finalise
    both read merged_papers directly — no second deduplication pass.
    """
    oa = state.get("openalex_results") or []
    asta_r = state.get("asta_results") or []

    seen: set[str] = set()
    merged: list[dict] = []
    for paper in oa + asta_r:
        key = (paper.get("title") or "").lower().strip()
        if key and key not in seen:
            seen.add(key)
            merged.append(paper)

    has_oa = bool(oa)
    has_asta = bool(asta_r)
    strategy = (
        "combined" if has_oa and has_asta
        else "openalex" if has_oa
        else "asta" if has_asta
        else "none"
    )

    return {
        "merged_papers": merged,
        "search_strategy": strategy,
        "source_count": len(merged),
    }


# ---------------------------------------------------------------------------
# Node: slr_synthesise — LLM synthesis over merged_papers
# ---------------------------------------------------------------------------


async def slr_synthesise(state: SLRAgentState, config: RunnableConfig) -> dict:
    """LLM: synthesise evidence from merged_papers.
    Uses expanded_queries as additional context when available.
    Reads merged_papers written by slr_collect_papers — no re-deduplication.
    """
    if state.get("synthesis") is not None:
        return {}

    from src.prompts.research_prompts import SLR_SYNTHESIS_SYSTEM, SLR_SYNTHESIS_TEMPLATE

    papers = state.get("merged_papers") or []
    if not papers:
        return {"synthesis": "No papers found for this query.", "success": True}

    papers_text = "\n\n".join(
        f"[{i}] {p.get('title', '')} ({p.get('authors', '')}, {p.get('year', '?')})\n"
        f"{(p.get('abstract') or '')[:300]}"
        for i, p in enumerate(papers[:20], 1)
    )

    expanded = state.get("expanded_queries") or []
    query_context = state["query"]
    if expanded:
        query_context = f"{state['query']}\n\nRelated phrasings: {'; '.join(expanded)}"

    model = (config.get("configurable") or {}).get("research_model", DEFAULT_RESEARCH_MODEL)
    try:
        synthesis = await acall_llm(
            SLR_SYNTHESIS_TEMPLATE.format(query=query_context, papers=papers_text),
            SLR_SYNTHESIS_SYSTEM,
            model=model,
            max_tokens=2048,
            config=config,
        )
    except Exception as exc:
        logger.error("slr_synthesise LLM failed: %s", exc)
        return {
            "synthesis": None,
            "errors": [f"slr_synthesise: {exc}"],
        }

    return {"synthesis": synthesis if isinstance(synthesis, str) else str(synthesis)}


# ---------------------------------------------------------------------------
# Node: slr_finalise — assemble result dict from state
# ---------------------------------------------------------------------------


async def slr_finalise(state: SLRAgentState, config: RunnableConfig) -> dict:
    """Assemble the final result dict. Reads merged_papers — no re-concatenation."""
    errors = state.get("errors") or []
    papers = state.get("merged_papers") or []
    result = {
        "task_id": state["task_id"],
        "query": state["query"],
        "thesis": state.get("synthesis") or "",
        "papers": papers,
        "source_count": state.get("source_count") or 0,
        "search_strategy": state.get("search_strategy") or "none",
        "success": not errors,
        "error_message": "; ".join(errors) if errors else None,
    }
    return {"result": result, "success": result["success"], "error_message": result["error_message"]}


# ---------------------------------------------------------------------------
# Graph compilation
# ---------------------------------------------------------------------------


def build_slr_graph() -> StateGraph:
    builder = StateGraph(SLRAgentState)

    builder.add_node("slr_start", slr_start)
    builder.add_node("slr_expand_queries", slr_expand_queries)
    # slr_plan_sources is a conditional edge function, not a node
    builder.add_node("slr_fetch_source", slr_fetch_source)
    builder.add_node("slr_collect_papers", slr_collect_papers)
    builder.add_node("slr_synthesise", slr_synthesise)
    builder.add_node("slr_finalise", slr_finalise)

    # ── Parallel branches from slr_start ─────────────────────────────────────
    # Branch 1: LLM query expansion (unconditional edge)
    # Branch 2: source fetch fan-out (conditional edge via slr_plan_sources × 0-2)
    builder.add_edge(START, "slr_start")
    builder.add_edge("slr_start", "slr_expand_queries")
    builder.add_conditional_edges(
        "slr_start",
        slr_plan_sources,
        {
            "slr_fetch_source": "slr_fetch_source",
            "slr_collect_papers": "slr_collect_papers",
        },
    )

    # ── Join at slr_collect_papers ────────────────────────────────────────────
    builder.add_edge("slr_expand_queries", "slr_collect_papers")
    builder.add_edge("slr_fetch_source", "slr_collect_papers")

    builder.add_edge("slr_collect_papers", "slr_synthesise")
    builder.add_edge("slr_synthesise", "slr_finalise")
    builder.add_edge("slr_finalise", END)

    return builder.compile()


slr_graph = build_slr_graph()
