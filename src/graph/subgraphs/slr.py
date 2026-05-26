"""SLR (Systematic Literature Review) subgraph — agent + ToolNode loop.

# SLR Subgraph topology (Option B: LLM-driven tool selection):
#
#   START
#     │
#     ▼
#   slr_agent ─── [has tool_calls?] ──► slr_tools ──► slr_agent (loop, max _MAX_ROUNDS)
#     │
#     │ [no tool_calls or rounds exhausted]
#     ▼
#   slr_collect_papers  ◄── extracts papers from ToolMessages in state.messages;
#     │                      deduplicates; writes merged_papers, search_strategy,
#     │                      source_count to state
#     ▼
#   slr_synthesise      ◄── LLM over merged_papers
#     │
#     ▼
#   slr_finalise        ◄── assembles result dict from state
#     │
#     ▼
#    END
#
# The LLM in slr_agent is bound to [search_openalex, search_asta] and prompted
# to call both tools with the primary query and 1-2 alternative phrasings.
# ToolNode executes all tool calls concurrently (LangGraph fan-out within the node).
# Tool call spans appear as proper entries in LangSmith / Langfuse traces.
"""
from __future__ import annotations

import ast
import json
import logging
from typing import Literal

from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from src.config import DEFAULT_RESEARCH_MODEL, DEFAULT_MAX_TOKENS
from src.core.llm_utils import _build_llm, acall_llm
from src.graph.agents.tools.search_tools import search_asta, search_openalex
from src.graph.state import SLRAgentState

logger = logging.getLogger(__name__)

_MAX_ROUNDS = 3   # safety cap — normally exits after 1-2 rounds


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_papers_from_content(content: str) -> list[dict]:
    """Parse a ToolMessage content string into a list of paper dicts."""
    if not content:
        return []
    try:
        result = json.loads(content)
        return result if isinstance(result, list) else []
    except (json.JSONDecodeError, ValueError):
        pass
    try:
        result = ast.literal_eval(content)
        return result if isinstance(result, list) else []
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Node: slr_agent — LLM with bound tools
# ---------------------------------------------------------------------------


async def slr_agent(state: SLRAgentState, config: RunnableConfig) -> dict:
    """LLM agent node: calls search_openalex and search_asta via tool use.

    On the first call the system + user messages are injected. Subsequent
    calls continue the conversation with the tool results already in state.
    Max _MAX_ROUNDS calls total; returns {} to force exit if exceeded.
    """
    rounds = state.get("agent_rounds") or 0
    if rounds >= _MAX_ROUNDS:
        return {}

    existing_messages: list[AnyMessage] = list(state.get("messages") or [])
    first_call = not existing_messages

    if first_call:
        from src.prompts.research_prompts import SLR_AGENT_SYSTEM, SLR_AGENT_USER
        existing_messages = [
            SystemMessage(content=SLR_AGENT_SYSTEM),
            HumanMessage(content=SLR_AGENT_USER.format(
                query=state["query"],
                context=state.get("context") or "",
            )),
        ]

    model_name = (config.get("configurable") or {}).get("research_model", DEFAULT_RESEARCH_MODEL)
    try:
        llm = _build_llm(model_name, max_tokens=1024).bind_tools(
            [search_openalex, search_asta]
        )
        response = await llm.ainvoke(existing_messages, config=config)
    except Exception as exc:
        logger.error("slr_agent LLM call failed: %s", exc)
        return {"errors": [f"slr_agent: {exc}"]}

    new_messages: list[AnyMessage] = (
        existing_messages + [response] if first_call else [response]
    )
    return {"messages": new_messages, "agent_rounds": rounds + 1}


# ---------------------------------------------------------------------------
# Routing: _route_slr — decides tool loop vs. collect
# ---------------------------------------------------------------------------


def _route_slr(state: SLRAgentState) -> Literal["slr_tools", "slr_collect_papers"]:
    """Route to slr_tools if the last message has pending tool calls, else collect."""
    messages: list[AnyMessage] = state.get("messages") or []
    if not messages:
        return "slr_collect_papers"
    last = messages[-1]
    rounds = state.get("agent_rounds") or 0
    if rounds >= _MAX_ROUNDS:
        return "slr_collect_papers"
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "slr_tools"
    return "slr_collect_papers"


# ---------------------------------------------------------------------------
# ToolNode: slr_tools — executes tool calls from the last AIMessage
# ---------------------------------------------------------------------------

slr_tools = ToolNode([search_openalex, search_asta])


# ---------------------------------------------------------------------------
# Node: slr_collect_papers — extract and deduplicate papers from ToolMessages
# ---------------------------------------------------------------------------


async def slr_collect_papers(state: SLRAgentState, config: RunnableConfig) -> dict:
    """Extract papers from ToolMessages in state.messages and deduplicate.

    ToolNode stores tool results in ToolMessage.content. Each message has a
    .name attribute matching the tool ("search_openalex" / "search_asta").
    Deduplicates by lowercased title; writes merged_papers, search_strategy,
    and source_count to state.
    """
    openalex_papers: list[dict] = []
    asta_papers: list[dict] = []

    for msg in (state.get("messages") or []):
        if not isinstance(msg, ToolMessage):
            continue
        papers = _parse_papers_from_content(msg.content)
        tool_name = getattr(msg, "name", "") or ""
        if tool_name == "search_openalex":
            openalex_papers.extend(papers)
        elif tool_name == "search_asta":
            asta_papers.extend(papers)

    seen: set[str] = set()
    merged: list[dict] = []
    for paper in openalex_papers + asta_papers:
        key = (paper.get("title") or "").lower().strip()
        if key and key not in seen:
            seen.add(key)
            merged.append(paper)

    has_oa = bool(openalex_papers)
    has_asta = bool(asta_papers)
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

    model = (config.get("configurable") or {}).get("research_model", DEFAULT_RESEARCH_MODEL)
    try:
        synthesis = await acall_llm(
            SLR_SYNTHESIS_TEMPLATE.format(query=state["query"], papers=papers_text),
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

    builder.add_node("slr_agent", slr_agent)
    builder.add_node("slr_tools", slr_tools)
    builder.add_node("slr_collect_papers", slr_collect_papers)
    builder.add_node("slr_synthesise", slr_synthesise)
    builder.add_node("slr_finalise", slr_finalise)

    builder.add_edge(START, "slr_agent")
    builder.add_conditional_edges(
        "slr_agent",
        _route_slr,
        {
            "slr_tools": "slr_tools",
            "slr_collect_papers": "slr_collect_papers",
        },
    )
    builder.add_edge("slr_tools", "slr_agent")
    builder.add_edge("slr_collect_papers", "slr_synthesise")
    builder.add_edge("slr_synthesise", "slr_finalise")
    builder.add_edge("slr_finalise", END)

    return builder.compile()


slr_graph = build_slr_graph()
