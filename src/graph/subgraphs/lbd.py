"""LBD (Literature-Based Discovery) subgraph — agent + ToolNode loop.

# LBD Subgraph topology (Option B: LLM-driven tool selection):
#
#   START
#     │
#     ▼
#   lbd_agent ─── [has tool_calls?] ──► lbd_tools ──► lbd_agent (loop, max _MAX_ROUNDS)
#     │
#     │ [no tool_calls or rounds exhausted]
#     ▼
#   lbd_collect_papers  ◄── extracts all search_asta ToolMessages from state.messages;
#     │                      deduplicates; writes merged_papers and seed_concepts to state
#     ▼
#   lbd_discover_connections ◄── LLM B-term extraction from merged_papers (Swanson ABC)
#     │
#     ▼
#   lbd_synthesise      ◄── LLM narrative over merged_papers + seed_concepts + B-terms
#     │
#     ▼
#   lbd_finalise        ◄── assembles result dict from state
#     │
#     ▼
#    END
#
# The LLM in lbd_agent is bound to [search_asta] and prompted to identify 3-5
# key concepts from the query, then call search_asta for each concept and once
# more for the full query. ToolNode executes all calls concurrently.
# Tool call spans appear as proper entries in LangSmith / Langfuse traces.
"""
from __future__ import annotations

import ast
import json
import logging
import re
from typing import Literal

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from src.config import DEFAULT_RESEARCH_MODEL
from src.core.llm_utils import _build_llm, acall_llm
from src.core.research_utils import deduplicate_papers
from src.graph.agents.tools.search_tools import search_asta
from src.graph.state import LBDAgentState

logger = logging.getLogger(__name__)

_MAX_ROUNDS = 3            # safety cap — normally exits after 1-2 rounds
_MAX_CONCEPTS_FOR_FETCH = 5


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


def _parse_concepts(text: str) -> list[str]:
    """Parse comma/newline separated concepts from LLM response."""
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [str(c).strip() for c in parsed if str(c).strip()]
    except (json.JSONDecodeError, TypeError):
        pass

    match = re.search(r'\[([^\]]+)\]', text)
    if match:
        parts = [p.strip().strip('"\'') for p in match.group(1).split(',')]
        return [p for p in parts if p]

    parts = re.split(r'[,\n]+', text)
    return [p.strip().strip('"\'- ') for p in parts if p.strip()][:10]


# ---------------------------------------------------------------------------
# Node: lbd_agent — LLM with bound search_asta tool
# ---------------------------------------------------------------------------


async def lbd_agent(state: LBDAgentState, config: RunnableConfig) -> dict:
    """LLM agent node: extracts concepts and calls search_asta for each.

    On the first call the system + user messages are injected. Subsequent
    calls continue the conversation with tool results already in state.
    Max _MAX_ROUNDS calls total; returns {} to force exit if exceeded.
    """
    rounds = state.get("agent_rounds") or 0
    if rounds >= _MAX_ROUNDS:
        return {}

    existing_messages: list[AnyMessage] = list(state.get("messages") or [])
    first_call = not existing_messages

    if first_call:
        from src.prompts.research_prompts import LBD_AGENT_SYSTEM, LBD_AGENT_USER
        existing_messages = [
            SystemMessage(content=LBD_AGENT_SYSTEM),
            HumanMessage(content=LBD_AGENT_USER.format(
                query=state["query"],
                context=state.get("context") or "",
            )),
        ]

    model_name = (config.get("configurable") or {}).get("research_model", DEFAULT_RESEARCH_MODEL)
    try:
        llm = _build_llm(model_name, max_tokens=1024).bind_tools([search_asta])
        response = await llm.ainvoke(existing_messages, config=config)
    except Exception as exc:
        logger.error("lbd_agent LLM call failed: %s", exc)
        return {"errors": [f"lbd_agent: {exc}"]}

    new_messages: list[AnyMessage] = (
        existing_messages + [response] if first_call else [response]
    )
    return {"messages": new_messages, "agent_rounds": rounds + 1}


# ---------------------------------------------------------------------------
# Routing: _route_lbd — decides tool loop vs. collect
# ---------------------------------------------------------------------------


def _route_lbd(state: LBDAgentState) -> Literal["lbd_tools", "lbd_collect_papers"]:
    """Route to lbd_tools if the last message has pending tool calls, else collect."""
    messages: list[AnyMessage] = state.get("messages") or []
    if not messages:
        return "lbd_collect_papers"
    last = messages[-1]
    rounds = state.get("agent_rounds") or 0
    if rounds >= _MAX_ROUNDS:
        return "lbd_collect_papers"
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "lbd_tools"
    return "lbd_collect_papers"


# ---------------------------------------------------------------------------
# ToolNode: lbd_tools — executes search_asta tool calls from the last AIMessage
# ---------------------------------------------------------------------------

lbd_tools = ToolNode([search_asta])


# ---------------------------------------------------------------------------
# Node: lbd_collect_papers — extract papers + concepts from messages
# ---------------------------------------------------------------------------


async def lbd_collect_papers(state: LBDAgentState, config: RunnableConfig) -> dict:
    """Extract papers from search_asta ToolMessages and concepts from tool call args.

    seed_concepts are derived from the query arguments of each search_asta call
    that differ from the main query (i.e. concept-level searches, not the broad
    search on the full query).
    """
    all_papers: list[dict] = []

    for msg in (state.get("messages") or []):
        if not isinstance(msg, ToolMessage):
            continue
        if getattr(msg, "name", "") != "search_asta":
            continue
        papers = _parse_papers_from_content(msg.content)
        all_papers.extend(papers)

    # Extract concept queries from tool_calls in AIMessages
    concepts: list[str] = []
    main_query_lower = state["query"].lower().strip()
    for msg in (state.get("messages") or []):
        if not isinstance(msg, AIMessage):
            continue
        for tc in getattr(msg, "tool_calls", None) or []:
            if tc.get("name") != "search_asta":
                continue
            q = (tc.get("args") or {}).get("query", "")
            if q and q.lower().strip() != main_query_lower:
                concepts.append(q)

    merged = deduplicate_papers(all_papers)

    return {
        "merged_papers": merged,
        "seed_concepts": concepts[:_MAX_CONCEPTS_FOR_FETCH] if concepts else [state["query"]],
        "paper_count": len(merged),
    }


# ---------------------------------------------------------------------------
# Node: lbd_discover_connections — LLM B-term extraction from merged_papers
# ---------------------------------------------------------------------------


async def lbd_discover_connections(state: LBDAgentState, config: RunnableConfig) -> dict:
    """LLM: extract B-term bridges from merged_papers for Swanson ABC discovery.

    Reads merged_papers written by lbd_collect_papers.
    Does NOT re-fetch or re-deduplicate — single dedup already done upstream.
    """
    if state.get("discovered_concepts") is not None:
        return {}

    from src.prompts.research_prompts import LBD_CONCEPT_SYSTEM, LBD_CONCEPT_TEMPLATE

    merged = state.get("merged_papers") or []
    if not merged:
        return {"discovered_concepts": []}

    papers_text = "\n\n".join(
        f"[{i}] {p.get('title', '')} ({p.get('year', '?')}): {(p.get('abstract') or '')[:250]}"
        for i, p in enumerate(merged[:20], 1)
    )

    model = (config.get("configurable") or {}).get("research_model", DEFAULT_RESEARCH_MODEL)
    b_term_prompt = LBD_CONCEPT_TEMPLATE.format(
        query=f"intermediary concepts bridging: {state['query']}\n\nEvidence:\n{papers_text}"
    )
    try:
        b_response = await acall_llm(b_term_prompt, LBD_CONCEPT_SYSTEM, model=model, config=config)
        b_terms = _parse_concepts(b_response)
    except Exception as exc:
        logger.warning("lbd_discover_connections b-term extraction failed: %s", exc)
        return {
            "discovered_concepts": [],
            "errors": [f"lbd_discover_connections: {exc}"],
        }

    discovered = [{"term": t, "type": "bridge"} for t in b_terms[:10]]
    return {"discovered_concepts": discovered}


# ---------------------------------------------------------------------------
# Node: lbd_synthesise — LLM narrative summary over merged_papers
# ---------------------------------------------------------------------------


async def lbd_synthesise(state: LBDAgentState, config: RunnableConfig) -> dict:
    """LLM: write narrative summary of discovered connections.
    Reads merged_papers written by lbd_collect_papers — no re-deduplication.
    """
    if state.get("narrative") is not None:
        return {}

    from src.prompts.research_prompts import LBD_SYNTHESIS_SYSTEM, LBD_SYNTHESIS_TEMPLATE

    papers = state.get("merged_papers") or []
    papers_text = "\n\n".join(
        f"[{i}] {p.get('title', '')} ({p.get('year', '?')}): {(p.get('abstract') or '')[:250]}"
        for i, p in enumerate(papers[:25], 1)
    )

    seed_concepts = state.get("seed_concepts") or []
    discovered = state.get("discovered_concepts") or []
    b_terms = [d.get("term", "") for d in discovered if d.get("type") == "bridge"]

    model = (config.get("configurable") or {}).get("research_model", DEFAULT_RESEARCH_MODEL)
    try:
        narrative = await acall_llm(
            LBD_SYNTHESIS_TEMPLATE.format(
                query=state["query"],
                a_terms=", ".join(seed_concepts),
                b_terms=", ".join(b_terms) if b_terms else "(none identified)",
                papers=papers_text,
            ),
            LBD_SYNTHESIS_SYSTEM,
            model=model,
            max_tokens=2048,
            config=config,
        )
    except Exception as exc:
        logger.error("lbd_synthesise LLM failed: %s", exc)
        return {
            "narrative": None,
            "errors": [f"lbd_synthesise: {exc}"],
        }

    return {"narrative": narrative if isinstance(narrative, str) else str(narrative)}


# ---------------------------------------------------------------------------
# Node: lbd_finalise — assembles LBDResult dict
# ---------------------------------------------------------------------------


async def lbd_finalise(state: LBDAgentState, config: RunnableConfig) -> dict:
    """Assemble final result dict. Reads merged_papers — no re-concatenation."""
    errors = state.get("errors") or []
    papers = state.get("merged_papers") or []
    seed_concepts = state.get("seed_concepts") or []
    discovered = state.get("discovered_concepts") or []
    all_concepts = seed_concepts + [d.get("term", "") for d in discovered if d.get("term")]

    result = {
        "task_id": state["task_id"],
        "query": state["query"],
        "thesis": state.get("narrative") or "",
        "concepts": list(dict.fromkeys(all_concepts))[:20],
        "papers": papers,
        "paper_count": len(papers),
        "status": "ok" if not errors else "error",
        "success": not errors,
        "error_message": "; ".join(errors) if errors else None,
    }
    return {"result": result, "success": result["success"], "error_message": result["error_message"]}


# ---------------------------------------------------------------------------
# Graph compilation
# ---------------------------------------------------------------------------


def build_lbd_graph():
    builder = StateGraph(LBDAgentState)

    builder.add_node("lbd_agent", lbd_agent)
    builder.add_node("lbd_tools", lbd_tools)
    builder.add_node("lbd_collect_papers", lbd_collect_papers)
    builder.add_node("lbd_discover_connections", lbd_discover_connections)
    builder.add_node("lbd_synthesise", lbd_synthesise)
    builder.add_node("lbd_finalise", lbd_finalise)

    builder.add_edge(START, "lbd_agent")
    builder.add_conditional_edges(
        "lbd_agent",
        _route_lbd,
        {
            "lbd_tools": "lbd_tools",
            "lbd_collect_papers": "lbd_collect_papers",
        },
    )
    builder.add_edge("lbd_tools", "lbd_agent")
    builder.add_edge("lbd_collect_papers", "lbd_discover_connections")
    builder.add_edge("lbd_discover_connections", "lbd_synthesise")
    builder.add_edge("lbd_synthesise", "lbd_finalise")
    builder.add_edge("lbd_finalise", END)

    return builder.compile()


lbd_graph = build_lbd_graph()
