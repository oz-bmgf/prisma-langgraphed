"""Deep Web research subgraph — 5 nodes.

# Step 1 — Audit findings vs src/graph/agents/deep_web_graph.py
# ──────────────────────────────────────────────────────────────────────────────
# | Pattern found                                     | File:line             | Fix applied                                             |
# |---------------------------------------------------|-----------------------|---------------------------------------------------------|
# | Missing config=config on acall_llm                | deep_web_graph.py:147 | config threaded in deep_web_search_round                |
# | Missing config=config on acall_llm                | deep_web_graph.py:214 | config threaded in deep_web_synthesise_fallback         |
# | Silent except in deep_web_synthesise_fallback:    | deep_web_graph.py:    | error surfaced to "errors" field; last answer kept as   |
# |   falls back to answers[-1], not surfaced         | 221-222               | fallback to preserve best-effort output                 |
# | asyncio.wait_for for primary O3 call              | deep_web_graph.py:42  | Keep: asyncio-APPROVED-3 (single external timeout —     |
# |                                                   |                       | no fan-out alternative for a one-shot primary call)     |
# | Missing status field in finalise result           | deep_web_graph.py:    | status: "ok"/"error" added for finalize.py gate         |
# |                                                   | 247-258               |                                                         |
# | Implementation in graph/agents/, not subgraphs/   | deep_web_graph.py     | moved to src/graph/subgraphs/deep_web.py                |
# ──────────────────────────────────────────────────────────────────────────────

# Deep Web Subgraph topology:
#
#   START
#     │
#     ▼
#   deep_web_try_primary          ← O3 deep research with asyncio timeout
#     │
#     │ [conditional via deep_web_route_after_primary]
#     │   primary.success=True  → "deep_web_finalise"                ──────────────────────┐
#     │   primary.success=False → Send("deep_web_search_round", ...) × DEEP_WEB_MAX_ROUNDS │
#     ▼                                                                                     │
#   deep_web_search_round (×N, parallel)                                                   │
#     │                                                                                     │
#     ▼                                                                                     │
#   deep_web_collect_rounds                                                                 │
#     │                                                                                     │
#     ▼                                                                                     │
#   deep_web_synthesise_fallback                                                            │
#     │                                                                                     │
#     └──────────────────────────────────────────────────────────────────────────────────┐ │
#                                                                                        ▼ ▼
#                                                                              deep_web_finalise
#                                                                                        │
#                                                                                       END
#
# deep_web_route_after_primary is a conditional edge routing function (not a node):
# returns "deep_web_finalise" on success, or list[Send] for fallback rounds on failure.
"""
from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from src.config import (
    DEFAULT_RESEARCH_MODEL,
    DEEP_WEB_FALLBACK_MODEL,
    DEEP_WEB_MAX_ROUNDS,
    DEEP_WEB_TIMEOUT_SECONDS,
)
from src.core.llm_utils import acall_llm
from src.graph.state import DeepWebAgentState, DeepWebSearchRoundState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Node: deep_web_try_primary — calls O3 deep_research with timeout
# ---------------------------------------------------------------------------


async def deep_web_try_primary(state: DeepWebAgentState, config: RunnableConfig) -> dict:
    """Try primary O3 deep research. Sets primary_result."""
    if state.get("primary_result") is not None:
        return {}

    from src.core.agents.deep_web import _primary_research
    from src.config import DEEP_WEB_PRIMARY_MODEL

    start = time.monotonic()
    called_at = datetime.now(timezone.utc).isoformat()
    try:
        # asyncio-APPROVED-3: wait_for wraps single external primary model call with timeout.
        # No fan-out alternative: this is one blocking call to an external service.
        result = await asyncio.wait_for(
            _primary_research(state["question"], state.get("context") or "", model=DEEP_WEB_PRIMARY_MODEL),
            timeout=DEEP_WEB_TIMEOUT_SECONDS,
        )
        duration_ms = int((time.monotonic() - start) * 1000)
        primary_result = {
            "answer": result.answer,
            "sources": result.sources,
            "model_used": result.model_used,
            "success": result.success,
            "error_message": result.error_message,
        }
        trace = {
            "tool_name": "deep_web_primary",
            "called_at": called_at,
            "duration_ms": duration_ms,
            "success": result.success,
            "model_used": result.model_used,
        }
        return {
            "primary_result": primary_result,
            "model_used": result.model_used,
            "tool_traces": [trace],
        }
    except (Exception, asyncio.TimeoutError) as exc:
        duration_ms = int((time.monotonic() - start) * 1000)
        trace = {
            "tool_name": "deep_web_primary",
            "called_at": called_at,
            "duration_ms": duration_ms,
            "success": False,
            "error_message": str(exc) or type(exc).__name__,
        }
        return {
            "primary_result": {"success": False, "error_message": str(exc) or type(exc).__name__, "answer": "", "sources": []},
            "tool_traces": [trace],
            "errors": [f"deep_web_primary: {type(exc).__name__}: {exc}"],
        }


# ---------------------------------------------------------------------------
# Conditional edge: deep_web_route_after_primary
# ---------------------------------------------------------------------------


async def deep_web_route_after_primary(state: DeepWebAgentState) -> list[Send] | str:
    """Route: primary success → finalise. Primary failure → dispatch fallback rounds."""
    if (state.get("primary_result") or {}).get("success"):
        return "deep_web_finalise"
    sends = []
    for round_i in range(DEEP_WEB_MAX_ROUNDS):
        sends.append(Send("deep_web_search_round", DeepWebSearchRoundState(
            round_number=round_i + 1,
            question=state["question"],
            prior_context=state.get("context") or "",
            result=None,
        )))
    return sends or "deep_web_finalise"


def deep_web_dispatch_rounds(state: DeepWebAgentState) -> list[Send] | str:
    """Fan out: one Send per fallback round (exposed for testing)."""
    sends = []
    for round_i in range(DEEP_WEB_MAX_ROUNDS):
        sends.append(Send("deep_web_search_round", DeepWebSearchRoundState(
            round_number=round_i + 1,
            question=state["question"],
            prior_context=state.get("context") or "",
            result=None,
        )))
    return sends or "deep_web_finalise"


# ---------------------------------------------------------------------------
# Node: deep_web_search_round — worker, one fallback round
# ---------------------------------------------------------------------------


async def deep_web_search_round(state: DeepWebSearchRoundState, config: RunnableConfig) -> dict:
    """Worker: run one fallback web search round. LLM synthesises over context."""
    if state.get("result") is not None:
        return {"search_round_results": [state["result"]]}

    from src.prompts.research_prompts import DEEP_WEB_FALLBACK_ROUND_TEMPLATE, DEEP_WEB_FALLBACK_SYSTEM

    start = time.monotonic()
    called_at = datetime.now(timezone.utc).isoformat()
    round_num = state["round_number"]

    try:
        model = (config.get("configurable") or {}).get("research_model", DEEP_WEB_FALLBACK_MODEL)
        prompt = DEEP_WEB_FALLBACK_ROUND_TEMPLATE.format(
            question=state["question"],
            context=state.get("prior_context") or "",
            round=round_num,
            rounds=DEEP_WEB_MAX_ROUNDS,
            prior="",
        )
        answer = await acall_llm(prompt, DEEP_WEB_FALLBACK_SYSTEM, model=model, config=config)
        duration_ms = int((time.monotonic() - start) * 1000)
        round_result = {
            "round_number": round_num,
            "answer": answer if isinstance(answer, str) else "",
            "sources": [],
            "success": True,
        }
        trace = {
            "tool_name": "deep_web_round",
            "called_at": called_at,
            "duration_ms": duration_ms,
            "success": True,
            "round_number": round_num,
        }
        return {"search_round_results": [round_result], "tool_traces": [trace]}
    except Exception as exc:
        duration_ms = int((time.monotonic() - start) * 1000)
        return {
            "search_round_results": [{"round_number": round_num, "answer": "", "success": False, "error": str(exc)}],
            "tool_traces": [{
                "tool_name": "deep_web_round",
                "called_at": called_at,
                "duration_ms": duration_ms,
                "success": False,
                "round_number": round_num,
                "error_message": str(exc),
            }],
            "errors": [f"deep_web_round_{round_num}: {exc}"],
        }


# ---------------------------------------------------------------------------
# Node: deep_web_collect_rounds — trivial join after fan-out
# ---------------------------------------------------------------------------


async def deep_web_collect_rounds(state: DeepWebAgentState, config: RunnableConfig) -> dict:
    """Join node after fallback round fan-out. No-op: reducers handle accumulation."""
    return {}


# ---------------------------------------------------------------------------
# Node: deep_web_synthesise_fallback — LLM synthesises all round results
# ---------------------------------------------------------------------------


async def deep_web_synthesise_fallback(state: DeepWebAgentState, config: RunnableConfig) -> dict:
    """LLM: synthesise all fallback round results into a coherent answer."""
    if state.get("fallback_synthesis") is not None:
        return {}

    from src.prompts.research_prompts import DEEP_WEB_FALLBACK_SYSTEM

    rounds = state.get("search_round_results") or []
    if not rounds:
        return {"fallback_synthesis": "No search rounds completed."}

    answers = [r.get("answer", "") for r in rounds if r.get("success") and r.get("answer")]
    if not answers:
        return {"fallback_synthesis": "All search rounds failed."}

    combined = "\n\n---\n\n".join(answers)
    model = (config.get("configurable") or {}).get("research_model", DEEP_WEB_FALLBACK_MODEL)
    try:
        synthesis = await acall_llm(
            f"Synthesise these {len(answers)} research rounds into a final comprehensive answer.\n\n"
            f"Question: {state['question']}\n\nRound answers:\n{combined}",
            DEEP_WEB_FALLBACK_SYSTEM,
            model=model,
            max_tokens=2048,
            config=config,
        )
    except Exception as exc:
        logger.error("deep_web_synthesise_fallback LLM failed: %s", exc)
        # Best-effort: use last available answer rather than returning nothing
        return {
            "fallback_synthesis": answers[-1],
            "errors": [f"deep_web_synthesise_fallback: {exc}"],
        }

    return {"fallback_synthesis": synthesis if isinstance(synthesis, str) else str(synthesis)}


# ---------------------------------------------------------------------------
# Node: deep_web_finalise — assembles DeepWebResult
# ---------------------------------------------------------------------------


async def deep_web_finalise(state: DeepWebAgentState, config: RunnableConfig) -> dict:
    """Assemble final result dict. Prefers primary result if it succeeded."""
    errors = state.get("errors") or []

    primary = state.get("primary_result") or {}
    if primary.get("success"):
        answer = primary.get("answer", "")
        sources = primary.get("sources") or []
        model_used = primary.get("model_used", "")
    else:
        answer = state.get("fallback_synthesis") or ""
        sources = []
        model_used = DEEP_WEB_FALLBACK_MODEL

    result = {
        "task_id": state["task_id"],
        "question": state["question"],
        "result": answer,
        "content": answer,
        "answer": answer,
        "sources": sources,
        "model_used": model_used,
        "search_rounds": len(state.get("search_round_results") or []),
        "status": "ok" if (bool(answer) and not (errors and not primary.get("success"))) else "error",
        "success": bool(answer) and not (errors and not primary.get("success")),
        "error_message": "; ".join(errors) if errors and not answer else None,
    }
    return {"result": result, "success": result["success"], "error_message": result["error_message"]}


# ---------------------------------------------------------------------------
# Graph compilation
# ---------------------------------------------------------------------------


def build_deep_web_graph():
    builder = StateGraph(DeepWebAgentState)
    builder.add_node("deep_web_try_primary", deep_web_try_primary)
    builder.add_node("deep_web_search_round", deep_web_search_round)
    builder.add_node("deep_web_collect_rounds", deep_web_collect_rounds)
    builder.add_node("deep_web_synthesise_fallback", deep_web_synthesise_fallback)
    builder.add_node("deep_web_finalise", deep_web_finalise)

    builder.add_edge(START, "deep_web_try_primary")
    builder.add_conditional_edges(
        "deep_web_try_primary",
        deep_web_route_after_primary,
        {
            "deep_web_search_round": "deep_web_search_round",
            "deep_web_finalise": "deep_web_finalise",
        },
    )
    builder.add_edge("deep_web_search_round", "deep_web_collect_rounds")
    builder.add_edge("deep_web_collect_rounds", "deep_web_synthesise_fallback")
    builder.add_edge("deep_web_synthesise_fallback", "deep_web_finalise")
    builder.add_edge("deep_web_finalise", END)

    return builder.compile()


deep_web_graph = build_deep_web_graph()
