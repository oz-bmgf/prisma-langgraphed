"""LBD (Literature-Based Discovery) subgraph — 7 nodes.

# Step 1 — Audit findings vs src/graph/agents/lbd_graph.py
# ──────────────────────────────────────────────────────────────────────────────
# | Pattern found                                     | File:line             | Fix applied                                             |
# |---------------------------------------------------|-----------------------|---------------------------------------------------------|
# | Missing config=config on acall_llm                | lbd_graph.py:72       | config threaded in lbd_extract_concepts                 |
# | Missing config=config on acall_llm                | lbd_graph.py:230      | config threaded in lbd_discover_connections             |
# | Missing config=config on acall_llm                | lbd_graph.py:276      | config threaded in lbd_synthesise                       |
# | Silent except in lbd_extract_concepts             | lbd_graph.py:78-81    | error surfaced to state "errors" field (fallback kept)  |
# | Silent except in lbd_discover_connections         | lbd_graph.py:233-234  | error surfaced to state "errors" field                  |
# | lbd_synthesise except: sets narrative to error    | lbd_graph.py:287-289  | narrative=None; error surfaced to "errors" field        |
# | Redundant dedup in lbd_discover_connections       | lbd_graph.py:211-217  | single dedup at join node; writes merged_papers         |
# | Redundant dedup in lbd_synthesise                 | lbd_graph.py:257-263  | reads merged_papers directly (no re-dedup)              |
# | lbd_finalise reads raw concept_papers accumulator | lbd_graph.py:302      | reads merged_papers instead                             |
# | Missing status field in finalise result           | lbd_graph.py:306-315  | status: "ok"/"error" added for finalize.py gate         |
# | Implementation in graph/agents/, not subgraphs/   | lbd_graph.py          | moved to src/graph/subgraphs/lbd.py                     |
# ──────────────────────────────────────────────────────────────────────────────

# LBD Subgraph topology:
#
#   START
#     │
#     ▼
#   lbd_start ──[branch 1: unconditional]──────────────────────────────────────────────┐
#     │                                                                                  │
#     │ [branch 2: unconditional]                                                        ▼
#     ▼                                                                         lbd_broad_search
#   lbd_extract_concepts                                                                 │
#     │                                                                                  │
#     │ [conditional via lbd_dispatch_concepts]                                          │
#     │   Send("lbd_fetch_concept_papers", LBDConceptFetchState) × N concepts           │
#     ▼                                                                                  │
#   lbd_fetch_concept_papers (×N, parallel)                                              │
#     │                                                                                  │
#     ▼                                                                                  │
#   lbd_collect_concept_papers  ◄──────────────────────────────────────────────────────┘
#     │                                   (all branches join here via LangGraph fan-in)
#     │
#     ▼
#   lbd_discover_connections ← single dedup: writes merged_papers; LLM extracts B-terms
#     │
#     ▼
#   lbd_synthesise            ← LLM narrative over merged_papers (no re-dedup)
#     │
#     ▼
#   lbd_finalise              ← assembles result dict from state
#     │
#     ▼
#    END
#
# lbd_dispatch_concepts is a conditional edge routing function (not a node):
# fans out one Send() per seed concept.
# lbd_discover_connections is the join node where lbd_collect_concept_papers
# (concept fetch chain) AND lbd_broad_search (broad ASTA search) both converge.
# At join time, both concept_papers and broad_search_results are in state —
# merged_papers is written here (single dedup point).
"""
from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime, timezone

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from src.config import DEFAULT_RESEARCH_MODEL
from src.core.llm_utils import acall_llm
from src.core.research_utils import deduplicate_papers
from src.graph.agents.tools.search_tools import search_asta
from src.graph.state import LBDAgentState, LBDConceptFetchState

logger = logging.getLogger(__name__)

_MAX_CONCEPTS_FOR_FETCH = 5


# ---------------------------------------------------------------------------
# Node: lbd_start — trivial entry enabling parallel fan-out
# ---------------------------------------------------------------------------


async def lbd_start(state: LBDAgentState, config: RunnableConfig) -> dict:
    """Trivial entry node — hub that fans out to lbd_extract_concepts and
    lbd_broad_search simultaneously from a single source node."""
    return {}


# ---------------------------------------------------------------------------
# Node: lbd_broad_search — broad ASTA search on original query (parallel)
# ---------------------------------------------------------------------------


async def lbd_broad_search(state: LBDAgentState, config: RunnableConfig) -> dict:
    """Broad ASTA search on original query. Runs in parallel with
    lbd_extract_concepts. Results join at lbd_discover_connections."""
    if state.get("broad_search_results") is not None:
        return {}

    try:
        papers = await search_asta.ainvoke({"query": state["query"], "top_k": 10})
        return {"broad_search_results": papers or []}
    except Exception as exc:
        logger.warning("lbd_broad_search failed: %s", exc)
        return {
            "broad_search_results": [],
            "errors": [f"lbd_broad_search: {exc}"],
        }


# ---------------------------------------------------------------------------
# Node: lbd_extract_concepts — LLM extracts seed concepts
# ---------------------------------------------------------------------------


async def lbd_extract_concepts(state: LBDAgentState, config: RunnableConfig) -> dict:
    """LLM: extract 3-7 seed concepts from query for Swanson ABC discovery."""
    if state.get("seed_concepts") is not None:
        return {}

    from src.prompts.research_prompts import LBD_CONCEPT_SYSTEM, LBD_CONCEPT_TEMPLATE

    model = (config.get("configurable") or {}).get("research_model", DEFAULT_RESEARCH_MODEL)
    try:
        response = await acall_llm(
            LBD_CONCEPT_TEMPLATE.format(query=state["query"]),
            LBD_CONCEPT_SYSTEM,
            model=model,
            config=config,
        )
        concepts = _parse_concepts(response)
    except Exception as exc:
        logger.warning("lbd_extract_concepts failed: %s", exc)
        return {
            "seed_concepts": [state["query"]],
            "errors": [f"lbd_extract_concepts: {exc}"],
        }

    return {"seed_concepts": concepts[:_MAX_CONCEPTS_FOR_FETCH]}


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
# Conditional edge: lbd_dispatch_concepts — pure router, returns list[Send]
# ---------------------------------------------------------------------------


async def lbd_dispatch_concepts(state: LBDAgentState) -> list[Send] | str:
    """Fan out: one Send per seed concept. Used as conditional edge function."""
    concepts = state.get("seed_concepts") or [state["query"]]
    sends = []
    for concept in concepts[:_MAX_CONCEPTS_FOR_FETCH]:
        sends.append(Send("lbd_fetch_concept_papers", LBDConceptFetchState(
            concept=concept,
            query=state["query"],
            top_k=15,
            result=None,
        )))
    return sends or "lbd_collect_concept_papers"


# ---------------------------------------------------------------------------
# Node: lbd_fetch_concept_papers — worker, HTTP only
# ---------------------------------------------------------------------------


async def lbd_fetch_concept_papers(state: LBDConceptFetchState, config: RunnableConfig) -> dict:
    """Worker: search ASTA for papers about one concept. HTTP only, no LLM."""
    if state.get("result") is not None:
        return {"concept_papers": state["result"]}

    start = time.monotonic()
    called_at = datetime.now(timezone.utc).isoformat()
    concept = state["concept"]

    try:
        papers = await search_asta.ainvoke({"query": concept, "top_k": state["top_k"]})
        duration_ms = int((time.monotonic() - start) * 1000)
        trace = {
            "tool_name": "search_asta",
            "called_at": called_at,
            "duration_ms": duration_ms,
            "success": True,
            "concept": concept,
            "result_count": len(papers) if papers else 0,
        }
        return {"concept_papers": papers or [], "tool_traces": [trace]}
    except Exception as exc:
        duration_ms = int((time.monotonic() - start) * 1000)
        return {
            "errors": [f"lbd_fetch_{concept}: {exc}"],
            "tool_traces": [{
                "tool_name": "search_asta",
                "called_at": called_at,
                "duration_ms": duration_ms,
                "success": False,
                "concept": concept,
                "error_message": str(exc),
            }],
        }


# ---------------------------------------------------------------------------
# Node: lbd_collect_concept_papers — reducer, deduplicates concept papers
# ---------------------------------------------------------------------------


async def lbd_collect_concept_papers(state: LBDAgentState, config: RunnableConfig) -> dict:
    """Deduplicate papers collected from concept fetch workers.
    Writes paper_count based on concept_papers only (broad_search_results not yet joined).
    merged_papers (concept + broad, fully deduped) is written by lbd_discover_connections.
    """
    all_papers = state.get("concept_papers") or []
    unique = deduplicate_papers(all_papers)
    return {"paper_count": len(unique)}


# ---------------------------------------------------------------------------
# Node: lbd_discover_connections — join node, single dedup, LLM B-term extraction
# ---------------------------------------------------------------------------


async def lbd_discover_connections(state: LBDAgentState, config: RunnableConfig) -> dict:
    """Join node: lbd_collect_concept_papers and lbd_broad_search both converge here.
    At this point both concept_papers and broad_search_results are in state.
    Single dedup point: writes merged_papers. Then extracts B-term bridges via LLM.
    """
    if state.get("discovered_concepts") is not None:
        return {}

    from src.prompts.research_prompts import LBD_CONCEPT_SYSTEM, LBD_CONCEPT_TEMPLATE

    merged = deduplicate_papers(
        (state.get("concept_papers") or []) + (state.get("broad_search_results") or [])
    )

    if not merged:
        return {"merged_papers": [], "discovered_concepts": []}

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
            "merged_papers": merged,
            "discovered_concepts": [],
            "errors": [f"lbd_discover_connections: {exc}"],
        }

    discovered = [{"term": t, "type": "bridge"} for t in b_terms[:10]]
    return {"merged_papers": merged, "discovered_concepts": discovered}


# ---------------------------------------------------------------------------
# Node: lbd_synthesise — LLM narrative summary over merged_papers
# ---------------------------------------------------------------------------


async def lbd_synthesise(state: LBDAgentState, config: RunnableConfig) -> dict:
    """LLM: write narrative summary of discovered connections.
    Reads merged_papers written by lbd_discover_connections — no re-deduplication.
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

    builder.add_node("lbd_start", lbd_start)
    builder.add_node("lbd_broad_search", lbd_broad_search)
    builder.add_node("lbd_extract_concepts", lbd_extract_concepts)
    # lbd_dispatch_concepts is a conditional edge function, not a node
    builder.add_node("lbd_fetch_concept_papers", lbd_fetch_concept_papers)
    builder.add_node("lbd_collect_concept_papers", lbd_collect_concept_papers)
    builder.add_node("lbd_discover_connections", lbd_discover_connections)
    builder.add_node("lbd_synthesise", lbd_synthesise)
    builder.add_node("lbd_finalise", lbd_finalise)

    # ── Parallel branches from lbd_start ─────────────────────────────────────
    builder.add_edge(START, "lbd_start")
    builder.add_edge("lbd_start", "lbd_extract_concepts")
    builder.add_edge("lbd_start", "lbd_broad_search")

    # Branch 1: concept extraction → per-concept fan-out → collect
    builder.add_conditional_edges(
        "lbd_extract_concepts",
        lbd_dispatch_concepts,
        {
            "lbd_fetch_concept_papers": "lbd_fetch_concept_papers",
            "lbd_collect_concept_papers": "lbd_collect_concept_papers",
        },
    )
    builder.add_edge("lbd_fetch_concept_papers", "lbd_collect_concept_papers")

    # ── Join at lbd_discover_connections ─────────────────────────────────────
    builder.add_edge("lbd_collect_concept_papers", "lbd_discover_connections")
    builder.add_edge("lbd_broad_search", "lbd_discover_connections")

    builder.add_edge("lbd_discover_connections", "lbd_synthesise")
    builder.add_edge("lbd_synthesise", "lbd_finalise")
    builder.add_edge("lbd_finalise", END)

    return builder.compile()


lbd_graph = build_lbd_graph()
