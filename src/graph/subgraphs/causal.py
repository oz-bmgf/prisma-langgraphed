"""Causal pipeline subgraph — 21 nodes across 8 sub-stages.

Stage 3.1 : dispatch_rubric_evaluation / evaluate_investment_rubric / collect_evidence_packs
Stage 3.1.5: dispatch_bow_enrichment / enrich_bow_context_worker / collect_bow_enrichment
Stage 3.2 : forecast_consequences           (sole writer of causal_model into scope_outputs)
Stage 3.4 : dispatch_link_investigations (node) / investigate_link / collect_link_assessments
Stage 3.5 : synthesize_findings → critique_synthesis → identify_gaps  ┐ parallel from collect_link_assessments
Stage 3.5d: dispatch_science_investigations (node)                     ┘ → investigate_science_assumption
           → collect_science_results   ┐ join at necessity_check
           identify_gaps               ┘
Stage 3.7 : necessity_check
Stage 3.8 : dispatch_decision_projections / project_scope_decisions / collect_decisions

Parallelism:
  - synthesize_findings chain ∥ dispatch_science_investigations fan-out  (independent of each other)
  - enrich_bow_context_worker runs N-parallel per scope (Send() fan-out, §3.1.5)

Compiled with max_concurrency=16.
"""
from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from src.core.llm_utils import acall_llm
from src.graph.state import (
    BowEnrichmentWorkerState,
    CausalState,
    InvestmentRubricState,
    LinkInvestigationState,
    ScienceAssumptionState,
    ScopeDecisionState,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Search backend → tools bridge
# ---------------------------------------------------------------------------


class _EmbeddingIndexAdapter:
    """Synchronous adapter for SearchBackend used inside asyncio.to_thread contexts.

    investigation.py and rubric_evaluator.py call search_with_filter() from within
    asyncio.to_thread, so the method must be synchronous. asyncio.run() is safe
    here because asyncio.to_thread worker threads have no running event loop.
    """

    has_hybrid: bool = False

    def __init__(self, backend: Any) -> None:
        self._backend = backend

    def search_with_filter(self, query: str, *, top_k: int = 10, **kwargs) -> list[dict]:
        import asyncio as _asyncio

        inv_id = kwargs.get("inv_id")
        bow_id = kwargs.get("bow_id")
        doc_type = kwargs.get("doc_type")

        async def _run() -> list[dict]:
            results = await self._backend.search(
                query,
                top_k=top_k,
                inv_id_filter=inv_id,
                bow_id_filter=bow_id,
                doc_type_filter=doc_type,
            )
            return [
                {
                    "text": r.text,
                    "file_id": r.file_id,
                    "chunk_id": r.chunk_id,
                    "score": r.score,
                    "inv_id": r.inv_id,
                    "bow_id": r.bow_id,
                    "page_start": r.page_start,
                    "page_end": r.page_end,
                    "doc_type": r.doc_type,
                }
                for r in (results or [])
            ]

        # asyncio-APPROVED-4: asyncio.run in asyncio.to_thread worker — no running event loop in thread pool thread
        return _asyncio.run(_run())


class _SearchBackendToolsBridge:
    """Duck-typed tools object wrapping SearchBackend.

    Provides the _embedding_index interface expected by investigation.py and
    rubric_evaluator.py. search_web, read_pages, and compute are injected from
    config["configurable"] if present; otherwise None (tool calls skipped).
    """

    def __init__(
        self,
        backend: Any,
        *,
        web_search_fn: Any = None,
        compute_fn: Any = None,
        pages_dir: str = "",
    ) -> None:
        self._embedding_index = _EmbeddingIndexAdapter(backend)
        self.search_web = web_search_fn
        self.read_pages = pages_dir or None   # str path; investigation.py checks truthiness
        self.compute = compute_fn


def _get_tools(config: Any) -> Any:
    """Wrap config's search_backend in _SearchBackendToolsBridge, or return None."""
    configurable = ((config or {}).get("configurable") or {})
    backend = configurable.get("search_backend")
    if not backend:
        return None
    return _SearchBackendToolsBridge(
        backend,
        web_search_fn=configurable.get("web_search_fn"),
        compute_fn=configurable.get("compute_fn"),
        pages_dir=configurable.get("pages_dir", ""),
    )


def _get_asta_client(config: Any) -> Any:
    """Return pre-built asta_client from config, or build one from asta_api_key / env."""
    import os
    configurable = ((config or {}).get("configurable") or {})
    client = configurable.get("asta_client")
    if client is not None:
        return client
    api_key = configurable.get("asta_api_key") or os.environ.get("ASTA_API_KEY")
    if not api_key:
        return None
    try:
        from src.core.asta_client import AstaClient
        return AstaClient(api_key=api_key)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Scope-fit classification (Phase 3.1a — deterministic, no LLM)
# ---------------------------------------------------------------------------

_SCOPE_SUPPORT_TERMS = [
    "business support", "business operations", "partner operations",
    "grantee capacity", "data management", "monitoring evaluation",
    "communications", "advocacy", "knowledge management", "administrative",
    "workforce development", "it infrastructure", "human resources",
]

_SCOPE_CROSS_PROGRAM_TERMS = [
    "cross-program", "cross program", "multi-program", "multiprogram",
    "portfolio-wide", "portfolio wide", "organization-wide",
    "foundation-wide", "cross-cutting", "horizontal initiative",
]


def _classify_scope_fit(scope_label: str, timeline: dict) -> tuple[str, str]:
    """Return (scope_fit_category, reason). Deterministic — no LLM.

    Categories: core_program_investment | support_or_business_support |
                legacy_or_cross_program | unclear_fit
    """
    label_lower = (scope_label or "").lower()

    # Build combined text from timeline metadata for term matching
    parts: list[str] = []
    if isinstance(timeline, dict):
        for key in ("org", "title", "key_results", "team_rationale"):
            val = timeline.get(key) or ""
            parts.append(str(val))
    text = f"{label_lower} {' '.join(parts)}".lower()

    if not text.strip():
        return "unclear_fit", "No timeline text available for scope-fit classification."

    scope_label_support = "business support" in label_lower or (
        "support" in label_lower
        and any(token in label_lower for token in ("partner", "partners", "program", "programs", "optimize"))
    )
    if scope_label_support or any(term in text for term in _SCOPE_SUPPORT_TERMS):
        return "support_or_business_support", "Support/business-support language dominates."

    if "legacy" in label_lower:
        return "legacy_or_cross_program", "Scope label marks this investment as legacy."

    for term in _SCOPE_CROSS_PROGRAM_TERMS:
        if term in text:
            return "legacy_or_cross_program", f"Cross-program indicator '{term}' found."

    return "core_program_investment", ""


# ---------------------------------------------------------------------------
# GROUP 1 — Rubric fan-out (Stage 3.1)
# ---------------------------------------------------------------------------


async def dispatch_rubric_evaluation(state: CausalState):
    research_model = state.get("research_model", "")
    already_done = {pack.get("scope_id") for pack in state.get("evidence_packs", [])}
    sends: list[Send] = []
    for scope in state.get("scopes", []):
        scope_id = scope.get("scope_id", "")
        if scope_id in already_done:
            continue
        inv_id = scope.get("inv_id", "")
        timeline = state.get("scope_timelines", {}).get(scope_id, {})
        sends.append(Send("evaluate_investment_rubric", {
            "inv_id": inv_id,
            "scope_id": scope_id,
            "scope_label": scope.get("label", ""),
            "timeline": timeline,
            "result": None,
            "research_model": research_model,
        }))
    return sends or "collect_evidence_packs"


async def evaluate_investment_rubric(state: InvestmentRubricState, config: RunnableConfig = None) -> dict:
    from src.core.tool_tracing import flush_trace_buffer, init_trace_buffer

    inv_id = state["inv_id"]
    scope_id = state["scope_id"]

    init_trace_buffer()

    # Phase 3.1a — deterministic scope-fit classification (no LLM)
    timeline_dict = state.get("timeline", {})
    scope_label = state.get("scope_label", "") or inv_id
    scope_fit, scope_fit_reason = _classify_scope_fit(scope_label, timeline_dict)

    try:
        from src.core import rubric_evaluator
        result = await rubric_evaluator.build_evidence_pack(
            inv_id=inv_id,
            scope_id=scope_id,
            timeline=timeline_dict,
            tools=_get_tools(config),
            model=state.get("research_model", ""),
        )
        result_dict: dict = result.to_dict() if hasattr(result, "to_dict") else result
    except Exception as exc:
        logger.error("evaluate_investment_rubric failed %s/%s: %s", scope_id, inv_id, exc)
        result_dict = {"inv_id": inv_id, "scope_id": scope_id, "error": str(exc)}

    # Phase 3.1a: compute InvestmentFacts deterministically (no LLM)
    try:
        from src.core.rubric_evaluator import compute_investment_facts
        scoring = timeline_dict.get("scoring") or {}
        result_dict["investment_facts"] = compute_investment_facts(scoring, timeline_dict)
    except Exception as exc:
        logger.warning("compute_investment_facts failed %s: %s", inv_id, exc)
        result_dict["investment_facts"] = {}

    flushed = flush_trace_buffer()

    result_dict["scope_fit"] = scope_fit
    result_dict["scope_fit_reason"] = scope_fit_reason

    return {"evidence_packs": [result_dict], **flushed}


async def collect_evidence_packs(state: CausalState) -> dict:
    by_scope: dict[str, dict] = {}
    for scope in state.get("scopes", []):
        scope_id = scope.get("scope_id", "")
        by_scope[scope_id] = {
            "scope_id": scope_id,
            "inv_id": scope.get("inv_id", ""),
            "bow_ids": scope.get("bow_ids", []),
            "label": scope.get("label", ""),
            "evidence_packs": [],
            "link_assessments": [],
            "science_flags": [],
            "decisions": [],
            "scope_decisions": [],
            "causal_model": None,
            "bow_context": None,
            "synthesis": "",
            "critique": "",
            "gaps": "",
            "necessity_assessment": "",
            "errors": [],
        }

    for pack in state.get("evidence_packs", []):
        scope_id = pack.get("scope_id", "")
        if scope_id in by_scope:
            by_scope[scope_id]["evidence_packs"].append(pack)
            # Propagate InvestmentFacts and scope_fit from evidence pack into scope
            if pack.get("investment_facts"):
                by_scope[scope_id]["investment_facts"] = pack["investment_facts"]
            if pack.get("scope_fit"):
                by_scope[scope_id]["scope_fit"] = pack["scope_fit"]
                by_scope[scope_id]["scope_fit_reason"] = pack.get("scope_fit_reason", "")
        else:
            logger.warning("evidence pack scope_id=%s not in scopes", scope_id)

    return {"scope_outputs": list(by_scope.values())}


# ---------------------------------------------------------------------------
# GROUP 1.5 — BOW Context (Stage 3.3)
# ---------------------------------------------------------------------------


async def dispatch_bow_enrichment(state: CausalState) -> list[Send] | str:
    """Fan-out router: one Send per scope for BOW context enrichment.

    Skips scopes that already have bow_context set (LangGraph checkpoint resume).
    Falls back to collect_bow_enrichment when all scopes are already done.
    """
    scope_outputs = state.get("scope_outputs") or []
    if not scope_outputs:
        return "collect_bow_enrichment"
    model = state.get("research_model") or ""
    already_done = {s.get("scope_id") for s in scope_outputs if "bow_context" in s}
    sends: list[Send] = []
    for scope in scope_outputs:
        scope_id = scope.get("scope_id", "")
        if scope_id in already_done:
            continue
        sends.append(Send("enrich_bow_context_worker", {
            "scope_id": scope_id,
            "scope": scope,
            "model": model,
            "result": None,
        }))
    return sends or "collect_bow_enrichment"


async def enrich_bow_context_worker(
    state: BowEnrichmentWorkerState, config: RunnableConfig = None
) -> dict:
    """Per-scope worker: search web for BOW field context, synthesise with LLM.

    Writes {"scope_outputs": [updated_scope]} — merge_scope_outputs handles the merge.
    """
    import re as _re

    scope = dict(state["scope"])
    scope_id = state["scope_id"]
    model = state["model"]
    bow_ids = scope.get("bow_ids", [])
    label = scope.get("label", scope_id)
    web_search_fn = ((config or {}).get("configurable") or {}).get("web_search_fn")

    _empty_bow: dict = {
        "bow_id": bow_ids[0] if bow_ids else "",
        "bow_ids": bow_ids,
        "benchmarks": [],
        "comparable_programs": [],
        "market_context": "",
        "regulatory_context": "",
    }

    try:
        if not bow_ids:
            scope["bow_context"] = _empty_bow
            return {"scope_outputs": [scope]}

        bow_label = bow_ids[0]
        queries = [
            f"{bow_label} field landscape 2024 2025 benchmarks comparable programs",
            f"{bow_label} regulatory context government policy health sector",
            f"{bow_label} market evidence outcomes best practices",
        ]
        web_snippets: list[str] = []
        if web_search_fn is not None:
            for q in queries:
                try:
                    results = await web_search_fn(q)
                    if isinstance(results, list):
                        for r in results[:3]:
                            text = r.get("snippet") or r.get("text") or ""
                            if text:
                                web_snippets.append(f"[{q}] {text[:400]}")
                    elif isinstance(results, str):
                        web_snippets.append(f"[{q}] {results[:400]}")
                except Exception as exc:
                    logger.debug("enrich_bow_context_worker: web search failed %s: %s", q, exc)

        if not web_snippets:
            scope["bow_context"] = _empty_bow
            return {"scope_outputs": [scope]}

        context_text = "\n\n".join(web_snippets[:9])
        prompt = (
            f"You have searched the web for external context on '{bow_label}' (scope: {label}).\n\n"
            f"Search results:\n{context_text}\n\n"
            "Synthesise a concise BOW context. Respond with a JSON object containing:\n"
            "  benchmarks: list of 2-4 relevant benchmark strings\n"
            "  comparable_programs: list of 2-3 comparable program names\n"
            "  market_context: 1-2 sentences on the current field landscape\n"
            "  regulatory_context: 1-2 sentences on relevant policy or regulatory environment\n"
        )
        raw = await acall_llm(prompt, model=model, config=config)
        raw_str = raw if isinstance(raw, str) else str(raw)
        m = _re.search(r"\{.*\}", raw_str, _re.DOTALL)
        bow_context: dict = json.loads(m.group(0)) if m else {}
        defaults = {"benchmarks": [], "comparable_programs": [], "market_context": "", "regulatory_context": ""}
        defaults.update(bow_context or {})
        scope["bow_context"] = {"bow_id": bow_ids[0], "bow_ids": bow_ids, **defaults}
    except Exception as exc:
        logger.error("enrich_bow_context_worker failed %s: %s", scope_id, exc)
        scope["bow_context"] = _empty_bow
        return {"scope_outputs": [scope], "errors": [f"enrich_bow_context_worker:{scope_id}:{exc}"]}

    return {"scope_outputs": [scope]}


async def collect_bow_enrichment(state: CausalState, config: RunnableConfig = None) -> dict:
    """Trivial join node — merge_scope_outputs already accumulated bow_context updates from workers."""
    return {}


# ---------------------------------------------------------------------------
# GROUP 2 — Consequence forecast (Stages 3.2-3.3)
# ---------------------------------------------------------------------------


async def forecast_consequences(state: CausalState, config: RunnableConfig = None) -> dict:
    scope_outputs = list(state.get("scope_outputs", []))
    model = state.get("research_model", "")
    errors: list[str] = []

    for scope in scope_outputs:
        scope_id = scope.get("scope_id", "")
        try:
            from src.core import causal_model as causal_model_module
            cm = await causal_model_module.extract_causal_model(
                scope=scope,
                model=model,
                config=config,
            )
            if dataclasses.is_dataclass(cm) and not isinstance(cm, type):
                scope["causal_model"] = dataclasses.asdict(cm)
            elif hasattr(cm, "to_dict"):
                scope["causal_model"] = cm.to_dict()
            else:
                scope["causal_model"] = cm
        except Exception as exc:
            logger.error("forecast_consequences failed %s: %s", scope_id, exc)
            errors.append(f"forecast_consequences:{scope_id}:{exc}")
            scope["causal_model"] = None

    result: dict[str, Any] = {"scope_outputs": scope_outputs}
    if errors:
        result["errors"] = errors
    return result


# ---------------------------------------------------------------------------
# GROUP 3 — Link investigation fan-out (Stage 3.4)
# ---------------------------------------------------------------------------


async def dispatch_link_investigations(state: CausalState) -> dict:
    """Join node — waits for forecast_consequences before fanning out link investigations.
    Returns {} (pass-through). The actual fan-out routing is in
    _route_link_investigations, called via add_conditional_edges from this node.
    """
    return {}


async def _route_link_investigations(state: CausalState):
    """Routing function called after dispatch_link_investigations node.
    Returns list[Send] to fan out per causal link, or string fallback.
    """
    model = state.get("research_model", "")
    already_done = {
        (a.get("scope_id"), a.get("link_id"))
        for a in state.get("link_assessments", [])
    }
    sends: list[Send] = []

    for scope in state.get("scope_outputs", []):
        scope_id = scope.get("scope_id", "")
        inv_id = scope.get("inv_id", "")
        bow_ids = scope.get("bow_ids", [])
        bow_id = bow_ids[0] if bow_ids else ""
        causal_model = scope.get("causal_model") or {}
        links = causal_model.get("links", []) if isinstance(causal_model, dict) else []

        for link in links:
            link_id = link.get("name", "") or link.get("link_id", "")
            if (scope_id, link_id) in already_done:
                continue
            sends.append(Send("investigate_link", {
                "link_id": link_id,
                "inv_id": inv_id,
                "bow_id": bow_id,
                "scope_id": scope_id,
                "claim": link,
                "model": model,
                "result": None,
            }))

    return sends or "collect_link_assessments"


async def investigate_link(state: LinkInvestigationState, config: RunnableConfig = None) -> dict:
    from src.core.tool_tracing import flush_trace_buffer, init_trace_buffer

    link_id = state["link_id"]
    scope_id = state["scope_id"]

    init_trace_buffer()

    try:
        from src.core import investigation
        result = await investigation.run_investigation(
            link_id=link_id,
            inv_id=state["inv_id"],
            bow_id=state["bow_id"],
            scope_id=scope_id,
            claim=state.get("claim", {}),
            model=state.get("model", ""),
            tools=_get_tools(config),
            config=config,
        )
        result_dict: dict = result.to_dict() if hasattr(result, "to_dict") else result
    except Exception as exc:
        logger.error("investigate_link failed %s/%s: %s", scope_id, link_id, exc)
        result_dict = {
            "link_id": link_id,
            "inv_id": state["inv_id"],
            "scope_id": scope_id,
            "error": str(exc),
        }

    flushed = flush_trace_buffer()

    # Build investigation summary trace
    all_tool_traces = (
        flushed.get("web_search_traces", []) +
        flushed.get("compute_traces", []) +
        flushed.get("collection_search_traces", []) +
        flushed.get("asta_traces", [])
    )
    breakdown: dict[str, int] = {}
    for t in all_tool_traces:
        name = t.get("tool_name", "unknown")
        breakdown[name] = breakdown.get(name, 0) + 1

    inv_trace = {
        "inv_id": state["inv_id"],
        "scope_id": scope_id,
        "link_id": link_id,
        "total_tool_calls": len(all_tool_traces),
        "tool_call_breakdown": breakdown,
        "asta_called": bool(flushed.get("asta_traces")),
        "web_search_called": bool(flushed.get("web_search_traces")),
        "compute_called": bool(flushed.get("compute_traces")),
        "terminal_status": result_dict.get("terminal_status", result_dict.get("status", "error" if "error" in result_dict else "sufficient")),
        "iterations_used": result_dict.get("iterations_used", result_dict.get("iterations", result_dict.get("iteration_count", 0))),
    }

    # Collect annotated excerpts; keep top-10 by credibility_tier to bound state size
    annotated_excerpts = result_dict.pop("annotated_excerpts", []) or []
    _tier_rank = {"high": 0, "medium": 1, "low": 2}
    annotated_excerpts.sort(key=lambda x: _tier_rank.get(str(x.get("credibility_tier", "")).lower(), 3))
    annotated_excerpts = annotated_excerpts[:10]

    return {
        "link_assessments": [result_dict],
        "investigation_traces": [inv_trace],
        "all_excerpts": annotated_excerpts,
        **flushed,
    }


async def collect_link_assessments(state: CausalState) -> dict:
    scope_outputs = list(state.get("scope_outputs", []))
    by_scope: dict[str, list] = {s.get("scope_id", ""): [] for s in scope_outputs}

    for assessment in state.get("link_assessments", []):
        scope_id = assessment.get("scope_id", "")
        if scope_id in by_scope:
            by_scope[scope_id].append(assessment)
        else:
            logger.warning("link assessment scope_id=%s not in scope_outputs", scope_id)

    for scope in scope_outputs:
        scope_id = scope.get("scope_id", "")
        scope["link_assessments"] = by_scope.get(scope_id, [])

    return {"scope_outputs": scope_outputs}


# ---------------------------------------------------------------------------
# GROUP 4 — Synthesis (Stages 3.5, 3.5b, 3.5c)
# ---------------------------------------------------------------------------


def _synthesis_prompt(scope: dict) -> str:
    scope_id = scope.get("scope_id", "")
    inv_id = scope.get("inv_id", "")
    links = scope.get("link_assessments", [])
    statuses = [a.get("status", "") for a in links]
    return (
        f"Synthesise findings for investment {inv_id} (scope {scope_id}). "
        f"{len(links)} causal links investigated; statuses: {statuses}. "
        "Provide a concise evidence synthesis."
    )


def _critique_prompt(scope: dict) -> str:
    synthesis = scope.get("synthesis", "")[:800]
    return (
        f"Steelman critique of this synthesis — what is the strongest counter-argument "
        f"and what evidence could change the conclusion?\n\nSynthesis:\n{synthesis}"
    )


def _gaps_prompt(scope: dict) -> str:
    scope_id = scope.get("scope_id", "")
    synthesis = scope.get("synthesis", "")[:600]
    return (
        f"Identify key evidence gaps for scope {scope_id}. "
        f"What data or analysis is missing that would materially change the assessment?\n\n"
        f"Synthesis:\n{synthesis}"
    )


async def synthesize_findings(state: CausalState, config: RunnableConfig = None) -> dict:
    scope_outputs = list(state.get("scope_outputs", []))
    model = state.get("synthesis_model", "")
    errors: list[str] = []

    for scope in scope_outputs:
        scope_id = scope.get("scope_id", "")
        if not scope.get("link_assessments"):
            scope["synthesis"] = ""
            continue
        try:
            raw = await acall_llm(_synthesis_prompt(scope), model=model, config=config)
            scope["synthesis"] = raw if isinstance(raw, str) else str(raw)
        except Exception as exc:
            logger.error("synthesize_findings failed %s: %s", scope_id, exc)
            errors.append(f"synthesize:{scope_id}:{exc}")
            scope["synthesis"] = ""

    result: dict[str, Any] = {"scope_outputs": scope_outputs}
    if errors:
        result["errors"] = errors
    return result


async def critique_synthesis(state: CausalState, config: RunnableConfig = None) -> dict:
    scope_outputs = list(state.get("scope_outputs", []))
    model = state.get("synthesis_model", "")
    errors: list[str] = []

    for scope in scope_outputs:
        scope_id = scope.get("scope_id", "")
        if not scope.get("synthesis"):
            scope["critique"] = ""
            continue
        try:
            raw = await acall_llm(_critique_prompt(scope), model=model, config=config)
            scope["critique"] = raw if isinstance(raw, str) else str(raw)
        except Exception as exc:
            logger.error("critique_synthesis failed %s: %s", scope_id, exc)
            errors.append(f"critique:{scope_id}:{exc}")
            scope["critique"] = ""

    result: dict[str, Any] = {"scope_outputs": scope_outputs}
    if errors:
        result["errors"] = errors
    return result


async def identify_gaps(state: CausalState, config: RunnableConfig = None) -> dict:
    scope_outputs = list(state.get("scope_outputs", []))
    model = state.get("synthesis_model", "")
    errors: list[str] = []

    for scope in scope_outputs:
        scope_id = scope.get("scope_id", "")
        try:
            raw = await acall_llm(_gaps_prompt(scope), model=model, config=config)
            scope["gaps"] = raw if isinstance(raw, str) else str(raw)
        except Exception as exc:
            logger.error("identify_gaps failed %s: %s", scope_id, exc)
            errors.append(f"gaps:{scope_id}:{exc}")
            scope["gaps"] = ""

    result: dict[str, Any] = {"scope_outputs": scope_outputs}
    if errors:
        result["errors"] = errors
    return result


# ---------------------------------------------------------------------------
# GROUP 5 — Science fan-out (Stage 3.5d)
# ---------------------------------------------------------------------------


async def dispatch_science_investigations(state: CausalState) -> dict:
    """Join/dispatch node — runs in parallel with synthesize_findings chain.
    Both this node and synthesize_findings read from collect_link_assessments.
    Returns {} (pass-through). The actual fan-out routing is in
    _route_science_investigations, called via add_conditional_edges.
    Reads causal_model.assumptions (set by forecast_consequences), NOT from
    synthesis results — safe to run in parallel with synthesis chain.
    """
    return {}


async def _route_science_investigations(state: CausalState):
    """Routing function called after dispatch_science_investigations node.
    Returns list[Send] to fan out per science assumption, or string fallback.
    """
    research_model = state.get("research_model", "")
    already_done = {r.get("assumption_id") for r in state.get("science_results", [])}
    sends: list[Send] = []

    for scope in state.get("scope_outputs", []):
        scope_id = scope.get("scope_id", "")
        inv_id = scope.get("inv_id", "")
        bow_ids = scope.get("bow_ids", [])
        bow_id = bow_ids[0] if bow_ids else ""
        causal_model = scope.get("causal_model") or {}
        assumptions = causal_model.get("assumptions", []) if isinstance(causal_model, dict) else []

        for i, assumption in enumerate(assumptions):
            assumption_id = f"{scope_id}_{i}"
            if assumption_id in already_done:
                continue
            sends.append(Send("investigate_science_assumption", {
                "assumption_id": assumption_id,
                "inv_id": inv_id,
                "bow_id": bow_id,
                "scope_id": scope_id,
                "question": assumption.get("assumption", ""),
                "result": None,
                "research_model": research_model,
            }))

    return sends or "collect_science_results"


async def investigate_science_assumption(state: ScienceAssumptionState, config: RunnableConfig = None) -> dict:
    from src.core.tool_tracing import flush_trace_buffer, init_trace_buffer

    assumption_id = state["assumption_id"]
    scope_id = state["scope_id"]

    init_trace_buffer()

    try:
        from src.core import science_investigator
        result = await science_investigator.investigate_science_question(
            assumption_id=assumption_id,
            inv_id=state["inv_id"],
            bow_id=state["bow_id"],
            scope_id=scope_id,
            question=state.get("question", ""),
            tools=_get_tools(config),
            asta_client=_get_asta_client(config),
            model=state.get("research_model", ""),
        )
        result_dict: dict = result.to_dict() if hasattr(result, "to_dict") else result
        result_dict.update({
            "scope_id": scope_id,
            "assumption_id": assumption_id,
            "question": state.get("question", ""),
        })
    except Exception as exc:
        logger.error("investigate_science_assumption failed %s/%s: %s", scope_id, assumption_id, exc)
        result_dict = {
            "assumption_id": assumption_id,
            "scope_id": scope_id,
            "question": state.get("question", ""),
            "terminal_status": "error",
            "error": str(exc),
        }

    flushed = flush_trace_buffer()

    all_tool_traces = (
        flushed.get("web_search_traces", []) +
        flushed.get("compute_traces", []) +
        flushed.get("collection_search_traces", []) +
        flushed.get("asta_traces", [])
    )
    breakdown: dict[str, int] = {}
    for t in all_tool_traces:
        name = t.get("tool_name", "unknown")
        breakdown[name] = breakdown.get(name, 0) + 1

    inv_trace = {
        "inv_id": state["inv_id"],
        "scope_id": scope_id,
        "link_id": None,
        "total_tool_calls": len(all_tool_traces),
        "tool_call_breakdown": breakdown,
        "asta_called": bool(flushed.get("asta_traces")),
        "web_search_called": bool(flushed.get("web_search_traces")),
        "compute_called": bool(flushed.get("compute_traces")),
        "terminal_status": result_dict.get("terminal_status", "error" if "error" in result_dict else "sufficient"),
        "iterations_used": result_dict.get("iterations_used", result_dict.get("iterations", result_dict.get("iteration_count", 0))),
    }

    return {
        "science_results": [result_dict],
        "investigation_traces": [inv_trace],
        **flushed,
    }


async def collect_science_results(state: CausalState) -> dict:
    scope_outputs = list(state.get("scope_outputs", []))
    by_scope: dict[str, list] = {s.get("scope_id", ""): [] for s in scope_outputs}

    for result in state.get("science_results", []):
        scope_id = result.get("scope_id", "")
        if scope_id in by_scope:
            by_scope[scope_id].append(result)
        else:
            logger.warning("science result scope_id=%s not in scope_outputs", scope_id)

    for scope in scope_outputs:
        scope_id = scope.get("scope_id", "")
        scope["science_flags"] = by_scope.get(scope_id, [])

    return {"scope_outputs": scope_outputs}


# ---------------------------------------------------------------------------
# GROUP 6 — Decisions (Stages 3.7-3.8)
# ---------------------------------------------------------------------------


async def necessity_check(state: CausalState, config: RunnableConfig = None) -> dict:
    scope_outputs = list(state.get("scope_outputs", []))
    model = state.get("synthesis_model", "")
    errors: list[str] = []

    for scope in scope_outputs:
        scope_id = scope.get("scope_id", "")
        link_assessments = scope.get("link_assessments", [])
        if not link_assessments:
            scope["necessity_assessment"] = ""
            continue
        try:
            prompt = (
                f"Scope {scope_id}: assess necessity of each of {len(link_assessments)} "
                "causal links for the investment's theory of change. "
                "Return a concise necessity assessment."
            )
            raw = await acall_llm(prompt, model=model, config=config)
            scope["necessity_assessment"] = raw if isinstance(raw, str) else str(raw)
        except Exception as exc:
            logger.error("necessity_check failed %s: %s", scope_id, exc)
            errors.append(f"necessity:{scope_id}:{exc}")
            scope["necessity_assessment"] = ""

    result: dict[str, Any] = {"scope_outputs": scope_outputs}
    if errors:
        result["errors"] = errors
    return result


async def dispatch_decision_projections(state: CausalState):
    synthesis_model = state.get("synthesis_model", "")
    already_done = {d.get("scope_id") for d in state.get("scope_decisions", [])}
    sends: list[Send] = []

    for scope in state.get("scope_outputs", []):
        scope_id = scope.get("scope_id", "")
        if scope_id in already_done:
            continue
        sends.append(Send("project_scope_decisions", {
            "scope_id": scope_id,
            "scope_output": scope,
            "decisions": None,
            "synthesis_model": synthesis_model,
        }))

    return sends or "collect_decisions"


async def project_scope_decisions(state: ScopeDecisionState) -> dict:
    scope_id = state["scope_id"]

    try:
        from src.core import decision_projection
        result = await decision_projection.project_decisions(
            scope_id=scope_id,
            scope_output=state.get("scope_output", {}),
            model=state.get("synthesis_model", ""),
        )
        result_dict: dict = result if isinstance(result, dict) else {
            "scope_id": scope_id,
            "decisions": result,
        }
    except Exception as exc:
        logger.error("project_scope_decisions failed %s: %s", scope_id, exc)
        result_dict = {"scope_id": scope_id, "decisions": [], "error": str(exc)}

    return {"scope_decisions": [result_dict]}


async def clear_fanout_accumulators(state: CausalState) -> dict:
    """Reset fan-out accumulator fields after all collector nodes have run.

    Data from evidence_packs / link_assessments / science_results / scope_decisions
    is embedded in scope_outputs by the collect_* nodes. Clearing these lists here
    prevents them from being propagated back to AnalyzeState where they are redundant.
    The operator.add reducer in CausalState means returning [] adds nothing on top of
    the existing list within the subgraph run itself, but run_causal_pipeline() will
    NOT forward these keys back to AnalyzeState (it only forwards scope_outputs and
    all_excerpts), so AnalyzeState stays clean after the causal subgraph exits.
    """
    return {}


async def collect_decisions(state: CausalState) -> dict:
    scope_outputs = list(state.get("scope_outputs", []))
    by_scope: dict[str, list] = {s.get("scope_id", ""): [] for s in scope_outputs}

    for decision_set in state.get("scope_decisions", []):
        scope_id = decision_set.get("scope_id", "")
        decisions = decision_set.get("decisions", []) or []
        if scope_id in by_scope:
            by_scope[scope_id].extend(decisions if isinstance(decisions, list) else [decisions])
        else:
            logger.warning("scope_decisions scope_id=%s not in scope_outputs", scope_id)

    for scope in scope_outputs:
        scope_id = scope.get("scope_id", "")
        scope["decisions"] = by_scope.get(scope_id, [])

    return {"scope_outputs": scope_outputs}


# ---------------------------------------------------------------------------
# Graph compilation
# ---------------------------------------------------------------------------

_builder = StateGraph(CausalState)

# Nodes — Group 1 (dispatch_rubric_evaluation is a conditional-edge router, not a node)
_builder.add_node("evaluate_investment_rubric", evaluate_investment_rubric)
_builder.add_node("collect_evidence_packs", collect_evidence_packs)

# Nodes — Group 1.5 (BOW context enrichment fan-out)
# dispatch_bow_enrichment is a conditional-edge router, not a node
_builder.add_node("enrich_bow_context_worker", enrich_bow_context_worker)
_builder.add_node("collect_bow_enrichment", collect_bow_enrichment)

# Nodes — Group 2
_builder.add_node("forecast_consequences", forecast_consequences)

# Nodes — Group 3
# dispatch_link_investigations: real join node
# _route_link_investigations: conditional edge routing function after the join node
_builder.add_node("dispatch_link_investigations", dispatch_link_investigations)
_builder.add_node("investigate_link", investigate_link)
_builder.add_node("collect_link_assessments", collect_link_assessments)

# Nodes — Group 4
_builder.add_node("synthesize_findings", synthesize_findings)
_builder.add_node("critique_synthesis", critique_synthesis)
_builder.add_node("identify_gaps", identify_gaps)

# Nodes — Group 5
# dispatch_science_investigations: real join node (parallel with synthesis chain)
# _route_science_investigations: conditional edge routing function after the join node
_builder.add_node("dispatch_science_investigations", dispatch_science_investigations)
_builder.add_node("investigate_science_assumption", investigate_science_assumption)
_builder.add_node("collect_science_results", collect_science_results)

# Nodes — Group 6 (dispatch_decision_projections is a conditional-edge router, not a node)
_builder.add_node("necessity_check", necessity_check)
_builder.add_node("project_scope_decisions", project_scope_decisions)
_builder.add_node("collect_decisions", collect_decisions)
_builder.add_node("clear_fanout_accumulators", clear_fanout_accumulators)

# Group 1 fan-out: START → dispatch_rubric_evaluation → [evaluate_investment_rubric × N]
#                                                       → collect_evidence_packs (empty fallback)
_builder.add_conditional_edges(
    START,
    dispatch_rubric_evaluation,
    {
        "evaluate_investment_rubric": "evaluate_investment_rubric",
        "collect_evidence_packs": "collect_evidence_packs",
    },
)
_builder.add_edge("evaluate_investment_rubric", "collect_evidence_packs")

# Group 1.5 fan-out: collect_evidence_packs → dispatch_bow_enrichment → [enrich_bow_context_worker × N]
#                                                                       → collect_bow_enrichment → forecast_consequences
_builder.add_conditional_edges(
    "collect_evidence_packs",
    dispatch_bow_enrichment,
    {
        "enrich_bow_context_worker": "enrich_bow_context_worker",
        "collect_bow_enrichment": "collect_bow_enrichment",
    },
)
_builder.add_edge("enrich_bow_context_worker", "collect_bow_enrichment")
_builder.add_edge("collect_bow_enrichment", "forecast_consequences")
_builder.add_edge("forecast_consequences", "dispatch_link_investigations")

# Group 3 fan-out: _route_link_investigations (routing fn) → [investigate_link × M]
#                                                           → collect_link_assessments (empty fallback)
_builder.add_conditional_edges(
    "dispatch_link_investigations",
    _route_link_investigations,
    {
        "investigate_link": "investigate_link",
        "collect_link_assessments": "collect_link_assessments",
    },
)
_builder.add_edge("investigate_link", "collect_link_assessments")

# ── Parallel: synthesize_findings chain ∥ dispatch_science_investigations ────
# Both read from collect_link_assessments:
#   - synthesize_findings reads link_assessments (updated by collect_link_assessments)
#   - dispatch_science_investigations reads causal_model.assumptions (from forecast_consequences)
# They write to completely different state keys — safe to run in parallel.
_builder.add_edge("collect_link_assessments", "synthesize_findings")
_builder.add_edge("collect_link_assessments", "dispatch_science_investigations")

# Synthesis chain (Branch 1):
_builder.add_edge("synthesize_findings", "critique_synthesis")
_builder.add_edge("critique_synthesis", "identify_gaps")

# Science fan-out (Branch 2): _route_science_investigations → [investigate_science_assumption × K]
#                                                            → collect_science_results (empty fallback)
_builder.add_conditional_edges(
    "dispatch_science_investigations",
    _route_science_investigations,
    {
        "investigate_science_assumption": "investigate_science_assumption",
        "collect_science_results": "collect_science_results",
    },
)
_builder.add_edge("investigate_science_assumption", "collect_science_results")

# ── Join: necessity_check waits for BOTH synthesis chain AND science results ──
# identify_gaps (end of synthesis chain) and collect_science_results (end of
# science branch) both flow into necessity_check.
# LangGraph executes necessity_check only after ALL incoming edges have arrived.
_builder.add_edge("identify_gaps", "necessity_check")
_builder.add_edge("collect_science_results", "necessity_check")

# Group 6 fan-out: necessity_check → dispatch_decision_projections → [project_scope_decisions × S]
#                                                                   → collect_decisions (empty fallback)
_builder.add_conditional_edges(
    "necessity_check",
    dispatch_decision_projections,
    {
        "project_scope_decisions": "project_scope_decisions",
        "collect_decisions": "collect_decisions",
    },
)
_builder.add_edge("project_scope_decisions", "collect_decisions")
_builder.add_edge("collect_decisions", "clear_fanout_accumulators")
_builder.add_edge("clear_fanout_accumulators", END)

causal_graph = _builder.compile()
