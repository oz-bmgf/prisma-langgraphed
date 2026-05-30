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
from pathlib import Path
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from src.core.llm_utils import acall_llm, safe_parse_json
from src.prompts.causal_prompts import NECESSITY_DISCOVER_SYSTEM, NECESSITY_VERIFY_SYSTEM
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


def _inject_search_config(config: Any, state: Any, *, inv_id: str = "", bow_id: str = "") -> Any:
    """Augment config["configurable"] with search_backend + per-link context.

    Investigation tools (search_investment, search_portfolio, search_bow, …) all
    read search_backend from config["configurable"].  When the graph runs through
    langgraph dev / Studio, configurable only has thread_id — search_backend is
    absent and every tool call returns "(search_backend not configured)".

    This helper mirrors the on-demand backend construction in _get_tools() and
    injects the result into a copy of the incoming config so the original is not
    mutated.  A pre-existing search_backend is never overwritten.
    """
    configurable = dict((config or {}).get("configurable") or {})
    if not configurable.get("search_backend"):
        ingested_dir = configurable.get("ingested_dir") or (state or {}).get("ingested_dir", "")
        collection_name = configurable.get("collection_name") or (state or {}).get("collection_name", "")
        if ingested_dir and collection_name:
            try:
                from src.backends.factory import build_search_backend
                configurable["search_backend"] = build_search_backend(ingested_dir, collection_name)
            except Exception as exc:
                logger.warning("_inject_search_config: could not build backend: %s", exc)
    # Per-link context — only set if absent so a pre-configured value is respected
    if inv_id and not configurable.get("inv_id"):
        configurable["inv_id"] = inv_id
    if bow_id and not configurable.get("bow_id"):
        configurable["bow_id"] = bow_id
    ingested_dir = configurable.get("ingested_dir") or (state or {}).get("ingested_dir", "")
    if ingested_dir and not configurable.get("pages_dir"):
        configurable["pages_dir"] = str(Path(ingested_dir) / "pages")
    doc_list = (state or {}).get("doc_list")
    if doc_list and not configurable.get("doc_list"):
        configurable["doc_list"] = doc_list
    return {**(config or {}), "configurable": configurable}


def _get_tools(config: Any, state: Any = None) -> Any:
    """Wrap config's search_backend in _SearchBackendToolsBridge, or return None.

    Falls back to building a backend from state.ingested_dir + state.collection_name
    when search_backend is absent from configurable (e.g. LangGraph Studio runs where
    the caller only supplies initial state and no configurable).
    """
    configurable = ((config or {}).get("configurable") or {})
    backend = configurable.get("search_backend")
    if not backend:
        ingested_dir = configurable.get("ingested_dir") or (state or {}).get("ingested_dir", "")
        collection_name = configurable.get("collection_name") or (state or {}).get("collection_name", "")
        if ingested_dir and collection_name:
            try:
                from src.backends.factory import build_search_backend
                backend = build_search_backend(ingested_dir, collection_name)
                logger.info(
                    "_get_tools: built search backend from state (ingested_dir=%s)", ingested_dir
                )
            except Exception as exc:
                logger.warning("_get_tools: could not build search backend: %s", exc)
                return None
        else:
            return None
    # Derive pages_dir: prefer configurable, fall back to {ingested_dir}/pages
    pages_dir = configurable.get("pages_dir", "") or (
        str(Path(ingested_dir) / "pages") if ingested_dir else ""
    )
    return _SearchBackendToolsBridge(
        backend,
        web_search_fn=configurable.get("web_search_fn"),
        compute_fn=configurable.get("compute_fn"),
        pages_dir=pages_dir,
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
        from src.core.agents.asta import AstaClient
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
    # F-049: dedup keyed by (scope_id, inv_id) — one Send per investment, not per scope
    already_done = {
        (pack.get("scope_id"), pack.get("inv_id"))
        for pack in state.get("evidence_packs", [])
    }
    sends: list[Send] = []
    for scope in state.get("scopes", []):
        scope_id = scope.get("scope_id", "")
        scope_tl = state.get("scope_timelines", {}).get(scope_id, {})
        # F-049: emit one Send per investment in the scope, not one per scope
        inv_ids = scope.get("inv_ids") or ([scope.get("inv_id", "")] if scope.get("inv_id") else [])
        for inv_id in inv_ids:
            if not inv_id:
                continue
            if (scope_id, inv_id) in already_done:
                continue
            # Consumers expect an InvestmentTimeline dict (title/org/scoring), not a ScopeTimeline.
            timeline = next(
                (inv for inv in (scope_tl.get("investments") or []) if inv.get("inv_id") == inv_id),
                scope_tl,  # fallback when investments list is absent
            )
            sends.append(Send("evaluate_investment_rubric", {
                "inv_id": inv_id,
                "scope_id": scope_id,
                "scope_label": scope.get("label", ""),
                "timeline": timeline,
                "result": None,
                "research_model": research_model,
                "ingested_dir": state.get("ingested_dir", ""),
                "collection_name": state.get("collection_name", ""),
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
            tools=_get_tools(config, state),
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
# GROUP 1.25 — BOW Causal Model (Stages 3.1a and 3.1b) — Orphans 4 & 5
# ---------------------------------------------------------------------------

_BOW_CAUSAL_SYSTEM = (
    "You are analyzing a global health investment portfolio for the "
    "Bill & Melinda Gates Foundation. Extract the shared theory of change "
    "for a group of investments that collectively pursue a strategic objective."
)

_BOW_CAUSAL_PROMPT = """\
These {n_inv} investments all contribute to "{bow_label}" in the {program} program.

PROGRAM THEORY OF CHANGE:
{program_toc}

INVESTMENTS (with per-investment summaries):
{inv_text}

Extract the SHARED theory of change — the causal chain from Foundation \
funding to impact that these investments collectively pursue.

Return JSON:
{{
  "theory_of_change": "BOW-level causal narrative (3-5 sentences)",
  "links": [
    {{
      "name": "descriptive link name",
      "from_stage": "stage A",
      "to_stage": "stage B",
      "mechanism": "how A leads to B",
      "assumptions": ["key assumption"],
      "failure_modes": ["what could go wrong"]
    }}
  ],
  "shared_assumptions": ["assumptions spanning the whole chain"],
  "investment_roles": {{
    "INV-XXX": ["link name 1", "link name 2"]
  }}
}}"""


async def extract_bow_causal_models(
    state: CausalState, config: RunnableConfig = None
) -> dict:
    """Phase 3.1a — BOW-level causal model extraction (Orphan 4).

    For each scope with >1 investment: synthesises a shared theory of change
    from per-investment causal models produced by forecast_consequences.
    Mirrors OLD causal_pipeline._phase31a_extract_bow_model.
    """
    import re as _re_bow
    scope_outputs = list(state.get("scope_outputs") or [])
    program_context = state.get("program_context") or {}
    model = state.get("research_model") or ""
    errors: list[str] = []

    # Build a lookup: inv_id → causal_model for fast access
    cm_by_inv: dict[str, dict] = {}
    for s in scope_outputs:
        cm = s.get("causal_model") or {}
        for iid in (s.get("inv_ids") or [s.get("inv_id", "")]):
            if iid and cm:
                cm_by_inv[iid] = cm

    bow_causal_models: dict[str, dict] = {}

    for scope in scope_outputs:
        scope_id = scope.get("scope_id", "")
        inv_ids = scope.get("inv_ids") or [scope.get("inv_id", "")]
        label = scope.get("label", scope_id)

        if len(inv_ids) <= 1:
            continue  # single-investment scope — skip BOW model

        inv_lines: list[str] = []
        for inv_id in sorted(inv_ids):
            cm = cm_by_inv.get(inv_id) or {}
            toc = (cm.get("theory_of_change") or "")[:200]
            inv_lines.append(f"- {inv_id}: {toc or '(no narrative)'}")

        prompt = _BOW_CAUSAL_PROMPT.format(
            n_inv=len(inv_ids),
            bow_label=label,
            program=program_context.get("program", ""),
            program_toc=(program_context.get("theory_of_change") or "")[:400],
            inv_text="\n".join(inv_lines),
        )
        try:
            raw = await acall_llm(prompt, system_msg=_BOW_CAUSAL_SYSTEM, model=model, config=config)
            raw_str = raw if isinstance(raw, str) else str(raw)
            m = _re_bow.search(r"\{.*\}", raw_str, _re_bow.DOTALL)
            parsed: dict = json.loads(m.group(0)) if m else {}
            bow_causal_models[scope_id] = parsed
            logger.info(
                "[%s] BOW causal model: %d links, %d investments mapped",
                scope_id, len(parsed.get("links", [])), len(parsed.get("investment_roles", {})),
            )
        except Exception as exc:
            logger.warning("[%s] BOW causal model extraction failed: %s", scope_id, exc)
            errors.append(f"extract_bow_causal_models:{scope_id}:{exc}")

    result: dict[str, Any] = {"bow_causal_models": bow_causal_models}
    if errors:
        result["errors"] = errors
    return result


def frame_investment_links(state: CausalState) -> dict:
    """Phase 3.1b — Investment link mapping and framing (Orphan 5).

    Case A: investment has causal links → tag with BOW mappings via word overlap.
    Case B: investment has 0 links (extraction failed) → create investment-specific
            framed links from BOW model so the investigation fan-out has claims.

    Mirrors OLD causal_pipeline Phase 3.1b.
    """
    bow_causal_models = state.get("bow_causal_models") or {}
    scope_outputs = list(state.get("scope_outputs") or [])

    for scope in scope_outputs:
        scope_id = scope.get("scope_id", "")
        bow_model = bow_causal_models.get(scope_id)
        if not bow_model:
            continue

        investment_roles: dict[str, list[str]] = bow_model.get("investment_roles", {})
        bow_links: list[dict] = bow_model.get("links", [])
        label = scope.get("label", scope_id)
        causal_model: dict = scope.get("causal_model") or {}
        inv_links: list[dict] = causal_model.get("links", [])

        inv_ids = scope.get("inv_ids") or [scope.get("inv_id", "")]

        for inv_id in inv_ids:
            inv_bow_roles = investment_roles.get(inv_id, [])
            if not inv_bow_roles:
                continue

            if inv_links:
                # Case A: tag existing links with BOW link name via word overlap
                bow_link_mappings: dict[str, list[str]] = causal_model.get("bow_link_mappings", {})
                for inv_link in inv_links:
                    inv_name = inv_link.get("name", "")
                    inv_words = set(inv_name.lower().split())
                    best, best_score = None, 0
                    for role in inv_bow_roles:
                        overlap = len(inv_words & set(role.lower().split()))
                        if overlap > best_score:
                            best_score, best = overlap, role
                    if best:
                        bow_link_mappings.setdefault(best, []).append(inv_name)
                causal_model["bow_link_mappings"] = bow_link_mappings
            else:
                # Case B: create investment-specific links from BOW model
                new_links: list[dict] = []
                bow_link_mappings = {}
                for role in inv_bow_roles:
                    bl = next((lk for lk in bow_links if lk.get("name") == role), None)
                    if bl:
                        inv_name = f"{inv_id} contribution to: {role}"[:80]
                        new_links.append({
                            "name": inv_name,
                            "from_stage": bl.get("from_stage", ""),
                            "to_stage": bl.get("to_stage", ""),
                            "mechanism": (
                                f"Within '{label}', {inv_id} advances '{role}' by: "
                                f"{bl.get('mechanism', '')[:200]}"
                            ),
                            "assumptions": bl.get("assumptions", []),
                            "failure_modes": bl.get("failure_modes", []),
                        })
                        bow_link_mappings.setdefault(role, []).append(inv_name)
                if new_links:
                    causal_model["links"] = new_links
                    causal_model["bow_link_mappings"] = bow_link_mappings
                    logger.info(
                        "[%s/%s] Created %d framed links from BOW model",
                        scope_id, inv_id, len(new_links),
                    )
            scope["causal_model"] = causal_model

    return {"scope_outputs": scope_outputs}


async def investigate_orphan_links(
    state: CausalState, config: RunnableConfig = None
) -> dict:
    """Phase 3.4b — Orphan BOW link investigation (Orphan 6).

    Identifies BOW causal links with no funding investment and investigates
    each gap: portfolio search + web search + LLM criticality assessment.
    Appends orphan findings to link_assessments so synthesize_findings
    can incorporate strategic gaps into the scope narrative.

    Mirrors OLD causal_pipeline Phase 3.4b.
    """
    import re as _re_orph
    bow_causal_models = state.get("bow_causal_models") or {}
    scope_outputs = state.get("scope_outputs") or []
    model = state.get("research_model") or ""
    errors: list[str] = []
    orphan_assessments: list[dict] = []

    for scope in scope_outputs:
        scope_id = scope.get("scope_id", "")
        bow_model = bow_causal_models.get(scope_id)
        if not bow_model:
            continue

        investment_roles: dict[str, list[str]] = bow_model.get("investment_roles", {})
        bow_links: list[dict] = bow_model.get("links", [])
        label = scope.get("label", scope_id)

        # Identify unfunded links
        funded_links: set[str] = set()
        for roles in investment_roles.values():
            funded_links.update(roles)
        orphan_links = [lk for lk in bow_links if lk.get("name", "") not in funded_links]

        if not orphan_links:
            continue

        logger.info("[%s] %d unfunded BOW links — investigating", scope_id, len(orphan_links))

        for orphan_link in orphan_links:
            link_name = orphan_link.get("name", "")
            mechanism = orphan_link.get("mechanism", "")[:200]
            search_query = f"{label}: {link_name} — {mechanism}"

            # Portfolio search
            portfolio_evidence = ""
            try:
                from src.tools.investigation_tools import search_portfolio
                p_result = await search_portfolio.ainvoke(
                    {"query": search_query, "top_k": 10},
                    config=config,
                )
                portfolio_evidence = str(p_result)[:2000]
            except Exception:
                pass

            # Web search
            web_evidence = ""
            try:
                from src.tools.investigation_tools import search_web
                w_result = await search_web.ainvoke(
                    {"query": search_query, "rationale": "gap evidence for unfunded BOW link"},
                    config=config,
                )
                if w_result and not w_result.startswith("[web search not configured"):
                    web_evidence = str(w_result)[:2000]
            except Exception:
                pass

            # LLM criticality assessment
            gap_prompt = (
                f"The BOW '{label}' has a theory of change with this UNFUNDED link:\n\n"
                f"LINK: {link_name}\n"
                f"From: {orphan_link.get('from_stage', '?')} → To: {orphan_link.get('to_stage', '?')}\n"
                f"Mechanism: {mechanism}\n"
                f"Assumptions: {orphan_link.get('assumptions', [])}\n\n"
                f"No current Foundation investment funds this stage.\n\n"
                f"PORTFOLIO EVIDENCE:\n{portfolio_evidence or '(none found)'}\n\n"
                f"EXTERNAL EVIDENCE:\n{web_evidence or '(none found)'}\n\n"
                "Assess this gap. Return JSON:\n"
                '{"criticality": "critical|high|moderate|low", '
                '"gap_acknowledged": true/false, '
                '"external_coverage": "who else covers this or empty", '
                '"risk_assessment": "what happens if unfunded", '
                '"recommendation": "what leadership should consider"}'
            )
            try:
                raw = await acall_llm(gap_prompt, model=model, config=config)
                raw_str = raw if isinstance(raw, str) else str(raw)
                m = _re_orph.search(r"\{.*\}", raw_str, _re_orph.DOTALL)
                assessment: dict = json.loads(m.group(0)) if m else {}
                orphan_assessments.append({
                    "scope_id": scope_id,
                    "link_id": f"ORPHAN:{scope_id}:{link_name[:30]}",
                    "link_name": link_name,
                    "from_stage": orphan_link.get("from_stage", ""),
                    "to_stage": orphan_link.get("to_stage", ""),
                    "mechanism": mechanism,
                    "criticality": assessment.get("criticality", "unknown"),
                    "gap_acknowledged": assessment.get("gap_acknowledged", False),
                    "external_coverage": assessment.get("external_coverage", ""),
                    "risk_assessment": assessment.get("risk_assessment", ""),
                    "recommendation": assessment.get("recommendation", ""),
                    "status": "not_answerable",
                    "confidence": "low",
                    "answer": assessment.get("risk_assessment", ""),
                    "evidence_refs": [],
                    "is_orphan_link": True,
                })
                logger.info(
                    "[%s] Orphan link '%s': criticality=%s",
                    scope_id, link_name[:40], assessment.get("criticality", "?"),
                )
            except Exception as exc:
                logger.warning("[%s] Orphan link assessment failed for %s: %s", scope_id, link_name[:40], exc)
                errors.append(f"investigate_orphan_links:{scope_id}:{link_name[:30]}:{exc}")

    result: dict[str, Any] = {"link_assessments": orphan_assessments}
    if errors:
        result["errors"] = errors
    return result


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
                "scope_label": scope.get("label", scope_id),
                "claim": link,
                "model": model,
                "result": None,
                "ingested_dir": state.get("ingested_dir", ""),
                "collection_name": state.get("collection_name", ""),
            }))

    return sends or "collect_link_assessments"


async def investigate_link(state: LinkInvestigationState, config: RunnableConfig = None) -> dict:
    from src.core.tool_tracing import flush_trace_buffer, init_trace_buffer

    link_id = state["link_id"]
    scope_id = state["scope_id"]

    init_trace_buffer()

    # Inject search_backend + per-link context into config so investigation tools
    # can resolve search_backend from config["configurable"].  Without this,
    # search_investment / search_portfolio / search_bow all return
    # "(search_backend not configured)" and accumulated_chunks stays empty.
    config = _inject_search_config(config, state, inv_id=state["inv_id"], bow_id=state.get("bow_id", ""))

    try:
        from src.core import investigation
        result = await investigation.run_investigation(
            link_id=link_id,
            inv_id=state["inv_id"],
            bow_id=state["bow_id"],
            scope_id=scope_id,
            claim=state.get("claim", {}),
            model=state.get("model", ""),
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

    # Collect annotated excerpts — copy, then strip from link_assessments to
    # bound state size.  Keep top-10 by tier priority.
    # Correct sort keys match what investigation.py writes: "tier1_primary" < "tier2_secondary" < "tier3_context".
    raw_excerpts: list[dict] = list(result_dict.get("annotated_excerpts", []) or [])
    result_dict.pop("annotated_excerpts", None)   # remove from link_assessments to keep state small
    _tier_rank = {"tier1_primary": 0, "tier2_secondary": 1, "tier3_context": 2}
    raw_excerpts.sort(key=lambda x: _tier_rank.get(str(x.get("credibility_tier", "")), 3))

    # Enrich with scope_label and link_name — context available only at this node level
    scope_label: str = state.get("scope_label", scope_id)
    link_name: str = (state.get("claim") or {}).get("name", link_id)
    annotated_excerpts = [
        {**ex, "scope_label": scope_label, "link_name": link_name}
        for ex in raw_excerpts[:10]
    ]

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
                "ingested_dir": state.get("ingested_dir", ""),
                "collection_name": state.get("collection_name", ""),
            }))

    return sends or "collect_science_results"


async def investigate_science_assumption(state: ScienceAssumptionState, config: RunnableConfig = None) -> dict:
    from src.core.tool_tracing import flush_trace_buffer, init_trace_buffer

    assumption_id = state["assumption_id"]
    scope_id = state["scope_id"]

    init_trace_buffer()

    # Same search_backend injection as investigate_link — science tools also read
    # search_backend from config["configurable"].
    config = _inject_search_config(config, state, inv_id=state["inv_id"], bow_id=state.get("bow_id", ""))

    try:
        from src.core import science_investigator
        result = await science_investigator.investigate_science_question(
            assumption_id=assumption_id,
            inv_id=state["inv_id"],
            bow_id=state["bow_id"],
            scope_id=scope_id,
            question=state.get("question", ""),
            config=config,
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


def _coerce_necessity_payload(
    parsed: dict,
    known_fields: frozenset,
    scope_id: str,
    *,
    web_searches_performed: int,
    all_web_sources: list,
    path_label: str,
    fallback_confidence_floor: bool,
) -> dict:
    """Coerce raw LLM JSON into a necessity assessment dict.

    Enforces two citation safeguards (mirrors old repo Phase 3.7):
    1. CITATION RULE — remove unsourced named-program data when the LLM made
       a differentiation/redundancy claim without providing source URLs.
    2. NO BACKFILL FROM WEB-ENGINE CITES unless the LLM also provided sources.
    """
    coerced = {k: v for k, v in parsed.items() if k in known_fields}

    llm_sources = list(coerced.get("sources") or [])
    has_llm_sources = bool(llm_sources)
    redundancy_text = coerced.get("redundancy_finding") or ""
    has_redundancy_claim = redundancy_text.strip().lower() not in ("", "none identified")
    differentiation = coerced.get("differentiation", "")
    has_substantive_diff = differentiation in ("high", "low")

    if not has_llm_sources and (has_substantive_diff or has_redundancy_claim):
        logger.warning(
            "[%s] necessity_check %s: LLM made differentiation/redundancy claim "
            "without citing sources — clearing unverified named programs and "
            "downgrading confidence",
            scope_id,
            path_label,
        )
        coerced["confidence"] = "low"
        if has_redundancy_claim:
            coerced["redundancy_finding"] = (
                "(unverified: LLM named overlapping programs without source "
                "citations; original text suppressed at type boundary)"
            )
        coerced["substitutes"] = []

    if has_llm_sources:
        existing = set(llm_sources)
        for s in all_web_sources:
            if s and s not in existing:
                existing.add(s)
        coerced["sources"] = list(existing)
    else:
        coerced["sources"] = []

    subs = coerced.get("substitutes", []) or []
    coerced["substitutes"] = [str(s) for s in subs if s] if isinstance(subs, list) else []

    coerced["scope_id"] = scope_id
    coerced["web_searches_performed"] = web_searches_performed
    if fallback_confidence_floor:
        coerced["confidence"] = "low"

    return coerced


_NECESSITY_KNOWN_FIELDS: frozenset = frozenset({
    "differentiation", "differentiation_rationale", "redundancy_finding",
    "counterfactual_loss", "marginal_contribution", "substitutes",
    "portfolio_relationship", "failure_mode_independence", "confidence", "sources",
})


async def necessity_check(state: CausalState, config: RunnableConfig = None) -> dict:
    """Phase 3.7 necessity check — 2-turn DISCOVER + VERIFY web-search loop.

    For each scope:
      Turn 1 (DISCOVER): web search for external programs on the same problem.
      Turn 2 (VERIFY): compare investment against candidates → structured JSON.
      Fallback: BoW-context-only assessment when Turn 1 finds no candidates or
        Turn 2 fails. Always emits a necessity_assessment; confidence='low' on
        fallback.

    Stores assessment as JSON string on scope["necessity_assessment"].
    Ported from old repo _phase37_necessity_check (causal_pipeline.py §3.7).
    """
    from src.tools.investigation_tools import search_web as _search_web

    scope_outputs = list(state.get("scope_outputs", []))
    model = state.get("synthesis_model", "")
    errors: list[str] = []

    for scope in scope_outputs:
        scope_id = scope.get("scope_id", "")
        link_assessments = scope.get("link_assessments", [])
        if not link_assessments:
            scope["necessity_assessment"] = ""
            continue

        # ── Build investment context ───────────────────────────────
        inv_id = scope.get("inv_id", "") or scope_id
        facts = scope.get("investment_facts") or {}
        org = facts.get("org", "")
        title = facts.get("title", "")
        approved = facts.get("approved_amount", 0)

        # Theory-of-change link name summary (mirrors old toc_summary)
        toc_summary = "; ".join(
            la.get("link_name", la.get("name", ""))
            for la in link_assessments[:5]
            if la.get("link_name") or la.get("name")
        )
        if not toc_summary:
            causal_model = scope.get("causal_model") or {}
            toc_links = causal_model.get("links", []) if isinstance(causal_model, dict) else []
            toc_summary = "; ".join(lk.get("name", "") for lk in toc_links[:5] if lk.get("name"))

        # BoW context block (mirrors old bc_summary)
        bow_context = scope.get("bow_context") or {}
        bc_parts: list[str] = []
        if isinstance(bow_context, dict):
            if bow_context.get("field_landscape"):
                bc_parts.append(f"Field: {bow_context['field_landscape'][:600]}")
            if bow_context.get("comparable_programs"):
                bc_parts.append(
                    "Already-identified comparable programs:\n  - "
                    + "\n  - ".join(str(p) for p in bow_context["comparable_programs"][:5])
                )
            if bow_context.get("benchmarks"):
                bm_lines = [str(b) for b in bow_context["benchmarks"][:3]]
                bc_parts.append("Benchmarks (top 3):\n" + "\n".join(bm_lines))
        bc_summary = "\n\n".join(bc_parts)

        narrative_excerpt = (scope.get("synthesis") or "")[:800]

        web_searches_performed = 0
        all_sources: list[str] = []

        # ── Turn 1: DISCOVER candidates (web search) ──────────────
        candidates: list[dict] = []
        try:
            discover_query = (
                f"{org or inv_id} {title} comparable programs external funder "
                f"alternative global health 2023 2024 2025"
            )
            discover_rationale = (
                f"Identify 2-5 external programs working on the same problem as {inv_id} "
                "at comparable maturity for necessity/differentiation assessment."
            )
            search_result = await _search_web(discover_query, discover_rationale, config)
            web_searches_performed += 1

            discover_prompt = (
                f"Investment: {inv_id} ({org}): {title} [${approved:,.0f}]\n"
                f"Theory of change (link summary): {toc_summary}\n\n"
                f"Key narrative excerpt:\n{narrative_excerpt}\n\n"
                f"BoW context:\n{bc_summary or '(no BoW context available)'}\n\n"
                f"Web search results:\n{search_result[:3000]}\n\n"
                "Based on the web search results above, identify 2-5 EXTERNAL programs "
                "working on the same problem at comparable maturity. Required: "
                "named program, funder/host, source URL."
            )
            discover_raw = await acall_llm(
                discover_prompt,
                system_msg=NECESSITY_DISCOVER_SYSTEM,
                model=model,
                config=config,
                max_tokens=1000,
            )
            parsed_d = safe_parse_json(discover_raw)
            if not (isinstance(parsed_d, dict) and parsed_d.get("_parse_failed")):
                raw_c = parsed_d.get("candidates", []) if isinstance(parsed_d, dict) else []
                if isinstance(raw_c, list):
                    candidates = [
                        c for c in raw_c
                        if isinstance(c, dict) and c.get("name") and c.get("source")
                    ]
        except Exception as exc:
            logger.warning("[%s] necessity_check Turn 1 (discover) failed: %s", scope_id, str(exc)[:120])

        # ── Turn 2: VERIFY differentiation (conditional on Turn 1) ──
        verify_parsed: dict = {}
        if candidates:
            try:
                cand_block = "\n".join(
                    f"- {c.get('name', '?')} ({c.get('funder', '?')}, "
                    f"{c.get('maturity_stage', '?')}): {c.get('source', '')}"
                    for c in candidates[:5]
                )
                verify_prompt = (
                    f"Investment: {inv_id} ({org}): {title} [${approved:,.0f}]\n"
                    f"Theory of change: {toc_summary}\n\n"
                    f"Narrative excerpt:\n{narrative_excerpt}\n\n"
                    f"BoW context:\n{bc_summary}\n\n"
                    f"Candidate external programs:\n{cand_block}\n\n"
                    "Compare the investment against each candidate. Produce "
                    "a NecessityAssessment per the system prompt. Cite source "
                    "URLs for every named program."
                )
                verify_raw = await acall_llm(
                    verify_prompt,
                    system_msg=NECESSITY_VERIFY_SYSTEM,
                    model=model,
                    config=config,
                    max_tokens=2000,
                )
                parsed_v = safe_parse_json(verify_raw)
                if isinstance(parsed_v, dict) and not parsed_v.get("_parse_failed"):
                    verify_parsed = parsed_v
            except Exception as exc:
                logger.warning("[%s] necessity_check Turn 2 (verify) failed: %s", scope_id, str(exc)[:120])

        # ── Build assessment from Turn 2 result ───────────────────
        if verify_parsed:
            coerced = _coerce_necessity_payload(
                verify_parsed,
                _NECESSITY_KNOWN_FIELDS,
                scope_id,
                web_searches_performed=web_searches_performed,
                all_web_sources=all_sources,
                path_label="Turn 2",
                fallback_confidence_floor=False,
            )
            scope["necessity_assessment"] = json.dumps(coerced)
            logger.info(
                "[%s] necessity_check complete (Turn 2): differentiation=%s, confidence=%s",
                scope_id,
                coerced.get("differentiation", "?"),
                coerced.get("confidence", "?"),
            )
            continue

        # ── Fallback: BoW-context-only assessment ─────────────────
        try:
            fallback_prompt = (
                f"Investment: {inv_id} ({org}): {title} [${approved:,.0f}]\n"
                f"Theory of change: {toc_summary}\n\n"
                f"Narrative excerpt:\n{narrative_excerpt}\n\n"
                f"BoW context (already-gathered field landscape and "
                f"comparable programs — NO additional web search):\n"
                f"{bc_summary or '(no BoW context available)'}\n\n"
                "Produce a NecessityAssessment using ONLY the BoW context "
                "above. Set confidence='low' since no external verification "
                "search was completed. Do NOT fabricate program names not "
                "already in the BoW context."
            )
            fb_raw = await acall_llm(
                fallback_prompt,
                system_msg=NECESSITY_VERIFY_SYSTEM,
                model=model,
                config=config,
                max_tokens=2000,
            )
            fb_parsed = (
                safe_parse_json(fb_raw) if isinstance(fb_raw, str)
                else (fb_raw if isinstance(fb_raw, dict) else {})
            )
            if fb_parsed and not fb_parsed.get("_parse_failed"):
                coerced = _coerce_necessity_payload(
                    fb_parsed,
                    _NECESSITY_KNOWN_FIELDS,
                    scope_id,
                    web_searches_performed=web_searches_performed,
                    all_web_sources=all_sources,
                    path_label="fallback",
                    fallback_confidence_floor=True,
                )
                scope["necessity_assessment"] = json.dumps(coerced)
                logger.info(
                    "[%s] necessity_check complete (fallback): differentiation=%s",
                    scope_id,
                    coerced.get("differentiation", "?"),
                )
                continue
        except Exception as exc:
            logger.warning("[%s] necessity_check fallback failed: %s", scope_id, str(exc)[:120])
            errors.append(f"necessity:{scope_id}:{exc}")

        # Last resort
        scope["necessity_assessment"] = json.dumps({
            "scope_id": scope_id,
            "confidence": "low",
            "differentiation_rationale": "(necessity check failed; defaulting to empty)",
            "web_searches_performed": web_searches_performed,
            "sources": [],
        })

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

# Nodes — Group 1.25 (BOW causal model + link framing + orphan investigation — Orphans 4–6)
_builder.add_node("extract_bow_causal_models", extract_bow_causal_models)
_builder.add_node("frame_investment_links", frame_investment_links)
_builder.add_node("investigate_orphan_links", investigate_orphan_links)

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

# Group 1.25 chain: collect_evidence_packs → extract_bow_causal_models → frame_investment_links
# (Orphans 4 & 5): extract BOW-level theory-of-change, then tag/create per-investment links
_builder.add_edge("collect_evidence_packs", "extract_bow_causal_models")
_builder.add_edge("extract_bow_causal_models", "frame_investment_links")

# Group 1.5 fan-out: frame_investment_links → dispatch_bow_enrichment → [enrich_bow_context_worker × N]
#                                                                      → collect_bow_enrichment → forecast_consequences
_builder.add_conditional_edges(
    "frame_investment_links",
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

# Orphan 6: investigate_orphan_links runs after collect_link_assessments before synthesis.
# Appends orphan gap findings to link_assessments so synthesize_findings sees them.
_builder.add_edge("collect_link_assessments", "investigate_orphan_links")

# ── Parallel: synthesize_findings chain ∥ dispatch_science_investigations ────
# Both read from investigate_orphan_links (which passes through to the same state):
#   - synthesize_findings reads link_assessments (now includes orphan findings)
#   - dispatch_science_investigations reads causal_model.assumptions
# They write to completely different state keys — safe to run in parallel.
_builder.add_edge("investigate_orphan_links", "synthesize_findings")
_builder.add_edge("investigate_orphan_links", "dispatch_science_investigations")

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
