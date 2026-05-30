"""Analyze subgraph — 13 nodes (ARCHITECTURE.md §3).

load_catalog                    ← Phase 0: load JSON artifacts when running standalone
orientation                     ← Phase 1: LLM portfolio overview → orientation_summary
compute_scopes                  ← Phase 2: scope computation + optional chunk-count filtering
build_timelines                 ← Phase 2.5: pure-local ScopeTimeline construction (no LLM)
dispatch_timeline_narratives    ← Phase 2.6 router: fan-out per scope
generate_scope_narrative        ← Phase 2.6 worker: rich multi-paragraph narrative per investment
collect_timeline_narratives     ← Phase 2.7 reducer: merge narratives back into scope_timelines
run_causal_pipeline             ← Phase 3: invokes causal subgraph; maps AnalyzeState ↔ CausalState
dispatch_investment_reports     ← Phase 3.5 router: fan-out per scope for AI-vs-team verdict
build_investment_report_worker  ← Phase 3.5 worker: blind AI verdict + divergence per scope
collect_investment_reports      ← Phase 3.5 join: trivial convergence node
dispatch_scope_sections         ← Phase 3.6 router: fan-out per scope for section draft
synthesize_scope_section_worker ← Phase 3.6 worker: ranked deviations + LLM narrative per scope
collect_scope_sections          ← Phase 3.6 join: trivial convergence node
cross_cutting_analysis          ← Phase 5: cross-scope pattern detection → clusters
quality_assessment              ← Phase 6a: coverage %, grade, confidence_map
assemble_report                 ← Phase 6b: calls src.core.report_assembler; writes final_report.md

Field alignment verified (AnalyzeState → CausalState):
  scopes           : Optional[list[dict]] → list[dict]            ✓ same name
  scope_timelines  : Optional[dict]       → dict                  ✓ same name
  research_model   : str                  → str                   ✓ same name
  synthesis_model  : str                  → str                   ✓ same name
  evidence_packs   : Annotated[...]       → Annotated[...]        ✓ same name
  link_assessments : Annotated[...]       → Annotated[...]        ✓ same name
  science_results  : Annotated[...]       → Annotated[...]        ✓ same name
  scope_decisions  : Annotated[...]       → Annotated[...]        ✓ same name
  scope_outputs    : Optional[list[dict]] ← list[dict]            ✓ same name (reverse)
  errors           : Annotated[list[str]] ← Annotated[list[str]]  ✓ same name (reverse)
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re as _re
from pathlib import Path
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from src.config import (
    ASSEMBLY_MAX_RETRIES,
    DEFAULT_ANALYSIS_MODEL as _DEFAULT_ANALYSIS_MODEL,
    DEFAULT_SYNTHESIS_MODEL as _DEFAULT_MODEL,
    NARRATOR_CALL_BUDGET,
    ORIENTATION_MAX_TOKENS,
)
from src.core.llm_utils import acall_llm
from src.prompts.tool_prompts import ORIENTATION_SYSTEM
from src.graph.state import (
    AnalyzeState,
    InvestmentNarrativeState,
    InvestmentReportWorkerState,
    ScopeSynthesisState,
    SectionDraftWorkerState,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# load_catalog — Phase 0 (no-op if data already in state)
# ---------------------------------------------------------------------------


async def load_catalog(state: AnalyzeState, config: RunnableConfig = None) -> dict:
    """Load catalog data from ingested_dir when fields are absent (standalone Studio use)."""
    if state.get("investment_scoring"):
        return {}
    ingested_dir = state.get("ingested_dir")
    if not ingested_dir:
        return {}
    base = Path(ingested_dir)

    def _read(path: Path):
        with open(path) as f:
            return json.load(f)

    try:
        return {
            # asyncio-APPROVED-1: to_thread wraps blocking JSON file read
            "doc_list": await asyncio.to_thread(_read, base / "doc_list.json"),
            # asyncio-APPROVED-1: to_thread wraps blocking JSON file read
            "investment_scoring": await asyncio.to_thread(_read, base / "investment_scoring.json"),
            # asyncio-APPROVED-1: to_thread wraps blocking JSON file read
            "bow_investment_map": await asyncio.to_thread(_read, base / "bow_investment_map.json"),
            # asyncio-APPROVED-1: to_thread wraps blocking JSON file read
            "investment_intelligence": await asyncio.to_thread(_read, base / "investment_intelligence.json"),
            "chunks_json_path": str(base / "embedding_index" / "chunks.json"),
            "pages_dir": str(base / "pages"),
        }
    except Exception as exc:
        logger.error("load_catalog failed: %s", exc)
        return {"errors": [f"load_catalog:{exc}"]}


# ---------------------------------------------------------------------------
# orientation — Phase 1
# ---------------------------------------------------------------------------


async def orientation(state: AnalyzeState, config: RunnableConfig = None) -> dict:
    """Build portfolio orientation summary.

    Uses investment metadata from JSON state fields. If a search backend is
    available via config, also retrieves top document summaries from the
    embedding index to match the OLD repo's _phase1_orient depth.
    """
    investment_scoring = state.get("investment_scoring") or {}
    bow_investment_map = state.get("bow_investment_map") or {}
    investment_intelligence = state.get("investment_intelligence") or {}
    doc_list = state.get("doc_list") or []
    focus = state.get("focus") or ""
    # F-002: orientation uses research_model (analysis-class = gpt-5.5), not synthesis_model
    model = state.get("research_model") or _DEFAULT_ANALYSIS_MODEL
    focus_str = f"\nFocus area: {focus}" if focus else ""

    # --- Bundles of Work ---
    bow_lines: list[str] = []
    for bow_id, bow_data in bow_investment_map.items():
        if isinstance(bow_data, dict):
            label = bow_data.get("bow_label") or bow_id
            inv_ids = bow_data.get("inv_ids") or []
        else:
            label = bow_id
            inv_ids = list(bow_data) if bow_data else []
        bow_lines.append(f"  - {label} ({bow_id}): {len(inv_ids)} investment(s)")

    # --- Investments ---
    inv_lines: list[str] = []
    for inv_id, inv in investment_scoring.items():
        if not isinstance(inv, dict):
            continue
        title = inv.get("title", inv_id)
        grantee = inv.get("grantee", "")
        geography = inv.get("geography", "")
        allocation = inv.get("allocation_usd", 0)
        maturity = inv.get("maturity_stage", "")
        start = inv.get("start_year", "")
        end = inv.get("end_year", "")
        intel = investment_intelligence.get(inv_id) or {}
        toc = (intel.get("theory_of_change") or "")[:300]
        timeline = (intel.get("timeline_summary") or "")[:200]

        line = f"  - **{title}** ({inv_id})"
        if grantee:
            line += f", grantee: {grantee}"
        if geography:
            line += f", geography: {geography}"
        if allocation:
            line += f", allocation: ${allocation:,.0f}"
        if maturity:
            line += f", stage: {maturity}"
        if start or end:
            line += f", period: {start}–{end}"
        if toc:
            line += f"\n    Theory of change: {toc}"
        if timeline:
            line += f"\n    Timeline: {timeline}"
        inv_lines.append(line)

    # --- Doc types ---
    doc_type_counts: dict[str, int] = {}
    for doc in doc_list:
        dt = doc.get("doc_type", "unknown") if isinstance(doc, dict) else "unknown"
        doc_type_counts[dt] = doc_type_counts.get(dt, 0) + 1
    doc_summary = (
        ", ".join(f"{n} {t}" for t, n in doc_type_counts.items())
        or f"{len(doc_list)} documents"
    )

    # --- F-020: Strategy docs — date-sorted descending (matches OLD _phase1_orient) ---
    strategy_docs = [
        d for d in doc_list
        if isinstance(d, dict) and d.get("collection") == "strategy"
    ]
    strategy_docs.sort(
        key=lambda d: d.get("doc_date") or d.get("date") or "0000-00-00",
        reverse=True,
    )
    strategy_section = ""
    if strategy_docs:
        strat_lines: list[str] = []
        for d in strategy_docs[:15]:
            fname = d.get("filename", d.get("file_id", ""))
            dt_val = d.get("doc_date") or d.get("date") or ""
            doc_type_val = d.get("doc_type", "")
            summary = (d.get("summary") or "")[:200]
            line = f"  [{dt_val}] {fname} ({doc_type_val})"
            if summary:
                line += f": {summary}"
            strat_lines.append(line)
        strategy_section = (
            f"\n\n## Strategy Documents (most recent first, top {len(strat_lines)})\n"
            + "\n".join(strat_lines)
        )

    # --- Optional: top document summaries from embedding index (Priority 3c) ---
    doc_excerpts_section = ""
    backend = ((config or {}).get("configurable") or {}).get("search_backend")
    _orient_traces: list[dict] = []
    if backend:
        try:
            results = await backend.search(
                "program theory of change goals outcomes investments",
                top_k=30,
            )
            excerpts: list[str] = []
            char_budget = 0
            for r in results or []:
                text = getattr(r, "text", "") or ""
                if char_budget + len(text) > 120_000:
                    break
                excerpts.append(text[:2000])
                char_budget += len(text)
            if excerpts:
                doc_excerpts_section = (
                    "\n\n## Representative Document Excerpts (top-30 search results)\n"
                    + "\n\n---\n\n".join(excerpts[:20])
                )
            # F-003: emit trace so orientation search appears in collection_search_traces
            _orient_traces.append({
                "node": "orientation",
                "query": "program theory of change goals outcomes investments",
                "top_k": 30,
                "result_count": len(results or []),
            })
        except Exception as exc:
            logger.debug("orientation: embedding search failed (non-fatal): %s", exc)

    prompt = (
        f"You are producing a structured portfolio orientation for an investment review.{focus_str}\n\n"
        f"## Bundles of Work ({len(bow_investment_map)})\n"
        + "\n".join(bow_lines or ["  (none)"])
        + f"\n\n## Investments ({len(investment_scoring)})\n"
        + "\n".join(inv_lines or ["  (none)"])
        + f"\n\n## Available Documents\n  {doc_summary}"
        + strategy_section
        + doc_excerpts_section
        + "\n\nRespond with a JSON object containing exactly these keys:\n"
        "  theory_of_change: string — 2-3 sentences on the portfolio's overall theory of change\n"
        "  major_bets: list of objects — the 3-5 most significant strategic bets, each with:\n"
        "    {bet: string, bows: list[string], amount_approx: string}\n"
        "  stated_priorities: list of strings — explicit review priorities or focus areas\n"
        "  key_timelines: list of objects — critical upcoming milestones, each with:\n"
        "    {milestone: string, target_date: string, status: 'on_track'|'at_risk'|'missed'}\n"
        "  portfolio_health_signals: list of strings — early signals about portfolio health\n"
        "  bow_summaries: dict mapping bow_id to a 1-sentence summary of that bundle\n"
        "  initial_concerns: list of strings — preliminary concerns for investigator focus\n\n"
        "Use the data above. Respond ONLY with the JSON object."
    )

    import re as _orient_re
    try:
        # F-021: use ORIENTATION_SYSTEM (with SAFETY_PREAMBLE) + enforce max_tokens=8000
        raw = await acall_llm(
            prompt,
            system_msg=ORIENTATION_SYSTEM,
            model=model,
            max_tokens=ORIENTATION_MAX_TOKENS,
            config=config,
        )
        raw_str = raw if isinstance(raw, str) else str(raw)
        m = _orient_re.search(r"\{.*\}", raw_str, _orient_re.DOTALL)
        program_context: dict = {}
        if m:
            try:
                program_context = json.loads(m.group(0))
            except Exception:
                pass
        if not program_context:
            # Fallback: wrap free text in minimal structure
            program_context = {
                "theory_of_change": raw_str[:500],
                "major_bets": [],
                "stated_priorities": [],
                "key_timelines": [],
                "portfolio_health_signals": [],
                "bow_summaries": {},
                "initial_concerns": [],
            }
        out: dict[str, Any] = {"program_context": program_context}
        if _orient_traces:
            out["collection_search_traces"] = _orient_traces
        return out
    except Exception as exc:
        logger.error("orientation failed: %s", exc)
        return {"errors": [f"orientation:{exc}"], "program_context": {}}


# ---------------------------------------------------------------------------
# compute_scopes — Phase 2
# ---------------------------------------------------------------------------


async def compute_scopes(state: AnalyzeState, config: RunnableConfig = None) -> dict:
    """Group BOWs by sub-topic and apply size gates to produce scopes.

    Size gates (match OLD repo):
      - Skip:  group chunk count < 200  (phantom — not indexed)
      - Split: group chunk count > 12K  OR  group has > 8 investments → one scope per BOW
      - Keep:  otherwise — one scope for the whole group

    Each scope carries: scope_id, label, bow_ids, inv_ids, chunk_count.
    Phantom investments (no indexed documents) are filtered out before grouping.
    """
    investment_scoring = state.get("investment_scoring") or {}
    bow_investment_map = state.get("bow_investment_map") or {}
    focus_bows = state.get("focus_bows")

    MIN_CHUNKS = int(os.environ.get("NQPR_MIN_CHUNKS_PER_BOW", "200"))  # skip threshold — matches OLD repo
    SPLIT_CHUNKS = 12_000  # split-by-BOW threshold
    SPLIT_INVS = 8       # split-by-investment-count threshold

    # --- Query chunk counts per BOW and phantom-investment filter ---
    bow_chunk_counts: dict[str, int] = {}
    doc_bearing_inv_ids: set[str] | None = None  # F-023: investments with ≥1 indexed chunk
    backend = ((config or {}).get("configurable") or {}).get("search_backend")
    if backend and hasattr(backend, "count_by_bow_id"):
        try:
            bow_chunk_counts = await backend.count_by_bow_id() or {}
        except Exception as exc:
            logger.debug("compute_scopes: count_by_bow_id failed (non-fatal): %s", exc)
    if backend and hasattr(backend, "distinct_inv_ids"):
        try:
            doc_bearing_inv_ids = set(await backend.distinct_inv_ids() or [])
        except Exception as exc:
            logger.debug("compute_scopes: distinct_inv_ids failed (non-fatal): %s", exc)

    # --- Filter out phantom BOWs (< MIN_CHUNKS when chunk data available) ---
    active_bow_ids: set[str] = set()
    for bow_id in bow_investment_map:
        if focus_bows and bow_id not in focus_bows:
            continue
        if bow_chunk_counts and bow_chunk_counts.get(bow_id, 0) < MIN_CHUNKS:
            logger.debug(
                "compute_scopes: skipping phantom BOW %s (%d chunks)",
                bow_id, bow_chunk_counts.get(bow_id, 0),
            )
            continue
        active_bow_ids.add(bow_id)

    # --- Group BOWs by sub-topic using bow_investment_map metadata ---
    # F-022: use "{topic} > {sub_topic}" grouping key to match old _compute_thread_scopes.
    # BOWs with empty topic/sub_topic produce key " > " and merge into one scope,
    # matching the old code where BOWNode defaults (topic="", sub_topic="") caused the
    # same behaviour.
    bow_meta: dict[str, dict] = {}  # bow_id → {topic, sub_topic}
    subtopic_to_bows: dict[str, list[str]] = {}
    for bow_id in active_bow_ids:
        bow_data = bow_investment_map.get(bow_id) or {}
        topic     = bow_data.get("topic",     "") if isinstance(bow_data, dict) else ""
        sub_topic = bow_data.get("sub_topic", "") if isinstance(bow_data, dict) else ""
        bow_meta[bow_id] = {"topic": topic, "sub_topic": sub_topic}
        group_key = f"{topic} > {sub_topic}"
        subtopic_to_bows.setdefault(group_key, []).append(bow_id)

    # --- Build scopes from groups ---
    scopes: list[dict] = []
    scope_idx = 0

    for group_key, bow_ids_in_group in subtopic_to_bows.items():
        # Collect all inv_ids and total chunk count for this group
        group_inv_ids: list[str] = []
        group_chunk_count = 0
        for bid in bow_ids_in_group:
            bow_data = bow_investment_map.get(bid) or {}
            inv_ids_for_bow = (
                bow_data.get("inv_ids", [])
                if isinstance(bow_data, dict)
                else list(bow_data) if bow_data else []
            )
            group_inv_ids.extend(inv_ids_for_bow)
            group_chunk_count += bow_chunk_counts.get(bid, 0)

        # De-duplicate inv_ids (an inv can belong to multiple BOWs)
        # F-023: also exclude phantom investments (in scoring but not indexed)
        seen_invs: set[str] = set()
        unique_inv_ids: list[str] = []
        for iid in group_inv_ids:
            if (
                iid not in seen_invs
                and iid in investment_scoring
                and (doc_bearing_inv_ids is None or iid in doc_bearing_inv_ids)
            ):
                seen_invs.add(iid)
                unique_inv_ids.append(iid)

        if not unique_inv_ids:
            continue

        # Size gate — skip tiny groups
        if bow_chunk_counts and group_chunk_count < MIN_CHUNKS:
            logger.debug(
                "compute_scopes: skipping group '%s' (%d chunks, %d invs)",
                group_key, group_chunk_count, len(unique_inv_ids),
            )
            continue

        # Size gate — split large groups
        needs_split = (
            (bow_chunk_counts and group_chunk_count > SPLIT_CHUNKS)
            or len(unique_inv_ids) > SPLIT_INVS
        )

        if needs_split and len(bow_ids_in_group) > 1:
            # One scope per BOW in the group
            for bid in bow_ids_in_group:
                bow_data = bow_investment_map.get(bid) or {}
                bow_inv_ids = (
                    bow_data.get("inv_ids", [])
                    if isinstance(bow_data, dict)
                    else list(bow_data) if bow_data else []
                )
                # Keep only inv_ids present in investment_scoring and indexed (F-023)
                bow_inv_ids = [i for i in bow_inv_ids if i in investment_scoring]
                if doc_bearing_inv_ids is not None:
                    bow_inv_ids = [i for i in bow_inv_ids if i in doc_bearing_inv_ids]
                if not bow_inv_ids:
                    continue
                bow_label = (
                    bow_data.get("bow_label") or bow_data.get("label") or bid
                    if isinstance(bow_data, dict) else bid
                )
                bow_chunks = bow_chunk_counts.get(bid, 0)
                # Primary inv_id for backward compat with nodes that read scope["inv_id"]
                primary_inv = bow_inv_ids[0]
                scopes.append({
                    "scope_id": f"scope_{scope_idx:04d}",
                    "inv_id": primary_inv,
                    "inv_ids": bow_inv_ids,
                    "bow_ids": [bid],
                    "label": f"{bow_label} — {bid}",
                    "chunk_count": bow_chunks,
                    "topic":     bow_meta.get(bid, {}).get("topic",     ""),
                    "sub_topic": bow_meta.get(bid, {}).get("sub_topic", ""),
                })
                scope_idx += 1
        else:
            # Keep whole group as single scope
            # F-022: label mirrors old code: take sub_topic part of key; fall back to BOW IDs.
            first_meta  = bow_meta.get(bow_ids_in_group[0], {})
            group_label = group_key.split(" > ")[-1] or ", ".join(bow_ids_in_group[:2])
            primary_inv = unique_inv_ids[0]
            scopes.append({
                "scope_id":  f"scope_{scope_idx:04d}",
                "inv_id":    primary_inv,
                "inv_ids":   unique_inv_ids,
                "bow_ids":   bow_ids_in_group,
                "label":     group_label,
                "chunk_count": group_chunk_count,
                "topic":     first_meta.get("topic",     ""),
                "sub_topic": first_meta.get("sub_topic", ""),
            })
            scope_idx += 1

    logger.info("compute_scopes: %d scopes from %d active BOWs", len(scopes), len(active_bow_ids))
    return {"scopes": scopes}


# ---------------------------------------------------------------------------
# build_timelines — Phase 2.5 (pure-local, no LLM)
# ---------------------------------------------------------------------------


async def build_timelines(state: AnalyzeState, config: RunnableConfig = None) -> dict:
    """Build rich ScopeTimeline objects for every scope. No LLM calls.

    Reads: scopes, doc_list, investment_scoring, investment_intelligence, pages_dir
    Writes: scope_timelines {scope_id → ScopeTimeline.to_dict()}

    Checks ingested_dir for a cached narrative file to pre-populate narratives
    (load_narratives returns False if stale/absent — build_timeline_narratives_async
    will then be called per-scope in generate_scope_narrative).
    """
    scopes = state.get("scopes") or []
    if not scopes:
        return {"scope_timelines": {}}

    doc_list = state.get("doc_list") or []
    scoring = state.get("investment_scoring") or {}
    intelligence = state.get("investment_intelligence") or {}
    pages_dir_str = state.get("pages_dir") or (
        ((config or {}).get("configurable") or {}).get("pages_dir", "")
    )
    pages_dir = Path(pages_dir_str) if pages_dir_str else None

    from src.core.investment_timeline import ScopeTimeline, build_scope_timeline

    scope_timelines: dict[str, ScopeTimeline] = {}

    def _build_all() -> dict[str, ScopeTimeline]:
        result: dict[str, ScopeTimeline] = {}
        for scope in scopes:
            sid = scope.get("scope_id", "")
            try:
                st = build_scope_timeline(
                    scope=scope,
                    doc_list=doc_list,
                    scoring=scoring,
                    pages_dir=pages_dir,
                    investment_intelligence=intelligence,
                )
                result[sid] = st
            except Exception as exc:
                logger.error("build_timelines: scope %s failed: %s", sid, exc)
        return result

    # asyncio-APPROVED-1: to_thread wraps blocking synchronous timeline construction
    scope_timelines = await asyncio.to_thread(_build_all)
    scope_timelines_dict = {sid: st.to_dict() for sid, st in scope_timelines.items()}

    # ── F-025: SHA256 narrative cache — cross-run reuse ───────────────────────
    ingested_dir = state.get("ingested_dir") or ""
    if ingested_dir:
        import hashlib as _hashlib
        cache_path = Path(ingested_dir) / "timeline_narratives.json"

        def _compute_hash(tl_dict: dict) -> str:
            # Hash structural data only (exclude existing narrative strings)
            structural = {
                sid: {k: v for k, v in tl.items() if k not in ("narrative", "investments")}
                for sid, tl in tl_dict.items()
            }
            return _hashlib.sha256(
                json.dumps(structural, sort_keys=True, default=str).encode()
            ).hexdigest()[:16]

        scope_hash = _compute_hash(scope_timelines_dict)

        def _load_cache() -> dict:
            if not cache_path.exists():
                return {}
            try:
                return json.loads(cache_path.read_text(encoding="utf-8"))
            except Exception:
                return {}

        # asyncio-APPROVED-1: to_thread wraps blocking file read
        cached = await asyncio.to_thread(_load_cache)
        if cached.get("hash") == scope_hash:
            cached_narratives: dict = cached.get("narratives", {})
            for sid, narrative in cached_narratives.items():
                if sid in scope_timelines_dict and narrative:
                    scope_timelines_dict[sid]["narrative"] = narrative
            logger.info(
                "build_timelines: SHA256 cache hit — %d narratives pre-loaded",
                len(cached_narratives),
            )
        # Store hash in state so collect_timeline_narratives can write it back
        return {"scope_timelines": scope_timelines_dict, "timeline_cache_hash": scope_hash}

    return {"scope_timelines": scope_timelines_dict}


# ---------------------------------------------------------------------------
# dispatch_investment_narratives — Phase 2.6 router (fan-out per investment)
# ---------------------------------------------------------------------------


async def dispatch_investment_narratives(state: AnalyzeState):
    """Fan-out: one Send per investment for per-investment LLM narrative generation.

    Two-level fan-out: investment narratives first, then scope synthesis.
    Resume-safe: skips investments already in investment_narrative_results.
    """
    scopes = state.get("scopes") or []
    if not scopes:
        return "collect_investment_narratives"

    scope_timelines = state.get("scope_timelines") or {}
    investment_scoring = state.get("investment_scoring") or {}
    model = state.get("synthesis_model") or _DEFAULT_MODEL

    already_done: set[str] = {
        f"{r['scope_id']}:{r['inv_id']}"
        for r in (state.get("investment_narrative_results") or [])
        if r.get("scope_id") and r.get("inv_id")
    }

    sends: list[Send] = []
    for scope in scopes:
        scope_id = scope.get("scope_id", "")
        inv_id = scope.get("inv_id", "")
        scope_tl_dict = scope_timelines.get(scope_id) or {}
        scope_label = scope_tl_dict.get("label", inv_id)
        investments = scope_tl_dict.get("investments") or []

        if investments:
            for inv_d in investments:
                iid = inv_d.get("inv_id", inv_id)
                if f"{scope_id}:{iid}" in already_done:
                    continue
                sends.append(Send("generate_investment_narrative", {
                    "scope_id": scope_id,
                    "scope_label": scope_label,
                    "inv_id": iid,
                    "inv_data": inv_d,
                    "model": model,
                }))
        else:
            # Fallback: no ScopeTimeline investments — send minimal financial dict
            if f"{scope_id}:{inv_id}" in already_done:
                continue
            inv_scoring = investment_scoring.get(inv_id) or {}
            sends.append(Send("generate_investment_narrative", {
                "scope_id": scope_id,
                "scope_label": scope_label,
                "inv_id": inv_id,
                "inv_data": {
                    "inv_id": inv_id,
                    "start": inv_scoring.get("start", "") if isinstance(inv_scoring, dict) else "",
                    "end": inv_scoring.get("end", "") if isinstance(inv_scoring, dict) else "",
                    "status": inv_scoring.get("status", "") if isinstance(inv_scoring, dict) else "",
                    "approved_amount": inv_scoring.get("approved_amount", 0) if isinstance(inv_scoring, dict) else 0,
                    "paid_amount": inv_scoring.get("paid_amount", 0) if isinstance(inv_scoring, dict) else 0,
                },
                "model": model,
            }))

    return sends or "collect_investment_narratives"


# ---------------------------------------------------------------------------
# generate_investment_narrative — Phase 2.6 per-investment worker
# ---------------------------------------------------------------------------


async def generate_investment_narrative(state: InvestmentNarrativeState) -> dict:
    """Generate a narrative for a single investment. One LLM call, no asyncio.gather."""
    from dataclasses import fields as dc_fields
    from src.core.investment_timeline import (
        DocumentEvent, InvestmentTimeline,
        _build_single_investment_narrative,
    )

    scope_id = state["scope_id"]
    scope_label = state.get("scope_label", "")
    inv_id = state["inv_id"]
    inv_data = state["inv_data"]
    model = state["model"]

    narrative = ""

    if inv_data.get("documents") is not None:
        # Rich path: reconstruct InvestmentTimeline and call the core narrative builder
        inv_fields = {f.name for f in dc_fields(InvestmentTimeline)}
        doc_fields = {f.name for f in dc_fields(DocumentEvent)}
        docs = [
            DocumentEvent(**{k: v for k, v in doc.items() if k in doc_fields})
            for doc in (inv_data.get("documents") or [])
        ]
        inv_obj = InvestmentTimeline(
            **{k: v for k, v in inv_data.items() if k in inv_fields and k != "documents"}
        )
        inv_obj.documents = docs
        try:
            await _build_single_investment_narrative(inv_obj, scope_id, scope_label, model)
            narrative = inv_obj.narrative or ""
        except Exception as exc:
            logger.warning(
                "generate_investment_narrative (rich) failed %s/%s: %s", scope_id, inv_id, exc
            )
    else:
        # Fallback path: minimal financial summary
        prompt = (
            f"Investment {inv_id} timeline narrative: "
            f"start={inv_data.get('start', '')}, end={inv_data.get('end', '')}, "
            f"status={inv_data.get('status', '')}, "
            f"approved=${float(inv_data.get('approved_amount', 0) or 0) / 1e6:.1f}M, "
            f"paid=${float(inv_data.get('paid_amount', 0) or 0) / 1e6:.1f}M. "
            "Summarise the financial timeline and execution status in 2-3 sentences."
        )
        try:
            narrative = str(await acall_llm(prompt, model=model))
        except Exception as exc:
            logger.warning(
                "generate_investment_narrative (fallback) failed %s/%s: %s", scope_id, inv_id, exc
            )

    return {"investment_narrative_results": [{
        "scope_id": scope_id,
        "inv_id": inv_id,
        "narrative": narrative,
        "inv_data": inv_data,
    }]}


# ---------------------------------------------------------------------------
# collect_investment_narratives — Phase 2.6 sync point
# ---------------------------------------------------------------------------


async def collect_investment_narratives(state: AnalyzeState) -> dict:
    """Sync point: all investment_narrative_results accumulated; dispatch_scope_syntheses fans out next."""
    n = len(state.get("investment_narrative_results") or [])
    logger.info("collect_investment_narratives: %d investment narratives ready", n)
    return {}


# ---------------------------------------------------------------------------
# dispatch_scope_syntheses — Phase 2.6 second router (fan-out per scope)
# ---------------------------------------------------------------------------


async def dispatch_scope_syntheses(state: AnalyzeState):
    """Fan-out: one Send per scope to generate the scope-level synthesis from investment narratives."""
    inv_results = state.get("investment_narrative_results") or []
    if not inv_results:
        return "collect_timeline_narratives"

    scope_timelines = state.get("scope_timelines") or {}
    model = state.get("synthesis_model") or _DEFAULT_MODEL

    already_done: set[str] = {
        r.get("scope_id", "")
        for r in (state.get("timeline_narrative_results") or [])
        if r.get("scope_id")
    }

    by_scope: dict[str, list[dict]] = {}
    for r in inv_results:
        sid = r.get("scope_id", "")
        if sid:
            by_scope.setdefault(sid, []).append(r)

    sends: list[Send] = []
    for scope_id, narratives in by_scope.items():
        if scope_id in already_done:
            continue
        scope_tl_dict = scope_timelines.get(scope_id) or {}
        sends.append(Send("generate_scope_synthesis", {
            "scope_id": scope_id,
            "scope_label": scope_tl_dict.get("label", ""),
            "investment_narratives": narratives,
            "scope_timeline_dict": scope_tl_dict,
            "model": model,
        }))

    return sends or "collect_timeline_narratives"


# ---------------------------------------------------------------------------
# generate_scope_synthesis — Phase 2.6 per-scope synthesis worker
# ---------------------------------------------------------------------------


async def generate_scope_synthesis(state: ScopeSynthesisState) -> dict:
    """Synthesise investment narratives into a scope-level paragraph. One LLM call."""
    from src.core.investment_timeline import SCOPE_SYNTHESIS_SYSTEM

    scope_id = state["scope_id"]
    scope_label = state.get("scope_label", "")
    inv_narratives = state.get("investment_narratives") or []
    scope_tl_dict = state.get("scope_timeline_dict") or {}
    model = state["model"]

    narrated = [n for n in inv_narratives if n.get("narrative")]

    # Rebuild updated scope_timeline_dict with per-investment narratives merged in
    scope_tl_out = dict(scope_tl_dict)
    scope_tl_out["scope_id"] = scope_id
    inv_narrative_map = {n["inv_id"]: n["narrative"] for n in narrated}
    scope_tl_out["investments"] = [
        {**inv_d, "narrative": inv_narrative_map.get(inv_d.get("inv_id", ""), inv_d.get("narrative", ""))}
        for inv_d in (scope_tl_dict.get("investments") or [])
    ]

    if not narrated:
        logger.warning("[%s] No investment narratives available; skipping scope synthesis", scope_id)
        return {"timeline_narrative_results": [scope_tl_out]}

    try:
        summaries = "\n\n---\n\n".join(
            f"**{n['inv_id']}:**\n{n['narrative'][:2000]}" for n in narrated
        )
        scope_raw = await acall_llm(
            f"# Scope: {scope_label}\n# {len(narrated)} investments\n\n"
            f"Per-investment narratives:\n\n{summaries}\n\nWrite the scope-level synthesis.",
            system_msg=SCOPE_SYNTHESIS_SYSTEM,
            model=model,
        )
        if len(str(scope_raw)) > 50:
            scope_tl_out["narrative"] = str(scope_raw)
    except Exception as exc:
        logger.warning("[%s] Scope synthesis failed: %s", scope_id, str(exc)[:120])

    return {"timeline_narrative_results": [scope_tl_out]}


# ---------------------------------------------------------------------------
# collect_timeline_narratives — Phase 2.7 (reducer → scope_timelines dict)
# ---------------------------------------------------------------------------


async def collect_timeline_narratives(state: AnalyzeState, config: RunnableConfig = None) -> dict:
    """Merge narrative results back into scope_timelines.

    F-025: After merging, writes {ingested_dir}/timeline_narratives.json with
    SHA256 hash so future runs can skip re-generating unchanged narratives.
    """
    results = state.get("timeline_narrative_results") or []
    existing = state.get("scope_timelines") or {}

    updated = dict(existing)
    for r in results:
        sid = r.get("scope_id")
        if not sid:
            continue
        if sid in updated:
            existing_tl = dict(updated[sid])
            existing_tl["narrative"] = r.get("narrative", existing_tl.get("narrative", ""))
            new_invs = r.get("investments") or []
            old_invs = {inv.get("inv_id", ""): inv for inv in existing_tl.get("investments", [])}
            for new_inv in new_invs:
                iid = new_inv.get("inv_id", "")
                if iid in old_invs:
                    old_invs[iid]["narrative"] = new_inv.get("narrative", old_invs[iid].get("narrative", ""))
                    old_invs[iid]["key_events"] = new_inv.get("key_events", old_invs[iid].get("key_events", []))
            existing_tl["investments"] = list(old_invs.values())
            updated[sid] = existing_tl
        else:
            updated[sid] = r

    # ── F-025: Write SHA256 cache so next run can skip unchanged narratives ──
    ingested_dir = state.get("ingested_dir") or ""
    scope_hash = state.get("timeline_cache_hash") or ""
    if ingested_dir and scope_hash:
        cache_path = Path(ingested_dir) / "timeline_narratives.json"
        cache_data = {
            "hash": scope_hash,
            "narratives": {sid: tl.get("narrative", "") for sid, tl in updated.items()},
        }
        def _write_cache():
            try:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(
                    json.dumps(cache_data, indent=2, default=str), encoding="utf-8"
                )
            except Exception as exc:
                logger.warning("timeline_narratives.json cache write failed: %s", exc)
        # asyncio-APPROVED-1: to_thread wraps blocking file write
        await asyncio.to_thread(_write_cache)

    return {"scope_timelines": updated}


# ---------------------------------------------------------------------------
# BOW enrichment web search helper (F-059)
# ---------------------------------------------------------------------------


async def _bow_web_search(query: str) -> str:
    """OpenAI Responses API web search for BOW context enrichment.

    Matches the OLD repo's _call_with_web_search pattern (thread_sub_agent.py):
    uses gpt-5.4 + {"type": "web_search"} tool via the OpenAI Responses API.
    Returns the response text as a string; enrich_bow_context_worker handles
    both str and list[dict] return types.

    Falls back to empty string on any error so BOW enrichment degrades
    gracefully rather than failing the scope.
    """
    import asyncio
    try:
        from openai import OpenAI
        client = OpenAI()

        def _sync_search(q: str) -> str:
            response = client.responses.create(
                model="gpt-5.4",
                input=[{"role": "user", "content": q}],
                tools=[{"type": "web_search"}],
            )
            return response.output_text or ""

        # asyncio-APPROVED-1: to_thread wraps blocking OpenAI Responses API call
        return await asyncio.to_thread(_sync_search, query)
    except Exception as exc:
        logger.debug("BOW web search failed for %r: %s", query[:60], str(exc)[:80])
        return ""


# ---------------------------------------------------------------------------
# run_causal_pipeline — Phase 3
# ---------------------------------------------------------------------------


async def run_causal_pipeline(
    state: AnalyzeState, config: RunnableConfig = None
) -> dict:
    from src.graph.subgraphs.causal import causal_graph  # lazy — allows test mocking

    # ── AnalyzeState → CausalState projection ──────────────────────────────
    causal_input: dict = {
        "scopes": state.get("scopes") or [],
        "scope_timelines": state.get("scope_timelines") or {},
        "research_model": state.get("research_model") or _DEFAULT_MODEL,
        "synthesis_model": state.get("synthesis_model") or _DEFAULT_MODEL,
        "program_context": state.get("program_context"),   # F-001: forward portfolio context to causal nodes
        # Pass through so causal worker _get_tools() can build a search backend when
        # search_backend is absent from config["configurable"] (e.g. langgraph dev).
        "ingested_dir": state.get("ingested_dir", ""),
        "collection_name": state.get("collection_name", ""),
        # reducer fields must be initialised to [] (AGENTS.md §4)
        "bow_causal_models": {},     # Orphan 4: extract_bow_causal_models writes this
        "evidence_packs": [],
        "link_assessments": [],
        "science_results": [],
        "scope_decisions": [],
        "scope_outputs": [],
        "all_excerpts": [],
        # trace reducer fields — must be [] not None (AGENTS.md §4)
        "asta_traces": [],
        "slr_traces": [],
        "lbd_traces": [],
        "deep_web_traces": [],
        "edison_traces": [],
        "web_search_traces": [],
        "compute_traces": [],
        "collection_search_traces": [],
        "investigation_traces": [],
        "errors": [],
    }

    # ── Augment config with web_search_fn for BOW enrichment (F-059) ──────────
    # enrich_bow_context_worker reads config["configurable"]["web_search_fn"].
    # If not already provided by the caller, wire in the OpenAI Responses API
    # web search (matching OLD repo thread_sub_agent._call_with_web_search) so
    # Phase 3.1.5 produces real benchmarks / comparable_programs instead of
    # silently returning _empty_bow on every scope.
    causal_config = dict(config or {})
    causal_configurable = dict(causal_config.get("configurable") or {})
    if "web_search_fn" not in causal_configurable:
        causal_configurable["web_search_fn"] = _bow_web_search
    causal_config["configurable"] = causal_configurable

    try:
        causal_result: dict = await causal_graph.ainvoke(causal_input, causal_config)
    except Exception as exc:
        logger.error("run_causal_pipeline failed: %s", exc)
        return {"errors": [f"run_causal_pipeline:{exc}"]}

    # ── CausalState → AnalyzeState projection ──────────────────────────────
    # evidence_packs / link_assessments / science_results / scope_decisions are NOT
    # forwarded: their data is embedded in scope_outputs by the collect_* nodes.
    # Omitting them keeps AnalyzeState.* accumulator fields at [] (their initialized
    # value), preventing unnecessary memory growth in subsequent analyze nodes.
    return {
        "scope_outputs": causal_result.get("scope_outputs") or [],
        "all_excerpts": causal_result.get("all_excerpts") or [],
        "asta_traces": causal_result.get("asta_traces") or [],
        "slr_traces": causal_result.get("slr_traces") or [],
        "lbd_traces": causal_result.get("lbd_traces") or [],
        "deep_web_traces": causal_result.get("deep_web_traces") or [],
        "edison_traces": causal_result.get("edison_traces") or [],
        "web_search_traces": causal_result.get("web_search_traces") or [],
        "compute_traces": causal_result.get("compute_traces") or [],
        "collection_search_traces": causal_result.get("collection_search_traces") or [],
        "investigation_traces": causal_result.get("investigation_traces") or [],
        "errors": causal_result.get("errors") or [],
    }


# ---------------------------------------------------------------------------
# build_investment_reports — Phase 3.5+ (per-investment AI-vs-team synthesis)
# ---------------------------------------------------------------------------


async def dispatch_investment_reports(state: AnalyzeState) -> list[Send] | str:
    """Fan-out router: one Send per scope for AI-vs-team investment report synthesis.

    Skips scopes that already have investment_report set (LangGraph checkpoint resume).
    Falls back to collect_investment_reports when all scopes are already done.
    """
    scope_outputs = state.get("scope_outputs") or []
    if not scope_outputs:
        return "collect_investment_reports"
    model = state.get("synthesis_model") or _DEFAULT_MODEL
    investment_scoring = state.get("investment_scoring") or {}
    already_done = {s.get("scope_id") for s in scope_outputs if "investment_report" in s}
    sends: list[Send] = []
    for scope in scope_outputs:
        scope_id = scope.get("scope_id", "")
        if scope_id in already_done:
            continue
        sends.append(Send("build_investment_report_worker", {
            "scope_id": scope_id,
            "scope": scope,
            "investment_scoring": investment_scoring,
            "model": model,
            "result": None,
        }))
    return sends or "collect_investment_reports"


async def build_investment_report_worker(
    state: InvestmentReportWorkerState, config: RunnableConfig = None
) -> dict:
    """Per-scope worker: blind AI verdict + team score divergence.

    Writes {"scope_outputs": [updated_scope]} — merge_scope_outputs handles the merge.
    """
    import re as _re

    scope = dict(state["scope"])
    scope_id = state["scope_id"]
    inv_id = scope.get("inv_id", "")
    investment_scoring = state["investment_scoring"]
    model = state["model"]
    link_assessments = scope.get("link_assessments") or []

    try:
        if not link_assessments:
            return {"scope_outputs": [scope]}

        statuses = [a.get("status", "") for a in link_assessments]
        confidence_levels = [a.get("confidence", "") for a in link_assessments]
        evidence_summary = "\n".join(
            f"  - Link {a.get('link_id', '?')}: {a.get('status', '?')} "
            f"(confidence={a.get('confidence', '?')}): {a.get('gap_description', '')[:150]}"
            for a in link_assessments[:10]
        )

        blind_prompt = (
            f"You are producing an independent investment assessment for {inv_id}.\n"
            "You have NOT been shown the program team's scores. "
            "Base your verdict solely on the evidence below.\n\n"
            f"Link assessment statuses: {statuses}\n"
            f"Confidence levels: {confidence_levels}\n\n"
            f"Evidence:\n{evidence_summary}\n\n"
            "Respond with a JSON object containing:\n"
            "  overall_status: one of [on_track, deviations_found, critical_risk, insufficient_evidence]\n"
            "  severity: one of [program_critical, pathway_altering, efficiency_reducing, acceptable]\n"
            "  ai_execution_verdict: your independent execution assessment (2-3 sentences)\n"
            "  ai_impact_verdict: your independent impact assessment (2-3 sentences)\n"
            "  key_risks: list of up to 5 key risk strings\n"
            "  key_strengths: list of up to 5 key strength strings\n"
            "  executive_summary: 2-3 sentence investment summary\n"
        )
        try:
            raw = await acall_llm(blind_prompt, model=model, config=config)
            raw_str = raw if isinstance(raw, str) else str(raw)
            m = _re.search(r"\{.*\}", raw_str, _re.DOTALL)
            blind_verdict: dict = json.loads(m.group(0)) if m else {}
        except Exception as exc:
            logger.warning("build_investment_report_worker blind verdict failed %s: %s", inv_id, exc)
            blind_verdict = {}

        team = investment_scoring.get(inv_id) or {}
        team_execution = team.get("execution_score", team.get("execution", "")) if isinstance(team, dict) else ""
        team_impact = team.get("impact_score", team.get("impact", "")) if isinstance(team, dict) else ""

        _severity_rank = {
            "program_critical": 3, "pathway_altering": 2,
            "efficiency_reducing": 1, "acceptable": 0,
            "critical_risk": 3, "deviations_found": 1, "on_track": 0,
        }
        _team_rank = {"low": 0, "medium": 1, "high": 2, "critical": 3}
        ai_rank = _severity_rank.get(blind_verdict.get("overall_status", ""), 0) + \
                  _severity_rank.get(blind_verdict.get("severity", ""), 0)
        team_rank = _team_rank.get(str(team_execution).lower(), -1)
        if team_rank < 0:
            divergence_severity = "unknown"
        elif abs(ai_rank - team_rank * 2) >= 4:
            divergence_severity = "program_critical"
        elif abs(ai_rank - team_rank * 2) >= 2:
            divergence_severity = "pathway_altering"
        elif abs(ai_rank - team_rank * 2) >= 1:
            divergence_severity = "efficiency_reducing"
        else:
            divergence_severity = "aligned"

        scope["investment_report"] = {
            **blind_verdict,
            "ai_execution_verdict": blind_verdict.get("ai_execution_verdict", ""),
            "ai_impact_verdict": blind_verdict.get("ai_impact_verdict", ""),
            "team_execution_score": team_execution,
            "team_impact_score": team_impact,
            "divergence_severity": divergence_severity,
        }
        scope["team_execution"] = team_execution
        scope["team_impact"] = team_impact
    except Exception as exc:
        logger.error("build_investment_report_worker failed %s: %s", scope_id, exc)
        return {"scope_outputs": [scope], "errors": [f"build_investment_report_worker:{scope_id}:{exc}"]}

    return {"scope_outputs": [scope]}


async def collect_investment_reports(state: AnalyzeState, config: RunnableConfig = None) -> dict:
    """Trivial join node — merge_scope_outputs already accumulated investment_report updates from workers."""
    return {}


# ---------------------------------------------------------------------------
# synthesize_scope_section — Phase 3.6 (SectionDraft with ranked deviations)
# ---------------------------------------------------------------------------


_SEVERITY_WEIGHT = {
    "program_critical": 4.0,
    "pathway_altering": 3.0,
    "efficiency_reducing": 1.5,
    "acceptable": 0.5,
    "aligned": 0.5,
    "unknown": 1.0,
}

# F-036: word budget per top-deviation severity (mirrors report_writer.WORD_BUDGETS)
_WORD_BUDGETS: dict[str, int] = {
    "program_critical": 1500,
    "pathway_altering": 1000,
    "efficiency_reducing": 500,
}

# F-030: 4-part essay format for section drafts (mirrors _SECTION_WRITER_SYSTEM in report_writer.py)
_SECTION_ESSAY_SYSTEM = """\
You are writing a section of an analytical portfolio review report for \
organisational leadership and programme teams.

For each finding, write in proper essay format:

1. **What we found.** (Topic sentence — the specific risk/gap/issue, naming \
investments, amounts, timelines. 1-2 sentences maximum.)

2. **Evidence.** (Specific documents and data. Cite sources using \
§-reference numbers from the Source Index, e.g. [§0001], [§0003].)

3. **Why this matters.** (Dollar amounts at risk, timeline consequences, \
downstream dependencies. No generic risk language.)

4. **The strongest counter-argument.** (Best case for why this is not a problem. \
Cite actual evidence for the counter-argument if it exists.)

5. **Assessment.** (Does the finding survive the steelman? State confidence \
level and WHY. Categorise as: factual/data error, reasoning gap, or \
strategic blind spot. State concrete action if needed.)

Write in prose, not bullet points. Use **bold** for subsection labels only. \
Do NOT use markdown headings. Every claim must cite at least one §-reference. \
Be specific — name investments, partners, amounts, dates.\
"""


def _count_refs(text: str) -> int:
    """Count distinct §NNNN citation references in text."""
    return len(set(_re.findall(r"§\d{4}", text)))


async def dispatch_scope_sections(state: AnalyzeState) -> list[Send] | str:
    """Fan-out router: one Send per scope for section draft synthesis.

    Skips scopes that already have section_draft set (LangGraph checkpoint resume).
    Falls back to collect_scope_sections when all scopes are already done.
    """
    scope_outputs = state.get("scope_outputs") or []
    if not scope_outputs:
        return "collect_scope_sections"
    model = state.get("synthesis_model") or _DEFAULT_MODEL
    already_done = {s.get("scope_id") for s in scope_outputs if "section_draft" in s}
    sends: list[Send] = []
    for scope in scope_outputs:
        scope_id = scope.get("scope_id", "")
        if scope_id in already_done:
            continue
        sends.append(Send("synthesize_scope_section_worker", {
            "scope_id": scope_id,
            "scope": scope,
            "model": model,
            "result": None,
        }))
    return sends or "collect_scope_sections"


async def synthesize_scope_section_worker(
    state: SectionDraftWorkerState, config: RunnableConfig = None
) -> dict:
    """Per-scope worker: produce a SectionDraft with ranked deviations.

    Writes {"scope_outputs": [updated_scope]} — merge_scope_outputs handles the merge.
    """
    scope = dict(state["scope"])
    scope_id = state["scope_id"]
    model = state["model"]
    inv_id = scope.get("inv_id", "")
    label = scope.get("label", scope_id)
    inv_report = scope.get("investment_report") or {}
    link_assessments = scope.get("link_assessments") or []
    investment_facts = scope.get("investment_facts") or {}

    try:
        raw_deviations: list[dict] = []
        for la in link_assessments:
            status = la.get("status", "")
            if status in ("deviation", "gap", "critical_risk", "deviations_found", "risk"):
                approved = investment_facts.get("approved_amount", 0)
                dollars_at_risk = float(approved) * 0.1 if approved else 0.0
                sev = inv_report.get("divergence_severity", "unknown")
                weight = _SEVERITY_WEIGHT.get(sev, 1.0)
                raw_deviations.append({
                    "link_id": la.get("link_id", ""),
                    "description": la.get("gap_description", la.get("claim_text", ""))[:200],
                    "dollars_at_risk": dollars_at_risk,
                    "severity": sev,
                    "score": dollars_at_risk * weight,
                })

        ranked_deviations = sorted(raw_deviations, key=lambda d: d["score"], reverse=True)

        # F-036: §-formatted source index from link_assessments
        source_index = [
            {
                "ref": f"§{i + 1:04d}",
                "link_id": la.get("link_id", f"link-{i}"),
                "description": la.get("gap_description", la.get("claim_text", ""))[:80],
                "status": la.get("status", ""),
            }
            for i, la in enumerate(link_assessments[:20])
        ]
        source_ref_text = "\n".join(
            f"  {s['ref']}: {s['link_id']} [{s['status']}] — {s['description']}"
            for s in source_index
        )
        # F-036: word budget keyed by top deviation severity
        top_severity = ranked_deviations[0]["severity"] if ranked_deviations else "pathway_altering"
        word_budget = _WORD_BUDGETS.get(top_severity, 300)

        prompt = (
            f"Produce a portfolio review section for investment {inv_id} (scope: {label}).\n\n"
            f"AI overall verdict: {inv_report.get('overall_status', 'unknown')}, "
            f"severity: {inv_report.get('severity', 'unknown')}\n"
            f"AI execution verdict: {inv_report.get('ai_execution_verdict', '')[:300]}\n\n"
            f"## Source Index (cite with §-refs in your text):\n{source_ref_text}\n\n"
            f"Write using the 4-part essay format from your instructions (~{word_budget} words total). "
            "Cover key findings, execution status, financial performance, and critical risks. "
            "Cite evidence using [§0001], [§0002] etc. from the Source Index above — NOT [1], [2]."
        )
        try:
            summary = await acall_llm(prompt, _SECTION_ESSAY_SYSTEM, model=model, config=config)
            summary_str = summary if isinstance(summary, str) else str(summary)
        except Exception as exc:
            logger.warning("synthesize_scope_section_worker LLM failed %s: %s", scope_id, exc)
            summary_str = inv_report.get("executive_summary", "")

        # F-036: retry once if output contains no §-refs and source index is non-empty
        if summary_str and source_index and _count_refs(summary_str) < 2:
            logger.info(
                "synthesize_scope_section_worker: no §-refs in output for %s, retrying",
                scope_id,
            )
            retry_prompt = (
                prompt
                + "\n\nYOUR PREVIOUS OUTPUT DID NOT CONTAIN §-REFERENCES. "
                "You MUST cite sources using §-format from the Source Index. "
                f"Available refs: {', '.join(s['ref'] for s in source_index[:10])}. "
                "Use [§0001], [§0002] etc. — NOT [1], [2]."
            )
            try:
                retry_raw = await acall_llm(retry_prompt, _SECTION_ESSAY_SYSTEM, model=model, config=config)
                summary_str = retry_raw if isinstance(retry_raw, str) else str(retry_raw)
            except Exception as exc:
                logger.warning(
                    "synthesize_scope_section_worker retry failed %s: %s", scope_id, exc
                )

        scope["section_draft"] = {
            "scope_id": scope_id,
            "heading": label,
            "summary": summary_str,
            "key_findings": inv_report.get("key_risks", []) + inv_report.get("key_strengths", []),
            "ranked_deviations": ranked_deviations[:10],
        }
    except Exception as exc:
        logger.error("synthesize_scope_section_worker failed %s: %s", scope_id, exc)
        return {"scope_outputs": [scope], "errors": [f"synthesize_scope_section_worker:{scope_id}:{exc}"]}

    return {"scope_outputs": [scope]}


async def collect_scope_sections(state: AnalyzeState, config: RunnableConfig = None) -> dict:
    """Trivial join node — merge_scope_outputs already accumulated section_draft updates from workers."""
    return {}


# ---------------------------------------------------------------------------
# cross_cutting_analysis — Phase 5
# ---------------------------------------------------------------------------


async def cross_cutting_analysis(state: AnalyzeState, config: RunnableConfig = None) -> dict:
    """Phase 4 — Cross-cutting analysis with pre-computed portfolio metrics.

    Pre-computes: total_approved_dollars, total_paid_dollars, at_risk_count,
    concentration_by_bow. Then calls LLM for structured essay + patterns.
    Returns typed CrossCuttingAnalysis dict in state.cross_cutting_analysis.
    Also populates state.clusters for backward compatibility.
    """
    scope_outputs = state.get("scope_outputs") or []
    model = state.get("synthesis_model") or _DEFAULT_MODEL

    if not scope_outputs:
        return {"cross_cutting_analysis": {}}

    import re as _re

    # ── F-026: Phase 4 pre-computation via OpenAI code_interpreter ────────────
    # Matches OLD repo's code_interpreter call (ANALYSIS_MODEL) for portfolio_metrics.
    # Falls back to pure-Python computation if code_interpreter is unavailable.
    portfolio_metrics: dict = {}

    async def _compute_metrics_code_interpreter() -> dict | None:
        """Use OpenAI code_interpreter to compute portfolio_metrics (Phase 5 pre-step)."""
        try:
            from openai import OpenAI
            from src.config import COMPUTE_MODEL as _ANALYSIS_MODEL
            client = OpenAI()

            scope_summary = json.dumps([
                {
                    "label": s.get("label"),
                    "approved": float((s.get("investment_facts") or {}).get("approved_amount", 0) or 0),
                    "paid": float((s.get("investment_facts") or {}).get("paid_amount", 0) or 0),
                    "severity": (s.get("investment_report") or {}).get("severity", ""),
                    "bow_ids": s.get("bow_ids") or [],
                }
                for s in scope_outputs
            ])

            code_prompt = (
                f"scope_outputs = {scope_summary}\n\n"
                "import json\n"
                "total_approved = sum(s['approved'] for s in scope_outputs)\n"
                "total_paid = sum(s['paid'] for s in scope_outputs)\n"
                "at_risk_count = sum(1 for s in scope_outputs if s['severity'] in ('program_critical','pathway_altering'))\n"
                "conc = {}\n"
                "for s in scope_outputs:\n"
                "    for bid in s['bow_ids']:\n"
                "        conc[bid] = conc.get(bid, 0) + s['approved']\n"
                "concentration_by_bow = {bid: round(v/max(total_approved,1)*100,1) for bid,v in conc.items()}\n"
                "result = {'total_approved_dollars': total_approved, 'total_paid_dollars': total_paid, "
                "'at_risk_count': at_risk_count, 'concentration_by_bow': concentration_by_bow, "
                f"'scope_count': {len(scope_outputs)}}}\n"
                "print(json.dumps(result))"
            )

            def _sync() -> str:
                response = client.responses.create(
                    model=_ANALYSIS_MODEL,
                    input=[{"role": "user", "content": code_prompt}],
                    tools=[{"type": "code_interpreter", "container": {"type": "auto"}}],
                )
                return response.output_text or ""

            # asyncio-APPROVED-1: to_thread wraps blocking OpenAI Responses API call
            output = await asyncio.to_thread(_sync)
            m = _re.search(r"\{[^{}]*total_approved_dollars[^{}]*\}", output, _re.DOTALL)
            if m:
                return json.loads(m.group(0))
        except Exception as exc:
            logger.debug("code_interpreter portfolio metrics failed (falling back): %s", exc)
        return None

    portfolio_metrics = await _compute_metrics_code_interpreter() or {}

    # Fallback: pure-Python metrics when code_interpreter unavailable
    if not portfolio_metrics:
        total_approved = sum(
            float((s.get("investment_facts") or {}).get("approved_amount", 0) or 0)
            for s in scope_outputs
        )
        total_paid = sum(
            float((s.get("investment_facts") or {}).get("paid_amount", 0) or 0)
            for s in scope_outputs
        )
        at_risk_count = sum(
            1 for s in scope_outputs
            if (s.get("investment_report") or {}).get("severity") in ("program_critical", "pathway_altering")
        )
        concentration: dict[str, float] = {}
        for s in scope_outputs:
            approved = float((s.get("investment_facts") or {}).get("approved_amount", 0) or 0)
            for bid in (s.get("bow_ids") or []):
                concentration[bid] = concentration.get(bid, 0) + approved
        portfolio_metrics = {
            "total_approved_dollars": round(total_approved, 2),
            "total_paid_dollars": round(total_paid, 2),
            "at_risk_count": at_risk_count,
            "concentration_by_bow": {
                bid: round(v / max(total_approved, 1) * 100, 1)
                for bid, v in concentration.items()
            },
            "scope_count": len(scope_outputs),
        }

    # ── Thread summaries for LLM ──────────────────────────────────────────────
    thread_summaries: list[str] = []
    for s in scope_outputs[:40]:
        inv_report = s.get("investment_report") or {}
        status = inv_report.get("overall_status", "unknown")
        sev = inv_report.get("severity", "")
        div = inv_report.get("divergence_severity", "")
        facts = s.get("investment_facts") or {}
        approved = facts.get("approved_amount", 0)
        thread_summaries.append(
            f"- {s.get('label', s.get('scope_id', '?'))}: "
            f"status={status} severity={sev} divergence={div} "
            f"approved=${float(approved or 0) / 1e6:.1f}M"
        )

    _ta = float(portfolio_metrics.get("total_approved_dollars", 0) or 0)
    _tp = float(portfolio_metrics.get("total_paid_dollars", 0) or 0)
    _arc = int(portfolio_metrics.get("at_risk_count", 0) or 0)
    _cbbow = portfolio_metrics.get("concentration_by_bow") or {}
    metrics_text = (
        f"Total approved: ${_ta / 1e6:.1f}M  "
        f"Total paid: ${_tp / 1e6:.1f}M  "
        f"At-risk investments: {_arc}  "
        f"Concentration (top BOWs): "
        + ", ".join(f"{bid}={pct}%" for bid, pct in
                    sorted(_cbbow.items(), key=lambda x: -x[1])[:5])
    )

    prompt = (
        f"You are analysing a portfolio of {len(scope_outputs)} investments for a quarterly review.\n\n"
        f"Portfolio metrics:\n{metrics_text}\n\n"
        "Investment scope summaries:\n"
        + "\n".join(thread_summaries)
        + "\n\nProvide a cross-cutting analysis. Respond with a JSON object containing:\n"
        "  patterns: list of 3-5 cross-portfolio pattern strings\n"
        "  contradictions: list of 2-4 contradiction or tension strings\n"
        "  shared_dependencies: list of 2-4 shared dependency strings\n"
        "  emergent_decisions: list of dicts [{title, description, urgency, affected_bow_ids}]\n"
        "  essay: 4-6 paragraph cross-cutting narrative essay covering the above\n"
    )

    try:
        raw = await acall_llm(prompt, model=model, config=config)
        raw_str = raw if isinstance(raw, str) else str(raw)
        m = _re.search(r"\{.*\}", raw_str, _re.DOTALL)
        llm_result: dict = json.loads(m.group(0)) if m else {}
    except Exception as exc:
        logger.error("cross_cutting_analysis LLM failed: %s", exc)
        return {
            "errors": [f"cross_cutting_analysis:{exc}"],
            "cross_cutting_analysis": {"portfolio_metrics": portfolio_metrics},
        }

    # ── F-027: Cluster identification — deterministic shortlist then LLM grouping ──
    # Mirrors OLD cluster_identification.identify_clusters: pick top findings by
    # severity/dollars, then one LLM call to group them into thematic clusters.
    clusters: list[dict] = []
    try:
        # Build a shortlist of high-severity deviations for cluster input
        deviation_shortlist: list[str] = []
        for s in scope_outputs:
            label = s.get("label", s.get("scope_id", "?"))
            for d in (s.get("section_draft") or {}).get("ranked_deviations", [])[:2]:
                desc = (d.get("description") or "")[:100]
                sev = d.get("severity", "")
                if desc and sev in ("program_critical", "pathway_altering"):
                    deviation_shortlist.append(f"- [{label}] {desc} ({sev})")
        if not deviation_shortlist:
            for s in scope_outputs[:8]:
                label = s.get("label", s.get("scope_id", "?"))
                for d in (s.get("section_draft") or {}).get("ranked_deviations", [])[:1]:
                    desc = (d.get("description") or "")[:100]
                    sev = d.get("severity", "")
                    if desc:
                        deviation_shortlist.append(f"- [{label}] {desc} ({sev})")

        if deviation_shortlist:
            cluster_prompt = (
                "Group the following investment deviations into 3–5 thematic clusters.\n\n"
                + "\n".join(deviation_shortlist[:20])
                + "\n\nReturn JSON: {\"clusters\": [{\"theme\": \"...\", \"description\": \"...\", "
                "\"scope_ids\": [\"...\"], \"risk_level\": \"high|medium|low\"}]}"
            )
            c_raw = await acall_llm(cluster_prompt, model=model, config=config)
            c_str = c_raw if isinstance(c_raw, str) else str(c_raw)
            import re as _re_cc
            c_m = _re_cc.search(r"\{.*\}", c_str, _re_cc.DOTALL)
            if c_m:
                clusters = json.loads(c_m.group(0)).get("clusters", [])
    except Exception as exc:
        logger.warning("cluster_identification failed (non-fatal): %s", exc)

    cross_cutting: dict[str, Any] = {
        "patterns": llm_result.get("patterns", []),
        "contradictions": llm_result.get("contradictions", []),
        "shared_dependencies": llm_result.get("shared_dependencies", []),
        "emergent_decisions": llm_result.get("emergent_decisions", []),
        "essay": llm_result.get("essay", raw_str),
        "portfolio_metrics": portfolio_metrics,
        "clusters": clusters,
    }

    return {"cross_cutting_analysis": cross_cutting, "clusters": clusters}


# ---------------------------------------------------------------------------
# quality_assessment — Phase 6a (documents coverage + grade)
# ---------------------------------------------------------------------------


async def quality_assessment(state: AnalyzeState, config: RunnableConfig = None) -> dict:
    """Compute coverage %, confidence grade, and quality metadata.

    Counts unique documents read from link_assessments (documents_read lists),
    compares against total doc_list length, and assigns an A/B/C/D grade.
    """
    doc_list = state.get("doc_list") or []
    scope_outputs = state.get("scope_outputs") or []
    scopes = state.get("scopes") or []
    # link_assessments are embedded in scope_outputs (no longer propagated at AnalyzeState level)
    link_assessments = [la for s in scope_outputs for la in (s.get("link_assessments") or [])]

    # documents_available = active investments with at least one indexed document
    # Use scope count (post-phantom-filter) as proxy when chunk counts aren't available
    active_inv_ids: set[str] = {
        iid
        for s in scopes
        for iid in (s.get("inv_ids") or [s.get("inv_id", "")])
        if iid
    }
    total_available = len(active_inv_ids) if active_inv_ids else len(doc_list)

    # Collect unique document IDs from annotated excerpts (primary) + investigation links
    docs_read: set[str] = set()
    for excerpt in (state.get("all_excerpts") or []):
        src = excerpt.get("source", "")
        if src:
            docs_read.add(src)
    for la in link_assessments:
        for doc_id in (la.get("documents_read") or []):
            docs_read.add(str(doc_id))
    # Also count from investigation_traces
    for trace in (state.get("investigation_traces") or []):
        for doc_id in (trace.get("documents_read") or []):
            docs_read.add(str(doc_id))

    total_read = len(docs_read)
    coverage_pct = total_read / max(total_available, 1)

    # Confidence map: infer from synthesis length and link assessment confidence
    confidence_map: dict[str, str] = {}
    low_confidence_scopes: list[str] = []
    for scope in scope_outputs:
        sid = scope.get("scope_id", "")
        las = scope.get("link_assessments") or []
        if not las:
            confidence_map[sid] = "low"
            low_confidence_scopes.append(sid)
            continue
        high_conf = sum(1 for la in las if la.get("confidence") in ("high", "very_high"))
        low_conf = sum(1 for la in las if la.get("confidence") in ("low", "very_low"))
        if high_conf > len(las) * 0.5:
            confidence_map[sid] = "high"
        elif low_conf > len(las) * 0.5:
            confidence_map[sid] = "low"
            low_confidence_scopes.append(sid)
        else:
            confidence_map[sid] = "medium"

    # A/B/C/D grade (matches OLD repo _phase5_quality thresholds)
    if coverage_pct > 0.5 and len(low_confidence_scopes) <= 2:
        grade = "A"
    elif coverage_pct > 0.3 and len(low_confidence_scopes) <= 4:
        grade = "B"
    elif coverage_pct > 0.15:
        grade = "C"
    else:
        grade = "D"

    quality_meta = {
        "documents_available": total_available,
        "documents_read": total_read,
        "coverage_pct": round(coverage_pct, 4),
        "grade": grade,
        "low_confidence_scopes": low_confidence_scopes,
    }
    logger.info(
        "quality_assessment: %d/%d docs read (%.1f%%), grade=%s",
        total_read, total_available, coverage_pct * 100, grade,
    )

    return {
        "coverage_pct": coverage_pct,
        "grade": grade,
        "confidence_map": confidence_map,
        "run_meta": quality_meta,
    }


# ---------------------------------------------------------------------------
# assemble_report — Phase 6b
# ---------------------------------------------------------------------------


async def assemble_report(state: AnalyzeState, config: RunnableConfig = None) -> dict:
    threads_dir = state.get("threads_dir") or ""
    errors: list[str] = []

    scope_outputs = list(state.get("scope_outputs") or [])
    all_excerpts = state.get("all_excerpts") or []

    # ── F-045: Build investment_sections from all_excerpts so read_evidence_pack works ──
    # Groups excerpt texts by inv_id and injects them into each scope's
    # "investment_sections" dict — the key that narration_tools.read_evidence_pack reads.
    inv_section_texts: dict[str, list[str]] = {}
    for ex in all_excerpts:
        iid = ex.get("inv_id", "")
        text = ex.get("text", ex.get("quote", ""))
        if iid and text:
            inv_section_texts.setdefault(iid, []).append(text[:500])
    for scope in scope_outputs:
        inv_sections: dict[str, str] = scope.get("investment_sections") or {}
        for iid in (scope.get("inv_ids") or [scope.get("inv_id", "")]):
            if iid and iid not in inv_sections and iid in inv_section_texts:
                inv_sections[iid] = "\n\n".join(inv_section_texts[iid][:20])
        if inv_sections:
            scope["investment_sections"] = inv_sections

    # ── F-032: Build narration configurable so NarrationToolbox tools resolve ──
    base_configurable = dict(((config or {}).get("configurable") or {}))
    budget_counter: list[int] = [0]   # F-046: shared mutable counter for per-narrator budget
    narration_configurable = {
        **base_configurable,
        "scope_outputs": {s.get("scope_id", ""): s for s in scope_outputs},
        "investment_scoring": state.get("investment_scoring") or {},
        "investment_intelligence": state.get("investment_intelligence") or {},
        "all_excerpts": all_excerpts,
        "relevance_subset": None,   # narrators see full portfolio at assembly time
        "narrator_budget": NARRATOR_CALL_BUDGET,
        "_narrator_call_counter": budget_counter,
    }
    # Preserve top-level config keys (callbacks, run_id, metadata) so LLM calls
    # inside report_assembler stay attached to the parent LangGraph trace in OTEL.
    # Only "configurable" is overridden with the enriched narration context.
    narration_config = {**(config or {}), "configurable": narration_configurable}

    # ── F-029: Retry loop — up to ASSEMBLY_MAX_RETRIES on structure validation failure ──
    def _validate_report_structure(md: str) -> bool:
        """Check that the report has at least 4 required H2 sections."""
        required = {"Executive Summary", "Portfolio Dashboard", "Cross-Cutting Analysis", "Bibliography"}
        found = set(_re.findall(r"^## (.+)$", md, _re.MULTILINE))
        return bool(required & found) and len(md) > 500

    from src.core import report_assembler
    report_dict: dict = {}
    final_report_md: str = ""
    last_exc: Exception | None = None

    for attempt in range(ASSEMBLY_MAX_RETRIES):
        try:
            budget_counter[0] = 0  # reset per attempt
            report = await report_assembler.assemble_report(
                scope_outputs=scope_outputs,
                cross_cutting_analysis=state.get("cross_cutting_analysis") or {},
                all_excerpts=all_excerpts,
                confidence_map=state.get("confidence_map") or {},
                coverage_pct=float(state.get("coverage_pct") or 0.0),
                grade=state.get("grade") or "D",
                model=state.get("synthesis_model", ""),
                config=narration_config,
            )
            report_dict = report if isinstance(report, dict) else {"body": str(report)}
            final_report_md = report_dict.get("markdown", report_dict.get("body", ""))
            if _validate_report_structure(final_report_md):
                break
            logger.warning(
                "assemble_report attempt %d/%d failed structure validation, retrying",
                attempt + 1, ASSEMBLY_MAX_RETRIES,
            )
        except Exception as exc:
            last_exc = exc
            logger.error("assemble_report attempt %d/%d raised: %s", attempt + 1, ASSEMBLY_MAX_RETRIES, exc)
            if attempt == ASSEMBLY_MAX_RETRIES - 1:
                return {"errors": [f"assemble_report:{exc}"]}

    if not final_report_md and last_exc:
        return {"errors": [f"assemble_report:{last_exc}"]}

    report_path: Path | None = None
    excerpts_csv_path: Path | None = None
    if threads_dir and final_report_md:
        report_dir = Path(threads_dir)
        try:
            # asyncio-APPROVED-1: to_thread wraps blocking mkdir
            await asyncio.to_thread(report_dir.mkdir, parents=True, exist_ok=True)
        except Exception:
            pass
        report_path = report_dir / "final_report.md"
        try:
            await asyncio.to_thread(report_path.write_text, final_report_md, "utf-8")
        except Exception as exc:
            logger.error("assemble_report write failed: %s", exc)
            errors.append(f"assemble_report:write:{exc}")

        # ── F-039b: Write excerpts.csv from all_excerpts (13-column legacy format) ──
        if all_excerpts:
            import csv, io
            def _write_csv() -> bytes:
                buf = io.StringIO()
                fieldnames = [
                    "ref_id", "excerpt_id", "inv_id", "scope_id", "link_id",
                    "file_id", "source_file", "page", "page_start", "page_end",
                    "source_type", "credibility_tier", "type", "significance",
                    "context_needed", "quote",
                ]
                writer = csv.DictWriter(buf, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(all_excerpts)
                return buf.getvalue().encode("utf-8")
            try:
                csv_bytes = await asyncio.to_thread(_write_csv)
                excerpts_csv_path = report_dir / "excerpts.csv"
                await asyncio.to_thread(excerpts_csv_path.write_bytes, csv_bytes)
            except Exception as exc:
                logger.warning("excerpts.csv write failed (non-fatal): %s", exc)

    result: dict[str, Any] = {
        "final_report_md": final_report_md,
        "analyst_report": report_dict,
        "final_report_md_path": str(report_path) if report_path is not None else None,
        "excerpts_csv_path": str(excerpts_csv_path) if excerpts_csv_path is not None else None,
        "bibliography": report_dict.get("bibliography") or [],
    }
    if errors:
        result["errors"] = errors
    return result


# ---------------------------------------------------------------------------
# verify_report — numerical & allocation verification
# ---------------------------------------------------------------------------


async def verify_report(state: AnalyzeState, config: RunnableConfig = None) -> dict:
    """Scan final_report.md for dollar figures and cross-reference against investment_scoring.

    Steps:
      1. Extract $X.XM / $XXM patterns from final_report_md.
      2. Cross-reference against investment_scoring approved/paid amounts (±10%).
      3. Call LLM with matching annotated_excerpts to verify unmatched figures.
      4. Store results in allocation_verification and numerical_verification.
      5. Build numerical_provenance from InvestmentFacts.
    """
    import re as _re
    final_report_md = state.get("final_report_md") or ""
    investment_scoring = state.get("investment_scoring") or {}
    all_excerpts = state.get("all_excerpts") or []
    model = state.get("synthesis_model") or _DEFAULT_MODEL
    errors: list[str] = []

    if not final_report_md:
        return {}

    # Extract dollar figures from report
    dollar_pattern = _re.compile(r'\$(\d+(?:\.\d+)?)\s*([MBK]?)', _re.IGNORECASE)
    found_figures: list[dict] = []
    for m in dollar_pattern.finditer(final_report_md):
        val = float(m.group(1))
        suffix = m.group(2).upper()
        if suffix == "M":
            val *= 1_000_000
        elif suffix == "B":
            val *= 1_000_000_000
        elif suffix == "K":
            val *= 1_000
        ctx_start = max(0, m.start() - 80)
        ctx_end = min(len(final_report_md), m.end() + 80)
        found_figures.append({
            "raw": m.group(0),
            "value": val,
            "context": final_report_md[ctx_start:ctx_end],
        })

    if not found_figures:
        return {"allocation_verification": [], "numerical_verification": [], "numerical_provenance": []}

    # Cross-reference against known investment amounts (±10% tolerance)
    inv_amounts: dict[str, dict] = {
        inv_id: {
            "approved": float(d.get("approved_amount", 0) or 0),
            "paid": float(d.get("paid_amount", 0) or 0),
        }
        for inv_id, d in investment_scoring.items()
        if isinstance(d, dict)
    }

    allocation_verification: list[dict] = []
    for fig in found_figures[:20]:
        val = fig["value"]
        match_inv = None
        match_type = None
        for inv_id, amounts in inv_amounts.items():
            for amt_type, amt_val in amounts.items():
                if amt_val > 0 and abs(val - amt_val) / max(amt_val, 1) < 0.10:
                    match_inv = inv_id
                    match_type = amt_type
                    break
            if match_inv:
                break
        supporting = [
            ex.get("text", "")[:200]
            for ex in all_excerpts
            if any(fig["raw"].replace(" ", "") in str(nf).replace(" ", "")
                   for nf in (ex.get("numerical_facts") or []))
        ][:2]
        allocation_verification.append({
            "figure": fig["raw"],
            "value": val,
            "context": fig["context"][:160],
            "matched_inv_id": match_inv,
            "matched_type": match_type,
            "supporting_excerpts": supporting,
            "status": "matched" if match_inv else "unmatched",
        })

    # LLM verification pass for unmatched figures
    unmatched = [v for v in allocation_verification if v["status"] == "unmatched"]
    numerical_verification: list[dict] = []
    if unmatched:
        excerpts_text = "\n".join(
            f"  [{ex.get('source', '?')}] {ex.get('text', '')[:200]}"
            for ex in all_excerpts[:10]
        )
        unmatched_text = "\n".join(
            f"  - {v['figure']}: \"{v['context']}\"" for v in unmatched[:5]
        )
        prompt = (
            "Review these dollar figures from a portfolio report that could not be "
            "cross-referenced against investment records.\n\n"
            f"Unmatched figures:\n{unmatched_text}\n\n"
            f"Evidence excerpts:\n{excerpts_text}\n\n"
            "For each figure determine: (a) verified by excerpts, "
            "(b) derived/aggregate, or (c) discrepancy requiring flagging. "
            "Respond with a JSON array: [{figure, verdict, explanation}]"
        )
        try:
            raw = await acall_llm(prompt, model=model, config=config)
            raw_str = raw if isinstance(raw, str) else str(raw)
            # Use raw_decode to stop at the first complete JSON value (avoids "Extra
            # data" errors when the LLM appends prose after the JSON array).
            arr_start = raw_str.find("[")
            if arr_start >= 0:
                parsed, _ = json.JSONDecoder().raw_decode(raw_str[arr_start:])
                numerical_verification = parsed if isinstance(parsed, list) else []
        except Exception as exc:
            logger.warning("verify_report LLM verification failed: %s", exc)
            errors.append(f"verify_report:llm:{exc}")

    # Build numerical_provenance from InvestmentFacts in scope_outputs
    numerical_provenance: list[dict] = []
    for s in (state.get("scope_outputs") or []):
        facts = s.get("investment_facts") or {}
        if facts:
            numerical_provenance.append({
                "inv_id": s.get("inv_id", ""),
                "scope_id": s.get("scope_id", ""),
                **facts,
            })

    # F-034: rewrite final_report_md in-place for discrepancy figures,
    # mirroring allocation_verifier.apply_allocation_edits() behaviour.
    discrepancy_figures = [
        v for v in numerical_verification
        if isinstance(v, dict) and v.get("verdict") == "discrepancy"
    ]
    annotated_md = final_report_md
    for disc in discrepancy_figures:
        fig_text = disc.get("figure", "")
        note = (disc.get("explanation") or "figure unverified")[:80]
        if fig_text and fig_text in annotated_md:
            annotated_md = annotated_md.replace(fig_text, f"{fig_text} [⚠ {note}]", 1)

    out: dict[str, Any] = {
        "allocation_verification": allocation_verification,
        "numerical_verification": numerical_verification,
        "numerical_provenance": numerical_provenance,
    }
    if annotated_md != final_report_md:
        out["final_report_md"] = annotated_md
    if errors:
        out["errors"] = errors
    return out


# ---------------------------------------------------------------------------
# Graph compilation
# ---------------------------------------------------------------------------

_builder = StateGraph(AnalyzeState)

_builder.add_node("load_catalog", load_catalog)
_builder.add_node("orientation", orientation)
_builder.add_node("compute_scopes", compute_scopes)
_builder.add_node("build_timelines", build_timelines)
_builder.add_node("generate_investment_narrative", generate_investment_narrative)
_builder.add_node("collect_investment_narratives", collect_investment_narratives)
_builder.add_node("generate_scope_synthesis", generate_scope_synthesis)
_builder.add_node("collect_timeline_narratives", collect_timeline_narratives)
_builder.add_node("run_causal_pipeline", run_causal_pipeline)
# dispatch_investment_reports and dispatch_scope_sections are conditional-edge routers, not nodes
_builder.add_node("build_investment_report_worker", build_investment_report_worker)
_builder.add_node("collect_investment_reports", collect_investment_reports)
_builder.add_node("synthesize_scope_section_worker", synthesize_scope_section_worker)
_builder.add_node("collect_scope_sections", collect_scope_sections)
_builder.add_node("cross_cutting_analysis", cross_cutting_analysis)
_builder.add_node("quality_assessment", quality_assessment)
_builder.add_node("assemble_report", assemble_report)
_builder.add_node("verify_report", verify_report)

_builder.add_edge(START, "load_catalog")
_builder.add_edge("load_catalog", "orientation")
_builder.add_edge("orientation", "compute_scopes")
_builder.add_edge("compute_scopes", "build_timelines")

# Phase 2.6 two-level fan-out:
# build_timelines → dispatch_investment_narratives (per-investment)
#   → generate_investment_narrative × N → collect_investment_narratives
#   → dispatch_scope_syntheses (per-scope)
#   → generate_scope_synthesis × M → collect_timeline_narratives
_builder.add_conditional_edges(
    "build_timelines",
    dispatch_investment_narratives,
    {
        "generate_investment_narrative": "generate_investment_narrative",
        "collect_investment_narratives": "collect_investment_narratives",
    },
)
_builder.add_edge("generate_investment_narrative", "collect_investment_narratives")
_builder.add_conditional_edges(
    "collect_investment_narratives",
    dispatch_scope_syntheses,
    {
        "generate_scope_synthesis": "generate_scope_synthesis",
        "collect_timeline_narratives": "collect_timeline_narratives",
    },
)
_builder.add_edge("generate_scope_synthesis", "collect_timeline_narratives")
_builder.add_edge("collect_timeline_narratives", "run_causal_pipeline")
# §3.5 fan-out: run_causal_pipeline → dispatch_investment_reports → [build_investment_report_worker × N]
#                                                                  → collect_investment_reports
_builder.add_conditional_edges(
    "run_causal_pipeline",
    dispatch_investment_reports,
    {
        "build_investment_report_worker": "build_investment_report_worker",
        "collect_investment_reports": "collect_investment_reports",
    },
)
_builder.add_edge("build_investment_report_worker", "collect_investment_reports")

# §3.6 fan-out: collect_investment_reports → dispatch_scope_sections → [synthesize_scope_section_worker × N]
#                                                                     → collect_scope_sections
_builder.add_conditional_edges(
    "collect_investment_reports",
    dispatch_scope_sections,
    {
        "synthesize_scope_section_worker": "synthesize_scope_section_worker",
        "collect_scope_sections": "collect_scope_sections",
    },
)
_builder.add_edge("synthesize_scope_section_worker", "collect_scope_sections")
_builder.add_edge("collect_scope_sections", "cross_cutting_analysis")
_builder.add_edge("cross_cutting_analysis", "quality_assessment")
_builder.add_edge("quality_assessment", "assemble_report")
_builder.add_edge("assemble_report", "verify_report")
_builder.add_edge("verify_report", END)

analyze_graph = _builder.compile()
