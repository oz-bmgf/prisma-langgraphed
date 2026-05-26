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
from pathlib import Path
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from src.config import DEFAULT_SYNTHESIS_MODEL as _DEFAULT_MODEL
from src.core.llm_utils import acall_llm
from src.graph.state import (
    AnalyzeState,
    InvestmentReportWorkerState,
    ScopeNarrativeState,
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
    model = state.get("synthesis_model") or _DEFAULT_MODEL

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

    # --- Optional: top document summaries from embedding index (Priority 3c) ---
    doc_excerpts_section = ""
    backend = ((config or {}).get("configurable") or {}).get("search_backend")
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
        except Exception as exc:
            logger.debug("orientation: embedding search failed (non-fatal): %s", exc)

    prompt = (
        f"You are producing a structured portfolio orientation for an investment review.{focus_str}\n\n"
        f"## Bundles of Work ({len(bow_investment_map)})\n"
        + "\n".join(bow_lines or ["  (none)"])
        + f"\n\n## Investments ({len(investment_scoring)})\n"
        + "\n".join(inv_lines or ["  (none)"])
        + f"\n\n## Available Documents\n  {doc_summary}"
        + doc_excerpts_section
        + "\n\nRespond with a JSON object containing exactly these keys:\n"
        "  theory_of_change: string — 2-3 sentences on the portfolio's overall theory of change\n"
        "  major_bets: list of strings — the 3-5 most significant strategic bets in this portfolio\n"
        "  stated_priorities: list of strings — explicit review priorities or focus areas\n"
        "  key_timelines: list of strings — critical upcoming milestones or decision points\n"
        "  portfolio_health_signals: list of strings — early signals about portfolio health\n"
        "  bow_summaries: dict mapping bow_id to a 1-sentence summary of that bundle\n"
        "  initial_concerns: list of strings — preliminary concerns for investigator focus\n\n"
        "Use the data above. Respond ONLY with the JSON object."
    )

    import re as _orient_re
    try:
        raw = await acall_llm(prompt, model=model, config=config)
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
        return {"program_context": program_context}
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

    MIN_CHUNKS = 200     # skip threshold
    SPLIT_CHUNKS = 12_000  # split-by-BOW threshold
    SPLIT_INVS = 8       # split-by-investment-count threshold

    # --- Query chunk counts per BOW ---
    bow_chunk_counts: dict[str, int] = {}
    backend = ((config or {}).get("configurable") or {}).get("search_backend")
    if backend and hasattr(backend, "count_by_bow_id"):
        try:
            bow_chunk_counts = await backend.count_by_bow_id() or {}
        except Exception as exc:
            logger.debug("compute_scopes: count_by_bow_id failed (non-fatal): %s", exc)

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
    # If bow_data has a "sub_topic" key, group by it; otherwise each BOW is its own group.
    subtopic_to_bows: dict[str, list[str]] = {}
    for bow_id in active_bow_ids:
        bow_data = bow_investment_map.get(bow_id) or {}
        sub_topic = (
            bow_data.get("sub_topic") or bow_data.get("focus_area") or bow_id
            if isinstance(bow_data, dict)
            else bow_id
        )
        subtopic_to_bows.setdefault(sub_topic, []).append(bow_id)

    # --- Build scopes from groups ---
    scopes: list[dict] = []
    scope_idx = 0

    for sub_topic, bow_ids_in_group in subtopic_to_bows.items():
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
        seen_invs: set[str] = set()
        unique_inv_ids: list[str] = []
        for iid in group_inv_ids:
            if iid not in seen_invs and iid in investment_scoring:
                seen_invs.add(iid)
                unique_inv_ids.append(iid)

        if not unique_inv_ids:
            continue

        # Size gate — skip tiny groups
        if bow_chunk_counts and group_chunk_count < MIN_CHUNKS:
            logger.debug(
                "compute_scopes: skipping group '%s' (%d chunks, %d invs)",
                sub_topic, group_chunk_count, len(unique_inv_ids),
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
                # Keep only inv_ids present in investment_scoring
                bow_inv_ids = [i for i in bow_inv_ids if i in investment_scoring]
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
                })
                scope_idx += 1
        else:
            # Keep whole group as single scope
            group_label = sub_topic if sub_topic != list(bow_ids_in_group)[0] else ", ".join(bow_ids_in_group[:2])
            primary_inv = unique_inv_ids[0]
            scopes.append({
                "scope_id": f"scope_{scope_idx:04d}",
                "inv_id": primary_inv,
                "inv_ids": unique_inv_ids,
                "bow_ids": bow_ids_in_group,
                "label": group_label,
                "chunk_count": group_chunk_count,
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

    return {"scope_timelines": {sid: st.to_dict() for sid, st in scope_timelines.items()}}


# ---------------------------------------------------------------------------
# dispatch_timeline_narratives — Phase 2.6 (pure router, returns list[Send])
# ---------------------------------------------------------------------------


async def dispatch_timeline_narratives(state: AnalyzeState):
    """Fan-out: one Send per scope for LLM narrative generation.

    Skips scopes whose scope_id is already present in timeline_narrative_results
    (i.e. this is a resumed run with a LangGraph checkpoint — no file cache used).
    """
    scopes = state.get("scopes") or []
    if not scopes:
        return "collect_timeline_narratives"

    scope_timelines = state.get("scope_timelines") or {}
    model = state.get("synthesis_model") or _DEFAULT_MODEL
    investment_scoring = state.get("investment_scoring") or {}

    # State-based resume check: skip scopes already in timeline_narrative_results
    already_done: set[str] = {
        r.get("scope_id", "")
        for r in (state.get("timeline_narrative_results") or [])
        if r.get("scope_id")
    }

    sends: list[Send] = []
    for scope in scopes:
        scope_id = scope.get("scope_id", "")
        inv_id = scope.get("inv_id", "")

        # Skip if already computed (resumed run via LangGraph checkpoint)
        if scope_id in already_done:
            continue

        # Use pre-built ScopeTimeline dict if available; fall back to basic data
        scope_tl = scope_timelines.get(scope_id) or {}
        investments_in_tl = scope_tl.get("investments") or []

        # Skip if scope timeline already has full narratives (shouldn't happen on fresh run)
        if investments_in_tl and all(
            inv.get("narrative") for inv in investments_in_tl
        ):
            continue

        inv_data = investment_scoring.get(inv_id) or {}
        timeline: dict[str, Any] = {
            "scope_id": scope_id,
            "inv_id": inv_id,
            "start": inv_data.get("start", "") if isinstance(inv_data, dict) else "",
            "end": inv_data.get("end", "") if isinstance(inv_data, dict) else "",
            "status": inv_data.get("status", "") if isinstance(inv_data, dict) else "",
            "approved_amount": inv_data.get("approved_amount", 0) if isinstance(inv_data, dict) else 0,
            "paid_amount": inv_data.get("paid_amount", 0) if isinstance(inv_data, dict) else 0,
            # Include the full pre-built ScopeTimeline dict so the worker has rich data
            "scope_timeline_dict": scope_tl,
        }
        sends.append(Send("generate_scope_narrative", {
            "scope_id": scope_id,
            "inv_id": inv_id,
            "timeline": timeline,
            "model": model,
            "result": None,
        }))

    return sends or "collect_timeline_narratives"


# ---------------------------------------------------------------------------
# generate_scope_narrative — Phase 2.6 (per-scope LLM worker)
# ---------------------------------------------------------------------------


async def generate_scope_narrative(state: ScopeNarrativeState) -> dict:
    """Generate rich multi-paragraph narrative(s) for an investment scope.

    Uses the pre-built ScopeTimeline from build_timelines when available,
    falling back to the basic financial timeline dict otherwise.
    """
    scope_id = state["scope_id"]
    inv_id = state["inv_id"]
    timeline = state["timeline"]
    model = state["model"]

    scope_tl_dict = timeline.get("scope_timeline_dict") or {}

    if scope_tl_dict.get("investments"):
        # Rich path: use build_timeline_narratives_async on the ScopeTimeline
        from src.core.investment_timeline import (
            InvestmentTimeline, DocumentEvent, ScopeTimeline,
            build_timeline_narratives_async,
        )
        from dataclasses import fields as dc_fields

        def _reconstruct() -> ScopeTimeline:
            inv_fields = {f.name for f in dc_fields(InvestmentTimeline)}
            doc_fields = {f.name for f in dc_fields(DocumentEvent)}
            inv_objs = []
            for inv_d in scope_tl_dict.get("investments", []):
                docs = [
                    DocumentEvent(**{k: v for k, v in doc.items() if k in doc_fields})
                    for doc in (inv_d.get("documents") or [])
                ]
                inv_obj = InvestmentTimeline(
                    **{k: v for k, v in inv_d.items() if k in inv_fields and k != "documents"}
                )
                inv_obj.documents = docs
                inv_objs.append(inv_obj)
            return ScopeTimeline(
                scope_id=scope_tl_dict.get("scope_id", scope_id),
                label=scope_tl_dict.get("label", inv_id),
                bow_ids=scope_tl_dict.get("bow_ids", []),
                investments=inv_objs,
                scope_flags=scope_tl_dict.get("scope_flags", []),
                narrative=scope_tl_dict.get("narrative", ""),
            )

        try:
            # asyncio-APPROVED-1: to_thread wraps dataclass reconstruction
            scope_timeline = await asyncio.to_thread(_reconstruct)
            await build_timeline_narratives_async(scope_timeline, model)
            updated_tl = scope_timeline.to_dict()
            updated_tl["scope_id"] = scope_id
            return {"timeline_narrative_results": [updated_tl]}
        except Exception as exc:
            logger.error("generate_scope_narrative (rich) failed %s: %s", scope_id, exc)
            return {
                "timeline_narrative_results": [{"scope_id": scope_id}],
                "errors": [f"timeline_narrative:{scope_id}:{exc}"],
            }

    # Fallback path: minimal 2-3 sentence narrative from basic financial data
    prompt = (
        f"Investment {inv_id} timeline narrative: "
        f"start={timeline.get('start', '')}, end={timeline.get('end', '')}, "
        f"status={timeline.get('status', '')}, "
        f"approved=${float(timeline.get('approved_amount', 0) or 0) / 1e6:.1f}M, "
        f"paid=${float(timeline.get('paid_amount', 0) or 0) / 1e6:.1f}M. "
        "Summarise the financial timeline and execution status in 2-3 sentences."
    )
    try:
        narrative = await acall_llm(prompt, model=model)
        timeline_out = dict(timeline)
        timeline_out["narrative"] = narrative if isinstance(narrative, str) else str(narrative)
        timeline_out["scope_id"] = scope_id
    except Exception as exc:
        logger.error("generate_scope_narrative (fallback) failed %s: %s", scope_id, exc)
        timeline_out = {"scope_id": scope_id}
        return {
            "timeline_narrative_results": [timeline_out],
            "errors": [f"timeline_narrative:{scope_id}:{exc}"],
        }

    return {"timeline_narrative_results": [timeline_out]}


# ---------------------------------------------------------------------------
# collect_timeline_narratives — Phase 2.7 (reducer → scope_timelines dict)
# ---------------------------------------------------------------------------


async def collect_timeline_narratives(state: AnalyzeState, config: RunnableConfig = None) -> dict:
    """Merge narrative results back into scope_timelines.

    LangGraph checkpointing handles resume — no file-based cache read or write.
    """
    results = state.get("timeline_narrative_results") or []
    existing = state.get("scope_timelines") or {}

    updated = dict(existing)
    for r in results:
        sid = r.get("scope_id")
        if not sid:
            continue
        if sid in updated:
            # Merge narrative fields back into existing ScopeTimeline dict
            existing_tl = dict(updated[sid])
            existing_tl["narrative"] = r.get("narrative", existing_tl.get("narrative", ""))
            # Merge per-investment narratives
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

    return {"scope_timelines": updated}


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
        # reducer fields must be initialised to [] (AGENTS.md §4)
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

    try:
        causal_result: dict = await causal_graph.ainvoke(causal_input, config)
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

        evidence_text = "\n".join(
            f"  - {la.get('link_id', '?')}: {la.get('status', '?')} — {la.get('gap_description', '')[:120]}"
            for la in link_assessments[:8]
        )
        prompt = (
            f"Produce a concise portfolio review section for investment {inv_id} (scope: {label}).\n\n"
            f"AI overall verdict: {inv_report.get('overall_status', 'unknown')}, "
            f"severity: {inv_report.get('severity', 'unknown')}\n"
            f"AI execution verdict: {inv_report.get('ai_execution_verdict', '')[:300]}\n\n"
            f"Link evidence:\n{evidence_text}\n\n"
            "Write a focused 3-4 paragraph narrative covering: key findings, "
            "execution status, financial performance, and critical risks. "
            "Be specific and data-driven. Do not pad with generic statements."
        )
        try:
            summary = await acall_llm(prompt, model=model, config=config)
            summary_str = summary if isinstance(summary, str) else str(summary)
        except Exception as exc:
            logger.warning("synthesize_scope_section_worker LLM failed %s: %s", scope_id, exc)
            summary_str = inv_report.get("executive_summary", "")

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

    # ── Phase 4 pre-computation (no LLM) ─────────────────────────────────────
    total_approved = 0.0
    total_paid = 0.0
    at_risk_count = 0
    concentration: dict[str, float] = {}

    for s in scope_outputs:
        facts = s.get("investment_facts") or {}
        approved = float(facts.get("approved_amount", 0) or 0)
        paid = float(facts.get("paid_amount", 0) or 0)
        total_approved += approved
        total_paid += paid
        if (s.get("investment_report") or {}).get("severity") in (
            "program_critical", "pathway_altering"
        ):
            at_risk_count += 1
        for bid in (s.get("bow_ids") or []):
            concentration[bid] = concentration.get(bid, 0) + approved

    concentration_by_bow: dict[str, float] = {
        bid: round(v / max(total_approved, 1) * 100, 1)
        for bid, v in concentration.items()
    }

    portfolio_metrics = {
        "total_approved_dollars": round(total_approved, 2),
        "total_paid_dollars": round(total_paid, 2),
        "at_risk_count": at_risk_count,
        "concentration_by_bow": concentration_by_bow,
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

    metrics_text = (
        f"Total approved: ${total_approved / 1e6:.1f}M  "
        f"Total paid: ${total_paid / 1e6:.1f}M  "
        f"At-risk investments: {at_risk_count}  "
        f"Concentration (top BOWs): "
        + ", ".join(f"{bid}={pct}%" for bid, pct in
                    sorted(concentration_by_bow.items(), key=lambda x: -x[1])[:5])
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

    cross_cutting: dict[str, Any] = {
        "patterns": llm_result.get("patterns", []),
        "contradictions": llm_result.get("contradictions", []),
        "shared_dependencies": llm_result.get("shared_dependencies", []),
        "emergent_decisions": llm_result.get("emergent_decisions", []),
        "essay": llm_result.get("essay", raw_str),
        "portfolio_metrics": portfolio_metrics,
    }

    return {"cross_cutting_analysis": cross_cutting}


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

    try:
        from src.core import report_assembler
        report = await report_assembler.assemble_report(
            scope_outputs=state.get("scope_outputs") or [],
            cross_cutting_analysis=state.get("cross_cutting_analysis") or {},
            all_excerpts=state.get("all_excerpts") or [],
            confidence_map=state.get("confidence_map") or {},
            coverage_pct=float(state.get("coverage_pct") or 0.0),
            grade=state.get("grade") or "D",
            model=state.get("synthesis_model", ""),
            config=config,
        )
        report_dict: dict = report if isinstance(report, dict) else {"body": str(report)}
        final_report_md: str = report_dict.get("markdown", report_dict.get("body", ""))
    except Exception as exc:
        logger.error("assemble_report failed: %s", exc)
        return {"errors": [f"assemble_report:{exc}"]}

    report_path: Path | None = None
    if threads_dir and final_report_md:
        report_path = Path(threads_dir) / "final_report.md"
        try:
            # asyncio-APPROVED-1: to_thread wraps blocking mkdir + write_text
            await asyncio.to_thread(report_path.parent.mkdir, parents=True, exist_ok=True)
            await asyncio.to_thread(report_path.write_text, final_report_md, "utf-8")
        except Exception as exc:
            logger.error("assemble_report write failed: %s", exc)
            errors.append(f"assemble_report:write:{exc}")

    result: dict[str, Any] = {
        "final_report_md": final_report_md,
        "analyst_report": report_dict,
        "final_report_md_path": str(report_path) if report_path is not None else None,
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
            m_arr = _re.search(r"\[.*\]", raw_str, _re.DOTALL)
            if m_arr:
                parsed = json.loads(m_arr.group(0))
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

    out: dict[str, Any] = {
        "allocation_verification": allocation_verification,
        "numerical_verification": numerical_verification,
        "numerical_provenance": numerical_provenance,
    }
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
_builder.add_node("generate_scope_narrative", generate_scope_narrative)
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

# Phase 2.6 fan-out: build_timelines → dispatch_timeline_narratives (router)
# → generate_scope_narrative (per-scope worker)
# → collect_timeline_narratives (reducer)
_builder.add_conditional_edges(
    "build_timelines",
    dispatch_timeline_narratives,
    {
        "generate_scope_narrative": "generate_scope_narrative",
        "collect_timeline_narratives": "collect_timeline_narratives",
    },
)
_builder.add_edge("generate_scope_narrative", "collect_timeline_narratives")
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
