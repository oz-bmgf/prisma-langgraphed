"""NarrationToolNode — 6 tools for exec summary and key insights narrators.

Used by lens narrators inside assemble_report. All corpus search delegates
to search_collection from collection_tools. Tools are thin retrieval wrappers;
no LLM calls inside tools except verify_claim (which routes through acall_llm).

Configurable keys consumed:
  search_backend          : SearchBackend instance
  doc_list                : list[dict]  — document catalog
  investment_scoring      : dict        — investment metadata keyed by inv_id
  investment_intelligence : dict        — per-investment intelligence keyed by inv_id
  scope_outputs           : dict        — scope_id → ScopeOutput dict (for evidence packs)
  pages_dir               : str | None  — pages directory for read_primary_document
  relevance_subset        : set[str] | None  — optional inv_id filter for narrators
  verifier_model          : str | None  — model for verify_claim (defaults to env or fast model)
  all_excerpts            : list[dict]  — all_excerpts from causal pipeline (F-045 fallback)
  narrator_budget         : int         — max tool calls per narrator session (F-046)
  _narrator_call_counter  : list[int]   — shared mutable [calls_used] counter (F-046)
"""
from __future__ import annotations

import json
import logging
import os
from typing import Optional

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from src.backends.base import SearchBackend
from src.config import DEFAULT_FAST_MODEL
from src.tools.collection_tools import _fmt_results

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Budget enforcement helper (F-046)
# ---------------------------------------------------------------------------

_BUDGET_EXHAUSTED = "[budget exhausted — wrap up now]"


def _use_budget(configurable: dict) -> bool:
    """Decrement the narrator call counter. Returns False when budget exhausted."""
    budget: int = configurable.get("narrator_budget") or 0
    if budget <= 0:
        return True  # no budget configured — no enforcement
    counter: list[int] = configurable.get("_narrator_call_counter") or [0]
    if counter[0] >= budget:
        return False
    counter[0] += 1
    return True


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@tool
async def list_filtered_investments(
    posture: Optional[str] = None,
    bow_id: Optional[str] = None,
    divergence_severity: Optional[str] = None,
    config: RunnableConfig = None,
) -> str:
    """List investments in scope, optionally filtered by posture, BOW, or divergence severity.

    Returns inv_id, title, approved amount, paid amount, allocation, posture,
    and AI/team divergence for each investment. Call at narrator start to
    identify relevant investments before reasoning. Bounded to 100 results.

    Args:
        posture:            Filter by posture (e.g. "program_critical", "pathway_altering",
                            "efficiency_reducing", "on_track").
        bow_id:             Filter to investments in a specific Bundle of Work.
        divergence_severity: Filter by AI/team divergence ("high", "medium", "low", "aligned").
    """
    configurable = (config or {}).get("configurable", {})
    if not _use_budget(configurable):
        return _BUDGET_EXHAUSTED

    investment_scoring: dict = configurable.get("investment_scoring") or {}
    relevance_subset: Optional[set] = configurable.get("relevance_subset")
    scope_outputs: dict = configurable.get("scope_outputs") or {}

    inv_ids = sorted(relevance_subset) if relevance_subset else sorted(investment_scoring.keys())

    # --- F-044: Apply filters ---
    if bow_id:
        bow_inv_ids: set[str] = set()
        for scope in scope_outputs.values():
            if isinstance(scope, dict) and bow_id in (scope.get("bow_ids") or []):
                bow_inv_ids.update(scope.get("inv_ids") or [])
        inv_ids = [i for i in inv_ids if i in bow_inv_ids]

    if posture:
        inv_ids = [
            i for i in inv_ids
            if (investment_scoring.get(i) or {}).get("posture", "").lower() == posture.lower()
        ]

    if divergence_severity:
        filtered_by_div: list[str] = []
        for iid in inv_ids:
            for scope in scope_outputs.values():
                if isinstance(scope, dict):
                    inv_report = scope.get("investment_report") or {}
                    if (
                        inv_report.get("divergence_severity", "").lower() == divergence_severity.lower()
                        and iid in (scope.get("inv_ids") or [])
                    ):
                        filtered_by_div.append(iid)
                        break
        inv_ids = filtered_by_div

    if not inv_ids:
        return "(no investments match the specified filters)"

    lines = [f"{len(inv_ids)} investments in scope:"]
    for iid in inv_ids[:100]:
        inv = investment_scoring.get(iid) or {}
        approved = float(inv.get("approved_amount", 0) or 0)
        paid = float(inv.get("paid_amount", 0) or 0)
        alloc = float(inv.get("allocation", 0) or 0)
        posture_val = inv.get("posture", "") or ""
        title = (inv.get("title", "") or "")[:60]
        # Divergence from scope_outputs
        div = ""
        for scope in scope_outputs.values():
            if isinstance(scope, dict) and iid in (scope.get("inv_ids") or []):
                div = (scope.get("investment_report") or {}).get("divergence_severity", "")
                break
        div_str = f" div={div}" if div else ""
        lines.append(
            f"- {iid} ({title}) "
            f"approved=${approved / 1e6:.1f}M paid=${paid / 1e6:.1f}M "
            f"alloc=${alloc / 1e6:.2f}M posture={posture_val}{div_str}"
        )
    if len(inv_ids) > 100:
        lines.append(f"… (+{len(inv_ids) - 100} more, truncated)")
    return "\n".join(lines)


@tool
async def get_inv_metadata(
    inv_id: str,
    config: RunnableConfig = None,
) -> str:
    """Return structured metadata for one investment.

    Cheaper than read_evidence_pack — use when you need scoring, allocation,
    and posture without full document evidence.

    Args:
        inv_id: Investment identifier (e.g. "INV-041892").
    """
    configurable = (config or {}).get("configurable", {})
    if not _use_budget(configurable):
        return _BUDGET_EXHAUSTED

    investment_scoring: dict = configurable.get("investment_scoring") or {}
    investment_intelligence: dict = configurable.get("investment_intelligence") or {}

    inv_score = investment_scoring.get(inv_id)
    inv_intel = investment_intelligence.get(inv_id) or {}

    if inv_score is None and not inv_intel:
        return f"(inv_id={inv_id} not found)"

    inv = inv_score or {}
    fields = {
        "inv_id": inv_id,
        "title": inv.get("title", inv_intel.get("title", "")),
        "org": inv.get("org", ""),
        "approved_amount_M": round(float(inv.get("approved_amount", 0) or 0) / 1e6, 2),
        "paid_amount_M": round(float(inv.get("paid_amount", 0) or 0) / 1e6, 2),
        "allocation_M": round(float(inv.get("allocation", 0) or 0) / 1e6, 2),
        "start": inv.get("start", ""),
        "end": inv.get("end", ""),
        "status": inv.get("status", ""),
        "posture": inv.get("posture", ""),
        "execution": inv.get("execution", ""),
        "impact": inv.get("impact", ""),
        "bow_id": inv.get("bow_id", ""),
        "bow_name": inv.get("bow_name", ""),
        "managing_team": inv.get("managing_team", ""),
    }
    vehicle_type = inv.get("vehicle_type", "") or ""
    if vehicle_type:
        fields["vehicle_type"] = vehicle_type

    lines = [f"{k}: {v}" for k, v in fields.items() if v != ""]

    # Append key intelligence fields if available
    key_results = inv_intel.get("key_results", "") or ""
    if key_results:
        lines.append(f"\nKey results:\n{key_results[:1000]}")

    return "\n".join(lines)


@tool
async def search_within_scope(
    query: str,
    top_k: int = 5,
    inv_id: Optional[str] = None,
    config: RunnableConfig = None,
) -> str:
    """Search the embedding index within the narrator's relevance scope.

    Returns top_k chunks with file_id, page range, score, and a text snippet.
    When inv_id is provided, scopes the search to that specific investment.
    When the toolbox has a relevance_subset, filters results to that subset.

    Args:
        query:  Natural-language question or keyword.
        top_k:  Number of results to return (default 5).
        inv_id: Optional investment filter; if omitted uses relevance_subset.
    """
    configurable = (config or {}).get("configurable", {})
    if not _use_budget(configurable):
        return _BUDGET_EXHAUSTED
    backend: SearchBackend = configurable["search_backend"]
    relevance_subset: Optional[set] = configurable.get("relevance_subset")

    if inv_id:
        results = await backend.search(query, top_k=top_k, inv_id_filter=inv_id)
    elif relevance_subset:
        wide = await backend.search(query, top_k=top_k * 6)
        results = [r for r in wide if (r.inv_id or "") in relevance_subset][:top_k]
    else:
        results = await backend.search(query, top_k=top_k)

    return _fmt_results(results)


@tool
async def read_evidence_pack(
    inv_id: str,
    config: RunnableConfig = None,
) -> str:
    """Return the full evidence pack for one investment.

    Looks up the scope-section body (Phase 3 analysis output) for this
    investment first. Falls back to investment metadata + key results if
    no scope body is available.

    Args:
        inv_id: Investment identifier.
    """
    configurable = (config or {}).get("configurable", {})
    if not _use_budget(configurable):
        return _BUDGET_EXHAUSTED
    scope_outputs: dict = configurable.get("scope_outputs") or {}
    investment_scoring: dict = configurable.get("investment_scoring") or {}
    investment_intelligence: dict = configurable.get("investment_intelligence") or {}

    # Find scope body — scope_outputs is keyed by scope_id, each scope has
    # per-investment content under its scope_output dict (written by assemble_report node)
    for scope_id, scope in scope_outputs.items():
        if isinstance(scope, dict):
            inv_sections = scope.get("investment_sections") or {}
            body = inv_sections.get(inv_id, "")
            if body:
                head = body[:6000]
                truncated = " […truncated…]" if len(body) > 6000 else ""
                return f"=== Scope section body for {inv_id} (scope {scope_id}) ===\n{head}{truncated}"

    # F-045: Second fallback — filter all_excerpts (from link investigations) by inv_id
    all_excerpts: list[dict] = configurable.get("all_excerpts") or []
    inv_excerpts = [ex for ex in all_excerpts if ex.get("inv_id", "") == inv_id]
    if inv_excerpts:
        lines = [f"=== Evidence excerpts for {inv_id} ({len(inv_excerpts)} items) ==="]
        for ex in inv_excerpts[:20]:
            ref = ex.get("ref_id", ex.get("excerpt_id", ""))
            src = ex.get("source_file", ex.get("source", ""))
            text = ex.get("text", ex.get("quote", ""))[:400]
            lines.append(f"\n[{ref}] {src}\n{text}")
        return "\n".join(lines)

    # Final fallback: metadata + intelligence
    inv = investment_scoring.get(inv_id) or {}
    inv_intel = investment_intelligence.get(inv_id) or {}
    if not inv and not inv_intel:
        return f"(inv_id={inv_id} not found)"

    lines = []
    title = inv.get("title", inv_intel.get("title", inv_id))
    approved = float(inv.get("approved_amount", 0) or 0)
    lines.append(f"{inv_id} — {title} (${approved / 1e6:.1f}M)")

    for field in ("key_results", "public_description", "strategic_goals"):
        value = (inv_intel.get(field) or inv.get(field) or "")[:2000]
        if value:
            lines.append(f"\n{field.replace('_', ' ').title()}:\n{value}")

    return "\n".join(lines)


@tool
async def read_primary_document(
    file_id: str,
    pages: Optional[str] = None,
    config: RunnableConfig = None,
) -> str:
    """Read text from a specific document via its chunks.

    Returns the first ~2000 characters of the document concatenated from
    chunks for that file_id, or a specific page range if pages is provided.

    Args:
        file_id: Document identifier.
        pages:   Optional page range as "N-M" (e.g. "3-7"). When omitted,
                 returns the opening content of the document.
    """
    import asyncio
    from pathlib import Path

    configurable = (config or {}).get("configurable", {})
    if not _use_budget(configurable):
        return _BUDGET_EXHAUSTED
    pages_dir = configurable.get("pages_dir")

    if not pages_dir:
        return "(pages_dir not configured)"

    from src.tools.collection_tools import _read_page_range

    if pages:
        try:
            lo_s, hi_s = pages.replace(" ", "").split("-")
            # asyncio-APPROVED-1: to_thread wraps blocking page file read
            return await asyncio.to_thread(
                _read_page_range, Path(pages_dir), file_id, int(lo_s), int(hi_s)
            )
        except (ValueError, TypeError):
            return f"(invalid pages format {pages!r} — use 'N-M')"

    # asyncio-APPROVED-1: to_thread wraps blocking page file read
    return await asyncio.to_thread(_read_page_range, Path(pages_dir), file_id, 1, 5)


@tool
async def verify_claim(
    claim: str,
    inv_id: Optional[str] = None,
    config: RunnableConfig = None,
) -> str:
    """Verify a factual claim against the evidence pack using a fast LLM pass.

    Returns "SUPPORTED", "CONTRADICTED", "UNVERIFIABLE", or "NEEDS_MORE_EVIDENCE"
    with a one-line rationale. Use to quality-check specific facts before
    including them in the final narrative.

    Args:
        claim:  The specific factual claim to verify (e.g. "enrollment reached
                80% of target by Q3 2024").
        inv_id: Optional investment context. When provided, the verification
                looks up the evidence pack for this investment.
    """
    configurable = (config or {}).get("configurable", {})
    if not _use_budget(configurable):
        return _BUDGET_EXHAUSTED
    acall_llm = configurable.get("acall_llm")
    verifier_model = configurable.get("verifier_model") or os.environ.get(
        "NQPR_VERIFIER_MODEL", DEFAULT_FAST_MODEL
    )

    # Build evidence context
    evidence_context = ""
    if inv_id:
        scope_outputs: dict = configurable.get("scope_outputs") or {}
        investment_scoring: dict = configurable.get("investment_scoring") or {}
        for scope_id, scope in scope_outputs.items():
            if isinstance(scope, dict):
                inv_sections = scope.get("investment_sections") or {}
                body = inv_sections.get(inv_id, "")
                if body:
                    evidence_context = body[:4000]
                    break
        if not evidence_context:
            inv = investment_scoring.get(inv_id) or {}
            evidence_context = (
                f"title={inv.get('title', '')} "
                f"approved=${float(inv.get('approved_amount', 0) or 0) / 1e6:.1f}M"
            )

    if acall_llm is None:
        return (
            f"[verify_claim: acall_llm not configured in configurable]\n"
            f"claim: {claim}\n"
            f"evidence: {evidence_context[:200]}"
        )

    prompt = (
        f"Claim: {claim}\n\n"
        f"Evidence context:\n{evidence_context[:3000] if evidence_context else '(no evidence context provided)'}\n\n"
        "Is this claim SUPPORTED, CONTRADICTED, UNVERIFIABLE, or NEEDS_MORE_EVIDENCE based on the evidence above? "
        "Reply with the verdict on the first line, then one sentence of rationale."
    )
    try:
        response = await acall_llm(prompt, model=verifier_model)
        return response
    except Exception as exc:
        logger.warning("verify_claim LLM call failed: %s", exc)
        return f"(verify_claim failed: {exc})"


# ---------------------------------------------------------------------------
# Exported tool list for ToolNode construction
# ---------------------------------------------------------------------------

NARRATION_TOOLS = [
    list_filtered_investments,
    get_inv_metadata,
    search_within_scope,
    read_evidence_pack,
    read_primary_document,
    verify_claim,
]
