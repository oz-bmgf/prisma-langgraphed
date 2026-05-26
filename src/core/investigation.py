"""Link investigation — multi-turn tool-calling loop for causal link assessment.

run_investigation() drives the loop:
  - Builds context from the causal link (claim dict)
  - Calls acall_llm with InvestigationActionsOutput schema each iteration
  - Dispatches tool actions via _execute_actions (async, asyncio.gather)
  - Detects saturation (3 consecutive empty rounds)
  - Enforces L4 coverage audit when NQPR_L4_COVERAGE_AUDIT=true
  - Returns InvestigationResult

Provider routing, consecutive empty check, and coverage audit are the three
critical invariants that must be preserved in all future refactors.

No LangGraph imports.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from src.config import (
    CONSECUTIVE_EMPTY_THRESHOLD,
    DEFAULT_MAX_TOKENS,
    DEFAULT_SYNTHESIS_MODEL,
    INVESTIGATION_L4_COVERAGE_AUDIT,
    MAX_INVESTIGATION_ITERATIONS,
)
from src.core.evidence_model import InvestigationResult
from src.core.llm_utils import acall_llm
from src.core.output_schemas import InvestigationActionsOutput
from src.prompts.tool_prompts import (
    INVESTIGATION_SYSTEM,
    INVESTIGATION_TOOL_DESCRIPTIONS,
    L4_COVERAGE_AUDIT_INSTRUCTION,
    L4_COVERAGE_AUDIT_ITEMS,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tool dispatcher (async)
# ---------------------------------------------------------------------------

_SUPPORTED_TOOLS = frozenset({
    "search_investment",
    "search_bow",
    "search_doc_type",
    "search_all",
    "search_web",
    "read_pages",
    "compute",
})


async def _execute_actions(
    actions: list[dict],
    tools: Any,
    inv_id: str,
    bow_id: str,
    model: str,
    facts: Any = None,
) -> tuple[list[dict], int]:
    """Async tool dispatcher — fan out tool actions concurrently.

    Returns (new_chunks, web_search_count). When tools is None or the
    tool action is unknown, returns empty results gracefully.
    """
    if not actions:
        return [], 0

    async def _dispatch_one(action: dict) -> tuple[list[dict], int]:
        tool = action.get("tool", "")
        query = action.get("query", "")
        if not query or tool not in _SUPPORTED_TOOLS:
            return [], 0

        if tools is None:
            return [], 0

        chunks: list[dict] = []
        web_count = 0
        try:
            idx = getattr(tools, "_embedding_index", None)
            _has_hybrid = idx is not None and hasattr(idx, "hybrid_search") and getattr(idx, "has_hybrid", False)

            def _search(q: str, *, top_k: int = 10, **kw) -> list[dict]:
                if _has_hybrid:
                    return idx.hybrid_search(q, top_k=top_k, embedding_weight=0.5, **kw)
                if idx is not None:
                    return idx.search_with_filter(q, top_k=top_k, **kw)
                return []

            if tool == "search_investment":
                # asyncio-APPROVED-1: to_thread wraps blocking embedding search
                raw = await asyncio.to_thread(_search, query, inv_id=inv_id, top_k=10)
                chunks.extend(raw)
            elif tool == "search_bow" and bow_id:
                # asyncio-APPROVED-1: to_thread wraps blocking embedding search
                raw = await asyncio.to_thread(_search, query, bow_id=bow_id, top_k=10)
                chunks.extend(raw)
            elif tool == "search_doc_type":
                doc_type = action.get("doc_type") or ""
                kw: dict = {"top_k": 8}
                if doc_type:
                    kw["doc_type"] = doc_type
                kw["inv_id"] = inv_id
                # asyncio-APPROVED-1: to_thread wraps blocking embedding search
                raw = await asyncio.to_thread(_search, query, **kw)
                chunks.extend(raw)
            elif tool == "search_all":
                # asyncio-APPROVED-1: to_thread wraps blocking embedding search
                raw = await asyncio.to_thread(_search, query, top_k=10)
                chunks.extend(raw)
            elif tool == "search_web":
                web_fn = getattr(tools, "search_web", None)
                if web_fn:
                    # asyncio-APPROVED-1: to_thread wraps blocking web HTTP call
                    result = await asyncio.to_thread(web_fn, query)
                    if result:
                        chunks.append({
                            "text": str(result)[:4000],
                            "file_id": f"web:{query[:40]}",
                            "filename": "web search",
                            "collection": "web",
                            "doc_type": "web",
                        })
                    web_count += 1
            elif tool == "read_pages":
                file_id = action.get("file_id", "")
                page_start = int(action.get("page_start", 1))
                page_end = int(action.get("page_end", page_start + 10))
                page_end = min(page_end, page_start + 20)
                read_fn = getattr(tools, "read_pages", None)
                if file_id and read_fn:
                    # asyncio-APPROVED-1: to_thread wraps blocking file read
                    text = await asyncio.to_thread(read_fn, file_id, page_start, page_end)
                    if text and len(text.strip()) > 50:
                        chunks.append({
                            "text": text,
                            "file_id": file_id,
                            "filename": f"{file_id} pp{page_start}-{page_end}",
                            "collection": "investment",
                        })
            elif tool == "compute" and facts:
                compute_fn = getattr(tools, "compute", None)
                if compute_fn:
                    # asyncio-APPROVED-1: to_thread wraps blocking compute call
                    result = await asyncio.to_thread(compute_fn, query, facts)
                    if result:
                        chunks.append({
                            "text": str(result)[:4000],
                            "file_id": "computed",
                            "filename": "compute tool",
                            "collection": "computed",
                        })
        except Exception as exc:
            logger.warning("_dispatch_one %s failed: %s", tool, str(exc)[:80])

        return chunks, web_count

    # asyncio-APPROVED-2: concurrent HTTP/search — fan out all tool actions in parallel
    dispatched = await asyncio.gather(*[_dispatch_one(a) for a in actions], return_exceptions=True)

    all_chunks: list[dict] = []
    web_total = 0
    for item in dispatched:
        if isinstance(item, BaseException):
            logger.warning("Tool action raised: %s", item)
            continue
        c, w = item
        all_chunks.extend(c)
        web_total += w

    return all_chunks, web_total


def _dedup_chunks(new_chunks: list[dict], existing: list[dict]) -> list[dict]:
    """Deduplicate new_chunks against existing accumulated evidence."""
    existing_ids = {
        c.get("chunk_id", c.get("file_id", "") + str(c.get("page_start", 0)))
        for c in existing
    }
    seen_groups: set[str] = set()
    for c in existing:
        vg = c.get("intelligence_version_group", "")
        if vg:
            seen_groups.add(vg)

    deduped: list[dict] = []
    for c in new_chunks:
        cid = c.get("chunk_id", c.get("file_id", "") + str(c.get("page_start", 0)))
        if cid in existing_ids:
            continue
        vg = c.get("intelligence_version_group", "")
        if vg and vg in seen_groups:
            continue
        if vg:
            seen_groups.add(vg)
        deduped.append(c)
        existing_ids.add(cid)
    return deduped


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

_EVIDENCE_BUDGET_CHARS = 400_000


def _format_evidence(chunks: list[dict]) -> str:
    parts: list[str] = []
    total = 0
    for i, chunk in enumerate(chunks):
        text = chunk.get("text", "")
        ref = f"§{i+1:04d}"
        fname = chunk.get("filename", "?")
        entry = f"[{ref}] {fname}:\n{text}"
        if total + len(entry) > _EVIDENCE_BUDGET_CHARS:
            break
        parts.append(entry)
        total += len(entry)
    return "\n\n".join(parts)


def _format_source_index(chunks: list[dict]) -> str:
    return "\n".join(
        f"  §{i+1:04d}: {c.get('filename', '?')}"
        for i, c in enumerate(chunks)
    )


def _build_first_prompt(claim: dict, accumulated_chunks: list[dict]) -> str:
    link_name = claim.get("name", claim.get("link_id", "unknown"))
    mechanism = claim.get("mechanism", "")
    assumptions = claim.get("assumptions", [])
    failure_modes = claim.get("failure_modes", [])
    dollars = claim.get("dollars_at_risk", 0)
    months = claim.get("months_at_risk", 0)

    parts = [
        f"CAUSAL LINK TO INVESTIGATE: {link_name}",
        f"From: {claim.get('from_stage', '?')} → To: {claim.get('to_stage', '?')}",
        f"Mechanism: {mechanism}",
        f"Key assumptions: {assumptions}",
        f"Failure modes: {failure_modes}",
    ]
    if dollars:
        parts.append(f"Stakes: ${dollars:,.0f} at risk / {months:.1f} months")

    if accumulated_chunks:
        parts.extend([
            f"\nINITIAL EVIDENCE ({len(accumulated_chunks)} excerpts):",
            _format_evidence(accumulated_chunks),
            f"\nSource index:\n{_format_source_index(accumulated_chunks)}",
        ])

    parts.append(
        "\nPlan your investigation. What evidence would confirm or challenge "
        "this link? Issue your first search actions."
    )
    return "\n".join(parts)


def _build_iteration_prompt(
    claim: dict,
    new_chunks: list[dict],
    all_chunks: list[dict],
    prev_output: InvestigationActionsOutput,
    iteration: int,
    max_iterations: int,
) -> str:
    link_name = claim.get("name", claim.get("link_id", "unknown"))
    new_evidence = (
        "\n\n".join(c.get("text", "") for c in new_chunks[:10])
        if new_chunks else "(no new evidence retrieved)"
    )
    l4_note = ""
    if INVESTIGATION_L4_COVERAGE_AUDIT and iteration >= max_iterations - 2:
        checklist = "\n".join(f"  {i+1}. {item}" for i, item in enumerate(L4_COVERAGE_AUDIT_ITEMS))
        l4_note = L4_COVERAGE_AUDIT_INSTRUCTION.format(checklist=checklist) + "\n\n"
    return (
        f"{l4_note}"
        f"LINK: {link_name}\n\n"
        f"YOUR PREVIOUS ANALYSIS (iteration {iteration}/{max_iterations}):\n"
        f"Status: {prev_output.status}\n"
        f"Answer: {prev_output.answer[:600]}\n\n"
        f"NEW EVIDENCE ({len(new_chunks)} new excerpts):\n{new_evidence}\n\n"
        f"ALL EVIDENCE ({len(all_chunks)} total):\n"
        f"{_format_evidence(all_chunks)}\n\n"
        f"Source index:\n{_format_source_index(all_chunks)}\n\n"
        f"Update your assessment. Continue searching or finalize?"
    )


def _build_system_prompt() -> str:
    return INVESTIGATION_SYSTEM.format(tool_descriptions=INVESTIGATION_TOOL_DESCRIPTIONS)


def _validate_coverage_audit(output: InvestigationActionsOutput) -> bool:
    """Return True if all 5 L4 coverage checklist items are addressed in the answer."""
    if not INVESTIGATION_L4_COVERAGE_AUDIT:
        return True
    answer_lower = output.answer.lower()
    keywords = [
        ["disbursement", "burn rate", "financial", "budget"],
        ["milestone", "deliverable", "on-track", "delay"],
        ["partnership", "grantee", "capacity", "execution"],
        ["external", "context", "policy", "landscape"],
        ["evidence quality", "freshness", "reporting", "contradict"],
    ]
    for group in keywords:
        if not any(kw in answer_lower for kw in group):
            return False
    return True


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

_TERMINAL_STATUSES = frozenset({"answered", "not_answerable", "unresolved_conflict"})


async def run_investigation(
    link_id: str,
    inv_id: str,
    bow_id: str,
    scope_id: str,
    claim: dict,
    model: str,
    *,
    tools: Any = None,
    config: Any = None,
    max_iterations: int = MAX_INVESTIGATION_ITERATIONS,
) -> InvestigationResult:
    """Run a multi-turn tool-calling investigation of one causal link.

    Dispatches tool actions asynchronously via asyncio.gather. Stops on:
    - terminal status (answered / not_answerable / unresolved_conflict)
    - empty next_actions
    - CONSECUTIVE_EMPTY_THRESHOLD consecutive rounds with no new chunks
    - max_iterations reached

    Returns InvestigationResult with all accumulated evidence and metadata.
    """
    t0 = time.time()
    system_msg = _build_system_prompt()

    accumulated_chunks: list[dict] = []
    tool_log: list[dict] = []
    web_searches = 0
    consecutive_empty = 0
    final_output = InvestigationActionsOutput(
        status="partially_answered",
        confidence="insufficient",
        answer="",
    )

    prompt = _build_first_prompt(claim, accumulated_chunks)

    for iteration in range(max_iterations):
        try:
            output: InvestigationActionsOutput = await acall_llm(
                prompt,
                system_msg=system_msg,
                model=model,
                output_schema=InvestigationActionsOutput,
                max_tokens=DEFAULT_MAX_TOKENS,
                config=config,
            )
        except Exception as exc:
            logger.warning("run_investigation acall_llm failed iteration %d: %s", iteration, exc)
            break

        final_output = output
        actions = [a.model_dump() for a in (output.next_actions or [])]

        tool_log.append({
            "iteration": iteration,
            "status": output.status,
            "action_count": len(actions),
        })

        # L4 gate: if coverage audit enabled and final, validate before accepting answer
        if output.status in _TERMINAL_STATUSES:
            if not _validate_coverage_audit(output):
                logger.debug("L4 coverage audit failed at iteration %d, continuing", iteration)
                # Force one more iteration to address gaps
                actions = [{
                    "tool": "search_investment",
                    "query": "financial performance milestone delivery partnership evidence quality",
                }]
            else:
                break

        if not actions:
            break

        new_chunks, wc = await _execute_actions(
            actions, tools, inv_id, bow_id, model
        )
        web_searches += wc
        new_chunks = _dedup_chunks(new_chunks, accumulated_chunks)

        if not new_chunks:
            consecutive_empty += 1
            if consecutive_empty >= CONSECUTIVE_EMPTY_THRESHOLD:
                logger.debug(
                    "run_investigation: %d consecutive empty rounds for %s, stopping",
                    consecutive_empty,
                    link_id,
                )
                break
        else:
            consecutive_empty = 0
            accumulated_chunks.extend(new_chunks)

        prompt = _build_iteration_prompt(
            claim, new_chunks, accumulated_chunks, output, iteration + 1, max_iterations
        )

    elapsed = time.time() - t0

    # Build annotated excerpts with credibility tier; dedup by 80-char prefix
    _CREDIBILITY_TIER = {
        "progress_report": "tier1_primary",
        "budget": "tier1_primary",
        "investment_document": "tier1_primary",
        "proposal": "tier2_secondary",
        "presentation": "tier2_secondary",
        "strategy": "tier3_context",
        "external": "tier3_context",
    }
    seen_prefixes: set[str] = set()
    annotated_excerpts: list[dict] = []
    for c in accumulated_chunks:
        text = c.get("text", "")
        prefix = text[:80]
        if prefix in seen_prefixes:
            continue
        seen_prefixes.add(prefix)
        doc_type = c.get("doc_type", "")
        credibility_tier = _CREDIBILITY_TIER.get(doc_type, "tier2_secondary")
        import re as _re_inv
        numerical_facts = _re_inv.findall(r'\$[\d,]+(?:\.\d+)?(?:\s*[MBK])?|\d+(?:\.\d+)?%', text)
        annotated_excerpts.append({
            "text": text[:500],
            "source": c.get("filename", c.get("file_id", "")),
            "page": c.get("page_start", 0),
            "significance": "supporting" if final_output.status == "answered" else "background",
            "numerical_facts": numerical_facts[:5],
            "credibility_tier": credibility_tier,
            "inv_id": inv_id,
            "scope_id": scope_id,
            "link_id": link_id,
        })

    return InvestigationResult(
        findings={
            "link_id": link_id,
            "status": final_output.status,
            "confidence": final_output.confidence,
            "evidence_refs": final_output.evidence_refs,
        },
        overall_assessment={
            "status": final_output.status,
            "confidence": final_output.confidence,
        },
        prose=final_output.answer,
        tool_log=tool_log,
        source_index=[
            {"ref": f"§{i+1:04d}", "filename": c.get("filename", "?"), "file_id": c.get("file_id", "")}
            for i, c in enumerate(accumulated_chunks)
        ],
        documents_read=list({c.get("file_id", "") for c in accumulated_chunks if c.get("file_id")}),
        annotated_excerpts=annotated_excerpts,
        iterations=len(tool_log),
        total_chunks_retrieved=len(accumulated_chunks),
        web_searches=web_searches,
        elapsed_seconds=elapsed,
        model=model,
        terminal_status=final_output.status,
    )
