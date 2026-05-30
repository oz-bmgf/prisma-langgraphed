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
    INVESTIGATION_USE_RESPONSES_API,
    MAX_INVESTIGATION_ITERATIONS,
)
from src.core.evidence_model import InvestigationResult
from src.core.llm_utils import acall_llm, acall_structured
from src.core.output_schemas import InvestigationActionsOutput
from src.tools.investigation_tools import (
    compute,
    read_document,
    search_bow,
    search_investment,
    search_policy,
    search_portfolio,
    search_science,
    search_web,
)
from src.tools.collection_tools import read_section
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
    "search_portfolio",
    "search_bow",
    "search_science",
    "search_policy",
    "search_doc_type",
    "search_all",
    "search_web",
    "read_pages",
    "read_document",
    "read_section",
    "compute",
    # submit_findings excluded — loop terminates via status field, not sentinel
})


_COLLECTION_TOOLS = frozenset({
    "search_investment",
    "search_portfolio",
    "search_bow",
    "search_science",
    "search_policy",
    "search_doc_type",
    "search_all",
})


async def _execute_actions(
    actions: list[dict],
    config: Any,
    inv_id: str,
    bow_id: str,
    model: str,
    facts: Any = None,
) -> tuple[list[dict], int]:
    """Async tool dispatcher — fan out tool actions concurrently via native @tool.ainvoke.

    Returns (new_chunks, web_search_count). Tracing is handled inside each
    @tool function via append_to_buffer. When search_backend is absent from
    config, collection search actions return empty results gracefully.
    """
    if not actions:
        return [], 0

    base_configurable: dict = dict(((config or {}).get("configurable") or {}))
    has_backend = base_configurable.get("search_backend") is not None

    # Per-action enriched configs inject inv_id / bow_id so the @tool functions
    # apply the right filter without needing per-call function arguments.
    # Top-level config keys (callbacks, run_id, metadata) are preserved so tool
    # spans remain attached to the parent LangGraph trace in OTEL.
    _inv_config = {**(config or {}), "configurable": {**base_configurable, "inv_id": inv_id, "bow_id": None}}
    _bow_config = {**(config or {}), "configurable": {**base_configurable, "inv_id": None, "bow_id": bow_id}}

    async def _dispatch_one(action: dict) -> tuple[list[dict], int]:
        tool_name = action.get("tool", "")
        query = action.get("query", "")
        if not query or tool_name not in _SUPPORTED_TOOLS:
            return [], 0
        if not has_backend and tool_name in _COLLECTION_TOOLS:
            return [], 0

        chunks: list[dict] = []
        web_count = 0
        try:
            if tool_name == "search_investment":
                text = await search_investment.ainvoke({"query": query}, config=_inv_config)
                chunks.append({"text": text, "file_id": f"inv:{query[:60]}", "filename": "collection search", "collection": "investment", "doc_type": "investment"})

            elif tool_name == "search_portfolio":
                collection = action.get("collection") or None
                text = await search_portfolio.ainvoke({"query": query, "collection": collection}, config=config)
                chunks.append({"text": text, "file_id": f"portfolio:{query[:60]}", "filename": "portfolio search", "collection": collection or "all", "doc_type": "all"})

            elif tool_name == "search_bow" and bow_id:
                text = await search_bow.ainvoke({"query": query}, config=_bow_config)
                chunks.append({"text": text, "file_id": f"bow:{query[:60]}", "filename": "BOW search", "collection": "investment", "doc_type": "investment"})

            elif tool_name == "search_science":
                text = await search_science.ainvoke({"query": query}, config=config)
                chunks.append({"text": text, "file_id": f"science:{query[:60]}", "filename": "science search", "collection": "strategy", "doc_type": "science"})

            elif tool_name == "search_policy":
                text = await search_policy.ainvoke({"query": query}, config=config)
                chunks.append({"text": text, "file_id": f"policy:{query[:60]}", "filename": "policy search", "collection": "strategy", "doc_type": "policy"})

            elif tool_name == "search_doc_type":
                doc_type = action.get("doc_type") or None
                text = await search_investment.ainvoke(
                    {"query": query, "doc_type": doc_type, "top_k": 8},
                    config=_inv_config,
                )
                chunks.append({"text": text, "file_id": f"inv:{query[:60]}", "filename": "collection search", "collection": "investment", "doc_type": doc_type or "investment"})

            elif tool_name == "search_all":
                text = await search_portfolio.ainvoke({"query": query}, config=config)
                chunks.append({"text": text, "file_id": f"all:{query[:60]}", "filename": "portfolio search", "collection": "investment", "doc_type": "all"})

            elif tool_name == "search_web":
                text = await search_web.ainvoke(
                    {"query": query, "rationale": action.get("rationale") or f"evidence for: {query}"},
                    config=config,
                )
                # F-056: stub text when Tavily absent must not count as evidence
                if text and not text.startswith("[web search not configured"):
                    chunks.append({"text": text, "file_id": f"web:{query[:40]}", "filename": "web search", "collection": "web", "doc_type": "web"})
                    web_count = 1

            elif tool_name in ("read_pages", "read_document"):
                file_id = action.get("file_id", "")
                section_id = action.get("section_id") or action.get("query") if tool_name == "read_document" and not action.get("file_id") else None
                page_start = int(action.get("page_start", 1))
                page_end = min(int(action.get("page_end", page_start + 10)), page_start + 20)
                if file_id:
                    text = await read_document.ainvoke(
                        {"file_id": file_id, "page_start": page_start, "page_end": page_end},
                        config=config,
                    )
                    if text and len(text.strip()) > 50:
                        chunks.append({"text": text, "file_id": file_id, "filename": f"{file_id} pp{page_start}-{page_end}", "collection": "investment"})

            elif tool_name == "read_section":
                file_id = action.get("file_id", "")
                section_id = action.get("section_id") or action.get("query", "")
                if file_id and section_id:
                    text = await read_section.ainvoke(
                        {"file_id": file_id, "section_id": section_id},
                        config=config,
                    )
                    if text and len(text.strip()) > 50:
                        chunks.append({"text": text, "file_id": file_id, "filename": f"{file_id}§{section_id[:30]}", "collection": "investment"})

            elif tool_name == "compute" and facts:
                text = await compute.ainvoke(
                    {"question": query, "data": str(facts)[:2000]},
                    config=config,
                )
                if text:
                    chunks.append({"text": text, "file_id": "computed", "filename": "compute tool", "collection": "computed"})

        except Exception as exc:
            logger.warning("_dispatch_one %s failed: %s", tool_name, str(exc)[:80])

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
    if INVESTIGATION_L4_COVERAGE_AUDIT:
        # F-054: inject checklist on EVERY iteration (not just last 2) so the model
        # is guided throughout, not just near the end of the loop.
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


def _is_openai_model(model: str) -> bool:
    """Return True when model is an OpenAI Responses-API-compatible model."""
    return model.startswith(("gpt-", "o1-", "o3-", "o4-"))


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
# Orphan 10 — Stateful OpenAI Responses API loop
# ---------------------------------------------------------------------------

async def _run_openai_responses_loop(
    link_id: str,
    inv_id: str,
    bow_id: str,
    scope_id: str,
    claim: dict,
    model: str,
    config: Any,
    max_iterations: int,
) -> "InvestigationResult":
    """Stateful investigation loop via OpenAI Responses API with previous_response_id.

    Matches old investigation_loop._run_openai_loop(): each iteration continues
    a persistent OpenAI Responses conversation thread instead of re-sending full
    accumulated context. This reduces token cost and preserves reasoning continuity.

    Falls back to the stateless acall_structured path on any Responses API error.
    """
    import asyncio as _asyncio
    import json as _json

    system_msg = _build_system_prompt()
    t0 = time.time()
    accumulated_chunks: list[dict] = []
    tool_log: list[dict] = []
    web_searches = 0
    consecutive_empty = 0
    prev_response_id: str | None = None

    # Build INVESTIGATION_TOOLS in OpenAI function-call format
    def _tool_schema(t: Any) -> dict:
        schema = t.args_schema.model_json_schema() if hasattr(t, "args_schema") else {}
        return {
            "type": "function",
            "name": t.name,
            "description": (t.description or "")[:500],
            "parameters": {
                "type": "object",
                "properties": {
                    k: v for k, v in schema.get("properties", {}).items()
                    if k != "config"
                },
                "required": [r for r in schema.get("required", []) if r != "config"],
                "additionalProperties": False,
            },
            "strict": False,
        }

    from src.tools.investigation_tools import INVESTIGATION_TOOLS
    oai_tools = [_tool_schema(t) for t in INVESTIGATION_TOOLS]

    try:
        from openai import OpenAI
        client = OpenAI()
    except Exception as exc:
        logger.warning("Responses API unavailable (%s); falling back to stateless loop", exc)
        return await _run_stateless_loop(
            link_id, inv_id, bow_id, scope_id, claim, model, config, max_iterations
        )

    first_prompt = _build_first_prompt(claim, [])
    initial_input = [
        {"role": "user", "content": [{"type": "input_text", "text": first_prompt}]}
    ]

    def _create_response(input_data: Any, prev_id: str | None) -> Any:
        kwargs: dict = dict(
            model=model,
            tools=oai_tools,
            instructions=system_msg,
        )
        if prev_id:
            kwargs["previous_response_id"] = prev_id
            kwargs["input"] = input_data   # tool outputs for this turn
        else:
            kwargs["input"] = input_data   # initial user message
        return client.responses.create(**kwargs)

    final_output = InvestigationActionsOutput(
        status="partially_answered", confidence="insufficient", answer="",
    )

    for iteration in range(max_iterations):
        try:
            if iteration == 0:
                response = await _asyncio.to_thread(_create_response, initial_input, None)
            else:
                response = await _asyncio.to_thread(_create_response, tool_outputs_for_next, prev_response_id)
            prev_response_id = response.id
        except Exception as exc:
            logger.warning("Responses API call failed at iteration %d: %s; switching to stateless", iteration, exc)
            break

        # Parse output items
        actions: list[dict] = []
        tool_calls: list[Any] = []
        prose_parts: list[str] = []

        for item in response.output:
            if item.type == "message":
                for block in item.content:
                    if hasattr(block, "text"):
                        prose_parts.append(block.text)
            elif item.type == "function_call":
                tool_calls.append(item)

        prose = "\n".join(prose_parts)

        # Try to extract structured output from prose
        try:
            import re as _re
            m = _re.search(r'\{[^{}]*"status"[^{}]*\}', prose, _re.DOTALL)
            if m:
                parsed = _json.loads(m.group(0))
                final_output = InvestigationActionsOutput(
                    status=parsed.get("status", "partially_answered"),
                    confidence=parsed.get("confidence", "low"),
                    answer=parsed.get("answer", prose[:600]),
                    evidence_refs=parsed.get("evidence_refs", []),
                )
        except Exception:
            final_output = InvestigationActionsOutput(
                status="partially_answered", confidence="low", answer=prose[:600],
            )

        # Convert function_calls to action dicts for _execute_actions
        for tc in tool_calls:
            try:
                args = _json.loads(tc.arguments) if isinstance(tc.arguments, str) else tc.arguments
                actions.append({"tool": tc.name, "query": args.get("query", ""), **args})
            except Exception:
                pass

        tool_log.append({
            "iteration": iteration,
            "status": final_output.status,
            "action_count": len(actions),
            "response_id": prev_response_id,
        })

        # Terminal status check
        if final_output.status in _TERMINAL_STATUSES:
            if not _validate_coverage_audit(final_output):
                actions = [{"tool": "search_investment",
                            "query": "financial performance milestone delivery evidence quality"}]
            else:
                break

        if not tool_calls and not actions:
            break

        # Execute tool actions
        new_chunks, wc = await _execute_actions(actions, config, inv_id, bow_id, model)
        web_searches += wc
        new_chunks = _dedup_chunks(new_chunks, accumulated_chunks)

        if not new_chunks:
            consecutive_empty += 1
            if consecutive_empty >= CONSECUTIVE_EMPTY_THRESHOLD:
                break
        else:
            consecutive_empty = 0
            accumulated_chunks.extend(new_chunks)

        # Build tool outputs for next Responses API turn
        tool_outputs_for_next = []
        for tc in tool_calls:
            # Find matching result text from executed actions
            result_text = f"(executed {tc.name}: {len(new_chunks)} chunks)"
            if new_chunks:
                result_text = "\n".join(c.get("text", "")[:300] for c in new_chunks[:5])
            tool_outputs_for_next.append({
                "type": "function_call_output",
                "call_id": getattr(tc, "call_id", tc.name),
                "output": result_text,
            })
        if not tool_outputs_for_next:
            break

    # Fall back to stateless path if no valid result
    if not final_output.answer and not accumulated_chunks:
        return await _run_stateless_loop(
            link_id, inv_id, bow_id, scope_id, claim, model, config, max_iterations
        )

    return _build_investigation_result(
        link_id, inv_id, scope_id, final_output, accumulated_chunks, tool_log, web_searches, time.time() - t0, model
    )


async def _run_stateless_loop(
    link_id: str,
    inv_id: str,
    bow_id: str,
    scope_id: str,
    claim: dict,
    model: str,
    config: Any,
    max_iterations: int,
) -> "InvestigationResult":
    """Stateless acall_structured investigation loop (non-OpenAI models or fallback)."""
    t0 = time.time()
    system_msg = _build_system_prompt()
    accumulated_chunks: list[dict] = []
    tool_log: list[dict] = []
    web_searches = 0
    consecutive_empty = 0
    final_output = InvestigationActionsOutput(
        status="partially_answered", confidence="insufficient", answer="",
    )
    prompt = _build_first_prompt(claim, accumulated_chunks)

    for iteration in range(max_iterations):
        try:
            output: InvestigationActionsOutput = await acall_structured(
                prompt,
                system_msg=system_msg,
                model=model,
                schema=InvestigationActionsOutput,
                max_tokens=DEFAULT_MAX_TOKENS,
                config=config,
            )
        except Exception as exc:
            logger.warning("run_investigation acall_structured failed iteration %d: %s", iteration, exc)
            break

        final_output = output
        actions = [a.model_dump() for a in (output.next_actions or [])]

        tool_log.append({
            "iteration": iteration,
            "status": output.status,
            "action_count": len(actions),
        })

        if output.status in _TERMINAL_STATUSES:
            if not _validate_coverage_audit(output):
                logger.debug("L4 coverage audit failed at iteration %d, continuing", iteration)
                actions = [{"tool": "search_investment",
                            "query": "financial performance milestone delivery partnership evidence quality"}]
            else:
                break

        if not actions:
            break

        new_chunks, wc = await _execute_actions(actions, config, inv_id, bow_id, model)
        web_searches += wc
        new_chunks = _dedup_chunks(new_chunks, accumulated_chunks)

        if not new_chunks:
            consecutive_empty += 1
            if consecutive_empty >= CONSECUTIVE_EMPTY_THRESHOLD:
                logger.debug(
                    "run_investigation: %d consecutive empty rounds for %s, stopping",
                    consecutive_empty, link_id,
                )
                break
        else:
            consecutive_empty = 0
            accumulated_chunks.extend(new_chunks)

        prompt = _build_iteration_prompt(
            claim, new_chunks, accumulated_chunks, output, iteration + 1, max_iterations
        )

    return _build_investigation_result(
        link_id, inv_id, scope_id, final_output, accumulated_chunks, tool_log, web_searches, time.time() - t0, model
    )


def _build_investigation_result(
    link_id: str,
    inv_id: str,
    scope_id: str,
    final_output: "InvestigationActionsOutput",
    accumulated_chunks: list[dict],
    tool_log: list[dict],
    web_searches: int,
    elapsed: float,
    model: str,
) -> "InvestigationResult":
    """Build InvestigationResult from accumulated loop state."""
    import re as _re_inv

    _CREDIBILITY_TIER = {
        "progress_report": "tier1_primary",
        "budget": "tier1_primary",
        "investment_document": "tier1_primary",
        "proposal": "tier2_secondary",
        "presentation": "tier2_secondary",
        "strategy": "tier3_context",
        "external": "tier3_context",
    }
    _answered = final_output.status == "answered"
    seen_prefixes: set[str] = set()
    annotated_excerpts: list[dict] = []
    ref_counter = 0
    for i, c in enumerate(accumulated_chunks):
        text = c.get("text", "")
        prefix = text[:80]
        if prefix in seen_prefixes:
            continue
        seen_prefixes.add(prefix)
        ref_counter += 1
        ref_id = f"§{ref_counter:04d}"
        doc_type = c.get("doc_type", "")
        source = c.get("filename", c.get("file_id", ""))
        credibility_tier = _CREDIBILITY_TIER.get(doc_type, "tier2_secondary")
        numerical_facts = _re_inv.findall(r'\$[\d,]+(?:\.\d+)?(?:\s*[MBK])?|\d+(?:\.\d+)?%', text)
        safe_link = link_id[:20].replace(" ", "_") if link_id else "lnk"
        annotated_excerpts.append({
            "ref_id": ref_id,
            "text": text[:500],
            "source": source,
            "credibility_tier": credibility_tier,
            "inv_id": inv_id,
            "scope_id": scope_id,
            "link_id": link_id,
            "file_id": c.get("file_id", ""),
            "page_start": c.get("page_start", 0),
            "page_end": c.get("page_end", 0),
            "excerpt_id": f"EX-{inv_id}-{safe_link}-{ref_counter:03d}",
            "source_file": source,
            "page": c.get("page_start", 0),
            "source_type": c.get("collection", doc_type),
            "type": "evidence" if _answered else "context",
            "quote": text[:500],
            "significance": "supporting" if _answered else "background",
            "context_needed": credibility_tier != "tier1_primary",
            "numerical_facts": numerical_facts[:5],
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
        # Routing fields for collect_link_assessments — without these, to_dict()
        # produces no scope_id and every assessment is silently dropped.
        scope_id=scope_id,
        link_id=link_id,
        inv_id=inv_id,
    )


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
    config: Any = None,
    max_iterations: int = MAX_INVESTIGATION_ITERATIONS,
) -> InvestigationResult:
    """Run a multi-turn tool-calling investigation of one causal link.

    Routes to the stateful OpenAI Responses API path (Orphan 10) when
    INVESTIGATION_USE_RESPONSES_API=true and the model is an OpenAI model;
    otherwise uses the stateless acall_structured path.

    Both paths stop on:
    - terminal status (answered / not_answerable / unresolved_conflict)
    - empty next_actions
    - CONSECUTIVE_EMPTY_THRESHOLD consecutive rounds with no new chunks
    - max_iterations reached
    """
    if INVESTIGATION_USE_RESPONSES_API and _is_openai_model(model):
        return await _run_openai_responses_loop(
            link_id, inv_id, bow_id, scope_id, claim, model, config, max_iterations
        )
    return await _run_stateless_loop(
        link_id, inv_id, bow_id, scope_id, claim, model, config, max_iterations
    )


async def _run_investigation_body(  # kept for backward compat references
    link_id: str,
    inv_id: str,
    bow_id: str,
    scope_id: str,
    claim: dict,
    model: str,
    *,
    config: Any = None,
    max_iterations: int = MAX_INVESTIGATION_ITERATIONS,
) -> InvestigationResult:
    return await run_investigation(
        link_id, inv_id, bow_id, scope_id, claim, model,
        config=config, max_iterations=max_iterations,
    )


# ---------------------------------------------------------------------------
# Old loop body (preserved below, now split into _run_stateless_loop + helpers)
# ---------------------------------------------------------------------------

