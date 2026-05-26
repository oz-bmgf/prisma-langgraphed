"""Science investigator — literature search for causal assumption validation.

investigate_science_question() drives the loop:
  - ASTA gate: must call search_asta at least once before status=evidence_gathered
  - ASTA soft cap: silently skips ASTA actions once asta_calls >= ASTA_SOFT_CAP
  - Consecutive empty check: 3 rounds with zero new chunks → insufficient_evidence
  - Tracks confirming_found / disconfirming_found from model self-report AND
    text heuristics on new chunks
  - Returns ScienceInvestigationResult

_execute_actions is imported from investigation (shared dispatcher, non-ASTA tools).
ASTA actions use the AstaClient (or no-op when asta_client=None).

No LangGraph imports.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from src.config import (
    ASTA_SOFT_CAP,
    CONSECUTIVE_EMPTY_THRESHOLD,
    DEFAULT_MAX_TOKENS,
    DEFAULT_SYNTHESIS_MODEL,
    SCIENCE_MAX_ITERATIONS,
)
from src.core.evidence_model import ScienceInvestigationResult
from src.core.investigation import _dedup_chunks, _execute_actions
from src.core.llm_utils import acall_llm
from src.core.output_schemas import ScienceActionsOutput
from src.prompts.tool_prompts import SCIENCE_INVESTIGATE_SYSTEM

logger = logging.getLogger(__name__)

_SCIENCE_TERMINAL = frozenset({"evidence_gathered", "insufficient_evidence", "blocked"})

# Heuristics for detecting confirming/disconfirming text in new chunks.
_CONFIRM_KEYWORDS = frozenset({
    "confirms", "supports", "demonstrates", "shows", "proves",
    "effective", "efficacious", "significant", "positive", "consistent with",
})
_DISCONFIRM_KEYWORDS = frozenset({
    "refutes", "contradicts", "challenges", "questions", "no evidence",
    "failed", "negative", "not significant", "inconsistent", "no effect",
})


def _has_text_evidence(text: str, keywords: frozenset[str]) -> bool:
    lower = text.lower()
    return any(kw in lower for kw in keywords)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


async def investigate_science_question(
    assumption_id: str,
    inv_id: str,
    bow_id: str,
    scope_id: str,
    question: str,
    *,
    asta_client: Any = None,
    tools: Any = None,
    model: str = DEFAULT_SYNTHESIS_MODEL,
    max_iterations: int = SCIENCE_MAX_ITERATIONS,
    asta_soft_cap: int = ASTA_SOFT_CAP,
) -> ScienceInvestigationResult:
    """Investigate one science assumption through ASTA + web search.

    ASTA gate enforcement: status=evidence_gathered is only accepted after
    search_asta has been called at least once. The gate note is injected into
    every iteration prompt until the gate is satisfied.

    Soft cap: once asta_calls >= asta_soft_cap, ASTA actions are silently
    dropped from the dispatch (does not prevent termination).

    Consecutive empty check: 3 rounds with zero new chunks forces
    terminal_status = "insufficient_evidence".
    """
    t0 = time.time()

    asta_called_ever = False
    confirming_found = False
    disconfirming_found = False
    asta_calls = 0
    web_calls = 0
    asta_hits: list[Any] = []
    blocked_items: list[str] = []
    accumulated_chunks: list[dict] = []
    consecutive_empty = 0

    final_output = ScienceActionsOutput(
        status="continue",
        confirming_evidence_found=False,
        disconfirming_evidence_found=False,
        answer="",
    )

    for iteration in range(max_iterations):
        gate_note = _build_gate_note(
            asta_called_ever, confirming_found, disconfirming_found
        )
        prompt = _build_prompt(
            question, assumption_id, accumulated_chunks, prev_output=final_output,
            iteration=iteration, gate_note=gate_note,
        )

        try:
            output: ScienceActionsOutput = await acall_llm(
                prompt,
                system_msg=SCIENCE_INVESTIGATE_SYSTEM,
                model=model,
                output_schema=ScienceActionsOutput,
                max_tokens=DEFAULT_MAX_TOKENS,
            )
        except Exception as exc:
            logger.warning("investigate_science_question LLM failed iteration %d: %s", iteration, exc)
            break

        final_output = output

        # Update gate flags from model self-report
        if output.confirming_evidence_found:
            confirming_found = True
        if output.disconfirming_evidence_found:
            disconfirming_found = True

        # ASTA gate check: evidence_gathered requires at least one ASTA call
        gate_blocked = output.status == "evidence_gathered" and not asta_called_ever
        if gate_blocked:
            logger.debug("Science ASTA gate blocked status=evidence_gathered at iteration %d", iteration)

        if output.status in _SCIENCE_TERMINAL and not gate_blocked:
            break

        actions = [a.model_dump() for a in (output.next_actions or [])]

        # When gate blocked and no actions returned, inject a forced ASTA search
        if not actions:
            if gate_blocked:
                actions = [{"tool": "search_asta", "query": question}]
            else:
                break

        # Split ASTA actions from non-ASTA actions
        asta_actions = [a for a in actions if a.get("tool") == "search_asta"]
        other_actions = [a for a in actions if a.get("tool") != "search_asta"]

        new_chunks: list[dict] = []
        wc = 0

        # Execute ASTA actions (subject to soft cap)
        if asta_actions and asta_calls < asta_soft_cap:
            n_asta = min(len(asta_actions), asta_soft_cap - asta_calls)
            for action in asta_actions[:n_asta]:
                asta_result = await _call_asta(action.get("query", ""), asta_client)
                asta_hits.extend(asta_result)
                asta_calls += 1
                asta_called_ever = True
                for hit in asta_result:
                    new_chunks.append({
                        "text": f"{hit.get('title', '')}. {hit.get('abstract', '')}",
                        "file_id": f"asta:{hit.get('paperId', '')}",
                        "filename": hit.get("title", "unknown paper"),
                        "collection": "science",
                        "doc_type": "science",
                    })
        elif asta_actions and asta_calls >= asta_soft_cap:
            logger.debug("ASTA soft cap reached (%d), skipping %d ASTA actions", asta_soft_cap, len(asta_actions))
            asta_called_ever = True

        # Execute non-ASTA actions via shared dispatcher
        if other_actions:
            other_chunks, wc = await _execute_actions(
                other_actions, tools, inv_id, bow_id, model
            )
            new_chunks.extend(other_chunks)
            web_calls += wc

        # Update confirming/disconfirming from text heuristics on new chunks
        for chunk in new_chunks:
            text = chunk.get("text", "")
            if _has_text_evidence(text, _CONFIRM_KEYWORDS):
                confirming_found = True
            if _has_text_evidence(text, _DISCONFIRM_KEYWORDS):
                disconfirming_found = True

        new_chunks = _dedup_chunks(new_chunks, accumulated_chunks)

        if not new_chunks:
            consecutive_empty += 1
            if consecutive_empty >= CONSECUTIVE_EMPTY_THRESHOLD:
                logger.debug(
                    "Science: %d consecutive empty rounds for %s, forcing insufficient_evidence",
                    consecutive_empty,
                    assumption_id,
                )
                final_output = ScienceActionsOutput(
                    status="insufficient_evidence",
                    confirming_evidence_found=confirming_found,
                    disconfirming_evidence_found=disconfirming_found,
                    answer=final_output.answer,
                )
                break
        else:
            consecutive_empty = 0
            accumulated_chunks.extend(new_chunks)

    elapsed = time.time() - t0
    return ScienceInvestigationResult(
        question_index=0,
        chunks=accumulated_chunks,
        asta_hits=asta_hits,
        blocked_items=blocked_items,
        iterations=max_iterations,
        asta_calls=asta_calls,
        web_calls=web_calls,
        terminal_status=final_output.status,
        elapsed_s=elapsed,
        answer=final_output.answer or "",
        question=question,
        scope_id=scope_id,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_gate_note(asta_called: bool, confirming: bool, disconfirming: bool) -> str:
    parts: list[str] = []
    if not asta_called:
        parts.append("GATE NOTE: search_asta has NOT been called yet — you MUST call it before status=evidence_gathered.")
    else:
        asta_status = "ASTA gate SATISFIED."
        if confirming and disconfirming:
            asta_status += " Both confirming and disconfirming evidence found."
        elif confirming:
            asta_status += " Confirming evidence found."
        elif disconfirming:
            asta_status += " Disconfirming evidence found."
        else:
            asta_status += " No confirming or disconfirming evidence found yet."
        parts.append(asta_status)
    return " | ".join(parts)


def _build_prompt(
    question: str,
    assumption_id: str,
    chunks: list[dict],
    *,
    prev_output: ScienceActionsOutput,
    iteration: int,
    gate_note: str,
) -> str:
    evidence_text = (
        "\n\n".join(
            f"[§{i+1:04d}] {c.get('filename', '?')}:\n{c.get('text', '')[:800]}"
            for i, c in enumerate(chunks[:30])
        )
        if chunks else "(no evidence collected yet)"
    )
    prev_answer = (prev_output.answer or "")[:500] if iteration > 0 else ""

    return (
        f"ASSUMPTION TO INVESTIGATE: {question}\n"
        f"Assumption ID: {assumption_id}\n\n"
        f"GATE STATUS: {gate_note}\n\n"
        f"{'PREVIOUS ANALYSIS: ' + prev_answer + chr(10) + chr(10) if prev_answer else ''}"
        f"ACCUMULATED EVIDENCE ({len(chunks)} items):\n{evidence_text}\n\n"
        f"Issue your next search actions. Set status=evidence_gathered only when "
        f"ASTA gate is satisfied and you have found relevant literature."
    )


async def _call_asta(query: str, asta_client: Any) -> list[dict]:
    if asta_client is None:
        return []
    import inspect
    try:
        search_fn = getattr(asta_client, "search", None)
        if search_fn is None:
            return []
        if inspect.iscoroutinefunction(search_fn):
            return await search_fn(query)
        # asyncio-APPROVED-1: to_thread wraps blocking ASTA HTTP call
        return await asyncio.to_thread(search_fn, query)
    except Exception as exc:
        logger.warning("_call_asta failed for query '%s': %s", query[:60], exc)
        return []
