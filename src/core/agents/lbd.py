"""LBD (Literature-Based Discovery) agent — Swanson's ABC model.

Finds indirect A→B→C connections between concepts not co-cited in literature.
Never raises — returns an error dict on failure.
"""
from __future__ import annotations

import asyncio
import logging
import re

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel

from src.config import DEFAULT_RESEARCH_MODEL, LBD_TIMEOUT_SECONDS
from src.core.agents.asta import AstaClient
from src.core.llm_utils import acall_llm

logger = logging.getLogger(__name__)


class LBDResult(BaseModel):
    task_id: str
    query: str
    thesis: str
    concepts: list[str] = []
    papers: list[dict] = []
    success: bool = True
    error_message: str | None = None


async def run(
    task_id: str,
    query: str,
    linked_scope: str = "",
    priority: str = "",
    *,
    model: str | None = None,
    timeout: int | None = None,
    config: RunnableConfig | None = None,
) -> dict:
    """Run literature-based discovery. Never raises."""
    research_model = model or DEFAULT_RESEARCH_MODEL
    max_secs = timeout or LBD_TIMEOUT_SECONDS

    try:
        # asyncio-APPROVED-3: wait_for wraps single external LBD pipeline call with timeout
        result = await asyncio.wait_for(
            _run_lbd(task_id, query, model=research_model, config=config),
            timeout=max_secs,
        )
    # asyncio-APPROVED-3: wait_for wraps single external LBD pipeline call with timeout
    except asyncio.TimeoutError:
        result = LBDResult(
            task_id=task_id,
            query=query,
            thesis="",
            success=False,
            error_message=f"LBD timed out after {max_secs}s",
        )
    except Exception as exc:
        logger.error("LBD failed %s: %s", task_id, exc)
        result = LBDResult(
            task_id=task_id,
            query=query,
            thesis="",
            success=False,
            error_message=str(exc),
        )

    return {
        "task_id": task_id,
        "task_type": "lbd",
        "linked_scope": linked_scope,
        "query": query,
        "thesis": result.thesis,
        "concepts": result.concepts,
        "results": result.papers,
        "success": result.success,
        "error_message": result.error_message,
    }


async def _run_lbd(task_id: str, query: str, *, model: str, config: RunnableConfig | None = None) -> LBDResult:
    from src.prompts.research_prompts import (
        LBD_CONCEPT_SYSTEM,
        LBD_CONCEPT_TEMPLATE,
        LBD_SYNTHESIS_SYSTEM,
        LBD_SYNTHESIS_TEMPLATE,
    )

    asta = AstaClient()

    # Step 1: Extract A-terms from query
    concept_response = await acall_llm(
        LBD_CONCEPT_TEMPLATE.format(query=query),
        system_msg=LBD_CONCEPT_SYSTEM,
        model=model,
        config=config,
    )
    a_terms = _parse_terms(concept_response) or [query]

    # Step 2: Search A→B (what does A connect to?)
    # asyncio-APPROVED-2: concurrent HTTP — parallel Asta searches for A-terms
    ab_results = await asyncio.gather(
        *[asta.search(term, max_results=15) for term in a_terms[:3]],
        return_exceptions=True,
    )
    ab_papers: list[dict] = []
    for r in ab_results:
        if not isinstance(r, Exception):
            ab_papers.extend(r)

    # Extract B-terms from A→B papers
    b_terms: list[str] = []
    if ab_papers:
        ab_context = _papers_to_context(ab_papers[:20])
        b_response = await acall_llm(
            LBD_CONCEPT_TEMPLATE.format(
                query=f"intermediary concepts bridging: {query}\n\nEvidence:\n{ab_context}"
            ),
            system_msg=LBD_CONCEPT_SYSTEM,
            model=model,
            config=config,
        )
        b_terms = _parse_terms(b_response)

    # Step 3: Search B→C
    bc_papers: list[dict] = []
    if b_terms:
        # asyncio-APPROVED-2: concurrent HTTP — parallel Asta searches for B-terms
        bc_results = await asyncio.gather(
            *[asta.search(f"{term} {query}", max_results=10) for term in b_terms[:3]],
            return_exceptions=True,
        )
        for r in bc_results:
            if not isinstance(r, Exception):
                bc_papers.extend(r)

    all_papers = _deduplicate(ab_papers + bc_papers)
    all_concepts = list(dict.fromkeys(a_terms + b_terms))  # preserve order, deduplicate

    context = _papers_to_context(all_papers[:25])
    thesis = await acall_llm(
        LBD_SYNTHESIS_TEMPLATE.format(
            query=query,
            a_terms=", ".join(a_terms),
            b_terms=", ".join(b_terms) if b_terms else "(none identified)",
            papers=context,
        ),
        system_msg=LBD_SYNTHESIS_SYSTEM,
        model=model,
        max_tokens=2048,
        config=config,
    )

    return LBDResult(
        task_id=task_id,
        query=query,
        thesis=thesis,
        concepts=all_concepts[:20],
        papers=all_papers[:25],
        success=True,
    )


def _parse_terms(response: str) -> list[str]:
    """Extract comma- or newline-separated terms from an LLM response."""
    match = re.search(r'\[([^\]]+)\]', response)
    if match:
        parts = [p.strip().strip('"\'') for p in match.group(1).split(',')]
        return [p for p in parts if p]
    parts = re.split(r'[,\n]+', response)
    return [p.strip().strip('"\'- ') for p in parts if p.strip()][:10]


def _papers_to_context(papers: list[dict]) -> str:
    lines: list[str] = []
    for i, p in enumerate(papers, 1):
        title = p.get("title", "")
        year = p.get("year", "?")
        abstract = (p.get("abstract") or "")[:250]
        lines.append(f"[{i}] {title} ({year}): {abstract}")
    return "\n\n".join(lines)


def _deduplicate(papers: list[dict]) -> list[dict]:
    seen: set[str] = set()
    result: list[dict] = []
    for p in papers:
        key = (p.get("title") or "").lower().strip()[:80]
        if key and key not in seen:
            seen.add(key)
            result.append(p)
    return result
