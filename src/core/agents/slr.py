"""SLR (Systematic Literature Review) agent.

Search OpenAlex and Asta concurrently → deduplicate by title → synthesise
thesis via acall_llm.

Never raises — returns an error dict on failure.
"""
from __future__ import annotations

import asyncio
import logging

from pydantic import BaseModel

from langchain_core.runnables import RunnableConfig

from src.config import DEFAULT_RESEARCH_MODEL, SLR_TIMEOUT_SECONDS
from src.core.agents.asta import AstaClient
from src.core.agents.openalex import OpenAlexClient
from src.core.llm_utils import acall_llm

logger = logging.getLogger(__name__)


class SLRResult(BaseModel):
    task_id: str
    query: str
    thesis: str
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
    """Run systematic literature review. Never raises."""
    research_model = model or DEFAULT_RESEARCH_MODEL
    max_secs = timeout or SLR_TIMEOUT_SECONDS

    try:
        # asyncio-APPROVED-3: wait_for wraps single external SLR pipeline call with timeout
        result = await asyncio.wait_for(
            _run_slr(task_id, query, model=research_model, config=config),
            timeout=max_secs,
        )
    # asyncio-APPROVED-3: wait_for wraps single external SLR pipeline call with timeout
    except asyncio.TimeoutError:
        result = SLRResult(
            task_id=task_id,
            query=query,
            thesis="",
            success=False,
            error_message=f"SLR timed out after {max_secs}s",
        )
    except Exception as exc:
        logger.error("SLR failed %s: %s", task_id, exc)
        result = SLRResult(
            task_id=task_id,
            query=query,
            thesis="",
            success=False,
            error_message=str(exc),
        )

    return {
        "task_id": task_id,
        "task_type": "slr",
        "linked_scope": linked_scope,
        "query": query,
        "thesis": result.thesis,
        "results": result.papers,
        "success": result.success,
        "error_message": result.error_message,
    }


async def _run_slr(task_id: str, query: str, *, model: str, config: RunnableConfig | None = None) -> SLRResult:
    from src.prompts.research_prompts import SLR_SYNTHESIS_SYSTEM, SLR_SYNTHESIS_TEMPLATE

    openalex = OpenAlexClient()
    asta = AstaClient()

    # asyncio-APPROVED-2: concurrent HTTP — OpenAlex and Asta searches in parallel
    openalex_papers, asta_papers = await asyncio.gather(
        openalex.search(query, max_results=20),
        asta.search(query, max_results=20),
        return_exceptions=True,
    )

    papers: list[dict] = []
    seen_titles: set[str] = set()

    for source in (openalex_papers, asta_papers):
        if isinstance(source, Exception):
            logger.warning("SLR search source failed: %s", source)
            continue
        for p in source:
            title_key = (p.get("title") or "").lower().strip()[:80]
            if title_key and title_key not in seen_titles:
                seen_titles.add(title_key)
                papers.append(p)

    if not papers:
        return SLRResult(
            task_id=task_id,
            query=query,
            thesis="No papers found for this query.",
            papers=[],
            success=True,
        )

    context_lines: list[str] = []
    for i, p in enumerate(papers[:30], 1):
        authors = p.get("authors", "")
        year = p.get("year", "?")
        title = p.get("title", "")
        abstract = (p.get("abstract") or "")[:300]
        context_lines.append(f"[{i}] {title} ({authors}, {year})\n{abstract}")

    context = "\n\n".join(context_lines)
    prompt = SLR_SYNTHESIS_TEMPLATE.format(query=query, papers=context)

    thesis = await acall_llm(
        prompt,
        system_msg=SLR_SYNTHESIS_SYSTEM,
        model=model,
        max_tokens=2048,
        config=config,
    )

    return SLRResult(
        task_id=task_id,
        query=query,
        thesis=thesis,
        papers=papers[:30],
        success=True,
    )
