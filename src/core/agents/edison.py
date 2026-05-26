"""Edison literature retrieval client.

Wraps the proprietary edison_client SDK with lazy import and graceful failure
when the SDK or EDISON_PLATFORM_API_KEY is absent.

Never raises — returns a dict with status="no_api_key"|"error" on failure.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from pydantic import BaseModel

from src.config import EDISON_API_KEY, EDISON_TIMEOUT_SECONDS

logger = logging.getLogger(__name__)


class EdisonPaper(BaseModel):
    paperId: str = ""
    title: str = ""
    year: int | None = None
    authors: str = ""
    abstract: str = ""
    url: str = ""


class EdisonQueryResult(BaseModel):
    task_id: str
    query: str
    status: str  # "ok" | "error" | "no_api_key" | "no_response" | "no_evidence"
    papers: list[EdisonPaper] = []
    thesis: str = ""
    error_message: str | None = None


async def run(
    task_id: str,
    query: str,
    original_query: str = "",
    linked_scope: str = "",
    priority: str = "",
    *,
    timeout: int | None = None,
) -> dict:
    """Query Edison for academic literature. Never raises."""
    if not EDISON_API_KEY:
        logger.warning("Edison: EDISON_PLATFORM_API_KEY not set — skipping task %s", task_id)
        return {
            "task_id": task_id,
            "task_type": "edison",
            "linked_scope": linked_scope,
            "query": query,
            "original_query": original_query,
            "status": "no_api_key",
            "papers": [],
            "thesis": "",
            "error_message": None,
        }

    _timeout = timeout or EDISON_TIMEOUT_SECONDS
    try:
        # asyncio-APPROVED-3: wait_for wraps single external Edison SDK call with timeout
        result = await asyncio.wait_for(
            _query_edison(task_id, query),
            timeout=_timeout,
        )
    # asyncio-APPROVED-3: wait_for wraps single external Edison SDK call with timeout
    except asyncio.TimeoutError:
        logger.warning("Edison: query timed out for %s", task_id)
        result = EdisonQueryResult(
            task_id=task_id,
            query=query,
            status="error",
            error_message=f"timed out after {_timeout}s",
        )
    except Exception as exc:
        logger.error("Edison query failed for %s: %s", task_id, exc)
        result = EdisonQueryResult(
            task_id=task_id,
            query=query,
            status="error",
            error_message=str(exc),
        )

    return {
        "task_id": task_id,
        "task_type": "edison",
        "linked_scope": linked_scope,
        "query": query,
        "original_query": original_query,
        "status": result.status,
        "papers": [p.model_dump() for p in result.papers],
        "thesis": result.thesis,
        "error_message": result.error_message,
    }


async def _query_edison(task_id: str, query: str) -> EdisonQueryResult:
    try:
        from edison_client import EdisonClient, JobNames  # type: ignore[import]
    except ImportError:
        logger.warning("edison_client SDK not installed; returning no_api_key")
        return EdisonQueryResult(task_id=task_id, query=query, status="no_api_key")

    def _sync_query() -> Any:
        client = EdisonClient(api_key=EDISON_API_KEY)
        return client.query(query, job_type=JobNames.LITERATURE_HIGH)

    # asyncio-APPROVED-1: to_thread wraps blocking Edison SDK client.query call
    raw = await asyncio.to_thread(_sync_query)
    return _parse_edison_result(task_id, query, raw)


def _parse_edison_result(task_id: str, query: str, raw: Any) -> EdisonQueryResult:
    if raw is None:
        return EdisonQueryResult(task_id=task_id, query=query, status="no_response")

    raw_papers: list[Any] = []
    if hasattr(raw, "papers"):
        raw_papers = raw.papers or []
    elif hasattr(raw, "results"):
        raw_papers = raw.results or []
    elif isinstance(raw, dict):
        raw_papers = raw.get("papers") or raw.get("results") or []

    papers: list[EdisonPaper] = []
    for rp in raw_papers:
        if isinstance(rp, dict):
            papers.append(EdisonPaper(
                paperId=str(rp.get("paperId") or rp.get("id") or ""),
                title=rp.get("title") or "",
                year=rp.get("year"),
                authors=rp.get("authors") or "",
                abstract=(rp.get("abstract") or "")[:800],
                url=rp.get("url") or "",
            ))
        else:
            papers.append(EdisonPaper(
                paperId=str(getattr(rp, "paperId", "") or ""),
                title=getattr(rp, "title", "") or "",
                year=getattr(rp, "year", None),
                authors=getattr(rp, "authors", "") or "",
                abstract=(getattr(rp, "abstract", "") or "")[:800],
                url=getattr(rp, "url", "") or "",
            ))

    thesis = ""
    if hasattr(raw, "synthesis"):
        thesis = str(raw.synthesis or "")
    elif hasattr(raw, "executive_summary"):
        thesis = str(raw.executive_summary or "")
    elif isinstance(raw, dict):
        thesis = str(raw.get("synthesis") or raw.get("executive_summary") or "")

    status = "ok" if papers else "no_evidence"
    return EdisonQueryResult(
        task_id=task_id,
        query=query,
        status=status,
        papers=papers,
        thesis=thesis,
    )
