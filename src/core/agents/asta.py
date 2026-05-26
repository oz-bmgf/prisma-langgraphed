"""Async Asta (Semantic Scholar MCP) client with Semantic Scholar public API fallback.

Primary path  : POST JSON-RPC 2.0 to ASTA_ENDPOINT with Bearer token.
                Uses MCP Streamable HTTP — requires Accept: application/json, text/event-stream.
                Tool: search_papers_by_relevance (keyword=), returns paperId+title only.
                IDs are then enriched via SS batch API to add abstract/authors/year.
Fallback path : Semantic Scholar public graph API (no key required).

Returns paper dicts with paperId, title, year, authors, abstract, url, source,
externalIds — same schema as OpenAlexClient.search().
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request

import httpx

from src.config import ASTA_API_KEY, ASTA_ENDPOINT

logger = logging.getLogger(__name__)

_SS_SEARCH_BASE = "https://api.semanticscholar.org/graph/v1/paper/search"
_SS_BATCH_BASE = "https://api.semanticscholar.org/graph/v1/paper/batch"
_SS_FIELDS = "paperId,title,authors,year,abstract,externalIds,url"


def _parse_sse(sse_body: str) -> list[dict]:
    """Extract JSON objects from SSE (text/event-stream) response body."""
    events = []
    for line in sse_body.splitlines():
        if line.startswith("data: "):
            try:
                events.append(json.loads(line[6:]))
            except json.JSONDecodeError:
                pass
    return events


class AstaClient:
    def __init__(
        self,
        api_key: str | None = None,
        endpoint: str | None = None,
    ) -> None:
        self._api_key = api_key or ASTA_API_KEY
        self._endpoint = endpoint or ASTA_ENDPOINT

    async def search(self, query: str, max_results: int = 50) -> list[dict]:
        """Search for papers. Falls back to Semantic Scholar if Asta is unavailable."""
        if self._api_key:
            try:
                results = await self._search_asta(query, max_results)
                if results:
                    return results
                # Empty result from ASTA — fall through to SS for better recall
                logger.debug("ASTA returned 0 results, falling back to Semantic Scholar")
            except Exception as exc:
                logger.warning(
                    "Asta MCP search failed, falling back to Semantic Scholar: %s", exc
                )
        return await self._search_semantic_scholar(query, max_results)

    async def _search_asta(self, query: str, max_results: int) -> list[dict]:
        # MCP Streamable HTTP requires both application/json and text/event-stream
        # in the Accept header; omitting text/event-stream causes a 406 response.
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "search_papers_by_relevance",
                "arguments": {"keyword": query, "limit": min(max_results, 100)},
            },
        }
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "Authorization": f"Bearer {self._api_key}",
        }
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(self._endpoint, json=payload, headers=headers)
            resp.raise_for_status()
            sse_body = resp.text  # Content-Type is text/event-stream

        paper_ids: list[str] = []
        id_to_title: dict[str, str] = {}

        for event in _parse_sse(sse_body):
            result = event.get("result") or {}
            if result.get("isError"):
                raise RuntimeError(f"ASTA tool error: {result.get('content')}")
            for item in result.get("content") or []:
                if item.get("type") != "text":
                    continue
                try:
                    paper = json.loads(item["text"])
                except (json.JSONDecodeError, KeyError):
                    continue
                pid = paper.get("paperId", "")
                if pid and pid not in id_to_title:
                    paper_ids.append(pid)
                    id_to_title[pid] = paper.get("title", "")
            if len(paper_ids) >= max_results:
                break

        if not paper_ids:
            return []

        # ASTA only returns paperId+title; enrich via SS batch API for abstracts/authors/year
        return await self._enrich_by_ids(paper_ids[:max_results], id_to_title)

    async def _enrich_by_ids(
        self, paper_ids: list[str], id_to_title: dict[str, str]
    ) -> list[dict]:
        """Fetch full metadata (abstract, authors, year) for paper IDs via SS batch API."""
        def _fetch() -> list[dict]:
            body = json.dumps({"ids": paper_ids}).encode()
            req = urllib.request.Request(
                f"{_SS_BATCH_BASE}?fields={_SS_FIELDS}",
                data=body,
                headers={
                    "Content-Type": "application/json",
                    "User-Agent": "nqpr-pipeline/1.0",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=20) as resp:
                return json.loads(resp.read())

        max_attempts = 4
        for attempt in range(max_attempts):
            try:
                # asyncio-APPROVED-1: to_thread wraps blocking urllib.request call
                papers = await asyncio.to_thread(_fetch)
                if not isinstance(papers, list):
                    papers = []
                break
            except urllib.error.HTTPError as exc:
                if exc.code == 429 and attempt < max_attempts - 1:
                    wait = 2 ** attempt  # 1s, 2s, 4s
                    logger.warning(
                        "ASTA SS-batch enrichment rate-limited (429); retrying in %ds (attempt %d/%d)",
                        wait, attempt + 1, max_attempts,
                    )
                    # asyncio-APPROVED-1: to_thread wraps blocking time.sleep call
                    await asyncio.to_thread(time.sleep, wait)
                    continue
                logger.warning("ASTA SS-batch enrichment failed: %s", exc)
                papers = [{"paperId": pid, "title": id_to_title.get(pid, "")} for pid in paper_ids]
                break
            except Exception as exc:
                logger.warning("ASTA SS-batch enrichment failed: %s", exc)
                papers = [{"paperId": pid, "title": id_to_title.get(pid, "")} for pid in paper_ids]
                break

        return [_normalize_ss_paper(p) for p in papers if p]

    async def _search_semantic_scholar(self, query: str, max_results: int) -> list[dict]:
        def _fetch(q: str, limit: int) -> list[dict]:
            url = (
                f"{_SS_SEARCH_BASE}"
                f"?query={urllib.parse.quote(q)}"
                f"&fields={_SS_FIELDS}"
                f"&limit={min(limit, 100)}"
            )
            req = urllib.request.Request(
                url, headers={"User-Agent": "nqpr-pipeline/1.0"}
            )
            try:
                with urllib.request.urlopen(req, timeout=20) as resp:
                    return json.loads(resp.read()).get("data") or []
            except urllib.error.HTTPError as exc:
                if exc.code == 429:
                    logger.warning("Semantic Scholar rate-limited (429); returning empty")
                    return []
                raise

        # asyncio-APPROVED-1: to_thread wraps blocking urllib.request call
        papers = await asyncio.to_thread(_fetch, query, max_results)
        return [_normalize_ss_paper(p) for p in papers[:max_results]]


def _normalize_ss_paper(p: dict) -> dict:
    authors_raw = p.get("authors") or []
    if isinstance(authors_raw, list):
        authors = ", ".join(a.get("name", "") for a in authors_raw[:5])
    else:
        authors = str(authors_raw)
    url = p.get("url") or ""
    return {
        "paperId": p.get("paperId", ""),
        "title": p.get("title", ""),
        "year": p.get("year"),
        "authors": authors,
        "abstract": (p.get("abstract") or "")[:1000],
        "url": url,
        "source": url,
        "externalIds": p.get("externalIds") or {},
    }
