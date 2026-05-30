"""Async Asta (Semantic Scholar MCP) client — snippet_search primary strategy.

Matches old prisma-ai-review AstaClient strategy="snippets":
  1. snippet_search via ASTA MCP endpoint — NL-friendly full-text passage
     search over 12M+ papers; returns ~500-word passage text per result.
  2. get_paper_batch via ASTA MCP endpoint — batch-enriches corpus IDs
     with DOI, full abstract, venue, open-access PDF URL.
  3. Deduplicates by corpus ID; keeps longer of snippet text vs abstract.

No Semantic Scholar public search or OpenAlex fallback in this client.
When ASTA_API_KEY is absent, search() returns []. The @tool layer in
science_tools.py:search_asta provides a Semantic Scholar public API
fallback for that case.

API: JSON-RPC 2.0 over SSE (MCP Streamable HTTP).
Endpoint: configured via ASTA_ENDPOINT (default https://asta-tools.allen.ai/mcp/v1).
"""
from __future__ import annotations

import asyncio
import json
import logging
import random
from typing import Any

import httpx

from src.config import (
    ASTA_API_KEY,
    ASTA_ENDPOINT,
    ASTA_MAX_RETRIES,
    ASTA_RETRY_BASE_DELAY,
    ASTA_TIMEOUT_SECONDS,
)

logger = logging.getLogger(__name__)

# Module-level result cache keyed by (query, max_results).
# Shared across all AstaClient instances (literature_tools.py constructs a new
# instance per tool call). Bounded at _CACHE_MAX to prevent unbounded growth
# over a long run. CPython dict ops are GIL-protected — no lock needed.
_search_cache: dict[tuple[str, int], list[dict]] = {}
_CACHE_MAX = 256


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
        """Search via snippet_search + get_paper_batch enrichment.

        Returns [] when ASTA_API_KEY is not set.
        Returns [] on any runtime failure (server error, timeout) after logging a warning.
        CancelledError from external task cancellation is NOT caught — it propagates.
        Caches results in the module-level _search_cache to suppress duplicate calls
        that arise when parallel LangGraph branches issue the same query.
        """
        if not self._api_key:
            return []
        cache_key = (query, max_results)
        if cache_key in _search_cache:
            logger.debug("ASTA cache hit for %r", query[:60])
            return _search_cache[cache_key]
        try:
            result = await self._snippet_search_and_enrich(query, max_results)
        except Exception as exc:
            logger.warning("ASTA search failed for %r: %s", query[:60], exc)
            return []
        if len(_search_cache) < _CACHE_MAX:
            _search_cache[cache_key] = result
        return result

    # ------------------------------------------------------------------
    # MCP transport
    # ------------------------------------------------------------------

    async def _call_mcp_tool(self, tool_name: str, arguments: dict) -> Any:
        """POST JSON-RPC 2.0 to the ASTA MCP endpoint; return parsed result.

        Handles SSE (text/event-stream) response format. Retries up to
        ASTA_MAX_RETRIES times on HTTP 429 with exponential backoff + jitter,
        honouring the Retry-After header when present. Returns the
        structuredContent dict if present, otherwise parses the first
        text content block as JSON.
        """
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": tool_name, "arguments": arguments},
        }
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "x-api-key": self._api_key,
        }
        for attempt in range(ASTA_MAX_RETRIES + 1):
            async with httpx.AsyncClient(timeout=ASTA_TIMEOUT_SECONDS) as client:
                # asyncio-APPROVED-3: wait_for wraps single external ASTA MCP HTTP call with total timeout
                # httpx read_timeout fires per-chunk, not per-request — SSE pings every ~15s keep it
                # alive indefinitely when the server stalls. wait_for enforces a hard ceiling on the
                # full SSE stream read so the call cannot hang longer than ASTA_TIMEOUT_SECONDS total.
                try:
                    resp = await asyncio.wait_for(
                        client.post(self._endpoint, json=payload, headers=headers),
                        timeout=ASTA_TIMEOUT_SECONDS,
                    )
                except asyncio.TimeoutError as exc:
                    raise TimeoutError(
                        f"ASTA {tool_name} timed out after {ASTA_TIMEOUT_SECONDS}s"
                    ) from exc

            if resp.status_code != 429:
                break
            if attempt >= ASTA_MAX_RETRIES:
                resp.raise_for_status()
            try:
                delay = min(float(resp.headers.get("Retry-After", "")), 60.0)
            except (ValueError, TypeError):
                delay = 0.0
            if delay < 1.0:
                delay = min(
                    ASTA_RETRY_BASE_DELAY * (2 ** attempt) + random.uniform(0.0, 1.0),
                    30.0,
                )
            logger.warning(
                "ASTA 429 on %s (attempt %d/%d) — backing off %.1fs",
                tool_name, attempt + 1, ASTA_MAX_RETRIES, delay,
            )
            # asyncio-APPROVED-4: sleep in ASTA 429 exponential-backoff retry
            await asyncio.sleep(delay)

        resp.raise_for_status()

        for event in _parse_sse(resp.text):
            result = event.get("result") or {}
            if result.get("isError"):
                raise RuntimeError(
                    f"ASTA tool error ({tool_name}): {result.get('content')}"
                )
            if "structuredContent" in result:
                return result["structuredContent"]
            for item in result.get("content") or []:
                if item.get("type") == "text":
                    try:
                        return json.loads(item["text"])
                    except (json.JSONDecodeError, KeyError):
                        pass
        return {}

    # ------------------------------------------------------------------
    # Core retrieval — snippet_search → get_paper_batch (OLD strategy)
    # ------------------------------------------------------------------

    async def _snippet_search_and_enrich(
        self,
        query: str,
        max_results: int,
    ) -> list[dict]:
        """Run snippet_search then batch-enrich via get_paper_batch.

        Mirrors old asta_client._search_snippets + _enrich_with_metadata:
        - Deduplicates by corpus ID (keeps best score + all snippet text)
        - Prefers longer of snippet text vs paper abstract after enrichment
        - Returns list[dict] with paperId/title/year/authors/abstract/externalIds
        """
        # Step 1: snippet_search — NL-friendly, returns passage text
        raw = await self._call_mcp_tool("snippet_search", {
            "query": query,
            "limit": min(max_results, 100),
        })

        data_list = self._extract_snippet_data(raw)

        # Step 2: Build per-paper dict; deduplicate by corpus ID
        papers: dict[str, dict] = {}  # corpus_id → result dict
        for entry in data_list:
            paper = entry.get("paper") or {}
            snippet = entry.get("snippet") or {}
            score = float(entry.get("score", 0.0))

            corpus_id = str(paper.get("corpusId", ""))
            if not corpus_id:
                continue

            snippet_text = snippet.get("text", "")

            if corpus_id not in papers:
                authors_raw = paper.get("authors", [])
                if authors_raw and isinstance(authors_raw[0], dict):
                    authors_str = ", ".join(a.get("name", "") for a in authors_raw)
                else:
                    authors_str = ", ".join(str(a) for a in authors_raw) if authors_raw else ""

                papers[corpus_id] = {
                    "paperId": corpus_id,
                    "title": paper.get("title", ""),
                    "year": paper.get("year"),
                    "authors": authors_str,
                    "abstract": snippet_text,   # snippet text as initial abstract
                    "url": paper.get("url", ""),
                    "externalIds": {},
                    "_score": score,
                }
            else:
                # Same paper appeared in multiple passages — append new text
                existing = papers[corpus_id]
                if snippet_text and snippet_text not in existing["abstract"]:
                    existing["abstract"] = existing["abstract"] + "\n\n" + snippet_text
                existing["_score"] = max(existing["_score"], score)

        if not papers:
            logger.info("ASTA snippet_search: 0 results for: %s", query[:60])
            return []

        # Step 3: Batch-enrich with DOI, full abstract, venue via get_paper_batch
        corpus_ids = list(papers.keys())
        try:
            batch_raw = await self._call_mcp_tool("get_paper_batch", {
                "ids": [f"CorpusId:{cid}" for cid in corpus_ids[:100]],
                "fields": (
                    "title,year,abstract,authors,venue,publicationDate,"
                    "url,isOpenAccess,openAccessPdf,externalIds"
                ),
            })
            batch_list = batch_raw.get("result", []) if isinstance(batch_raw, dict) else []

            for i, enriched in enumerate(batch_list or []):
                if not isinstance(enriched, dict):
                    continue
                # Correlate by corpusId or by position (batch preserves request order)
                cid = str(enriched.get("corpusId", ""))
                if not cid and i < len(corpus_ids):
                    cid = corpus_ids[i]
                if cid not in papers:
                    continue

                p = papers[cid]
                p["externalIds"] = enriched.get("externalIds") or {}

                # Keep longer of snippet text vs paper abstract — matches OLD logic
                paper_abstract = enriched.get("abstract") or ""
                if paper_abstract and len(paper_abstract) > len(p["abstract"] or ""):
                    p["abstract"] = paper_abstract

                if enriched.get("year"):
                    p["year"] = enriched["year"]
                if enriched.get("venue"):
                    p["journal"] = enriched["venue"]

                authors_raw = enriched.get("authors", [])
                if authors_raw and isinstance(authors_raw[0], dict):
                    p["authors"] = ", ".join(a.get("name", "") for a in authors_raw)

                oa_pdf = enriched.get("openAccessPdf") or {}
                if isinstance(oa_pdf, dict) and oa_pdf.get("url"):
                    p["pdf_url"] = oa_pdf["url"]

        except Exception as exc:
            logger.warning(
                "ASTA get_paper_batch enrichment failed — returning snippet-only data: %s", exc
            )

        # Step 4: Sort by relevance, strip internal keys, cap at max_results
        result = sorted(papers.values(), key=lambda x: x["_score"], reverse=True)
        result = result[:max_results]
        for p in result:
            p.pop("_score", None)

        logger.info(
            "ASTA snippet_search: %d papers from %d passages for: %s",
            len(result), len(data_list), query[:60],
        )
        return result

    @staticmethod
    def _extract_snippet_data(raw: Any) -> list[dict]:
        """Extract [{paper, snippet, score}, ...] list from snippet_search result."""
        if isinstance(raw, dict):
            inner = raw.get("result", raw)
            if isinstance(inner, dict):
                return inner.get("data", [])
            if isinstance(inner, list):
                return inner
        return []
