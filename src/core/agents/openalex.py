"""Async OpenAlex client for systematic literature retrieval.

Returns paper dicts with paperId, title, year, authors, abstract, url, source,
externalIds. All fields are always present; missing values are empty strings or None.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import httpx

from src.config import OPENALEX_EMAIL, OPENALEX_MAX_RESULTS

logger = logging.getLogger(__name__)

_OPENALEX_BASE = "https://api.openalex.org"
_MIN_REQUEST_INTERVAL = 0.11  # ~9 req/s — polite pool below their 10/s cap


class OpenAlexClient:
    def __init__(
        self,
        email: str | None = None,
        max_results: int = OPENALEX_MAX_RESULTS,
    ) -> None:
        self._email = email or OPENALEX_EMAIL
        self._max_results = max_results
        self._last_request_at: float = 0.0

    async def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request_at
        wait = _MIN_REQUEST_INTERVAL - elapsed
        if wait > 0:
            # asyncio-APPROVED-4: sleep in throttle — infrastructure rate limiter
            await asyncio.sleep(wait)
        self._last_request_at = time.monotonic()

    def _user_agent(self) -> str:
        if self._email:
            return f"nqpr-pipeline/1.0 (mailto:{self._email})"
        return "nqpr-pipeline/1.0"

    async def search(
        self,
        query: str,
        max_results: int | None = None,
        year_start: int | None = None,
        year_end: int | None = None,
    ) -> list[dict]:
        """Search OpenAlex works. Returns up to max_results normalised paper dicts."""
        n = max_results or self._max_results
        headers = {"User-Agent": self._user_agent()}
        params: dict[str, Any] = {
            "search": query,
            "per_page": min(n, 200),
            "select": (
                "id,title,authorships,publication_year,"
                "abstract_inverted_index,primary_location,doi,ids"
            ),
        }
        if year_start or year_end:
            lo = year_start or 1900
            hi = year_end or 2100
            params["filter"] = f"publication_year:{lo}-{hi}"

        results: list[dict] = []
        cursor = "*"

        async with httpx.AsyncClient(timeout=30.0) as client:
            while len(results) < n:
                await self._throttle()
                params["cursor"] = cursor
                try:
                    resp = await client.get(
                        f"{_OPENALEX_BASE}/works",
                        headers=headers,
                        params=params,
                    )
                    resp.raise_for_status()
                    data = resp.json()
                except Exception as exc:
                    logger.warning("OpenAlex request failed: %s", exc)
                    break

                items = data.get("results") or []
                if not items:
                    break

                for item in items:
                    results.append(_normalize_work(item))
                    if len(results) >= n:
                        break

                meta = data.get("meta") or {}
                cursor = meta.get("next_cursor")
                if not cursor:
                    break

        return results[:n]


def _reconstruct_abstract(inverted_index: dict | None) -> str:
    if not inverted_index:
        return ""
    positions: list[tuple[int, str]] = []
    for word, idxs in inverted_index.items():
        for pos in idxs:
            positions.append((pos, word))
    positions.sort()
    return " ".join(w for _, w in positions)


def _normalize_work(item: dict) -> dict:
    authorships = item.get("authorships") or []
    authors = ", ".join(
        a.get("author", {}).get("display_name", "")
        for a in authorships[:5]
        if a.get("author")
    )
    abstract = _reconstruct_abstract(item.get("abstract_inverted_index"))
    location = item.get("primary_location") or {}
    url = location.get("landing_page_url") or item.get("doi") or ""
    ext = item.get("ids") or {}
    return {
        "paperId": item.get("id", ""),
        "title": item.get("title", ""),
        "year": item.get("publication_year"),
        "authors": authors,
        "abstract": abstract[:1000],
        "url": url,
        "source": url,
        "externalIds": {
            "doi": ext.get("doi", ""),
            "pmid": ext.get("pmid", ""),
        },
    }
