from __future__ import annotations

import logging

from langchain_core.tools import tool

from src.config import OPENALEX_EMAIL, OPENALEX_MAX_RESULTS, ASTA_API_KEY, ASTA_ENDPOINT
from src.core.agents.openalex import OpenAlexClient
from src.core.agents.asta import AstaClient

logger = logging.getLogger(__name__)

_openalex_client: OpenAlexClient | None = None
_asta_client: AstaClient | None = None


def _get_openalex() -> OpenAlexClient:
    global _openalex_client
    if _openalex_client is None:
        _openalex_client = OpenAlexClient(email=OPENALEX_EMAIL, max_results=OPENALEX_MAX_RESULTS)
    return _openalex_client


def _get_asta() -> AstaClient:
    global _asta_client
    if _asta_client is None:
        _asta_client = AstaClient(api_key=ASTA_API_KEY, endpoint=ASTA_ENDPOINT)
    return _asta_client


@tool
async def search_openalex(query: str, top_k: int = 20) -> list[dict]:
    """Search OpenAlex (200M+ academic papers). Returns paper dicts with title, abstract, doi, year, citation_count."""
    client = _get_openalex()
    results = await client.search(query, max_results=top_k)
    return results if isinstance(results, list) else []


@tool
async def search_asta(query: str, top_k: int = 20) -> list[dict]:
    """Search Semantic Scholar (225M+ papers) via ASTA. Returns paper dicts with title, abstract, year, tldr, citation_count."""
    client = _get_asta()
    results = await client.search(query, max_results=top_k)
    return results if isinstance(results, list) else []
