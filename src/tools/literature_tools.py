"""Literature retrieval tools for SLR and LBD agents.

search_asta_papers and search_openalex_papers are @tool-decorated so calls
via .ainvoke(input, config=config) appear as tool-use spans in LangSmith /
Langfuse traces when LangchainInstrumentor is active.
"""
from __future__ import annotations

from langchain_core.tools import tool

from src.core.agents.asta import AstaClient
from src.core.agents.openalex import OpenAlexClient


@tool
async def search_asta_papers(query: str, max_results: int = 20) -> list[dict]:
    """Search Asta (Semantic Scholar MCP) for academic papers by keyword.

    Falls back to the Semantic Scholar public API when ASTA_API_KEY is unset
    or the MCP endpoint is unreachable. Returns paper dicts with paperId,
    title, year, authors, abstract, url.
    """
    return await AstaClient().search(query, max_results=max_results)


@tool
async def search_openalex_papers(query: str, max_results: int = 20) -> list[dict]:
    """Search OpenAlex for academic papers by keyword.

    Uses cursor-based pagination with polite-pool rate limiting. Returns
    paper dicts with paperId, title, year, authors, abstract, url.
    """
    return await OpenAlexClient().search(query, max_results=max_results)
