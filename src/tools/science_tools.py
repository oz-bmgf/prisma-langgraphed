"""ScienceToolNode — InvestigationToolNode + search_asta.

Extends the investigation tool set with ASTA (Semantic Scholar) access
for Phase 3.5d science validation. At least one search_asta call is
required before the loop may terminate with status=evidence_gathered.

Configurable keys: same as InvestigationToolNode.
  asta_api_key : str | None  — ASTA API key (falls back to ASTA_API_KEY env var)
"""
from __future__ import annotations

import logging
import os
from typing import Optional

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from src.tools.investigation_tools import INVESTIGATION_TOOLS

logger = logging.getLogger(__name__)


@tool
async def search_asta(
    query: str,
    config: RunnableConfig = None,
) -> str:
    """Search the Asta scientific corpus (Semantic Scholar, 225M+ papers, 12M+ full-text).

    Use natural-language questions about scientific mechanisms, efficacy data,
    effect sizes, trial results, or comparable program outcomes. This is the
    PRIMARY source for published peer-reviewed evidence. You MUST call this
    at least once per investigation before emitting status=evidence_gathered.

    Args:
        query: Natural-language scientific question (e.g.
               "RTS,S malaria vaccine efficacy in children under 5").
    """
    import time
    from datetime import datetime, timezone
    from src.core.tool_tracing import append_to_buffer

    configurable = (config or {}).get("configurable", {})
    api_key = configurable.get("asta_api_key") or os.environ.get("ASTA_API_KEY")

    start = time.monotonic()
    started_at = datetime.now(timezone.utc).isoformat()
    result_text = ""
    result_count = 0
    top_paper_ids: list[str] = []
    top_titles: list[str] = []
    success = True
    error_message = None

    try:
        from src.core.agents.asta import AstaClient  # lazy import — not available in all envs
        client = AstaClient(api_key=api_key)
        results = await client.search(query)
        result_count = len(results) if results else 0
        if not results:
            result_text = f"(no ASTA results for: {query!r})"
        else:
            top_paper_ids = [str(r.get("paperId", "")) for r in results[:5]]
            top_titles = [r.get("title", "") for r in results[:5]]
            lines = [f"{len(results)} ASTA results for: {query!r}"]
            for i, r in enumerate(results, 1):
                lines.append(
                    f"\n[{i}] {r.get('title', '')} ({r.get('year', '?')})\n"
                    f"    Authors: {r.get('authors', '-')}\n"
                    f"    Abstract: {r.get('abstract', '')[:400]}\n"
                    f"    PMID/DOI: {r.get('externalIds', {})}"
                )
            result_text = "\n".join(lines)
    except ImportError:
        pass
    except Exception as exc:
        logger.warning("ASTA search failed: %s", exc)
        result_text = f"(ASTA search failed: {exc})"
        success = False
        error_message = str(exc)

    if not result_text:
        # Fallback: use Semantic Scholar public API when AstaClient is unavailable
        try:
            import asyncio
            import json
            import urllib.parse
            import urllib.request

            def _fetch_semantic_scholar(q: str) -> str:
                url = (
                    "https://api.semanticscholar.org/graph/v1/paper/search"
                    f"?query={urllib.parse.quote(q)}"
                    "&fields=title,authors,year,abstract,externalIds"
                    "&limit=5"
                )
                req = urllib.request.Request(url, headers={"User-Agent": "nqpr-pipeline/1.0"})
                with urllib.request.urlopen(req, timeout=15) as resp:
                    data = json.loads(resp.read())
                return data

            # asyncio-APPROVED-1: to_thread wraps blocking urllib.request Semantic Scholar call
            raw_data = await asyncio.to_thread(_fetch_semantic_scholar, query)
            papers = raw_data.get("data") or []
            result_count = len(papers)
            top_paper_ids = [p.get("paperId", "") for p in papers[:5]]
            top_titles = [p.get("title", "") for p in papers[:5]]
            if not papers:
                result_text = f"(no Semantic Scholar results for: {query!r})"
            else:
                lines = [f"{len(papers)} papers for: {query!r}"]
                for i, p in enumerate(papers, 1):
                    authors = ", ".join(a.get("name", "") for a in (p.get("authors") or [])[:3])
                    lines.append(
                        f"\n[{i}] {p.get('title', '')} ({p.get('year', '?')})\n"
                        f"    Authors: {authors}\n"
                        f"    Abstract: {(p.get('abstract') or '')[:400]}"
                    )
                result_text = "\n".join(lines)
        except Exception as exc:
            logger.warning("Semantic Scholar fallback failed: %s", exc)
            result_text = (
                f"[ASTA/Semantic Scholar unavailable — configure ASTA_API_KEY or network]\n"
                f"query: {query}"
            )
            success = False
            error_message = str(exc)

    duration_ms = int((time.monotonic() - start) * 1000)
    append_to_buffer("asta_traces", {
        "tool_name": "search_asta",
        "called_at": started_at,
        "duration_ms": duration_ms,
        "success": success,
        "error_message": error_message,
        "query": query,
        "result_count": result_count,
        "top_paper_ids": top_paper_ids,
        "top_titles": top_titles,
        "index_used": "semantic_scholar",
    })

    return result_text


# ---------------------------------------------------------------------------
# Exported tool list for ToolNode construction
# ---------------------------------------------------------------------------

SCIENCE_TOOLS = [search_asta] + list(INVESTIGATION_TOOLS)
