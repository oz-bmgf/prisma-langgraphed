"""ScienceToolNode — 8-tool set for Phase 3.5d science assumption validation.

Matches OLD science_investigator.py tool vocabulary: search_asta (external
literature gate), search_bow / search_science / search_policy (scoped local
collection), search_web, read_document, compute, read_section.

Intentionally excludes submit_findings (conflicts with the science loop's
own status=evidence_gathered termination protocol), and excludes the document
navigation tools (list_documents, read_document_summary, get_document_structure)
which are link-investigation-only helpers.

Configurable keys: same as InvestigationToolNode.
  asta_api_key : str | None  — ASTA API key (falls back to ASTA_API_KEY env var)

search_asta error handling mirrors old science_investigator._search_asta:
  - ImportError (AstaClient missing): returns error message, marks success=False
  - Runtime failure (server error, network): returns error message, marks success=False
  - No Semantic Scholar fallback — matches old-repo behavior
"""
from __future__ import annotations

import logging
import os

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from src.tools.investigation_tools import (
    compute,
    read_document,
    read_section,
    search_bow,
    search_policy,
    search_science,
    search_web,
)

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
        result_text = f"(ASTA unavailable — configure ASTA_API_KEY)\nquery: {query}"
        success = False
        error_message = "AstaClient not importable"
    except Exception as exc:
        logger.warning("ASTA search failed: %s", exc)
        result_text = f"(ASTA search failed: {exc})"
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
        "index_used": "asta",
    })

    return result_text


# ---------------------------------------------------------------------------
# Exported tool list for ToolNode construction
# ---------------------------------------------------------------------------

SCIENCE_TOOLS = [
    search_asta,        # external literature — must call at least once (ASTA gate)
    search_bow,         # sibling investments in same BOW
    search_science,     # local science/lit docs (global scope, no inv filter)
    search_policy,      # local WHO/policy docs (global scope, no inv filter)
    search_web,         # recent developments, preprints, registry data
    read_document,      # full page or section text
    compute,            # arithmetic on verified facts
    read_section,       # named section read
    # Excluded: submit_findings (conflicts with evidence_gathered termination)
    # Excluded: list_documents, read_document_summary, get_document_structure
    #           (link-investigation nav tools; not needed for science loop)
]
