"""InvestigationToolNode — 10 tools for NQPR causal link investigations.

Used by investigate_link worker nodes (Phase 3.4). All corpus search
delegates to search_collection / read_section / read_pages from
collection_tools. Web search uses Tavily if TAVILY_API_KEY is set,
otherwise returns a structured stub. compute() evaluates simple Python
arithmetic and falls back to structured passthrough.

Configurable keys consumed:
  search_backend  : SearchBackend instance
  pages_dir       : str  — absolute path to pages directory
  doc_list        : list[dict]  — document catalog
  inv_id          : str | None  — investment filter for search_investment
  bow_id          : str | None  — BOW filter passed down from scope context
"""
from __future__ import annotations

import json
import logging
import math
import os
from typing import Optional

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from src.tools.collection_tools import (
    _fmt_results,
    _get_section_page_range,
    _read_page_range,
    read_section,
)
from src.backends.base import SearchBackend

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@tool
async def search_investment(
    query: str,
    doc_type: Optional[str] = None,
    top_k: Optional[int] = None,
    config: RunnableConfig = None,
) -> str:
    """Search this investment's documents to discover relevant files, pages, and sections.

    Returns short document excerpts with §-references and doc_type annotations.
    Use search to triage, then inspect full sections with get_document_structure,
    read_section, or read_document.

    Args:
        query:    Natural-language question to search for.
        doc_type: Optional filter by document type. Pass null for no filter.
                  Use list_documents to discover available types.
        top_k:    Number of results (default 25, pass null for default).
    """
    import time
    from datetime import datetime, timezone
    try:
        from src.core.tool_tracing import append_to_buffer as _append
    except ImportError:
        _append = None

    configurable = (config or {}).get("configurable", {})
    backend: SearchBackend = configurable["search_backend"]
    inv_id: Optional[str] = configurable.get("inv_id")
    k = top_k if top_k is not None else 25

    start = time.monotonic()
    started_at = datetime.now(timezone.utc).isoformat()

    results = await backend.search(
        query,
        top_k=k,
        inv_id_filter=inv_id,
        doc_type_filter=doc_type,
    )

    duration_ms = int((time.monotonic() - start) * 1000)
    if _append is not None:
        _append("collection_search_traces", {
            "tool_name": "search_investment",
            "called_at": started_at,
            "duration_ms": duration_ms,
            "success": True,
            "error_message": None,
            "query": query,
            "backend": "local",
            "inv_id_filter": inv_id,
            "bow_id_filter": None,
            "top_k": k,
            "result_count": len(results),
            "top_chunk_ids": [r.chunk_id for r in results[:5]],
        })

    return _fmt_results(results)


@tool
async def search_portfolio(
    query: str,
    collection: Optional[str] = None,
    doc_type: Optional[str] = None,
    top_k: Optional[int] = None,
    config: RunnableConfig = None,
) -> str:
    """Search across all investments or the full portfolio index.

    Use for cross-investment patterns, strategy context, or portfolio-level
    evidence. Read the underlying documents before concluding from search hits.

    Args:
        query:      Natural-language question.
        collection: Filter: "investment", "strategy", or null for all.
        doc_type:   Optional filter by document type. Pass null for no filter.
        top_k:      Number of results (default 10, pass null for default).
    """
    import time
    from datetime import datetime, timezone
    try:
        from src.core.tool_tracing import append_to_buffer as _append
    except ImportError:
        _append = None

    configurable = (config or {}).get("configurable", {})
    backend: SearchBackend = configurable["search_backend"]
    k = top_k if top_k is not None else 10

    start = time.monotonic()
    started_at = datetime.now(timezone.utc).isoformat()

    results = await backend.search(
        query,
        top_k=k,
        collection_filter=collection,
        doc_type_filter=doc_type,
    )

    duration_ms = int((time.monotonic() - start) * 1000)
    if _append is not None:
        _append("collection_search_traces", {
            "tool_name": "search_portfolio",
            "called_at": started_at,
            "duration_ms": duration_ms,
            "success": True,
            "error_message": None,
            "query": query,
            "backend": "local",
            "inv_id_filter": None,
            "bow_id_filter": None,
            "top_k": k,
            "result_count": len(results),
            "top_chunk_ids": [r.chunk_id for r in results[:5]],
        })

    return _fmt_results(results)


@tool
async def search_web(
    query: str,
    rationale: str,
    config: RunnableConfig = None,
) -> str:
    """Search the public web for external evidence.

    Use for regulatory updates, WHO guidance, published studies, news,
    trial registry data, and recent developments not in the document collection.

    Args:
        query:     Web search query string.
        rationale: One sentence explaining why this external evidence is needed.
    """
    import time
    from datetime import datetime, timezone
    from src.core.tool_tracing import append_to_buffer

    configurable = (config or {}).get("configurable", {})
    inv_id: str | None = configurable.get("inv_id")

    api_key = os.environ.get("TAVILY_API_KEY")
    start = time.monotonic()
    started_at = datetime.now(timezone.utc).isoformat()
    result_count = 0
    top_urls: list[str] = []
    success = True
    error_message = None

    if not api_key:
        result_text = (
            f"[web search not configured — TAVILY_API_KEY not set]\n"
            f"query: {query}\nrationale: {rationale}"
        )
        success = False
        error_message = "TAVILY_API_KEY not set"
    else:
        try:
            from tavily import TavilyClient  # type: ignore[import]
            client = TavilyClient(api_key=api_key)
            response = client.search(query, max_results=5, search_depth="advanced")
            hits = response.get("results", [])
            result_count = len(hits)
            top_urls = [h.get("url", "") for h in hits[:5]]
            if not hits:
                result_text = f"(no web results for: {query!r})"
            else:
                lines = [f"{len(hits)} web results for: {query!r}"]
                for i, h in enumerate(hits, 1):
                    lines.append(
                        f"\n[{i}] {h.get('title', '')} ({h.get('url', '')})\n"
                        f"    {h.get('content', '')[:400]}"
                    )
                result_text = "\n".join(lines)
        except Exception as exc:
            logger.warning("web search failed: %s", exc)
            result_text = f"(web search failed: {exc})"
            success = False
            error_message = str(exc)

    duration_ms = int((time.monotonic() - start) * 1000)
    append_to_buffer("web_search_traces", {
        "tool_name": "search_web",
        "called_at": started_at,
        "duration_ms": duration_ms,
        "success": success,
        "error_message": error_message,
        "query": query,
        "inv_id": inv_id,
        "result_count": result_count,
        "top_urls": top_urls,
    })

    return result_text


@tool
async def read_document(
    file_id: str,
    page_start: Optional[int] = None,
    page_end: Optional[int] = None,
    section_id: Optional[str] = None,
    config: RunnableConfig = None,
) -> str:
    """Read specific pages or a named section of a document.

    If section_id is provided, reads the full section (preferred — more
    precise). Otherwise reads the page range [page_start, page_end].
    Use after search to get full context beyond the 500-char excerpt.

    Args:
        file_id:    Document identifier from search results.
        page_start: First page to read (if using page range).
        page_end:   Last page to read (if using page range).
        section_id: Section ID from get_document_structure. When provided,
                    reads the full section instead of a page range.
    """
    configurable = (config or {}).get("configurable", {})
    pages_dir = configurable.get("pages_dir")
    doc_list: list[dict] = configurable.get("doc_list") or []

    if not pages_dir:
        return "(pages_dir not configured)"

    import asyncio
    from pathlib import Path

    if section_id:
        ps, pe, err = _get_section_page_range(doc_list, file_id, section_id)
        if err:
            return err
        # asyncio-APPROVED-1: to_thread wraps blocking page file read
        return await asyncio.to_thread(_read_page_range, Path(pages_dir), file_id, ps, pe)

    if page_start is not None and page_end is not None:
        # asyncio-APPROVED-1: to_thread wraps blocking page file read
        return await asyncio.to_thread(_read_page_range, Path(pages_dir), file_id, page_start, page_end)

    return "(provide either section_id or page_start + page_end)"


@tool
async def compute(
    question: str,
    data: Optional[str] = None,
    config: RunnableConfig = None,
) -> str:
    """Perform a numerical calculation — burn rate, enrollment projection, cost comparison.

    Evaluates simple Python arithmetic. For complex calculations, returns the
    question and data structured for your next reasoning step.

    Args:
        question: What to compute (e.g. "What is the annual burn rate?").
        data:     The numbers and context to work with (e.g. "Budget: $7.4M over 3 years").
    """
    import ast
    import time
    from datetime import datetime, timezone
    from src.core.tool_tracing import append_to_buffer

    start = time.monotonic()
    started_at = datetime.now(timezone.utc).isoformat()
    output_type = "text"
    result_text = ""

    try:
        result = ast.literal_eval(question.strip())
        result_text = f"Result: {result}"
        output_type = "number"
    except (ValueError, SyntaxError):
        pass

    if not result_text:
        try:
            safe_ns = {"__builtins__": {}, "math": math}
            result = eval(question.strip(), safe_ns)  # noqa: S307
            result_text = f"Result: {result}"
            output_type = "number"
        except Exception:
            pass

    if not result_text:
        parts = [f"Computation requested: {question}"]
        if data:
            parts.append(f"Input data: {data}")
        parts.append("Perform the calculation in your next response using the data above.")
        result_text = "\n".join(parts)

    duration_ms = int((time.monotonic() - start) * 1000)
    append_to_buffer("compute_traces", {
        "tool_name": "compute",
        "called_at": started_at,
        "duration_ms": duration_ms,
        "success": True,
        "error_message": None,
        "code_snippet": question[:200],
        "output_type": output_type,
        "output_summary": result_text[:200],
    })

    return result_text


@tool
async def submit_findings(
    findings: list[dict],
    overall_assessment: dict,
    config: RunnableConfig = None,
) -> str:
    """Submit your final structured findings when the investigation is complete.

    Call this ONLY when you have searched enough evidence and are ready to
    conclude. The investigation loop will terminate after this call.

    Args:
        findings: List of findings, each with statement, finding_type,
                  severity, confidence, evidence_refs, rationale,
                  and numerical_claims.
        overall_assessment: Object with status, summary, and evidence_gaps.
    """
    return json.dumps(
        {"findings": findings, "overall_assessment": overall_assessment},
        indent=2,
    )


@tool
async def list_documents(
    scope: str,
    doc_type: Optional[str] = None,
    config: RunnableConfig = None,
) -> str:
    """List all available documents for this investment or the full portfolio.

    Returns file_id, filename, doc_type, page count, date, and summary.
    Use this to discover what documents exist before reading them.

    Args:
        scope:    "this_investment" for the current investment, "portfolio" for all.
        doc_type: Optional filter by document type. Pass null to list all.
    """
    configurable = (config or {}).get("configurable", {})
    doc_list: list[dict] = configurable.get("doc_list") or []
    inv_id: Optional[str] = configurable.get("inv_id")

    if scope == "this_investment" and inv_id:
        docs = [d for d in doc_list if d.get("inv_id") == inv_id]
    else:
        docs = doc_list

    if doc_type:
        docs = [d for d in docs if d.get("doc_type") == doc_type]

    if not docs:
        return f"(no documents found for scope={scope!r} doc_type={doc_type!r})"

    lines = [f"{len(docs)} documents:"]
    for d in docs:
        lines.append(
            f"  file_id={d.get('file_id', '')} "
            f"type={d.get('doc_type', '-')} "
            f"pages={d.get('total_pages', '?')} "
            f"date={d.get('date', '-')} "
            f"filename={d.get('filename', '-')}\n"
            f"    {(d.get('summary') or '')[:150]}"
        )
    return "\n".join(lines)


@tool
async def read_document_summary(
    file_id: str,
    config: RunnableConfig = None,
) -> str:
    """Read the full summary and section outline of a specific document.

    Returns a summary, document type, section list with page ranges,
    and key metadata. Use this to understand what a document covers before
    deciding which sections to read in full.

    Args:
        file_id: Document identifier from search results or list_documents.
    """
    import asyncio
    import json as _json
    from pathlib import Path

    configurable = (config or {}).get("configurable", {})
    pages_dir = configurable.get("pages_dir")
    doc_list: list[dict] = configurable.get("doc_list") or []

    doc_entry = next((d for d in doc_list if d.get("file_id") == file_id), None)

    summary = (doc_entry or {}).get("summary", "") if doc_entry else ""

    if not summary and pages_dir:
        index_path = Path(pages_dir) / file_id / "index.json"
        if index_path.exists():
            try:
                # asyncio-APPROVED-1: to_thread wraps blocking file read for document index
                page_idx = _json.loads(await asyncio.to_thread(index_path.read_text))
                summary = page_idx.get("summary", "")
            except (Exception,):
                pass

    if doc_entry is None:
        if not summary:
            return f"(document {file_id!r} not found)"
        return f"Summary for {file_id}:\n{summary}"

    lines = [
        f"file_id: {file_id}",
        f"filename: {doc_entry.get('filename', '-')}",
        f"doc_type: {doc_entry.get('doc_type', '-')}",
        f"total_pages: {doc_entry.get('total_pages', '?')}",
        f"date: {doc_entry.get('date', '-')}",
        f"inv_id: {doc_entry.get('inv_id', '-')}",
        f"bow_id: {doc_entry.get('bow_id', '-')}",
        "",
        f"Summary:\n{summary or '(no summary available)'}",
    ]

    sections = doc_entry.get("sections") or []
    if sections:
        lines.append(f"\nSections ({len(sections)}):")
        for s in sections:
            lines.append(
                f"  {s.get('id', '?')} — {s.get('label', '')} "
                f"(pp. {s.get('page_start', '?')}–{s.get('page_end', '?')})"
            )

    return "\n".join(lines)


@tool
async def get_document_structure(
    file_id: str,
    config: RunnableConfig = None,
) -> str:
    """Get the table of contents and section map for a specific document.

    Returns section IDs, labels, page ranges, and table/figure flags.
    Use BEFORE read_document or read_section to find the right section_id.

    Args:
        file_id: Document identifier from search results or list_documents.
    """
    import asyncio
    import json as _json
    from pathlib import Path

    configurable = (config or {}).get("configurable", {})
    pages_dir = configurable.get("pages_dir")
    doc_list: list[dict] = configurable.get("doc_list") or []

    doc_entry = next((d for d in doc_list if d.get("file_id") == file_id), None)
    sections: list[dict] = []

    if doc_entry:
        sections = doc_entry.get("sections") or []

    if not sections and pages_dir:
        index_path = Path(pages_dir) / file_id / "index.json"
        if index_path.exists():
            try:
                # asyncio-APPROVED-1: to_thread wraps blocking file read for document index
                page_idx = _json.loads(await asyncio.to_thread(index_path.read_text))
                sections = page_idx.get("sections") or []
            except (Exception,):
                pass

    if not sections:
        return f"(no section structure available for {file_id})"

    lines = [f"Structure of {file_id} ({len(sections)} sections):"]
    for s in sections:
        has_data = ""
        if s.get("has_table"):
            has_data += " [TABLE]"
        if s.get("has_figure"):
            has_data += " [FIGURE]"
        lines.append(
            f"  {s.get('id', '?')} — {s.get('label', '')} "
            f"pp. {s.get('page_start', '?')}–{s.get('page_end', '?')}{has_data}"
        )
    return "\n".join(lines)


# re-export read_section from collection_tools so it's available in this module
# without duplication — InvestigationToolNode uses the identical function.


# ---------------------------------------------------------------------------
# Exported tool list for ToolNode construction
# ---------------------------------------------------------------------------

INVESTIGATION_TOOLS = [
    search_investment,
    search_portfolio,
    search_web,
    read_document,
    compute,
    submit_findings,
    list_documents,
    read_document_summary,
    get_document_structure,
    read_section,
]
