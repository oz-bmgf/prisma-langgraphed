"""InvestigationToolNode — 13 tools for NQPR causal link investigations.

Used by investigate_link worker nodes (Phase 3.4). All corpus search
delegates to search_collection / read_section / read_pages from
collection_tools. Web search uses OpenAI Responses API (gpt-5.4 +
web_search tool), matching the OLD repo's thread_sub_agent pattern.

Configurable keys consumed:
  search_backend  : SearchBackend instance
  pages_dir       : str  — absolute path to pages directory
  doc_list        : list[dict]  — document catalog
  inv_id          : str | None  — investment filter for search_investment
  bow_id          : str | None  — BOW filter for search_bow (scope context)
"""
from __future__ import annotations

import json
import logging
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
    backend: SearchBackend = configurable.get("search_backend")
    if backend is None:
        return "(search_backend not configured)"
    inv_id: Optional[str] = configurable.get("inv_id")
    bow_id: Optional[str] = configurable.get("bow_id")
    k = top_k if top_k is not None else 25

    start = time.monotonic()
    started_at = datetime.now(timezone.utc).isoformat()

    results = await backend.search(
        query,
        top_k=k,
        inv_id_filter=inv_id,
        bow_id_filter=bow_id,
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
    backend: SearchBackend = configurable.get("search_backend")
    if backend is None:
        return "(search_backend not configured)"
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
async def search_bow(
    query: str,
    top_k: Optional[int] = None,
    config: RunnableConfig = None,
) -> str:
    """Search all investments in this body of work (BOW / scope).

    Use for cross-investment patterns within the same outcome area —
    sibling program comparisons, shared assumptions, BOW-level evidence.
    Broader than search_investment; narrower than search_portfolio.

    Args:
        query: Natural-language question to search for.
        top_k: Number of results (default 10, pass null for default).
    """
    import time
    from datetime import datetime, timezone
    try:
        from src.core.tool_tracing import append_to_buffer as _append
    except ImportError:
        _append = None

    configurable = (config or {}).get("configurable", {})
    backend: SearchBackend = configurable.get("search_backend")
    if backend is None:
        return "(search_backend not configured)"
    bow_id: Optional[str] = configurable.get("bow_id")
    if not bow_id:
        # No BOW context — fall back to unfiltered portfolio search
        return "(search_bow called without bow_id context — use search_portfolio instead)"
    k = top_k if top_k is not None else 10

    start = time.monotonic()
    started_at = datetime.now(timezone.utc).isoformat()

    results = await backend.search(
        query,
        top_k=k,
        bow_id_filter=bow_id,
    )

    duration_ms = int((time.monotonic() - start) * 1000)
    if _append is not None:
        _append("collection_search_traces", {
            "tool_name": "search_bow",
            "called_at": started_at,
            "duration_ms": duration_ms,
            "success": True,
            "error_message": None,
            "query": query,
            "backend": "local",
            "inv_id_filter": None,
            "bow_id_filter": bow_id,
            "top_k": k,
            "result_count": len(results),
            "top_chunk_ids": [r.chunk_id for r in results[:5]],
        })

    return _fmt_results(results)


@tool
async def search_science(
    query: str,
    top_k: Optional[int] = None,
    config: RunnableConfig = None,
) -> str:
    """Search published scientific literature already in the document collection.

    Use for peer-reviewed evidence on mechanisms, efficacy, epidemiology,
    or trial results that are already ingested. Searches all collections
    with doc_type=science — no investment or BOW scope filter applied.
    For external literature not in the collection, use search_asta instead.

    Args:
        query: Natural-language scientific question.
        top_k: Number of results (default 10, pass null for default).
    """
    import time
    from datetime import datetime, timezone
    try:
        from src.core.tool_tracing import append_to_buffer as _append
    except ImportError:
        _append = None

    configurable = (config or {}).get("configurable", {})
    backend: SearchBackend = configurable.get("search_backend")
    if backend is None:
        return "(search_backend not configured)"
    k = top_k if top_k is not None else 10

    start = time.monotonic()
    started_at = datetime.now(timezone.utc).isoformat()

    # Global scope: no inv_id or bow_id filter — science docs span the whole collection
    results = await backend.search(
        query,
        top_k=k,
        doc_type_filter="science",
    )

    duration_ms = int((time.monotonic() - start) * 1000)
    if _append is not None:
        _append("collection_search_traces", {
            "tool_name": "search_science",
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
async def search_policy(
    query: str,
    top_k: Optional[int] = None,
    config: RunnableConfig = None,
) -> str:
    """Search WHO guidance, normative body documents, and policy papers in the collection.

    Use for regulatory context, WHO recommendations, SAGE reports, policy
    frameworks, and government guidance. Searches all collections with
    doc_type=policy — no investment or BOW scope filter applied.

    Args:
        query: Natural-language policy question.
        top_k: Number of results (default 10, pass null for default).
    """
    import time
    from datetime import datetime, timezone
    try:
        from src.core.tool_tracing import append_to_buffer as _append
    except ImportError:
        _append = None

    configurable = (config or {}).get("configurable", {})
    backend: SearchBackend = configurable.get("search_backend")
    if backend is None:
        return "(search_backend not configured)"
    k = top_k if top_k is not None else 10

    start = time.monotonic()
    started_at = datetime.now(timezone.utc).isoformat()

    # Global scope: no inv_id or bow_id filter — policy docs span the whole collection
    results = await backend.search(
        query,
        top_k=k,
        doc_type_filter="policy",
    )

    duration_ms = int((time.monotonic() - start) * 1000)
    if _append is not None:
        _append("collection_search_traces", {
            "tool_name": "search_policy",
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
    Uses OpenAI Responses API (gpt-5.4 + web_search tool).

    Args:
        query:     Web search query string.
        rationale: One sentence explaining why this external evidence is needed.
    """
    import asyncio
    import time
    from datetime import datetime, timezone
    from src.core.tool_tracing import append_to_buffer

    configurable = (config or {}).get("configurable", {})
    inv_id: str | None = configurable.get("inv_id")

    start = time.monotonic()
    started_at = datetime.now(timezone.utc).isoformat()
    result_count = 0
    success = True
    error_message = None

    try:
        from openai import OpenAI
        client = OpenAI()

        def _sync_search() -> str:
            response = client.responses.create(
                model="gpt-5.4",
                input=[
                    {
                        "role": "system",
                        "content": "Search for factual, current information. "
                                   "Focus on publicly available data.",
                    },
                    {
                        "role": "user",
                        "content": f"{rationale}\n\nSearch query: {query}",
                    },
                ],
                tools=[{"type": "web_search"}],
            )
            return response.output_text or ""

        # asyncio-APPROVED-1: to_thread wraps blocking OpenAI Responses API call
        result_text = await asyncio.to_thread(_sync_search)
        result_count = 1 if result_text else 0

    except Exception as exc:
        logger.warning("web search failed: %s", exc)
        result_text = f"[web search not configured — {exc}]"
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
        "top_urls": [],
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
    """Execute a numerical computation via OpenAI code_interpreter (sandboxed Python).

    Use for: burn rates, runway projections, enrollment completion rates, date spans,
    financial ratios. Describe the computation in natural language; provide numbers in data.

    Args:
        question: What to compute in natural language (e.g. "What is the annual burn rate?").
        data:     Numbers and context to work with (e.g. "Budget: $7.4M, Period: 3 years").
    """
    import asyncio
    import time
    from datetime import datetime, timezone
    from src.config import COMPUTE_MODEL
    from src.core.tool_tracing import append_to_buffer

    start = time.monotonic()
    started_at = datetime.now(timezone.utc).isoformat()
    result_text = ""
    success = True
    error_message = None

    try:
        from openai import OpenAI
        client = OpenAI()

        input_content = question
        if data:
            input_content = f"Data:\n{data[:3000]}\n\nCompute: {question}\n\nShow your work and give a clear numerical answer."

        def _sync_compute() -> str:
            response = client.responses.create(
                model=COMPUTE_MODEL,
                input=[{
                    "role": "system",
                    "content": "You are a numerical analyst. Compute precisely and show your work.",
                }, {
                    "role": "user",
                    "content": input_content,
                }],
                tools=[{"type": "code_interpreter", "container": {"type": "auto"}}],
            )
            return response.output_text or ""

        # asyncio-APPROVED-1: to_thread wraps blocking OpenAI Responses API call
        result_text = await asyncio.to_thread(_sync_compute)
        if not result_text:
            result_text = "(computation returned no output)"

    except Exception as exc:
        logger.warning("compute tool (code_interpreter) failed: %s", exc)
        result_text = f"(computation failed: {exc})"
        success = False
        error_message = str(exc)

    duration_ms = int((time.monotonic() - start) * 1000)
    append_to_buffer("compute_traces", {
        "tool_name": "compute",
        "called_at": started_at,
        "duration_ms": duration_ms,
        "success": success,
        "error_message": error_message,
        "code_snippet": question[:200],
        "output_type": "code_interpreter",
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
    search_bow,
    search_science,
    search_policy,
    search_web,
    read_document,
    compute,
    list_documents,
    read_document_summary,
    get_document_structure,
    read_section,
    # submit_findings intentionally excluded: loop terminates via
    # InvestigationActionsOutput.status (answered / not_answerable /
    # unresolved_conflict) + empty next_actions, not via a sentinel tool call.
    # Including it caused the LLM to call it as a next_action which was
    # silently dropped by _execute_actions, wasting an iteration.
]
