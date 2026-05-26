"""CollectionToolNode — 6 search and retrieval tools for NQPR agents.

All search calls delegate to the SearchBackend injected via
config["configurable"]["search_backend"]. File reading is done from
config["configurable"]["pages_dir"]. Document metadata (sections, doc_type)
is resolved from config["configurable"]["doc_list"].

Configurable keys consumed:
  search_backend  : SearchBackend instance (required for search_collection)
  pages_dir       : str  — absolute path to pages/{file_id}/ directory tree
  doc_list        : list[dict]  — document catalog loaded by load_collection
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
from pathlib import Path
from typing import Optional

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from src.backends.base import SearchBackend, SearchResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared formatting helpers
# ---------------------------------------------------------------------------


def _fmt_results(results: list[SearchResult]) -> str:
    if not results:
        return "(no results)"
    lines = []
    for i, r in enumerate(results, 1):
        snippet = r.text.replace("\n", " ")[:400]
        lines.append(
            f"[{i}] score={r.score:.3f} file={r.file_id[:60]} "
            f"inv={r.inv_id or '-'} bow={r.bow_id or '-'} "
            f"pages={r.page_start}–{r.page_end} doc_type={r.doc_type or '-'}\n"
            f"    {snippet}"
        )
    return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# File-reading helpers (sync — called via asyncio.to_thread)
# ---------------------------------------------------------------------------


def _read_page_range(pages_dir: Path, file_id: str, page_start: int, page_end: int) -> str:
    doc_dir = pages_dir / file_id
    if not doc_dir.is_dir():
        return f"(no pages directory for {file_id})"
    parts: list[str] = []
    for pg in range(page_start, min(page_end, page_start + 50) + 1):
        txt_path = doc_dir / f"p{pg:03d}.txt"
        content = txt_path.read_text(encoding="utf-8") if txt_path.exists() else "[text not available]"
        parts.append(f"--- Page {pg} ---\n{content}")
    return "\n\n".join(parts) if parts else f"(no pages in range {page_start}–{page_end})"


def _load_page_image(pages_dir: Path, file_id: str, page: int) -> Optional[bytes]:
    img_path = pages_dir / file_id / f"p{page:03d}.png"
    return img_path.read_bytes() if img_path.exists() else None


def _load_section_images(
    pages_dir: Path,
    file_id: str,
    page_start: int,
    page_end: int,
    max_images: int,
) -> list[tuple[int, bytes]]:
    manifest_path = pages_dir / file_id / "manifest.json"
    page_meta: dict[int, dict] = {}
    if manifest_path.exists():
        try:
            for pm in json.loads(manifest_path.read_text()).get("pages", []):
                page_meta[pm["page"]] = pm
        except (json.JSONDecodeError, OSError):
            pass

    all_pages = list(range(page_start, page_end + 1))
    data_pages = [p for p in all_pages if page_meta.get(p, {}).get("has_table") or page_meta.get(p, {}).get("has_figure")]
    ordered = (data_pages + [p for p in all_pages if p not in data_pages])[:max_images]

    results: list[tuple[int, bytes]] = []
    for pg in ordered:
        img_path = pages_dir / file_id / f"p{pg:03d}.png"
        if img_path.exists():
            results.append((pg, img_path.read_bytes()))
    return results


def _get_section_page_range(doc_list: list[dict], file_id: str, section_id: str) -> tuple[Optional[int], Optional[int], Optional[str]]:
    """Return (page_start, page_end, error_msg) for a section."""
    doc_entry = next((d for d in doc_list if d.get("file_id") == file_id), None)
    if doc_entry is None:
        return None, None, f"(document {file_id!r} not found in catalog)"
    section = next((s for s in (doc_entry.get("sections") or []) if s.get("id") == section_id), None)
    if section is None:
        return None, None, f"(section {section_id!r} not found in {file_id})"
    return section.get("page_start", 1), section.get("page_end", section.get("page_start", 1)), None


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@tool
async def search_collection(
    query: str,
    top_k: int = 20,
    collection: Optional[str] = None,
    bow_id: Optional[str] = None,
    inv_id: Optional[str] = None,
    doc_type: Optional[str] = None,
    include_aux: bool = False,
    config: RunnableConfig = None,
) -> str:
    """Search the document collection using vector + keyword hybrid retrieval.

    Returns ranked document excerpts with file IDs, page ranges, scores,
    and source attribution. Use for triage — always follow up with
    read_section or read_pages to get full context on promising hits.

    Args:
        query:       Natural-language question or keyword query.
        top_k:       Number of results to return (default 20).
        collection:  Filter by collection: "investment", "strategy", or None for all.
        bow_id:      Filter to one bundle-of-work and its cross-cutting strategy docs.
        inv_id:      Filter to one specific investment's documents.
        doc_type:    Filter by document type (e.g. "progress_report", "strategy").
        include_aux: Also search auxiliary collections from aux_collections config.
    """
    configurable = (config or {}).get("configurable", {})
    backend: SearchBackend = configurable["search_backend"]

    import time
    from datetime import datetime, timezone
    try:
        from src.core.tool_tracing import append_to_buffer as _append
    except ImportError:
        _append = None

    start = time.monotonic()
    started_at = datetime.now(timezone.utc).isoformat()

    results = await backend.search(
        query,
        top_k=top_k,
        collection_filter=collection,
        bow_id_filter=bow_id,
        inv_id_filter=inv_id,
        doc_type_filter=doc_type,
    )

    if include_aux:
        aux_backends: dict[str, SearchBackend] = configurable.get("aux_backends") or {}
        for _team, aux in aux_backends.items():
            try:
                aux_results = await aux.search(query, top_k=top_k // 2)
                results = results + aux_results
            except Exception as exc:
                logger.warning("aux backend search failed: %s", exc)
        results = sorted(results, key=lambda r: r.score, reverse=True)[:top_k]

    duration_ms = int((time.monotonic() - start) * 1000)
    if _append is not None:
        backend_name = type(backend).__name__.lower()
        if "qdrant" in backend_name:
            backend_str = "qdrant"
        elif "azure" in backend_name:
            backend_str = "azure"
        else:
            backend_str = "local"
        _append("collection_search_traces", {
            "tool_name": "search_collection",
            "called_at": started_at,
            "duration_ms": duration_ms,
            "success": True,
            "error_message": None,
            "query": query,
            "backend": backend_str,
            "inv_id_filter": inv_id,
            "bow_id_filter": bow_id,
            "top_k": top_k,
            "result_count": len(results),
            "top_chunk_ids": [r.chunk_id for r in results[:5]],
        })

    return _fmt_results(results)


@tool
async def read_section(
    file_id: str,
    section_id: str,
    config: RunnableConfig = None,
) -> str:
    """Read the full text of a named section from a document.

    Use after get_document_structure to identify the correct section_id.
    Returns complete section text, never truncated.

    Args:
        file_id:    Document identifier from search results.
        section_id: Section ID from get_document_structure (e.g. "C001-S3").
    """
    configurable = (config or {}).get("configurable", {})
    pages_dir = configurable.get("pages_dir")
    doc_list: list[dict] = configurable.get("doc_list") or []

    if not pages_dir:
        return "(pages_dir not configured)"

    page_start, page_end, err = _get_section_page_range(doc_list, file_id, section_id)
    if err:
        return err

    # asyncio-APPROVED-1: to_thread wraps blocking page file read
    return await asyncio.to_thread(_read_page_range, Path(pages_dir), file_id, page_start, page_end)


@tool
async def read_pages(
    file_id: str,
    page_start: int,
    page_end: int,
    config: RunnableConfig = None,
) -> str:
    """Read specific pages of a document (full text, never truncated).

    Use when you need surrounding context outside a named section boundary,
    or when a search result references a specific page range you want in full.

    Args:
        file_id:    Document identifier from search results.
        page_start: First page to read (1-indexed).
        page_end:   Last page to read (inclusive).
    """
    configurable = (config or {}).get("configurable", {})
    pages_dir = configurable.get("pages_dir")
    if not pages_dir:
        return "(pages_dir not configured)"

    # asyncio-APPROVED-1: to_thread wraps blocking page file read
    return await asyncio.to_thread(_read_page_range, Path(pages_dir), file_id, page_start, page_end)


@tool
async def read_key_docs(
    inv_id: str,
    config: RunnableConfig = None,
) -> str:
    """List key documents for an investment, prioritised by document type.

    Returns file_id, filename, doc_type, and summary for each document.
    Use to orient on what documents are available before deciding what to read.

    Args:
        inv_id: Investment identifier (e.g. "INV-041892").
    """
    configurable = (config or {}).get("configurable", {})
    doc_list: list[dict] = configurable.get("doc_list") or []

    docs = [d for d in doc_list if d.get("inv_id") == inv_id]
    if not docs:
        return f"(no documents found for investment {inv_id})"

    _priority = {
        "progress_report": 0, "final_report": 1, "proposal": 2,
        "amendment": 3, "budget": 4, "milestone": 5, "deliverable": 6,
    }
    docs_sorted = sorted(docs, key=lambda d: _priority.get(d.get("doc_type", ""), 99))

    lines = [f"{len(docs_sorted)} documents for {inv_id}:"]
    for d in docs_sorted:
        lines.append(
            f"  file_id={d.get('file_id', '')} "
            f"type={d.get('doc_type', '-')} "
            f"filename={d.get('filename', '-')}\n"
            f"    {(d.get('summary') or '')[:200]}"
        )
    return "\n".join(lines)


@tool
async def read_page_image(
    file_id: str,
    page: int,
    config: RunnableConfig = None,
) -> str:
    """Read a rendered page image (PNG) for multimodal evidence analysis.

    Returns a base64-encoded data URI ready for use in a multimodal message,
    or an informational message if the image is unavailable.

    Use when a page contains tables, charts, or figures that require visual
    analysis — always try read_pages first for text-extractable content.

    Args:
        file_id: Document identifier from search results.
        page:    Page number (1-indexed).
    """
    configurable = (config or {}).get("configurable", {})
    pages_dir = configurable.get("pages_dir")
    if not pages_dir:
        return "(pages_dir not configured)"

    # asyncio-APPROVED-1: to_thread wraps blocking page image file read
    img_bytes = await asyncio.to_thread(_load_page_image, Path(pages_dir), file_id, page)
    if img_bytes is None:
        return f"(image not available for {file_id} page {page})"
    return f"data:image/png;base64,{base64.b64encode(img_bytes).decode()}"


@tool
async def get_page_images_for_section(
    file_id: str,
    section_id: str,
    max_images: int = 5,
    config: RunnableConfig = None,
) -> str:
    """Get rendered page images for all pages in a section.

    Returns base64-encoded PNG data URIs separated by '---'.
    Pages containing tables or figures are returned first.
    Returns an informational message if no images are available.

    Args:
        file_id:    Document identifier.
        section_id: Section ID from get_document_structure.
        max_images: Maximum number of page images to return (default 5).
    """
    configurable = (config or {}).get("configurable", {})
    pages_dir = configurable.get("pages_dir")
    doc_list: list[dict] = configurable.get("doc_list") or []

    if not pages_dir:
        return "(pages_dir not configured)"

    page_start, page_end, err = _get_section_page_range(doc_list, file_id, section_id)
    if err:
        return err

    # asyncio-APPROVED-1: to_thread wraps blocking section image file reads
    images = await asyncio.to_thread(
        _load_section_images, Path(pages_dir), file_id, page_start, page_end, max_images
    )
    if not images:
        return f"(no images available for {file_id} section {section_id})"

    parts = [f"data:image/png;base64,{base64.b64encode(img).decode()}" for _, img in images]
    return "\n---\n".join(parts)


# ---------------------------------------------------------------------------
# Exported tool list for ToolNode construction
# ---------------------------------------------------------------------------

COLLECTION_TOOLS = [
    search_collection,
    read_section,
    read_pages,
    read_key_docs,
    read_page_image,
    get_page_images_for_section,
]
