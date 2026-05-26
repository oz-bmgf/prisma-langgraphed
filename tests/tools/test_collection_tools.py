"""Unit tests for src/tools/collection_tools.py.

Uses a MockSearchBackend that returns 2 fixed SearchResult objects.
No real file I/O; pages_dir is a tmp_path fixture.
"""
import base64
import json
from pathlib import Path

import pytest

from src.backends.base import SearchResult
from src.tools.collection_tools import (
    COLLECTION_TOOLS,
    get_page_images_for_section,
    read_key_docs,
    read_page_image,
    read_pages,
    read_section,
    search_collection,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


FIXED_RESULTS = [
    SearchResult(
        chunk_id="c1", text="Enrollment reached 80% of target.", score=0.91,
        file_id="INV-001__progress_2024", inv_id="INV-001", bow_id="BOW-A",
        page_start=3, page_end=4, doc_type="progress_report",
    ),
    SearchResult(
        chunk_id="c2", text="Budget utilisation is 73% year-to-date.", score=0.85,
        file_id="INV-001__budget_2024", inv_id="INV-001", bow_id="BOW-A",
        page_start=1, page_end=1, doc_type="budget",
    ),
]


class MockSearchBackend:
    async def search(self, query, **kwargs):
        return FIXED_RESULTS

    async def distinct_inv_ids(self):
        return ["INV-001"]

    async def distinct_bow_ids(self):
        return ["BOW-A"]

    async def count_by_bow_id(self):
        return {"BOW-A": 2}


def _config(extra: dict | None = None):
    base = {"configurable": {"search_backend": MockSearchBackend()}}
    if extra:
        base["configurable"].update(extra)
    return base


def _make_doc_list():
    return [
        {
            "file_id": "INV-001__progress_2024",
            "filename": "progress_2024.pdf",
            "inv_id": "INV-001",
            "doc_type": "progress_report",
            "summary": "Annual progress report.",
            "sections": [
                {"id": "S1", "label": "Executive Summary", "page_start": 1, "page_end": 2},
                {"id": "S2", "label": "Progress Narrative", "page_start": 3, "page_end": 5},
            ],
        },
        {
            "file_id": "INV-001__budget_2024",
            "filename": "budget_2024.xlsx",
            "inv_id": "INV-001",
            "doc_type": "budget",
            "summary": "Budget summary.",
            "sections": [],
        },
    ]


# ---------------------------------------------------------------------------
# search_collection
# ---------------------------------------------------------------------------


async def test_search_collection_returns_results():
    result = await search_collection.ainvoke({"query": "enrollment"}, config=_config())
    assert "INV-001__progress_2024" in result
    assert "Enrollment reached 80%" in result
    assert "score=0.910" in result


async def test_search_collection_no_results():
    class EmptyBackend:
        async def search(self, *a, **kw):
            return []
        async def distinct_inv_ids(self): return []
        async def distinct_bow_ids(self): return []
        async def count_by_bow_id(self): return {}

    result = await search_collection.ainvoke(
        {"query": "nothing"},
        config={"configurable": {"search_backend": EmptyBackend()}},
    )
    assert result == "(no results)"


async def test_search_collection_tool_count():
    assert len(COLLECTION_TOOLS) == 6


# ---------------------------------------------------------------------------
# read_section
# ---------------------------------------------------------------------------


async def test_read_section_found(tmp_path: Path):
    doc_dir = tmp_path / "INV-001__progress_2024"
    doc_dir.mkdir()
    for pg, text in [(1, "exec summary text"), (2, "more exec text")]:
        (doc_dir / f"p{pg:03d}.txt").write_text(text)

    config = _config({"pages_dir": str(tmp_path), "doc_list": _make_doc_list()})
    result = await read_section.ainvoke(
        {"file_id": "INV-001__progress_2024", "section_id": "S1"},
        config=config,
    )
    assert "exec summary text" in result
    assert "Page 1" in result


async def test_read_section_not_found(tmp_path: Path):
    config = _config({"pages_dir": str(tmp_path), "doc_list": _make_doc_list()})
    result = await read_section.ainvoke(
        {"file_id": "INV-001__progress_2024", "section_id": "SXXX"},
        config=config,
    )
    assert "not found" in result


async def test_read_section_no_pages_dir():
    config = _config({"doc_list": _make_doc_list()})
    result = await read_section.ainvoke(
        {"file_id": "INV-001__progress_2024", "section_id": "S1"},
        config=config,
    )
    assert "pages_dir not configured" in result


# ---------------------------------------------------------------------------
# read_pages
# ---------------------------------------------------------------------------


async def test_read_pages_returns_text(tmp_path: Path):
    doc_dir = tmp_path / "INV-001__progress_2024"
    doc_dir.mkdir()
    (doc_dir / "p003.txt").write_text("Enrollment data here.")

    config = _config({"pages_dir": str(tmp_path)})
    result = await read_pages.ainvoke(
        {"file_id": "INV-001__progress_2024", "page_start": 3, "page_end": 3},
        config=config,
    )
    assert "Enrollment data here." in result
    assert "Page 3" in result


async def test_read_pages_missing_dir(tmp_path: Path):
    config = _config({"pages_dir": str(tmp_path)})
    result = await read_pages.ainvoke(
        {"file_id": "NONEXISTENT_FILE", "page_start": 1, "page_end": 2},
        config=config,
    )
    assert "no pages directory" in result


# ---------------------------------------------------------------------------
# read_key_docs
# ---------------------------------------------------------------------------


async def test_read_key_docs_found():
    config = _config({"doc_list": _make_doc_list()})
    result = await read_key_docs.ainvoke({"inv_id": "INV-001"}, config=config)
    assert "INV-001__progress_2024" in result
    assert "progress_report" in result
    assert "budget" in result


async def test_read_key_docs_not_found():
    config = _config({"doc_list": _make_doc_list()})
    result = await read_key_docs.ainvoke({"inv_id": "INV-999"}, config=config)
    assert "no documents found" in result


async def test_read_key_docs_priority_order():
    config = _config({"doc_list": _make_doc_list()})
    result = await read_key_docs.ainvoke({"inv_id": "INV-001"}, config=config)
    # progress_report (priority 0) should appear before budget (priority 4)
    assert result.index("progress_report") < result.index("budget")


# ---------------------------------------------------------------------------
# read_page_image
# ---------------------------------------------------------------------------


async def test_read_page_image_found(tmp_path: Path):
    doc_dir = tmp_path / "INV-001__progress_2024"
    doc_dir.mkdir()
    png_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
    (doc_dir / "p003.png").write_bytes(png_bytes)

    config = _config({"pages_dir": str(tmp_path)})
    result = await read_page_image.ainvoke(
        {"file_id": "INV-001__progress_2024", "page": 3},
        config=config,
    )
    assert result.startswith("data:image/png;base64,")
    decoded = base64.b64decode(result.split(",", 1)[1])
    assert decoded == png_bytes


async def test_read_page_image_not_found(tmp_path: Path):
    config = _config({"pages_dir": str(tmp_path)})
    result = await read_page_image.ainvoke(
        {"file_id": "NONEXISTENT", "page": 1},
        config=config,
    )
    assert "not available" in result


# ---------------------------------------------------------------------------
# get_page_images_for_section
# ---------------------------------------------------------------------------


async def test_get_page_images_for_section(tmp_path: Path):
    doc_dir = tmp_path / "INV-001__progress_2024"
    doc_dir.mkdir()
    png_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 50
    for pg in [1, 2]:
        (doc_dir / f"p{pg:03d}.png").write_bytes(png_bytes)

    config = _config({"pages_dir": str(tmp_path), "doc_list": _make_doc_list()})
    result = await get_page_images_for_section.ainvoke(
        {"file_id": "INV-001__progress_2024", "section_id": "S1"},
        config=config,
    )
    assert "data:image/png;base64," in result


async def test_get_page_images_section_not_found(tmp_path: Path):
    config = _config({"pages_dir": str(tmp_path), "doc_list": _make_doc_list()})
    result = await get_page_images_for_section.ainvoke(
        {"file_id": "INV-001__progress_2024", "section_id": "SXXX"},
        config=config,
    )
    assert "not found" in result
