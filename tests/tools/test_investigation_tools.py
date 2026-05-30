"""Unit tests for src/tools/investigation_tools.py."""
import json
from pathlib import Path

import pytest

from src.backends.base import SearchResult
from src.tools.investigation_tools import (
    INVESTIGATION_TOOLS,
    compute,
    get_document_structure,
    list_documents,
    read_document,
    read_document_summary,
    search_investment,
    search_portfolio,
    submit_findings,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


FIXED_RESULTS = [
    SearchResult(
        chunk_id="c1", text="Phase 2 trial completed.", score=0.88,
        file_id="INV-002__progress_2024", inv_id="INV-002", bow_id="BOW-B",
        page_start=5, page_end=6, doc_type="progress_report",
    ),
    SearchResult(
        chunk_id="c2", text="Budget overspend detected.", score=0.81,
        file_id="INV-002__budget_2024", inv_id="INV-002", bow_id="BOW-B",
        page_start=2, page_end=2, doc_type="budget",
    ),
]


class MockSearchBackend:
    async def search(self, query, **kwargs):
        return FIXED_RESULTS
    async def distinct_inv_ids(self): return ["INV-002"]
    async def distinct_bow_ids(self): return ["BOW-B"]
    async def count_by_bow_id(self): return {"BOW-B": 2}


def _doc_list():
    return [
        {
            "file_id": "INV-002__progress_2024",
            "filename": "progress_2024.pdf",
            "inv_id": "INV-002",
            "doc_type": "progress_report",
            "total_pages": 10,
            "date": "2024-06-01",
            "summary": "Progress report summary.",
            "sections": [
                {"id": "A1", "label": "Background", "page_start": 1, "page_end": 3,
                 "has_table": True, "has_figure": False},
                {"id": "A2", "label": "Results", "page_start": 5, "page_end": 7,
                 "has_table": False, "has_figure": True},
            ],
        },
        {
            "file_id": "INV-002__budget_2024",
            "filename": "budget_2024.xlsx",
            "inv_id": "INV-002",
            "doc_type": "budget",
            "total_pages": 3,
            "date": "2024-01-15",
            "summary": "Budget summary.",
            "sections": [],
        },
    ]


def _config(extra: dict | None = None):
    base = {
        "configurable": {
            "search_backend": MockSearchBackend(),
            "inv_id": "INV-002",
            "doc_list": _doc_list(),
        }
    }
    if extra:
        base["configurable"].update(extra)
    return base


# ---------------------------------------------------------------------------
# search_investment
# ---------------------------------------------------------------------------


async def test_search_investment_uses_inv_id():
    result = await search_investment.ainvoke({"query": "trial results"}, config=_config())
    assert "INV-002__progress_2024" in result
    assert "Phase 2 trial completed." in result


async def test_search_investment_tool_count():
    assert len(INVESTIGATION_TOOLS) == 10


# ---------------------------------------------------------------------------
# search_portfolio
# ---------------------------------------------------------------------------


async def test_search_portfolio_returns_results():
    result = await search_portfolio.ainvoke({"query": "malaria trial"}, config=_config())
    assert "Phase 2 trial completed." in result


async def test_search_portfolio_with_collection_filter():
    result = await search_portfolio.ainvoke(
        {"query": "strategy", "collection": "strategy"}, config=_config()
    )
    assert result  # just check it doesn't raise


# ---------------------------------------------------------------------------
# read_document
# ---------------------------------------------------------------------------


async def test_read_document_by_section(tmp_path: Path):
    doc_dir = tmp_path / "INV-002__progress_2024"
    doc_dir.mkdir()
    (doc_dir / "p005.txt").write_text("Results section content.")
    (doc_dir / "p006.txt").write_text("More results.")

    config = _config({"pages_dir": str(tmp_path)})
    result = await read_document.ainvoke(
        {"file_id": "INV-002__progress_2024", "section_id": "A2"},
        config=config,
    )
    assert "Results section content." in result


async def test_read_document_by_page_range(tmp_path: Path):
    doc_dir = tmp_path / "INV-002__progress_2024"
    doc_dir.mkdir()
    (doc_dir / "p002.txt").write_text("Page two content.")

    config = _config({"pages_dir": str(tmp_path)})
    result = await read_document.ainvoke(
        {"file_id": "INV-002__progress_2024", "page_start": 2, "page_end": 2},
        config=config,
    )
    assert "Page two content." in result


async def test_read_document_no_params():
    config = _config({"pages_dir": "/tmp"})
    result = await read_document.ainvoke({"file_id": "INV-002__progress_2024"}, config=config)
    assert "provide either" in result


# ---------------------------------------------------------------------------
# compute
# ---------------------------------------------------------------------------


async def test_compute_simple_arithmetic():
    result = await compute.ainvoke({"question": "2 + 2"}, config=_config())
    assert "4" in result


async def test_compute_math_expression():
    result = await compute.ainvoke({"question": "1000000 / 3"}, config=_config())
    assert result  # just check no crash


async def test_compute_passthrough():
    result = await compute.ainvoke(
        {"question": "What is the annual burn rate?", "data": "Budget: $7.4M over 3 years"},
        config=_config(),
    )
    assert "annual burn rate" in result.lower()
    assert "7.4" in result  # model may format as "$7.4M", "$7.4 million", etc.


# ---------------------------------------------------------------------------
# submit_findings
# ---------------------------------------------------------------------------


async def test_submit_findings_returns_json():
    findings = [
        {
            "statement": "Enrollment is on track.",
            "finding_type": "strength",
            "severity": "on_track_strong",
            "confidence": "high",
            "evidence_refs": ["§p3"],
            "rationale": "80% target met.",
            "numerical_claims": [],
        }
    ]
    overall = {"status": "on_track", "summary": "Good progress.", "evidence_gaps": []}
    result = await submit_findings.ainvoke(
        {"findings": findings, "overall_assessment": overall},
        config=_config(),
    )
    parsed = json.loads(result)
    assert parsed["findings"][0]["statement"] == "Enrollment is on track."
    assert parsed["overall_assessment"]["status"] == "on_track"


# ---------------------------------------------------------------------------
# list_documents
# ---------------------------------------------------------------------------


async def test_list_documents_this_investment():
    result = await list_documents.ainvoke({"scope": "this_investment"}, config=_config())
    assert "INV-002__progress_2024" in result
    assert "INV-002__budget_2024" in result


async def test_list_documents_with_doc_type_filter():
    result = await list_documents.ainvoke(
        {"scope": "this_investment", "doc_type": "progress_report"},
        config=_config(),
    )
    assert "INV-002__progress_2024" in result
    assert "INV-002__budget_2024" not in result


async def test_list_documents_portfolio():
    result = await list_documents.ainvoke({"scope": "portfolio"}, config=_config())
    assert "progress_report" in result


# ---------------------------------------------------------------------------
# read_document_summary
# ---------------------------------------------------------------------------


async def test_read_document_summary_from_doc_list():
    result = await read_document_summary.ainvoke(
        {"file_id": "INV-002__progress_2024"}, config=_config()
    )
    assert "Progress report summary." in result
    assert "doc_type: progress_report" in result


async def test_read_document_summary_not_found():
    result = await read_document_summary.ainvoke({"file_id": "NONEXISTENT"}, config=_config())
    assert "not found" in result


# ---------------------------------------------------------------------------
# get_document_structure
# ---------------------------------------------------------------------------


async def test_get_document_structure_from_doc_list():
    result = await get_document_structure.ainvoke(
        {"file_id": "INV-002__progress_2024"}, config=_config()
    )
    assert "A1" in result
    assert "Background" in result
    assert "[TABLE]" in result
    assert "A2" in result
    assert "[FIGURE]" in result


async def test_get_document_structure_no_sections():
    result = await get_document_structure.ainvoke(
        {"file_id": "INV-002__budget_2024"}, config=_config()
    )
    assert "no section structure" in result
