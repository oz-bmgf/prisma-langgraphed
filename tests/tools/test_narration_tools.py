"""Unit tests for src/tools/narration_tools.py."""
from pathlib import Path

import pytest

from src.backends.base import SearchResult
from src.tools.narration_tools import (
    NARRATION_TOOLS,
    get_inv_metadata,
    list_filtered_investments,
    read_evidence_pack,
    read_primary_document,
    search_within_scope,
    verify_claim,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


FIXED_RESULTS = [
    SearchResult(
        chunk_id="c1", text="Key results for INV-003.", score=0.90,
        file_id="INV-003__progress_2024", inv_id="INV-003", bow_id="BOW-C",
        page_start=2, page_end=3, doc_type="progress_report",
    ),
]


class MockSearchBackend:
    async def search(self, query, **kwargs):
        return FIXED_RESULTS
    async def distinct_inv_ids(self): return ["INV-003"]
    async def distinct_bow_ids(self): return ["BOW-C"]
    async def count_by_bow_id(self): return {"BOW-C": 1}


def _investment_scoring():
    return {
        "INV-003": {
            "title": "Malaria Vaccine Trial Phase 3",
            "org": "Wellcome Sanger Institute",
            "approved_amount": 12_000_000,
            "paid_amount": 8_500_000,
            "allocation": 2_000_000,
            "posture": "active",
            "execution": "on_track",
            "impact": "high",
            "bow_id": "BOW-C",
            "bow_name": "Malaria Vaccines",
            "managing_team": "Discovery and Translational Sciences",
        },
        "INV-004": {
            "title": "Polio Eradication Support",
            "org": "WHO",
            "approved_amount": 5_000_000,
            "paid_amount": 3_200_000,
            "allocation": 900_000,
            "posture": "active",
            "execution": "at_risk",
        },
    }


def _config(extra: dict | None = None):
    base = {
        "configurable": {
            "search_backend": MockSearchBackend(),
            "investment_scoring": _investment_scoring(),
            "investment_intelligence": {
                "INV-003": {"key_results": "Phase 3 enrollment complete."}
            },
            "scope_outputs": {},
            "doc_list": [],
        }
    }
    if extra:
        base["configurable"].update(extra)
    return base


# ---------------------------------------------------------------------------
# list_filtered_investments
# ---------------------------------------------------------------------------


async def test_list_filtered_investments_all():
    result = await list_filtered_investments.ainvoke({}, config=_config())
    assert "INV-003" in result
    assert "INV-004" in result
    assert "Malaria Vaccine Trial Phase 3" in result


async def test_list_filtered_investments_with_subset():
    config = _config({"relevance_subset": {"INV-003"}})
    result = await list_filtered_investments.ainvoke({}, config=config)
    assert "INV-003" in result
    assert "INV-004" not in result


async def test_list_filtered_investments_empty():
    config = _config({"investment_scoring": {}})
    result = await list_filtered_investments.ainvoke({}, config=config)
    assert "no investments in scope" in result


# ---------------------------------------------------------------------------
# get_inv_metadata
# ---------------------------------------------------------------------------


async def test_get_inv_metadata_found():
    result = await get_inv_metadata.ainvoke({"inv_id": "INV-003"}, config=_config())
    assert "Malaria Vaccine Trial Phase 3" in result
    assert "approved_amount_M: 12.0" in result
    assert "posture: active" in result


async def test_get_inv_metadata_includes_key_results():
    result = await get_inv_metadata.ainvoke({"inv_id": "INV-003"}, config=_config())
    assert "Phase 3 enrollment complete." in result


async def test_get_inv_metadata_not_found():
    result = await get_inv_metadata.ainvoke({"inv_id": "INV-999"}, config=_config())
    assert "not found" in result


# ---------------------------------------------------------------------------
# search_within_scope
# ---------------------------------------------------------------------------


async def test_search_within_scope_returns_results():
    result = await search_within_scope.ainvoke(
        {"query": "trial enrollment"}, config=_config()
    )
    assert "INV-003__progress_2024" in result
    assert "Key results for INV-003." in result


async def test_search_within_scope_with_inv_id():
    result = await search_within_scope.ainvoke(
        {"query": "trial enrollment", "inv_id": "INV-003"}, config=_config()
    )
    assert "INV-003" in result


async def test_search_within_scope_with_subset():
    config = _config({"relevance_subset": {"INV-003"}})
    result = await search_within_scope.ainvoke({"query": "trial"}, config=config)
    assert isinstance(result, str)


# ---------------------------------------------------------------------------
# read_evidence_pack
# ---------------------------------------------------------------------------


async def test_read_evidence_pack_metadata_fallback():
    result = await read_evidence_pack.ainvoke({"inv_id": "INV-003"}, config=_config())
    assert "Malaria Vaccine Trial Phase 3" in result


async def test_read_evidence_pack_scope_body():
    scope_outputs = {
        "SCOPE-1": {
            "investment_sections": {
                "INV-003": "Full scope body with detailed analysis of the Phase 3 trial."
            }
        }
    }
    config = _config({"scope_outputs": scope_outputs})
    result = await read_evidence_pack.ainvoke({"inv_id": "INV-003"}, config=config)
    assert "Full scope body" in result
    assert "SCOPE-1" in result


async def test_read_evidence_pack_not_found():
    result = await read_evidence_pack.ainvoke({"inv_id": "INV-999"}, config=_config())
    assert "not found" in result


# ---------------------------------------------------------------------------
# read_primary_document
# ---------------------------------------------------------------------------


async def test_read_primary_document(tmp_path: Path):
    doc_dir = tmp_path / "INV-003__progress_2024"
    doc_dir.mkdir()
    for pg in range(1, 6):
        (doc_dir / f"p{pg:03d}.txt").write_text(f"Page {pg} content.")

    config = _config({"pages_dir": str(tmp_path)})
    result = await read_primary_document.ainvoke(
        {"file_id": "INV-003__progress_2024"}, config=config
    )
    assert "Page 1 content." in result


async def test_read_primary_document_with_pages(tmp_path: Path):
    doc_dir = tmp_path / "INV-003__progress_2024"
    doc_dir.mkdir()
    (doc_dir / "p004.txt").write_text("Page 4 clinical data.")
    (doc_dir / "p005.txt").write_text("Page 5 conclusions.")

    config = _config({"pages_dir": str(tmp_path)})
    result = await read_primary_document.ainvoke(
        {"file_id": "INV-003__progress_2024", "pages": "4-5"}, config=config
    )
    assert "Page 4 clinical data." in result
    assert "Page 5 conclusions." in result


async def test_read_primary_document_invalid_pages(tmp_path: Path):
    config = _config({"pages_dir": str(tmp_path)})
    result = await read_primary_document.ainvoke(
        {"file_id": "INV-003__progress_2024", "pages": "badformat"}, config=config
    )
    assert "invalid pages format" in result


# ---------------------------------------------------------------------------
# verify_claim
# ---------------------------------------------------------------------------


async def test_verify_claim_no_llm():
    result = await verify_claim.ainvoke({"claim": "Enrollment reached 80%."}, config=_config())
    assert "verify_claim" in result or "acall_llm" in result


async def test_verify_claim_with_mock_llm():
    async def mock_acall_llm(messages, model):
        return "SUPPORTED\nEvidence confirms 80% enrollment milestone was met in Q3."

    config = _config({"acall_llm": mock_acall_llm})
    result = await verify_claim.ainvoke({"claim": "Enrollment reached 80%."}, config=config)
    assert "SUPPORTED" in result


# ---------------------------------------------------------------------------
# Tool count
# ---------------------------------------------------------------------------


async def test_narration_tools_count():
    assert len(NARRATION_TOOLS) == 6
