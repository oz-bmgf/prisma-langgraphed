"""Unit tests for src/graph/nodes/analyze.py."""
from __future__ import annotations

import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.graph.nodes.analyze import analyze


def _make_state(**overrides) -> dict:
    base = {
        "program": "Malaria",
        "run_name": "test-run",
        "collection_name": "malaria",
        "base_dir": "/tmp/base",
        "ingested_dir": "/tmp/ingested",
        "doc_list": [{"file_id": "f1"}],
        "investment_scoring": {"INV-01": 0.9},
        "bow_investment_map": {"BOW-A": ["INV-01"]},
        "investment_intelligence": {"INV-01": {}},
        "chunks_json_path": "/tmp/chunks.json",
        "pages_dir": "/tmp/pages",
        "research_model": "claude-sonnet-4-6",
        "synthesis_model": "claude-sonnet-4-6",
        "focus": None,
        "focus_bows": None,
        "aux_collections": None,
        "threads_dir": None,
        "orientation_summary": None,
        "scopes": None,
        "scope_timelines": None,
        "clusters": None,
        "scope_outputs": None,
        "analyst_report": None,
        "final_report_md": None,
        "numerical_provenance": None,
        "verification_sources": None,
        "run_meta": None,
    }
    base.update(overrides)
    return base


_MOCK_RESULT = {
    "threads_dir": "/tmp/threads",
    "final_report_md": "# Report",
    "analyst_report": {"threads": []},
    "scope_outputs": [],
    "scopes": [],
    "scope_timelines": {},
    "evidence_packs": [],
    "link_assessments": [],
    "science_results": [],
    "scope_decisions": [],
    "errors": [],
}


# ---------------------------------------------------------------------------
# invokes analyze_graph
# ---------------------------------------------------------------------------


async def test_invokes_analyze_graph():
    mock_graph = MagicMock()
    mock_graph.ainvoke = AsyncMock(return_value=_MOCK_RESULT)

    mock_module = MagicMock()
    mock_module.analyze_graph = mock_graph

    with patch.dict(sys.modules, {"src.graph.subgraphs.analyze": mock_module}):
        result = await analyze(_make_state(), {})

    mock_graph.ainvoke.assert_called_once()


# ---------------------------------------------------------------------------
# asserts on missing collection inputs
# ---------------------------------------------------------------------------


async def test_asserts_doc_list_not_none():
    with pytest.raises(AssertionError, match="load_collection"):
        await analyze(_make_state(doc_list=None), {})


async def test_asserts_investment_scoring_not_none():
    with pytest.raises(AssertionError, match="load_collection"):
        await analyze(_make_state(investment_scoring=None), {})


async def test_asserts_chunks_json_path_not_none():
    with pytest.raises(AssertionError, match="load_collection"):
        await analyze(_make_state(chunks_json_path=None), {})


# ---------------------------------------------------------------------------
# merges subgraph outputs back into WorkflowState fields
# ---------------------------------------------------------------------------


async def test_merges_outputs():
    mock_graph = MagicMock()
    mock_graph.ainvoke = AsyncMock(return_value={
        **_MOCK_RESULT,
        "final_report_md": "# Enriched",
        "analyst_report": {"threads": [{"id": "S1"}]},
        "errors": ["minor warning"],
    })
    mock_module = MagicMock()
    mock_module.analyze_graph = mock_graph

    with patch.dict(sys.modules, {"src.graph.subgraphs.analyze": mock_module}):
        result = await analyze(_make_state(), {})

    assert result["final_report_md"] == "# Enriched"
    assert result["analyst_report"] == {"threads": [{"id": "S1"}]}
    assert result["errors"] == ["minor warning"]


# ---------------------------------------------------------------------------
# passes correct fields to subgraph
# ---------------------------------------------------------------------------


async def test_passes_correct_fields_to_subgraph():
    mock_graph = MagicMock()
    captured_input = {}

    async def _capture(input_dict, config):
        captured_input.update(input_dict)
        return _MOCK_RESULT

    mock_graph.ainvoke = _capture
    mock_module = MagicMock()
    mock_module.analyze_graph = mock_graph

    with patch.dict(sys.modules, {"src.graph.subgraphs.analyze": mock_module}):
        await analyze(_make_state(focus="vaccines", focus_bows=["BOW-A"]), {})

    assert captured_input["program"] == "Malaria"
    assert captured_input["focus"] == "vaccines"
    assert captured_input["focus_bows"] == ["BOW-A"]
    assert captured_input["research_model"] == "claude-sonnet-4-6"
    assert captured_input["evidence_packs"] == []
