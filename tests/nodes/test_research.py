"""Unit tests for src/graph/nodes/research.py."""
from __future__ import annotations

import sys
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.graph.nodes.research import research


def _make_state(**overrides) -> dict:
    base = {
        "program": "Malaria",
        "research_model": "claude-sonnet-4-6",
        "synthesis_model": "claude-sonnet-4-6",
        "research_plan": [
            {"id": "RQ-001", "query": "Q1", "type": "slr", "priority": "important", "linked_scope": "S1"},
        ],
        "output_dir": "/tmp/output",
        "threads_dir": None,
    }
    base.update(overrides)
    return base


_MOCK_RESULT = {
    "research_results": [{"task_id": "RQ-001", "status": "ok", "findings": "..."}],
    "dispatch_results": [{"task_id": "RQ-001", "findings": "..."}],
    "edison_results": [],
    "errors": [],
}


# ---------------------------------------------------------------------------
# invokes research_graph
# ---------------------------------------------------------------------------


async def test_invokes_research_graph():
    mock_graph = MagicMock()
    mock_graph.ainvoke = AsyncMock(return_value=_MOCK_RESULT)
    mock_module = MagicMock()
    mock_module.research_graph = mock_graph

    with __import__("unittest.mock", fromlist=["patch"]).patch.dict(
        sys.modules, {"src.graph.subgraphs.research": mock_module}
    ):
        result = await research(_make_state(), {})

    mock_graph.ainvoke.assert_called_once()


# ---------------------------------------------------------------------------
# returns expected keys
# ---------------------------------------------------------------------------


async def test_returns_expected_keys():
    mock_graph = MagicMock()
    mock_graph.ainvoke = AsyncMock(return_value=_MOCK_RESULT)
    mock_module = MagicMock()
    mock_module.research_graph = mock_graph

    from unittest.mock import patch
    with patch.dict(sys.modules, {"src.graph.subgraphs.research": mock_module}):
        result = await research(_make_state(), {})

    assert "research_dir" in result
    assert "research_results" in result
    assert "dispatch_results" in result
    assert "edison_results" in result
    assert "research_ok_count" in result
    assert "errors" in result


# ---------------------------------------------------------------------------
# passes plan + research_dir to subgraph
# ---------------------------------------------------------------------------


async def test_passes_correct_fields_to_subgraph():
    captured = {}

    async def _capture(input_dict, config):
        captured.update(input_dict)
        return _MOCK_RESULT

    mock_graph = MagicMock()
    mock_graph.ainvoke = _capture
    mock_module = MagicMock()
    mock_module.research_graph = mock_graph

    from unittest.mock import patch
    with patch.dict(sys.modules, {"src.graph.subgraphs.research": mock_module}):
        await research(_make_state(output_dir="/tmp/out"), {})

    assert captured["research_plan"][0]["id"] == "RQ-001"
    assert captured["research_dir"].endswith("/research")
    assert captured["research_results"] == []


# ---------------------------------------------------------------------------
# counts ok results
# ---------------------------------------------------------------------------


async def test_counts_ok_results():
    mock_graph = MagicMock()
    mock_graph.ainvoke = AsyncMock(return_value={
        **_MOCK_RESULT,
        "research_results": [
            {"task_id": "RQ-001", "status": "ok"},
            {"task_id": "RQ-002", "status": "error"},
            {"task_id": "RQ-003", "status": "ok"},
        ],
    })
    mock_module = MagicMock()
    mock_module.research_graph = mock_graph

    from unittest.mock import patch
    with patch.dict(sys.modules, {"src.graph.subgraphs.research": mock_module}):
        result = await research(_make_state(), {})

    assert result["research_ok_count"] == 2


# ---------------------------------------------------------------------------
# empty plan — still works
# ---------------------------------------------------------------------------


async def test_empty_plan_works():
    mock_graph = MagicMock()
    mock_graph.ainvoke = AsyncMock(return_value={
        "research_results": [], "dispatch_results": [], "edison_results": [], "errors": []
    })
    mock_module = MagicMock()
    mock_module.research_graph = mock_graph

    from unittest.mock import patch
    with patch.dict(sys.modules, {"src.graph.subgraphs.research": mock_module}):
        result = await research(_make_state(research_plan=[]), {})

    assert result["research_results"] == []
    assert result["research_ok_count"] == 0
