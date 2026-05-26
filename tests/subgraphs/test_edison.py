"""Tests for src/graph/subgraphs/edison.py — canonical Edison subgraph location."""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from src.graph.subgraphs.edison import (
    edison_finalise,
    edison_graph,
    edison_rewrite_query,
    edison_search,
    route_rewrite_entry,
)
from src.graph.state import EdisonAgentState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _state(**overrides) -> dict:
    base: dict = {
        "task_id": "E1",
        "original_query": "efficacy of artemisinin combination therapy",
        "context": "",
        "skip_rewrite": False,
        "rewritten_query": None,
        "papers": None,
        "result_count": 0,
        "tool_traces": [],
        "result": None,
        "success": False,
        "error_message": None,
        "errors": [],
    }
    base.update(overrides)
    return base


_PAPERS = [
    {"title": "ACT efficacy RCT", "abstract": "ACT reduced parasitemia.", "year": 2022},
]


# ---------------------------------------------------------------------------
# test_edison_dispatch_skips_completed
# Task spec: route_rewrite_entry routes correctly; idempotency in search node
# ---------------------------------------------------------------------------


def test_edison_route_skips_rewrite_when_flagged():
    """route_rewrite_entry returns 'edison_search' when skip_rewrite=True."""
    state = _state(skip_rewrite=True)
    assert route_rewrite_entry(state) == "edison_search"


def test_edison_route_rewrites_when_not_flagged():
    """route_rewrite_entry returns 'edison_rewrite_query' when skip_rewrite=False."""
    state = _state(skip_rewrite=False)
    assert route_rewrite_entry(state) == "edison_rewrite_query"


async def test_edison_rewrite_skips_when_already_done():
    """edison_rewrite_query returns {} when rewritten_query already set."""
    state = _state(rewritten_query="already rewritten")
    result = await edison_rewrite_query(state, {})
    assert result == {}


async def test_edison_search_skips_when_already_done():
    """edison_search returns {} when papers already set (idempotency guard)."""
    state = _state(papers=[{"title": "Cached paper"}])
    result = await edison_search(state, {})
    assert result == {}


# ---------------------------------------------------------------------------
# test_edison_worker_pure
# Task spec: edison_search returns plain dict, no file side effects
# ---------------------------------------------------------------------------


async def test_edison_search_pure_success():
    """edison_search returns papers and trace on success."""
    fake_papers = [{"title": "Paper X", "abstract": "content", "year": 2022}]
    state = _state(rewritten_query="artemisinin ACT")
    with patch("src.graph.subgraphs.edison.search_edison") as mock_tool:
        mock_tool.ainvoke = AsyncMock(return_value=fake_papers)
        result = await edison_search(state, {})

    assert isinstance(result, dict)
    assert result["papers"] == fake_papers
    assert result["result_count"] == 1
    assert result["tool_traces"][0]["success"] is True


async def test_edison_search_error_surfaces_to_errors_field():
    """edison_search tool failure goes into errors[], not raises."""
    state = _state()
    with patch("src.graph.subgraphs.edison.search_edison") as mock_tool:
        mock_tool.ainvoke = AsyncMock(side_effect=RuntimeError("API error"))
        result = await edison_search(state, {})

    assert "errors" in result
    assert result["papers"] == []
    assert result["tool_traces"][0]["success"] is False


# ---------------------------------------------------------------------------
# test_edison_graph_has_all_nodes
# ---------------------------------------------------------------------------


def test_edison_graph_has_all_nodes():
    """Compiled graph nodes match topology declared in module docstring."""
    graph_nodes = set(edison_graph.get_graph().nodes.keys()) - {"__start__", "__end__"}
    expected = {
        "edison_rewrite_query",
        "edison_search",
        "edison_finalise",
    }
    assert graph_nodes == expected


def test_edison_graph_compiles():
    assert edison_graph is not None


# ---------------------------------------------------------------------------
# test_edison_handles_empty_results
# ---------------------------------------------------------------------------


async def test_edison_finalise_no_papers_sets_no_evidence_status():
    """edison_finalise sets status: no_evidence when no papers and no errors."""
    state = _state(papers=[], errors=[])
    result = await edison_finalise(state, {})
    assert result["result"]["status"] == "no_evidence"
    assert result["result"]["success"] is True


async def test_edison_finalise_with_papers_sets_ok_status():
    """edison_finalise sets status: ok when papers are found."""
    state = _state(papers=_PAPERS)
    result = await edison_finalise(state, {})
    assert result["result"]["status"] == "ok"
    assert result["result"]["success"] is True


async def test_edison_finalise_error_sets_error_status():
    """edison_finalise sets status: error when there are errors and no papers."""
    state = _state(papers=[], errors=["edison_search: API error"])
    result = await edison_finalise(state, {})
    assert result["result"]["status"] == "error"
    assert result["result"]["success"] is False


async def test_edison_rewrite_query_error_surfaces_and_keeps_fallback():
    """edison_rewrite_query LLM error surfaces to errors[] and falls back to original."""
    state = _state()
    with patch("src.graph.subgraphs.edison.acall_llm", new=AsyncMock(side_effect=RuntimeError("LLM down"))):
        result = await edison_rewrite_query(state, {})
    assert result["rewritten_query"] == state["original_query"]
    assert any("edison_rewrite_query" in e for e in result.get("errors", []))


# ---------------------------------------------------------------------------
# config threading
# ---------------------------------------------------------------------------


async def test_edison_rewrite_query_passes_config_to_acall_llm():
    """acall_llm must receive config= so LLM calls appear as child spans."""
    captured: list[dict] = []

    async def _mock_acall(*args, config=None, **kwargs):
        captured.append({"config": config})
        return "rewritten query"

    state = _state()
    with patch("src.graph.subgraphs.edison.acall_llm", side_effect=_mock_acall):
        await edison_rewrite_query(state, {"configurable": {"research_model": "test-model"}})

    assert len(captured) == 1
    assert captured[0]["config"] is not None


# ---------------------------------------------------------------------------
# Full dry-run through compiled graph
# ---------------------------------------------------------------------------


async def test_edison_graph_skip_rewrite_path():
    """End-to-end: skip_rewrite=True bypasses rewrite node."""
    fake_papers = [{"title": "Paper X", "abstract": "content", "year": 2022}]
    with patch("src.graph.subgraphs.edison.search_edison") as mock_tool:
        mock_tool.ainvoke = AsyncMock(return_value=fake_papers)
        final = await edison_graph.ainvoke(_state(skip_rewrite=True))
    assert final["result"]["status"] == "ok"
    assert final.get("rewritten_query") is None


async def test_edison_graph_with_rewrite_path():
    """End-to-end: rewrite runs when skip_rewrite=False."""
    fake_rewrite = "artemisinin ACT malaria treatment"
    fake_papers = [{"title": "Paper Y", "abstract": "content", "year": 2023}]
    with (
        patch("src.graph.subgraphs.edison.acall_llm", new=AsyncMock(return_value=fake_rewrite)),
        patch("src.graph.subgraphs.edison.search_edison") as mock_tool,
    ):
        mock_tool.ainvoke = AsyncMock(return_value=fake_papers)
        final = await edison_graph.ainvoke(_state(skip_rewrite=False))
    assert final["rewritten_query"] == fake_rewrite
    assert final["result"]["status"] == "ok"
