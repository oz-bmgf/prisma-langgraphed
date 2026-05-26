from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, patch

from src.graph.agents.edison_graph import (
    edison_graph,
    route_rewrite_entry,
    edison_rewrite_query,
    edison_search,
    edison_finalise,
)
from src.graph.state import EdisonAgentState


def minimal_edison_state(**overrides) -> EdisonAgentState:
    base: EdisonAgentState = {
        "task_id": "ed-001",
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


def test_edison_graph_compiles():
    assert edison_graph is not None
    nodes = list(edison_graph.nodes.keys())
    assert "edison_search" in nodes
    assert "edison_finalise" in nodes


def test_route_rewrite_skips_when_flagged():
    state = minimal_edison_state(skip_rewrite=True)
    route = route_rewrite_entry(state)
    assert route == "edison_search"


def test_route_rewrite_rewrites_when_not_skipped():
    state = minimal_edison_state(skip_rewrite=False)
    route = route_rewrite_entry(state)
    assert route == "edison_rewrite_query"


@pytest.mark.asyncio
async def test_edison_rewrite_query_returns_rewritten():
    with patch("src.graph.subgraphs.edison.acall_llm", new=AsyncMock(return_value="artemisinin ACT malaria therapy")):
        state = minimal_edison_state()
        result = await edison_rewrite_query(state, {})
    assert "rewritten_query" in result
    assert result["rewritten_query"] == "artemisinin ACT malaria therapy"


@pytest.mark.asyncio
async def test_edison_rewrite_query_falls_back_on_error():
    with patch("src.graph.subgraphs.edison.acall_llm", new=AsyncMock(side_effect=Exception("LLM error"))):
        state = minimal_edison_state()
        result = await edison_rewrite_query(state, {})
    assert result["rewritten_query"] == state["original_query"]


@pytest.mark.asyncio
async def test_edison_search_success():
    fake_papers = [{"title": "Paper X", "abstract": "content", "year": 2022}]
    with patch("src.graph.subgraphs.edison.search_edison") as mock_tool:
        mock_tool.ainvoke = AsyncMock(return_value=fake_papers)
        state = minimal_edison_state(rewritten_query="artemisinin ACT")
        result = await edison_search(state, {})
    assert "papers" in result
    assert result["result_count"] == 1
    assert result["tool_traces"][0]["success"] is True


@pytest.mark.asyncio
async def test_edison_search_handles_error():
    with patch("src.graph.subgraphs.edison.search_edison") as mock_tool:
        mock_tool.ainvoke = AsyncMock(side_effect=Exception("API error"))
        state = minimal_edison_state()
        result = await edison_search(state, {})
    assert result["papers"] == []
    assert "errors" in result
    assert result["tool_traces"][0]["success"] is False


@pytest.mark.asyncio
async def test_edison_finalise_with_papers():
    state = minimal_edison_state(
        papers=[{"title": "Paper X"}],
        result_count=1,
        rewritten_query="artemisinin ACT malaria",
    )
    result = await edison_finalise(state, {})
    assert result["result"]["status"] == "ok"
    assert result["success"] is True


@pytest.mark.asyncio
async def test_edison_finalise_no_papers():
    state = minimal_edison_state(papers=[], result_count=0)
    result = await edison_finalise(state, {})
    assert result["result"]["status"] == "no_evidence"


@pytest.mark.asyncio
async def test_edison_graph_skip_rewrite():
    fake_papers = [{"title": "Paper X", "abstract": "content", "year": 2022}]
    with patch("src.graph.subgraphs.edison.search_edison") as mock_tool:
        mock_tool.ainvoke = AsyncMock(return_value=fake_papers)
        result_state = await edison_graph.ainvoke(minimal_edison_state(skip_rewrite=True))
    # When skip_rewrite, rewritten_query stays None
    assert result_state.get("rewritten_query") is None


@pytest.mark.asyncio
async def test_edison_graph_with_rewrite():
    fake_rewrite = "artemisinin ACT malaria treatment"
    fake_papers = [{"title": "Paper Y", "abstract": "content", "year": 2023}]
    with (
        patch("src.graph.subgraphs.edison.acall_llm", new=AsyncMock(return_value=fake_rewrite)),
        patch("src.graph.subgraphs.edison.search_edison") as mock_tool,
    ):
        mock_tool.ainvoke = AsyncMock(return_value=fake_papers)
        result_state = await edison_graph.ainvoke(minimal_edison_state(skip_rewrite=False))
    assert result_state.get("rewritten_query") is not None
    assert result_state.get("rewritten_query") == fake_rewrite
