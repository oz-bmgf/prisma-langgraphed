from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, patch

from src.graph.agents.slr_graph import (
    slr_graph,
    slr_plan_sources,  # conditional edge function
    slr_fetch_source,
    slr_collect_papers,
    slr_finalise,
)
from src.graph.state import SLRAgentState, SLRFetchState


def minimal_slr_state(**overrides) -> SLRAgentState:
    base: SLRAgentState = {
        "task_id": "test-001",
        "query": "malaria net efficacy",
        "context": "",
        "top_k": 5,
        "openalex_results": [],
        "asta_results": [],
        "merged_papers": None,
        "synthesis": None,
        "source_count": 0,
        "search_strategy": "none",
        "tool_traces": [],
        "result": None,
        "success": False,
        "error_message": None,
        "errors": [],
        "expanded_queries": None,
    }
    base.update(overrides)
    return base


def test_slr_graph_compiles():
    assert slr_graph is not None
    nodes = list(slr_graph.nodes.keys())
    # slr_plan_sources is a conditional edge function, not a node
    assert "slr_fetch_source" in nodes
    assert "slr_collect_papers" in nodes
    assert "slr_synthesise" in nodes
    assert "slr_finalise" in nodes


@pytest.mark.asyncio
async def test_slr_plan_sources_returns_two_sends():
    from langgraph.types import Send
    state = minimal_slr_state()
    result = await slr_plan_sources(state)
    assert isinstance(result, list)
    assert len(result) == 2
    sources = {s.arg["source"] for s in result}
    assert sources == {"openalex", "asta"}


@pytest.mark.asyncio
async def test_slr_fetch_source_openalex_success():
    fake_papers = [{"title": "Paper A", "abstract": "abc", "year": 2020}]
    with patch("src.graph.subgraphs.slr.search_openalex") as mock_tool:
        mock_tool.ainvoke = AsyncMock(return_value=fake_papers)
        state: SLRFetchState = {"source": "openalex", "query": "malaria", "top_k": 5, "result": None}
        result = await slr_fetch_source(state, {})
    assert "openalex_results" in result
    assert len(result["openalex_results"]) >= 1
    assert result["tool_traces"][0]["success"] is True


@pytest.mark.asyncio
async def test_slr_fetch_source_asta_success():
    fake_papers = [{"title": "Paper B", "abstract": "xyz", "year": 2021}]
    with patch("src.graph.subgraphs.slr.search_asta") as mock_tool:
        mock_tool.ainvoke = AsyncMock(return_value=fake_papers)
        state: SLRFetchState = {"source": "asta", "query": "malaria", "top_k": 5, "result": None}
        result = await slr_fetch_source(state, {})
    assert "asta_results" in result
    assert len(result["asta_results"]) >= 1
    assert result["tool_traces"][0]["success"] is True


@pytest.mark.asyncio
async def test_slr_fetch_source_handles_error():
    with patch("src.graph.subgraphs.slr.search_openalex") as mock_tool:
        mock_tool.ainvoke = AsyncMock(side_effect=Exception("network error"))
        state: SLRFetchState = {"source": "openalex", "query": "malaria", "top_k": 5, "result": None}
        result = await slr_fetch_source(state, {})
    assert "errors" in result
    assert len(result["errors"]) > 0
    assert result["tool_traces"][0]["success"] is False


@pytest.mark.asyncio
async def test_slr_collect_papers_deduplicates():
    paper_a = {"title": "Paper A", "abstract": "abc"}
    paper_b = {"title": "Paper B", "abstract": "def"}
    paper_b_dup = {"title": "Paper B", "abstract": "def (dup)"}  # same title
    state = minimal_slr_state(
        openalex_results=[paper_a, paper_b],
        asta_results=[paper_b_dup, paper_a],  # duplicates
    )
    result = await slr_collect_papers(state, {})
    assert result["source_count"] == 2  # deduplicated


@pytest.mark.asyncio
async def test_slr_collect_papers_strategy_combined():
    state = minimal_slr_state(
        openalex_results=[{"title": "A"}],
        asta_results=[{"title": "B"}],
    )
    result = await slr_collect_papers(state, {})
    assert result["search_strategy"] == "combined"


@pytest.mark.asyncio
async def test_slr_collect_papers_strategy_openalex_only():
    state = minimal_slr_state(
        openalex_results=[{"title": "A"}],
        asta_results=[],
    )
    result = await slr_collect_papers(state, {})
    assert result["search_strategy"] == "openalex"


@pytest.mark.asyncio
async def test_slr_finalise_success():
    state = minimal_slr_state(
        openalex_results=[{"title": "A"}],
        asta_results=[{"title": "B"}],
        synthesis="Strong evidence found.",
        source_count=2,
        search_strategy="combined",
    )
    result = await slr_finalise(state, {})
    assert result["result"]["success"] is True
    assert result["result"]["thesis"] == "Strong evidence found."
    assert result["success"] is True


@pytest.mark.asyncio
async def test_slr_finalise_with_errors():
    state = minimal_slr_state(
        errors=["fetch_openalex: timeout"],
        synthesis=None,
    )
    result = await slr_finalise(state, {})
    assert result["result"]["success"] is False
    assert result["success"] is False


@pytest.mark.asyncio
async def test_slr_graph_dry_run():
    fake_papers = [
        {"title": "Study A", "abstract": "Found malaria nets effective", "year": 2020},
        {"title": "Study B", "abstract": "Nets reduce transmission", "year": 2021},
    ]
    with (
        patch("src.graph.subgraphs.slr.search_openalex") as mock_oa,
        patch("src.graph.subgraphs.slr.search_asta") as mock_asta,
        patch("src.graph.subgraphs.slr.acall_llm", new=AsyncMock(return_value="Strong evidence found.")),
    ):
        mock_oa.ainvoke = AsyncMock(return_value=fake_papers)
        mock_asta.ainvoke = AsyncMock(return_value=[])
        result_state = await slr_graph.ainvoke(minimal_slr_state())

    assert result_state["result"] is not None
    assert result_state["source_count"] >= 2
    assert result_state["result"]["thesis"] == "Strong evidence found."
