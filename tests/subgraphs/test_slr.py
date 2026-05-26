"""Tests for src/graph/subgraphs/slr.py — canonical SLR subgraph location."""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from langgraph.types import Send

from src.graph.subgraphs.slr import (
    slr_collect_papers,
    slr_expand_queries,
    slr_fetch_source,
    slr_finalise,
    slr_graph,
    slr_plan_sources,
    slr_synthesise,
)
from src.graph.state import SLRAgentState, SLRFetchState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _state(**overrides) -> dict:
    base: dict = {
        "task_id": "T1",
        "query": "malaria net efficacy",
        "context": "",
        "top_k": 10,
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


_PAPERS = [
    {"title": "Net efficacy RCT", "abstract": "85% reduction.", "year": 2022, "authors": "Smith J"},
    {"title": "Resistance patterns", "abstract": "IR spreading.", "year": 2021, "authors": "Jones K"},
]


# ---------------------------------------------------------------------------
# test_slr_dispatch_skips_completed
# Task spec: prime state with one existing result, assert dispatch emits N-1 sends
# ---------------------------------------------------------------------------


async def test_slr_dispatch_skips_completed_openalex():
    """slr_plan_sources skips openalex when openalex_results is already populated."""
    state = _state(openalex_results=_PAPERS)  # openalex already done
    result = await slr_plan_sources(state)
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0].arg["source"] == "asta"


async def test_slr_dispatch_skips_completed_asta():
    """slr_plan_sources skips asta when asta_results is already populated."""
    state = _state(asta_results=_PAPERS)  # asta already done
    result = await slr_plan_sources(state)
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0].arg["source"] == "openalex"


async def test_slr_dispatch_skips_all_when_both_done():
    """slr_plan_sources returns passthrough string when both sources already fetched."""
    state = _state(openalex_results=_PAPERS, asta_results=_PAPERS)
    result = await slr_plan_sources(state)
    assert result == "slr_collect_papers"


async def test_slr_dispatch_sends_both_on_fresh_state():
    """slr_plan_sources emits 2 Sends when neither source has results."""
    state = _state()
    result = await slr_plan_sources(state)
    assert isinstance(result, list)
    assert len(result) == 2
    sources = {s.arg["source"] for s in result}
    assert sources == {"openalex", "asta"}
    assert all(s.node == "slr_fetch_source" for s in result)


# ---------------------------------------------------------------------------
# test_slr_worker_pure
# Task spec: call worker with fixed payload, assert dict return + no side effects
# ---------------------------------------------------------------------------


async def test_slr_worker_pure_openalex():
    """slr_fetch_source returns a plain dict with openalex_results and no file I/O."""
    fake_papers = [{"title": "Paper A", "abstract": "abc", "year": 2020}]
    worker_state: SLRFetchState = {
        "source": "openalex",
        "query": "malaria",
        "top_k": 5,
        "result": None,
    }
    with patch("src.graph.subgraphs.slr.search_openalex") as mock_tool:
        mock_tool.ainvoke = AsyncMock(return_value=fake_papers)
        result = await slr_fetch_source(worker_state, {})

    assert isinstance(result, dict)
    assert "openalex_results" in result
    assert result["openalex_results"] == fake_papers
    assert result["tool_traces"][0]["success"] is True


async def test_slr_worker_pure_asta():
    """slr_fetch_source returns a plain dict with asta_results."""
    fake_papers = [{"title": "Paper B", "abstract": "xyz", "year": 2021}]
    worker_state: SLRFetchState = {
        "source": "asta",
        "query": "malaria",
        "top_k": 5,
        "result": None,
    }
    with patch("src.graph.subgraphs.slr.search_asta") as mock_tool:
        mock_tool.ainvoke = AsyncMock(return_value=fake_papers)
        result = await slr_fetch_source(worker_state, {})

    assert isinstance(result, dict)
    assert "asta_results" in result
    assert result["asta_results"] == fake_papers


async def test_slr_worker_error_surfaces_to_errors_field():
    """slr_fetch_source network failure goes into errors[], not raises."""
    worker_state: SLRFetchState = {
        "source": "openalex",
        "query": "malaria",
        "top_k": 5,
        "result": None,
    }
    with patch("src.graph.subgraphs.slr.search_openalex") as mock_tool:
        mock_tool.ainvoke = AsyncMock(side_effect=RuntimeError("network timeout"))
        result = await slr_fetch_source(worker_state, {})

    assert "errors" in result
    assert any("openalex" in e for e in result["errors"])
    assert result["tool_traces"][0]["success"] is False


async def test_slr_worker_skips_when_result_pre_populated():
    """slr_fetch_source returns early when result is already set (idempotency guard)."""
    prepopulated = [{"title": "Cached paper"}]
    worker_state: SLRFetchState = {
        "source": "openalex",
        "query": "malaria",
        "top_k": 5,
        "result": prepopulated,
    }
    result = await slr_fetch_source(worker_state, {})
    assert result == {"openalex_results": prepopulated}


# ---------------------------------------------------------------------------
# test_slr_graph_has_all_nodes
# Task spec: compiled graph nodes match topology comment
# ---------------------------------------------------------------------------


def test_slr_graph_has_all_nodes():
    """Compiled graph nodes match the topology declared in the module docstring."""
    graph_nodes = set(slr_graph.get_graph().nodes.keys()) - {"__start__", "__end__"}
    # slr_plan_sources is a conditional edge function, not a node
    expected = {
        "slr_start",
        "slr_expand_queries",
        "slr_fetch_source",
        "slr_collect_papers",
        "slr_synthesise",
        "slr_finalise",
    }
    assert graph_nodes == expected


def test_slr_graph_compiles():
    assert slr_graph is not None


# ---------------------------------------------------------------------------
# test_slr_handles_empty_results
# Task spec: synthesise returns None gracefully when all workers returned errors
# ---------------------------------------------------------------------------


async def test_slr_synthesise_returns_no_papers_message_when_merged_papers_empty():
    """slr_synthesise sets a graceful message when merged_papers is empty."""
    state = _state(merged_papers=[])
    result = await slr_synthesise(state, {})
    assert "synthesis" in result
    assert "No papers found" in result["synthesis"]


async def test_slr_synthesise_skips_when_already_done():
    """slr_synthesise returns {} when synthesis is already set (idempotency)."""
    state = _state(synthesis="Existing synthesis.")
    result = await slr_synthesise(state, {})
    assert result == {}


async def test_slr_finalise_none_when_llm_errors():
    """slr_finalise sets success=False and empty thesis when synthesis is None."""
    state = _state(errors=["slr_synthesise: LLM failed"], synthesis=None, merged_papers=[])
    result = await slr_finalise(state, {})
    assert result["result"]["success"] is False
    assert result["result"]["thesis"] == ""


async def test_slr_synthesise_llm_failure_surfaces_error():
    """slr_synthesise LLM exception surfaces to errors field, synthesis is None."""
    state = _state(merged_papers=_PAPERS)
    with patch("src.graph.subgraphs.slr.acall_llm", new=AsyncMock(side_effect=RuntimeError("LLM down"))):
        result = await slr_synthesise(state, {})
    assert result["synthesis"] is None
    assert any("slr_synthesise" in e for e in result.get("errors", []))


# ---------------------------------------------------------------------------
# slr_collect_papers — deduplication and merged_papers
# ---------------------------------------------------------------------------


async def test_slr_collect_papers_writes_merged_papers():
    """slr_collect_papers produces merged_papers with deduplication applied once."""
    paper_a = {"title": "Paper A", "abstract": "abc"}
    paper_b = {"title": "Paper B", "abstract": "def"}
    paper_b_dup = {"title": "Paper B", "abstract": "def (dup)"}
    state = _state(openalex_results=[paper_a, paper_b], asta_results=[paper_b_dup])

    result = await slr_collect_papers(state, {})

    assert "merged_papers" in result
    titles = [p["title"] for p in result["merged_papers"]]
    assert titles.count("Paper B") == 1
    assert result["source_count"] == 2
    assert result["search_strategy"] == "combined"


async def test_slr_collect_papers_strategy_openalex_only():
    state = _state(openalex_results=[{"title": "A"}], asta_results=[])
    result = await slr_collect_papers(state, {})
    assert result["search_strategy"] == "openalex"


async def test_slr_collect_papers_empty_sources():
    state = _state()
    result = await slr_collect_papers(state, {})
    assert result["merged_papers"] == []
    assert result["source_count"] == 0
    assert result["search_strategy"] == "none"


# ---------------------------------------------------------------------------
# slr_expand_queries — config threading + error surfacing
# ---------------------------------------------------------------------------


async def test_slr_expand_queries_passes_config_to_acall_llm():
    """acall_llm must receive config= so LLM calls appear as child spans."""
    state = _state()
    captured: list[dict] = []

    async def _mock_acall(*args, config=None, **kwargs):
        captured.append({"config": config})
        return '{"queries": ["alternative 1"]}'

    with patch("src.graph.subgraphs.slr.acall_llm", side_effect=_mock_acall):
        await slr_expand_queries(state, {"configurable": {"research_model": "test-model"}})

    assert len(captured) == 1
    assert captured[0]["config"] is not None


async def test_slr_expand_queries_surfaces_errors():
    """slr_expand_queries exception goes to errors field, not swallowed."""
    state = _state()
    with patch("src.graph.subgraphs.slr.acall_llm", new=AsyncMock(side_effect=RuntimeError("LLM error"))):
        result = await slr_expand_queries(state, {})
    assert result.get("expanded_queries") == []
    assert any("slr_expand_queries" in e for e in result.get("errors", []))


async def test_slr_expand_queries_skips_when_already_done():
    state = _state(expanded_queries=["alt 1"])
    result = await slr_expand_queries(state, {})
    assert result == {}


# ---------------------------------------------------------------------------
# Full dry-run through compiled graph
# ---------------------------------------------------------------------------


async def test_slr_graph_dry_run():
    """End-to-end dry run through compiled slr_graph with mocked I/O."""
    fake_papers = [
        {"title": "Study A", "abstract": "Found nets effective", "year": 2020},
        {"title": "Study B", "abstract": "Nets reduce transmission", "year": 2021},
    ]
    with (
        patch("src.graph.subgraphs.slr.search_openalex") as mock_oa,
        patch("src.graph.subgraphs.slr.search_asta") as mock_asta,
        patch("src.graph.subgraphs.slr.acall_llm", new=AsyncMock(return_value="Strong evidence.")),
    ):
        mock_oa.ainvoke = AsyncMock(return_value=fake_papers)
        mock_asta.ainvoke = AsyncMock(return_value=[])
        final = await slr_graph.ainvoke(_state())

    assert final["result"] is not None
    assert final["result"]["thesis"] == "Strong evidence."
    assert final["merged_papers"] == fake_papers
    assert final["source_count"] == 2
    assert final["search_strategy"] == "openalex"
