"""Tests for src/graph/subgraphs/slr.py — canonical SLR subgraph location."""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from src.graph.subgraphs.slr import (
    _parse_papers_from_content,
    _route_slr,
    slr_agent,
    slr_collect_papers,
    slr_finalise,
    slr_graph,
    slr_synthesise,
)
from src.graph.state import SLRAgentState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _state(**overrides) -> dict:
    base: dict = {
        "task_id": "T1",
        "query": "malaria net efficacy",
        "context": "",
        "top_k": 10,
        "messages": [],
        "agent_rounds": 0,
        "merged_papers": None,
        "synthesis": None,
        "source_count": 0,
        "search_strategy": "none",
        "tool_traces": [],
        "result": None,
        "success": False,
        "error_message": None,
        "errors": [],
    }
    base.update(overrides)
    return base


_PAPERS = [
    {"title": "Net efficacy RCT", "abstract": "85% reduction.", "year": 2022, "authors": "Smith J"},
    {"title": "Resistance patterns", "abstract": "IR spreading.", "year": 2021, "authors": "Jones K"},
]


# ---------------------------------------------------------------------------
# _parse_papers_from_content
# ---------------------------------------------------------------------------


def test_parse_papers_json():
    assert _parse_papers_from_content(json.dumps(_PAPERS)) == _PAPERS


def test_parse_papers_python_repr():
    assert _parse_papers_from_content(str(_PAPERS)) == _PAPERS


def test_parse_papers_invalid():
    assert _parse_papers_from_content("not a list") == []
    assert _parse_papers_from_content("") == []


# ---------------------------------------------------------------------------
# _route_slr
# ---------------------------------------------------------------------------


def test_route_slr_empty_messages():
    assert _route_slr(_state()) == "slr_collect_papers"


def test_route_slr_has_tool_calls():
    ai_msg = AIMessage(content="", tool_calls=[
        {"name": "search_openalex", "args": {"query": "malaria", "top_k": 20}, "id": "1"},
    ])
    assert _route_slr(_state(messages=[ai_msg], agent_rounds=1)) == "slr_tools"


def test_route_slr_no_tool_calls():
    ai_msg = AIMessage(content="Done.")
    assert _route_slr(_state(messages=[ai_msg], agent_rounds=1)) == "slr_collect_papers"


def test_route_slr_max_rounds():
    ai_msg = AIMessage(content="", tool_calls=[{"name": "search_asta", "args": {}, "id": "1"}])
    assert _route_slr(_state(messages=[ai_msg], agent_rounds=3)) == "slr_collect_papers"


# ---------------------------------------------------------------------------
# slr_collect_papers — extracts papers from ToolMessages
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_slr_collect_papers_combined():
    msgs = [
        ToolMessage(content=json.dumps([_PAPERS[0]]), tool_call_id="1", name="search_openalex"),
        ToolMessage(content=json.dumps([_PAPERS[1]]), tool_call_id="2", name="search_asta"),
    ]
    result = await slr_collect_papers(_state(messages=msgs), {})
    assert result["source_count"] == 2
    assert result["search_strategy"] == "combined"


@pytest.mark.asyncio
async def test_slr_collect_papers_deduplicates():
    paper = {"title": "Shared Paper"}
    msgs = [
        ToolMessage(content=json.dumps([paper, {"title": "Only OA"}]), tool_call_id="1", name="search_openalex"),
        ToolMessage(content=json.dumps([paper, {"title": "Only ASTA"}]), tool_call_id="2", name="search_asta"),
    ]
    result = await slr_collect_papers(_state(messages=msgs), {})
    assert result["source_count"] == 3  # "Shared Paper" deduped


@pytest.mark.asyncio
async def test_slr_collect_papers_openalex_only():
    msgs = [ToolMessage(content=json.dumps([{"title": "A"}]), tool_call_id="1", name="search_openalex")]
    result = await slr_collect_papers(_state(messages=msgs), {})
    assert result["search_strategy"] == "openalex"


@pytest.mark.asyncio
async def test_slr_collect_papers_empty_messages():
    result = await slr_collect_papers(_state(), {})
    assert result["merged_papers"] == []
    assert result["source_count"] == 0
    assert result["search_strategy"] == "none"


@pytest.mark.asyncio
async def test_slr_collect_papers_python_repr_content():
    """ToolNode may serialize list[dict] as str(list) — ast.literal_eval fallback."""
    msgs = [ToolMessage(content=str(_PAPERS), tool_call_id="1", name="search_asta")]
    result = await slr_collect_papers(_state(messages=msgs), {})
    assert result["source_count"] == 2


# ---------------------------------------------------------------------------
# slr_synthesise
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_slr_synthesise_returns_no_papers_message_when_merged_papers_empty():
    result = await slr_synthesise(_state(merged_papers=[]), {})
    assert "No papers found" in result["synthesis"]


@pytest.mark.asyncio
async def test_slr_synthesise_skips_when_already_done():
    result = await slr_synthesise(_state(synthesis="Existing synthesis."), {})
    assert result == {}


@pytest.mark.asyncio
async def test_slr_synthesise_llm_failure_surfaces_error():
    state = _state(merged_papers=_PAPERS)
    with patch("src.graph.subgraphs.slr.acall_llm", new=AsyncMock(side_effect=RuntimeError("LLM down"))):
        result = await slr_synthesise(state, {})
    assert result["synthesis"] is None
    assert any("slr_synthesise" in e for e in result.get("errors", []))


@pytest.mark.asyncio
async def test_slr_synthesise_passes_config_to_acall_llm():
    """acall_llm must receive config= so LLM calls appear as child spans."""
    captured: list[dict] = []

    async def _mock_acall(*args, config=None, **kwargs):
        captured.append({"config": config})
        return "synthesis text"

    state = _state(merged_papers=_PAPERS)
    with patch("src.graph.subgraphs.slr.acall_llm", side_effect=_mock_acall):
        await slr_synthesise(state, {"configurable": {"research_model": "test-model"}})

    assert len(captured) == 1
    assert captured[0]["config"] is not None


# ---------------------------------------------------------------------------
# slr_finalise
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_slr_finalise_success():
    state = _state(
        merged_papers=_PAPERS,
        synthesis="Strong evidence.",
        source_count=2,
        search_strategy="combined",
    )
    result = await slr_finalise(state, {})
    assert result["result"]["success"] is True
    assert result["result"]["thesis"] == "Strong evidence."


@pytest.mark.asyncio
async def test_slr_finalise_none_when_llm_errors():
    state = _state(errors=["slr_synthesise: LLM failed"], synthesis=None, merged_papers=[])
    result = await slr_finalise(state, {})
    assert result["result"]["success"] is False
    assert result["result"]["thesis"] == ""


# ---------------------------------------------------------------------------
# slr_agent — round limit guard
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_slr_agent_respects_max_rounds():
    state = _state(agent_rounds=3)  # at _MAX_ROUNDS
    result = await slr_agent(state, {})
    assert result == {}


@pytest.mark.asyncio
async def test_slr_agent_injects_initial_messages():
    final_response = AIMessage(content="", tool_calls=[
        {"name": "search_openalex", "args": {"query": "malaria net efficacy", "top_k": 20}, "id": "tc1"},
    ])
    mock_llm = MagicMock()
    mock_bound = MagicMock()
    mock_bound.ainvoke = AsyncMock(return_value=final_response)

    with (
        patch("src.graph.subgraphs.slr._build_llm", return_value=mock_llm),
        patch.object(mock_llm, "bind_tools", return_value=mock_bound),
    ):
        result = await slr_agent(_state(), {})

    assert "messages" in result
    assert result["agent_rounds"] == 1
    # First call: SystemMessage + HumanMessage + AIMessage
    assert len(result["messages"]) == 3


# ---------------------------------------------------------------------------
# Graph topology
# ---------------------------------------------------------------------------


def test_slr_graph_has_all_nodes():
    graph_nodes = set(slr_graph.get_graph().nodes.keys()) - {"__start__", "__end__"}
    expected = {
        "slr_agent",
        "slr_tools",
        "slr_collect_papers",
        "slr_synthesise",
        "slr_finalise",
    }
    assert graph_nodes == expected


def test_slr_graph_compiles():
    assert slr_graph is not None
