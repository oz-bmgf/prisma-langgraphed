from __future__ import annotations

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from src.graph.agents.slr_graph import (
    _parse_papers_from_content,
    _route_slr,
    slr_agent,
    slr_collect_papers,
    slr_finalise,
    slr_graph,
    slr_synthesise,
)
from src.graph.state import SLRAgentState


def minimal_slr_state(**overrides) -> SLRAgentState:
    base: SLRAgentState = {
        "task_id": "test-001",
        "query": "malaria net efficacy",
        "context": "",
        "top_k": 5,
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


# ---------------------------------------------------------------------------
# Graph compilation
# ---------------------------------------------------------------------------


def test_slr_graph_compiles():
    assert slr_graph is not None
    nodes = list(slr_graph.nodes.keys())
    assert "slr_agent" in nodes
    assert "slr_tools" in nodes
    assert "slr_collect_papers" in nodes
    assert "slr_synthesise" in nodes
    assert "slr_finalise" in nodes


# ---------------------------------------------------------------------------
# _parse_papers_from_content
# ---------------------------------------------------------------------------


def test_parse_papers_from_json():
    papers = [{"title": "A", "year": 2020}, {"title": "B"}]
    assert _parse_papers_from_content(json.dumps(papers)) == papers


def test_parse_papers_from_python_repr():
    papers = [{"title": "A", "year": 2020}]
    assert _parse_papers_from_content(str(papers)) == papers


def test_parse_papers_empty():
    assert _parse_papers_from_content("") == []
    assert _parse_papers_from_content("not a list") == []


# ---------------------------------------------------------------------------
# _route_slr
# ---------------------------------------------------------------------------


def test_route_slr_no_messages():
    state = minimal_slr_state()
    assert _route_slr(state) == "slr_collect_papers"


def test_route_slr_ai_message_with_tool_calls():
    ai_msg = AIMessage(content="", tool_calls=[{"name": "search_openalex", "args": {}, "id": "1"}])
    state = minimal_slr_state(messages=[ai_msg], agent_rounds=1)
    assert _route_slr(state) == "slr_tools"


def test_route_slr_ai_message_no_tool_calls():
    ai_msg = AIMessage(content="Done searching.")
    state = minimal_slr_state(messages=[ai_msg], agent_rounds=1)
    assert _route_slr(state) == "slr_collect_papers"


def test_route_slr_max_rounds_forces_collect():
    ai_msg = AIMessage(content="", tool_calls=[{"name": "search_asta", "args": {}, "id": "1"}])
    state = minimal_slr_state(messages=[ai_msg], agent_rounds=3)  # at _MAX_ROUNDS
    assert _route_slr(state) == "slr_collect_papers"


# ---------------------------------------------------------------------------
# slr_collect_papers
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_slr_collect_papers_extracts_from_tool_messages():
    oa_papers = [{"title": "Paper A", "year": 2020}]
    asta_papers = [{"title": "Paper B", "year": 2021}]
    msgs = [
        ToolMessage(content=json.dumps(oa_papers), tool_call_id="1", name="search_openalex"),
        ToolMessage(content=json.dumps(asta_papers), tool_call_id="2", name="search_asta"),
    ]
    state = minimal_slr_state(messages=msgs)
    result = await slr_collect_papers(state, {})
    assert result["source_count"] == 2
    assert result["search_strategy"] == "combined"
    titles = {p["title"] for p in result["merged_papers"]}
    assert "Paper A" in titles
    assert "Paper B" in titles


@pytest.mark.asyncio
async def test_slr_collect_papers_deduplicates():
    papers_oa = [{"title": "Shared Paper"}, {"title": "Only OA"}]
    papers_asta = [{"title": "Shared Paper"}, {"title": "Only ASTA"}]
    msgs = [
        ToolMessage(content=json.dumps(papers_oa), tool_call_id="1", name="search_openalex"),
        ToolMessage(content=json.dumps(papers_asta), tool_call_id="2", name="search_asta"),
    ]
    state = minimal_slr_state(messages=msgs)
    result = await slr_collect_papers(state, {})
    assert result["source_count"] == 3  # "Shared Paper" deduped


@pytest.mark.asyncio
async def test_slr_collect_papers_openalex_only():
    papers = [{"title": "A"}]
    msgs = [ToolMessage(content=json.dumps(papers), tool_call_id="1", name="search_openalex")]
    state = minimal_slr_state(messages=msgs)
    result = await slr_collect_papers(state, {})
    assert result["search_strategy"] == "openalex"


@pytest.mark.asyncio
async def test_slr_collect_papers_no_messages():
    state = minimal_slr_state()
    result = await slr_collect_papers(state, {})
    assert result["source_count"] == 0
    assert result["search_strategy"] == "none"


# ---------------------------------------------------------------------------
# slr_finalise
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_slr_finalise_success():
    state = minimal_slr_state(
        merged_papers=[{"title": "A"}, {"title": "B"}],
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


# ---------------------------------------------------------------------------
# slr_graph dry run (mocked LLM + tools)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_slr_graph_dry_run():
    fake_papers_oa = [
        {"title": "Study A", "abstract": "Found malaria nets effective", "year": 2020},
    ]
    fake_papers_asta = [
        {"title": "Study B", "abstract": "Nets reduce transmission", "year": 2021},
    ]

    # Mock the LLM to return an AIMessage with tool calls on first call,
    # then a plain AIMessage on second call (no more tool calls).
    tool_call_response = AIMessage(
        content="",
        tool_calls=[
            {"name": "search_openalex", "args": {"query": "malaria net efficacy", "top_k": 20}, "id": "tc1"},
            {"name": "search_asta", "args": {"query": "malaria net efficacy", "top_k": 20}, "id": "tc2"},
        ],
    )
    final_response = AIMessage(content="Search complete.")

    call_count = 0

    async def mock_llm_invoke(messages, config=None, **kwargs):
        nonlocal call_count
        call_count += 1
        return tool_call_response if call_count == 1 else final_response

    mock_llm = MagicMock()
    mock_llm.ainvoke = mock_llm_invoke

    mock_bound = MagicMock()
    mock_bound.ainvoke = mock_llm_invoke

    with (
        patch("src.graph.subgraphs.slr._build_llm", return_value=mock_llm),
        patch.object(mock_llm, "bind_tools", return_value=mock_bound),
        patch("src.graph.subgraphs.slr.search_openalex") as mock_oa,
        patch("src.graph.subgraphs.slr.search_asta") as mock_asta,
        patch("src.graph.subgraphs.slr.acall_llm", new=AsyncMock(return_value="Strong evidence found.")),
    ):
        # ToolNode calls .invoke on the tool; patch the module-level references
        mock_oa.ainvoke = AsyncMock(return_value=fake_papers_oa)
        mock_asta.ainvoke = AsyncMock(return_value=fake_papers_asta)

        # Run via slr_agent directly (avoid full graph integration complexity in unit test)
        state = minimal_slr_state()
        agent_result = await slr_agent(state, {})
        assert "messages" in agent_result
        assert agent_result["agent_rounds"] == 1
