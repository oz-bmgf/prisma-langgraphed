from __future__ import annotations

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from src.graph.subgraphs.lbd import (
    _parse_concepts,
    _parse_papers_from_content,
    _route_lbd,
    lbd_agent,
    lbd_collect_papers,
    lbd_discover_connections,
    lbd_finalise,
    lbd_graph,
)
from src.graph.state import LBDAgentState


def minimal_lbd_state(**overrides) -> LBDAgentState:
    base: LBDAgentState = {
        "task_id": "lbd-001",
        "query": "malaria iron deficiency anemia connection",
        "context": "",
        "messages": [],
        "agent_rounds": 0,
        "seed_concepts": None,
        "merged_papers": None,
        "discovered_concepts": None,
        "narrative": None,
        "paper_count": 0,
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


def test_lbd_graph_compiles():
    assert lbd_graph is not None
    nodes = list(lbd_graph.nodes.keys())
    assert "lbd_agent" in nodes
    assert "lbd_tools" in nodes
    assert "lbd_collect_papers" in nodes
    assert "lbd_discover_connections" in nodes
    assert "lbd_synthesise" in nodes
    assert "lbd_finalise" in nodes


# ---------------------------------------------------------------------------
# _parse_concepts
# ---------------------------------------------------------------------------


def test_parse_concepts_from_list():
    result = _parse_concepts('["malaria", "iron deficiency", "anemia"]')
    assert "malaria" in result
    assert "anemia" in result


def test_parse_concepts_from_csv():
    result = _parse_concepts("malaria, iron deficiency, anemia, immune response")
    assert len(result) >= 3


# ---------------------------------------------------------------------------
# _route_lbd
# ---------------------------------------------------------------------------


def test_route_lbd_no_messages():
    state = minimal_lbd_state()
    assert _route_lbd(state) == "lbd_collect_papers"


def test_route_lbd_ai_message_with_tool_calls():
    ai_msg = AIMessage(content="", tool_calls=[{"name": "search_asta", "args": {}, "id": "1"}])
    state = minimal_lbd_state(messages=[ai_msg], agent_rounds=1)
    assert _route_lbd(state) == "lbd_tools"


def test_route_lbd_ai_message_no_tool_calls():
    ai_msg = AIMessage(content="Search complete.")
    state = minimal_lbd_state(messages=[ai_msg], agent_rounds=1)
    assert _route_lbd(state) == "lbd_collect_papers"


def test_route_lbd_max_rounds_forces_collect():
    ai_msg = AIMessage(content="", tool_calls=[{"name": "search_asta", "args": {}, "id": "1"}])
    state = minimal_lbd_state(messages=[ai_msg], agent_rounds=3)  # at _MAX_ROUNDS
    assert _route_lbd(state) == "lbd_collect_papers"


# ---------------------------------------------------------------------------
# lbd_collect_papers
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_lbd_collect_papers_extracts_papers():
    papers_a = [{"title": "Paper A"}, {"title": "Paper B"}]
    papers_b = [{"title": "Paper C"}]
    msgs = [
        ToolMessage(content=json.dumps(papers_a), tool_call_id="1", name="search_asta"),
        ToolMessage(content=json.dumps(papers_b), tool_call_id="2", name="search_asta"),
    ]
    state = minimal_lbd_state(messages=msgs)
    result = await lbd_collect_papers(state, {})
    assert result["paper_count"] == 3
    assert len(result["merged_papers"]) == 3


@pytest.mark.asyncio
async def test_lbd_collect_papers_deduplicates():
    papers = [
        {"title": "Paper A"},
        {"title": "Paper B"},
        {"title": "Paper A"},  # duplicate
    ]
    msgs = [ToolMessage(content=json.dumps(papers), tool_call_id="1", name="search_asta")]
    state = minimal_lbd_state(messages=msgs)
    result = await lbd_collect_papers(state, {})
    assert result["paper_count"] == 2


@pytest.mark.asyncio
async def test_lbd_collect_papers_extracts_concepts_from_tool_calls():
    ai_msg = AIMessage(
        content="",
        tool_calls=[
            {"name": "search_asta", "args": {"query": "malaria", "top_k": 15}, "id": "1"},
            {"name": "search_asta", "args": {"query": "iron deficiency", "top_k": 15}, "id": "2"},
            # broad search matches main query — should not be included as concept
            {"name": "search_asta", "args": {"query": "malaria iron deficiency anemia connection", "top_k": 15}, "id": "3"},
        ],
    )
    msgs = [
        ai_msg,
        ToolMessage(content=json.dumps([{"title": "A"}]), tool_call_id="1", name="search_asta"),
        ToolMessage(content=json.dumps([{"title": "B"}]), tool_call_id="2", name="search_asta"),
        ToolMessage(content=json.dumps([]), tool_call_id="3", name="search_asta"),
    ]
    state = minimal_lbd_state(messages=msgs)
    result = await lbd_collect_papers(state, {})
    assert "malaria" in result["seed_concepts"]
    assert "iron deficiency" in result["seed_concepts"]
    # broad search query should not appear
    assert "malaria iron deficiency anemia connection" not in result["seed_concepts"]


@pytest.mark.asyncio
async def test_lbd_collect_papers_fallback_concept():
    # No AI messages → seed_concepts falls back to the query itself
    msgs = [ToolMessage(content=json.dumps([{"title": "A"}]), tool_call_id="1", name="search_asta")]
    state = minimal_lbd_state(messages=msgs)
    result = await lbd_collect_papers(state, {})
    assert result["seed_concepts"] == [state["query"]]


# ---------------------------------------------------------------------------
# lbd_finalise
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_lbd_finalise_success():
    state = minimal_lbd_state(
        narrative="Found indirect connections via bridging concepts.",
        merged_papers=[{"title": "A"}, {"title": "B"}],
        seed_concepts=["malaria", "anemia"],
        discovered_concepts=[{"term": "iron", "type": "bridge"}],
    )
    result = await lbd_finalise(state, {})
    assert result["result"]["success"] is True
    assert result["result"]["thesis"] == "Found indirect connections via bridging concepts."
    assert "malaria" in result["result"]["concepts"]


@pytest.mark.asyncio
async def test_lbd_finalise_with_errors():
    state = minimal_lbd_state(errors=["lbd_agent: timeout"])
    result = await lbd_finalise(state, {})
    assert result["result"]["success"] is False
    assert result["result"]["status"] == "error"


# ---------------------------------------------------------------------------
# lbd_agent — basic invocation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_lbd_agent_injects_initial_messages():
    final_response = AIMessage(content="", tool_calls=[
        {"name": "search_asta", "args": {"query": "malaria", "top_k": 15}, "id": "tc1"},
    ])

    mock_llm = MagicMock()
    mock_bound = MagicMock()
    mock_bound.ainvoke = AsyncMock(return_value=final_response)

    with (
        patch("src.graph.subgraphs.lbd._build_llm", return_value=mock_llm),
        patch.object(mock_llm, "bind_tools", return_value=mock_bound),
    ):
        state = minimal_lbd_state()
        result = await lbd_agent(state, {})

    assert "messages" in result
    assert result["agent_rounds"] == 1
    # First call should inject SystemMessage + HumanMessage + AIMessage
    assert len(result["messages"]) == 3


@pytest.mark.asyncio
async def test_lbd_agent_respects_max_rounds():
    state = minimal_lbd_state(agent_rounds=3)  # at _MAX_ROUNDS
    result = await lbd_agent(state, {})
    assert result == {}
