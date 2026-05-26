"""Tests for src/graph/subgraphs/lbd.py — canonical LBD subgraph location."""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from src.graph.subgraphs.lbd import (
    _parse_concepts,
    _parse_papers_from_content,
    _route_lbd,
    lbd_agent,
    lbd_collect_papers,
    lbd_discover_connections,
    lbd_finalise,
    lbd_graph,
    lbd_synthesise,
)
from src.graph.state import LBDAgentState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _state(**overrides) -> dict:
    base: dict = {
        "task_id": "L1",
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


_PAPERS = [
    {"title": "Malaria iron link", "abstract": "Iron deficiency reduces immunity.", "year": 2022},
    {"title": "Anemia tropics", "abstract": "Anemia prevalent in tropics.", "year": 2021},
]


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


def test_route_lbd_empty_messages():
    assert _route_lbd(_state()) == "lbd_collect_papers"


def test_route_lbd_has_tool_calls():
    ai_msg = AIMessage(content="", tool_calls=[
        {"name": "search_asta", "args": {"query": "malaria", "top_k": 15}, "id": "1"},
    ])
    assert _route_lbd(_state(messages=[ai_msg], agent_rounds=1)) == "lbd_tools"


def test_route_lbd_no_tool_calls():
    ai_msg = AIMessage(content="Searches complete.")
    assert _route_lbd(_state(messages=[ai_msg], agent_rounds=1)) == "lbd_collect_papers"


def test_route_lbd_max_rounds():
    ai_msg = AIMessage(content="", tool_calls=[{"name": "search_asta", "args": {}, "id": "1"}])
    assert _route_lbd(_state(messages=[ai_msg], agent_rounds=3)) == "lbd_collect_papers"


# ---------------------------------------------------------------------------
# lbd_collect_papers — extracts papers + concepts from messages
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_lbd_collect_papers_extracts_papers():
    msgs = [
        ToolMessage(content=json.dumps(_PAPERS[:1]), tool_call_id="1", name="search_asta"),
        ToolMessage(content=json.dumps(_PAPERS[1:]), tool_call_id="2", name="search_asta"),
    ]
    result = await lbd_collect_papers(_state(messages=msgs), {})
    assert result["paper_count"] == 2


@pytest.mark.asyncio
async def test_lbd_collect_papers_deduplicates():
    paper = {"title": "Shared"}
    msgs = [
        ToolMessage(content=json.dumps([paper, {"title": "A"}]), tool_call_id="1", name="search_asta"),
        ToolMessage(content=json.dumps([paper, {"title": "B"}]), tool_call_id="2", name="search_asta"),
    ]
    result = await lbd_collect_papers(_state(messages=msgs), {})
    assert result["paper_count"] == 3  # "Shared" deduped


@pytest.mark.asyncio
async def test_lbd_collect_papers_extracts_concepts():
    ai_msg = AIMessage(
        content="",
        tool_calls=[
            {"name": "search_asta", "args": {"query": "malaria", "top_k": 15}, "id": "1"},
            {"name": "search_asta", "args": {"query": "iron deficiency", "top_k": 15}, "id": "2"},
            # broad search = main query; should not appear as concept
            {"name": "search_asta", "args": {"query": "malaria iron deficiency anemia connection", "top_k": 15}, "id": "3"},
        ],
    )
    msgs = [
        ai_msg,
        ToolMessage(content=json.dumps([{"title": "A"}]), tool_call_id="1", name="search_asta"),
        ToolMessage(content=json.dumps([{"title": "B"}]), tool_call_id="2", name="search_asta"),
        ToolMessage(content=json.dumps([]), tool_call_id="3", name="search_asta"),
    ]
    result = await lbd_collect_papers(_state(messages=msgs), {})
    assert "malaria" in result["seed_concepts"]
    assert "iron deficiency" in result["seed_concepts"]
    assert "malaria iron deficiency anemia connection" not in result["seed_concepts"]


@pytest.mark.asyncio
async def test_lbd_collect_papers_fallback_concept():
    msgs = [ToolMessage(content=json.dumps([{"title": "A"}]), tool_call_id="1", name="search_asta")]
    result = await lbd_collect_papers(_state(messages=msgs), {})
    assert result["seed_concepts"] == ["malaria iron deficiency anemia connection"]


# ---------------------------------------------------------------------------
# lbd_discover_connections — reads merged_papers, extracts B-terms
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_lbd_discover_connections_empty_merged_papers():
    result = await lbd_discover_connections(_state(merged_papers=[]), {})
    assert result["discovered_concepts"] == []


@pytest.mark.asyncio
async def test_lbd_discover_connections_extracts_b_terms():
    state = _state(merged_papers=_PAPERS)
    with patch("src.graph.subgraphs.lbd.acall_llm", new=AsyncMock(return_value='["bridge_concept"]')):
        result = await lbd_discover_connections(state, {})
    assert result["discovered_concepts"] == [{"term": "bridge_concept", "type": "bridge"}]
    # merged_papers is NOT returned since it's already in state
    assert "merged_papers" not in result


@pytest.mark.asyncio
async def test_lbd_discover_connections_skips_when_already_done():
    state = _state(discovered_concepts=[{"term": "x", "type": "bridge"}])
    result = await lbd_discover_connections(state, {})
    assert result == {}


@pytest.mark.asyncio
async def test_lbd_discover_connections_surfaces_llm_error():
    state = _state(merged_papers=_PAPERS)
    with patch("src.graph.subgraphs.lbd.acall_llm", new=AsyncMock(side_effect=RuntimeError("LLM down"))):
        result = await lbd_discover_connections(state, {})
    assert result["discovered_concepts"] == []
    assert any("lbd_discover_connections" in e for e in result.get("errors", []))


@pytest.mark.asyncio
async def test_lbd_discover_connections_passes_config_to_acall_llm():
    captured: list[dict] = []

    async def _mock_acall(*args, config=None, **kwargs):
        captured.append({"config": config})
        return '["bridge_concept"]'

    state = _state(merged_papers=_PAPERS)
    with patch("src.graph.subgraphs.lbd.acall_llm", side_effect=_mock_acall):
        await lbd_discover_connections(state, {"configurable": {"research_model": "test-model"}})

    assert len(captured) == 1
    assert captured[0]["config"] is not None


# ---------------------------------------------------------------------------
# lbd_synthesise
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_lbd_synthesise_skips_when_already_done():
    result = await lbd_synthesise(_state(narrative="Existing narrative."), {})
    assert result == {}


@pytest.mark.asyncio
async def test_lbd_synthesise_llm_failure_surfaces_error():
    state = _state(merged_papers=_PAPERS)
    with patch("src.graph.subgraphs.lbd.acall_llm", new=AsyncMock(side_effect=RuntimeError("LLM down"))):
        result = await lbd_synthesise(state, {})
    assert result["narrative"] is None
    assert any("lbd_synthesise" in e for e in result.get("errors", []))


# ---------------------------------------------------------------------------
# lbd_finalise
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_lbd_finalise_success():
    state = _state(
        merged_papers=_PAPERS,
        narrative="Found indirect connections.",
        seed_concepts=["malaria", "anemia"],
        discovered_concepts=[{"term": "iron", "type": "bridge"}],
    )
    result = await lbd_finalise(state, {})
    assert result["result"]["success"] is True
    assert result["result"]["thesis"] == "Found indirect connections."
    assert "malaria" in result["result"]["concepts"]
    assert result["result"]["status"] == "ok"


@pytest.mark.asyncio
async def test_lbd_finalise_adds_status_field():
    state = _state(merged_papers=_PAPERS, narrative="connections found", errors=[])
    result = await lbd_finalise(state, {})
    assert result["result"]["status"] == "ok"

    state_err = _state(errors=["lbd_agent: timeout"])
    result_err = await lbd_finalise(state_err, {})
    assert result_err["result"]["status"] == "error"


@pytest.mark.asyncio
async def test_lbd_finalise_reads_merged_papers():
    state = _state(
        merged_papers=_PAPERS,
        narrative="connections found",
    )
    result = await lbd_finalise(state, {})
    assert result["result"]["papers"] == _PAPERS
    assert result["result"]["paper_count"] == 2


# ---------------------------------------------------------------------------
# lbd_agent — round limit guard
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_lbd_agent_respects_max_rounds():
    state = _state(agent_rounds=3)
    result = await lbd_agent(state, {})
    assert result == {}


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
        result = await lbd_agent(_state(), {})

    assert "messages" in result
    assert result["agent_rounds"] == 1
    # First call: SystemMessage + HumanMessage + AIMessage
    assert len(result["messages"]) == 3


# ---------------------------------------------------------------------------
# Graph topology
# ---------------------------------------------------------------------------


def test_lbd_graph_has_all_nodes():
    graph_nodes = set(lbd_graph.get_graph().nodes.keys()) - {"__start__", "__end__"}
    expected = {
        "lbd_agent",
        "lbd_tools",
        "lbd_collect_papers",
        "lbd_discover_connections",
        "lbd_synthesise",
        "lbd_finalise",
    }
    assert graph_nodes == expected


def test_lbd_graph_compiles():
    assert lbd_graph is not None
