from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.graph.agents.deep_web_graph import (
    deep_web_graph,
    deep_web_route_after_primary,  # combined routing+dispatch conditional edge
    deep_web_try_primary,
    deep_web_dispatch_rounds,  # exposed for testing, not used in graph directly
    deep_web_search_round,
    deep_web_finalise,
)
from src.graph.state import DeepWebAgentState, DeepWebSearchRoundState


def minimal_dw_state(**overrides) -> DeepWebAgentState:
    base: DeepWebAgentState = {
        "task_id": "dw-001",
        "question": "What is the efficacy of insecticide-treated nets against malaria?",
        "context": "",
        "model_used": None,
        "primary_result": None,
        "search_round_results": [],
        "fallback_synthesis": None,
        "tool_traces": [],
        "result": None,
        "success": False,
        "error_message": None,
        "errors": [],
    }
    base.update(overrides)
    return base


def test_deep_web_graph_compiles():
    assert deep_web_graph is not None
    nodes = list(deep_web_graph.nodes.keys())
    assert "deep_web_try_primary" in nodes
    # deep_web_dispatch_rounds is NOT a node — it's a conditional edge router combined with route_after_primary
    assert "deep_web_search_round" in nodes
    assert "deep_web_finalise" in nodes


@pytest.mark.asyncio
async def test_route_after_primary_success_returns_str():
    state = minimal_dw_state(primary_result={"success": True, "answer": "nets are effective"})
    result = await deep_web_route_after_primary(state)
    assert result == "deep_web_finalise"


@pytest.mark.asyncio
async def test_route_after_primary_failure_returns_sends():
    from langgraph.types import Send
    state = minimal_dw_state(primary_result={"success": False, "error_message": "API unavailable"})
    result = await deep_web_route_after_primary(state)
    assert isinstance(result, list)
    assert all(isinstance(s, Send) for s in result)


@pytest.mark.asyncio
async def test_route_after_primary_none_returns_sends():
    from langgraph.types import Send
    state = minimal_dw_state(primary_result=None)
    result = await deep_web_route_after_primary(state)
    assert isinstance(result, list)
    assert all(isinstance(s, Send) for s in result)


@pytest.mark.asyncio
async def test_deep_web_try_primary_success():
    from src.core.agents.deep_web import DeepWebResult
    mock_result = DeepWebResult(
        question="test",
        answer="nets reduce malaria by 50%",
        sources=["https://example.com"],
        model_used="o3-deep-research",
        search_rounds=1,
        success=True,
    )
    with patch("src.core.agents.deep_web._primary_research", new=AsyncMock(return_value=mock_result)):
        state = minimal_dw_state()
        result = await deep_web_try_primary(state, {})
    assert result["primary_result"]["success"] is True
    assert result["primary_result"]["answer"] == "nets reduce malaria by 50%"
    assert result["tool_traces"][0]["success"] is True


@pytest.mark.asyncio
async def test_deep_web_try_primary_failure():
    with patch("src.core.agents.deep_web._primary_research", new=AsyncMock(side_effect=Exception("API unavailable"))):
        state = minimal_dw_state()
        result = await deep_web_try_primary(state, {})
    assert result["primary_result"]["success"] is False
    assert "errors" in result


def test_deep_web_dispatch_rounds_returns_sends():
    from langgraph.types import Send
    from src.config import DEEP_WEB_MAX_ROUNDS
    state = minimal_dw_state()
    result = deep_web_dispatch_rounds(state)
    assert isinstance(result, list)
    assert len(result) == DEEP_WEB_MAX_ROUNDS
    for s in result:
        assert isinstance(s, Send)
        assert s.node == "deep_web_search_round"


@pytest.mark.asyncio
async def test_deep_web_search_round_success():
    with patch("src.graph.subgraphs.deep_web.acall_llm", new=AsyncMock(return_value="Round 1 findings.")):
        state: DeepWebSearchRoundState = {
            "round_number": 1,
            "question": "malaria nets efficacy",
            "prior_context": "",
            "result": None,
        }
        result = await deep_web_search_round(state, {})
    assert "search_round_results" in result
    assert result["search_round_results"][0]["success"] is True
    assert result["search_round_results"][0]["answer"] == "Round 1 findings."


@pytest.mark.asyncio
async def test_deep_web_search_round_error():
    with patch("src.graph.subgraphs.deep_web.acall_llm", new=AsyncMock(side_effect=Exception("LLM error"))):
        state: DeepWebSearchRoundState = {
            "round_number": 2,
            "question": "test",
            "prior_context": "",
            "result": None,
        }
        result = await deep_web_search_round(state, {})
    assert result["search_round_results"][0]["success"] is False
    assert "errors" in result


@pytest.mark.asyncio
async def test_deep_web_finalise_uses_primary_when_available():
    state = minimal_dw_state(
        primary_result={"success": True, "answer": "Primary answer", "sources": ["url1"], "model_used": "o3"},
    )
    result = await deep_web_finalise(state, {})
    assert result["result"]["result"] == "Primary answer"
    assert result["result"]["model_used"] == "o3"


@pytest.mark.asyncio
async def test_deep_web_finalise_uses_fallback():
    state = minimal_dw_state(
        primary_result={"success": False},
        fallback_synthesis="Fallback synthesis text.",
    )
    result = await deep_web_finalise(state, {})
    assert result["result"]["result"] == "Fallback synthesis text."


@pytest.mark.asyncio
async def test_deep_web_graph_primary_path():
    from src.core.agents.deep_web import DeepWebResult
    mock_result = DeepWebResult(
        question="malaria nets efficacy",
        answer="Strong evidence nets reduce malaria.",
        sources=[],
        model_used="o3-deep-research",
        success=True,
    )
    with patch("src.core.agents.deep_web._primary_research", new=AsyncMock(return_value=mock_result)):
        result_state = await deep_web_graph.ainvoke(minimal_dw_state())
    assert result_state["result"] is not None
    assert result_state["result"]["success"] is True
    assert "Strong evidence" in result_state["result"]["result"]


@pytest.mark.asyncio
async def test_deep_web_graph_fallback_path():
    with (
        patch("src.core.agents.deep_web._primary_research", new=AsyncMock(side_effect=Exception("No API"))),
        patch("src.graph.subgraphs.deep_web.acall_llm", new=AsyncMock(return_value="Fallback round answer.")),
    ):
        result_state = await deep_web_graph.ainvoke(minimal_dw_state())
    assert result_state["result"] is not None
    # Should have run fallback rounds
    assert len(result_state.get("search_round_results", [])) > 0
