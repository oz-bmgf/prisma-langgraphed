"""Tests for src/graph/subgraphs/deep_web.py — canonical Deep Web subgraph location."""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from langgraph.types import Send

from src.graph.subgraphs.deep_web import (
    deep_web_collect_rounds,
    deep_web_dispatch_rounds,
    deep_web_finalise,
    deep_web_graph,
    deep_web_route_after_primary,
    deep_web_search_round,
    deep_web_synthesise_fallback,
    deep_web_try_primary,
)
from src.graph.state import DeepWebAgentState, DeepWebSearchRoundState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _state(**overrides) -> dict:
    base: dict = {
        "task_id": "DW1",
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


# ---------------------------------------------------------------------------
# test_deep_web_dispatch_skips_completed
# Task spec: router sends to finalise when primary succeeded; dispatches rounds when not
# ---------------------------------------------------------------------------


async def test_deep_web_dispatch_goes_to_finalise_on_primary_success():
    """deep_web_route_after_primary returns 'deep_web_finalise' when primary succeeded."""
    state = _state(primary_result={"success": True, "answer": "nets are effective"})
    result = await deep_web_route_after_primary(state)
    assert result == "deep_web_finalise"


async def test_deep_web_dispatch_sends_rounds_on_primary_failure():
    """deep_web_route_after_primary emits Send()s when primary failed."""
    from src.config import DEEP_WEB_MAX_ROUNDS
    state = _state(primary_result={"success": False, "error_message": "timeout"})
    result = await deep_web_route_after_primary(state)
    assert isinstance(result, list)
    assert len(result) == DEEP_WEB_MAX_ROUNDS
    assert all(isinstance(s, Send) for s in result)
    assert all(s.node == "deep_web_search_round" for s in result)


async def test_deep_web_dispatch_sends_rounds_when_primary_none():
    """deep_web_route_after_primary dispatches rounds when primary_result is None."""
    state = _state(primary_result=None)
    result = await deep_web_route_after_primary(state)
    assert isinstance(result, list)
    assert len(result) > 0


def test_deep_web_dispatch_rounds_helper_exposed_for_testing():
    """deep_web_dispatch_rounds (sync helper) returns correct round count."""
    from src.config import DEEP_WEB_MAX_ROUNDS
    state = _state()
    result = deep_web_dispatch_rounds(state)
    assert isinstance(result, list)
    assert len(result) == DEEP_WEB_MAX_ROUNDS


# ---------------------------------------------------------------------------
# test_deep_web_worker_pure
# Task spec: worker returns plain dict, no side effects
# ---------------------------------------------------------------------------


async def test_deep_web_worker_pure_success():
    """deep_web_search_round returns search_round_results and trace on success."""
    worker_state: DeepWebSearchRoundState = {
        "round_number": 1,
        "question": "malaria nets efficacy",
        "prior_context": "",
        "result": None,
    }
    with patch("src.graph.subgraphs.deep_web.acall_llm", new=AsyncMock(return_value="Round 1 findings.")):
        result = await deep_web_search_round(worker_state, {})

    assert isinstance(result, dict)
    assert "search_round_results" in result
    assert result["search_round_results"][0]["success"] is True
    assert result["search_round_results"][0]["answer"] == "Round 1 findings."
    assert result["tool_traces"][0]["success"] is True


async def test_deep_web_worker_error_surfaces_to_errors_field():
    """deep_web_search_round LLM failure goes into errors[], not raises."""
    worker_state: DeepWebSearchRoundState = {
        "round_number": 2,
        "question": "malaria nets efficacy",
        "prior_context": "",
        "result": None,
    }
    with patch("src.graph.subgraphs.deep_web.acall_llm", new=AsyncMock(side_effect=RuntimeError("LLM error"))):
        result = await deep_web_search_round(worker_state, {})

    assert "errors" in result
    assert result["search_round_results"][0]["success"] is False
    assert result["tool_traces"][0]["success"] is False


async def test_deep_web_worker_skips_when_result_pre_populated():
    """deep_web_search_round returns early when result already set (idempotency)."""
    prepopulated = {"round_number": 1, "answer": "cached answer", "success": True}
    worker_state: DeepWebSearchRoundState = {
        "round_number": 1,
        "question": "test",
        "prior_context": "",
        "result": prepopulated,
    }
    result = await deep_web_search_round(worker_state, {})
    assert result == {"search_round_results": [prepopulated]}


# ---------------------------------------------------------------------------
# test_deep_web_graph_has_all_nodes
# ---------------------------------------------------------------------------


def test_deep_web_graph_has_all_nodes():
    """Compiled graph nodes match topology declared in module docstring."""
    graph_nodes = set(deep_web_graph.get_graph().nodes.keys()) - {"__start__", "__end__"}
    expected = {
        "deep_web_try_primary",
        "deep_web_search_round",
        "deep_web_collect_rounds",
        "deep_web_synthesise_fallback",
        "deep_web_finalise",
    }
    assert graph_nodes == expected


def test_deep_web_graph_compiles():
    assert deep_web_graph is not None


# ---------------------------------------------------------------------------
# test_deep_web_handles_empty_results
# ---------------------------------------------------------------------------


async def test_deep_web_synthesise_fallback_no_rounds():
    """deep_web_synthesise_fallback returns graceful message when no rounds."""
    state = _state(search_round_results=[])
    result = await deep_web_synthesise_fallback(state, {})
    assert "fallback_synthesis" in result
    assert "No search rounds" in result["fallback_synthesis"]


async def test_deep_web_synthesise_fallback_all_rounds_failed():
    """deep_web_synthesise_fallback handles all-failed rounds gracefully."""
    state = _state(search_round_results=[
        {"round_number": 1, "answer": "", "success": False},
    ])
    result = await deep_web_synthesise_fallback(state, {})
    assert "fallback_synthesis" in result
    assert "failed" in result["fallback_synthesis"].lower()


async def test_deep_web_synthesise_fallback_skips_when_done():
    """deep_web_synthesise_fallback returns {} when fallback_synthesis already set."""
    state = _state(fallback_synthesis="Existing synthesis.")
    result = await deep_web_synthesise_fallback(state, {})
    assert result == {}


async def test_deep_web_synthesise_fallback_llm_failure_surfaces_error():
    """LLM failure in deep_web_synthesise_fallback surfaces to errors; uses best-effort fallback."""
    state = _state(search_round_results=[
        {"round_number": 1, "answer": "Round 1 answer", "success": True},
    ])
    with patch("src.graph.subgraphs.deep_web.acall_llm", new=AsyncMock(side_effect=RuntimeError("LLM down"))):
        result = await deep_web_synthesise_fallback(state, {})
    assert any("deep_web_synthesise_fallback" in e for e in result.get("errors", []))
    assert result["fallback_synthesis"] == "Round 1 answer"


async def test_deep_web_finalise_adds_status_field():
    """deep_web_finalise adds status: ok/error for finalize.py enrichment gate."""
    state = _state(primary_result={"success": True, "answer": "Primary answer", "sources": [], "model_used": "o3"})
    result = await deep_web_finalise(state, {})
    assert result["result"]["status"] == "ok"


async def test_deep_web_finalise_prefers_primary():
    """deep_web_finalise uses primary answer when primary succeeded."""
    state = _state(
        primary_result={"success": True, "answer": "Primary answer", "sources": ["url1"], "model_used": "o3"},
    )
    result = await deep_web_finalise(state, {})
    assert result["result"]["result"] == "Primary answer"
    assert result["result"]["model_used"] == "o3"


async def test_deep_web_finalise_falls_back_to_synthesis():
    """deep_web_finalise uses fallback_synthesis when primary failed."""
    state = _state(
        primary_result={"success": False},
        fallback_synthesis="Fallback answer.",
    )
    result = await deep_web_finalise(state, {})
    assert result["result"]["result"] == "Fallback answer."


# ---------------------------------------------------------------------------
# config threading
# ---------------------------------------------------------------------------


async def test_deep_web_search_round_passes_config_to_acall_llm():
    """acall_llm must receive config= so LLM calls appear as child spans."""
    captured: list[dict] = []

    async def _mock_acall(*args, config=None, **kwargs):
        captured.append({"config": config})
        return "Round answer."

    worker_state: DeepWebSearchRoundState = {
        "round_number": 1,
        "question": "malaria nets",
        "prior_context": "",
        "result": None,
    }
    with patch("src.graph.subgraphs.deep_web.acall_llm", side_effect=_mock_acall):
        await deep_web_search_round(worker_state, {"configurable": {"research_model": "test-model"}})

    assert len(captured) == 1
    assert captured[0]["config"] is not None


# ---------------------------------------------------------------------------
# Full dry-run through compiled graph
# ---------------------------------------------------------------------------


async def test_deep_web_graph_primary_path():
    """End-to-end via primary path."""
    from src.core.agents.deep_web import DeepWebResult
    mock_result = DeepWebResult(
        question="malaria nets efficacy",
        answer="Strong evidence nets reduce malaria.",
        sources=[],
        model_used="o3-deep-research",
        success=True,
    )
    with patch("src.core.agents.deep_web._primary_research", new=AsyncMock(return_value=mock_result)):
        final = await deep_web_graph.ainvoke(_state())
    assert final["result"] is not None
    assert final["result"]["success"] is True
    assert "Strong evidence" in final["result"]["result"]


async def test_deep_web_graph_fallback_path():
    """End-to-end via fallback path when primary fails."""
    with (
        patch("src.core.agents.deep_web._primary_research", new=AsyncMock(side_effect=Exception("No API"))),
        patch("src.graph.subgraphs.deep_web.acall_llm", new=AsyncMock(return_value="Fallback round answer.")),
    ):
        final = await deep_web_graph.ainvoke(_state())
    assert final["result"] is not None
    assert len(final.get("search_round_results", [])) > 0
