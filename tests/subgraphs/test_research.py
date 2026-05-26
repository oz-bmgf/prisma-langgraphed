"""Unit tests for src/graph/subgraphs/research.py."""
from __future__ import annotations

import json
import sys
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from langgraph.types import Send

from src.graph.subgraphs.research import (
    aggregate_research_results,
    deep_web_worker,
    edison_worker,
    fan_out_research_tasks,
    lbd_worker,
    research_graph,
    slr_worker,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _task(task_id: str, task_type: str, query: str = "q", linked_scope: str = "S1") -> dict:
    return {"id": task_id, "type": task_type, "query": query, "linked_scope": linked_scope, "priority": "important"}


def _base_state(**overrides: Any) -> dict:
    state: dict = {
        "research_plan": [],
        "research_dir": "",
        "research_results": [],
        "dispatch_results": None,
        "edison_results": None,
        "errors": [],
    }
    state.update(overrides)
    return state


def _worker_state(
    task_id: str = "T1",
    task_type: str = "slr",
    query: str = "test query",
    linked_scope: str = "S1",
) -> dict:
    return {
        "task_id": task_id,
        "task_type": task_type,
        "query": query,
        "linked_scope": linked_scope,
        "priority": "important",
        "result": None,
    }


# ---------------------------------------------------------------------------
# fan_out_research_tasks
# ---------------------------------------------------------------------------


async def test_fan_out_correct_count():
    plan = [
        _task("T1", "slr"),
        _task("T2", "lbd"),
        _task("T3", "deep_web"),
        _task("T4", "edison"),
    ]
    state = _base_state(research_plan=plan)
    sends = await fan_out_research_tasks(state)
    assert len(sends) == 4


async def test_fan_out_routes_by_type():
    plan = [
        _task("T1", "slr"),
        _task("T2", "lbd"),
        _task("T3", "deep_web"),
        _task("T4", "edison"),
    ]
    state = _base_state(research_plan=plan)
    sends = await fan_out_research_tasks(state)

    by_id = {s.arg["task_id"]: s.node for s in sends}
    assert by_id["T1"] == "slr_worker"
    assert by_id["T2"] == "lbd_worker"
    assert by_id["T3"] == "deep_web_worker"
    assert by_id["T4"] == "edison_worker"


async def test_fan_out_unknown_type_falls_back_to_deep_web():
    plan = [_task("T1", "unknown_type")]
    state = _base_state(research_plan=plan)
    sends = await fan_out_research_tasks(state)
    assert sends[0].node == "deep_web_worker"


async def test_fan_out_empty_plan():
    # Empty plan returns fallback string (conditional edge router pattern)
    state = _base_state(research_plan=[])
    result = await fan_out_research_tasks(state)
    assert result == "aggregate_research_results"


async def test_fan_out_skips_already_done_tasks():
    """Router skips task_ids already in research_results — state is the single source of truth."""
    plan = [_task("T1", "slr"), _task("T2", "lbd")]
    state = _base_state(
        research_plan=plan,
        research_results=[{"task_id": "T1", "task_type": "slr"}],
    )
    sends = await fan_out_research_tasks(state)
    assert len(sends) == 1
    assert sends[0].arg["task_id"] == "T2"


async def test_fan_out_no_research_dir_in_payload():
    plan = [_task("T1", "slr")]
    state = _base_state(research_plan=plan)
    sends = await fan_out_research_tasks(state)
    assert "research_dir" not in sends[0].arg


async def test_fan_out_maps_type_key_to_task_type():
    # research_plan items use "type" key per ARCHITECTURE.md §2
    plan = [{"id": "T1", "type": "slr", "query": "q", "linked_scope": "S1", "priority": "critical"}]
    state = _base_state(research_plan=plan)
    sends = await fan_out_research_tasks(state)
    assert sends[0].node == "slr_worker"
    assert sends[0].arg["task_type"] == "slr"


async def test_fan_out_also_accepts_task_type_key():
    # Handles both "type" and "task_type" key spellings
    plan = [{"id": "T1", "task_type": "lbd", "query": "q", "linked_scope": "S1", "priority": "important"}]
    state = _base_state(research_plan=plan)
    sends = await fan_out_research_tasks(state)
    assert sends[0].node == "lbd_worker"


# ---------------------------------------------------------------------------
# slr_worker
# ---------------------------------------------------------------------------


async def test_slr_worker_always_calls_graph():
    """Workers always run — idempotency is handled by the router, not workers."""
    state_return = {"result": {"task_id": "T1", "thesis": "", "papers": [], "success": True}}
    with patch("src.graph.subgraphs.research.slr_graph") as mock_graph:
        mock_graph.ainvoke = AsyncMock(return_value=state_return)
        await slr_worker(_worker_state("T1", "slr"))
    mock_graph.ainvoke.assert_called_once()


async def test_slr_worker_cache_miss():
    state_return = {"result": {"task_id": "T1", "thesis": "test thesis", "papers": [], "success": True}}
    with patch("src.graph.subgraphs.research.slr_graph") as mock_graph:
        mock_graph.ainvoke = AsyncMock(return_value=state_return)
        result = await slr_worker(_worker_state("T1", "slr"))

    mock_graph.ainvoke.assert_called_once()
    assert result["research_results"][0]["task_type"] == "slr"
    assert result["research_results"][0]["thesis"] == "test thesis"


async def test_slr_worker_error_returns_stub():
    with patch("src.graph.subgraphs.research.slr_graph") as mock_graph:
        mock_graph.ainvoke = AsyncMock(side_effect=RuntimeError("API error"))
        result = await slr_worker(_worker_state("T1", "slr"))

    assert "error" in result["research_results"][0]
    assert result["research_results"][0]["task_type"] == "slr"


# ---------------------------------------------------------------------------
# lbd_worker
# ---------------------------------------------------------------------------


async def test_lbd_worker_always_calls_graph():
    """Workers always run — idempotency is handled by the router, not workers."""
    state_return = {"result": {"task_id": "T2", "thesis": "", "concepts": [], "papers": [], "success": True}}
    with patch("src.graph.subgraphs.research.lbd_graph") as mock_graph:
        mock_graph.ainvoke = AsyncMock(return_value=state_return)
        await lbd_worker(_worker_state("T2", "lbd"))
    mock_graph.ainvoke.assert_called_once()


async def test_lbd_worker_cache_miss():
    state_return = {"result": {"task_id": "T2", "thesis": "", "concepts": ["malaria"], "papers": [], "success": True}}
    with patch("src.graph.subgraphs.research.lbd_graph") as mock_graph:
        mock_graph.ainvoke = AsyncMock(return_value=state_return)
        result = await lbd_worker(_worker_state("T2", "lbd"))

    mock_graph.ainvoke.assert_called_once()
    assert result["research_results"][0]["task_type"] == "lbd"
    assert result["research_results"][0]["concepts"] == ["malaria"]


# ---------------------------------------------------------------------------
# deep_web_worker
# ---------------------------------------------------------------------------


async def test_deep_web_worker_always_calls_graph():
    """Workers always run — idempotency is handled by the router, not workers."""
    state_return = {"result": {"task_id": "T3", "result": "answer", "sources": [], "model_used": "", "search_rounds": 1, "success": True}}
    with patch("src.graph.subgraphs.research.deep_web_graph") as mock_graph:
        mock_graph.ainvoke = AsyncMock(return_value=state_return)
        await deep_web_worker(_worker_state("T3", "deep_web"))
    mock_graph.ainvoke.assert_called_once()


async def test_deep_web_worker_cache_miss():
    state_return = {"result": {"task_id": "T3", "result": "web answer", "answer": "web answer", "sources": [], "model_used": "o3", "search_rounds": 1, "success": True}}
    with patch("src.graph.subgraphs.research.deep_web_graph") as mock_graph:
        mock_graph.ainvoke = AsyncMock(return_value=state_return)
        result = await deep_web_worker(_worker_state("T3", "deep_web"))

    mock_graph.ainvoke.assert_called_once()
    assert result["research_results"][0]["task_type"] == "deep_web"
    assert result["research_results"][0]["result"] == "web answer"


# ---------------------------------------------------------------------------
# edison_worker
# ---------------------------------------------------------------------------


async def test_edison_worker_always_calls_graph():
    """Workers always run — idempotency is handled by the router, not workers."""
    state_return = {"result": {"task_id": "T4", "papers": [], "success": True}, "rewritten_query": "q"}
    with patch("src.graph.subgraphs.research.edison_graph") as mock_graph:
        mock_graph.ainvoke = AsyncMock(return_value=state_return)
        await edison_worker(_worker_state("T4", "edison"))
    mock_graph.ainvoke.assert_called_once()


async def test_edison_worker_calls_graph_with_original_query():
    state_return = {
        "result": {"task_id": "T4", "papers": [], "status": "ok", "success": True},
        "rewritten_query": "rewritten: malaria vaccine efficacy RCT",
    }
    with patch("src.graph.subgraphs.research.edison_graph") as mock_graph:
        mock_graph.ainvoke = AsyncMock(return_value=state_return)
        result = await edison_worker(_worker_state("T4", "edison", query="malaria vaccines"))

    mock_graph.ainvoke.assert_called_once()
    call_input = mock_graph.ainvoke.call_args[0][0]
    assert call_input["original_query"] == "malaria vaccines"
    assert result["research_results"][0]["rewritten_query"] == "rewritten: malaria vaccine efficacy RCT"


async def test_edison_worker_falls_back_to_original_on_graph_error():
    with patch("src.graph.subgraphs.research.edison_graph") as mock_graph:
        mock_graph.ainvoke = AsyncMock(side_effect=RuntimeError("graph fail"))
        result = await edison_worker(_worker_state("T4", "edison", query="original query"))

    assert result["research_results"][0]["rewritten_query"] == "original query"
    assert result["research_results"][0]["task_type"] == "edison"


async def test_edison_worker_error_returns_stub():
    with patch("src.graph.subgraphs.research.edison_graph") as mock_graph:
        mock_graph.ainvoke = AsyncMock(side_effect=RuntimeError("timeout"))
        result = await edison_worker(_worker_state("T4", "edison"))

    assert "error" in result["research_results"][0]
    assert result["research_results"][0]["task_type"] == "edison"


# ---------------------------------------------------------------------------
# aggregate_research_results
# ---------------------------------------------------------------------------


async def test_aggregate_splits_dispatch_and_edison():
    state = _base_state(research_results=[
        {"task_id": "T1", "task_type": "slr"},
        {"task_id": "T2", "task_type": "lbd"},
        {"task_id": "T3", "task_type": "deep_web"},
        {"task_id": "T4", "task_type": "edison"},
    ])
    result = await aggregate_research_results(state)
    assert len(result["dispatch_results"]) == 3
    assert len(result["edison_results"]) == 1
    assert result["edison_results"][0]["task_id"] == "T4"


async def test_aggregate_returns_split_results_in_state(tmp_path):
    state = _base_state(
        research_dir=str(tmp_path),
        research_results=[
            {"task_id": "T1", "task_type": "slr"},
            {"task_id": "T2", "task_type": "edison"},
        ],
    )
    result = await aggregate_research_results(state)

    assert result["dispatch_results"][0]["task_id"] == "T1"
    assert result["edison_results"][0]["task_id"] == "T2"
    # intermediate JSON files must NOT be written (state-first design)
    assert not (tmp_path / "dispatch_results.json").exists()
    assert not (tmp_path / "edison_results.json").exists()


async def test_aggregate_empty_results():
    state = _base_state(research_results=[])
    result = await aggregate_research_results(state)
    assert result["dispatch_results"] == []
    assert result["edison_results"] == []


# ---------------------------------------------------------------------------
# Compiled graph
# ---------------------------------------------------------------------------


def test_research_graph_compiles():
    assert research_graph is not None


def test_research_graph_has_all_nodes():
    # fan_out_research_tasks is a conditional edge router, not a node
    expected = {
        "slr_worker",
        "lbd_worker",
        "deep_web_worker",
        "edison_worker",
        "aggregate_research_results",
    }
    graph_nodes = set(research_graph.get_graph().nodes.keys()) - {"__start__", "__end__"}
    assert graph_nodes == expected
