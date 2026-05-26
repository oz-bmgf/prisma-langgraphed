"""Tests for src.core.tool_tracing — decorator and context buffer."""
from __future__ import annotations

import os
from pathlib import Path
from unittest import mock

import pytest


# ---------------------------------------------------------------------------
# test_traced_tool_success
# ---------------------------------------------------------------------------


async def test_traced_tool_success():
    from src.core.tool_tracing import traced_tool

    @traced_tool("test_tool")
    async def my_fn(x: int) -> int:
        return x * 2

    result, trace = await my_fn(5)
    assert result == 10
    assert trace["tool_name"] == "test_tool"
    assert "called_at" in trace
    assert isinstance(trace["duration_ms"], int)
    assert trace["success"] is True
    assert "error_message" not in trace or trace.get("error_message") is None


# ---------------------------------------------------------------------------
# test_traced_tool_failure
# ---------------------------------------------------------------------------


async def test_traced_tool_failure():
    from src.core.tool_tracing import traced_tool

    @traced_tool("failing_tool")
    async def bad_fn() -> str:
        raise ValueError("simulated failure")

    result, trace = await bad_fn()
    assert result is None
    assert trace["tool_name"] == "failing_tool"
    assert trace["success"] is False
    assert "simulated failure" in trace["error_message"]
    assert isinstance(trace["duration_ms"], int)


# ---------------------------------------------------------------------------
# test_trace_buffer_init_and_flush
# ---------------------------------------------------------------------------


def test_trace_buffer_init_and_flush():
    from src.core.tool_tracing import append_to_buffer, flush_trace_buffer, init_trace_buffer

    init_trace_buffer()
    append_to_buffer("web_search_traces", {"tool_name": "search_web", "query": "malaria"})
    flushed = flush_trace_buffer()

    assert len(flushed["web_search_traces"]) == 1
    assert flushed["web_search_traces"][0]["query"] == "malaria"
    # Buffer is empty after flush
    flushed2 = flush_trace_buffer()
    assert flushed2["web_search_traces"] == []


# ---------------------------------------------------------------------------
# test_asta_trace_schema
# ---------------------------------------------------------------------------


def test_asta_trace_schema():
    from src.core.output_schemas import AstaSearchTrace

    trace = AstaSearchTrace(
        tool_name="search_asta",
        called_at="2026-05-22T00:00:00+00:00",
        duration_ms=340,
        success=True,
        query="malaria vaccine efficacy",
        result_count=5,
        top_paper_ids=["abc123"],
        top_titles=["RTS,S trial"],
        index_used="semantic_scholar",
    )
    dumped = trace.model_dump()
    assert dumped["tool_name"] == "search_asta"
    assert dumped["result_count"] == 5
    assert isinstance(dumped["top_paper_ids"], list)


# ---------------------------------------------------------------------------
# test_investigation_trace_in_worker_result
# ---------------------------------------------------------------------------


async def test_investigation_trace_in_worker_result():
    """investigate_link returns investigation_traces in its result dict."""
    from src.graph.subgraphs.causal import investigate_link

    state = {
        "link_id": "L001",
        "inv_id": "INV-001",
        "bow_id": "BOW-001",
        "scope_id": "SCOPE-001",
        "claim": {"assumption": "test claim"},
        "model": "claude-sonnet-4-6",
        "result": None,
        "cache_dir": "",
    }

    async def _fake_run(**kwargs) -> dict:
        return {
            "link_id": "L001",
            "status": "sufficient",
            "iterations_used": 3,
            "terminal_status": "sufficient",
        }

    with mock.patch("src.core.investigation.run_investigation", new=_fake_run):
        result = await investigate_link(state)

    assert "link_assessments" in result
    assert "investigation_traces" in result
    assert len(result["investigation_traces"]) == 1
    inv_trace = result["investigation_traces"][0]
    assert "tool_call_breakdown" in inv_trace
    assert isinstance(inv_trace["tool_call_breakdown"], dict)


# ---------------------------------------------------------------------------
# test_summarise_traces
# ---------------------------------------------------------------------------


def test_summarise_traces():
    from src.core.tool_tracing import summarise_traces

    state = {
        "asta_traces": [
            {"tool_name": "search_asta", "duration_ms": 300, "success": True},
            {"tool_name": "search_asta", "duration_ms": 380, "success": True},
            {"tool_name": "search_asta", "duration_ms": 340, "success": False},
        ],
        "web_search_traces": [
            {"tool_name": "search_web", "duration_ms": 900, "success": True},
            {"tool_name": "search_web", "duration_ms": 870, "success": True},
        ],
    }
    summary = summarise_traces(state)
    assert summary["asta"]["count"] == 3
    assert summary["asta"]["error_count"] == 1
    assert summary["web_search"]["count"] == 2
    assert summary["web_search"]["error_count"] == 0


# ---------------------------------------------------------------------------
# test_api_traces_endpoint
# ---------------------------------------------------------------------------


async def test_api_traces_endpoint():
    """GET /runs/{thread_id}/traces returns trace fields."""
    from httpx import AsyncClient, ASGITransport
    from src.api import app

    mock_state_values = {
        "asta_traces": [{"tool_name": "search_asta", "query": "malaria"}],
        "slr_traces": [],
        "lbd_traces": [],
        "deep_web_traces": [],
        "edison_traces": [],
        "web_search_traces": [],
        "compute_traces": [],
        "collection_search_traces": [],
        "investigation_traces": [],
    }

    class _MockState:
        values = mock_state_values
        next = []

    with mock.patch.object(app.state, "graph", create=True) as mock_graph:
        mock_graph.aget_state = mock.AsyncMock(return_value=_MockState())

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/runs/MOCK%3A%3Atest-01/traces")

    assert resp.status_code == 200
    data = resp.json()
    assert "asta_traces" in data
    assert data["asta_traces"][0]["tool_name"] == "search_asta"
