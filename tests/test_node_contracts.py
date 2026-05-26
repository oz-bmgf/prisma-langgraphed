"""Node boundary contract tests (Task D — correctness audit).

Verifies that every node passes exactly the right data to the next node:
- trace reducer fields are initialised to [] in every subgraph input dict
- trace reducer fields are propagated back through every subgraph return dict
- WorkflowState.create_initial_state contains no extraneous fields
- research subgraph graph topology: fan_out is a router (not a node), workers
  all have edges to aggregate_research_results
"""
from __future__ import annotations

import sys
from unittest.mock import AsyncMock, MagicMock

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ALL_TRACE_FIELDS = [
    "asta_traces",
    "slr_traces",
    "lbd_traces",
    "deep_web_traces",
    "edison_traces",
    "web_search_traces",
    "compute_traces",
    "collection_search_traces",
    "investigation_traces",
]

_RESEARCH_TRACE_FIELDS = [
    "slr_traces",
    "lbd_traces",
    "deep_web_traces",
    "edison_traces",
]


def _minimal_analyze_state() -> dict:
    return {
        "program": "TestProg",
        "run_name": "test",
        "collection_name": "test",
        "base_dir": "/tmp/base",
        "ingested_dir": "/tmp/ingested",
        "doc_list": [{"file_id": "f1"}],
        "investment_scoring": {"INV-01": 0.9},
        "bow_investment_map": {"BOW-A": ["INV-01"]},
        "investment_intelligence": {"INV-01": {}},
        "chunks_json_path": "/tmp/chunks.json",
        "pages_dir": "/tmp/pages",
        "research_model": "claude-sonnet-4-6",
        "synthesis_model": "claude-sonnet-4-6",
        "focus": None,
        "focus_bows": None,
        "aux_collections": None,
        "threads_dir": None,
        "orientation_summary": None,
        "scopes": None,
        "scope_timelines": None,
        "clusters": None,
        "scope_outputs": None,
        "analyst_report": None,
        "final_report_md": None,
        "numerical_provenance": None,
        "verification_sources": None,
        "run_meta": None,
    }


def _empty_analyze_result() -> dict:
    return {
        "threads_dir": None,
        "final_report_md": "# R",
        "analyst_report": {},
        "scope_outputs": [],
        "scopes": [],
        "scope_timelines": {},
        "orientation_summary": "",
        "clusters": [],
        "evidence_packs": [],
        "link_assessments": [],
        "science_results": [],
        "scope_decisions": [],
        "excerpts_csv_path": None,
        "numerical_provenance": None,
        "verification_sources": None,
        "run_meta": None,
        **{f: [] for f in _ALL_TRACE_FIELDS},
        "errors": [],
    }


def _empty_research_result() -> dict:
    return {
        "research_results": [],
        "dispatch_results": [],
        "edison_results": [],
        **{f: [] for f in _RESEARCH_TRACE_FIELDS},
        "errors": [],
    }


# ---------------------------------------------------------------------------
# Contract 1: analyze node — trace fields initialised in subgraph input
# ---------------------------------------------------------------------------


async def test_analyze_node_initialises_all_trace_fields_in_subgraph_input():
    """analyze node must send all 9 trace reducer fields as [] to analyze_graph."""
    from src.graph.nodes.analyze import analyze

    captured: dict = {}

    async def _capture(input_dict, config):
        captured.update(input_dict)
        return _empty_analyze_result()

    mock_graph = MagicMock()
    mock_graph.ainvoke = _capture
    mock_module = MagicMock()
    mock_module.analyze_graph = mock_graph

    with __import__("unittest.mock", fromlist=["patch"]).patch.dict(
        sys.modules, {"src.graph.subgraphs.analyze": mock_module}
    ):
        await analyze(_minimal_analyze_state(), {})

    for field in _ALL_TRACE_FIELDS:
        assert field in captured, f"analyze_input missing trace field: {field}"
        assert captured[field] == [], f"analyze_input.{field} must be [], got {captured[field]!r}"


# ---------------------------------------------------------------------------
# Contract 2: analyze node — trace fields propagated back to WorkflowState
# ---------------------------------------------------------------------------


async def test_analyze_node_propagates_trace_fields_to_workflow_state():
    """analyze node return dict must include all 9 trace fields from the subgraph result."""
    from src.graph.nodes.analyze import analyze

    trace_payload = {f: [{"tool_name": f}] for f in _ALL_TRACE_FIELDS}
    mock_result = {**_empty_analyze_result(), **trace_payload}

    mock_graph = MagicMock()
    mock_graph.ainvoke = AsyncMock(return_value=mock_result)
    mock_module = MagicMock()
    mock_module.analyze_graph = mock_graph

    with __import__("unittest.mock", fromlist=["patch"]).patch.dict(
        sys.modules, {"src.graph.subgraphs.analyze": mock_module}
    ):
        result = await analyze(_minimal_analyze_state(), {})

    for field in _ALL_TRACE_FIELDS:
        assert field in result, f"analyze return missing trace field: {field}"
        assert result[field] == [{"tool_name": field}], (
            f"analyze.{field} not propagated: {result[field]!r}"
        )


# ---------------------------------------------------------------------------
# Contract 3: research node — trace fields initialised in subgraph input
# ---------------------------------------------------------------------------


async def test_research_node_initialises_trace_fields_in_subgraph_input():
    """research node must send all 4 research trace fields as [] to research_graph."""
    from src.graph.nodes.research import research

    captured: dict = {}

    async def _capture(input_dict, config):
        captured.update(input_dict)
        return _empty_research_result()

    mock_graph = MagicMock()
    mock_graph.ainvoke = _capture
    mock_module = MagicMock()
    mock_module.research_graph = mock_graph

    state = {
        "program": "X",
        "research_plan": [],
        "output_dir": "/tmp/out",
        "threads_dir": None,
        "research_model": "claude-sonnet-4-6",
        "synthesis_model": "claude-sonnet-4-6",
    }

    with __import__("unittest.mock", fromlist=["patch"]).patch.dict(
        sys.modules, {"src.graph.subgraphs.research": mock_module}
    ):
        await research(state, {})

    for field in _RESEARCH_TRACE_FIELDS:
        assert field in captured, f"research_input missing trace field: {field}"
        assert captured[field] == [], f"research_input.{field} must be [], got {captured[field]!r}"


# ---------------------------------------------------------------------------
# Contract 4: research node — trace fields propagated back to WorkflowState
# ---------------------------------------------------------------------------


async def test_research_node_propagates_trace_fields_to_workflow_state():
    """research node return dict must include all 4 trace fields from the subgraph result."""
    from src.graph.nodes.research import research

    trace_payload = {f: [{"tool_name": f}] for f in _RESEARCH_TRACE_FIELDS}
    mock_result = {**_empty_research_result(), **trace_payload}

    mock_graph = MagicMock()
    mock_graph.ainvoke = AsyncMock(return_value=mock_result)
    mock_module = MagicMock()
    mock_module.research_graph = mock_graph

    state = {
        "program": "X",
        "research_plan": [],
        "output_dir": "/tmp/out",
        "threads_dir": None,
        "research_model": "claude-sonnet-4-6",
        "synthesis_model": "claude-sonnet-4-6",
    }

    with __import__("unittest.mock", fromlist=["patch"]).patch.dict(
        sys.modules, {"src.graph.subgraphs.research": mock_module}
    ):
        result = await research(state, {})

    for field in _RESEARCH_TRACE_FIELDS:
        assert field in result, f"research return missing trace field: {field}"
        assert result[field] == [{"tool_name": field}], (
            f"research.{field} not propagated: {result[field]!r}"
        )


# ---------------------------------------------------------------------------
# Contract 5: run_causal_pipeline — trace fields initialised in causal_input
# ---------------------------------------------------------------------------


async def test_run_causal_pipeline_initialises_all_trace_fields():
    """run_causal_pipeline must send all 9 trace reducer fields as [] to causal_graph."""
    from src.graph.subgraphs.analyze import run_causal_pipeline

    captured: dict = {}

    async def _capture(input_dict, config):
        captured.update(input_dict)
        return {
            "scope_outputs": [],
            "evidence_packs": [],
            "link_assessments": [],
            "science_results": [],
            "scope_decisions": [],
            **{f: [] for f in _ALL_TRACE_FIELDS},
            "errors": [],
        }

    mock_causal_graph = MagicMock()
    mock_causal_graph.ainvoke = _capture
    mock_causal_module = MagicMock()
    mock_causal_module.causal_graph = mock_causal_graph

    state = {
        "scopes": [],
        "scope_timelines": {},
        "research_model": "claude-sonnet-4-6",
        "synthesis_model": "claude-sonnet-4-6",
        "threads_dir": "/tmp/threads",
    }

    with __import__("unittest.mock", fromlist=["patch"]).patch.dict(
        sys.modules, {"src.graph.subgraphs.causal": mock_causal_module}
    ):
        await run_causal_pipeline(state)

    for field in _ALL_TRACE_FIELDS:
        assert field in captured, f"causal_input missing trace field: {field}"
        assert captured[field] == [], f"causal_input.{field} must be [], got {captured[field]!r}"


# ---------------------------------------------------------------------------
# Contract 6: run_causal_pipeline — trace fields propagated back to AnalyzeState
# ---------------------------------------------------------------------------


async def test_run_causal_pipeline_propagates_trace_fields_to_analyze_state():
    """run_causal_pipeline must propagate all 9 trace fields from causal result."""
    from src.graph.subgraphs.analyze import run_causal_pipeline

    trace_payload = {f: [{"tool_name": f}] for f in _ALL_TRACE_FIELDS}
    causal_result = {
        "scope_outputs": [],
        "evidence_packs": [],
        "link_assessments": [],
        "science_results": [],
        "scope_decisions": [],
        **trace_payload,
        "errors": [],
    }

    mock_causal_graph = MagicMock()
    mock_causal_graph.ainvoke = AsyncMock(return_value=causal_result)
    mock_causal_module = MagicMock()
    mock_causal_module.causal_graph = mock_causal_graph

    state = {
        "scopes": [],
        "scope_timelines": {},
        "research_model": "claude-sonnet-4-6",
        "synthesis_model": "claude-sonnet-4-6",
        "threads_dir": None,
    }

    with __import__("unittest.mock", fromlist=["patch"]).patch.dict(
        sys.modules, {"src.graph.subgraphs.causal": mock_causal_module}
    ):
        result = await run_causal_pipeline(state)

    for field in _ALL_TRACE_FIELDS:
        assert field in result, f"run_causal_pipeline return missing trace field: {field}"
        assert result[field] == [{"tool_name": field}], (
            f"run_causal_pipeline.{field} not propagated: {result[field]!r}"
        )


# ---------------------------------------------------------------------------
# Contract 7: create_initial_state — correct reducer fields, no extraneous fields
# ---------------------------------------------------------------------------


def test_create_initial_state_contains_all_trace_reducer_fields():
    """create_initial_state must include all 9 trace reducer fields initialised as []."""
    from src.graph.workflow import create_initial_state

    state = create_initial_state(
        program="P",
        run_name="r",
        collection_name="c",
        base_dir="/b",
        ingested_dir="/i",
    )

    for field in _ALL_TRACE_FIELDS:
        assert field in state, f"create_initial_state missing: {field}"
        assert state[field] == [], f"create_initial_state.{field} must be [], got {state[field]!r}"


def test_create_initial_state_does_not_contain_timeline_narrative_results():
    """timeline_narrative_results is AnalyzeState-internal and must not be in WorkflowState."""
    from src.graph.workflow import create_initial_state

    state = create_initial_state(
        program="P",
        run_name="r",
        collection_name="c",
        base_dir="/b",
        ingested_dir="/i",
    )

    assert "timeline_narrative_results" not in state, (
        "timeline_narrative_results is AnalyzeState-internal; must not appear in WorkflowState"
    )


# ---------------------------------------------------------------------------
# Contract 8: research subgraph graph topology
# ---------------------------------------------------------------------------


def test_research_subgraph_fan_out_is_not_a_node():
    """fan_out_research_tasks must be a conditional edge router, not a node."""
    from src.graph.subgraphs.research import research_graph

    node_names = set(research_graph.get_graph().nodes.keys()) - {"__start__", "__end__"}
    assert "fan_out_research_tasks" not in node_names, (
        "fan_out_research_tasks is a conditional edge router and must not appear as a node"
    )


def test_research_subgraph_all_worker_nodes_present():
    """All 4 worker nodes and the aggregator must be present in the research graph."""
    from src.graph.subgraphs.research import research_graph

    node_names = set(research_graph.get_graph().nodes.keys()) - {"__start__", "__end__"}
    for expected in ("slr_worker", "lbd_worker", "deep_web_worker", "edison_worker", "aggregate_research_results"):
        assert expected in node_names, f"research graph missing node: {expected}"


def test_research_subgraph_workers_connect_to_aggregate():
    """Each worker node must have an edge to aggregate_research_results.

    Uses research_graph.builder.edges because Send()-dispatched worker edges are
    not reflected in get_graph() static rendering — they ARE wired, just not rendered.
    """
    from src.graph.subgraphs.research import research_graph

    # research_graph.builder.edges contains the statically registered edges
    edges = set(research_graph.builder.edges)
    workers = ("slr_worker", "lbd_worker", "deep_web_worker", "edison_worker")
    for worker in workers:
        assert (worker, "aggregate_research_results") in edges, (
            f"{worker} has no edge to aggregate_research_results"
        )


# ---------------------------------------------------------------------------
# Contract 9: fan_out_research_tasks — return type contract
# ---------------------------------------------------------------------------


async def test_fan_out_returns_list_of_send_for_non_empty_plan():
    """Non-empty plan must yield a list of Send objects (not a string)."""
    from langgraph.types import Send
    from src.graph.subgraphs.research import fan_out_research_tasks

    state = {
        "research_plan": [{"id": "T1", "type": "slr", "query": "q", "linked_scope": "S", "priority": "important"}],
        "research_dir": "",
    }
    result = await fan_out_research_tasks(state)
    assert isinstance(result, list)
    assert all(isinstance(s, Send) for s in result)


async def test_fan_out_returns_fallback_string_for_empty_plan():
    """Empty plan must return the fallback node name string (not [])."""
    from src.graph.subgraphs.research import fan_out_research_tasks

    state = {"research_plan": [], "research_dir": ""}
    result = await fan_out_research_tasks(state)
    assert isinstance(result, str)
    assert result == "aggregate_research_results"
