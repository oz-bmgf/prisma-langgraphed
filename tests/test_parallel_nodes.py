from __future__ import annotations

import pytest
from src.graph.state import merge_scope_outputs


def test_merge_scope_outputs_merges_by_scope_id():
    a = [{"scope_id": "s1", "causal_model": {"links": []}}]
    b = [{"scope_id": "s1", "bow_context": {"label": "X"}}]
    result = merge_scope_outputs(a, b)
    assert len(result) == 1
    assert "causal_model" in result[0]
    assert "bow_context" in result[0]
    assert result[0]["scope_id"] == "s1"


def test_merge_scope_outputs_appends_new_scopes():
    a = [{"scope_id": "s1", "causal_model": {}}]
    b = [{"scope_id": "s2", "causal_model": {}}]
    result = merge_scope_outputs(a, b)
    assert len(result) == 2
    ids = {r["scope_id"] for r in result}
    assert ids == {"s1", "s2"}


def test_merge_scope_outputs_empty_inputs():
    assert merge_scope_outputs([], []) == []
    assert merge_scope_outputs([{"scope_id": "s1"}], []) == [{"scope_id": "s1"}]
    assert merge_scope_outputs([], [{"scope_id": "s2"}]) == [{"scope_id": "s2"}]


def test_merge_scope_outputs_none_inputs():
    """None inputs are treated as empty lists — defensive for AnalyzeState initialisation."""
    assert merge_scope_outputs(None, None) == []
    assert merge_scope_outputs(None, [{"scope_id": "s1"}]) == [{"scope_id": "s1"}]
    assert merge_scope_outputs([{"scope_id": "s1"}], None) == [{"scope_id": "s1"}]


def test_merge_scope_outputs_update_wins_on_conflict():
    a = [{"scope_id": "s1", "label": "old"}]
    b = [{"scope_id": "s1", "label": "new", "extra": "yes"}]
    result = merge_scope_outputs(a, b)
    assert result[0]["label"] == "new"
    assert result[0]["extra"] == "yes"


def test_merge_scope_outputs_ignores_missing_scope_id():
    a = [{"causal_model": {}}]  # no scope_id
    b = [{"scope_id": "s1", "bow_context": {}}]
    result = merge_scope_outputs(a, b)
    assert len(result) == 1
    assert result[0]["scope_id"] == "s1"


def test_merge_scope_outputs_multiple_scopes():
    a = [
        {"scope_id": "s1", "causal_model": {"links": [1, 2]}},
        {"scope_id": "s2", "causal_model": {"links": [3]}},
    ]
    b = [
        {"scope_id": "s1", "bow_context": {"bow_id": "bow1"}},
        {"scope_id": "s3", "bow_context": {"bow_id": "bow3"}},
    ]
    result = merge_scope_outputs(a, b)
    by_id = {r["scope_id"]: r for r in result}
    assert set(by_id.keys()) == {"s1", "s2", "s3"}
    assert "causal_model" in by_id["s1"]
    assert "bow_context" in by_id["s1"]
    assert "causal_model" in by_id["s2"]
    assert "bow_context" not in by_id["s2"]
    assert "bow_context" in by_id["s3"]
    assert "causal_model" not in by_id["s3"]


def test_causal_subgraph_parallel_branches_compile():
    from src.graph.subgraphs.causal import causal_graph
    drawable = causal_graph.get_graph()
    edges = [(e.source, e.target) for e in drawable.edges]
    # gather_bow_context was replaced by dispatch_bow_enrichment fan-out (§3.1.5).
    # collect_evidence_packs now fans out to enrich_bow_context_worker and collect_bow_enrichment.
    cp_targets = [t for s, t in edges if s == "collect_evidence_packs"]
    assert "enrich_bow_context_worker" in cp_targets or "collect_bow_enrichment" in cp_targets, \
        f"expected bow enrichment fan-out targets from collect_evidence_packs: {cp_targets}"

    cbe_targets = [t for s, t in edges if s == "collect_bow_enrichment"]
    assert "forecast_consequences" in cbe_targets, \
        f"forecast_consequences not a target of collect_bow_enrichment: {cbe_targets}"

    # collect_link_assessments should have two outgoing edges (Change 2)
    cla_targets = [t for s, t in edges if s == "collect_link_assessments"]
    assert "synthesize_findings" in cla_targets, \
        f"synthesize_findings not a target of collect_link_assessments: {cla_targets}"
    assert "dispatch_science_investigations" in cla_targets, \
        f"dispatch_science_investigations not a target of collect_link_assessments: {cla_targets}"


def test_causal_join_nodes():
    from src.graph.subgraphs.causal import causal_graph
    edges = [(e.source, e.target) for e in causal_graph.get_graph().edges]
    # dispatch_link_investigations source is forecast_consequences (sequential: collect_bow_enrichment → forecast_consequences → dispatch_link_investigations)
    dli_sources = [s for s, t in edges if t == "dispatch_link_investigations"]
    assert "forecast_consequences" in dli_sources, \
        f"forecast_consequences not a source of dispatch_link_investigations: {dli_sources}"
    # necessity_check should have two incoming edges
    nc_sources = [s for s, t in edges if t == "necessity_check"]
    assert "identify_gaps" in nc_sources, \
        f"identify_gaps not a source of necessity_check: {nc_sources}"
    assert "collect_science_results" in nc_sources, \
        f"collect_science_results not a source of necessity_check: {nc_sources}"


def test_slr_graph_has_agent_tool_loop():
    """SLR uses an agent + ToolNode loop (Option B: LLM-driven tool selection)."""
    from src.graph.subgraphs.slr import slr_graph
    nodes = list(slr_graph.nodes.keys())
    assert "slr_agent" in nodes, f"slr_agent not in nodes: {nodes}"
    assert "slr_tools" in nodes, f"slr_tools not in nodes: {nodes}"

    drawable = slr_graph.get_graph()
    edges = [(e.source, e.target) for e in drawable.edges]
    # slr_tools must feed back into slr_agent (the loop)
    tools_targets = [t for s, t in edges if s == "slr_tools"]
    assert "slr_agent" in tools_targets, \
        f"slr_tools → slr_agent edge missing: {tools_targets}"


def test_lbd_graph_has_agent_tool_loop():
    """LBD uses an agent + ToolNode loop (Option B: LLM-driven tool selection)."""
    from src.graph.subgraphs.lbd import lbd_graph
    nodes = list(lbd_graph.nodes.keys())
    assert "lbd_agent" in nodes, f"lbd_agent not in nodes: {nodes}"
    assert "lbd_tools" in nodes, f"lbd_tools not in nodes: {nodes}"

    drawable = lbd_graph.get_graph()
    edges = [(e.source, e.target) for e in drawable.edges]
    # lbd_tools must feed back into lbd_agent (the loop)
    tools_targets = [t for s, t in edges if s == "lbd_tools"]
    assert "lbd_agent" in tools_targets, \
        f"lbd_tools → lbd_agent edge missing: {tools_targets}"
    # lbd_collect_papers must flow into lbd_discover_connections
    collect_targets = [t for s, t in edges if s == "lbd_collect_papers"]
    assert "lbd_discover_connections" in collect_targets, \
        f"lbd_collect_papers → lbd_discover_connections edge missing: {collect_targets}"


@pytest.mark.asyncio
async def test_merge_scope_outputs_reducer_wired_in_causal_state():
    import typing
    from src.graph.state import CausalState, merge_scope_outputs
    hints = typing.get_type_hints(CausalState, include_extras=True)
    scope_hint = hints.get("scope_outputs")
    if scope_hint is None:
        pytest.skip("scope_outputs not in CausalState hints")
    args = typing.get_args(scope_hint)
    assert merge_scope_outputs in args, \
        f"merge_scope_outputs not in scope_outputs annotation args: {args}"


@pytest.mark.asyncio
async def test_merge_scope_outputs_reducer_wired_in_analyze_state():
    import typing
    from src.graph.state import AnalyzeState, merge_scope_outputs
    hints = typing.get_type_hints(AnalyzeState, include_extras=True)
    scope_hint = hints.get("scope_outputs")
    if scope_hint is None:
        pytest.skip("scope_outputs not in AnalyzeState hints")
    args = typing.get_args(scope_hint)
    assert merge_scope_outputs in args, \
        f"merge_scope_outputs not in AnalyzeState.scope_outputs annotation args: {args}"
