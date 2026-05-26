from __future__ import annotations

import pytest


def _get_all_graphs():
    graphs = {}
    from src.graph.subgraphs.causal import causal_graph
    graphs["causal"] = causal_graph
    from src.graph.subgraphs.analyze import analyze_graph
    graphs["analyze"] = analyze_graph
    from src.graph.subgraphs.research import research_graph
    graphs["research"] = research_graph
    from src.graph.agents.slr_graph import slr_graph
    graphs["slr"] = slr_graph
    from src.graph.agents.lbd_graph import lbd_graph
    graphs["lbd"] = lbd_graph
    from src.graph.agents.deep_web_graph import deep_web_graph
    graphs["deep_web"] = deep_web_graph
    from src.graph.agents.edison_graph import edison_graph
    graphs["edison"] = edison_graph
    return graphs


def test_all_graphs_compile():
    graphs = _get_all_graphs()
    assert len(graphs) == 7


def _get_dynamic_fanout_workers(drawable) -> set[str]:
    """Return nodes that appear isolated in static topology — these are Send()
    fan-out workers whose edges are created dynamically at runtime, not at
    graph compilation time. They are valid graph nodes; the check skips them."""
    all_nodes = set(drawable.nodes.keys())
    nodes_with_incoming = {e.target for e in drawable.edges}
    nodes_with_outgoing = {e.source for e in drawable.edges}
    # A node is a dynamic worker if it has neither static incoming NOR static outgoing edges
    # (excluding __start__ and __end__ which are structurally isolated by design)
    isolated = all_nodes - nodes_with_incoming - nodes_with_outgoing - {"__start__", "__end__"}
    return isolated


def test_no_isolated_nodes():
    """Every non-dynamic node must have at least one incoming edge (except __start__).
    Dynamic Send() fan-out workers are detected and excluded automatically.
    """
    for name, g in _get_all_graphs().items():
        drawable = g.get_graph()
        all_nodes = set(drawable.nodes.keys())
        nodes_with_incoming = {e.target for e in drawable.edges}
        dynamic_workers = _get_dynamic_fanout_workers(drawable)
        # __start__ has no incoming edge by definition
        isolated = all_nodes - nodes_with_incoming - {"__start__"} - dynamic_workers
        assert not isolated, f"{name}: isolated nodes (no incoming edges): {isolated}"


def test_no_dead_end_non_terminal_nodes():
    """Every non-END, non-dynamic node must have at least one outgoing edge.
    Dynamic Send() fan-out workers are detected and excluded automatically.
    """
    for name, g in _get_all_graphs().items():
        drawable = g.get_graph()
        all_nodes = set(drawable.nodes.keys())
        nodes_with_outgoing = {e.source for e in drawable.edges}
        dynamic_workers = _get_dynamic_fanout_workers(drawable)
        # __end__ is the only legitimate dead end
        dead_ends = all_nodes - nodes_with_outgoing - {"__end__"} - dynamic_workers
        assert not dead_ends, f"{name}: dead-end nodes (no outgoing edges): {dead_ends}"


def test_causal_graph_has_expected_nodes():
    from src.graph.subgraphs.causal import causal_graph
    nodes = set(causal_graph.nodes.keys())
    expected = {
        "evaluate_investment_rubric",
        "collect_evidence_packs",
        "forecast_consequences",
        "enrich_bow_context_worker",
        "collect_bow_enrichment",
        "dispatch_link_investigations",
        "investigate_link",
        "collect_link_assessments",
        "synthesize_findings",
        "critique_synthesis",
        "identify_gaps",
        "dispatch_science_investigations",
        "investigate_science_assumption",
        "collect_science_results",
        "necessity_check",
        "project_scope_decisions",
        "collect_decisions",
    }
    missing = expected - nodes
    assert not missing, f"causal graph missing nodes: {missing}"


def test_slr_graph_has_expected_nodes():
    from src.graph.agents.slr_graph import slr_graph
    nodes = set(slr_graph.nodes.keys())
    expected = {
        "slr_agent",
        "slr_tools",
        "slr_collect_papers",
        "slr_synthesise",
        "slr_finalise",
    }
    missing = expected - nodes
    assert not missing, f"slr graph missing nodes: {missing}"


def test_lbd_graph_has_expected_nodes():
    from src.graph.agents.lbd_graph import lbd_graph
    nodes = set(lbd_graph.nodes.keys())
    expected = {
        "lbd_agent",
        "lbd_tools",
        "lbd_collect_papers",
        "lbd_discover_connections",
        "lbd_synthesise",
        "lbd_finalise",
    }
    missing = expected - nodes
    assert not missing, f"lbd graph missing nodes: {missing}"
