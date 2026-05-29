"""Unit tests for src/graph/workflow.py."""
from __future__ import annotations

from unittest.mock import patch

from langgraph.graph import END

from src.graph.workflow import (
    _HUMAN_INTERRUPT_NODES,
    compile_graph,
    create_initial_state,
    make_thread_id,
    route_after_precheck,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _state(**overrides) -> dict:
    base = {"precheck_passed": None}
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# compile_graph
# ---------------------------------------------------------------------------


def test_graph_compiles():
    graph = compile_graph()
    assert graph is not None


def test_compile_graph_default_no_interrupts():
    with patch("src.graph.workflow.CHECKPOINT_HUMAN_INTERRUPTS", False):
        graph = compile_graph()
        assert graph.interrupt_before_nodes == [], \
            f"Expected no interrupts, got: {graph.interrupt_before_nodes}"


def test_compile_graph_interactive_flag():
    graph = compile_graph(human_interrupts=True)
    interrupts = graph.interrupt_before_nodes
    assert len(interrupts) == len(_HUMAN_INTERRUPT_NODES)
    assert "analyze" in interrupts


def test_graph_has_all_nodes():
    graph = compile_graph()
    node_names = set(graph.get_graph().nodes.keys()) - {"__start__", "__end__"}
    expected = {
        "load_collection",
        "precheck",
        "analyze",
        "rerender",
        "deliver",
    }
    assert node_names == expected


# ---------------------------------------------------------------------------
# route_after_precheck
# ---------------------------------------------------------------------------


def test_route_after_precheck_pass():
    assert route_after_precheck(_state(precheck_passed=True)) == "analyze"


def test_route_after_precheck_fail():
    assert route_after_precheck(_state(precheck_passed=False)) == END


def test_route_after_precheck_none_fails():
    assert route_after_precheck(_state(precheck_passed=None)) == END


# ---------------------------------------------------------------------------
# create_initial_state
# ---------------------------------------------------------------------------


def test_create_initial_state_initialises_reducer_fields():
    state = create_initial_state(
        program="Malaria",
        run_name="crimson-falcon",
        collection_name="malaria",
        base_dir="/tmp/base",
        ingested_dir="/tmp/ingested",
    )
    # All Annotated[list, operator.add] reducer fields must be [] not None
    assert state["evidence_packs"] == []
    assert state["link_assessments"] == []
    assert state["science_results"] == []
    assert state["scope_decisions"] == []
    assert state["research_results"] == []
    assert state["errors"] == []


def test_create_initial_state_sets_identity_fields():
    state = create_initial_state(
        program="HIV",
        run_name="silver-osprey",
        collection_name="hiv",
        base_dir="/qpr",
        ingested_dir="/qpr/hiv-ingested",
        research_model="claude-opus-4-7",
        synthesis_model="claude-sonnet-4-6",
    )
    assert state["program"] == "HIV"
    assert state["run_name"] == "silver-osprey"
    assert state["collection_name"] == "hiv"
    assert state["research_model"] == "claude-opus-4-7"
    assert state["synthesis_model"] == "claude-sonnet-4-6"


def test_create_initial_state_optional_fields_default_none():
    state = create_initial_state(
        program="Malaria",
        run_name="test-run",
        collection_name="malaria",
        base_dir="/tmp",
        ingested_dir="/tmp/ingested",
    )
    assert state["focus"] is None
    assert state["focus_bows"] is None
    assert state["aux_collections"] is None


def test_create_initial_state_accepts_optional_overrides():
    state = create_initial_state(
        program="Malaria",
        run_name="test-run",
        collection_name="malaria",
        base_dir="/tmp",
        ingested_dir="/tmp/ingested",
        focus="Vaccine delivery",
        focus_bows=["BOW-A", "BOW-B"],
    )
    assert state["focus"] == "Vaccine delivery"
    assert state["focus_bows"] == ["BOW-A", "BOW-B"]


def test_create_initial_state_kwargs_forwarded():
    state = create_initial_state(
        program="TB",
        run_name="test",
        collection_name="tb",
        base_dir="/tmp",
        ingested_dir="/tmp/tb-ingested",
        current_stage="precheck",
    )
    assert state["current_stage"] == "precheck"


# ---------------------------------------------------------------------------
# make_thread_id
# ---------------------------------------------------------------------------


def test_make_thread_id_format():
    assert make_thread_id("Malaria", "crimson-falcon") == "Malaria::crimson-falcon"
    assert make_thread_id("HIV", "silver-osprey") == "HIV::silver-osprey"
