"""Unit tests for src/graph/subgraphs/causal.py — all 21 nodes."""
from __future__ import annotations

import sys
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langgraph.types import Send

from src.graph.subgraphs.causal import (
    _route_link_investigations,
    _route_science_investigations,
    causal_graph,
    collect_bow_enrichment,
    collect_decisions,
    collect_evidence_packs,
    collect_link_assessments,
    collect_science_results,
    critique_synthesis,
    dispatch_bow_enrichment,
    dispatch_decision_projections,
    dispatch_link_investigations,
    dispatch_rubric_evaluation,
    dispatch_science_investigations,
    enrich_bow_context_worker,
    evaluate_investment_rubric,
    forecast_consequences,
    identify_gaps,
    investigate_link,
    investigate_science_assumption,
    necessity_check,
    project_scope_decisions,
    synthesize_findings,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _base_state(**overrides: Any) -> dict:
    state: dict = {
        "scopes": [],
        "scope_timelines": {},
        "research_model": "claude-sonnet-4-6",
        "synthesis_model": "claude-sonnet-4-6",
        "evidence_packs": [],
        "link_assessments": [],
        "science_results": [],
        "scope_decisions": [],
        "scope_outputs": [],
        "errors": [],
    }
    state.update(overrides)
    return state


def _scope(scope_id: str = "S1", inv_id: str = "INV-001", bow_ids: list[str] | None = None) -> dict:
    return {
        "scope_id": scope_id,
        "inv_id": inv_id,
        "bow_ids": bow_ids or ["BOW-A"],
        "label": f"scope {scope_id}",
    }


# ---------------------------------------------------------------------------
# GROUP 1 — dispatch_rubric_evaluation
# ---------------------------------------------------------------------------


async def test_dispatch_rubric_evaluation_sends_one_per_scope():
    state = _base_state(
        scopes=[_scope("S1", "INV-001"), _scope("S2", "INV-002")],
        scope_timelines={"S1": {"start": "2024-01"}, "S2": {"start": "2024-06"}},
    )
    sends = await dispatch_rubric_evaluation(state)
    assert len(sends) == 2
    assert all(s.node == "evaluate_investment_rubric" for s in sends)
    ids = {s.arg["inv_id"] for s in sends}
    assert ids == {"INV-001", "INV-002"}


async def test_dispatch_rubric_evaluation_empty_scopes():
    state = _base_state(scopes=[])
    result = await dispatch_rubric_evaluation(state)
    assert result == "collect_evidence_packs"


async def test_dispatch_rubric_evaluation_passes_timeline():
    state = _base_state(
        scopes=[_scope("S1", "INV-001")],
        scope_timelines={"S1": {"start": "2024-01", "end": "2026-12"}},
    )
    sends = await dispatch_rubric_evaluation(state)
    assert sends[0].arg["timeline"] == {"start": "2024-01", "end": "2026-12"}


async def test_dispatch_rubric_evaluation_skips_already_done_scopes():
    state = _base_state(
        scopes=[_scope("S1"), _scope("S2")],
        evidence_packs=[{"scope_id": "S1"}],
    )
    sends = await dispatch_rubric_evaluation(state)
    assert len(sends) == 1
    assert sends[0].arg["scope_id"] == "S2"


async def test_dispatch_rubric_evaluation_no_cache_dir_in_payload():
    state = _base_state(scopes=[_scope()])
    sends = await dispatch_rubric_evaluation(state)
    assert "cache_dir" not in sends[0].arg


# ---------------------------------------------------------------------------
# GROUP 1 — evaluate_investment_rubric (worker)
# ---------------------------------------------------------------------------


async def test_evaluate_investment_rubric_calls_core():
    mock_rubric_mod = MagicMock()
    mock_pack = MagicMock()
    mock_pack.to_dict.return_value = {"inv_id": "INV-001", "scope_id": "S1", "chunks": []}
    mock_rubric_mod.build_evidence_pack = AsyncMock(return_value=mock_pack)

    with patch.dict(sys.modules, {"src.core.rubric_evaluator": mock_rubric_mod}):
        result = await evaluate_investment_rubric({
            "inv_id": "INV-001", "scope_id": "S1",
            "timeline": {}, "result": None,
        })

    assert "evidence_packs" in result
    assert len(result["evidence_packs"]) == 1
    assert result["evidence_packs"][0]["inv_id"] == "INV-001"


async def test_evaluate_investment_rubric_error_returns_stub():
    with patch("src.core.rubric_evaluator.build_evidence_pack", new=AsyncMock(side_effect=RuntimeError("network error"))):
        result = await evaluate_investment_rubric({
            "inv_id": "INV-001", "scope_id": "S1",
            "timeline": {}, "result": None,
        })

    assert "evidence_packs" in result
    assert "error" in result["evidence_packs"][0]


async def test_dispatch_rubric_evaluation_state_skip_means_no_worker_called():
    """Router skips already-done scopes — state is the single source of truth."""
    state = _base_state(
        scopes=[_scope("S1")],
        evidence_packs=[{"scope_id": "S1", "inv_id": "INV-001"}],
    )
    sends = await dispatch_rubric_evaluation(state)
    assert sends == "collect_evidence_packs"


# ---------------------------------------------------------------------------
# GROUP 1 — collect_evidence_packs (reducer)
# ---------------------------------------------------------------------------


async def test_collect_evidence_packs_groups_by_scope():
    state = _base_state(
        scopes=[_scope("S1", "INV-001"), _scope("S2", "INV-002")],
        evidence_packs=[
            {"inv_id": "INV-001", "scope_id": "S1", "chunks": []},
            {"inv_id": "INV-002", "scope_id": "S2", "chunks": []},
        ],
    )
    result = await collect_evidence_packs(state)
    outputs = {s["scope_id"]: s for s in result["scope_outputs"]}
    assert "S1" in outputs and "S2" in outputs
    assert len(outputs["S1"]["evidence_packs"]) == 1
    assert len(outputs["S2"]["evidence_packs"]) == 1


async def test_collect_evidence_packs_initialises_all_scopes():
    state = _base_state(
        scopes=[_scope("S1"), _scope("S2")],
        evidence_packs=[],
    )
    result = await collect_evidence_packs(state)
    assert len(result["scope_outputs"]) == 2
    for so in result["scope_outputs"]:
        assert so["evidence_packs"] == []
        assert so["link_assessments"] == []


# ---------------------------------------------------------------------------
# GROUP 2 — forecast_consequences
# ---------------------------------------------------------------------------


async def test_forecast_consequences_attaches_causal_model():
    mock_cm = MagicMock()
    mock_cm.to_dict.return_value = {"links": [{"name": "L1"}], "assumptions": []}

    scope_outputs = [{"scope_id": "S1", "inv_id": "INV-001", "bow_ids": ["BOW-A"]}]
    state = _base_state(scope_outputs=scope_outputs)

    with patch("src.core.causal_model.extract_causal_model", new=AsyncMock(return_value=mock_cm)):
        result = await forecast_consequences(state)

    updated = result["scope_outputs"][0]
    assert updated["causal_model"]["links"][0]["name"] == "L1"


async def test_forecast_consequences_handles_error():
    scope_outputs = [{"scope_id": "S1", "inv_id": "INV-001"}]
    state = _base_state(scope_outputs=scope_outputs)

    with patch("src.core.causal_model.extract_causal_model", new=AsyncMock(side_effect=RuntimeError("LLM fail"))):
        result = await forecast_consequences(state)

    assert result["scope_outputs"][0]["causal_model"] is None
    assert len(result.get("errors", [])) == 1


# ---------------------------------------------------------------------------
# GROUP 2 — dispatch_bow_enrichment / enrich_bow_context_worker / collect_bow_enrichment
# ---------------------------------------------------------------------------


async def test_dispatch_bow_enrichment_sends_one_per_scope():
    scope_outputs = [
        {"scope_id": "S1", "inv_id": "INV-001", "bow_ids": ["BOW-A"]},
        {"scope_id": "S2", "inv_id": "INV-002", "bow_ids": ["BOW-B"]},
    ]
    state = _base_state(scope_outputs=scope_outputs)
    sends = await dispatch_bow_enrichment(state)
    assert isinstance(sends, list)
    assert len(sends) == 2
    assert all(s.node == "enrich_bow_context_worker" for s in sends)
    assert {s.arg["scope_id"] for s in sends} == {"S1", "S2"}


async def test_dispatch_bow_enrichment_skips_already_enriched():
    scope_outputs = [
        {"scope_id": "S1", "inv_id": "INV-001", "bow_ids": ["BOW-A"], "bow_context": {"bow_id": "BOW-A"}},
        {"scope_id": "S2", "inv_id": "INV-002", "bow_ids": ["BOW-B"]},
    ]
    state = _base_state(scope_outputs=scope_outputs)
    sends = await dispatch_bow_enrichment(state)
    assert len(sends) == 1
    assert sends[0].arg["scope_id"] == "S2"


async def test_dispatch_bow_enrichment_empty_scope_outputs():
    state = _base_state(scope_outputs=[])
    result = await dispatch_bow_enrichment(state)
    assert result == "collect_bow_enrichment"


async def test_enrich_bow_context_worker_returns_scope_with_bow_context():
    # With bow_ids but no web_search_fn in config → returns _empty_bow (no LLM call)
    scope = {"scope_id": "S1", "inv_id": "INV-001", "bow_ids": ["BOW-A", "BOW-B"]}
    worker_state = {"scope_id": "S1", "scope": scope, "model": "claude-sonnet-4-6", "result": None}
    result = await enrich_bow_context_worker(worker_state)
    assert "scope_outputs" in result
    assert len(result["scope_outputs"]) == 1
    updated = result["scope_outputs"][0]
    assert updated["scope_id"] == "S1"
    assert "bow_context" in updated
    assert updated["bow_context"]["bow_id"] == "BOW-A"


async def test_enrich_bow_context_worker_empty_bow_ids():
    scope = {"scope_id": "S1", "inv_id": "INV-001", "bow_ids": []}
    worker_state = {"scope_id": "S1", "scope": scope, "model": "", "result": None}
    result = await enrich_bow_context_worker(worker_state)
    updated = result["scope_outputs"][0]
    assert updated["bow_context"]["bow_id"] == ""


async def test_enrich_bow_context_worker_error_in_web_search_still_returns_scope():
    # Simulate web_search_fn raising; worker falls back to _empty_bow
    async def bad_search(q: str):
        raise RuntimeError("network error")

    scope = {"scope_id": "S1", "inv_id": "INV-001", "bow_ids": ["BOW-A"]}
    worker_state = {"scope_id": "S1", "scope": scope, "model": "", "result": None}
    config = {"configurable": {"web_search_fn": bad_search}}
    result = await enrich_bow_context_worker(worker_state, config=config)
    assert "scope_outputs" in result
    assert result["scope_outputs"][0]["scope_id"] == "S1"
    assert "bow_context" in result["scope_outputs"][0]


async def test_collect_bow_enrichment_is_trivial_join():
    state = _base_state(scope_outputs=[{"scope_id": "S1", "bow_context": {}}])
    result = await collect_bow_enrichment(state)
    assert result == {}


# ---------------------------------------------------------------------------
# GROUP 3 — dispatch_link_investigations
# ---------------------------------------------------------------------------


async def test_dispatch_link_investigations_is_passthrough():
    """dispatch_link_investigations is now a join node (returns {})."""
    scope_outputs = [{"scope_id": "S1", "inv_id": "INV-001", "bow_ids": ["BOW-A"],
                      "causal_model": {"links": [{"name": "L1"}]}}]
    state = _base_state(scope_outputs=scope_outputs)
    result = await dispatch_link_investigations(state)
    assert result == {}


async def test_dispatch_link_investigations_one_send_per_link():
    """_route_link_investigations (routing fn) emits one Send per causal link."""
    scope_outputs = [{
        "scope_id": "S1", "inv_id": "INV-001", "bow_ids": ["BOW-A"],
        "causal_model": {"links": [{"name": "L1"}, {"name": "L2"}, {"name": "L3"}]},
    }]
    state = _base_state(scope_outputs=scope_outputs)
    sends = await _route_link_investigations(state)
    assert len(sends) == 3
    assert all(s.node == "investigate_link" for s in sends)
    link_ids = {s.arg["link_id"] for s in sends}
    assert link_ids == {"L1", "L2", "L3"}


async def test_dispatch_link_investigations_no_causal_model():
    """_route_link_investigations falls back to 'collect_link_assessments' string when no links."""
    scope_outputs = [{"scope_id": "S1", "inv_id": "INV-001", "bow_ids": [], "causal_model": None}]
    state = _base_state(scope_outputs=scope_outputs)
    result = await _route_link_investigations(state)
    assert result == "collect_link_assessments"


async def test_dispatch_link_investigations_multiple_scopes():
    """_route_link_investigations emits Sends for links across all scopes."""
    scope_outputs = [
        {"scope_id": "S1", "inv_id": "INV-001", "bow_ids": ["BOW-A"],
         "causal_model": {"links": [{"name": "L1"}]}},
        {"scope_id": "S2", "inv_id": "INV-002", "bow_ids": ["BOW-B"],
         "causal_model": {"links": [{"name": "L2"}, {"name": "L3"}]}},
    ]
    state = _base_state(scope_outputs=scope_outputs)
    sends = await _route_link_investigations(state)
    assert len(sends) == 3
    scope_ids = {s.arg["scope_id"] for s in sends}
    assert scope_ids == {"S1", "S2"}


# ---------------------------------------------------------------------------
# GROUP 3 — investigate_link (worker)
# ---------------------------------------------------------------------------


async def test_investigate_link_calls_core():
    mock_result = MagicMock()
    mock_result.to_dict.return_value = {
        "link_id": "L1", "inv_id": "INV-001", "scope_id": "S1", "status": "confirmed"
    }

    with patch("src.core.investigation.run_investigation", new=AsyncMock(return_value=mock_result)):
        result = await investigate_link({
            "link_id": "L1", "inv_id": "INV-001", "bow_id": "BOW-A",
            "scope_id": "S1", "claim": {"name": "L1"}, "model": "m", "result": None,
        })

    assert "link_assessments" in result
    assert result["link_assessments"][0]["status"] == "confirmed"


async def test_investigate_link_error_returns_stub():
    with patch("src.core.investigation.run_investigation", new=AsyncMock(side_effect=RuntimeError("timeout"))):
        result = await investigate_link({
            "link_id": "L1", "inv_id": "INV-001", "bow_id": "BOW-A",
            "scope_id": "S1", "claim": {}, "model": "m", "result": None,
        })

    assert "error" in result["link_assessments"][0]


async def test_route_link_investigations_skips_already_done():
    """Router skips links already in link_assessments — state is the single source of truth."""
    scope_outputs = [
        {"scope_id": "S1", "inv_id": "INV-001", "bow_ids": ["BOW-A"],
         "causal_model": {"links": [{"name": "L1"}, {"name": "L2"}]}},
    ]
    state = _base_state(
        scope_outputs=scope_outputs,
        link_assessments=[{"scope_id": "S1", "link_id": "L1"}],
    )
    sends = await _route_link_investigations(state)
    assert len(sends) == 1
    assert sends[0].arg["link_id"] == "L2"


async def test_route_link_investigations_no_cache_dir_in_payload():
    scope_outputs = [
        {"scope_id": "S1", "inv_id": "INV-001", "bow_ids": ["BOW-A"],
         "causal_model": {"links": [{"name": "L1"}]}},
    ]
    state = _base_state(scope_outputs=scope_outputs)
    sends = await _route_link_investigations(state)
    assert "cache_dir" not in sends[0].arg


# ---------------------------------------------------------------------------
# GROUP 3 — collect_link_assessments (reducer)
# ---------------------------------------------------------------------------


async def test_collect_link_assessments_groups_by_scope():
    scope_outputs = [
        {"scope_id": "S1", "inv_id": "INV-001", "link_assessments": []},
        {"scope_id": "S2", "inv_id": "INV-002", "link_assessments": []},
    ]
    state = _base_state(
        scope_outputs=scope_outputs,
        link_assessments=[
            {"link_id": "L1", "scope_id": "S1"},
            {"link_id": "L2", "scope_id": "S1"},
            {"link_id": "L3", "scope_id": "S2"},
        ],
    )
    result = await collect_link_assessments(state)
    by_scope = {s["scope_id"]: s for s in result["scope_outputs"]}
    assert len(by_scope["S1"]["link_assessments"]) == 2
    assert len(by_scope["S2"]["link_assessments"]) == 1


# ---------------------------------------------------------------------------
# GROUP 4 — synthesize_findings
# ---------------------------------------------------------------------------


async def test_synthesize_findings_calls_acall_llm():
    scope_outputs = [{
        "scope_id": "S1", "inv_id": "INV-001",
        "link_assessments": [{"status": "confirmed"}],
        "synthesis": "",
    }]
    state = _base_state(scope_outputs=scope_outputs, synthesis_model="test-model")

    mock_acall = AsyncMock(return_value="synthesis text")
    with patch("src.graph.subgraphs.causal.acall_llm", mock_acall):
        result = await synthesize_findings(state)

    assert result["scope_outputs"][0]["synthesis"] == "synthesis text"
    mock_acall.assert_called_once()


async def test_synthesize_findings_skips_empty_link_assessments():
    scope_outputs = [{"scope_id": "S1", "link_assessments": [], "synthesis": ""}]
    state = _base_state(scope_outputs=scope_outputs)

    mock_acall = AsyncMock(return_value="should not be called")
    with patch("src.graph.subgraphs.causal.acall_llm", mock_acall):
        result = await synthesize_findings(state)

    mock_acall.assert_not_called()
    assert result["scope_outputs"][0]["synthesis"] == ""


# ---------------------------------------------------------------------------
# GROUP 4 — critique_synthesis
# ---------------------------------------------------------------------------


async def test_critique_synthesis_calls_acall_llm():
    scope_outputs = [{"scope_id": "S1", "synthesis": "A synthesis text", "critique": ""}]
    state = _base_state(scope_outputs=scope_outputs)

    mock_acall = AsyncMock(return_value="critique text")
    with patch("src.graph.subgraphs.causal.acall_llm", mock_acall):
        result = await critique_synthesis(state)

    assert result["scope_outputs"][0]["critique"] == "critique text"


async def test_critique_synthesis_skips_empty_synthesis():
    scope_outputs = [{"scope_id": "S1", "synthesis": "", "critique": ""}]
    state = _base_state(scope_outputs=scope_outputs)

    mock_acall = AsyncMock(return_value="should not be called")
    with patch("src.graph.subgraphs.causal.acall_llm", mock_acall):
        result = await critique_synthesis(state)

    mock_acall.assert_not_called()
    assert result["scope_outputs"][0]["critique"] == ""


# ---------------------------------------------------------------------------
# GROUP 4 — identify_gaps
# ---------------------------------------------------------------------------


async def test_identify_gaps_calls_acall_llm():
    scope_outputs = [{"scope_id": "S1", "synthesis": "synthesis", "gaps": ""}]
    state = _base_state(scope_outputs=scope_outputs)

    mock_acall = AsyncMock(return_value="gaps text")
    with patch("src.graph.subgraphs.causal.acall_llm", mock_acall):
        result = await identify_gaps(state)

    assert result["scope_outputs"][0]["gaps"] == "gaps text"


# ---------------------------------------------------------------------------
# GROUP 5 — dispatch_science_investigations
# ---------------------------------------------------------------------------


async def test_dispatch_science_investigations_one_send_per_assumption():
    scope_outputs = [{
        "scope_id": "S1", "inv_id": "INV-001", "bow_ids": ["BOW-A"],
        "causal_model": {
            "assumptions": [
                {"assumption": "Assume A"},
                {"assumption": "Assume B"},
            ],
        },
    }]
    state = _base_state(scope_outputs=scope_outputs)
    sends = await _route_science_investigations(state)
    assert len(sends) == 2
    assert all(s.node == "investigate_science_assumption" for s in sends)


async def test_dispatch_science_investigations_no_assumptions():
    scope_outputs = [{"scope_id": "S1", "inv_id": "INV-001", "bow_ids": [], "causal_model": None}]
    state = _base_state(scope_outputs=scope_outputs)
    result = await _route_science_investigations(state)
    assert result == "collect_science_results"


async def test_dispatch_science_investigations_is_passthrough():
    """dispatch_science_investigations is now a join node (returns {})."""
    scope_outputs = [{"scope_id": "S1", "inv_id": "INV-001", "bow_ids": [],
                      "causal_model": None}]
    state = _base_state(scope_outputs=scope_outputs)
    result = await dispatch_science_investigations(state)
    assert result == {}


# ---------------------------------------------------------------------------
# GROUP 5 — investigate_science_assumption (worker)
# ---------------------------------------------------------------------------


async def test_investigate_science_assumption_calls_core():
    mock_result = MagicMock()
    mock_result.to_dict.return_value = {
        "assumption_id": "S1_0", "scope_id": "S1", "terminal_status": "evidence_gathered"
    }

    with patch("src.core.science_investigator.investigate_science_question", new=AsyncMock(return_value=mock_result)):
        result = await investigate_science_assumption({
            "assumption_id": "S1_0", "inv_id": "INV-001", "bow_id": "BOW-A",
            "scope_id": "S1", "question": "Does A cause B?", "result": None,
        })

    assert "science_results" in result
    assert result["science_results"][0]["terminal_status"] == "evidence_gathered"


async def test_investigate_science_assumption_error_returns_stub():
    with patch("src.core.science_investigator.investigate_science_question", new=AsyncMock(side_effect=RuntimeError("API fail"))):
        result = await investigate_science_assumption({
            "assumption_id": "S1_0", "inv_id": "INV-001", "bow_id": "BOW-A",
            "scope_id": "S1", "question": "?", "result": None,
        })

    assert result["science_results"][0]["terminal_status"] == "error"


# ---------------------------------------------------------------------------
# GROUP 5 — collect_science_results (reducer)
# ---------------------------------------------------------------------------


async def test_collect_science_results_attaches_to_scopes():
    scope_outputs = [
        {"scope_id": "S1", "science_flags": []},
        {"scope_id": "S2", "science_flags": []},
    ]
    state = _base_state(
        scope_outputs=scope_outputs,
        science_results=[
            {"assumption_id": "S1_0", "scope_id": "S1"},
            {"assumption_id": "S2_0", "scope_id": "S2"},
            {"assumption_id": "S2_1", "scope_id": "S2"},
        ],
    )
    result = await collect_science_results(state)
    by_scope = {s["scope_id"]: s for s in result["scope_outputs"]}
    assert len(by_scope["S1"]["science_flags"]) == 1
    assert len(by_scope["S2"]["science_flags"]) == 2


# ---------------------------------------------------------------------------
# GROUP 6 — necessity_check
# ---------------------------------------------------------------------------


async def test_necessity_check_calls_acall_llm_per_scope():
    scope_outputs = [
        {"scope_id": "S1", "link_assessments": [{"status": "confirmed"}]},
        {"scope_id": "S2", "link_assessments": [{"status": "weak"}]},
    ]
    state = _base_state(scope_outputs=scope_outputs)

    mock_acall = AsyncMock(return_value="necessity assessment")
    with patch("src.graph.subgraphs.causal.acall_llm", mock_acall):
        result = await necessity_check(state)

    assert mock_acall.call_count == 2
    for so in result["scope_outputs"]:
        assert so["necessity_assessment"] == "necessity assessment"


async def test_necessity_check_skips_empty_links():
    scope_outputs = [{"scope_id": "S1", "link_assessments": []}]
    state = _base_state(scope_outputs=scope_outputs)

    mock_acall = AsyncMock(return_value="should not be called")
    with patch("src.graph.subgraphs.causal.acall_llm", mock_acall):
        result = await necessity_check(state)

    mock_acall.assert_not_called()
    assert result["scope_outputs"][0]["necessity_assessment"] == ""


# ---------------------------------------------------------------------------
# GROUP 6 — dispatch_decision_projections
# ---------------------------------------------------------------------------


async def test_dispatch_decision_projections_one_send_per_scope():
    scope_outputs = [
        {"scope_id": "S1", "inv_id": "INV-001"},
        {"scope_id": "S2", "inv_id": "INV-002"},
        {"scope_id": "S3", "inv_id": "INV-003"},
    ]
    state = _base_state(scope_outputs=scope_outputs)
    sends = await dispatch_decision_projections(state)
    assert len(sends) == 3
    assert all(s.node == "project_scope_decisions" for s in sends)
    scope_ids = {s.arg["scope_id"] for s in sends}
    assert scope_ids == {"S1", "S2", "S3"}


async def test_dispatch_decision_projections_empty_scopes():
    state = _base_state(scope_outputs=[])
    result = await dispatch_decision_projections(state)
    assert result == "collect_decisions"


# ---------------------------------------------------------------------------
# GROUP 6 — project_scope_decisions (worker)
# ---------------------------------------------------------------------------


async def test_project_scope_decisions_calls_core():
    mock_dp_mod = MagicMock()
    mock_dp_mod.project_decisions = AsyncMock(return_value={
        "scope_id": "S1", "decisions": [{"decision_type": "accelerate"}]
    })

    with patch.dict(sys.modules, {"src.core.decision_projection": mock_dp_mod}):
        result = await project_scope_decisions({
            "scope_id": "S1",
            "scope_output": {"scope_id": "S1"},
            "decisions": None,
        })

    assert "scope_decisions" in result
    assert result["scope_decisions"][0]["scope_id"] == "S1"


async def test_project_scope_decisions_error_returns_stub():
    with patch("src.core.decision_projection.project_decisions", new=AsyncMock(side_effect=RuntimeError("LLM fail"))):
        result = await project_scope_decisions({
            "scope_id": "S1",
            "scope_output": {},
            "decisions": None,
        })

    assert result["scope_decisions"][0]["decisions"] == []
    assert "error" in result["scope_decisions"][0]


async def test_dispatch_decision_projections_skips_already_done():
    """Router skips scopes already in scope_decisions — state is the single source of truth."""
    scope_outputs = [
        {"scope_id": "S1", "inv_id": "INV-001"},
        {"scope_id": "S2", "inv_id": "INV-002"},
    ]
    state = _base_state(
        scope_outputs=scope_outputs,
        scope_decisions=[{"scope_id": "S1", "decisions": []}],
    )
    sends = await dispatch_decision_projections(state)
    assert len(sends) == 1
    assert sends[0].arg["scope_id"] == "S2"


async def test_dispatch_decision_projections_no_cache_dir_in_payload():
    scope_outputs = [{"scope_id": "S1", "inv_id": "INV-001"}]
    state = _base_state(scope_outputs=scope_outputs)
    sends = await dispatch_decision_projections(state)
    assert "cache_dir" not in sends[0].arg


# ---------------------------------------------------------------------------
# GROUP 6 — collect_decisions (reducer)
# ---------------------------------------------------------------------------


async def test_collect_decisions_attaches_to_scopes():
    scope_outputs = [
        {"scope_id": "S1", "decisions": []},
        {"scope_id": "S2", "decisions": []},
    ]
    state = _base_state(
        scope_outputs=scope_outputs,
        scope_decisions=[
            {"scope_id": "S1", "decisions": [{"decision_type": "accelerate"}, {"decision_type": "halt"}]},
            {"scope_id": "S2", "decisions": [{"decision_type": "monitor"}]},
        ],
    )
    result = await collect_decisions(state)
    by_scope = {s["scope_id"]: s for s in result["scope_outputs"]}
    assert len(by_scope["S1"]["decisions"]) == 2
    assert len(by_scope["S2"]["decisions"]) == 1


async def test_collect_decisions_handles_empty_scope_decisions():
    scope_outputs = [{"scope_id": "S1", "decisions": []}]
    state = _base_state(scope_outputs=scope_outputs, scope_decisions=[])
    result = await collect_decisions(state)
    assert result["scope_outputs"][0]["decisions"] == []


# ---------------------------------------------------------------------------
# Compiled graph
# ---------------------------------------------------------------------------


def test_causal_graph_compiles():
    assert causal_graph is not None


def test_causal_graph_has_all_nodes():
    # dispatch_link_investigations and dispatch_science_investigations are real
    # join nodes (not conditional-edge routers). _route_* functions do the routing.
    # gather_bow_context → replaced by dispatch_bow_enrichment fan-out (§3.1.5).
    # dispatch_bow_enrichment is a conditional-edge routing function, not a node —
    # same pattern as dispatch_timeline_narratives in the analyze subgraph.
    expected_nodes = {
        "evaluate_investment_rubric",
        "collect_evidence_packs",
        "enrich_bow_context_worker",
        "collect_bow_enrichment",
        "forecast_consequences",
        "dispatch_link_investigations",   # join node: collect_bow_enrichment → here
        "investigate_link",
        "collect_link_assessments",
        "synthesize_findings",
        "critique_synthesis",
        "identify_gaps",
        "dispatch_science_investigations",  # join node: parallel with synthesis chain
        "investigate_science_assumption",
        "collect_science_results",
        "necessity_check",
        "project_scope_decisions",
        "collect_decisions",
        "clear_fanout_accumulators",
    }
    graph_nodes = set(causal_graph.get_graph().nodes.keys()) - {"__start__", "__end__"}
    assert graph_nodes == expected_nodes
