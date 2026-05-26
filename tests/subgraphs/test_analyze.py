"""Unit tests for src/graph/subgraphs/analyze.py."""
from __future__ import annotations

import sys
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from src.graph.subgraphs.analyze import (
    analyze_graph,
    assemble_report,
    build_investment_report_worker,
    collect_investment_reports,
    collect_scope_sections,
    collect_timeline_narratives,
    compute_scopes,
    cross_cutting_analysis,
    dispatch_investment_reports,
    dispatch_scope_sections,
    dispatch_timeline_narratives,
    generate_scope_narrative,
    load_catalog,
    orientation,
    run_causal_pipeline,
    synthesize_scope_section_worker,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _base_state(**overrides: Any) -> dict:
    state: dict = {
        "program": "Malaria",
        "collection_name": "malaria",
        "base_dir": "/tmp/base",
        "ingested_dir": "/tmp/ingested",
        "doc_list": [],
        "investment_scoring": {},
        "bow_investment_map": {},
        "investment_intelligence": {},
        "chunks_json_path": "",
        "pages_dir": "",
        "focus": None,
        "focus_bows": None,
        "aux_collections": None,
        "threads_dir": None,
        "research_model": "claude-sonnet-4-6",
        "synthesis_model": "claude-sonnet-4-6",
        "program_context": None,
        "scopes": None,
        "scope_timelines": None,
        "scope_outputs": None,
        "analyst_report": None,
        "final_report_md": None,
        "excerpts_csv_path": None,
        "numerical_provenance": None,
        "verification_sources": None,
        "allocation_verification_path": None,
        "numerical_verification_path": None,
        "run_meta": None,
        "evidence_packs": [],
        "link_assessments": [],
        "science_results": [],
        "scope_decisions": [],
        "errors": [],
    }
    state.update(overrides)
    return state


def _investment_scoring() -> dict:
    return {
        "INV-001": {"title": "Malaria Vaccine Trial", "start": "2023-01", "end": "2025-12",
                    "status": "active", "approved_amount": 10_000_000, "paid_amount": 6_000_000},
        "INV-002": {"title": "RTS,S Deployment", "start": "2022-06", "end": "2026-06",
                    "status": "active", "approved_amount": 5_000_000, "paid_amount": 2_500_000},
    }


def _bow_investment_map() -> dict:
    return {
        "BOW-A": ["INV-001"],
        "BOW-B": ["INV-001", "INV-002"],
    }


# ---------------------------------------------------------------------------
# orientation
# ---------------------------------------------------------------------------


async def test_orientation_calls_acall_llm():
    state = _base_state(
        investment_scoring=_investment_scoring(),
        bow_investment_map=_bow_investment_map(),
        doc_list=[{}, {}],
    )
    mock_acall = AsyncMock(return_value="Portfolio covers 2 investments in malaria vaccines.")
    with patch("src.graph.subgraphs.analyze.acall_llm", mock_acall):
        result = await orientation(state)

    mock_acall.assert_called_once()
    # orientation() now returns program_context dict (Fix 2d); free-text fallback wraps in struct
    assert "program_context" in result
    assert isinstance(result["program_context"], dict)


async def test_orientation_includes_focus_in_prompt():
    state = _base_state(focus="Vaccine delivery scale-up")
    mock_acall = AsyncMock(return_value="summary with focus")
    with patch("src.graph.subgraphs.analyze.acall_llm", mock_acall):
        await orientation(state)

    call_args = mock_acall.call_args
    prompt = call_args[0][0]  # first positional arg is the prompt string
    assert "Vaccine delivery scale-up" in prompt


async def test_orientation_error_returns_empty_summary():
    state = _base_state()
    mock_acall = AsyncMock(side_effect=RuntimeError("LLM fail"))
    with patch("src.graph.subgraphs.analyze.acall_llm", mock_acall):
        result = await orientation(state)

    assert result.get("program_context") == {}
    assert len(result.get("errors", [])) > 0


# ---------------------------------------------------------------------------
# compute_scopes
# ---------------------------------------------------------------------------


async def test_compute_scopes_one_per_investment():
    state = _base_state(
        investment_scoring=_investment_scoring(),
        bow_investment_map=_bow_investment_map(),
    )
    result = await compute_scopes(state)
    # INV-001 appears in BOW-A and BOW-B; INV-002 in BOW-B
    # New BOW-grouping: each BOW becomes its own scope; scope["inv_id"] is primary
    # INV-002 is in BOW-B scope's inv_ids (secondary); INV-001 is primary in BOW-A and BOW-B
    scopes = result["scopes"]
    assert len(scopes) > 0
    all_primary_inv_ids = {s["inv_id"] for s in scopes}
    all_member_inv_ids = {iid for s in scopes for iid in s.get("inv_ids", [s["inv_id"]])}
    assert "INV-001" in all_primary_inv_ids
    assert "INV-002" in all_member_inv_ids
    assert all("scope_id" in s for s in scopes)
    assert all("bow_ids" in s for s in scopes)


async def test_compute_scopes_respects_focus_bows():
    state = _base_state(
        investment_scoring=_investment_scoring(),
        bow_investment_map=_bow_investment_map(),
        focus_bows=["BOW-A"],  # only BOW-A → only INV-001
    )
    result = await compute_scopes(state)
    inv_ids = {s["inv_id"] for s in result["scopes"]}
    assert "INV-001" in inv_ids
    assert "INV-002" not in inv_ids


async def test_compute_scopes_empty_map():
    state = _base_state(investment_scoring={}, bow_investment_map={})
    result = await compute_scopes(state)
    assert result["scopes"] == []


async def test_compute_scopes_skips_investments_with_no_bows():
    state = _base_state(
        investment_scoring={"INV-ORPHAN": {"title": "No BOW"}},
        bow_investment_map={"BOW-A": ["INV-001"]},
    )
    result = await compute_scopes(state)
    inv_ids = {s["inv_id"] for s in result["scopes"]}
    assert "INV-ORPHAN" not in inv_ids


# ---------------------------------------------------------------------------
# dispatch_timeline_narratives / generate_scope_narrative / collect_timeline_narratives
# ---------------------------------------------------------------------------


async def test_dispatch_timeline_narratives_returns_sends():
    from langgraph.types import Send
    scopes = [
        {"scope_id": "scope_0000", "inv_id": "INV-001", "bow_ids": ["BOW-A"]},
        {"scope_id": "scope_0001", "inv_id": "INV-002", "bow_ids": ["BOW-B"]},
    ]
    state = _base_state(scopes=scopes, investment_scoring=_investment_scoring())
    sends = await dispatch_timeline_narratives(state)
    assert len(sends) == 2
    assert all(isinstance(s, Send) for s in sends)
    assert all(s.node == "generate_scope_narrative" for s in sends)
    scope_ids = {s.arg["scope_id"] for s in sends}
    assert "scope_0000" in scope_ids
    assert "scope_0001" in scope_ids


async def test_dispatch_timeline_narratives_empty_scopes():
    state = _base_state(scopes=[])
    result = await dispatch_timeline_narratives(state)
    assert result == "collect_timeline_narratives"


async def test_generate_scope_narrative_calls_llm():
    from src.graph.state import ScopeNarrativeState
    node_state: ScopeNarrativeState = {
        "scope_id": "scope_0000",
        "inv_id": "INV-001",
        "timeline": {"scope_id": "scope_0000", "inv_id": "INV-001",
                     "start": "2023-01", "end": "2025-12", "status": "active",
                     "approved_amount": 10_000_000, "paid_amount": 6_000_000},
        "model": "claude-sonnet-4-6",
        "result": None,
    }
    mock_acall = AsyncMock(return_value="Timeline narrative.")
    with patch("src.graph.subgraphs.analyze.acall_llm", mock_acall):
        result = await generate_scope_narrative(node_state)

    mock_acall.assert_called_once()
    assert result["timeline_narrative_results"][0]["narrative"] == "Timeline narrative."
    assert result["timeline_narrative_results"][0]["inv_id"] == "INV-001"


async def test_collect_timeline_narratives_builds_dict():
    state = _base_state(timeline_narrative_results=[
        {"scope_id": "scope_0000", "inv_id": "INV-001", "narrative": "N1"},
        {"scope_id": "scope_0001", "inv_id": "INV-002", "narrative": "N2"},
    ])
    result = await collect_timeline_narratives(state)
    timelines = result["scope_timelines"]
    assert "scope_0000" in timelines
    assert "scope_0001" in timelines
    assert timelines["scope_0000"]["inv_id"] == "INV-001"


async def test_collect_timeline_narratives_empty():
    state = _base_state(timeline_narrative_results=[])
    result = await collect_timeline_narratives(state)
    assert result["scope_timelines"] == {}


# ---------------------------------------------------------------------------
# run_causal_pipeline
# ---------------------------------------------------------------------------


def _causal_output() -> dict:
    return {
        "scope_outputs": [{"scope_id": "scope_0000", "inv_id": "INV-001"}],
        "evidence_packs": [{"inv_id": "INV-001", "scope_id": "scope_0000"}],
        "link_assessments": [{"link_id": "L1", "scope_id": "scope_0000"}],
        "science_results": [],
        "scope_decisions": [],
        "errors": [],
    }


async def test_run_causal_pipeline_maps_input_fields():
    mock_causal_mod = MagicMock()
    mock_graph = MagicMock()
    mock_graph.ainvoke = AsyncMock(return_value=_causal_output())
    mock_causal_mod.causal_graph = mock_graph

    state = _base_state(
        scopes=[{"scope_id": "scope_0000", "inv_id": "INV-001"}],
        scope_timelines={"scope_0000": {"start": "2023-01"}},
        threads_dir="/tmp/threads",
    )

    with patch.dict(sys.modules, {"src.graph.subgraphs.causal": mock_causal_mod}):
        await run_causal_pipeline(state)

    call_args = mock_graph.ainvoke.call_args
    causal_input = call_args[0][0]

    assert causal_input["scopes"] == [{"scope_id": "scope_0000", "inv_id": "INV-001"}]
    assert causal_input["scope_timelines"] == {"scope_0000": {"start": "2023-01"}}
    assert causal_input["research_model"] == "claude-sonnet-4-6"
    assert causal_input["synthesis_model"] == "claude-sonnet-4-6"
    assert "cache_dir" not in causal_input
    assert causal_input["evidence_packs"] == []
    assert causal_input["link_assessments"] == []
    assert causal_input["science_results"] == []
    assert causal_input["scope_decisions"] == []


async def test_run_causal_pipeline_maps_output_fields():
    mock_causal_mod = MagicMock()
    mock_graph = MagicMock()
    mock_graph.ainvoke = AsyncMock(return_value=_causal_output())
    mock_causal_mod.causal_graph = mock_graph

    state = _base_state(
        scopes=[{"scope_id": "scope_0000"}],
        scope_timelines={},
    )

    with patch.dict(sys.modules, {"src.graph.subgraphs.causal": mock_causal_mod}):
        result = await run_causal_pipeline(state)

    assert result["scope_outputs"] == [{"scope_id": "scope_0000", "inv_id": "INV-001"}]
    # Fix 2c: evidence_packs / link_assessments / science_results / scope_decisions are NOT
    # forwarded to AnalyzeState — their data is embedded in scope_outputs by collect_* nodes.
    assert "evidence_packs" not in result
    assert "link_assessments" not in result
    assert "science_results" not in result
    assert "scope_decisions" not in result


async def test_run_causal_pipeline_no_cache_dir_in_input():
    """cache_dir must never appear in causal_input — state is the single source of truth."""
    mock_causal_mod = MagicMock()
    mock_graph = MagicMock()
    mock_graph.ainvoke = AsyncMock(return_value=_causal_output())
    mock_causal_mod.causal_graph = mock_graph

    for threads_dir in ("/var/qpr/threads", None):
        state = _base_state(threads_dir=threads_dir)
        with patch.dict(sys.modules, {"src.graph.subgraphs.causal": mock_causal_mod}):
            await run_causal_pipeline(state)
        causal_input = mock_graph.ainvoke.call_args[0][0]
        assert "cache_dir" not in causal_input, f"cache_dir must not be in causal_input (threads_dir={threads_dir!r})"


async def test_run_causal_pipeline_error_returns_error_list():
    mock_causal_mod = MagicMock()
    mock_graph = MagicMock()
    mock_graph.ainvoke = AsyncMock(side_effect=RuntimeError("subgraph crash"))
    mock_causal_mod.causal_graph = mock_graph

    state = _base_state()
    with patch.dict(sys.modules, {"src.graph.subgraphs.causal": mock_causal_mod}):
        result = await run_causal_pipeline(state)

    assert len(result.get("errors", [])) > 0
    assert "run_causal_pipeline" in result["errors"][0]


# ---------------------------------------------------------------------------
# cross_cutting_analysis
# ---------------------------------------------------------------------------


async def test_cross_cutting_analysis_calls_acall_llm():
    scope_outputs = [
        {"scope_id": "S0", "synthesis": "Vaccine trial on track."},
        {"scope_id": "S1", "synthesis": "Delivery pipeline delayed."},
    ]
    state = _base_state(scope_outputs=scope_outputs)
    mock_acall = AsyncMock(return_value="Cross-cutting: supply chain risk across both scopes.")
    with patch("src.graph.subgraphs.analyze.acall_llm", mock_acall):
        result = await cross_cutting_analysis(state)

    mock_acall.assert_called_once()
    # Fix 2e: cross_cutting_analysis no longer returns clusters; returns cross_cutting_analysis dict
    assert "cross_cutting_analysis" in result
    assert "clusters" not in result


async def test_cross_cutting_analysis_empty_scope_outputs():
    state = _base_state(scope_outputs=[])
    mock_acall = AsyncMock(return_value="should not be called")
    with patch("src.graph.subgraphs.analyze.acall_llm", mock_acall):
        result = await cross_cutting_analysis(state)

    mock_acall.assert_not_called()
    assert "clusters" not in result
    assert result.get("cross_cutting_analysis") == {}


# ---------------------------------------------------------------------------
# assemble_report
# ---------------------------------------------------------------------------


async def test_assemble_report_calls_core():
    mock_ra = MagicMock()
    mock_ra.assemble_report = AsyncMock(return_value={
        "markdown": "# Final Report\nContent here.",
        "body": "Content here.",
    })

    state = _base_state(scope_outputs=[{"scope_id": "S0"}])

    with patch.dict(sys.modules, {"src.core.report_assembler": mock_ra}):
        result = await assemble_report(state)

    mock_ra.assemble_report.assert_called_once()
    assert result["final_report_md"] == "# Final Report\nContent here."
    assert "analyst_report" in result


async def test_assemble_report_writes_file(tmp_path):
    mock_ra = MagicMock()
    mock_ra.assemble_report = AsyncMock(return_value={"markdown": "# Report"})

    state = _base_state(threads_dir=str(tmp_path))

    with patch.dict(sys.modules, {"src.core.report_assembler": mock_ra}):
        await assemble_report(state)

    report_path = tmp_path / "final_report.md"
    assert report_path.exists()
    assert report_path.read_text() == "# Report"


async def test_assemble_report_error_returns_errors():
    mock_ra = MagicMock()
    mock_ra.assemble_report = AsyncMock(side_effect=RuntimeError("assembly failed"))

    state = _base_state()
    with patch.dict(sys.modules, {"src.core.report_assembler": mock_ra}):
        result = await assemble_report(state)

    assert len(result.get("errors", [])) > 0
    assert "assemble_report" in result["errors"][0]


# ---------------------------------------------------------------------------
# dispatch_investment_reports / build_investment_report_worker / collect_investment_reports
# ---------------------------------------------------------------------------


def _scope_output(scope_id: str = "S1", inv_id: str = "INV-001") -> dict:
    return {
        "scope_id": scope_id,
        "inv_id": inv_id,
        "bow_ids": ["BOW-A"],
        "evidence_packs": [],
        "link_assessments": [],
        "causal_model": None,
    }


async def test_dispatch_investment_reports_sends_one_per_scope():
    scope_outputs = [_scope_output("S1"), _scope_output("S2", "INV-002")]
    state = _base_state(scope_outputs=scope_outputs, investment_scoring=_investment_scoring())
    sends = await dispatch_investment_reports(state)
    assert isinstance(sends, list)
    assert len(sends) == 2
    assert all(s.node == "build_investment_report_worker" for s in sends)
    assert {s.arg["scope_id"] for s in sends} == {"S1", "S2"}


async def test_dispatch_investment_reports_skips_already_done():
    scope_outputs = [
        {**_scope_output("S1"), "investment_report": {"verdict": "ok"}},
        _scope_output("S2", "INV-002"),
    ]
    state = _base_state(scope_outputs=scope_outputs, investment_scoring=_investment_scoring())
    sends = await dispatch_investment_reports(state)
    assert len(sends) == 1
    assert sends[0].arg["scope_id"] == "S2"


async def test_dispatch_investment_reports_empty_scope_outputs():
    state = _base_state(scope_outputs=[])
    result = await dispatch_investment_reports(state)
    assert result == "collect_investment_reports"


async def test_build_investment_report_worker_empty_link_assessments_returns_scope():
    # With no link_assessments, worker returns early without investment_report
    scope = _scope_output("S1")
    worker_state = {
        "scope_id": "S1",
        "scope": scope,
        "investment_scoring": _investment_scoring(),
        "model": "claude-sonnet-4-6",
        "result": None,
    }
    result = await build_investment_report_worker(worker_state)
    assert "scope_outputs" in result
    assert result["scope_outputs"][0]["scope_id"] == "S1"


async def test_build_investment_report_worker_with_link_assessments_adds_investment_report():
    scope = {
        **_scope_output("S1"),
        "link_assessments": [
            {"link_id": "L1", "status": "on_track", "confidence": "high", "gap_description": ""},
        ],
    }
    worker_state = {
        "scope_id": "S1",
        "scope": scope,
        "investment_scoring": _investment_scoring(),
        "model": "claude-sonnet-4-6",
        "result": None,
    }
    llm_response = '{"overall_status": "on_track", "severity": "acceptable", "ai_execution_verdict": "Good", "ai_impact_verdict": "Strong", "key_risks": [], "key_strengths": [], "executive_summary": "On track."}'
    with patch("src.graph.subgraphs.analyze.acall_llm", new=AsyncMock(return_value=llm_response)):
        result = await build_investment_report_worker(worker_state)
    assert "scope_outputs" in result
    updated = result["scope_outputs"][0]
    assert "investment_report" in updated
    assert updated["investment_report"]["divergence_severity"] is not None


async def test_build_investment_report_worker_error_still_returns_scope():
    scope = {
        **_scope_output("S1"),
        "link_assessments": [{"link_id": "L1", "status": "on_track", "confidence": "high", "gap_description": ""}],
    }
    worker_state = {"scope_id": "S1", "scope": scope, "investment_scoring": {}, "model": "", "result": None}
    with patch("src.graph.subgraphs.analyze.acall_llm", new=AsyncMock(side_effect=RuntimeError("fail"))):
        result = await build_investment_report_worker(worker_state)
    assert "scope_outputs" in result
    assert result["scope_outputs"][0]["scope_id"] == "S1"


async def test_collect_investment_reports_is_trivial_join():
    state = _base_state(scope_outputs=[_scope_output()])
    result = await collect_investment_reports(state)
    assert result == {}


# ---------------------------------------------------------------------------
# dispatch_scope_sections / synthesize_scope_section_worker / collect_scope_sections
# ---------------------------------------------------------------------------


async def test_dispatch_scope_sections_sends_one_per_scope():
    scope_outputs = [_scope_output("S1"), _scope_output("S2", "INV-002")]
    state = _base_state(scope_outputs=scope_outputs)
    sends = await dispatch_scope_sections(state)
    assert isinstance(sends, list)
    assert len(sends) == 2
    assert all(s.node == "synthesize_scope_section_worker" for s in sends)
    assert {s.arg["scope_id"] for s in sends} == {"S1", "S2"}


async def test_dispatch_scope_sections_skips_already_drafted():
    scope_outputs = [
        {**_scope_output("S1"), "section_draft": {"narrative": "done"}},
        _scope_output("S2", "INV-002"),
    ]
    state = _base_state(scope_outputs=scope_outputs)
    sends = await dispatch_scope_sections(state)
    assert len(sends) == 1
    assert sends[0].arg["scope_id"] == "S2"


async def test_dispatch_scope_sections_empty_scope_outputs():
    state = _base_state(scope_outputs=[])
    result = await dispatch_scope_sections(state)
    assert result == "collect_scope_sections"


async def test_synthesize_scope_section_worker_returns_scope_with_section_draft():
    scope = _scope_output("S1")
    worker_state = {"scope_id": "S1", "scope": scope, "model": "claude-sonnet-4-6", "result": None}
    with patch("src.graph.subgraphs.analyze.acall_llm", new=AsyncMock(return_value="## Section\nDetailed narrative.")):
        result = await synthesize_scope_section_worker(worker_state)

    assert "scope_outputs" in result
    updated = result["scope_outputs"][0]
    assert updated["scope_id"] == "S1"
    assert "section_draft" in updated
    assert updated["section_draft"]["scope_id"] == "S1"


async def test_synthesize_scope_section_worker_llm_error_falls_back_to_executive_summary():
    scope = {**_scope_output("S1"), "investment_report": {"executive_summary": "Fallback text."}}
    worker_state = {"scope_id": "S1", "scope": scope, "model": "", "result": None}
    with patch("src.graph.subgraphs.analyze.acall_llm", new=AsyncMock(side_effect=RuntimeError("fail"))):
        result = await synthesize_scope_section_worker(worker_state)
    assert "scope_outputs" in result
    updated = result["scope_outputs"][0]
    assert "section_draft" in updated
    assert updated["section_draft"]["summary"] == "Fallback text."


async def test_collect_scope_sections_is_trivial_join():
    state = _base_state(scope_outputs=[_scope_output()])
    result = await collect_scope_sections(state)
    assert result == {}


# ---------------------------------------------------------------------------
# Compiled graph
# ---------------------------------------------------------------------------


def test_analyze_graph_compiles():
    assert analyze_graph is not None


def test_analyze_graph_has_all_nodes():
    # dispatch_investment_reports and dispatch_scope_sections are conditional-edge routing
    # functions, not nodes — same pattern as dispatch_timeline_narratives.
    expected = {
        "load_catalog",
        "orientation",
        "compute_scopes",
        "build_timelines",
        "generate_scope_narrative",
        "collect_timeline_narratives",
        "run_causal_pipeline",
        "build_investment_report_worker",
        "collect_investment_reports",
        "synthesize_scope_section_worker",
        "collect_scope_sections",
        "cross_cutting_analysis",
        "quality_assessment",
        "assemble_report",
        "verify_report",
    }
    graph_nodes = set(analyze_graph.get_graph().nodes.keys()) - {"__start__", "__end__"}
    assert graph_nodes == expected
