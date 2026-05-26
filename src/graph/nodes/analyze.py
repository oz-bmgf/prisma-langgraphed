"""analyze node — delegates to the analyze subgraph (ARCHITECTURE.md §3)."""
from __future__ import annotations

from pathlib import Path

from langchain_core.runnables import RunnableConfig

from src.graph.state import WorkflowState


async def analyze(state: WorkflowState, config: RunnableConfig) -> dict:
    # lazy import — allows mocking analyze_graph in tests
    from src.graph.subgraphs.analyze import analyze_graph

    assert state["doc_list"] is not None, "load_collection must run before analyze"
    assert state["investment_scoring"] is not None, "load_collection must run before analyze"
    assert state["bow_investment_map"] is not None, "load_collection must run before analyze"
    assert state["investment_intelligence"] is not None, "load_collection must run before analyze"
    assert state["chunks_json_path"] is not None, "load_collection must run before analyze"
    assert state["pages_dir"] is not None, "load_collection must run before analyze"

    # Derive threads_dir if absent — LangGraph Studio / direct ainvoke callers may omit it,
    # but assemble_report needs it to write final_report.md and final_report_md_path to state.
    threads_dir: str | None = state.get("threads_dir")
    if not threads_dir:
        base_dir = state.get("base_dir", "")
        program = state.get("program", "")
        run_name = state.get("run_name", "")
        if base_dir and program and run_name:
            threads_dir = str(Path(base_dir) / f"{program}-experiments" / f"run-{run_name}")

    analyze_input = {
        "program": state["program"],
        "collection_name": state["collection_name"],
        "base_dir": state["base_dir"],
        "ingested_dir": state["ingested_dir"],
        "doc_list": state["doc_list"],
        "investment_scoring": state["investment_scoring"],
        "bow_investment_map": state["bow_investment_map"],
        "investment_intelligence": state["investment_intelligence"],
        "chunks_json_path": state["chunks_json_path"],
        "pages_dir": state["pages_dir"],
        "focus": state.get("focus"),
        "focus_bows": state.get("focus_bows"),
        "aux_collections": state.get("aux_collections"),
        "threads_dir": threads_dir,
        "research_model": state["research_model"],
        "synthesis_model": state["synthesis_model"],
        # fan-out reducer fields — must be [] not None
        "evidence_packs": [],
        "link_assessments": [],
        "science_results": [],
        "scope_decisions": [],
        "timeline_narrative_results": [],
        "all_excerpts": [],
        # trace reducer fields — must be [] not None (AGENTS.md §4)
        "asta_traces": [],
        "slr_traces": [],
        "lbd_traces": [],
        "deep_web_traces": [],
        "edison_traces": [],
        "web_search_traces": [],
        "compute_traces": [],
        "collection_search_traces": [],
        "investigation_traces": [],
        "errors": [],
        # optional carry-through fields for resume
        "program_context": state.get("program_context"),
        "scopes": state.get("scopes"),
        "scope_timelines": state.get("scope_timelines"),
        "cross_cutting_analysis": state.get("cross_cutting_analysis"),
        "scope_outputs": state.get("scope_outputs"),
        "analyst_report": state.get("analyst_report"),
        "final_report_md": state.get("final_report_md"),
        "bibliography": state.get("bibliography"),
        "run_meta": state.get("run_meta"),
        "coverage_pct": state.get("coverage_pct"),
        "grade": state.get("grade"),
        "confidence_map": state.get("confidence_map"),
        "allocation_verification": state.get("allocation_verification"),
        "numerical_verification": state.get("numerical_verification"),
        "numerical_provenance": state.get("numerical_provenance"),
    }

    result = await analyze_graph.ainvoke(analyze_input, config)

    return {
        "threads_dir": threads_dir,
        "final_report_md_path": result.get("final_report_md_path"),
        "final_report_md": result.get("final_report_md"),
        "analyst_report": result.get("analyst_report"),
        "scope_outputs": result.get("scope_outputs"),
        "program_context": result.get("program_context"),
        "scopes": result.get("scopes"),
        "scope_timelines": result.get("scope_timelines"),
        "cross_cutting_analysis": result.get("cross_cutting_analysis"),
        "bibliography": result.get("bibliography"),
        # Clear fan-out accumulators — data is embedded in scope_outputs after analysis.
        # _take_update reducer in WorkflowState allows overwriting to [] here.
        "evidence_packs": [],
        "link_assessments": [],
        "science_results": [],
        "scope_decisions": [],
        "all_excerpts": result.get("all_excerpts") or [],
        "excerpts_csv_path": result.get("excerpts_csv_path"),
        "allocation_verification": result.get("allocation_verification"),
        "numerical_verification": result.get("numerical_verification"),
        "numerical_provenance": result.get("numerical_provenance"),
        "run_meta": result.get("run_meta"),
        "coverage_pct": result.get("coverage_pct"),
        "grade": result.get("grade"),
        "confidence_map": result.get("confidence_map"),
        "asta_traces": result.get("asta_traces") or [],
        "slr_traces": result.get("slr_traces") or [],
        "lbd_traces": result.get("lbd_traces") or [],
        "deep_web_traces": result.get("deep_web_traces") or [],
        "edison_traces": result.get("edison_traces") or [],
        "web_search_traces": result.get("web_search_traces") or [],
        "compute_traces": result.get("compute_traces") or [],
        "collection_search_traces": result.get("collection_search_traces") or [],
        "investigation_traces": result.get("investigation_traces") or [],
        "errors": result.get("errors") or [],
    }
