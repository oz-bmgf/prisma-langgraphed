"""research node — delegates to the research dispatch subgraph (ARCHITECTURE.md §5)."""
from __future__ import annotations

from pathlib import Path

from langchain_core.runnables import RunnableConfig

from src.graph.state import WorkflowState


async def research(state: WorkflowState, config: RunnableConfig) -> dict:
    # lazy import — allows mocking research_graph in tests
    from src.graph.subgraphs.research import research_graph

    research_plan = state.get("research_plan") or []
    threads_dir = state.get("threads_dir") or ""
    research_dir = str(Path(threads_dir) / "research") if threads_dir else ""

    research_input = {
        "research_plan": research_plan,
        "research_dir": research_dir,
        # fan-out reducer fields — must be [] not None (AGENTS.md §4)
        "research_results": [],
        "slr_traces": [],
        "lbd_traces": [],
        "deep_web_traces": [],
        "edison_traces": [],
        "errors": [],
    }

    result = await research_graph.ainvoke(research_input, config)

    research_results = result.get("research_results") or []
    return {
        "research_dir": research_dir,
        "research_results": research_results,
        "research_ok_count": sum(1 for r in research_results if r.get("status") == "ok"),
        "slr_traces": result.get("slr_traces") or [],
        "lbd_traces": result.get("lbd_traces") or [],
        "deep_web_traces": result.get("deep_web_traces") or [],
        "edison_traces": result.get("edison_traces") or [],
        "errors": result.get("errors") or [],
    }
