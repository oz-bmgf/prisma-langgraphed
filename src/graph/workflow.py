"""Top-level NQPR workflow graph (ARCHITECTURE.md §2).

Entry:   START → load_collection
Exits:   deliver → END  (success)
         precheck → END (precheck failure)

The graph checkpoints after every node whenever a checkpointer is attached.
Human-in-the-loop interrupts are OFF by default (pipeline runs unattended).
Enable with human_interrupts=True or NQPR_HUMAN_INTERRUPTS=true.

Thread identity:
  thread_id = f"{program}::{run_name}"
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from langgraph.graph import END, START, StateGraph

from src.config import (
    CHECKPOINT_HUMAN_INTERRUPTS,
    DEFAULT_RESEARCH_MODEL,
    DEFAULT_SYNTHESIS_MODEL,
)
from src.graph.nodes.analyze import analyze
from src.graph.nodes.approve_report import approve_report
from src.graph.nodes.deliver import deliver
from src.graph.nodes.finalize import finalize
from src.graph.nodes.load_collection import load_collection
from src.graph.nodes.precheck import precheck
from src.graph.nodes.prepare_research import prepare_research
from src.graph.nodes.rerender import rerender
from src.graph.nodes.research import research
from src.graph.nodes.review_research_plan import review_research_plan
from src.graph.state import WorkflowState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Conditional routing functions (synchronous — no I/O)
# ---------------------------------------------------------------------------


def route_after_precheck(state: WorkflowState) -> str:
    return "analyze" if state.get("precheck_passed") else END


def route_after_research_plan_review(state: WorkflowState) -> str:
    return "research" if state.get("research_plan_approved") else "prepare_research"


def route_after_report_approval(state: WorkflowState) -> str:
    return "deliver" if state.get("report_approved") else "finalize"


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

_builder = StateGraph(WorkflowState)

# Nodes
_builder.add_node("load_collection", load_collection)
_builder.add_node("precheck", precheck)
_builder.add_node("analyze", analyze)
_builder.add_node("prepare_research", prepare_research)
_builder.add_node("review_research_plan", review_research_plan)
_builder.add_node("research", research)
_builder.add_node("finalize", finalize)
_builder.add_node("rerender", rerender)
_builder.add_node("approve_report", approve_report)
_builder.add_node("deliver", deliver)

# Edges — per ARCHITECTURE.md §2
_builder.add_edge(START, "load_collection")
_builder.add_edge("load_collection", "precheck")
_builder.add_conditional_edges(
    "precheck",
    route_after_precheck,
    {"analyze": "analyze", END: END},
)
_builder.add_edge("analyze", "prepare_research")
_builder.add_edge("prepare_research", "review_research_plan")
_builder.add_conditional_edges(
    "review_research_plan",
    route_after_research_plan_review,
    {"research": "research", "prepare_research": "prepare_research"},
)
_builder.add_edge("research", "finalize")
_builder.add_edge("finalize", "rerender")
_builder.add_edge("rerender", "approve_report")
_builder.add_conditional_edges(
    "approve_report",
    route_after_report_approval,
    {"deliver": "deliver", "finalize": "finalize"},
)
_builder.add_edge("deliver", END)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

# Nodes where human review can optionally pause execution.
# Only active when human_interrupts=True is passed to compile_graph().
_HUMAN_INTERRUPT_NODES = [
    "analyze",
    "review_research_plan",
    "research",
    "approve_report",
]


def compile_graph(checkpointer=None, *, human_interrupts: bool | None = None):
    """Compile and return the workflow graph.

    Args:
        checkpointer: a LangGraph checkpointer instance (from build_checkpointer()).
            When provided, state is persisted after every node. When None,
            the graph runs without persistence (unit tests only).
        human_interrupts: controls whether execution pauses at human review nodes.
            None  → reads NQPR_HUMAN_INTERRUPTS env var (default False = unattended).
            True  → pause at analyze, review_research_plan, research, approve_report.
            False → run straight through without pausing.

    The checkpointer saves state after EVERY node regardless of human_interrupts.
    Interrupts only control whether execution pauses waiting for human input.

    Returns:
        CompiledStateGraph ready for ainvoke / astream.
    """
    if human_interrupts is None:
        human_interrupts = CHECKPOINT_HUMAN_INTERRUPTS

    interrupt_before = _HUMAN_INTERRUPT_NODES if human_interrupts else []

    return _builder.compile(
        checkpointer=checkpointer,
        interrupt_before=interrupt_before,
    )


# Pre-compiled for langgraph dev / Studio (platform injects its own checkpointer).
workflow_graph = _builder.compile(interrupt_before=[])


def create_initial_state(
    program: str,
    run_name: str,
    collection_name: str,
    base_dir: str,
    ingested_dir: str,
    research_model: str = DEFAULT_RESEARCH_MODEL,
    synthesis_model: str = DEFAULT_SYNTHESIS_MODEL,
    focus: Optional[str] = None,
    focus_bows: Optional[list[str]] = None,
    aux_collections: Optional[list[str]] = None,
    **kwargs: Any,
) -> dict:
    """Build a valid initial WorkflowState dict.

    threads_dir is derived from base_dir + program + run_name as the canonical
    output directory for all run artifacts (reports, research/, causal_cache/).

    All Annotated reducer fields are initialised to [] (required by AGENTS.md §4).
    """
    threads_dir = str(Path(base_dir) / f"{program}-experiments" / f"run-{run_name}")
    return {
        # Run identity
        "program": program,
        "run_name": run_name,
        "collection_name": collection_name,
        "base_dir": base_dir,
        "ingested_dir": ingested_dir,
        "research_model": research_model,
        "synthesis_model": synthesis_model,
        "threads_dir": threads_dir,
        "focus": focus,
        "focus_bows": focus_bows,
        "aux_collections": aux_collections,
        # Fan-out reducer fields — must be [] not None (AGENTS.md §4)
        "evidence_packs": [],
        "link_assessments": [],
        "science_results": [],
        "scope_decisions": [],
        "research_results": [],
        "all_excerpts": [],
        "errors": [],
        "asta_traces": [],
        "slr_traces": [],
        "lbd_traces": [],
        "deep_web_traces": [],
        "edison_traces": [],
        "web_search_traces": [],
        "compute_traces": [],
        "collection_search_traces": [],
        "investigation_traces": [],
        **kwargs,
    }


def make_thread_id(program: str, run_name: str) -> str:
    """Return the canonical thread_id for a pipeline run."""
    return f"{program}::{run_name}"
