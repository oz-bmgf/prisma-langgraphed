"""review_research_plan node — human-in-the-loop interrupt (AGENTS.md §5 Phase I)."""
from __future__ import annotations

from langchain_core.runnables import RunnableConfig
from langgraph.types import interrupt

from src.graph.state import WorkflowState


async def review_research_plan(state: WorkflowState, config: RunnableConfig) -> dict:
    # Non-interactive path: already decided (e.g. mock input or unattended run)
    if state.get("research_plan_approved") is not None:
        return {}

    resume_value = interrupt({  # NOT awaited — synchronous sentinel
        "stage": "review_research_plan",
        "program": state["program"],
        "research_plan_md_path": state.get("research_plan_md_path"),
        "task_count": len(state.get("research_plan") or []),
        "question": (
            "Review and edit research_plan.md, then reply:\n"
            "  'approve'               — dispatch all tasks\n"
            "  'regenerate'            — rebuild the plan\n"
            "  {'prune': [task_id...]} — remove specific tasks, then dispatch"
        ),
    })

    if resume_value == "approve":
        return {"research_plan_approved": True}

    if resume_value == "regenerate":
        return {"research_plan_approved": False}

    if isinstance(resume_value, dict) and "prune" in resume_value:
        prune_ids = set(resume_value["prune"])
        pruned = [t for t in (state.get("research_plan") or []) if t.get("id") not in prune_ids]
        return {
            "research_plan": pruned,
            "research_plan_approved": True,
        }

    # unrecognised input → treat as approve
    return {"research_plan_approved": True}
