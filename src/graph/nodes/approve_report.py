"""approve_report node — human-in-the-loop interrupt before delivery."""
from __future__ import annotations

from langchain_core.runnables import RunnableConfig
from langgraph.types import interrupt

from src.graph.state import WorkflowState


async def approve_report(state: WorkflowState, config: RunnableConfig) -> dict:
    # Non-interactive path: already decided (e.g. mock input or unattended run)
    if state.get("report_approved") is not None:
        return {}

    resume_value = interrupt({  # NOT awaited — synchronous sentinel
        "stage": "approve_report",
        "program": state["program"],
        "final_report_wresearch_md_path": state.get("final_report_wresearch_md_path"),
        "question": (
            "Review the final report, then reply:\n"
            "  'approve' — deliver the report\n"
            "  'revise'  — return to finalize for another enrichment pass"
        ),
    })

    if resume_value == "approve":
        return {"report_approved": True}

    if resume_value == "revise":
        return {"report_approved": False}

    # unrecognised input → treat as approve
    return {"report_approved": True}
