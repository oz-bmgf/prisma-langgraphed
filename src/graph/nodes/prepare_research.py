"""prepare_research node — builds research plan from analyst findings."""
from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Optional

from langchain_core.runnables import RunnableConfig

from src.core.llm_utils import acall_llm
from src.graph.state import WorkflowState
from src.prompts.research_prompts import (
    DE_NOVO_SYSTEM as _DE_NOVO_SYSTEM,
    DE_NOVO_TEMPLATE as _DE_NOVO_PROMPT,
    REVIEW_SYSTEM as _REVIEW_SYSTEM,
    REVIEW_TEMPLATE as _REVIEW_PROMPT,
)

VALID_TYPES = {"slr", "lbd", "deep_web", "internal", "edison"}
VALID_PRIORITIES = {"critical", "important", "nice_to_have"}

_TYPE_LABELS = {
    "slr": "Systematic Literature Review (SLR)",
    "lbd": "Literature-Based Discovery (LBD)",
    "deep_web": "Deep Web / Grey Literature",
    "edison": "Edison Scientific Literature Review",
    "internal": "Internal (Program Team)",
}
_PRIO_BADGE = {"critical": "CRITICAL", "important": "IMPORTANT", "nice_to_have": "NICE"}


# ---------------------------------------------------------------------------
# Helpers (synchronous — called via asyncio.to_thread)
# ---------------------------------------------------------------------------


def _extract_from_analyst_report(analyst_report: dict) -> list[dict]:
    candidates = []
    for thread in analyst_report.get("threads", []):
        tid = thread.get("id", "")
        for dr in thread.get("deep_research_needed", []):
            candidates.append({
                "query": dr.get("question", ""),
                "type": "deep_web",
                "rationale": dr.get("why", ""),
                "what_it_would_change": dr.get("expected_impact", ""),
                "priority": "important",
                "source": "analyst",
                "linked_scope": tid,
            })
        for gap in thread.get("gaps", []):
            if isinstance(gap, str) and gap.strip():
                candidates.append({
                    "query": gap.strip(),
                    "type": "deep_web",
                    "rationale": f"Unresolved gap from thread {tid}",
                    "what_it_would_change": "",
                    "priority": "nice_to_have",
                    "source": "analyst",
                    "linked_scope": tid,
                })
    return [c for c in candidates if c.get("query", "").strip()]


def _parse_json_from_text(text: str) -> object:
    text = text.strip()
    # strip markdown code fences
    m = re.search(r"```(?:json)?\s*([\s\S]+?)```", text)
    if m:
        text = m.group(1).strip()
    return json.loads(text)


def _write_plan_md(items: list[dict], path: Path) -> None:
    lines = [f"# Research Plan\n\nTotal questions: {len(items)}\n"]
    by_type: dict[str, list[dict]] = {}
    for c in items:
        by_type.setdefault(c.get("type", "deep_web"), []).append(c)
    for rt in ("slr", "lbd", "deep_web", "edison", "internal"):
        group = by_type.get(rt, [])
        if not group:
            continue
        lines.append(f"## {_TYPE_LABELS.get(rt, rt)} ({len(group)})\n")
        for c in group:
            badge = _PRIO_BADGE.get(c.get("priority", ""), "")
            lines.append(f"### {c['id']} [{badge}]\n")
            lines.append(f"**Query:** {c['query']}\n")
            if c.get("rationale"):
                lines.append(f"**Rationale:** {c['rationale']}\n")
            if c.get("what_it_would_change"):
                lines.append(f"**Impact:** {c['what_it_would_change']}\n")
            lines.append("---\n")
    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------


async def prepare_research(state: WorkflowState, config: RunnableConfig) -> dict:
    output_dir = state.get("threads_dir") or state.get("output_dir") or ""
    out_path = Path(output_dir) if output_dir else None

    research_model: str = state["research_model"]

    # Collect seed candidates from analyst report
    analyst_report: Optional[dict] = state.get("analyst_report") or {}
    candidates: list[dict] = _extract_from_analyst_report(analyst_report)

    # Generate de novo questions from the report markdown
    report_md: str = state.get("final_report_md") or ""
    if report_md:
        prompt = _DE_NOVO_PROMPT.format(report_text=report_md[:150_000])
        raw = await acall_llm(prompt, _DE_NOVO_SYSTEM, model=research_model)
        try:
            parsed = _parse_json_from_text(raw)
            if isinstance(parsed, list):
                items = parsed
            elif isinstance(parsed, dict):
                items = parsed.get("items", [parsed] if "query" in parsed else [])
            else:
                items = []
            for it in items:
                it.setdefault("source", "de_novo")
                if it.get("type") not in VALID_TYPES:
                    it["type"] = "deep_web"
            candidates.extend(items)
        except (json.JSONDecodeError, ValueError):
            pass

    # Review + validate routing via LLM
    if candidates:
        for i, c in enumerate(candidates):
            c.setdefault("id", f"RQ-{i:03d}")
        review_input = [
            {"id": c["id"], "query": c["query"], "type": c.get("type", "deep_web"),
             "priority": c.get("priority", "important")}
            for c in candidates
        ]
        try:
            raw_review = await acall_llm(
                _REVIEW_PROMPT.format(questions_json=json.dumps(review_input, indent=2)),
                _REVIEW_SYSTEM,
                model=research_model,
            )
            parsed_review = _parse_json_from_text(raw_review)
            if isinstance(parsed_review, list):
                reviews = parsed_review
            elif isinstance(parsed_review, dict):
                reviews = parsed_review.get("items", [])
            else:
                reviews = []
            by_id = {r["id"]: r for r in reviews if isinstance(r, dict) and "id" in r}
            kept = []
            for c in candidates:
                rev = by_id.get(c.get("id", ""), {})
                if rev.get("status") == "drop":
                    continue
                if rev.get("status") == "fix":
                    if rev.get("corrected_type") in VALID_TYPES:
                        c["type"] = rev["corrected_type"]
                    if rev.get("corrected_query"):
                        c["query"] = rev["corrected_query"]
                kept.append(c)
            candidates = kept
        except (json.JSONDecodeError, ValueError):
            pass

    # Sort and assign final IDs
    type_ord = {"slr": 0, "lbd": 1, "deep_web": 2, "edison": 3, "internal": 4}
    prio_ord = {"critical": 0, "important": 1, "nice_to_have": 2}
    candidates.sort(key=lambda c: (
        prio_ord.get(c.get("priority", ""), 3),
        type_ord.get(c.get("type", ""), 5),
    ))
    output_keys = ("id", "query", "type", "rationale", "what_it_would_change",
                   "priority", "source", "linked_scope")
    plan: list[dict] = []
    for i, c in enumerate(candidates):
        item = {k: c.get(k, "") for k in output_keys}
        item["id"] = f"RQ-{i + 1:03d}"
        plan.append(item)

    # Write human-readable MD deliverable if output_dir is set
    md_path_str: Optional[str] = None
    if out_path:
        # asyncio-APPROVED-1: to_thread wraps blocking mkdir
        await asyncio.to_thread(out_path.mkdir, parents=True, exist_ok=True)
        md_path = out_path / "research_plan.md"
        # asyncio-APPROVED-1: to_thread wraps blocking file write
        await asyncio.to_thread(_write_plan_md, plan, md_path)
        md_path_str = str(md_path)

    return {
        "research_plan": plan,
        "research_plan_md_path": md_path_str,
    }
