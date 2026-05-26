"""finalize node — injects research results and rewrites the report."""
from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Optional

from langchain_core.runnables import RunnableConfig

from src.core.llm_utils import acall_llm
from src.graph.state import WorkflowState
from src.prompts.report_prompts import (
    EXEC_SUMMARY_SYSTEM as _EXEC_SUMMARY_SYSTEM,
    KEY_FINDINGS_SYSTEM as _KEY_FINDINGS_SYSTEM,
    SCOPE_ENRICHMENT_SYSTEM as _SCOPE_ENRICHMENT_SYSTEM,
)

_TYPE_LABELS = {
    "slr": "Systematic Literature Review",
    "lbd": "Literature-Based Discovery",
    "deep_web": "Deep Web Research",
    "edison": "Edison Scientific Literature Review",
}


# ---------------------------------------------------------------------------
# Matching helpers (synchronous — no LLM)
# ---------------------------------------------------------------------------


def _parse_scope_headings(report_md: str) -> list[tuple[str, int, int]]:
    lines = report_md.split("\n")
    sections: list[tuple[str, int, int]] = []
    for i, line in enumerate(lines):
        if line.startswith("## ") or line.startswith("### "):
            if sections:
                prev_heading, prev_start, _ = sections[-1]
                sections[-1] = (prev_heading, prev_start, i)
            heading = re.sub(r"^#+\s+", "", line).strip()
            sections.append((heading, i, len(lines)))
    return sections


def _match_research_to_scopes(
    report_md: str, research_results: list[dict]
) -> dict[str, list[dict]]:
    sections = _parse_scope_headings(report_md)
    matched: dict[str, list[dict]] = {}
    for result in research_results:
        if result.get("status") != "ok":
            continue
        linked = result.get("linked_scope", "")
        placed = False
        if linked:
            for heading, _, _ in sections:
                if linked.lower() in heading.lower():
                    matched.setdefault(heading, []).append(result)
                    placed = True
                    break
        if not placed:
            query = result.get("query", result.get("question", ""))
            query_words = {w.lower() for w in query.split() if len(w) > 3}
            for heading, _, _ in sections:
                heading_words = {w.lower() for w in re.findall(r"\w+", heading) if len(w) > 3}
                if len(query_words & heading_words) >= 3:
                    matched.setdefault(heading, []).append(result)
                    break
    return matched


def _build_appendices(research_results: list[dict]) -> str:
    ok_results = [r for r in research_results if r.get("status") == "ok"]
    if not ok_results:
        return ""
    type_order = {"slr": 0, "lbd": 1, "deep_web": 2, "edison": 3}
    ok_results.sort(key=lambda r: (type_order.get(r.get("channel", ""), 4), r.get("task_id", "")))
    parts = ["\n---\n", "# Appendix: External Research Evidence\n"]
    current_type = None
    for r in ok_results:
        channel = r.get("channel", "")
        if channel != current_type:
            current_type = channel
            label = _TYPE_LABELS.get(channel, channel)
            parts.append(f"\n## {label} Results\n")
        tid = r.get("task_id", "?")
        question = r.get("query", r.get("question", ""))
        thesis = r.get("thesis", "")
        parts.append(f"### {tid}: {question}\n")
        parts.append(thesis)
        parts.append("")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Async enrichment passes
# ---------------------------------------------------------------------------


async def _enrich_one_scope(
    heading: str,
    results: list[dict],
    section_body: str,
    model: str,
) -> str:
    research_parts = []
    for r in results:
        label = _TYPE_LABELS.get(r.get("channel", ""), r.get("channel", "unknown"))
        question = r.get("query", r.get("question", ""))
        thesis = r.get("thesis", "")
        research_parts.append(f"**{label}: {question}**\n\n{thesis}")

    prompt = (
        f"## Internal Assessment Section: {heading}\n\n{section_body}\n\n"
        f"## External Research Results ({len(results)} queries):\n\n"
        + "\n\n---\n\n".join(research_parts)
    )
    return await acall_llm(prompt, _SCOPE_ENRICHMENT_SYSTEM, model=model)


async def _enrich_scope_sections(
    report_md: str, matched: dict[str, list[dict]], model: str
) -> str:
    if not matched:
        return report_md

    lines = report_md.split("\n")

    async def _process_heading(heading: str, results: list[dict]) -> tuple[int, str]:
        heading_line = None
        for i, line in enumerate(lines):
            if re.sub(r"^#+\s+", "", line).strip() == heading:
                heading_line = i
                break
        if heading_line is None:
            return -1, ""

        heading_text = lines[heading_line]
        level = len(heading_text.split()[0]) if heading_text.startswith("#") else 2
        section_end = len(lines)
        for j in range(heading_line + 1, len(lines)):
            if lines[j].startswith("#"):
                next_level = len(lines[j].split()[0])
                if next_level <= level:
                    section_end = j
                    break

        insert_at = section_end
        for j in range(heading_line + 1, section_end):
            if lines[j].strip().lower().startswith("### recommended research"):
                insert_at = j
                break

        body_preview = "\n".join(lines[heading_line + 1:min(heading_line + 61, insert_at)])
        summary = await _enrich_one_scope(heading, results, body_preview, model)

        block = (
            "\n#### External Research Evidence\n\n"
            "*The following evidence was gathered from published scientific literature "
            "and web sources to complement the internal document assessment.*\n\n"
            + summary + "\n"
        )
        return insert_at, block

    tasks = [_process_heading(h, r) for h, r in matched.items()]
    # asyncio-APPROVED-2: concurrent LLM — scope enrichment per heading (≤15).
    # Convertible in principle but not worth the cost: each Send() worker would need a full
    # copy of `lines` (thousands of entries) in its payload, and the collect node must still
    # do the reverse-order line insertion — no checkpointing benefit for this scale.
    insertions = await asyncio.gather(*tasks)

    for insert_at, block in sorted(insertions, key=lambda x: x[0], reverse=True):
        if insert_at >= 0 and block:
            lines.insert(insert_at, block)

    return "\n".join(lines)


async def _rewrite_key_findings(report_md: str, research_results: list[dict], model: str) -> str:
    ok_results = [r for r in research_results if r.get("status") == "ok"]
    if not ok_results:
        return report_md

    section_start = -1
    for name in ("## Key Insights", "## Key Findings"):
        idx = report_md.find(name)
        if idx >= 0:
            section_start = idx
            break
    if section_start < 0:
        return report_md

    section_end = report_md.find("\n## ", section_start + 10)
    if section_end < 0:
        section_end = len(report_md)

    original_section = report_md[section_start:section_end]
    summaries = []
    for r in ok_results:
        tid = r.get("task_id", "?")
        channel = r.get("channel", "?")
        question = r.get("query", r.get("question", ""))
        thesis = r.get("thesis", "")
        summaries.append(f"**{tid} [{channel}]**: {question}\nKey finding: {thesis}")

    prompt = (
        f"## Original Key Insights:\n{original_section}\n\n"
        f"## External Research Results ({len(summaries)} queries):\n\n"
        + "\n\n---\n\n".join(summaries)
    )
    supplement = await acall_llm(prompt, _KEY_FINDINGS_SYSTEM, model=model)
    if not supplement or len(supplement) < 100:
        return report_md

    insert_text = (
        "\n\n*The following findings are based on systematic literature reviews, "
        "literature-based discovery, and deep web research conducted to address "
        "the scientific uncertainties identified in this assessment.*\n\n"
        + supplement + "\n"
    )
    return report_md[:section_end] + insert_text + report_md[section_end:]


async def _rewrite_executive_summary(
    report_md: str, research_results: list[dict], model: str
) -> str:
    ok_results = [r for r in research_results if r.get("status") == "ok"]
    if not ok_results:
        return report_md

    es_start = report_md.find("## Executive Summary")
    if es_start < 0:
        return report_md
    es_end = report_md.find("\n## ", es_start + 10)
    if es_end < 0:
        es_end = report_md.find("\n# ", es_start + 10)
    if es_end < 0:
        return report_md

    original_es = report_md[es_start:es_end]
    highlights = []
    for r in ok_results:
        tid = r.get("task_id", "?")
        channel = r.get("channel", "?")
        question = r.get("query", r.get("question", ""))
        thesis = r.get("thesis", "")
        highlights.append(f"- {tid} [{channel}]: {question} -> {thesis}")

    prompt = (
        f"## Original Executive Summary:\n{original_es}\n\n"
        f"## External Research Findings ({len(ok_results)} queries completed):\n"
        + "\n".join(highlights)
    )
    new_es = await acall_llm(prompt, _EXEC_SUMMARY_SYSTEM, model=model)
    if not new_es or len(new_es) < 200:
        return report_md

    return (
        report_md[:es_start]
        + "## Executive Summary\n\n" + new_es + "\n"
        + report_md[es_end:]
    )


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------


async def finalize(state: WorkflowState, config: RunnableConfig) -> dict:
    report_md: Optional[str] = state.get("final_report_md") or ""
    research_results: list[dict] = list(state.get("research_results") or [])
    synthesis_model: str = state["synthesis_model"]

    threads_dir = state.get("threads_dir") or ""

    # Promote ASTA science investigation results from causal scope_outputs into
    # research_results so they participate in all downstream enrichment passes.
    seen_ids = {r.get("task_id") for r in research_results}
    for scope in state.get("scope_outputs") or []:
        scope_id = scope.get("scope_id", "")
        for flag in scope.get("science_flags") or []:
            if flag.get("terminal_status") not in ("evidence_gathered", "insufficient_evidence"):
                continue
            answer = flag.get("answer") or ""
            question = flag.get("question") or flag.get("assumption", "")
            if not question:
                continue
            task_id = f"asta:{scope_id}:{flag.get('assumption_id', question[:40])}"
            if task_id in seen_ids:
                continue
            seen_ids.add(task_id)
            research_results.append({
                "task_id": task_id,
                "task_type": "asta",
                "channel": "asta",
                "linked_scope": scope_id,
                "query": question,
                "thesis": answer,
                "status": "ok" if answer else "no_evidence",
                "success": bool(answer),
            })

    ok_count = sum(1 for r in research_results if r.get("status") == "ok")

    if ok_count > 0 and report_md:
        # Match research to scope sections
        matched = _match_research_to_scopes(report_md, research_results)

        # Run enrichment passes — scope enrichment in parallel, then sequential rewrites
        report_md = await _enrich_scope_sections(report_md, matched, synthesis_model)
        report_md = await _rewrite_key_findings(report_md, research_results, synthesis_model)
        report_md = await _rewrite_executive_summary(report_md, research_results, synthesis_model)

        # Add appendices
        appendices = _build_appendices(research_results)
        if appendices:
            report_md += "\n" + appendices

    # Write output
    output_path_str: Optional[str] = None
    if threads_dir and report_md:
        out_dir = Path(threads_dir)
        # asyncio-APPROVED-1: to_thread wraps blocking mkdir
        await asyncio.to_thread(out_dir.mkdir, parents=True, exist_ok=True)
        output_path = out_dir / "final_report_wresearch.md"
        # asyncio-APPROVED-1: to_thread wraps blocking Path.write_text
        await asyncio.to_thread(output_path.write_text, report_md, "utf-8")
        output_path_str = str(output_path)

    return {
        "final_report_wresearch_md_path": output_path_str,
        "final_report_wresearch_md": report_md if report_md else None,
    }
