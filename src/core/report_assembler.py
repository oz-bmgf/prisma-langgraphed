"""Assemble the analyst report markdown from analyze-subgraph outputs.

Full structure (Phase 6 spec):
  1. Executive Summary (LLM — v2 typology-driven)
  2. Table of Contents (auto-generated from section headings)
  3. Portfolio Dashboard (pure computation)
  4. Body Sections — BOW-count routing:
       single-BOW scope → under that BOW heading
       multi-BOW scope  → Cross-Cutting Region only (never duplicated)
  5. Cross-Cutting Region: essay + findings + emergent decisions
  6. Bibliography (deduplicated by file path, sorted alphabetically)
  7. Appendices:
       A. Thread stats (per-scope label, investments, links, confidence)
       B. BOW roster (BOW id, investment count, chunk count)
       C. Coverage index (per-investment documents_read/available)

Called by assemble_report node (analyze.py):
    scope_outputs, cross_cutting_analysis, orientation_summary,
    all_excerpts, confidence_map, coverage_pct, grade, model, threads_dir
"""
from __future__ import annotations

import asyncio
import logging
import re
from typing import Any

from src.config import DEFAULT_SYNTHESIS_MODEL as _DEFAULT_MODEL
from src.core.llm_utils import acall_llm
from src.core.report_charts import render_confusion_matrix as _render_confusion_matrix
from src.core.report_charts import render_scatter_plot as _render_scatter_plot

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Chart generation (delegated to report_charts — matplotlib, inline base64 PNG)
# ---------------------------------------------------------------------------


def _try_render_confusion_matrix(scope_outputs: list[dict]) -> str | None:
    """Team-vs-AI risk severity confusion matrix as inline base64 PNG markdown snippet."""
    b64 = _render_confusion_matrix(scope_outputs, {})
    if b64 is None:
        return None
    return (
        "**Team vs AI Risk Severity**\n\n"
        f"![Team vs AI severity confusion matrix](data:image/png;base64,{b64})\n\n"
        "*Rows = team risk_severity; columns = AI severity. "
        "Diagonal cells (green border) = agreement. "
        "Color intensity ∝ investment count.*\n"
    )


def _try_render_bow_scatter(bow_id: str, scopes: list[dict]) -> str | None:
    """Execution-rate vs approved-amount scatter for one BOW as inline PNG snippet."""
    b64 = _render_scatter_plot(
        bow_id, scopes, {},
        x_axis="execution_rate",
        y_axis="approved_amount",
    )
    if b64 is None:
        return None
    return f"![{bow_id} scatter](data:image/png;base64,{b64})\n\n"


# ---------------------------------------------------------------------------
# Credibility tier display
# ---------------------------------------------------------------------------

_TIER_DISPLAY = {
    "tier1_primary": "Primary",
    "tier2_secondary": "Secondary",
    "tier3_context": "Context",
}


# ---------------------------------------------------------------------------
# Executive Summary (LLM call)
# ---------------------------------------------------------------------------


async def _run_premise_investigation(
    major_bets: list,
    config: Any,
    model: str,
) -> str:
    """Orphan 9 / F-033 — 3-pass per-bet ASTA scientific premise cascade.

    Matches old exec_summary/premise_investigator.run_per_bet_investigation():
    Pass 1: search_asta for each bet to gather scientific evidence.
    Pass 2: extract key findings from the retrieved evidence.
    Pass 3: synthesize a per-bet executive summary paragraph.

    Returns a multi-paragraph string ready for injection into the exec summary.
    Degrades gracefully — empty string if no ASTA access or no bets.
    """
    if not major_bets:
        return ""

    try:
        from src.tools.science_tools import search_asta
    except ImportError:
        return ""

    bet_paragraphs: list[str] = []

    for bet_obj in major_bets[:5]:   # cap at 5 bets
        if isinstance(bet_obj, dict):
            bet_str = bet_obj.get("bet", "")
            bows = bet_obj.get("bows", [])
            amount = bet_obj.get("amount_approx", "")
        else:
            bet_str = str(bet_obj)
            bows = []
            amount = ""

        if not bet_str:
            continue

        bet_label = bet_str[:80]
        bows_str = f" (BOWs: {', '.join(bows[:3])})" if bows else ""
        amount_str = f" ~{amount}" if amount else ""

        # ── Pass 1: ASTA investigation ─────────────────────────────────────────
        asta_evidence: list[str] = []
        for query_suffix in [
            f"{bet_str} evidence efficacy outcomes",
            f"{bet_str} clinical trial results systematic review",
        ]:
            try:
                result = await search_asta.ainvoke({"query": query_suffix[:200]}, config=config)
                if result and not result.startswith(("(", "[ASTA")):
                    asta_evidence.append(result[:1500])
            except Exception:
                pass

        # ── Pass 2: key findings extraction ───────────────────────────────────
        if asta_evidence:
            findings_prompt = (
                f"Strategic bet: {bet_label}{bows_str}{amount_str}\n\n"
                f"Scientific evidence retrieved:\n"
                + "\n\n---\n\n".join(asta_evidence[:2])
                + "\n\nIn 2-3 sentences, what are the key scientific findings relevant to "
                "this bet? Note evidence quality, gaps, and uncertainties."
            )
            try:
                key_findings = str(await acall_llm(findings_prompt, model=model, config=config))[:600]
            except Exception:
                key_findings = "(insufficient scientific evidence)"
        else:
            key_findings = "(no external scientific evidence found in ASTA corpus)"

        # ── Pass 3: per-bet exec summary paragraph ─────────────────────────────
        para_prompt = (
            f"Strategic bet: {bet_label}{bows_str}{amount_str}\n\n"
            f"Science assessment: {key_findings}\n\n"
            "Write one concise executive summary paragraph (3-4 sentences) for this "
            "strategic bet. Assess the scientific evidence base, any critical gaps or "
            "risks, and the implication for Foundation investment decisions. "
            "Be specific and factual."
        )
        try:
            paragraph = str(await acall_llm(para_prompt, model=model, config=config))[:700]
            bet_paragraphs.append(f"**{bet_label}**\n\n{paragraph}")
        except Exception:
            bet_paragraphs.append(f"**{bet_label}**: (science assessment unavailable)")

    return "\n\n".join(bet_paragraphs)


async def _run_narration_pass(config: Any, model: str) -> str:
    """F-032: Run a brief NarrationToolbox evidence-gathering pass before synthesis.

    Calls list_filtered_investments + 3 search_within_scope queries to surface
    cross-investment patterns. Degrades gracefully when search_backend is absent.
    """
    try:
        from src.tools.narration_tools import list_filtered_investments, search_within_scope
        configurable = ((config or {}).get("configurable") or {})
        if not configurable.get("search_backend"):
            return ""

        parts: list[str] = []

        inv_list = await list_filtered_investments.ainvoke({}, config=config)
        parts.append(f"INVESTMENTS IN SCOPE:\n{inv_list[:1500]}")

        for q in [
            "program critical pathway altering risk high severity investments",
            "science assumption evidence weak insufficient literature",
            "execution milestone delivery delay partner capacity risk",
        ]:
            result = await search_within_scope.ainvoke({"query": q, "top_k": 3}, config=config)
            parts.append(f"EVIDENCE — {q}:\n{result[:600]}")

        return "\n\n".join(parts)
    except Exception as exc:
        logger.debug("narration pass failed (non-fatal): %s", exc)
        return ""


async def _build_executive_summary(
    cross_cutting: dict,
    scope_outputs: list[dict],
    model: str,
    config: Any = None,
) -> str:
    metrics = cross_cutting.get("portfolio_metrics") or {}
    total_approved = float(metrics.get("total_approved_dollars", 0) or 0)
    total_paid = float(metrics.get("total_paid_dollars", 0) or 0)
    at_risk = int(metrics.get("at_risk_count", 0) or 0)
    scope_count = len(scope_outputs)

    top_deviations: list[str] = []
    for s in scope_outputs:
        for d in (s.get("section_draft") or {}).get("ranked_deviations", [])[:2]:
            if d.get("description"):
                top_deviations.append(
                    f"  - {s.get('label', s.get('scope_id', '?'))}: "
                    f"{d['description'][:120]} "
                    f"[severity={d.get('severity', 'unknown')}]"
                )

    # F-032: gather live evidence via NarrationToolbox before synthesis
    narration_evidence = await _run_narration_pass(config, model)

    # F-033 / Orphan 9: 3-pass per-bet ASTA scientific premise cascade
    major_bets: list = cross_cutting.get("major_bets") or []
    if not major_bets:
        for s in scope_outputs[:5]:
            bets = (s.get("program_context") or {}).get("major_bets") or []
            if bets:
                major_bets = bets
                break
    premise_evidence = await _run_premise_investigation(major_bets, config, model)

    prompt = (
        "You are writing the executive summary for a quarterly portfolio review report.\n\n"
        f"Portfolio: {scope_count} investments analysed, "
        f"${total_approved / 1e6:.1f}M approved, "
        f"${total_paid / 1e6:.1f}M paid, "
        f"{at_risk} investments at pathway-altering or program-critical risk.\n\n"
        f"Patterns identified:\n"
        + "\n".join(f"  - {p}" for p in cross_cutting.get("patterns", [])[:5])
        + "\n\nTop deviations by risk:\n"
        + ("\n".join(top_deviations[:8]) if top_deviations else "  (none above threshold)")
        + "\n\nEmergent decisions:\n"
        + "\n".join(
            f"  - {d.get('title', '?')}: {d.get('description', '')[:100]}"
            for d in cross_cutting.get("emergent_decisions", [])[:4]
        )
        + (f"\n\nNARRATION EVIDENCE (from portfolio index):\n{narration_evidence}"
           if narration_evidence else "")
        + (f"\n\nPER-BET SCIENTIFIC PREMISE ASSESSMENT (3-pass ASTA investigation):\n{premise_evidence}"
           if premise_evidence else "")
        + "\n\nWrite a 4-5 paragraph executive summary. Cover:\n"
        "1. Risk disagreements between AI assessment and team scores\n"
        "2. Material deviations by dollar volume and severity\n"
        "3. Science and evidence dependencies at risk (draw on the per-bet science assessments above)\n"
        "4. Portfolio patterns and strategic implications\n\n"
        "Be specific, factual, and action-oriented. Cite specific investments. No hedging."
    )
    try:
        result = await acall_llm(prompt, model=model, config=config)
        return result if isinstance(result, str) else str(result)
    except Exception as exc:
        logger.warning("_build_executive_summary LLM failed: %s", exc)
        return cross_cutting.get("essay", "*Executive summary pending.*")[:2000]


# ---------------------------------------------------------------------------
# Portfolio Dashboard (pure computation)
# ---------------------------------------------------------------------------


def _build_portfolio_dashboard(
    scope_outputs: list[dict],
    coverage_pct: float,
    grade: str,
    confidence_map: dict,
) -> str:
    metrics = {}
    for s in scope_outputs:
        facts = s.get("investment_facts") or {}
        metrics["total_approved"] = metrics.get("total_approved", 0) + float(facts.get("approved_amount", 0) or 0)
        metrics["total_paid"] = metrics.get("total_paid", 0) + float(facts.get("paid_amount", 0) or 0)

    at_risk_dollars = sum(
        float((s.get("investment_facts") or {}).get("approved_amount", 0) or 0)
        for s in scope_outputs
        if (s.get("investment_report") or {}).get("severity") in (
            "program_critical", "pathway_altering"
        )
    )

    # Timeline slips: past planned end with execution_rate < 0.8
    slip_count = sum(
        1 for s in scope_outputs
        if float((s.get("investment_facts") or {}).get("timeline_slip_months", 0) or 0) > 0
        and float((s.get("investment_facts") or {}).get("execution_rate", 1) or 1) < 0.8
    )

    scope_fit_counts: dict[str, int] = {}
    for s in scope_outputs:
        fit = s.get("scope_fit", "unknown")
        scope_fit_counts[fit] = scope_fit_counts.get(fit, 0) + 1

    lines = [
        "## Portfolio Dashboard\n",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Coverage grade | **{grade}** ({coverage_pct * 100:.1f}% documents reviewed) |",
        f"| Investments analysed | {len(scope_outputs)} |",
        f"| Total approved | ${metrics.get('total_approved', 0) / 1e6:.1f}M |",
        f"| Total disbursed | ${metrics.get('total_paid', 0) / 1e6:.1f}M |",
        f"| Dollars at risk (pathway+ severity) | ${at_risk_dollars / 1e6:.1f}M |",
        f"| Timeline slips | {slip_count} investment(s) past end date with <80% execution |",
    ]
    if scope_fit_counts:
        lines.append(f"| Partner mix | {', '.join(f'{v} {k}' for k, v in scope_fit_counts.items())} |")

    high_conf = sum(1 for v in confidence_map.values() if v == "high")
    low_conf = sum(1 for v in confidence_map.values() if v == "low")
    lines.append(f"| Confidence | {high_conf} high / {len(confidence_map) - high_conf - low_conf} medium / {low_conf} low |")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# BOW routing
# ---------------------------------------------------------------------------


def _route_scopes(scope_outputs: list[dict]) -> tuple[dict[str, list[dict]], list[dict]]:
    """Split scopes into single-BOW (routed under BOW heading) and multi-BOW (cross-cutting only)."""
    single_bow: dict[str, list[dict]] = {}   # bow_id → [scope, ...]
    multi_bow: list[dict] = []

    for s in scope_outputs:
        bow_ids = s.get("bow_ids") or []
        if len(bow_ids) == 1:
            bid = bow_ids[0]
            single_bow.setdefault(bid, []).append(s)
        else:
            multi_bow.append(s)

    return single_bow, multi_bow


def _scope_body_section(scope: dict) -> str:
    """Render one scope's SectionDraft as markdown."""
    draft = scope.get("section_draft") or {}
    label = draft.get("heading") or scope.get("label") or scope.get("scope_id", "Unknown")
    summary = draft.get("summary") or scope.get("synthesis") or ""
    deviations = draft.get("ranked_deviations") or []
    inv_report = scope.get("investment_report") or {}
    facts = scope.get("investment_facts") or {}
    bow_context = scope.get("bow_context") or {}

    parts = [f"#### {label}\n"]

    # Quick-look metadata line
    meta_items = []
    inv_ids = scope.get("inv_ids") or [scope.get("inv_id", "")]
    meta_items.append(f"Investments: {', '.join(inv_ids)}")
    if facts.get("approved_amount"):
        meta_items.append(f"Approved: ${float(facts['approved_amount']) / 1e6:.1f}M")
    if facts.get("execution_rate") is not None:
        meta_items.append(f"Execution: {float(facts['execution_rate']) * 100:.0f}%")
    overall = inv_report.get("overall_status", "")
    if overall:
        meta_items.append(f"Status: **{overall}**")
    div = inv_report.get("divergence_severity", "")
    if div and div != "aligned":
        meta_items.append(f"AI/Team divergence: **{div}**")
    parts.append("*" + " | ".join(meta_items) + "*\n")

    if summary:
        parts.append(summary)
        parts.append("")

    if bow_context.get("market_context"):
        parts.append(f"**External context:** {bow_context['market_context']}")
        if bow_context.get("regulatory_context"):
            parts.append(bow_context["regulatory_context"])
        parts.append("")

    if deviations:
        parts.append("**Ranked deviations (by dollars at risk × severity):**")
        for d in deviations[:5]:
            dollars = float(d.get("dollars_at_risk", 0) or 0)
            parts.append(
                f"- {d.get('description', '?')[:120]} "
                f"*(${dollars / 1e6:.1f}M at risk, {d.get('severity', '?')})*"
            )
        parts.append("")

    ai_exec = inv_report.get("ai_execution_verdict", "")
    if ai_exec:
        parts.append(f"**AI execution verdict:** {ai_exec[:300]}")
        parts.append("")

    risks = inv_report.get("key_risks", [])
    if risks:
        parts.append("**Key risks:** " + "; ".join(str(r)[:80] for r in risks[:3]))
        parts.append("")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Bibliography
# ---------------------------------------------------------------------------


def _build_bibliography(all_excerpts: list[dict]) -> tuple[str, list[dict]]:
    """Deduplicate excerpts by source path and build bibliography section."""
    seen: dict[str, dict] = {}
    for ex in all_excerpts:
        src = ex.get("source", "")
        if src and src not in seen:
            seen[src] = ex

    sorted_sources = sorted(seen.values(), key=lambda x: (x.get("source") or "").lower())

    if not sorted_sources:
        return "*No cited sources.*", []

    lines = ["## Bibliography\n"]
    bib_list: list[dict] = []
    for i, src_dict in enumerate(sorted_sources):
        src = src_dict.get("source", "")
        tier = _TIER_DISPLAY.get(src_dict.get("credibility_tier", ""), "")
        tier_str = f" [{tier}]" if tier else ""
        lines.append(f"{i+1}. `{src}`{tier_str}")
        bib_list.append({
            "filename": src,
            "credibility_tier": src_dict.get("credibility_tier", ""),
        })

    return "\n".join(lines), bib_list


# ---------------------------------------------------------------------------
# Appendices
# ---------------------------------------------------------------------------


def _build_appendices(
    scope_outputs: list[dict],
    confidence_map: dict,
    all_excerpts: list[dict],
) -> str:
    parts = ["## Appendices\n"]

    # Appendix A — Thread stats
    parts.append("### Appendix A: Thread Statistics\n")
    parts.append("| Scope | Investments | Links investigated | Confidence |")
    parts.append("|-------|------------|-------------------|------------|")
    for s in scope_outputs:
        label = s.get("label") or s.get("scope_id", "?")
        inv_ids = s.get("inv_ids") or [s.get("inv_id", "?")]
        links = len(s.get("link_assessments") or [])
        conf = confidence_map.get(s.get("scope_id", ""), "unknown")
        parts.append(f"| {label[:40]} | {len(inv_ids)} | {links} | {conf} |")
    parts.append("")

    # Appendix B — BOW roster
    parts.append("### Appendix B: Bundle of Work Roster\n")
    bow_stats: dict[str, dict] = {}
    for s in scope_outputs:
        for bid in (s.get("bow_ids") or []):
            if bid not in bow_stats:
                bow_stats[bid] = {"inv_count": 0, "chunk_count": 0}
            bow_stats[bid]["inv_count"] += len(s.get("inv_ids") or [s.get("inv_id", "")])
            bow_stats[bid]["chunk_count"] = max(
                bow_stats[bid]["chunk_count"],
                int(s.get("chunk_count", 0) or 0),
            )
    parts.append("| BOW ID | Investments | Chunks indexed |")
    parts.append("|--------|------------|----------------|")
    for bid, stats in sorted(bow_stats.items()):
        parts.append(f"| {bid} | {stats['inv_count']} | {stats['chunk_count']:,} |")
    parts.append("")

    # Appendix C — Coverage index
    parts.append("### Appendix C: Coverage Index\n")
    parts.append("| Investment | Documents read | Coverage |")
    parts.append("|-----------|---------------|---------|")
    inv_docs: dict[str, set[str]] = {}
    for ex in all_excerpts:
        iid = ex.get("inv_id", "")
        src = ex.get("source", "")
        if iid and src:
            inv_docs.setdefault(iid, set()).add(src)
    for s in scope_outputs:
        for iid in (s.get("inv_ids") or [s.get("inv_id", "")]):
            docs_read = len(inv_docs.get(iid, set()))
            parts.append(f"| {iid} | {docs_read} | — |")
    parts.append("")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Main assembly function
# ---------------------------------------------------------------------------


async def assemble_report(
    scope_outputs: list[dict],
    *,
    cross_cutting_analysis: dict | None = None,
    all_excerpts: list[dict] | None = None,
    confidence_map: dict | None = None,
    coverage_pct: float = 0.0,
    grade: str = "D",
    model: str = "",
    config: Any = None,
    # Kept for call-site backward compatibility; ignored — data is in cross_cutting_analysis
    clusters: list[dict] | None = None,
    orientation_summary: str = "",
) -> dict[str, Any]:
    """Assemble a full-structure portfolio review report.

    Returns {"markdown": str, "body": str, "bibliography": list[dict]}.
    """
    model = model or _DEFAULT_MODEL
    all_excerpts = all_excerpts or []
    confidence_map = confidence_map or {}
    cross_cutting = cross_cutting_analysis or {}

    sections: list[str] = []

    # ── 1. Title ─────────────────────────────────────────────────────────────
    sections.append("# Portfolio Analysis Report\n")

    # ── 2. Executive Summary (LLM) ────────────────────────────────────────────
    exec_summary = await _build_executive_summary(cross_cutting, scope_outputs, model, config=config)
    sections.append("## Executive Summary\n")
    sections.append(exec_summary)
    sections.append("")

    # ── 3. Portfolio Dashboard ────────────────────────────────────────────────
    sections.append(_build_portfolio_dashboard(scope_outputs, coverage_pct, grade, confidence_map))
    sections.append("")

    # ── 3b. Team-vs-AI confusion matrix (optional, requires Pillow) ───────────
    cm_snippet = _try_render_confusion_matrix(scope_outputs)
    if cm_snippet:
        sections.append(cm_snippet)
        sections.append("")

    # ── 4. Body Sections (BOW routing) ────────────────────────────────────────
    single_bow_map, multi_bow_scopes = _route_scopes(scope_outputs)

    if single_bow_map:
        sections.append("## Investment Analysis by Bundle of Work\n")
        for bow_id, bow_scopes in sorted(single_bow_map.items()):
            sections.append(f"### {bow_id}\n")
            scatter = _try_render_bow_scatter(bow_id, bow_scopes)
            if scatter:
                sections.append(scatter)
            for scope in bow_scopes:
                sections.append(_scope_body_section(scope))
            sections.append("")

    # ── 5. Cross-Cutting Region ───────────────────────────────────────────────
    sections.append("## Cross-Cutting Analysis\n")

    essay = cross_cutting.get("essay", "")
    if essay:
        sections.append(essay)
        sections.append("")

    # Multi-BOW scopes go here (never in individual BOW sections)
    if multi_bow_scopes:
        sections.append("### Multi-BOW Scope Findings\n")
        for scope in multi_bow_scopes:
            sections.append(_scope_body_section(scope))

    patterns = cross_cutting.get("patterns", [])
    if patterns:
        sections.append("### Portfolio Patterns\n")
        for p in patterns:
            sections.append(f"- {p}")
        sections.append("")

    contradictions = cross_cutting.get("contradictions", [])
    if contradictions:
        sections.append("### Contradictions and Tensions\n")
        for c in contradictions:
            sections.append(f"- {c}")
        sections.append("")

    shared_deps = cross_cutting.get("shared_dependencies", [])
    if shared_deps:
        sections.append("### Shared Dependencies\n")
        for d in shared_deps:
            sections.append(f"- {d}")
        sections.append("")

    emergent = cross_cutting.get("emergent_decisions", [])
    if emergent:
        sections.append("### Emergent Decisions\n")
        for ed in emergent:
            title = ed.get("title", "Untitled")
            desc = ed.get("description", "")
            urgency = ed.get("urgency", "")
            urgency_str = f" *(urgency: {urgency})*" if urgency else ""
            sections.append(f"**{title}**{urgency_str}: {desc}")
            sections.append("")

    # ── 6. Bibliography ───────────────────────────────────────────────────────
    bib_text, bib_list = _build_bibliography(all_excerpts)
    sections.append(bib_text)
    sections.append("")

    # ── 7. Appendices ─────────────────────────────────────────────────────────
    sections.append(_build_appendices(scope_outputs, confidence_map, all_excerpts))

    # ── Table of Contents (inserted after title) ──────────────────────────────
    toc_lines = ["## Table of Contents\n"]
    h2_pattern = re.compile(r"^## (.+)$", re.MULTILINE)
    h3_pattern = re.compile(r"^### (.+)$", re.MULTILINE)
    full_body = "\n".join(sections)
    for m in h2_pattern.finditer(full_body):
        heading = m.group(1)
        anchor = re.sub(r"[^a-z0-9-]", "", heading.lower().replace(" ", "-"))
        toc_lines.append(f"- [{heading}](#{anchor})")
        for m3 in h3_pattern.finditer(full_body[m.end():]):
            sub = m3.group(1)
            sub_anchor = re.sub(r"[^a-z0-9-]", "", sub.lower().replace(" ", "-"))
            toc_lines.append(f"  - [{sub}](#{sub_anchor})")
            if m3.end() > 2000:
                break
    toc_text = "\n".join(toc_lines) + "\n"

    # Insert ToC after the title
    full_body = full_body.replace("# Portfolio Analysis Report\n", "# Portfolio Analysis Report\n\n" + toc_text + "\n", 1)

    return {
        "markdown": full_body,
        "body": full_body,
        "bibliography": bib_list,
    }
