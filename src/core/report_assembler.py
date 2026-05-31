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
from typing import Any, Optional

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
        scopes, {},
        x_axis="execution_rate",
        y_axis="approved_amount",
        title=bow_id,
    )
    if b64 is None:
        return None
    return f"![{bow_id} scatter](data:image/png;base64,{b64})\n\n"


def _try_render_portfolio_scatter(
    scope_outputs: list[dict],
    investment_scoring: dict,
) -> str | None:
    """Portfolio-wide execution-rate vs approved-amount scatter as inline base64 PNG snippet."""
    b64 = _render_scatter_plot(scope_outputs, investment_scoring)
    if b64 is None:
        return None
    return (
        "**Execution Rate vs Approved Amount**\n\n"
        f"![Portfolio scatter](data:image/png;base64,{b64})\n\n"
        "*X = execution rate (paid/approved); Y = approved amount ($M). "
        "Color = AI-assessed severity.*\n"
    )


# ---------------------------------------------------------------------------
# Scoring Comparison (BoW-level divergence + confusion matrix + scatter)
# ---------------------------------------------------------------------------


def _build_scoring_comparison(
    scope_outputs: list[dict],
    investment_scoring: Optional[dict],
) -> str:
    """Build the ## Scoring Comparison section.

    Mirrors old report_assembler.py ## Scoring Comparison content:
      - BoW-level aggregate divergence table (sorted by at-risk dollars)
      - Per-investment AI-vs-team verdict table (pathway-altering+ only)
      - Team-vs-AI confusion matrix PNG
      - Portfolio execution-rate scatter PNG
    """
    investment_scoring = investment_scoring or {}
    parts = ["## Scoring Comparison\n"]
    parts.append(
        "*Verdict legend: **E** Exceeds · **M** Meets · **B** Below · "
        "**T** Too Early to Tell · **—** Not rated.*\n"
    )

    # ── BoW-level divergence table ────────────────────────────────────────
    bow_rows: dict[str, dict] = {}
    for s in scope_outputs:
        for bid in (s.get("bow_ids") or []):
            if bid not in bow_rows:
                bow_rows[bid] = {
                    "label": (s.get("label") or bid)[:60],
                    "inv_count": 0,
                    "program_critical": 0,
                    "pathway_altering": 0,
                    "total_approved": 0.0,
                    "at_risk_dollars": 0.0,
                }
            row = bow_rows[bid]
            n_inv = len(s.get("inv_ids") or [s.get("inv_id", "")])
            row["inv_count"] += n_inv
            inv_report = s.get("investment_report") or {}
            div_sev = (inv_report.get("divergence_severity") or "").lower()
            if div_sev == "program_critical":
                row["program_critical"] += 1
            elif div_sev == "pathway_altering":
                row["pathway_altering"] += 1
            facts = s.get("investment_facts") or {}
            approved = float(facts.get("approved_amount", 0) or 0)
            row["total_approved"] += approved
            if div_sev in ("program_critical", "pathway_altering"):
                row["at_risk_dollars"] += approved

    if bow_rows:
        sorted_bows = sorted(bow_rows.items(), key=lambda x: -x[1]["at_risk_dollars"])
        lines = ["**BoW-level scoring divergence** (sorted by dollars at pathway-altering+ risk)\n"]
        lines.append("| BoW | Investments | AI Critical | AI Pathway | At-Risk ($M) |")
        lines.append("|-----|------------|------------|-----------|-------------|")
        for bid, r in sorted_bows:
            at_risk_str = f"${r['at_risk_dollars'] / 1e6:.1f}M" if r["at_risk_dollars"] else "—"
            lines.append(
                f"| {r['label']} | {r['inv_count']} "
                f"| {r['program_critical']} | {r['pathway_altering']} | {at_risk_str} |"
            )
        parts.append("\n".join(lines))
        parts.append("")

    # ── Per-investment divergence table (pathway-altering and program-critical) ──
    diverging: list[dict] = []
    for s in scope_outputs:
        inv_report = s.get("investment_report") or {}
        div_sev = (inv_report.get("divergence_severity") or "").lower()
        if div_sev not in ("program_critical", "pathway_altering"):
            continue
        facts = s.get("investment_facts") or {}
        inv_ids = s.get("inv_ids") or [s.get("inv_id", "")]
        approved = float(facts.get("approved_amount", 0) or 0)
        diverging.append({
            "inv_id": inv_ids[0] if inv_ids else "?",
            "approved": approved,
            "team_exec": inv_report.get("team_execution_score", "—") or "—",
            "ai_verdict": inv_report.get("overall_status", "—") or "—",
            "ai_sev": inv_report.get("severity", "—") or "—",
            "div_sev": div_sev,
        })
    diverging.sort(key=lambda x: -x["approved"])

    if diverging:
        lines = [
            "**AI-vs-team scoring divergence** "
            "(pathway-altering and program-critical investments)\n"
        ]
        lines.append("| Investment | Approved | Team Execution | AI Verdict | Severity |")
        lines.append("|-----------|---------|---------------|-----------|---------|")
        for d in diverging[:25]:
            approved_str = f"${d['approved'] / 1e6:.1f}M" if d["approved"] else "—"
            lines.append(
                f"| **{d['inv_id']}** | {approved_str} | {d['team_exec']} "
                f"| {d['ai_verdict']} | **{d['div_sev']}** |"
            )
        parts.append("\n".join(lines))
        parts.append("")

    # ── Team-vs-AI confusion matrix ───────────────────────────────────────
    cm_snippet = _try_render_confusion_matrix(scope_outputs)
    if cm_snippet:
        parts.append(cm_snippet)
        parts.append("")

    # ── Portfolio scatter ─────────────────────────────────────────────────
    scatter_snippet = _try_render_portfolio_scatter(scope_outputs, investment_scoring)
    if scatter_snippet:
        parts.append(scatter_snippet)
        parts.append("")

    return "\n".join(parts)


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


def _strip_leading_heading(text: str) -> str:
    """Remove the first line if it is a markdown heading (# … or ## …).

    LLMs sometimes open their response with a section title that duplicates
    the heading the assembler has already emitted (e.g. "## Executive Summary"
    or "# Calibration"). Stripping it prevents double-headings and heading-
    level conflicts in the final PDF.
    """
    lines = text.splitlines()
    if lines and re.match(r"^#{1,6}\s", lines[0]):
        return "\n".join(lines[1:]).lstrip("\n")
    return text


async def _build_executive_summary(
    cross_cutting: dict,
    scope_outputs: list[dict],
    model: str,
    config: Any = None,
    program: str = "",
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

    program_label = program or "Portfolio"
    prompt = (
        f"You are writing the executive summary for the {program_label} quarterly portfolio review.\n\n"
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
        "Be specific, factual, and action-oriented. Cite specific investments. No hedging.\n"
        "Do NOT add a heading or title line — output prose paragraphs only."
    )
    try:
        result = await acall_llm(prompt, model=model, config=config)
        text = result if isinstance(result, str) else str(result)
        return _strip_leading_heading(text)
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
# Key Insights (per-bet LLM narrative with alignment verdicts)
# ---------------------------------------------------------------------------


async def _build_key_insights(
    scope_outputs: list[dict],
    cross_cutting: dict,
    model: str,
    config: Any = None,
) -> str:
    """Build the ## Key Insights section — per-bet narrative with alignment verdicts.

    Mirrors old report_assembler.py ## Key Insights content:
      - Alignment verdicts legend
      - Per-bet analysis: deviations review, premises review, patterns review,
        alignment verdict (confirms / requires update / invalidates / neutral)
      - Cross-bet patterns
      - Calibration paragraph
    """
    # Resolve major_bets from cross_cutting or scope program_context
    major_bets: list = cross_cutting.get("major_bets") or []
    if not major_bets:
        for s in scope_outputs[:8]:
            bets = (s.get("program_context") or {}).get("major_bets") or []
            if bets:
                major_bets = bets
                break
    # Fall back to clusters as bet-like groupings when no named bets available
    if not major_bets:
        for c in (cross_cutting.get("clusters") or []):
            theme = c.get("theme", "")
            if theme:
                major_bets.append({
                    "bet": theme,
                    "bows": c.get("scope_ids") or [],
                    "amount_approx": "",
                })

    if not major_bets and not scope_outputs:
        return ""

    parts = ["## Key Insights\n"]
    parts.append(
        "*Alignment verdicts — **confirms**: documented evidence supports "
        "the stated strategic claim · **requires update**: the claim needs "
        "revision given new evidence (the world has changed since it was written) · "
        "**invalidates**: the claim is no longer tenable on the documented record · "
        "**neutral**: no implication either direction this cycle.*\n"
    )

    # Group scopes by bow_id for fast lookup
    bow_scope_map: dict[str, list[dict]] = {}
    for s in scope_outputs:
        for bid in (s.get("bow_ids") or [s.get("scope_id", "")]):
            bow_scope_map.setdefault(bid, []).append(s)

    async def _analyze_one_bet(bet_obj: Any) -> str:
        """Generate analysis for one strategic bet. Returns empty string on failure."""
        if isinstance(bet_obj, dict):
            bet_str = bet_obj.get("bet", "")
            bows = bet_obj.get("bows") or []
            amount = bet_obj.get("amount_approx", "")
        else:
            bet_str = str(bet_obj)
            bows = []
            amount = ""
        if not bet_str:
            return ""

        # Gather relevant scopes for this bet's BOWs
        relevant: list[dict] = []
        for bid in bows:
            relevant.extend(bow_scope_map.get(bid, []))
        if not relevant:
            relevant = scope_outputs[:12]  # fallback when no BOW mapping

        # Compact scope summaries for the prompt
        scope_lines: list[str] = []
        for s in relevant[:10]:
            inv_ids = s.get("inv_ids") or [s.get("inv_id", "")]
            inv_report = s.get("investment_report") or {}
            facts = s.get("investment_facts") or {}
            draft = s.get("section_draft") or {}
            approved = float(facts.get("approved_amount", 0) or 0)
            deviations = [
                d.get("description", "")[:100]
                for d in (draft.get("ranked_deviations") or [])[:3]
            ]
            line = (
                f"  {', '.join(inv_ids)}"
                f"  team={inv_report.get('team_execution_score', '?')}"
                f"  ai_verdict={inv_report.get('overall_status', '?')}"
                f"  divergence={inv_report.get('divergence_severity', '?')}"
                f"  approved=${approved / 1e6:.1f}M"
            )
            if deviations:
                line += "\n  deviations: " + "; ".join(deviations)
            scope_lines.append(line)

        prompt = (
            f"Write the '{bet_str}' section of the Key Insights chapter "
            "in a quarterly portfolio review report.\n\n"
            f"**Strategic bet:** {bet_str}"
            + (f"\nBundles of Work: {', '.join(bows)}" if bows else "")
            + (f"\nApproximate exposure: {amount}" if amount else "")
            + "\n\nRelevant investment findings:\n"
            + ("\n\n".join(scope_lines) if scope_lines else "(no investment data)")
            + "\n\nWrite 3-5 tight paragraphs structured as:\n"
            "1. **Deviations review:** flag specific investments by ID. Cite dollar amounts, "
            "timeline slips, milestone misses, and months at risk.\n"
            "2. **Premises review:** assess whether the scientific/strategic premises of this "
            "bet are supported. Note evidence gaps and contested assumptions.\n"
            "3. **Patterns review:** cross-investment patterns (measurement gaps, partner "
            "concentration, execution-lane risks, outputs-without-outcomes).\n"
            "4. **Alignment verdict:** exactly one of **confirms** / **requires update** / "
            "**invalidates** / **neutral** — one sentence of rationale. "
            "Format as: 'Alignment verdict: requires update — <rationale>.'\n\n"
            "Be specific. Cite investment IDs and dollar amounts. No hedging."
        )

        try:
            text = await acall_llm(prompt, model=model, config=config)
            if not isinstance(text, str) or not text.strip():
                return ""
            return f"### {bet_str[:120]}\n\n{_strip_leading_heading(text.strip())}\n"
        except Exception as exc:
            logger.warning("_build_key_insights bet '%s' failed: %s", bet_str[:40], exc)
            return ""

    # Run per-bet analyses in parallel (capped at 6 bets)
    bet_results = await asyncio.gather(
        *[_analyze_one_bet(b) for b in major_bets[:6]],
        return_exceptions=True,
    )
    for result in bet_results:
        if isinstance(result, str) and result.strip():
            parts.append(result)

    # ── Cross-bet patterns ─────────────────────────────────────────────────
    patterns = cross_cutting.get("patterns") or []
    if patterns:
        parts.append("### Cross-bet patterns\n")
        for p in patterns[:8]:
            parts.append(f"- {p}")
        parts.append("")

    # ── Calibration paragraph ─────────────────────────────────────────────
    contradictions = cross_cutting.get("contradictions") or []
    if patterns or contradictions:
        calib_prompt = (
            "Write a 'Calibration' paragraph (3-5 sentences) for a portfolio review report. "
            "Cover: (1) what the assessment is confident about and why, "
            "(2) where significant uncertainty remains, "
            "(3) what could not be determined from the available evidence.\n\n"
            + ("Cross-cutting patterns:\n" + "\n".join(f"- {p}" for p in patterns[:5])
               if patterns else "")
            + ("\n\nContradictions:\n" + "\n".join(f"- {c}" for c in contradictions[:3])
               if contradictions else "")
        )
        try:
            calib_text = await acall_llm(calib_prompt, model=model, config=config)
            if isinstance(calib_text, str) and calib_text.strip():
                parts.append("### Calibration\n")
                parts.append(_strip_leading_heading(calib_text.strip()))
                parts.append("")
        except Exception as exc:
            logger.warning("_build_key_insights calibration failed: %s", exc)

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
    investment_scoring: dict | None = None,
    program: str = "",
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
    report_title = f"# {program} — Portfolio Risk Assessment\n" if program else "# Portfolio Risk Assessment\n"
    sections.append(report_title)

    # ── 2. Executive Summary (LLM) ────────────────────────────────────────────
    exec_summary = await _build_executive_summary(
        cross_cutting, scope_outputs, model, config=config, program=program
    )
    sections.append("## Executive Summary\n")
    sections.append(exec_summary)
    sections.append("")

    # ── 3. Portfolio Dashboard ────────────────────────────────────────────────
    sections.append(_build_portfolio_dashboard(scope_outputs, coverage_pct, grade, confidence_map))
    sections.append("")

    # ── 4. Scoring Comparison (divergence tables + confusion matrix + scatter) ──
    # Mirrors old report_assembler.py ## Scoring Comparison section which lives
    # in the focused PDF (before the ## Investment Analysis split marker).
    scoring_comp = _build_scoring_comparison(scope_outputs, investment_scoring)
    sections.append(scoring_comp)
    sections.append("")

    # ── 5. Key Insights (per-bet LLM narrative with alignment verdicts) ────────
    # Mirrors old report_assembler.py ## Key Insights section — the analytical
    # body of the focused PDF (Bet 1-N + cross-bet patterns + calibration).
    key_insights = await _build_key_insights(
        scope_outputs, cross_cutting, model, config=config
    )
    if key_insights:
        sections.append(key_insights)
        sections.append("")

    # ── 6. Body Sections (BOW routing) ────────────────────────────────────────
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

    # ── 7. Cross-Cutting Region ───────────────────────────────────────────────
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

    # ── 8. Bibliography ───────────────────────────────────────────────────────
    bib_text, bib_list = _build_bibliography(all_excerpts)
    sections.append(bib_text)
    sections.append("")

    # ── 9. Appendices ─────────────────────────────────────────────────────────
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

    # Insert ToC after the first H1 title (program name varies)
    full_body = re.sub(
        r"(^# .+\n)",
        lambda m: m.group(1) + "\n" + toc_text + "\n",
        full_body,
        count=1,
        flags=re.MULTILINE,
    )

    return {
        "markdown": full_body,
        "body": full_body,
        "bibliography": bib_list,
    }
