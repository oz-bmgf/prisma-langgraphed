"""Decision projection — evidence-weighted action decisions from scope output.

project_decisions() drives the full pipeline:
  1. Build a goal-anchored prompt (§1 BoW goal → §2 necessity role → §3 evidence → §4 ask)
  2. Call LLM with DecisionProjectionOutput schema
  3. Sanitize each candidate (validate type, require recommended_action + triggering_links)
  4. Filter via §1a evidence gate (low-evidence decisions need thin type OR corroboration≥2)
  5. Rank by rank score: corroboration × materiality × log10(dollars+10) × urgency × evidence
  6. Apply caps: max DECISION_MAX_PER_INV per INV, max DECISION_MAX_PER_SCOPE total

Ported from old-repo decision_projection.py _build_decisions_prompt §1–§4 structure.
Key adaptation: takes scope_output dict (not ScopeOutput dataclass); necessity_assessment
is a JSON string on scope_output; financial data looked up from causal_model.links.

DECISION_TYPE_VOCABULARY and _THIN_EVIDENCE_DECISION_TYPES are the two
canonical sets used by the §1a gate.

No LangGraph imports.
"""
from __future__ import annotations

import json
import logging
import math
from typing import Any

from src.config import (
    DECISION_MAX_PER_INV,
    DECISION_MAX_PER_SCOPE,
    DEFAULT_MAX_TOKENS,
    DEFAULT_SYNTHESIS_MODEL,
)
from src.core.evidence_model import Decision
from src.core.llm_utils import acall_llm, acall_structured
from src.core.output_schemas import DecisionCandidate, DecisionProjectionOutput

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type vocabulary
# ---------------------------------------------------------------------------

DECISION_TYPE_VOCABULARY: frozenset[str] = frozenset({
    "approve_with_conditions",
    "defer_pending_data",
    "request_progress_packet",
    "extend_no_cost",
    "supplement",
    "redirect_funds",
    "terminate_unless_resolved",
    "escalate_to_leadership",
    "escalate_to_partner",
    "schedule_review",
    "validate_assumption",
    "align_with_strategy_team",
    "approve_as_is",
    "monitor",
    "decommission_layer",
})

# Types that bypass §1a low-evidence gate.
_THIN_EVIDENCE_DECISION_TYPES: frozenset[str] = frozenset({
    "request_progress_packet",
    "validate_assumption",
})

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a portfolio decision analyst. You are NOT writing analysis prose
— you are converting an analyst's per-investment causal-link findings
into a small set of specific, actionable decisions that a foundation
portfolio review meeting can act on this week.

Each decision you emit must:
1. Connect to the BoW's stated strategic goal (`goal_link`)
2. Pick a `decision_type` from this exact vocabulary:
   approve_with_conditions, defer_pending_data, request_progress_packet,
   extend_no_cost, supplement, redirect_funds, terminate_unless_resolved,
   escalate_to_leadership, escalate_to_partner, schedule_review,
   validate_assumption, align_with_strategy_team, approve_as_is,
   monitor, decommission_layer
3. Cite the specific LinkAssessment(s) that trigger it via
   `triggering_link_ids` (use the exact link_name values you were given)
4. Be specific: name the INV, name the action verb, and either name
   a date OR a deliverable the team must produce.

Bad: "Monitor INV-X for ongoing risk."
Good: "Defer INV-X supplement until Q1 2026 retrospective progress packet
       documents the previously-reported milestone slip on antigen design."

Do NOT estimate $ amounts or timelines yourself — those will be sourced
from the LinkAssessment fields directly after you emit.

Return ONLY valid JSON. No prose outside JSON."""

_TASK_INSTRUCTIONS = """\
For each INV in this scope, decide if a portfolio decision is warranted
RIGHT NOW given the goal + role + evidence. Emit 0-3 decisions per INV
(most INVs will have 0-1). You may also emit scope-level decisions
(inv_id="") when 3+ INVs surface the same recommendation.

For each decision return:

{
  "decisions": [
    {
      "inv_id": "INV-XXXXXX",
      "decision_type": "<from controlled vocab>",
      "recommended_action": "<specific 1-2 sentence action with verb + target>",
      "goal_link": "<one sentence: how this serves the BoW's stated goal>",
      "substitution_path": "<from necessity.substitutes: what absorbs the gap if this investment stops>",
      "triggering_link_ids": ["<exact link_name from the evidence above>", ...],
      "corroboration_count": <int: number of independent evidence sources>,
      "cost_impact_dollars": <float: dollars at risk; cite from link assessments above>,
      "timeline_impact_months": <float: months at risk; cite from link assessments above>,
      "confidence": "high|moderate|low|insufficient",
      "urgency": "immediate|quarterly|annual_cycle",
      "materiality": "material|uncertain|not_material",
      "rationale": "<1-2 sentence evidence basis>"
    }
  ]
}

Rules:
- `decision_type` MUST be exactly one of the 15 vocabulary verbs listed
  in the system prompt. Decisions with an unknown verb will be dropped.
- `triggering_link_ids` MUST cite at least one link_name from the
  evidence section above. Decisions citing zero links will be dropped.
- `substitution_path` comes from the necessity assessment substitutes listed
  in §2 above. Leave empty if no substitutes are listed.
- Prefer `monitor` only when no other action fits.
- If an INV is clearly on-track with strong evidence, emit `approve_as_is`.
- An empty decisions list is valid if no action is warranted right now."""


def _render_link_for_prompt(la: dict, link_financial: dict | None = None) -> str:
    """Render one link assessment dict as evidence input for the decision prompt.

    `link_financial` maps link_name → causal_model link dict for financial
    fields (dollars_at_risk, months_at_risk) that are not stored on assessment
    dicts in the new repo's InvestigationResult schema.
    """
    link_name = la.get("link_name", "") or la.get("link_id", "")
    findings = la.get("findings") or {}
    status = la.get("terminal_status") or findings.get("status") or la.get("status", "")
    confidence = la.get("confidence") or findings.get("confidence", "")
    evidence_refs = la.get("evidence_refs") or findings.get("evidence_refs", [])
    prose = la.get("prose", "")

    lk = (link_financial or {}).get(link_name, {})
    dollars = la.get("dollars_at_risk") or lk.get("dollars_at_risk", 0) or 0
    months = la.get("months_at_risk") or lk.get("months_at_risk", 0) or 0

    lines = [f"\n- **{link_name}** [status={status or '?'}, confidence={confidence or '?'}]"]
    if prose:
        lines.append(f"  Analysis: {prose[:400]}")
    if dollars:
        lines.append(f"  Dollars at risk: ${float(dollars):,.0f}")
    if months:
        lines.append(f"  Months at risk: {float(months):.1f}")
    if evidence_refs:
        lines.append(f"  Evidence refs: {', '.join(str(r) for r in evidence_refs[:6])}")
    return "\n".join(lines)


def _build_decisions_prompt(scope_id: str, scope_output: dict) -> str:
    """Goal-anchored prompt: §1 BoW goal → §2 necessity role → §3 evidence → §4 ask.

    Ported from old-repo decision_projection._build_decisions_prompt (§1-§4 structure).
    """
    inv_id = scope_output.get("inv_id", "")
    label = scope_output.get("label", scope_id)
    parts = [f"# Scope: {label}"]

    # ── §1. BoW strategic goal (goal-anchor for every decision) ──────
    parts.append("\n## What this Body of Work is trying to achieve")
    bow_context = scope_output.get("bow_context") or {}
    field_landscape = bow_context.get("field_landscape", "") if isinstance(bow_context, dict) else ""
    if field_landscape:
        parts.append(field_landscape[:1500])
    else:
        parts.append(
            f"(No BoWContext.field_landscape available for {scope_id}; "
            f"infer the goal from the scope label and investment titles below.)"
        )

    # ── §2. Per-INV portfolio role (necessity assessment) ────────────
    parts.append("\n## How each investment is meant to serve that goal")
    na_raw = scope_output.get("necessity_assessment") or ""
    na: dict = {}
    if na_raw:
        try:
            na = json.loads(na_raw) if isinstance(na_raw, str) else (na_raw if isinstance(na_raw, dict) else {})
        except Exception:
            na = {}

    facts = scope_output.get("investment_facts") or {}
    org = facts.get("org", "")
    title = facts.get("title", "")
    title_bit = f": {title}" if title else ""
    parts.append(f"\n**{inv_id}**{title_bit} (org: {org or '?'})")

    if na:
        parts.append(
            f"  Portfolio role: {na.get('portfolio_relationship', '') or 'unclear'}"
            f", differentiation={na.get('differentiation', '') or 'unknown'}"
            f", marginal_contribution={na.get('marginal_contribution', '') or 'unknown'}"
        )
        rationale = (na.get("differentiation_rationale", "") or "")[:300]
        if rationale:
            parts.append(f"  Why differentiated: {rationale}")
        counterfactual = (na.get("counterfactual_loss", "") or "")[:200]
        if counterfactual:
            parts.append(f"  If this didn't exist: {counterfactual}")
        substitutes = na.get("substitutes", []) or []
        if substitutes:
            parts.append(f"  Substitutes available: {', '.join(str(s) for s in substitutes[:3])}")
    else:
        parts.append("  (no necessity assessment available)")

    # ── §3. Evidence per INV (link assessments) ───────────────────────
    parts.append("\n## Evidence: is each investment delivering its role?")
    parts.append(
        "Each link below carries the analyst's per-link findings: "
        "prose analysis, evidence references, and financial risk estimates. "
        "Use these directly — do not re-derive."
    )

    link_assessments = scope_output.get("link_assessments", []) or []
    causal_model = scope_output.get("causal_model") or {}
    cm_links = causal_model.get("links", []) if isinstance(causal_model, dict) else []
    link_financial: dict = {lk.get("name", ""): lk for lk in cm_links if lk.get("name")}

    if link_assessments:
        parts.append(f"\n### {inv_id} link assessments")
        for la in link_assessments:
            parts.append(_render_link_for_prompt(la, link_financial))
    else:
        parts.append("  (no link assessments available)")

    # ── §4. The ask ───────────────────────────────────────────────────
    parts.append("\n## Your task")
    parts.append(_TASK_INSTRUCTIONS)

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Sanitization, scoring, and filtering
# ---------------------------------------------------------------------------


def _sanitize_candidate(candidate: DecisionCandidate) -> DecisionCandidate | None:
    """Return validated candidate or None if it fails basic sanity checks.

    Deduplicates triggering_link_ids (order-preserving) to prevent inflated
    corroboration_count when the LLM emits the same link name twice — mirrors
    the dict.fromkeys() dedup in old-repo decision_projection._sanitize_candidate.
    """
    dt = (candidate.decision_type or "").strip().lower()
    if dt not in DECISION_TYPE_VOCABULARY:
        logger.debug("Decision type '%s' not in vocabulary — dropping", dt)
        return None
    if not (candidate.recommended_action or "").strip():
        logger.debug("Empty recommended_action for type '%s' — dropping", dt)
        return None
    # Deduplicate triggering_link_ids while preserving insertion order.
    # Without this, duplicate link names inflate corroboration_count and can
    # falsely satisfy the §1a gate's corroboration >= 2 threshold.
    deduped_links = list(dict.fromkeys(
        str(x).strip() for x in (candidate.triggering_link_ids or [])
        if isinstance(x, str) and str(x).strip()
    ))
    if not deduped_links:
        logger.debug("Empty triggering_link_ids for type '%s' — dropping", dt)
        return None
    return candidate.model_copy(update={"triggering_link_ids": deduped_links})


def _compute_rank_score(candidate: DecisionCandidate) -> float:
    """Rank score: corroboration × materiality × log10(dollars+10) × urgency × evidence.

    Weight values copied from old-repo decision_projection._compute_rank_score:
      materiality: material=1.0, uncertain=0.5, not_material=0.0
                   (aliases: high→material, medium→uncertain, low→not_material)
      urgency:     immediate=1.0, quarterly=0.6, annual_cycle=0.3
                   (aliases: near_term→quarterly, medium_term→annual_cycle, long_term→annual_cycle)
      evidence:    high=1.0, moderate=0.7, low=0.3, insufficient=0.1
                   (confidence field; moderate=medium alias included)

    Unknown values fall back to the lowest non-zero bucket so missing fields
    don't promote a decision past those with explicit positive signals.
    """
    cor = float(max(1, candidate.corroboration_count))

    mat_map: dict[str, float] = {
        # Canonical OLD vocabulary
        "material":     1.0,
        "uncertain":    0.5,
        "not_material": 0.0,
        # NEW aliases (from old prompt vocabulary)
        "high":   1.0,   # high ≡ material
        "medium": 0.5,   # medium ≡ uncertain
        "low":    0.0,   # low ≡ not_material
    }
    urg_map: dict[str, float] = {
        # Canonical OLD vocabulary
        "immediate":    1.0,
        "quarterly":    0.6,
        "annual_cycle": 0.3,
        # Aliases for backward compatibility
        "near_term":    0.6,   # near_term ≡ quarterly
        "medium_term":  0.3,   # medium_term ≡ annual_cycle
        "long_term":    0.3,   # long_term ≡ annual_cycle
    }
    evd_map: dict[str, float] = {
        # OLD vocabulary (confidence / evidence_level field)
        "high":         1.0,
        "moderate":     0.7,
        "medium":       0.7,   # alias for moderate
        "low":          0.3,
        "insufficient": 0.1,
    }

    mat = mat_map.get((candidate.materiality or "").strip().lower(), 0.5)
    urg = urg_map.get((candidate.urgency or "").strip().lower(), 0.3)
    evd = evd_map.get((candidate.confidence or "").strip().lower(), 0.3)
    dollars = max(0.0, candidate.cost_impact_dollars)

    return cor * mat * max(math.log10(dollars + 10.0), 1.0) * urg * evd


def _section1a_gate(candidate: DecisionCandidate) -> bool:
    """§1a evidence gate.

    Passes if:
    - decision_type is in _THIN_EVIDENCE_DECISION_TYPES (always pass), OR
    - confidence != "low", OR
    - corroboration_count >= 2
    Returns True (pass) or False (reject).
    """
    dt = (candidate.decision_type or "").strip().lower()
    if dt in _THIN_EVIDENCE_DECISION_TYPES:
        return True
    if (candidate.confidence or "").lower() != "low":
        return True
    return candidate.corroboration_count >= 2


def _apply_caps(
    decisions: list[Decision],
    inv_id: str,
) -> list[Decision]:
    """Apply per-INV and per-scope caps.

    - Max DECISION_MAX_PER_INV per inv_id (decisions with empty inv_id do not count).
    - Max DECISION_MAX_PER_SCOPE total across the scope.
    Decisions are assumed pre-sorted by rank_score descending.
    """
    per_inv: dict[str, int] = {}
    result: list[Decision] = []

    for d in decisions:
        if len(result) >= DECISION_MAX_PER_SCOPE:
            break
        d_inv = d.inv_id or ""
        if d_inv:
            count = per_inv.get(d_inv, 0)
            if count >= DECISION_MAX_PER_INV:
                continue
            per_inv[d_inv] = count + 1
        result.append(d)

    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def project_decisions(
    scope_id: str,
    scope_output: dict,
    *,
    model: str = DEFAULT_SYNTHESIS_MODEL,
) -> dict:
    """Project actionable decisions from scope evidence.

    Returns a dict with {"scope_id": scope_id, "decisions": [<Decision dicts>]}.
    """
    inv_id: str = scope_output.get("inv_id", "")
    prompt = _build_decisions_prompt(scope_id, scope_output)

    try:
        projection: DecisionProjectionOutput = await acall_structured(
            prompt,
            system_msg=_SYSTEM_PROMPT,
            model=model,
            schema=DecisionProjectionOutput,
            max_tokens=DEFAULT_MAX_TOKENS,
        )
        candidates = projection.decisions
    except Exception as exc:
        logger.warning("project_decisions LLM failed for %s: %s", scope_id, exc)
        candidates = []

    # Sanitize, gate, rank, cap
    valid: list[tuple[float, Decision]] = []
    for candidate in candidates:
        sanitized = _sanitize_candidate(candidate)
        if sanitized is None:
            continue
        if not _section1a_gate(sanitized):
            logger.debug("§1a gate rejected decision type=%s for scope %s", sanitized.decision_type, scope_id)
            continue
        rank = _compute_rank_score(sanitized)
        decision = Decision(
            inv_id=inv_id,
            bow_ids=scope_output.get("bow_ids", []) or [],
            decision_type=sanitized.decision_type,
            recommended_action=sanitized.recommended_action,
            goal_link=sanitized.goal_link,
            substitution_path=sanitized.substitution_path,
            triggering_link_ids=sanitized.triggering_link_ids,
            triggering_evidence=sanitized.triggering_evidence,
            corroboration_count=sanitized.corroboration_count,
            cost_impact_dollars=sanitized.cost_impact_dollars,
            timeline_impact_months=sanitized.timeline_impact_months,
            confidence=sanitized.confidence,
            urgency=sanitized.urgency,
            materiality=sanitized.materiality,
            rank_score=rank,
        )
        valid.append((rank, decision))

    valid.sort(key=lambda x: -x[0])
    decisions = _apply_caps([d for _, d in valid], inv_id=inv_id)

    return {
        "scope_id": scope_id,
        "decisions": [d.to_dict() for d in decisions],
    }
