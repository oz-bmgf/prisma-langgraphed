"""Decision projection — evidence-weighted action decisions from scope output.

project_decisions() drives the full pipeline:
  1. Build a prompt from scope link assessments and science flags
  2. Call LLM with DecisionProjectionOutput schema
  3. Sanitize each candidate (validate type, require recommended_action + triggering_links)
  4. Filter via §1a evidence gate (low-evidence decisions need thin type OR corroboration≥2)
  5. Rank by rank score: corroboration × materiality × log10(dollars+10) × urgency × evidence
  6. Apply caps: max DECISION_MAX_PER_INV per INV, max DECISION_MAX_PER_SCOPE total

DECISION_TYPE_VOCABULARY and _THIN_EVIDENCE_DECISION_TYPES are the two
canonical sets used by the §1a gate.

No LangGraph imports.
"""
from __future__ import annotations

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
CONTEXT: You are projecting actionable portfolio management decisions from evidence \
gathered during a Gates Foundation investment review.

For each significant finding from link assessments and science flags, recommend \
ONLY valid decision types. Return ONLY valid JSON. No prose outside JSON.

VALID DECISION TYPES (exactly one per decision):
approve_with_conditions, defer_pending_data, request_progress_packet,
extend_no_cost, supplement, redirect_funds, terminate_unless_resolved,
escalate_to_leadership, escalate_to_partner, schedule_review, validate_assumption,
align_with_strategy_team, approve_as_is, monitor, decommission_layer

RULES:
1. Each decision must have: non-empty recommended_action, non-empty triggering_link_ids.
2. corroboration_count = number of INDEPENDENT evidence sources (not just citation count).
3. urgency: immediate (action needed now) | near_term (next 30 days) | \
   medium_term (next quarter) | long_term (strategic planning horizon).
4. materiality: high (program-critical) | medium (efficiency) | low (informational).
5. Focus on the 3-8 highest-impact decisions. Quality over quantity.
"""


def _build_decisions_prompt(scope_id: str, scope_output: dict) -> str:
    link_assessments = scope_output.get("link_assessments", []) or []
    science_flags = scope_output.get("science_flags", []) or []
    synthesis = scope_output.get("synthesis", "") or ""
    gaps = scope_output.get("gaps", "") or ""
    inv_id = scope_output.get("inv_id", "")

    la_lines = []
    for la in link_assessments[:10]:
        link_id = la.get("link_id", "") or la.get("link_name", "")
        status = la.get("status", "")
        confidence = la.get("confidence", "")
        dollars = la.get("dollars_at_risk", 0)
        prose = (la.get("prose", "") or la.get("findings", {}).get("status", ""))[:200]
        la_lines.append(f"  [{link_id}] status={status} confidence={confidence} dollars={dollars:,.0f}: {prose}")

    sf_lines = []
    for sf in science_flags[:5]:
        ts = sf.get("terminal_status", "")
        q = sf.get("question", sf.get("answer", ""))[:100]
        sf_lines.append(f"  terminal={ts}: {q}")

    return (
        f"Scope: {scope_id}\nInvestment: {inv_id}\n\n"
        f"LINK ASSESSMENTS:\n" + ("\n".join(la_lines) or "  (none)") + "\n\n"
        f"SCIENCE FLAGS:\n" + ("\n".join(sf_lines) or "  (none)") + "\n\n"
        f"SYNTHESIS:\n{synthesis[:600]}\n\n"
        f"GAPS:\n{gaps[:300]}\n\n"
        "Project actionable decisions for leadership review."
    )


# ---------------------------------------------------------------------------
# Sanitization, scoring, and filtering
# ---------------------------------------------------------------------------


def _sanitize_candidate(candidate: DecisionCandidate) -> DecisionCandidate | None:
    """Return validated candidate or None if it fails basic sanity checks."""
    dt = (candidate.decision_type or "").strip().lower()
    if dt not in DECISION_TYPE_VOCABULARY:
        logger.debug("Decision type '%s' not in vocabulary — dropping", dt)
        return None
    if not (candidate.recommended_action or "").strip():
        logger.debug("Empty recommended_action for type '%s' — dropping", dt)
        return None
    if not candidate.triggering_link_ids:
        logger.debug("Empty triggering_link_ids for type '%s' — dropping", dt)
        return None
    return candidate


def _compute_rank_score(candidate: DecisionCandidate) -> float:
    """Rank score: corroboration × materiality × log10(dollars+10) × urgency × evidence.

    Higher score = higher priority. Formula mirrors _compute_rank_score in old
    decision_projection.py. Scalar mappings:
      materiality: high=3, medium=2, low=1
      urgency: immediate=4, near_term=3, medium_term=2, long_term=1
      confidence (evidence quality): high=3, medium=2, low=1
    """
    cor = float(max(1, candidate.corroboration_count))
    mat_map = {"high": 3.0, "medium": 2.0, "low": 1.0}
    urg_map = {"immediate": 4.0, "near_term": 3.0, "medium_term": 2.0, "long_term": 1.0}
    evd_map = {"high": 3.0, "medium": 2.0, "low": 1.0}

    mat = mat_map.get((candidate.materiality or "").lower(), 1.0)
    urg = urg_map.get((candidate.urgency or "").lower(), 1.0)
    evd = evd_map.get((candidate.confidence or "").lower(), 1.0)
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
