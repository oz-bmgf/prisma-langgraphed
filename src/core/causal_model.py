"""Causal model extraction, assumption risk-ranking, and consequence forecasting.

Four public functions form a sequential pipeline:
  extract_causal_model   → CausalModel  (LLM extracts ToC chain from chunks)
  rank_assumptions       → CausalModel  (LLM classifies + _RISK_MATRIX scores)
  forecast_consequences  → CausalModel  (LLM forecasts $ / months per link)
  make_investigation_claims → list[dict] (builds investigation task dicts)

No LangGraph imports. No hardcoded model names or thresholds.
"""
from __future__ import annotations

import logging
import math
from typing import Any

from src.config import (
    CAUSAL_MAX_TOKENS,
    DEFAULT_MAX_TOKENS,
    DEFAULT_SYNTHESIS_MODEL,
)
from src.core.evidence_model import CausalLink, CausalModel, ScoredAssumption
from src.core.llm_utils import acall_llm, acall_structured
from src.core.output_schemas import (
    CausalModelExtraction,
    ConsequenceForecast,
    ForecastOutput,
    RankedAssumptionsOutput,
)
from src.prompts.causal_prompts import (
    ASSUMPTION_RANKING_PROMPT,
    ASSUMPTION_RANKING_SYSTEM,
    CAUSAL_EXTRACTION_PROMPT,
    CAUSAL_EXTRACTION_SYSTEM,
    CONSEQUENCE_FORECAST_PROMPT,
    CONSEQUENCE_FORECAST_SYSTEM,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Risk matrix — deterministic (consequence, uncertainty) → (label, sort_key)
# sort_key: 1 = most critical, 9 = negligible
# ---------------------------------------------------------------------------

_RISK_MATRIX: dict[tuple[str, str], tuple[str, int]] = {
    # Copied verbatim from old-repo causal_model.py — do not modify sort keys
    # without also updating the equivalent in prisma-ai-review.
    # sort_key: 1 = most critical, 9 = negligible
    ("terminal", "high"):     ("critical", 1),
    ("terminal", "moderate"): ("critical", 2),
    ("major",    "high"):     ("high",     3),
    ("terminal", "low"):      ("high",     4),  # well-understood terminal risk stays HIGH
    ("major",    "moderate"): ("medium",   5),  # was wrongly ("high", 4) in new repo
    ("minor",    "high"):     ("medium",   6),  # was wrongly ("medium", 5) in new repo
    ("major",    "low"):      ("low",      7),
    ("minor",    "moderate"): ("low",      8),
    ("minor",    "low"):      ("low",      9),  # label was wrongly "negligible" in new repo
}

# Keywords that flag an assumption as science-heavy (add web-search suggestions).
_SCIENCE_KEYWORDS: frozenset[str] = frozenset({
    "efficacy",
    "effectiveness",
    "rct",
    "clinical trial",
    "vaccine",
    "immunogenicity",
    "antibody",
    "immune",
    "pathogen",
    "epidemiology",
    "prevalence",
    "incidence",
    "mortality",
    "morbidity",
    "seroprevalence",
    "protection",
    "dosing",
    "pharmacokinetics",
})

_MAX_CHUNK_TEXT_CHARS = 80_000
_MAX_CHUNKS = 50


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def extract_causal_model(
    scope: dict,
    model: str = DEFAULT_SYNTHESIS_MODEL,
    config: object = None,
) -> CausalModel:
    """Extract a full theory-of-change causal model from scope evidence.

    Reads evidence chunks from scope["evidence_packs"], calls the LLM to
    extract the causal chain, then calls rank_assumptions and
    forecast_consequences before returning a fully populated CausalModel.
    """
    inv_id: str = scope.get("inv_id", "")

    chunks: list[dict] = []
    scoring: dict[str, Any] = {}
    for pack in scope.get("evidence_packs", []):
        if isinstance(pack, dict):
            chunks.extend(pack.get("chunks", []))
            scoring.update(pack.get("scoring", {}))

    chunk_text = "\n\n".join(
        c.get("text", "") for c in chunks[:_MAX_CHUNKS]
    )[:_MAX_CHUNK_TEXT_CHARS]

    prompt = CAUSAL_EXTRACTION_PROMPT.format(
        inv_id=inv_id,
        chunk_text=chunk_text or "(no evidence chunks available — use general knowledge about this investment type)",
    )

    try:
        extracted: CausalModelExtraction = await acall_structured(
            prompt,
            system_msg=CAUSAL_EXTRACTION_SYSTEM,
            model=model,
            schema=CausalModelExtraction,
            max_tokens=CAUSAL_MAX_TOKENS,  # 32000 — matches OLD; large ToC chains need room
        )
        links = [
            CausalLink(
                name=lk.name,
                from_stage=lk.from_stage,
                to_stage=lk.to_stage,
                mechanism=lk.mechanism,
                assumptions=lk.assumptions,
                metrics=lk.metrics,
                failure_modes=lk.failure_modes,
            )
            for lk in extracted.links
        ]
        cm = CausalModel(
            inv_id=inv_id,
            theory_of_change=extracted.theory_of_change,
            links=links,
            outcome_statement=extracted.outcome_statement,
        )
    except Exception as exc:
        logger.warning("extract_causal_model LLM call failed for %s: %s", inv_id, exc)
        cm = CausalModel(inv_id=inv_id)

    if not cm.links:
        logger.info("extract_causal_model: no links extracted for %s, using minimal fallback", inv_id)
        cm.links = [
            CausalLink(
                name="Intervention→Outcome",
                from_stage="ACTIVITIES",
                to_stage="OUTCOMES",
                mechanism="Investment activities produce measurable outcomes via program delivery.",
                assumptions=["Grantee executes activities as planned"],
                metrics=["milestone completion rate"],
                failure_modes=["execution failure", "context change"],
            )
        ]

    cm = await rank_assumptions(cm, model=model)

    timeline: dict = scope.get("scope_timelines", {}).get(scope.get("scope_id", ""), {})
    cm = await forecast_consequences(cm, scoring=scoring, timeline=timeline, model=model)

    return cm


async def rank_assumptions(
    cm: CausalModel,
    model: str = DEFAULT_SYNTHESIS_MODEL,
) -> CausalModel:
    """Classify each link assumption by consequence/uncertainty and sort by risk.

    Uses LLM to classify consequence (terminal|major|minor) and uncertainty
    (high|moderate|low) for each assumption, then applies _RISK_MATRIX
    deterministically to assign risk_rank (1=most critical). Returns a new
    CausalModel with assumptions sorted ascending by risk_rank.
    """
    all_assumptions: list[tuple[str, str]] = []
    for link in cm.links:
        for assumption_text in link.assumptions:
            all_assumptions.append((assumption_text, link.name))

    if not all_assumptions:
        return cm

    assumptions_text = "\n".join(
        f"{i+1}. [{link_name}] {text}"
        for i, (text, link_name) in enumerate(all_assumptions)
    )

    prompt = ASSUMPTION_RANKING_PROMPT.format(
        inv_id=cm.inv_id,
        theory_of_change=cm.theory_of_change[:500] if cm.theory_of_change else "(unknown)",
        assumptions_text=assumptions_text,
    )

    try:
        ranked: RankedAssumptionsOutput = await acall_structured(
            prompt,
            system_msg=ASSUMPTION_RANKING_SYSTEM,
            model=model,
            schema=RankedAssumptionsOutput,
            max_tokens=DEFAULT_MAX_TOKENS,
        )
        scored: list[ScoredAssumption] = []
        for ar in ranked.assumptions:
            consequence = ar.consequence.strip().lower()
            uncertainty = ar.uncertainty.strip().lower()
            _label, sort_key = _RISK_MATRIX.get(
                (consequence, uncertainty), ("unknown", 99)
            )
            scored.append(
                ScoredAssumption(
                    assumption=ar.assumption,
                    causal_link=ar.causal_link,
                    consequence=consequence,
                    uncertainty=uncertainty,
                    if_wrong=ar.if_wrong,
                    investigation_question=ar.investigation_question,
                    risk_rank=sort_key,
                )
            )
        scored.sort(key=lambda a: a.risk_rank)
        cm.assumptions = scored
    except Exception as exc:
        logger.warning("rank_assumptions LLM call failed for %s: %s", cm.inv_id, exc)
        cm.assumptions = [
            ScoredAssumption(
                assumption=text,
                causal_link=link_name,
                consequence="major",
                uncertainty="moderate",
                risk_rank=4,
                investigation_question=f"What evidence supports or refutes: {text}?",
            )
            for text, link_name in all_assumptions
        ]

    return cm


async def forecast_consequences(
    cm: CausalModel,
    scoring: dict[str, Any],
    timeline: dict[str, Any],
    model: str = DEFAULT_SYNTHESIS_MODEL,
) -> CausalModel:
    """Forecast dollar and month exposure per causal link.

    Basel-EAD constraint: dollars_at_risk per link is capped at approved_amount
    (the principal can only be lost once). Uses max() aggregation when multiple
    forecasts address the same link — never sum.
    """
    if not cm.links:
        return cm

    approved_amount: float = float(scoring.get("approved_amount", 0) or 0)
    duration_months: int = int(timeline.get("duration_months", 24) or 24)

    links_text = "\n".join(
        f"- {lk.name}: {lk.from_stage}→{lk.to_stage}. Mechanism: {lk.mechanism}. "
        f"Failure modes: {', '.join(lk.failure_modes[:2]) or 'unknown'}"
        for lk in cm.links
    )

    prompt = CONSEQUENCE_FORECAST_PROMPT.format(
        inv_id=cm.inv_id,
        approved_amount=approved_amount,
        duration_months=duration_months,
        links_text=links_text,
    )

    dollars_by_link: dict[str, float] = {}
    months_by_link: dict[str, float] = {}

    try:
        forecast_result: ForecastOutput = await acall_structured(
            prompt,
            system_msg=CONSEQUENCE_FORECAST_SYSTEM,
            model=model,
            schema=ForecastOutput,
            max_tokens=DEFAULT_MAX_TOKENS,
        )
        for fc in forecast_result.forecasts:
            name = fc.link_name
            dollars = max(0.0, fc.dollars_at_risk)
            months = max(0.0, fc.months_at_risk)
            if approved_amount > 0:
                dollars = min(dollars, approved_amount)
            months = min(months, float(duration_months))
            # max() aggregation — never sum across multiple forecasts for same link
            dollars_by_link[name] = max(dollars_by_link.get(name, 0.0), dollars)
            months_by_link[name] = max(months_by_link.get(name, 0.0), months)
    except Exception as exc:
        logger.warning("forecast_consequences LLM call failed for %s: %s", cm.inv_id, exc)

    for link in cm.links:
        link.dollars_at_risk = dollars_by_link.get(link.name, 0.0)
        link.months_at_risk = months_by_link.get(link.name, 0.0)

    return cm


def _make_search_suggestions(assumption: ScoredAssumption) -> list[str]:
    """Generate targeted search queries for this assumption.

    Mirrors old-repo causal_model._make_search_suggestions. Science-heavy
    assumptions get ASTA/web queries; financial assumptions get investment queries.
    """
    text = assumption.assumption.lower()
    question = assumption.investigation_question or assumption.assumption
    suggestions: list[str] = []

    if any(kw in text for kw in _SCIENCE_KEYWORDS):
        suggestions.append(f"search_asta: {question}")
        suggestions.append(f"search_web: peer-reviewed evidence on {question}")
    if any(kw in text for kw in ("financial", "budget", "disbursement", "funding", "runway")):
        suggestions.append(f"search_investment: budget actuals disbursement {question}")
    if any(kw in text for kw in ("partner", "grantee", "capacity", "delivery", "milestone")):
        suggestions.append(f"search_investment: {question}")
    if any(kw in text for kw in ("policy", "regulation", "government", "ministry")):
        suggestions.append(f"search_policy: {question}")
        suggestions.append(f"search_web: {question}")

    if not suggestions:
        # Generic fallback — always produce at least one suggestion
        suggestions.append(f"search_investment: {question}")

    return suggestions[:4]


def make_investigation_claims(
    cm: CausalModel,
    bow_id: str,
    inv_id: str,
) -> list[dict]:
    """Build investigation task dicts from the ranked assumption list.

    Each dict is a Send payload for the LangGraph fan-out.
    Fields added to match old-repo InvestigationClaim:
      priority         — terminal→critical, major→important, else→exploratory
      suggested_searches — targeted queries from _make_search_suggestions()
    task_id format: "{inv_id}-assumption-{i+1:03d}"
    """
    claims: list[dict] = []
    inv_id = inv_id or cm.inv_id

    _PRIORITY_MAP = {
        "terminal": "critical",
        "major":    "important",
    }

    for i, assumption in enumerate(cm.assumptions):
        text_lower = assumption.assumption.lower()
        is_science = any(kw in text_lower for kw in _SCIENCE_KEYWORDS)

        priority = _PRIORITY_MAP.get(assumption.consequence, "exploratory")
        suggested_searches = _make_search_suggestions(assumption)

        claim: dict = {
            "task_id": f"{inv_id}-assumption-{i+1:03d}",
            "inv_id": inv_id,
            "bow_id": bow_id,
            "assumption": assumption.assumption,
            "causal_link": assumption.causal_link,
            "consequence": assumption.consequence,
            "uncertainty": assumption.uncertainty,
            "if_wrong": assumption.if_wrong,
            "investigation_question": assumption.investigation_question,
            "risk_rank": assumption.risk_rank,
            "priority": priority,
            "suggested_searches": suggested_searches,
        }
        if is_science:
            claim["web_search_hint"] = (
                f"Search for peer-reviewed evidence on: {assumption.investigation_question}"
            )
        claims.append(claim)

    return claims
