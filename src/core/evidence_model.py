"""Core evidence data model — dataclasses shared across all pipeline stages.

No LangGraph imports. Pure data containers used by rubric_evaluator,
investigation, science_investigator, causal_model, decision_projection,
and report_assembler. Worker nodes serialise these to dict before writing
to WorkflowState fan-out fields.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Supporting types
# ---------------------------------------------------------------------------


@dataclass
class Evidence:
    text: str = ""
    source_ref: str = ""
    doc_type: str = ""
    inv_id: str = ""
    file_id: str = ""
    page: int = 0
    score: float = 0.0


@dataclass
class NumericalFact:
    value: str = ""
    label: str = ""
    source_ref: str = ""
    inv_id: str = ""
    file_id: str = ""
    page: int = 0


@dataclass
class ScenarioResult:
    scenario: str = ""
    likelihood: str = ""
    impact: str = ""
    time_horizon: str = ""
    drivers: list[str] = field(default_factory=list)
    mitigation: str = ""


@dataclass
class ScienceFlag:
    assumption: str = ""
    status: str = ""            # "supported" | "challenged" | "insufficient_evidence" | "blocked"
    confidence: str = ""
    key_evidence: str = ""
    pivotal_because: str = ""
    recommendation: str = ""


# ---------------------------------------------------------------------------
# Causal model types (produced by causal_model.extract_causal_model)
# ---------------------------------------------------------------------------


@dataclass
class CausalLink:
    name: str = ""
    from_stage: str = ""
    to_stage: str = ""
    mechanism: str = ""
    assumptions: list[str] = field(default_factory=list)
    metrics: list[str] = field(default_factory=list)
    failure_modes: list[str] = field(default_factory=list)
    dollars_at_risk: float = 0.0
    months_at_risk: float = 0.0


@dataclass
class ScoredAssumption:
    assumption: str = ""
    causal_link: str = ""
    source_type: str = ""
    if_wrong: str = ""
    consequence: str = ""
    uncertainty: str = ""
    risk_rank: int = 0
    investigation_question: str = ""


@dataclass
class CausalModel:
    inv_id: str = ""
    theory_of_change: str = ""
    links: list[CausalLink] = field(default_factory=list)
    assumptions: list[ScoredAssumption] = field(default_factory=list)
    outcome_statement: str = ""


# ---------------------------------------------------------------------------
# Investment-level types
# ---------------------------------------------------------------------------


@dataclass
class InvestmentFacts:
    inv_id: str = ""
    org: str = ""
    title: str = ""
    approved_amount: float = 0.0
    paid_amount: float = 0.0
    outstanding_balance: float = 0.0
    grant_start: str = ""
    grant_end: str = ""
    duration_months: int = 0
    elapsed_months: int = 0
    remaining_months: int = 0
    burn_rate_per_month: float = 0.0
    runway_months: float = 0.0
    disbursement_pct: float = 0.0
    monthly_budget: float = 0.0
    budget_vs_actual_ratio: float = 0.0
    pct_time_elapsed: float = 0.0
    time_vs_spend_ratio: float = 0.0
    is_overrun: bool = False
    months_overrun: int = 0
    enrollment_target: int = 0
    enrollment_actual: int = 0
    sites_planned: int = 0
    sites_active: int = 0
    doc_count: int = 0
    latest_doc_date: str = ""
    months_since_latest: int = 0
    sources: dict[str, str] = field(default_factory=dict)
    unverified_fields: list[str] = field(default_factory=list)


@dataclass
class InvestmentEvidencePack:
    """Rubric evaluation output for one investment (Phase 3.1 worker output)."""
    inv_id: str = ""
    scope_id: str = ""
    timeline: dict = field(default_factory=dict)    # serialised InvestmentTimeline
    chunks: list[dict] = field(default_factory=list)
    source_index: list[dict] = field(default_factory=list)
    scoring: dict = field(default_factory=dict)
    local_scores: dict = field(default_factory=dict)
    visual_images: list[bytes] = field(default_factory=list)
    facts: Optional[InvestmentFacts] = None
    causal_model: Optional[CausalModel] = None

    def to_dict(self) -> dict:
        import dataclasses
        return dataclasses.asdict(self)


# ---------------------------------------------------------------------------
# Link assessment (Phase 3.4 worker output)
# ---------------------------------------------------------------------------


@dataclass
class LinkAssessment:
    """Investigation result for one causal link."""
    link_name: str = ""
    link_id: str = ""
    inv_id: str = ""
    bow_id: str = ""
    scope_id: str = ""
    status: str = ""
    confidence: str = ""
    evidence_level: str = ""
    evidence_basis: str = ""
    evidence_expectation: str = ""
    team_prior: str = ""
    agrees_with_team: bool = True
    evidence_for: str = ""
    evidence_against: str = ""
    evidence_refs: list[str] = field(default_factory=list)
    metric_actual: str = ""
    metric_expected: str = ""
    gap_description: str = ""
    scenarios: list[ScenarioResult] = field(default_factory=list)
    forecast: str = ""
    dollars_at_risk: float = 0.0
    months_at_risk: float = 0.0
    upstream_dependencies: list[str] = field(default_factory=list)
    downstream_impacts: list[str] = field(default_factory=list)
    compounds_with: list[str] = field(default_factory=list)
    root_cause: str = ""
    unresolved_questions: list[str] = field(default_factory=list)
    recommended_research: Any = None
    consequence_pathway: str = ""
    magnitude_reasoning: str = ""
    likelihood_reasoning: str = ""
    materiality: str = ""
    leadership_options: list[str] = field(default_factory=list)
    annotated_excerpts: list[dict] = field(default_factory=list)
    source_index: list[dict] = field(default_factory=list)
    numerical_facts: list[NumericalFact] = field(default_factory=list)
    extracted_quotes: list[dict] = field(default_factory=list)
    iterations: int = 0
    web_searches: int = 0
    documents_read: list[str] = field(default_factory=list)
    tool_log: list[dict] = field(default_factory=list)
    section_reads: list[dict] = field(default_factory=list)
    usable_read_count: int = 0
    empty_read_count: int = 0
    searches_before_first_usable_read: int = 0
    first_usable_read_tool: str = ""
    used_fallback_batch: bool = False
    fallback_usable_reads_added: int = 0
    had_substantive_without_usable_read: bool = False

    def to_dict(self) -> dict:
        import dataclasses
        return dataclasses.asdict(self)


# ---------------------------------------------------------------------------
# Science investigation types (Phase 3.5d)
# ---------------------------------------------------------------------------


@dataclass
class ScienceQuestion:
    assumption: str = ""
    causal_link: str = ""
    pivotal_because: str = ""
    current_evidence: str = ""
    search_queries: list[str] = field(default_factory=list)


@dataclass
class ScienceInvestigationResult:
    """Phase 3.5d worker output for one science assumption."""
    question_index: int = 0
    chunks: list[dict] = field(default_factory=list)
    asta_hits: list[Any] = field(default_factory=list)
    blocked_items: list[str] = field(default_factory=list)
    iterations: int = 0
    asta_calls: int = 0
    web_calls: int = 0
    terminal_status: str = ""
    elapsed_s: float = 0.0
    answer: str = ""       # LLM synthesis from ScienceActionsOutput.answer
    question: str = ""     # The assumption text passed in
    scope_id: str = ""     # For grouping in collect_science_results

    def to_dict(self) -> dict:
        import dataclasses
        return dataclasses.asdict(self)


# ---------------------------------------------------------------------------
# Decision (Phase 3.8 worker output)
# ---------------------------------------------------------------------------


@dataclass
class Decision:
    """Action-impact decision projected from scope evidence."""
    inv_id: str = ""
    bow_id: str = ""
    bow_ids: list[str] = field(default_factory=list)
    decision_type: str = ""         # "accelerate" | "halt" | "pivot" | "monitor" | "close"
    recommended_action: str = ""
    goal_link: str = ""
    substitution_path: str = ""
    triggering_link_ids: list[str] = field(default_factory=list)
    triggering_evidence: list[str] = field(default_factory=list)
    aggregate_evidence_level: str = ""
    corroboration_count: int = 0
    cost_impact_dollars: float = 0.0
    timeline_impact_months: float = 0.0
    confidence: str = ""
    urgency: str = ""
    materiality: str = ""
    rank_score: float = 0.0

    def to_dict(self) -> dict:
        import dataclasses
        return dataclasses.asdict(self)


# ---------------------------------------------------------------------------
# Finding / AdjudicatedFinding
# ---------------------------------------------------------------------------


@dataclass
class Finding:
    id: str = ""
    statement: str = ""
    finding_type: str = ""          # "risk" | "strength" | "gap" | "contradiction" | "pattern"
    severity: str = ""
    confidence: str = ""
    confidence_rationale: str = ""
    evidence_for: list[Evidence] = field(default_factory=list)
    evidence_against: list[Evidence] = field(default_factory=list)
    affected_bows: list[str] = field(default_factory=list)
    affected_investments: list[str] = field(default_factory=list)
    addressable_by: str = ""


@dataclass
class AdjudicatedFinding:
    finding_id: str = ""
    inv_id: str = ""
    claim: str = ""
    severity: str = ""
    confidence: str = ""
    rubric_dimensions: list[str] = field(default_factory=list)
    gap_ids: list[str] = field(default_factory=list)
    highlight_ids: list[str] = field(default_factory=list)
    evidence_refs: list[str] = field(default_factory=list)
    unresolved: str = ""
    decision_implication: str = ""
    classification: str = ""
    source_data_age_months: Optional[int] = None


@dataclass
class ResearchThread:
    id: str = ""
    hypothesis: str = ""
    question: str = ""
    priority: str = ""
    bows: list[str] = field(default_factory=list)
    investments: list[str] = field(default_factory=list)
    reading_plan: list[str] = field(default_factory=list)
    web_searches: list[str] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)
    documents_read: list[str] = field(default_factory=list)
    web_results_used: int = 0
    deep_research_needed: list[dict] = field(default_factory=list)
    confidence: str = ""
    confidence_rationale: str = ""
    gaps: list[str] = field(default_factory=list)
    status: str = "pending"
    rounds_completed: int = 0
    section_draft: Any = None


# ---------------------------------------------------------------------------
# Scope-level aggregation (produced by collect/reducer nodes)
# ---------------------------------------------------------------------------


@dataclass
class BOWContext:
    bow_id: str = ""
    bow_name: str = ""
    strategy_summary: str = ""
    inv_ids: list[str] = field(default_factory=list)
    portfolio_patterns: list[str] = field(default_factory=list)


@dataclass
class ScopeOutput:
    """Aggregated output for one analysis scope (investment × BOW thread)."""
    scope_id: str = ""
    label: str = ""
    inv_id: str = ""
    bow_ids: list[str] = field(default_factory=list)
    timeline: Optional[dict] = None             # serialised ScopeTimeline
    evidence_packs: list[InvestmentEvidencePack] = field(default_factory=list)
    link_assessments: list[LinkAssessment] = field(default_factory=list)
    science_flags: list[ScienceFlag] = field(default_factory=list)
    adjudicated_findings: list[AdjudicatedFinding] = field(default_factory=list)
    decisions: list[Decision] = field(default_factory=list)
    causal_model: Optional[CausalModel] = None
    bow_context: Optional[BOWContext] = None
    synthesis: str = ""
    critique: str = ""
    gaps: str = ""
    necessity_assessment: str = ""
    section_body: str = ""          # rendered markdown section for this scope
    investment_sections: dict[str, str] = field(default_factory=dict)  # inv_id → markdown
    scenario_results: list[ScenarioResult] = field(default_factory=list)
    scope_decisions: list[dict] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        import dataclasses
        return dataclasses.asdict(self)


# ---------------------------------------------------------------------------
# Investigation result (produced by investigation.run_investigation)
# ---------------------------------------------------------------------------


@dataclass
class InvestigationResult:
    findings: dict = field(default_factory=dict)
    overall_assessment: dict = field(default_factory=dict)
    prose: str = ""
    tool_log: list[dict] = field(default_factory=list)
    source_index: list[dict] = field(default_factory=list)
    documents_read: list[str] = field(default_factory=list)
    section_reads: list[dict] = field(default_factory=list)
    annotated_excerpts: list[dict] = field(default_factory=list)  # with credibility_tier
    iterations: int = 0
    total_chunks_retrieved: int = 0
    web_searches: int = 0
    elapsed_seconds: float = 0.0
    model: str = ""
    terminal_status: str = ""
    # Routing fields — required by collect_link_assessments to merge results into the
    # correct scope's link_assessments list. Matches ScienceInvestigationResult pattern.
    scope_id: str = ""
    link_id: str = ""
    inv_id: str = ""

    def to_dict(self) -> dict:
        import dataclasses
        return dataclasses.asdict(self)
