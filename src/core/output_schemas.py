"""Pydantic v2 structured output schemas for NQPR pipeline LLM calls.

Usage with acall_structured:
    result = await acall_structured(prompt, model=model, schema=ScopesOutput)
    # result is a validated ScopesOutput instance
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class OrientationOutput(BaseModel):
    portfolio_summary: str = Field(description="High-level summary of the portfolio")
    key_themes: list[str] = Field(description="Major thematic areas across investments")
    recommended_focus_areas: list[str] = Field(description="Areas warranting deep analysis")


class ScopeDesign(BaseModel):
    scope_id: str
    scope_label: str
    bow_ids: list[str]
    inv_ids: list[str]
    research_questions: list[str]


class ScopesOutput(BaseModel):
    scopes: list[ScopeDesign]


class CausalLink(BaseModel):
    link_id: str
    from_node: str
    to_node: str
    assumption: str
    uncertainty: float = Field(ge=0.0, le=1.0)
    consequence: float = Field(ge=0.0, le=1.0)


class CausalModelOutput(BaseModel):
    inv_id: str
    links: list[CausalLink]
    dollars_by_link: dict[str, float]
    months_by_link: dict[str, float]


class FindingSynthesis(BaseModel):
    scope_id: str
    key_findings: list[str]
    supporting_evidence: list[str]
    confidence: str = Field(description="high | medium | low")
    gaps: list[str]


class NecessityAssessmentOutput(BaseModel):
    link_id: str
    necessity_score: float = Field(ge=0.0, le=1.0)
    rationale: str


class DecisionOutput(BaseModel):
    decision_type: str
    description: str
    rationale: str
    affected_inv_ids: list[str]
    confidence: str = Field(description="high | medium | low")


class ResearchPlanItem(BaseModel):
    id: str
    type: str = Field(description="slr | lbd | deep_web | edison | internal")
    query: str
    priority: str = Field(description="critical | important | nice_to_have")
    linked_scope: str


class ResearchPlanOutput(BaseModel):
    tasks: list[ResearchPlanItem]


class EdisonRewrittenQuery(BaseModel):
    original_query: str
    rewritten_query: str
    rationale: str


# ── Tool call trace schemas ────────────────────────────────────────────────────

from typing import Literal


class ToolCallTrace(BaseModel):
    """Base metadata for any external tool call."""
    tool_name: str
    called_at: str              # ISO-8601 UTC timestamp
    duration_ms: int            # wall-clock time
    success: bool
    error_message: str | None = None


class AstaSearchTrace(ToolCallTrace):
    tool_name: Literal["search_asta"] = "search_asta"
    query: str
    rewritten_query: str | None = None
    result_count: int
    top_paper_ids: list[str]    # first 5 Semantic Scholar IDs
    top_titles: list[str]       # first 5 paper titles
    index_used: str = "semantic_scholar"


class SLRSearchTrace(ToolCallTrace):
    tool_name: Literal["slr_worker"] = "slr_worker"
    query: str
    task_id: str
    linked_scope: str
    result_count: int
    top_source_urls: list[str]  # first 5 URLs or identifiers
    agent_model: str = ""


class LBDSearchTrace(ToolCallTrace):
    tool_name: Literal["lbd_worker"] = "lbd_worker"
    query: str
    task_id: str
    linked_scope: str
    result_count: int
    concepts_discovered: list[str]  # first 10


class DeepWebTrace(ToolCallTrace):
    tool_name: Literal["deep_web_worker"] = "deep_web_worker"
    query: str
    task_id: str
    model_used: str = ""
    result_summary_chars: int   # length of result text
    sources_cited: list[str]    # URLs extracted from result


class EdisonSearchTrace(ToolCallTrace):
    tool_name: Literal["edison_worker"] = "edison_worker"
    original_query: str
    rewritten_query: str | None
    task_id: str
    result_count: int
    top_paper_ids: list[str]


class WebSearchTrace(ToolCallTrace):
    tool_name: Literal["search_web"] = "search_web"
    query: str
    inv_id: str | None = None
    result_count: int
    top_urls: list[str]         # first 5 URLs


class CodeInterpreterTrace(ToolCallTrace):
    tool_name: Literal["compute"] = "compute"
    code_snippet: str           # first 200 chars of code/question run
    output_type: str = "text"   # "number" | "dataframe" | "text"
    output_summary: str         # first 200 chars of output


class CollectionSearchTrace(ToolCallTrace):
    tool_name: str = "search_collection"  # may be search_collection, search_investment, search_portfolio
    query: str
    backend: str = "local"      # "local" | "qdrant" | "azure"
    inv_id_filter: str | None = None
    bow_id_filter: str | None = None
    top_k: int
    result_count: int
    top_chunk_ids: list[str]    # first 5 chunk_ids returned


class InvestigationToolTrace(BaseModel):
    """Aggregated trace for one full investigation loop."""
    inv_id: str
    scope_id: str
    link_id: str | None = None
    total_tool_calls: int
    tool_call_breakdown: dict[str, int]  # {tool_name: call_count}
    asta_called: bool
    web_search_called: bool
    compute_called: bool
    terminal_status: str        # "sufficient" | "insufficient" | "error"
    iterations_used: int


# ── Causal model extraction schemas ──────────────────────────────────────────


class CausalLinkSchema(BaseModel):
    name: str = Field(description="Short descriptive name, e.g. 'Training→CapacityGain'")
    from_stage: str = Field(description="Upstream stage: FUNDING|ACTIVITIES|OUTPUTS|OUTCOMES|IMPACT")
    to_stage: str = Field(description="Downstream stage: FUNDING|ACTIVITIES|OUTPUTS|OUTCOMES|IMPACT")
    mechanism: str = Field(description="How the upstream stage causes the downstream stage")
    assumptions: list[str] = Field(default=[], description="Key assumptions required for this link")
    metrics: list[str] = Field(default=[], description="Observable indicators of link health")
    failure_modes: list[str] = Field(default=[], description="Ways this link could break")


class CausalModelExtraction(BaseModel):
    theory_of_change: str = Field(description="One-paragraph theory of change")
    outcome_statement: str = Field(description="Primary intended outcome of the investment")
    links: list[CausalLinkSchema] = Field(description="3-7 causal links in the theory of change chain")


class AssumptionRisk(BaseModel):
    assumption: str = Field(description="The assumption text")
    causal_link: str = Field(description="Name of the associated causal link")
    consequence: str = Field(description="terminal | major | minor — severity if assumption fails")
    uncertainty: str = Field(description="high | moderate | low — probability assumption is wrong")
    if_wrong: str = Field(description="Concrete outcome if this assumption fails")
    investigation_question: str = Field(description="Specific question to verify this assumption")


class RankedAssumptionsOutput(BaseModel):
    assumptions: list[AssumptionRisk]


class ConsequenceForecast(BaseModel):
    link_name: str = Field(description="Name of the causal link being forecast")
    dollars_at_risk: float = Field(ge=0.0, description="Dollars at risk if link breaks (Basel-EAD: ≤ approved_amount)")
    months_at_risk: float = Field(ge=0.0, description="Timeline months lost if link breaks")
    rationale: str = Field(description="Brief justification for the dollar and month estimates")


class ForecastOutput(BaseModel):
    forecasts: list[ConsequenceForecast]


# ── Investigation loop schemas ────────────────────────────────────────────────


class InvestigationAction(BaseModel):
    tool: str = Field(description="Tool name: search_investment|search_bow|search_doc_type|search_all|search_web|read_pages|compute")
    query: str = Field(description="Natural-language question to research")
    doc_type: str | None = Field(default=None, description="Doc type filter for search_doc_type")
    rationale: str = Field(default="", description="Why this search is needed")


class InvestigationActionsOutput(BaseModel):
    status: str = Field(description="answered|partially_answered|not_answerable|unresolved_conflict")
    confidence: str = Field(description="high|moderate|low|insufficient")
    answer: str = Field(default="", description="Current analysis and assessment")
    evidence_refs: list[str] = Field(default=[], description="§-refs cited in this analysis")
    next_actions: list[InvestigationAction] = Field(default=[], description="Tool calls to execute next; empty when done")


# ── Science investigation schemas ─────────────────────────────────────────────


class ScienceAction(BaseModel):
    tool: str = Field(description="Tool: search_asta|search_web|search_investment|search_all")
    query: str = Field(description="Search query")
    rationale: str = Field(default="")


class ScienceActionsOutput(BaseModel):
    status: str = Field(description="continue|evidence_gathered|insufficient_evidence|blocked")
    confirming_evidence_found: bool = Field(default=False, description="Found literature supporting the assumption")
    disconfirming_evidence_found: bool = Field(default=False, description="Found literature challenging the assumption")
    answer: str = Field(default="", description="Current synthesis of scientific evidence")
    next_actions: list[ScienceAction] = Field(default=[], description="Next tool calls; empty when done")


# ── Decision projection schemas ───────────────────────────────────────────────


class DecisionCandidate(BaseModel):
    decision_type: str = Field(
        description=(
            "One of: approve_with_conditions, defer_pending_data, request_progress_packet, "
            "extend_no_cost, supplement, redirect_funds, terminate_unless_resolved, "
            "escalate_to_leadership, escalate_to_partner, schedule_review, validate_assumption, "
            "align_with_strategy_team, approve_as_is, monitor, decommission_layer"
        )
    )
    recommended_action: str = Field(description="Specific action leadership should take")
    goal_link: str = Field(default="", description="Causal link this decision addresses")
    triggering_link_ids: list[str] = Field(description="IDs of links that triggered this decision")
    triggering_evidence: list[str] = Field(default=[], description="Evidence items supporting this decision")
    corroboration_count: int = Field(ge=0, description="Number of independent evidence sources")
    cost_impact_dollars: float = Field(ge=0.0, description="Expected financial impact")
    timeline_impact_months: float = Field(ge=0.0, description="Expected timeline impact in months")
    urgency: str = Field(description="immediate|near_term|medium_term|long_term")
    materiality: str = Field(description="high|medium|low")
    confidence: str = Field(description="high|medium|low")
    rationale: str = Field(default="", description="Evidence basis for this decision")


class DecisionProjectionOutput(BaseModel):
    decisions: list[DecisionCandidate]


# ── Rubric evaluation schemas ─────────────────────────────────────────────────


class StrategyQueryList(BaseModel):
    queries: list[str] = Field(description="10 natural-language queries to retrieve strategy context")
