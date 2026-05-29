from __future__ import annotations

import operator
from typing import Annotated, Optional, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


def merge_scope_outputs(existing: list[dict] | None, update: list[dict] | None) -> list[dict]:
    """
    Deep-merge scope_outputs by scope_id.
    Used when parallel nodes both write to scope_outputs but to different keys.
    Example:
      existing = [{"scope_id": "s1", "causal_model": {...}}]
      update   = [{"scope_id": "s1", "bow_context": {...}}]
      result   = [{"scope_id": "s1", "causal_model": {...}, "bow_context": {...}}]
    New scope_ids are appended. Known scope_ids are shallow-merged.
    None inputs are treated as empty lists (defensive — callers should pass []).
    """
    by_id: dict[str, dict] = {}
    for scope in (existing or []):
        sid = scope.get("scope_id")
        if sid is not None:
            by_id[sid] = dict(scope)
    for scope in (update or []):
        sid = scope.get("scope_id")
        if sid is not None:
            if sid in by_id:
                by_id[sid].update(scope)
            else:
                by_id[sid] = dict(scope)
        # scopes without scope_id are dropped — they're malformed
    return list(by_id.values())


def _take_update(existing: list | None, update: list | None) -> list:
    """Reducer that replaces with the update value (last writer wins).

    Used for WorkflowState fan-out accumulator fields that are written
    exactly once (by the analyze bridge node) after the subgraph
    consolidates data into scope_outputs. Avoids unbounded growth when
    those fields are cleared to [] post-analysis.
    """
    return update if update is not None else (existing or [])


# ── Research agent sub-states (for agent graph fan-outs) ──────────────────────


class SLRAgentState(TypedDict):
    task_id: str
    query: str
    context: str
    top_k: int
    messages: Annotated[list[AnyMessage], add_messages]
    agent_rounds: int
    merged_papers: Optional[list[dict]]          # written once by slr_collect_papers
    synthesis: Optional[str]
    source_count: int
    search_strategy: str
    tool_traces: Annotated[list[dict], operator.add]
    result: Optional[dict]
    success: bool
    error_message: Optional[str]
    errors: Annotated[list[str], operator.add]


class LBDAgentState(TypedDict):
    task_id: str
    query: str
    context: str
    messages: Annotated[list[AnyMessage], add_messages]
    agent_rounds: int
    seed_concepts: Optional[list[str]]           # extracted from tool call args in lbd_collect_papers
    merged_papers: Optional[list[dict]]          # written once by lbd_collect_papers
    discovered_concepts: Optional[list[dict]]
    narrative: Optional[str]
    paper_count: int
    tool_traces: Annotated[list[dict], operator.add]
    result: Optional[dict]
    success: bool
    error_message: Optional[str]
    errors: Annotated[list[str], operator.add]


class DeepWebAgentState(TypedDict):
    task_id: str
    question: str
    context: str
    model_used: Optional[str]
    primary_result: Optional[dict]
    search_round_results: Annotated[list[dict], operator.add]
    fallback_synthesis: Optional[str]
    tool_traces: Annotated[list[dict], operator.add]
    result: Optional[dict]
    success: bool
    error_message: Optional[str]
    errors: Annotated[list[str], operator.add]


class EdisonAgentState(TypedDict):
    task_id: str
    original_query: str
    context: str
    skip_rewrite: bool
    rewritten_query: Optional[str]
    papers: Optional[list[dict]]
    result_count: int
    tool_traces: Annotated[list[dict], operator.add]
    result: Optional[dict]
    success: bool
    error_message: Optional[str]
    errors: Annotated[list[str], operator.add]


class DeepWebSearchRoundState(TypedDict):
    round_number: int
    question: str
    prior_context: str
    result: Optional[dict]


# ── Sub-state slices sent via Send() ─────────────────────────────────────────


class InvestmentRubricState(TypedDict):
    """Sent per-investment into evaluate_investment_rubric (Stage 3.1)."""
    inv_id: str
    scope_id: str
    scope_label: Optional[str]       # human-readable scope label for scope-fit classification
    timeline: dict                   # serialised InvestmentTimeline
    result: Optional[dict]           # InvestmentEvidencePack — filled by worker
    research_model: str              # model used for LLM calls in this worker
    ingested_dir: str                # fallback for _get_tools when search_backend absent from config
    collection_name: str             # fallback for _get_tools when search_backend absent from config


class LinkInvestigationState(TypedDict):
    """Sent per-causal-link into investigate_link (Stage 3.4)."""
    link_id: str
    inv_id: str
    bow_id: str
    scope_id: str
    scope_label: str                  # human-readable scope label for excerpt CSV output
    claim: dict                      # serialised InvestigationClaim
    model: str
    result: Optional[dict]           # LinkAssessment — filled by worker
    ingested_dir: str                # fallback for _get_tools when search_backend absent from config
    collection_name: str             # fallback for _get_tools when search_backend absent from config


class ScienceAssumptionState(TypedDict):
    """Sent per-science-assumption into investigate_science_assumption (Stage 3.5d)."""
    assumption_id: str
    inv_id: str
    bow_id: str
    scope_id: str
    question: str
    result: Optional[dict]           # ScienceInvestigationResult — filled by worker
    research_model: str              # model used for LLM calls in this worker
    ingested_dir: str                # fallback for _get_tools when search_backend absent from config
    collection_name: str             # fallback for _get_tools when search_backend absent from config


class InvestmentNarrativeState(TypedDict):
    """Sent per-investment into generate_investment_narrative (analyze subgraph §2.6 fan-out)."""
    scope_id: str
    scope_label: str
    inv_id: str
    inv_data: dict     # InvestmentTimeline.to_dict() or minimal financial dict for fallback
    model: str


class ScopeSynthesisState(TypedDict):
    """Sent per-scope into generate_scope_synthesis (analyze subgraph §2.6 second fan-out)."""
    scope_id: str
    scope_label: str
    investment_narratives: list    # [{scope_id, inv_id, narrative, inv_data}, ...]
    scope_timeline_dict: dict
    model: str


class BowEnrichmentWorkerState(TypedDict):
    """Sent per-scope into enrich_bow_context_worker (causal subgraph Send() fan-out §1.5)."""
    scope_id: str
    scope: dict                      # full scope dict (bow_ids, label, dispatch_results)
    model: str
    result: Optional[dict]           # unused — worker writes directly to scope_outputs


class InvestmentReportWorkerState(TypedDict):
    """Sent per-scope into build_investment_report_worker (analyze subgraph Send() fan-out §3.5)."""
    scope_id: str
    scope: dict                      # full scope dict (link_assessments, inv_id, etc.)
    investment_scoring: dict         # {inv_id: InvestmentDetail} — needed for team scores
    model: str
    result: Optional[dict]           # unused — worker writes directly to scope_outputs


class SectionDraftWorkerState(TypedDict):
    """Sent per-scope into synthesize_scope_section_worker (analyze subgraph Send() fan-out §3.6)."""
    scope_id: str
    scope: dict                      # full scope dict (investment_report, link_assessments, etc.)
    model: str
    result: Optional[dict]           # unused — worker writes directly to scope_outputs


class ResearchTaskState(TypedDict):
    """Sent per-research-task into the appropriate worker node (Stage 4)."""
    task_id: str
    task_type: str                   # "slr" | "lbd" | "deep_web" | "edison"
    query: str
    linked_scope: str
    priority: str                    # "critical" | "important" | "nice_to_have"
    result: Optional[dict]           # filled by worker


class ScopeDecisionState(TypedDict):
    """Sent per-scope into project_scope_decisions (Stage 3.8)."""
    scope_id: str
    scope_output: dict               # serialised ScopeOutput
    decisions: Optional[list[dict]]  # list[Decision] — filled by worker
    synthesis_model: str             # model used for LLM calls in this worker


# ── Top-level WorkflowState ───────────────────────────────────────────────────


class WorkflowState(TypedDict):
    """
    Single state object threaded through all pipeline stages.
    Collection artifacts (ingested_dir and below) are pre-existing inputs,
    read at pipeline startup. Pipeline outputs (analyst_report and below)
    are written by each stage.
    """

    # ── Run identity ──────────────────────────────────────────────────────────
    program: str                          # e.g. "Malaria", "HIV"
    run_name: str                         # human-readable slug, e.g. "crimson-falcon"
    collection_name: str                  # alias used in search backend routing
    base_dir: str                         # path to ~/qpr-collections
    threads_dir: Optional[str]            # {base_dir}/{program}-experiments/run-{run_name}
    focus: Optional[str]                  # free-text focus area for analyze
    focus_bows: Optional[list[str]]       # restrict analysis to these BOW IDs
    aux_collections: Optional[list[str]]  # cross-corpus collection names for asset fanout
    research_model: str                   # model used for investigation/research LLM calls
    synthesis_model: str                  # model used for synthesis and report-writing LLM calls

    # ── Pre-existing collection inputs (read from {program}-ingested/) ─────────
    # Populated by load_collection; never mutated after that.
    ingested_dir: Optional[str]            # absolute path to {program}-ingested/ (derived from base_dir+program if absent)
    doc_list: Optional[list[dict]]        # content of doc_list.json
    investment_scoring: Optional[dict]    # {inv_id: InvestmentDetail}
    bow_investment_map: Optional[dict]    # {bow_id: [inv_id, ...]}
    investment_bow_rows: Optional[list]   # raw rows from investment_bow_rows.json
    investment_intelligence: Optional[dict]  # {inv_id: InvestmentIntelligence}
    chunks_json_path: Optional[str]       # embedding_index/chunks.json
    pages_dir: Optional[str]              # pages/{file_id}/ tree root

    # ── Precheck gate ─────────────────────────────────────────────────────────
    precheck_passed: Optional[bool]
    precheck_report: Optional[str]        # formatted text — also written to disk by precheck node

    # ── Stage 2 outputs (under threads_dir/) ──────────────────────────────────
    final_report_md_path: Optional[str]   # human-readable deliverable — keep
    final_report_md: Optional[str]
    analyst_report: Optional[dict]        # serialised AnalystReport (in-state; no path)
    scope_outputs: Optional[list[dict]]   # list[ScopeOutput] (in-state; no path)
    excerpts_csv_path: Optional[str]      # written by deliver.py
    numerical_provenance: Optional[list[dict]]   # from InvestmentFacts computation
    verification_sources: Optional[list[dict]]
    allocation_verification: Optional[list[dict]]    # from verify_report
    numerical_verification: Optional[list[dict]]     # from verify_report
    bibliography: Optional[list[dict]]   # deduplicated cited sources
    run_meta: Optional[dict]
    coverage_pct: Optional[float]
    grade: Optional[str]
    confidence_map: Optional[dict]        # {scope_id: "high"|"medium"|"low"}

    # Intermediate analyze state (subgraph-internal, checkpointed for resume)
    program_context: Optional[dict]       # structured ProgramContext (Phase 1)
    scopes: Optional[list[dict]]          # list[Scope]
    scope_timelines: Optional[dict]       # {scope_id: ScopeTimeline}
    cross_cutting_analysis: Optional[dict]    # typed CrossCuttingAnalysis (Phase 4)
    allocation_verification_path: Optional[str]
    numerical_verification_path: Optional[str]

    # Fan-out collector fields — written by Send() worker nodes.
    # Use _take_update reducer (not operator.add) so the analyze bridge can
    # clear these to [] after analysis consolidates data into scope_outputs.
    evidence_packs: Annotated[list[dict], _take_update]
    link_assessments: Annotated[list[dict], _take_update]
    science_results: Annotated[list[dict], _take_update]
    scope_decisions: Annotated[list[dict], _take_update]

    # Annotated excerpts from all link investigations (accumulated; written by deliver.py)
    all_excerpts: Annotated[list[dict], operator.add]

    # ── Tool call traces (accumulated across all nodes) ────────────────────────
    asta_traces: Annotated[list[dict], operator.add]
    slr_traces: Annotated[list[dict], operator.add]
    lbd_traces: Annotated[list[dict], operator.add]
    deep_web_traces: Annotated[list[dict], operator.add]
    edison_traces: Annotated[list[dict], operator.add]
    web_search_traces: Annotated[list[dict], operator.add]
    compute_traces: Annotated[list[dict], operator.add]
    collection_search_traces: Annotated[list[dict], operator.add]
    investigation_traces: Annotated[list[dict], operator.add]

    # ── Stage 3 outputs ────────────────────────────────────────────────────────
    research_plan: Optional[list[dict]]   # [{id, type, query, priority, linked_scope}]
    research_plan_md_path: Optional[str]  # human-readable deliverable — keep
    research_plan_approved: Optional[bool]

    # ── Stage 4 outputs (under threads_dir/research/) ─────────────────────────
    research_dir: Optional[str]
    research_results: Annotated[list[dict], operator.add]    # per-task result dicts

    # ── Stage 5 outputs ────────────────────────────────────────────────────────
    final_report_wresearch_md_path: Optional[str]
    final_report_wresearch_md: Optional[str]
    final_report_pdf_path: Optional[str]  # written by rerender (focused.pdf)
    report_approved: Optional[bool]

    # ── Error / status ─────────────────────────────────────────────────────────
    errors: Annotated[list[str], operator.add]
    current_stage: Optional[str]


# ── Subgraph states (standalone TypedDicts — not subclasses of WorkflowState) ─


class AnalyzeState(TypedDict):
    """State for the analyze subgraph (§3). Standalone — not a subclass of WorkflowState."""

    # ── Collection inputs (read-only, passed in from WorkflowState) ────────────
    program: str
    collection_name: str
    base_dir: str
    ingested_dir: str
    doc_list: list[dict]
    investment_scoring: dict
    bow_investment_map: dict
    investment_intelligence: dict
    chunks_json_path: str
    pages_dir: str
    focus: Optional[str]
    focus_bows: Optional[list[str]]
    aux_collections: Optional[list[str]]

    # ── Run context ────────────────────────────────────────────────────────────
    threads_dir: Optional[str]
    research_model: str
    synthesis_model: str

    # ── Phase outputs (written progressively by subgraph nodes) ────────────────
    program_context: Optional[dict]      # structured ProgramContext (Phase 1)
    scopes: Optional[list[dict]]
    scope_timelines: Optional[dict]
    cross_cutting_analysis: Optional[dict]    # typed CrossCuttingAnalysis (Phase 4)
    scope_outputs: Annotated[list[dict], merge_scope_outputs]  # forward-compatible with parallel causal branches
    analyst_report: Optional[dict]
    final_report_md: Optional[str]
    final_report_md_path: Optional[str]   # written by assemble_report
    bibliography: Optional[list[dict]]   # deduplicated cited sources
    excerpts_csv_path: Optional[str]
    numerical_provenance: Optional[list[dict]]   # from InvestmentFacts computation
    verification_sources: Optional[list[dict]]
    allocation_verification: Optional[list[dict]]    # from verify_report
    allocation_verification_path: Optional[str]
    numerical_verification: Optional[list[dict]]     # from verify_report
    numerical_verification_path: Optional[str]
    run_meta: Optional[dict]
    coverage_pct: Optional[float]
    grade: Optional[str]
    confidence_map: Optional[dict]    # {scope_id: "high"|"medium"|"low"}

    # ── Fan-out reducer fields (Send() workers write into these) ───────────────
    evidence_packs: Annotated[list[dict], operator.add]
    link_assessments: Annotated[list[dict], operator.add]
    science_results: Annotated[list[dict], operator.add]
    scope_decisions: Annotated[list[dict], operator.add]
    investment_narrative_results: Annotated[list[dict], operator.add]  # per-investment; feeds dispatch_scope_syntheses
    timeline_narrative_results: Annotated[list[dict], operator.add]
    all_excerpts: Annotated[list[dict], operator.add]   # top-10 per link by credibility_tier desc, ~50 chars each

    # ── Tool call traces (accumulated across all nodes) ────────────────────────
    asta_traces: Annotated[list[dict], operator.add]
    slr_traces: Annotated[list[dict], operator.add]
    lbd_traces: Annotated[list[dict], operator.add]
    deep_web_traces: Annotated[list[dict], operator.add]
    edison_traces: Annotated[list[dict], operator.add]
    web_search_traces: Annotated[list[dict], operator.add]
    compute_traces: Annotated[list[dict], operator.add]
    collection_search_traces: Annotated[list[dict], operator.add]
    investigation_traces: Annotated[list[dict], operator.add]

    # ── Error accumulator ──────────────────────────────────────────────────────
    errors: Annotated[list[str], operator.add]


class CausalState(TypedDict):
    """State for the causal pipeline subgraph (§4). Standalone — not a subclass of AnalyzeState."""

    # ── Inputs (passed in from AnalyzeState) ───────────────────────────────────
    scopes: list[dict]
    scope_timelines: dict
    research_model: str
    synthesis_model: str

    # ── Fan-out reducer fields ──────────────────────────────────────────────────
    evidence_packs: Annotated[list[dict], operator.add]      # InvestmentEvidencePack per inv
    link_assessments: Annotated[list[dict], operator.add]    # LinkAssessment per link
    science_results: Annotated[list[dict], operator.add]     # ScienceInvestigationResult per assumption
    scope_decisions: Annotated[list[dict], operator.add]     # Decision lists per scope

    # ── Tool call traces (accumulated across all nodes) ────────────────────────
    asta_traces: Annotated[list[dict], operator.add]
    slr_traces: Annotated[list[dict], operator.add]
    lbd_traces: Annotated[list[dict], operator.add]
    deep_web_traces: Annotated[list[dict], operator.add]
    edison_traces: Annotated[list[dict], operator.add]
    web_search_traces: Annotated[list[dict], operator.add]
    compute_traces: Annotated[list[dict], operator.add]
    collection_search_traces: Annotated[list[dict], operator.add]
    investigation_traces: Annotated[list[dict], operator.add]

    # ── Progressive outputs (written by collect/reducer nodes) ─────────────────
    # merge_scope_outputs reducer deep-merges by scope_id — safe for parallel
    # branches writing to different keys within each scope dict.
    scope_outputs: Annotated[list[dict], merge_scope_outputs]  # list[ScopeOutput], built progressively

    # Annotated excerpts accumulated from link investigations
    all_excerpts: Annotated[list[dict], operator.add]   # top-10 per link by credibility_tier desc

    # ── Error accumulator ──────────────────────────────────────────────────────
    errors: Annotated[list[str], operator.add]


class ResearchDispatchState(TypedDict):
    """State for the research dispatch subgraph (§5). Standalone — not a subclass of WorkflowState."""

    # ── Inputs (passed in from WorkflowState) ──────────────────────────────────
    research_plan: list[dict]    # [{id, type, query, priority, linked_scope}]
    research_dir: str

    # ── Fan-out reducer field ───────────────────────────────────────────────────
    research_results: Annotated[list[dict], operator.add]    # per-task result dicts

    # ── Research tool call traces ──────────────────────────────────────────────
    slr_traces: Annotated[list[dict], operator.add]
    lbd_traces: Annotated[list[dict], operator.add]
    deep_web_traces: Annotated[list[dict], operator.add]
    edison_traces: Annotated[list[dict], operator.add]

    # ── Aggregated outputs (split by type in aggregate_research_results) ──────────
    dispatch_results: list[dict]   # non-edison research results
    edison_results: list[dict]     # edison-type research results

    # ── Error accumulator ──────────────────────────────────────────────────────
    errors: Annotated[list[str], operator.add]


# ── Diagnostic graph states (§13) ─────────────────────────────────────────────


class EvidenceAuditState(TypedDict):
    """State for the evidence_audit diagnostic graph (§13).

    Post-hoc analysis of a completed analyst run. Independent of the main
    pipeline thread_id. No LLM calls unless skip_llm_expected_docs=False.
    """

    # ── Inputs ────────────────────────────────────────────────────────────────
    program: str                          # e.g. "VDEV", "Malaria"
    data_root: str                        # path to ~/qpr-collections
    run_dir_name: str                     # e.g. "phaseA-data-prep-full"
    top_n_files: int                      # top-N files for influence ranking (default 25)
    skip_llm_expected_docs: bool          # if False, run LLM-based expected-docs audit
    output_xlsx: bool                     # write .xlsx workbook (controlled by config flag)
    output_diagnosis: bool                # write diagnosis JSON (controlled by config flag)

    # ── Loaded artifacts (from load_artifacts) ────────────────────────────────
    analyst_report: Optional[dict]        # analyst_report.json
    doc_list: Optional[list[dict]]        # doc_list.json
    investment_scoring: Optional[dict]    # investment_scoring.json
    run_dir: Optional[str]               # resolved run dir path

    # ── Audit result (from run_audit) ─────────────────────────────────────────
    audit: Optional[dict]                 # full audit dict (+ evidence_audit.json written)

    # ── Output paths (from write_* nodes) ─────────────────────────────────────
    brief_md: Optional[str]              # rendered markdown brief
    brief_path: Optional[str]            # path to team_brief.md
    workbook_path: Optional[str]         # path to evidence_audit.xlsx (if output_xlsx)
    rollup_md_path: Optional[str]        # path to cross_program_rollup.md (if rollup ran)

    # ── Error accumulator ─────────────────────────────────────────────────────
    errors: Annotated[list[str], operator.add]


class FindingVerificationState(TypedDict):
    """Sent per-finding into the verify_finding worker node (gs_verifier §13)."""

    finding: dict              # gold standard finding dict
    scope_label: str           # human-readable scope label
    finding_type: str          # classified finding type
    evidence_bundle: str       # pre-rendered evidence text
    program: str
    as_of_date: str            # ISO date string "YYYY-MM-DD"
    verifier_a_model: str      # primary verifier model
    verifier_b_model: str      # independent second-opinion model
    result: Optional[dict]     # filled by verify_finding worker


class GsVerifierState(TypedDict):
    """State for the gs_verifier graph (§13).

    Re-verifies gold standard findings using dual LLM verifiers in parallel,
    reconciles verdicts, and produces tiered gold output.
    """

    # ── Inputs ────────────────────────────────────────────────────────────────
    program: str
    gold_path: str                        # path to gold_standard_verified.json
    out_path: str                         # path for gold_v3_reverified.json output
    as_of_date: str                       # ISO date string "YYYY-MM-DD"
    verifier_a_model: str                 # primary model (default: claude-opus-4-7)
    verifier_b_model: str                 # second-opinion model (default: claude-sonnet-4-6)
    skip_causal: bool                     # skip causal model refresh

    # ── Loaded (from load_gold) ───────────────────────────────────────────────
    gold_data: Optional[dict]             # full gold standard JSON
    doc_list: Optional[list[dict]]        # for building evidence bundles
    investment_scoring: Optional[dict]    # for investment context

    # ── Fan-out reducer (from verify_finding workers) ─────────────────────────
    verdicts: Annotated[list[dict], operator.add]    # one per finding

    # ── Outputs ───────────────────────────────────────────────────────────────
    reconciled_path: Optional[str]        # path to gold_v3_reverified.json
    tiered_gold_dir: Optional[str]        # directory with tier1/tier2/tier3 files
    gold_v4_path: Optional[str]           # path to gold_v4.json
    status_counts: Optional[dict]         # {agree_retain, agree_reject, ...}

    # ── Error accumulator ─────────────────────────────────────────────────────
    errors: Annotated[list[str], operator.add]
