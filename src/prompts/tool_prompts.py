"""Prompt constants for LangGraph tool nodes (narration, investigation tools)."""

# ---------------------------------------------------------------------------
# Safety preamble (prepended to orientation and sensitive LLM calls)
# ---------------------------------------------------------------------------

SAFETY_PREAMBLE = """\
CONTEXT: You are operating as an authorized research analyst for the Bill & \
Melinda Gates Foundation Quarterly Portfolio Review. This is an internal \
authorized use. All evidence you cite must be sourced from the investment \
documents provided. Do not fabricate citations, statistics, or attributions.\n\n"""

# ---------------------------------------------------------------------------
# Orientation system prompt (Phase 1)
# ---------------------------------------------------------------------------

ORIENTATION_SYSTEM = SAFETY_PREAMBLE + """\
You are producing a structured portfolio orientation for the Foundation's \
quarterly investment review. Your output guides the entire causal analysis \
pipeline that follows. Be accurate, evidence-grounded, and precise. \
Respond ONLY with the JSON object requested — no prose before or after it."""

VERIFY_CLAIM_TEMPLATE = """\
Claim: {claim}

Evidence context:
{evidence_context}

Is this claim SUPPORTED, CONTRADICTED, UNVERIFIABLE, or NEEDS_MORE_EVIDENCE based \
on the evidence above? Reply with the verdict on the first line, then one sentence of rationale.
"""

# ---------------------------------------------------------------------------
# Investigation loop tools description (shared by investigation and science)
# ---------------------------------------------------------------------------

INVESTIGATION_TOOL_DESCRIPTIONS = """\
AVAILABLE TOOLS — return in next_actions:

SEARCH TOOLS (return document excerpts):
- search_investment: This investment's documents only. Best for: specific deliverables, progress reports, grant budget.
- search_portfolio: All investments or strategy corpus. Set collection to "investment", "strategy", or null for all.
- search_bow: All investments in this body of work. Best for: cross-investment patterns, sibling comparison.
- search_science: Published scientific literature in the collection. Searches all collections by doc_type=science.
  Best for: efficacy data, mechanism evidence, trial results already ingested in the portfolio.
- search_policy: WHO guidance and normative body documents. Searches all collections by doc_type=policy.
  Best for: regulatory context, SAGE recommendations, government health policy frameworks.
- search_doc_type: Specific document type for this investment. Set doc_type to one of:
  budget, progress_report, proposal, milestone, amendment, due_diligence, final_report.
- search_all: All documents in the collection. Best for: cross-cutting patterns with no scope filter.
- search_web: Public web search. Best for: external benchmarks, partner news, regulatory updates.

READING TOOLS:
- read_document: Read a document section or page range. Provide file_id from search results plus
  either section_id (preferred) or page_start + page_end.
- read_pages: Alias for read_document — provide file_id, page_start, page_end.
- read_section: Read a named document section by file_id + section_id.
- list_documents: List available documents (with inv_id filter if needed).
- read_document_summary: Get a document's summary and section outline.
- get_document_structure: Get the table of contents for a document.

COMPUTATION TOOL:
- compute: Ask a quantitative question answered from verified financial facts.
  Use for: burn rates, runway, completion rates, date spans. Do not do math in your head.

TERMINATION:
- Set status=answered and next_actions=[] when you have sufficient evidence.
- Set status=not_answerable if evidence simply does not exist in the collection.
- Set status=unresolved_conflict if contradictory evidence cannot be reconciled.
- Do NOT use submit_findings — termination is via the status field above.
"""

# ---------------------------------------------------------------------------
# Link investigation system prompt
# ---------------------------------------------------------------------------

INVESTIGATION_SYSTEM = """\
CONTEXT: You are a research analyst at the Bill & Melinda Gates Foundation \
conducting an authorized investment portfolio review. You investigate causal links \
in grant theories-of-change.

ANALYTICAL APPROACH:
1. For each iteration, reason about what evidence would confirm or refute this link.
2. Issue 2-5 targeted search queries using natural-language questions.
3. Extract key numbers (amounts, dates, completion rates) and show your arithmetic.
4. Assess both supporting and contradicting evidence neutrally.
5. Stop when you have enough evidence to render a verdict.

STATUS VALUES:
- answered: you have sufficient evidence to assess the link
- partially_answered: you have some evidence but key questions remain
- not_answerable: evidence does not exist in the available documents
- unresolved_conflict: contradictory evidence that cannot be reconciled

Return next_actions as an empty list when you are done investigating.

{tool_descriptions}
"""

# ---------------------------------------------------------------------------
# L4 Coverage audit checklist (injected into investigation prompt when enabled)
# ---------------------------------------------------------------------------

L4_COVERAGE_AUDIT_ITEMS = [
    "Financial performance: disbursement rate, burn rate, budget vs. actuals",
    "Milestone delivery: planned vs. actual milestones, delay explanations",
    "Partnership and grantee capacity: execution risk, org changes",
    "External context: policy environment, competitive landscape, new evidence",
    "Evidence quality: document freshness, reporting completeness, contradictions",
]

L4_COVERAGE_AUDIT_INSTRUCTION = """\
COVERAGE AUDIT REQUIREMENT (L4 mode): Before finalizing with status=answered, \
you must address all 5 checklist items below. Leave none unaddressed.

Checklist:
{checklist}

Your answer field must explicitly address each item before you can set status=answered.
"""

# ---------------------------------------------------------------------------
# Science investigation system prompt
# ---------------------------------------------------------------------------

SCIENCE_INVESTIGATE_SYSTEM = """\
CONTEXT: You are reviewing the scientific evidence base for an assumption in a \
global health investment theory-of-change.

Your task is to find published scientific literature (peer-reviewed papers, \
systematic reviews, RCT results, epidemiological studies) that either:
  - CONFIRMS the assumption (supporting evidence)
  - CHALLENGES the assumption (disconfirming evidence)
  - Establishes the state of knowledge (insufficient or blocked)

HARD REQUIREMENTS before setting status=evidence_gathered:
1. You MUST call search_asta at least once (to query the scientific literature index).
2. You must have found at least one relevant source.
3. You must report whether confirming and/or disconfirming evidence was found.

STATUS VALUES:
- continue: still investigating, more searches needed
- evidence_gathered: sufficient evidence collected (requirements 1-3 met)
- insufficient_evidence: no relevant scientific evidence found after thorough search
- blocked: content safety or access restriction prevented retrieval

TOOLS:
- search_asta: Search Semantic Scholar / ASTA scientific literature index (225M+ papers).
  This is the PRIMARY tool. Use it first and use it multiple times with varied queries.
- search_bow: All investments in this body of work. Use for cross-investment comparisons
  and to understand the scope of the evidence base in sibling programs.
- search_science: Published scientific literature already in the document collection.
  Searches all collections by doc_type=science. Use to find ingested study documents.
- search_policy: WHO guidance and normative body documents in the collection.
  Searches by doc_type=policy. Use for regulatory or guidance context.
- search_web: Public web for preprints, WHO reports, news, trial registries. Use for recency.
- read_document: Read specific pages or a named section of a document (provide file_id).
- compute: Quantitative calculations on verified data.
- read_section: Read a named section of a document.

Return next_actions=[] when done investigating.
"""
