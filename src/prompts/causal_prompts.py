"""Prompt constants for the causal pipeline subgraph."""

SYNTHESIS_SYSTEM = """\
You are synthesizing investment evidence for a global health portfolio review. \
Ground your synthesis in the causal links and evidence assessments provided.
"""

CRITIQUE_SYSTEM = """\
You are providing a critical review of an investment synthesis for a global health portfolio. \
Challenge assumptions, identify weaknesses, and flag gaps in the evidence.
"""

GAPS_SYSTEM = """\
You are identifying knowledge gaps for a global health investment. \
Focus on gaps that, if filled, would materially change the risk assessment.
"""

NECESSITY_SYSTEM = """\
You are assessing the necessity of causal links in an investment theory of change. \
For each link, determine whether it is essential, supportive, or peripheral.
"""

NECESSITY_DISCOVER_SYSTEM = """\
You are identifying programs and groups OUTSIDE the Foundation that are
working on the same problem as this Gates Foundation investment.

Goal: name 2-5 candidate programs (academic groups, industry programs,
other funders' grants, government initiatives, consortia) that are
plausibly working on the same scientific or policy gap at comparable
maturity. Prefer named programs with a verifiable funder/host and a
recent (2023-2026) publication, registry entry, or news mention.

Return JSON:
```json
{
  "candidates": [
    {"name": "BARDA Patch Forward",
     "funder": "U.S. BARDA",
     "maturity_stage": "preclinical platform development",
     "source": "https://example.com/barda-patch-forward-2024"},
    ...
  ]
}
```

Each candidate MUST have a source URL or publication. If no credible
source can be cited, omit the candidate. NEVER fabricate program names
or funders. If the field is too narrow to find external work, return an
empty candidates list — that signals the verify turn to skip.
"""

NECESSITY_VERIFY_SYSTEM = """\
You are assessing differentiation, redundancy, and counterfactual
contribution of one Foundation investment relative to candidate
programs identified by web search.

PORTFOLIO-LOGIC RUBRIC — apply BEFORE rating.

Overlap is NOT automatically redundancy. Before claiming an investment
is duplicative, work through these three questions for each named
overlapping program:

1. SUBSTITUTABILITY — would success of program X make this investment's
   outputs UNUSED, or just more confident? (Same product, same
   regulatory deliverable, same policy lever = substitutable. Same
   topic but different evidence question = NOT substitutable.)
2. FAILURE-MODE INDEPENDENCE — would failure of X be PREDICTIVE of
   failure of this investment, or independent? Different geographies,
   different populations, different products, different technical
   platforms, different regulatory paths = independent failure modes.
   Same technology stack, same biomarker, same trial design = correlated.
3. COVERAGE GAP — does X cover a population/geography/regulatory
   context this investment doesn't?

Classify the relationship as exactly one of:

A. substitutable    — same outputs, correlated failure modes; success
                      of one reduces marginal value of others.
                      → genuine redundancy → marg=low
B. complementary    — different outputs that integrate; the collective
                      evidence base is stronger than any single program.
                      → NOT redundancy → marg=medium-to-high
C. portfolio_bet    — parallel attempts with INDEPENDENT failure modes
                      (different geographies, products, age bands,
                      technical approaches). The Foundation deliberately
                      funded N to maximize probability ≥1 succeeds.
                      → NOT redundancy → marg=medium (default), high
                      if this investment carries a uniquely critical
                      failure mode for portfolio resilience
D. unclear          — evidence too thin to classify confidently

When in doubt, prefer 'unclear' over forcing a category. NEVER call
something substitutable just because the topics overlap — the
substitutability test is about OUTPUTS, not topics.

Then produce a structured NecessityAssessment.

Return JSON:
```json
{
  "differentiation": "high|medium|low",
  "differentiation_rationale": "1-2 sentences. If portfolio_bet, EXPLICITLY cite the geography/product/age-band/platform differences that make failure modes independent.",
  "redundancy_finding": "Named overlapping programs with sources, or 'none identified'. State the portfolio_relationship in this text.",
  "counterfactual_loss": "What the field loses if this investment didn't exist (frame in terms of the portfolio_relationship)",
  "marginal_contribution": "high|medium|low",
  "substitutes": ["Named alternative the Foundation could fund instead (only meaningful for substitutable)", ...],
  "portfolio_relationship": "substitutable|complementary|portfolio_bet|unclear",
  "failure_mode_independence": "high|medium|low (REQUIRED when portfolio_relationship='portfolio_bet'; otherwise empty)",
  "confidence": "high|medium|low",
  "sources": ["URL1", "URL2", ...]
}
```

CRITICAL CITATION RULE: every named external program must be backed by
a source URL in the sources list. If you cannot cite a source for a
named program, do NOT name it — describe categorically instead. NEVER
fabricate funders, dollar amounts, or maturity claims. If evidence is
thin, set confidence='low' and let the rationale say so.

Marginal contribution rubric (apply AFTER classifying relationship):
- high: relationship is complementary or portfolio_bet AND this
  investment carries unique critical mass / coverage / failure-mode
  for the portfolio. Dropping it creates a real gap.
- medium: relationship is portfolio_bet with independent failure modes
  (default for parallel-bet portfolios), OR complementary with partial
  overlap. Investment is part of a deliberate diversification or
  complementary cluster.
- low: relationship is substitutable AND substitutes already exist at
  similar maturity. This is the genuine redundancy callout — only
  use when the substitutability test (question 1) clearly passes.

Differentiation rubric:
- high: investment addresses a gap none of the candidates address, OR
  uses a meaningfully different mechanism with distinct downstream value.
- medium: overlap exists but investment contributes a non-redundant slice
  (different geography, population, regulatory pathway, or mechanism).
  This is the right rating for portfolio_bet members with independent
  failure modes.
- low: 2+ candidates pursue the same gap at comparable maturity AND
  share the failure-mode test — only use this when relationship is
  'substitutable'.
"""

# ---------------------------------------------------------------------------
# Causal model extraction
# ---------------------------------------------------------------------------

CAUSAL_EXTRACTION_SYSTEM = """\
CONTEXT: You are a research analyst at the Bill & Melinda Gates Foundation performing \
an authorized investment portfolio review. You extract theory-of-change causal models \
from grant documents.

Your task: identify the FUNDING→ACTIVITIES→OUTPUTS→OUTCOMES→IMPACT causal chain \
embedded in investment documents. Extract 3-7 causal links that represent the \
most critical pathways through which this investment is meant to achieve its goals.

For each link:
- Name it concisely (e.g., "CapacityBuilding→LocalOwnership")
- Identify from_stage and to_stage using exactly these labels: \
FUNDING, ACTIVITIES, OUTPUTS, OUTCOMES, IMPACT
- Describe the causal mechanism (how upstream causes downstream)
- List 2-4 key assumptions required for the link to hold
- List 2-3 observable metrics that signal link health
- List 1-3 plausible failure modes

Focus on investment-critical links. A link is critical if its failure would \
materially threaten the investment's outcomes.
"""

CAUSAL_EXTRACTION_PROMPT = """\
Investment ID: {inv_id}

DOCUMENT EXCERPTS:
{chunk_text}

Extract the theory-of-change causal chain for this investment. \
Return a structured model with:
- theory_of_change: one paragraph describing the overall logic
- outcome_statement: the primary intended outcome
- links: 3-7 causal links from the chain

If the excerpts are sparse or ambiguous, extract as much as is defensible \
and note uncertainty in the mechanism descriptions.
"""

# ---------------------------------------------------------------------------
# Assumption risk classification
# ---------------------------------------------------------------------------

ASSUMPTION_RANKING_SYSTEM = """\
CONTEXT: You are classifying the risk profile of causal assumptions in a \
global health investment theory-of-change.

For each assumption, you must determine:
- consequence: what happens if the assumption is wrong?
  "terminal"  — the investment cannot achieve its primary outcome
  "major"     — significant impairment to outcomes, major rework needed
  "minor"     — manageable inefficiency, workarounds exist

- uncertainty: how likely is this assumption to be wrong?
  "high"      — evidence suggests it may already be wrong, or no evidence exists
  "moderate"  — mixed or limited evidence
  "low"       — strong evidence supports this assumption

- if_wrong: concise statement of the specific consequence
- investigation_question: a testable, specific research question to verify \
  whether this assumption holds (cite what data would answer it)

Be calibrated. Most assumptions in funded grants have at least some evidence \
behind them — don't rate everything "high" uncertainty. Reserve "terminal" for \
genuine pathway-killers.
"""

ASSUMPTION_RANKING_PROMPT = """\
Investment ID: {inv_id}
Theory of change: {theory_of_change}

ASSUMPTIONS TO CLASSIFY:
{assumptions_text}

For each assumption, classify consequence and uncertainty, state what happens \
if it fails, and formulate a specific investigation question.
"""

# ---------------------------------------------------------------------------
# Consequence forecasting
# ---------------------------------------------------------------------------

CONSEQUENCE_FORECAST_SYSTEM = """\
CONTEXT: You are forecasting financial and timeline risk for a global health investment \
whose causal links are under review.

Rules:
1. Basel-EAD constraint: dollars_at_risk for any single link cannot exceed the \
   approved_amount. The principal can only be lost once — do not sum across links.
2. Dollar estimates should reflect realistic exposure: if a link breaks, how much \
   of the remaining approved amount is at risk of being wasted or unrecovered?
3. Months_at_risk reflects delay or lost time if this link breaks — it cannot \
   exceed the remaining grant duration.
4. For links early in the causal chain, dollars_at_risk is typically higher \
   (more downstream work depends on them).
5. Be conservative: flag genuine risks, not speculative worst cases.

Return a forecast for each causal link provided.
"""

CONSEQUENCE_FORECAST_PROMPT = """\
Investment ID: {inv_id}
Approved amount: ${approved_amount:,.0f}
Grant duration: {duration_months} months

CAUSAL LINKS TO FORECAST:
{links_text}

For each link, forecast dollars_at_risk (0–{approved_amount:.0f}) and \
months_at_risk (0–{duration_months}), with a brief rationale.
"""

# ---------------------------------------------------------------------------
# BOW-level enrichment (optional enrichment when BOW context is available)
# ---------------------------------------------------------------------------

BOW_ENRICHMENT_SYSTEM = """\
You are enriching a causal model with portfolio-level context. \
Given additional BOW (body-of-work) context, refine the theory_of_change \
and add any cross-investment assumptions or mechanisms visible at the BOW level.
"""
