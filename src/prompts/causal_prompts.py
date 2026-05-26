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
