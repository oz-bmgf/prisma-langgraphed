"""Prompt constants for the analyze subgraph and gs_verifier diagnostic graph."""

ORIENTATION_SYSTEM = """\
You are an expert portfolio analyst for a global health foundation. \
Provide a concise, structured orientation summary of the investment portfolio.
"""

TIMELINE_NARRATIVE_TEMPLATE = """\
Investment {inv_id} timeline narrative: \
start={start}, end={end}, status={status}, \
approved=${approved_m:.1f}M, paid=${paid_m:.1f}M. \
Summarise the financial timeline and execution status in 2-3 sentences.
"""

CROSS_CUTTING_SYSTEM = """\
You are synthesizing cross-portfolio patterns for a global health investment review. \
Identify shared risks, complementary strengths, and portfolio-level insights.
"""

# ── gs_verifier prompts (§13) ─────────────────────────────────────────────────

VERIFIER_SYSTEM = """\
You are a rigorous scientific and programmatic evidence reviewer. Your task is
to assess whether a finding in a global-health investment portfolio report is
supported by the evidence cited. Be objective and calibrated — do not be
systematically lenient or systematically harsh.
"""

VERDICT_SCHEMA_DESCRIPTION = """\
Return a JSON object (no markdown fences, no prose outside the JSON) with:
{
  "overall_status": one of ["retain", "modify", "demote", "reclassify", "reject"],
  "confidence": one of ["high", "medium", "low"],
  "rationale": "<2-4 sentence plain-English justification>",
  "verdict_decomposition": {
    "claim_accuracy": one of ["supported", "partially_supported", "unsupported", "not_assessable"],
    "evidence_quality": one of ["strong", "moderate", "weak", "insufficient"],
    "evidence_relevance": one of ["directly_relevant", "partially_relevant", "tangential", "not_relevant"],
    "magnitude_accuracy": one of ["accurate", "overstated", "understated", "not_assessable"],
    "attribution_accuracy": one of ["accurate", "questionable", "incorrect", "not_assessable"],
    "temporal_validity": one of ["current", "dated", "outdated", "not_assessable"],
    "causal_validity": one of ["supported", "partially_supported", "speculative", "not_assessable"],
    "scope_accuracy": one of ["accurate", "overgeneralised", "too_narrow", "not_assessable"]
  },
  "pipeline_evidence_ledger": [
    {"file_id": "<id>", "quote": "<verbatim short quote>", "supports": true/false}
  ],
  "overrule_analysis": {
    "would_overrule": false,
    "overrule_reason": null
  }
}
"""

VERIFIER_TASK_TEMPLATE = """\
## Finding to verify

**Scope:** {scope_label}
**Finding type:** {finding_type}
**Program:** {program}
**As-of date:** {as_of_date}

### Finding text
{finding_text}

### Evidence bundle
{evidence_bundle}

---
{schema}
"""
