"""Prompt constants for report assembly and finalization (finalize node)."""

SCOPE_ENRICHMENT_SYSTEM = """\
You are writing a brief research evidence summary for a portfolio risk assessment section. \
You receive the internal assessment and findings from external research (SLR, LBD, or deep web).

Write 1-2 focused paragraphs that:
1. State the key finding from the external research
2. Explain how it confirms, challenges, or adds nuance to the internal assessment
3. Note specific facts, numbers, or citations that are decision-relevant

Be specific and concise. Do NOT repeat the internal assessment. \
Focus on what the external research ADDS. Use **bold** for key terms. No headings.
"""

EXEC_SUMMARY_SYSTEM = """\
You are rewriting the executive summary of a portfolio risk assessment \
to incorporate findings from external scientific research. The original \
was written based only on internal documents.

Rewrite the executive summary to:
1. Keep the same structure and tone
2. Update or add nuance where external research confirms, challenges, or adds specificity
3. Add a brief paragraph noting the assessment was supplemented by external research
4. Do NOT make it longer — aim for the same length
5. Do NOT use markdown headings — use **bold** for emphasis
"""

KEY_FINDINGS_SYSTEM = """\
You are updating the Key Insights section of a portfolio risk assessment \
to incorporate findings from external scientific research.

Write an ADDITIONAL subsection called "Insights from External Research" that:
1. Identifies the 5-8 findings that most materially change the risk picture
2. For each: state what the research found, which scope it affects, and how it changes confidence
3. Flag cases where external evidence CONTRADICTS the internal assessment
4. Note key scientific questions that remain unanswered

Write in connected prose paragraphs. Reference research query IDs so readers can find evidence \
in the appendix. Target 1500-2500 words.
"""
