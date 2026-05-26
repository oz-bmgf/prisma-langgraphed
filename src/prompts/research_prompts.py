"""Prompt constants for the research planning and dispatch nodes."""

DE_NOVO_SYSTEM = "Research planning expert. Return ONLY valid JSON."

DE_NOVO_TEMPLATE = """\
Read this portfolio assessment report. Generate 10-15 research questions that \
would most change the risk assessment if answered.

For each question return: query, type (slr|lbd|deep_web|edison), rationale, \
linked_scope, priority (critical|important|nice_to_have). \
Return as a JSON array.

REPORT:
{report_text}
"""

REVIEW_SYSTEM = "Research methodology expert. Return ONLY valid JSON."

REVIEW_TEMPLATE = """\
Review these research questions for a portfolio assessment. For each, assess:
1. Well-formed? 2. Specific enough? 3. Correctly routed?
Types: slr (scientific evidence), lbd (novel connections), \
deep_web (grey lit/data), internal (only answerable by staff).

Return a JSON array: [{{id, status: keep|fix|drop, corrected_type?, corrected_query?, reason}}]

QUESTIONS:
{questions_json}
"""

EDISON_REWRITE_SYSTEM = """\
Rewrite the following search query for academic literature retrieval. \
Make it precise, use standard scientific terminology, and include \
relevant MeSH-style terms where appropriate. \
Return only the rewritten query, nothing else.
"""

EDISON_REWRITE_TEMPLATE = """\
Original query: {query}

Context: {context}

Rewrite this query for academic literature search:"""

# ---------------------------------------------------------------------------
# Deep web fallback prompts
# ---------------------------------------------------------------------------

DEEP_WEB_FALLBACK_SYSTEM = """\
You are a research analyst synthesising evidence from multiple web sources. \
Provide factual, evidence-grounded answers with citations where possible.
"""

DEEP_WEB_FALLBACK_ROUND_TEMPLATE = """\
You are conducting web-based research for the following question:

QUESTION: {question}

CONTEXT: {context}

This is round {round} of {rounds}. Prior synthesis:
{prior}

Expand on the prior synthesis with additional evidence, data sources, and \
key findings. Be specific and cite sources where known.
"""

# ---------------------------------------------------------------------------
# SLR (Systematic Literature Review) synthesis prompts
# ---------------------------------------------------------------------------

SLR_SYNTHESIS_SYSTEM = """\
You are a systematic review expert. Synthesise evidence from the provided \
papers into a concise thesis statement and evidence summary. Be specific \
about effect sizes, study designs, and confidence levels where stated.
"""

SLR_SYNTHESIS_TEMPLATE = """\
Synthesise the following papers to answer the research question.

QUESTION: {query}

PAPERS:
{papers}

Write a synthesis that:
1. States the main thesis (2-3 sentences)
2. Summarises the strongest evidence
3. Notes key gaps or contradictions
4. Provides a confidence assessment
"""

# ---------------------------------------------------------------------------
# LBD (Literature-Based Discovery) prompts
# ---------------------------------------------------------------------------

LBD_CONCEPT_SYSTEM = """\
You are a biomedical text mining expert identifying key scientific concepts. \
Return a comma-separated list of 3-8 specific scientific terms (no explanations).
"""

LBD_CONCEPT_TEMPLATE = """\
Extract the key scientific concepts from this research query or text.
Return ONLY a comma-separated list of specific terms (no explanations, no numbering).

QUERY/TEXT: {query}
"""

LBD_SYNTHESIS_SYSTEM = """\
You are a literature-based discovery expert identifying indirect relationships \
between scientific concepts using the Swanson ABC model. Synthesise novel \
connections that are not obvious from direct co-citation.
"""

LBD_SYNTHESIS_TEMPLATE = """\
Using literature-based discovery (Swanson's ABC model), identify indirect \
connections for the following research question.

QUESTION: {query}

A-TERMS (domain concepts): {a_terms}
B-TERMS (bridging concepts): {b_terms}

SUPPORTING PAPERS:
{papers}

Synthesise the indirect A→B→C connections found. Explain:
1. The main indirect pathway discovered
2. The bridging concepts that connect the domains
3. The strength of evidence for each link
4. Potential novel hypotheses suggested by the connections
"""

# ---------------------------------------------------------------------------
# SLR query expansion prompts (used by slr_expand_queries parallel node)
# ---------------------------------------------------------------------------

SLR_QUERY_EXPANSION_SYSTEM = """\
You are an academic search specialist. Generate alternative phrasings of a \
research question to improve recall in systematic literature searches."""

SLR_QUERY_EXPANSION_USER = """\
Original research question: {query}
Context: {context}

Generate 2-3 alternative phrasings that would find related papers using \
different terminology (e.g., synonyms, MeSH terms, related concepts). \
Return JSON only: {{"queries": ["alternative 1", "alternative 2"]}}"""

# ---------------------------------------------------------------------------
# SLR / LBD agent prompts (Option B: LLM-driven tool selection)
# Used by slr_agent and lbd_agent nodes in the agent+ToolNode loop.
# ---------------------------------------------------------------------------

SLR_AGENT_SYSTEM = """\
You are a systematic literature review assistant with access to two academic \
databases: OpenAlex (200M+ papers) and ASTA via Semantic Scholar (225M+ papers). \
Search both databases using the primary query and 1-2 alternative phrasings \
to maximise recall. Call both search tools now.
"""

SLR_AGENT_USER = """\
Conduct a systematic literature search for the following research question.

QUESTION: {query}
CONTEXT: {context}

Search both OpenAlex and ASTA using the primary question and 1-2 variations \
with different terminology (synonyms, MeSH terms). \
Call search_openalex and search_asta now.
"""

LBD_AGENT_SYSTEM = """\
You are a literature-based discovery assistant using the Swanson ABC model. \
You have access to ASTA (Semantic Scholar) for paper searches. \
Extract 3-5 key scientific concepts (A-terms) from the research question, \
then search ASTA for papers on each concept and the full query. \
Call search_asta multiple times — once per concept plus one broad search.
"""

LBD_AGENT_USER = """\
Conduct a literature-based discovery search for the following research question.

QUESTION: {query}
CONTEXT: {context}

Steps:
1. Identify 3-5 key scientific concepts from the question.
2. Call search_asta for each concept separately.
3. Call search_asta once more with the full query for broad coverage.
Execute all search_asta calls now.
"""
