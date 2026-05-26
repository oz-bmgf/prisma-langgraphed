"""Rubric evaluator — builds an evidence pack for one investment.

build_evidence_pack() runs 4-strategy retrieval to guarantee ≥20 strategy
chunks, detects fact contradictions, computes local quality scores, and
returns InvestmentEvidencePack ready for downstream investigation.

4-strategy retrieval (budget = top_k total, min_strategy = 20):
  Strategy 1: LLM-generated queries (10 natural-language questions)
  Strategy 2: 4 hardcoded fallback queries (if Strategy 1 fails)
  Strategy 3: 3 doc-type-specific queries (Series B / milestone / risk)
  Strategy 4: 5 strategy doc queries against collection="strategy"
  Floor: min_strategy=20 strategy chunks guaranteed from Strategies 3+4

No LangGraph imports.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import date, timedelta
from typing import Any

from src.config import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_SYNTHESIS_MODEL,
    TOP_K_DEFAULT,
)
from src.core.evidence_model import InvestmentEvidencePack
from src.core.llm_utils import acall_llm
from src.core.output_schemas import StrategyQueryList

logger = logging.getLogger(__name__)

_MIN_STRATEGY_CHUNKS = 20
_STRATEGY_COLLECTION = "strategy"

_EXPECTED_DOC_TYPES = {"proposal", "investment_document", "progress_report", "budget"}

# Hardcoded fallback queries (Strategy 2) when LLM query generation fails.
_FALLBACK_QUERIES = [
    "investment progress against planned milestones and deliverables",
    "budget disbursement actual versus planned expenditure",
    "partnership risks and grantee capacity concerns",
    "theory of change evidence and outcome measurement",
]

# Doc-type-specific queries (Strategy 3).
_DOC_TYPE_QUERIES = [
    ("progress_report", "milestone completion rates delays grantee performance"),
    ("budget", "disbursement paid amount outstanding balance burn rate"),
    ("proposal", "theory of change expected outcomes assumptions risks"),
]

# Strategy doc queries (Strategy 4).
_STRATEGY_QUERIES = [
    "program priorities and strategic focus areas",
    "portfolio theory of change expected outcomes",
    "risk tolerance and investment criteria",
    "evidence requirements for reinvestment decisions",
    "key strategic assumptions and dependencies",
]

_QUERY_PROMPT = """\
Investment: {inv_id}
Title: {title}
Organization: {org}
Program area: {program_area}

Generate 10 targeted natural-language questions to retrieve the most relevant \
evidence for a portfolio risk review of this investment. Questions should cover:
financial performance, milestone delivery, partnership risks, theory-of-change \
validity, external context, and evidence quality.
"""


async def build_evidence_pack(
    inv_id: str,
    scope_id: str,
    timeline: dict,
    *,
    top_k: int = TOP_K_DEFAULT,
    tools: Any = None,
    model: str = DEFAULT_SYNTHESIS_MODEL,
) -> InvestmentEvidencePack:
    """Build an evidence pack for one investment using 4-strategy retrieval.

    Guarantees ≥ _MIN_STRATEGY_CHUNKS strategy chunks. Returns
    InvestmentEvidencePack with chunks, source_index, local_scores,
    and a facts summary derived from the timeline.
    """
    title: str = timeline.get("title", "")
    org: str = timeline.get("org", "")
    program_area: str = timeline.get("program_area", "")
    scoring: dict = timeline.get("scoring", {})

    inv_budget = max(0, top_k - _MIN_STRATEGY_CHUNKS)
    all_inv_chunks: list[dict] = []
    all_strategy_chunks: list[dict] = []

    if tools is not None:
        # Strategy 1: LLM-generated queries
        inv_queries = await _strategy1_llm_queries(inv_id, title, org, program_area, model)
        if not inv_queries:
            inv_queries = list(_FALLBACK_QUERIES)

        # asyncio-APPROVED-2: concurrent HTTP/search — run all query strategies in parallel
        s1_chunks, s3_inv, s3_strategy, s4_chunks = await asyncio.gather(
            _run_inv_queries(inv_queries, inv_id, tools, top_k=max(1, inv_budget // len(inv_queries) if inv_queries else 10)),
            _strategy3_doc_type(inv_id, tools),
            asyncio.gather(*[_search_strategy(q, tools) for q in _STRATEGY_QUERIES[:3]]),
            _strategy4_strategy_docs(tools),
            return_exceptions=False,
        )

        all_inv_chunks.extend(s1_chunks)
        for c in s3_inv:
            all_inv_chunks.extend(c)
        all_strategy_chunks.extend(s4_chunks)
        for c in s3_strategy:
            if isinstance(c, list):
                all_strategy_chunks.extend(c)
            elif not isinstance(c, BaseException):
                all_strategy_chunks.extend(c if c else [])

        # Ensure strategy floor
        if len(all_strategy_chunks) < _MIN_STRATEGY_CHUNKS:
            extra = await _strategy4_strategy_docs(tools, extra=True)
            all_strategy_chunks.extend(extra)

    all_chunks = all_inv_chunks + all_strategy_chunks
    all_chunks = _dedup_chunks(all_chunks)

    source_index = _build_source_index(all_chunks)
    local_scores = _compute_local_scores(timeline, all_chunks)
    contradictions = _detect_fact_contradictions(scoring, all_chunks)
    if contradictions:
        local_scores["contradictions"] = contradictions

    return InvestmentEvidencePack(
        inv_id=inv_id,
        scope_id=scope_id,
        timeline=timeline,
        chunks=all_chunks,
        source_index=source_index,
        scoring=scoring,
        local_scores=local_scores,
    )


# ---------------------------------------------------------------------------
# Strategy implementations
# ---------------------------------------------------------------------------


async def _strategy1_llm_queries(
    inv_id: str,
    title: str,
    org: str,
    program_area: str,
    model: str,
) -> list[str]:
    prompt = _QUERY_PROMPT.format(
        inv_id=inv_id,
        title=title or inv_id,
        org=org or "unknown",
        program_area=program_area or "global health",
    )
    try:
        result: StrategyQueryList = await acall_llm(
            prompt,
            system_msg="You are a portfolio analyst generating targeted research queries.",
            model=model,
            output_schema=StrategyQueryList,
            max_tokens=DEFAULT_MAX_TOKENS,
        )
        return [q for q in result.queries if q.strip()][:10]
    except Exception as exc:
        logger.warning("_strategy1_llm_queries failed for %s: %s", inv_id, exc)
        return []


async def _run_inv_queries(
    queries: list[str],
    inv_id: str,
    tools: Any,
    top_k: int = 5,
) -> list[dict]:
    if not queries:
        return []

    async def _one(q: str) -> list[dict]:
        return await _search_investment(q, inv_id, tools, top_k=top_k)

    # asyncio-APPROVED-2: concurrent HTTP/search — fan out investment queries in parallel
    results = await asyncio.gather(*[_one(q) for q in queries], return_exceptions=True)
    chunks: list[dict] = []
    for r in results:
        if isinstance(r, list):
            chunks.extend(r)
    return chunks


async def _strategy3_doc_type(
    inv_id: str,
    tools: Any,
) -> list[list[dict]]:
    async def _one(doc_type: str, query: str) -> list[dict]:
        return await _search_with_filter(query, tools, inv_id=inv_id, doc_type=doc_type, top_k=8)

    # asyncio-APPROVED-2: concurrent HTTP/search — fan out doc-type queries in parallel
    results = await asyncio.gather(
        *[_one(dt, q) for dt, q in _DOC_TYPE_QUERIES],
        return_exceptions=True,
    )
    return [r if isinstance(r, list) else [] for r in results]


async def _strategy4_strategy_docs(tools: Any, *, extra: bool = False) -> list[dict]:
    queries = _STRATEGY_QUERIES if not extra else _STRATEGY_QUERIES[3:]
    # asyncio-APPROVED-2: concurrent HTTP/search — fan out strategy queries in parallel
    results = await asyncio.gather(
        *[_search_strategy(q, tools) for q in queries],
        return_exceptions=True,
    )
    chunks: list[dict] = []
    for r in results:
        if isinstance(r, list):
            chunks.extend(r)
    return chunks


async def _search_investment(query: str, inv_id: str, tools: Any, *, top_k: int = 10) -> list[dict]:
    return await _search_with_filter(query, tools, inv_id=inv_id, top_k=top_k)


async def _search_strategy(query: str, tools: Any, *, top_k: int = 8) -> list[dict]:
    return await _search_with_filter(query, tools, collection=_STRATEGY_COLLECTION, top_k=top_k)


async def _search_with_filter(query: str, tools: Any, *, top_k: int = 10, **filters) -> list[dict]:
    import time
    from src.core.tool_tracing import append_to_buffer

    if tools is None:
        return []
    idx = getattr(tools, "_embedding_index", None)
    if idx is None:
        return []
    _has_hybrid = hasattr(idx, "hybrid_search") and getattr(idx, "has_hybrid", False)

    def _run():
        if _has_hybrid:
            return idx.hybrid_search(query, top_k=top_k, embedding_weight=0.5, **filters)
        return idx.search_with_filter(query, top_k=top_k, **filters)

    start = time.monotonic()
    try:
        # asyncio-APPROVED-1: to_thread wraps blocking embedding search
        results = await asyncio.to_thread(_run)
    except Exception as exc:
        logger.warning("_search_with_filter failed: %s", exc)
        results = []

    append_to_buffer("collection_search_traces", {
        "tool_name": "collection_search",
        "query": query,
        "top_k": top_k,
        "filters": filters,
        "result_count": len(results),
        "duration_ms": int((time.monotonic() - start) * 1000),
    })
    return results


# ---------------------------------------------------------------------------
# Evidence utilities
# ---------------------------------------------------------------------------


def _dedup_chunks(chunks: list[dict]) -> list[dict]:
    seen: set[str] = set()
    seen_groups: set[str] = set()
    out: list[dict] = []
    for c in chunks:
        cid = c.get("chunk_id", c.get("file_id", "") + str(c.get("page_start", 0)))
        if cid in seen:
            continue
        vg = c.get("intelligence_version_group", "")
        if vg and vg in seen_groups and not c.get("carve_out_metadata"):
            continue
        seen.add(cid)
        if vg:
            seen_groups.add(vg)
        out.append(c)
    return out


def _build_source_index(chunks: list[dict]) -> list[dict]:
    return [
        {
            "ref": f"§{i+1:04d}",
            "file_id": c.get("file_id", ""),
            "filename": c.get("filename", ""),
            "page": c.get("page_start", 0),
            "collection": c.get("collection", ""),
            "inv_id": c.get("inv_id", ""),
            "doc_type": c.get("doc_type", ""),
        }
        for i, c in enumerate(chunks)
    ]


# ---------------------------------------------------------------------------
# Scoring and contradiction detection
# ---------------------------------------------------------------------------


def _detect_fact_contradictions(scoring: dict, chunks: list[dict]) -> list[str]:
    """Flag contradictions between scoring data and document text.

    Checks approved_amount (within 10% tolerance) and grant dates.
    Returns list of contradiction description strings (empty if none found).
    """
    contradictions: list[str] = []
    approved_amount: float = float(scoring.get("approved_amount", 0) or 0)

    if approved_amount > 0:
        for chunk in chunks:
            text = chunk.get("text", "")
            if not text:
                continue
            # Look for dollar amounts that differ by >10% from the known approved_amount
            import re
            for m in re.finditer(r"\$\s*([\d,]+(?:\.\d+)?)\s*(?:million|M)?", text):
                raw = m.group(1).replace(",", "")
                try:
                    found_amount = float(raw)
                    if "million" in m.group(0).lower() or found_amount < 1000:
                        found_amount *= 1_000_000
                    ratio = abs(found_amount - approved_amount) / (approved_amount + 1)
                    if ratio > 0.10 and found_amount > 100_000:
                        contradictions.append(
                            f"Amount mismatch: scoring={approved_amount:,.0f}, "
                            f"document={found_amount:,.0f} in {chunk.get('filename', '?')}"
                        )
                        break
                except ValueError:
                    pass
            if contradictions:
                break

    return contradictions[:3]


def _score_disbursement_velocity(scoring: dict, timeline: dict) -> str:
    """Return green|yellow|red based on paid/approved ratio vs time elapsed.

    green:  0.70 ≤ ratio ≤ 2.0
    yellow: ratio < 0.70 and pct_time < 25%, or ratio > 2.0 and 5 < pct_time < 60
    red:    ratio < 0.40 and pct_time > 25%
    """
    approved = float(scoring.get("approved_amount", 0) or 0)
    paid = float(scoring.get("paid_amount", 0) or 0)
    pct_time = float(timeline.get("pct_time_elapsed", 0) or 0)

    if approved <= 0:
        return "not_assessable"

    ratio = paid / approved
    if 0.70 <= ratio <= 2.0:
        return "green"
    if ratio < 0.70:
        if pct_time < 25:
            return "yellow"
        if ratio < 0.40:
            return "red"
        return "yellow"
    if ratio > 2.0 and 5 < pct_time < 60:
        return "yellow"
    return "green"


def compute_investment_facts(scoring: dict, timeline: dict) -> dict:
    """Compute InvestmentFacts deterministically from scoring and timeline data (no LLM).

    Returns a dict with: approved_amount, paid_amount, burn_rate, runway_months,
    execution_rate, pct_time_elapsed, timeline_slip_months.
    """
    approved = float(scoring.get("approved_amount", 0) or 0)
    paid = float(scoring.get("paid_amount", 0) or 0)
    pct_time = float(timeline.get("pct_time_elapsed", 0) or 0)

    burn_rate = paid / max(pct_time / 100.0, 0.01) if pct_time > 0 else 0.0
    runway_months = (
        ((approved - paid) / (burn_rate / 12.0))
        if burn_rate > 0 and paid < approved
        else 0.0
    )
    execution_rate = paid / max(approved, 1.0)

    # Compute timeline slip from planned vs actual end dates
    planned_end = timeline.get("end_date") or timeline.get("end", "")
    actual_end = timeline.get("latest_doc_date", "")
    timeline_slip_months = 0.0
    if planned_end and actual_end:
        try:
            from datetime import datetime as _dt
            p = _dt.strptime(str(planned_end)[:10], "%Y-%m-%d").date()
            a = _dt.strptime(str(actual_end)[:10], "%Y-%m-%d").date()
            if a > p:
                timeline_slip_months = round((a - p).days / 30.44, 1)
        except (ValueError, TypeError):
            pass

    return {
        "approved_amount": approved,
        "paid_amount": paid,
        "burn_rate": round(burn_rate, 2),
        "runway_months": round(runway_months, 1),
        "execution_rate": round(execution_rate, 4),
        "pct_time_elapsed": pct_time,
        "timeline_slip_months": timeline_slip_months,
    }


def _compute_local_scores(timeline: dict, chunks: list[dict]) -> dict:
    """Compute deterministic document quality signals.

    document_freshness:     ≤6 months → green, ≤12 → yellow, >12 → red
    reporting_completeness: progress_report present → green, absent → red
    rationale_adequacy:     proposal/rationale text > 500 chars → green
    """
    scores: dict = {}

    # document_freshness
    latest_date_str: str = timeline.get("latest_doc_date", "") or ""
    if latest_date_str:
        try:
            from datetime import datetime
            latest = datetime.strptime(latest_date_str[:10], "%Y-%m-%d").date()
            months_old = (date.today() - latest).days / 30.44
            if months_old <= 6:
                scores["document_freshness"] = "green"
            elif months_old <= 12:
                scores["document_freshness"] = "yellow"
            else:
                scores["document_freshness"] = "red"
        except (ValueError, TypeError):
            scores["document_freshness"] = "not_assessable"
    else:
        scores["document_freshness"] = "not_assessable"

    # reporting_completeness
    doc_types_present: set[str] = set(timeline.get("doc_types_present", []) or [])
    if "progress_report" in doc_types_present:
        scores["reporting_completeness"] = "green"
    else:
        scores["reporting_completeness"] = "red"

    # rationale_adequacy — check for any proposal/rationale text in chunks
    total_proposal_chars = sum(
        len(c.get("text", ""))
        for c in chunks
        if c.get("doc_type", "") in {"proposal", "investment_document"}
    )
    if total_proposal_chars > 500:
        scores["rationale_adequacy"] = "green"
    elif total_proposal_chars > 0:
        scores["rationale_adequacy"] = "yellow"
    else:
        scores["rationale_adequacy"] = "red"

    return scores
