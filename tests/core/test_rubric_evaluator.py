"""Unit tests for src/core/rubric_evaluator.py."""
from __future__ import annotations

from datetime import date
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.rubric_evaluator import (
    _compute_local_scores,
    _dedup_chunks,
    _detect_fact_contradictions,
    _score_disbursement_velocity,
    build_evidence_pack,
)
from src.core.output_schemas import StrategyQueryList


# ---------------------------------------------------------------------------
# _dedup_chunks
# ---------------------------------------------------------------------------


def test_dedup_chunks_removes_exact_duplicates():
    chunks = [
        {"file_id": "f1", "text": "a"},
        {"file_id": "f1", "text": "a"},
        {"file_id": "f2", "text": "b"},
    ]
    result = _dedup_chunks(chunks)
    assert len(result) == 2


def test_dedup_chunks_preserves_order_of_first_occurrence():
    chunks = [
        {"file_id": "f3", "text": "c"},
        {"file_id": "f1", "text": "a"},
        {"file_id": "f1", "text": "a-dup"},
    ]
    result = _dedup_chunks(chunks)
    assert result[0]["file_id"] == "f3"


def test_dedup_chunks_respects_version_group():
    chunks = [
        {"intelligence_version_group": "g1", "file_id": "f1", "text": "v1"},
        {"intelligence_version_group": "g1", "file_id": "f2", "text": "v2-same-group"},
        {"intelligence_version_group": "g2", "file_id": "f3", "text": "v3-different"},
    ]
    result = _dedup_chunks(chunks)
    assert len(result) == 2
    assert result[1]["intelligence_version_group"] == "g2"


# ---------------------------------------------------------------------------
# _score_disbursement_velocity
# ---------------------------------------------------------------------------


def test_score_disbursement_velocity_green_on_track():
    scoring = {"approved_amount": 1_000_000, "paid_amount": 800_000}
    timeline = {"pct_time_elapsed": 80}
    assert _score_disbursement_velocity(scoring, timeline) == "green"


def test_score_disbursement_velocity_red_underspend_late():
    scoring = {"approved_amount": 1_000_000, "paid_amount": 200_000}
    timeline = {"pct_time_elapsed": 80}
    assert _score_disbursement_velocity(scoring, timeline) == "red"


def test_score_disbursement_velocity_yellow_early_underspend():
    scoring = {"approved_amount": 1_000_000, "paid_amount": 300_000}
    timeline = {"pct_time_elapsed": 10}
    assert _score_disbursement_velocity(scoring, timeline) == "yellow"


def test_score_disbursement_velocity_not_assessable_when_no_approved():
    assert _score_disbursement_velocity({"approved_amount": 0}, {}) == "not_assessable"


# ---------------------------------------------------------------------------
# _compute_local_scores
# ---------------------------------------------------------------------------


def test_compute_local_scores_freshness_green_recent():
    today_str = date.today().strftime("%Y-%m-%d")
    timeline = {"latest_doc_date": today_str, "doc_types_present": ["progress_report"]}
    scores = _compute_local_scores(timeline, [])
    assert scores["document_freshness"] == "green"


def test_compute_local_scores_freshness_red_stale():
    timeline = {"latest_doc_date": "2020-01-01", "doc_types_present": []}
    scores = _compute_local_scores(timeline, [])
    assert scores["document_freshness"] == "red"


def test_compute_local_scores_reporting_completeness_red_no_progress_report():
    timeline = {"doc_types_present": ["proposal"]}
    scores = _compute_local_scores(timeline, [])
    assert scores["reporting_completeness"] == "red"


def test_compute_local_scores_rationale_adequacy_green():
    chunks = [{"doc_type": "proposal", "text": "x" * 600}]
    timeline = {"doc_types_present": []}
    scores = _compute_local_scores(timeline, chunks)
    assert scores["rationale_adequacy"] == "green"


# ---------------------------------------------------------------------------
# _detect_fact_contradictions
# ---------------------------------------------------------------------------


def test_detect_fact_contradictions_no_contradiction():
    scoring = {"approved_amount": 5_000_000}
    chunks = [{"text": "The project received $5,000,000 in total.", "filename": "proposal.pdf"}]
    result = _detect_fact_contradictions(scoring, chunks)
    assert result == []


def test_detect_fact_contradictions_empty_scoring():
    result = _detect_fact_contradictions({}, [{"text": "$1,000,000"}])
    assert result == []


# ---------------------------------------------------------------------------
# build_evidence_pack
# ---------------------------------------------------------------------------


async def test_build_evidence_pack_no_tools_returns_empty_pack():
    result = await build_evidence_pack(
        inv_id="INV-TEST",
        scope_id="S-01",
        timeline={"doc_types_present": ["progress_report"], "latest_doc_date": "2025-01-01"},
        tools=None,
    )
    assert result.inv_id == "INV-TEST"
    assert result.scope_id == "S-01"
    assert result.chunks == []


async def test_build_evidence_pack_computes_local_scores():
    result = await build_evidence_pack(
        inv_id="INV-SCORES",
        scope_id="S-02",
        timeline={
            "doc_types_present": ["progress_report", "proposal"],
            "latest_doc_date": date.today().strftime("%Y-%m-%d"),
        },
        tools=None,
    )
    assert "document_freshness" in result.local_scores
    assert result.local_scores["document_freshness"] == "green"
