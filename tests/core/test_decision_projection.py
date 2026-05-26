"""Unit tests for src/core/decision_projection.py."""
from __future__ import annotations

import math
from unittest.mock import AsyncMock, patch

import pytest

from src.config import DECISION_MAX_PER_INV, DECISION_MAX_PER_SCOPE
from src.core.decision_projection import (
    DECISION_TYPE_VOCABULARY,
    _THIN_EVIDENCE_DECISION_TYPES,
    _apply_caps,
    _compute_rank_score,
    _sanitize_candidate,
    _section1a_gate,
    project_decisions,
)
from src.core.evidence_model import Decision
from src.core.output_schemas import DecisionCandidate, DecisionProjectionOutput


# ---------------------------------------------------------------------------
# DECISION_TYPE_VOCABULARY
# ---------------------------------------------------------------------------


def test_vocabulary_has_15_types():
    assert len(DECISION_TYPE_VOCABULARY) == 15


def test_thin_evidence_types_are_subset_of_vocabulary():
    assert _THIN_EVIDENCE_DECISION_TYPES.issubset(DECISION_TYPE_VOCABULARY)


# ---------------------------------------------------------------------------
# _sanitize_candidate
# ---------------------------------------------------------------------------


def test_sanitize_accepts_valid_candidate():
    c = DecisionCandidate(
        decision_type="monitor",
        recommended_action="Watch quarterly progress",
        triggering_link_ids=["link-001"],
        corroboration_count=2,
        cost_impact_dollars=100_000,
        timeline_impact_months=3,
        urgency="medium_term",
        materiality="medium",
        confidence="medium",
    )
    assert _sanitize_candidate(c) is not None


def test_sanitize_rejects_invalid_type():
    c = DecisionCandidate(
        decision_type="buy_more",
        recommended_action="Do something",
        triggering_link_ids=["link-001"],
        corroboration_count=1,
        cost_impact_dollars=0,
        timeline_impact_months=0,
        urgency="immediate",
        materiality="high",
        confidence="high",
    )
    assert _sanitize_candidate(c) is None


def test_sanitize_rejects_empty_recommended_action():
    c = DecisionCandidate(
        decision_type="monitor",
        recommended_action="",
        triggering_link_ids=["link-001"],
        corroboration_count=1,
        cost_impact_dollars=0,
        timeline_impact_months=0,
        urgency="immediate",
        materiality="high",
        confidence="high",
    )
    assert _sanitize_candidate(c) is None


def test_sanitize_rejects_empty_triggering_links():
    c = DecisionCandidate(
        decision_type="monitor",
        recommended_action="Watch it",
        triggering_link_ids=[],
        corroboration_count=1,
        cost_impact_dollars=0,
        timeline_impact_months=0,
        urgency="immediate",
        materiality="high",
        confidence="high",
    )
    assert _sanitize_candidate(c) is None


# ---------------------------------------------------------------------------
# _section1a_gate
# ---------------------------------------------------------------------------


def test_section1a_gate_passes_thin_evidence_type():
    c = DecisionCandidate(
        decision_type="request_progress_packet",
        recommended_action="Request report",
        triggering_link_ids=["l1"],
        corroboration_count=0,
        cost_impact_dollars=0,
        timeline_impact_months=0,
        urgency="immediate",
        materiality="low",
        confidence="low",
    )
    assert _section1a_gate(c) is True


def test_section1a_gate_passes_high_confidence():
    c = DecisionCandidate(
        decision_type="terminate_unless_resolved",
        recommended_action="Terminate",
        triggering_link_ids=["l1"],
        corroboration_count=1,
        cost_impact_dollars=0,
        timeline_impact_months=0,
        urgency="immediate",
        materiality="high",
        confidence="high",
    )
    assert _section1a_gate(c) is True


def test_section1a_gate_rejects_low_confidence_low_corroboration():
    c = DecisionCandidate(
        decision_type="terminate_unless_resolved",
        recommended_action="Terminate",
        triggering_link_ids=["l1"],
        corroboration_count=1,
        cost_impact_dollars=0,
        timeline_impact_months=0,
        urgency="immediate",
        materiality="high",
        confidence="low",
    )
    assert _section1a_gate(c) is False


def test_section1a_gate_passes_low_confidence_with_corroboration():
    c = DecisionCandidate(
        decision_type="terminate_unless_resolved",
        recommended_action="Terminate",
        triggering_link_ids=["l1", "l2"],
        corroboration_count=2,
        cost_impact_dollars=0,
        timeline_impact_months=0,
        urgency="immediate",
        materiality="high",
        confidence="low",
    )
    assert _section1a_gate(c) is True


# ---------------------------------------------------------------------------
# _compute_rank_score
# ---------------------------------------------------------------------------


def test_compute_rank_score_ordering():
    high = DecisionCandidate(
        decision_type="monitor", recommended_action="act", triggering_link_ids=["l1"],
        corroboration_count=5, cost_impact_dollars=1_000_000,
        timeline_impact_months=6, urgency="immediate", materiality="high", confidence="high",
    )
    low = DecisionCandidate(
        decision_type="monitor", recommended_action="act", triggering_link_ids=["l1"],
        corroboration_count=1, cost_impact_dollars=1_000,
        timeline_impact_months=1, urgency="long_term", materiality="low", confidence="low",
    )
    assert _compute_rank_score(high) > _compute_rank_score(low)


# ---------------------------------------------------------------------------
# _apply_caps
# ---------------------------------------------------------------------------


def _make_decision(inv_id: str, rank: float = 1.0) -> Decision:
    return Decision(
        inv_id=inv_id,
        decision_type="monitor",
        recommended_action="Watch",
        triggering_link_ids=["l1"],
        rank_score=rank,
    )


def test_apply_caps_respects_per_inv_limit():
    decisions = [_make_decision("INV-001") for _ in range(DECISION_MAX_PER_INV + 2)]
    result = _apply_caps(decisions, inv_id="INV-001")
    inv_count = sum(1 for d in result if d.inv_id == "INV-001")
    assert inv_count == DECISION_MAX_PER_INV


def test_apply_caps_respects_scope_limit():
    decisions = [_make_decision(f"INV-{i:03d}") for i in range(DECISION_MAX_PER_SCOPE + 5)]
    result = _apply_caps(decisions, inv_id="")
    assert len(result) == DECISION_MAX_PER_SCOPE


def test_apply_caps_empty_inv_id_not_counted():
    decisions = [_make_decision("") for _ in range(DECISION_MAX_PER_INV + 2)]
    result = _apply_caps(decisions, inv_id="")
    assert len(result) == min(len(decisions), DECISION_MAX_PER_SCOPE)


# ---------------------------------------------------------------------------
# project_decisions
# ---------------------------------------------------------------------------


async def test_project_decisions_returns_dict_with_scope_id():
    mock_output = DecisionProjectionOutput(
        decisions=[
            DecisionCandidate(
                decision_type="monitor",
                recommended_action="Schedule quarterly review",
                goal_link="Training→Outcomes",
                triggering_link_ids=["link-001"],
                corroboration_count=2,
                cost_impact_dollars=500_000,
                timeline_impact_months=3,
                urgency="medium_term",
                materiality="medium",
                confidence="medium",
            )
        ]
    )
    with patch("src.core.decision_projection.acall_llm", new=AsyncMock(return_value=mock_output)):
        result = await project_decisions(
            "S-01",
            {"inv_id": "INV-001", "link_assessments": [], "science_flags": []},
            model="claude-haiku-4-5-20251001",
        )
    assert result["scope_id"] == "S-01"
    assert len(result["decisions"]) == 1


async def test_project_decisions_section1a_gate_drops_low_evidence():
    mock_output = DecisionProjectionOutput(
        decisions=[
            DecisionCandidate(
                decision_type="terminate_unless_resolved",
                recommended_action="Terminate immediately",
                triggering_link_ids=["link-001"],
                corroboration_count=1,
                cost_impact_dollars=0,
                timeline_impact_months=0,
                urgency="immediate",
                materiality="high",
                confidence="low",
            )
        ]
    )
    with patch("src.core.decision_projection.acall_llm", new=AsyncMock(return_value=mock_output)):
        result = await project_decisions(
            "S-02",
            {"inv_id": "INV-002", "link_assessments": [], "science_flags": []},
            model="claude-haiku-4-5-20251001",
        )
    assert len(result["decisions"]) == 0


async def test_project_decisions_handles_llm_failure():
    with patch("src.core.decision_projection.acall_llm", new=AsyncMock(side_effect=RuntimeError("down"))):
        result = await project_decisions(
            "S-03",
            {"inv_id": "INV-003"},
            model="claude-haiku-4-5-20251001",
        )
    assert result["scope_id"] == "S-03"
    assert result["decisions"] == []
