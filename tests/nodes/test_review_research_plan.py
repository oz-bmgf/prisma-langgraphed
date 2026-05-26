"""Unit tests for src/graph/nodes/review_research_plan.py."""
from __future__ import annotations

from unittest.mock import patch

import pytest

from src.graph.nodes.review_research_plan import review_research_plan


def _make_state(**overrides) -> dict:
    base = {
        "program": "Malaria",
        "research_plan_md_path": "/tmp/research_plan.md",
        "research_plan": [
            {"id": "RQ-001", "query": "Q1", "type": "slr"},
            {"id": "RQ-002", "query": "Q2", "type": "deep_web"},
            {"id": "RQ-003", "query": "Q3", "type": "lbd"},
        ],
        "research_plan_approved": None,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# "approve" → approved=True
# ---------------------------------------------------------------------------


async def test_approve_returns_approved_true():
    with patch("src.graph.nodes.review_research_plan.interrupt", return_value="approve"):
        result = await review_research_plan(_make_state(), {})
    assert result == {"research_plan_approved": True}


# ---------------------------------------------------------------------------
# "regenerate" → approved=False
# ---------------------------------------------------------------------------


async def test_regenerate_returns_approved_false():
    with patch("src.graph.nodes.review_research_plan.interrupt", return_value="regenerate"):
        result = await review_research_plan(_make_state(), {})
    assert result == {"research_plan_approved": False}


# ---------------------------------------------------------------------------
# {"prune": [...]} → plan pruned, approved=True
# ---------------------------------------------------------------------------


async def test_prune_removes_tasks_and_approves():
    with patch("src.graph.nodes.review_research_plan.interrupt",
               return_value={"prune": ["RQ-002"]}):
        result = await review_research_plan(_make_state(), {})

    assert result["research_plan_approved"] is True
    remaining_ids = [t["id"] for t in result["research_plan"]]
    assert "RQ-001" in remaining_ids
    assert "RQ-003" in remaining_ids
    assert "RQ-002" not in remaining_ids


# ---------------------------------------------------------------------------
# unrecognised value → treat as approve
# ---------------------------------------------------------------------------


async def test_unrecognised_value_treats_as_approve():
    with patch("src.graph.nodes.review_research_plan.interrupt", return_value="something_else"):
        result = await review_research_plan(_make_state(), {})
    assert result == {"research_plan_approved": True}


# ---------------------------------------------------------------------------
# interrupt payload has expected fields
# ---------------------------------------------------------------------------


async def test_interrupt_payload_fields():
    captured = {}

    def _capture_interrupt(payload):
        captured.update(payload)
        return "approve"

    with patch("src.graph.nodes.review_research_plan.interrupt", side_effect=_capture_interrupt):
        await review_research_plan(_make_state(), {})

    assert captured["stage"] == "review_research_plan"
    assert captured["program"] == "Malaria"
    assert captured["task_count"] == 3
    assert "question" in captured
