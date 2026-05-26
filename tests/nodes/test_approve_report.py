"""Unit tests for src/graph/nodes/approve_report.py."""
from __future__ import annotations

from unittest.mock import patch

import pytest

from src.graph.nodes.approve_report import approve_report


def _make_state(**overrides) -> dict:
    base = {
        "program": "Malaria",
        "final_report_wresearch_md_path": "/tmp/final_report_wresearch.md",
        "report_approved": None,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# "approve" → report_approved=True
# ---------------------------------------------------------------------------


async def test_approve_returns_true():
    with patch("src.graph.nodes.approve_report.interrupt", return_value="approve"):
        result = await approve_report(_make_state(), {})
    assert result == {"report_approved": True}


# ---------------------------------------------------------------------------
# "revise" → report_approved=False
# ---------------------------------------------------------------------------


async def test_revise_returns_false():
    with patch("src.graph.nodes.approve_report.interrupt", return_value="revise"):
        result = await approve_report(_make_state(), {})
    assert result == {"report_approved": False}


# ---------------------------------------------------------------------------
# unrecognised → treat as approve
# ---------------------------------------------------------------------------


async def test_unrecognised_treats_as_approve():
    with patch("src.graph.nodes.approve_report.interrupt", return_value="something_unknown"):
        result = await approve_report(_make_state(), {})
    assert result == {"report_approved": True}


# ---------------------------------------------------------------------------
# interrupt payload has expected fields
# ---------------------------------------------------------------------------


async def test_interrupt_payload_fields():
    captured = {}

    def _capture(payload):
        captured.update(payload)
        return "approve"

    with patch("src.graph.nodes.approve_report.interrupt", side_effect=_capture):
        await approve_report(_make_state(), {})

    assert captured["stage"] == "approve_report"
    assert captured["program"] == "Malaria"
    assert "question" in captured
