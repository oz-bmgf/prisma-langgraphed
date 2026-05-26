"""Unit tests for src/graph/nodes/prepare_research.py."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from src.graph.nodes.prepare_research import prepare_research


def _make_state(**overrides) -> dict:
    base = {
        "research_model": "claude-sonnet-4-6",
        "synthesis_model": "claude-sonnet-4-6",
        "output_dir": None,
        "threads_dir": None,
        "final_report_md": None,
        "analyst_report": None,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# returns expected keys
# ---------------------------------------------------------------------------


async def test_returns_expected_keys():
    _de_novo_response = json.dumps([
        {"query": "What is the evidence for vaccine efficacy?", "type": "slr",
         "rationale": "Key question", "priority": "critical", "linked_scope": "S1"}
    ])
    _review_response = json.dumps([
        {"id": "RQ-000", "status": "keep"}
    ])

    responses = [_de_novo_response, _review_response]
    call_count = 0

    async def _mock_acall_llm(*args, **kwargs):
        nonlocal call_count
        r = responses[call_count % len(responses)]
        call_count += 1
        return r

    with patch("src.graph.nodes.prepare_research.acall_llm", side_effect=_mock_acall_llm):
        result = await prepare_research(
            _make_state(final_report_md="Some report text"),
            {},
        )

    assert "research_plan" in result
    assert "research_plan_md_path" in result


# ---------------------------------------------------------------------------
# extracts from analyst report
# ---------------------------------------------------------------------------


async def test_extracts_from_analyst_report():
    analyst_report = {
        "threads": [
            {
                "id": "S1",
                "deep_research_needed": [
                    {"question": "What is efficacy?", "why": "Gap", "expected_impact": "Changes score"}
                ],
                "gaps": ["Need more data on X"],
            }
        ]
    }

    _review_response = json.dumps([{"id": "RQ-000", "status": "keep"},
                                    {"id": "RQ-001", "status": "keep"}])

    async def _mock_acall_llm(*args, **kwargs):
        return _review_response

    with patch("src.graph.nodes.prepare_research.acall_llm", side_effect=_mock_acall_llm):
        result = await prepare_research(
            _make_state(analyst_report=analyst_report),
            {},
        )

    plan = result["research_plan"]
    assert isinstance(plan, list)
    assert len(plan) >= 2
    queries = [item["query"] for item in plan]
    assert any("efficacy" in q.lower() for q in queries)


# ---------------------------------------------------------------------------
# writes files to disk
# ---------------------------------------------------------------------------


async def test_writes_files_to_disk(tmp_path):
    out_dir = tmp_path / "output"

    _de_novo_response = json.dumps([
        {"query": "Test question", "type": "slr", "priority": "important", "linked_scope": ""}
    ])
    _review_response = json.dumps([{"id": "RQ-000", "status": "keep"}])
    responses = [_de_novo_response, _review_response]
    call_count = 0

    async def _mock_acall_llm(*args, **kwargs):
        nonlocal call_count
        r = responses[call_count % len(responses)]
        call_count += 1
        return r

    with patch("src.graph.nodes.prepare_research.acall_llm", side_effect=_mock_acall_llm):
        result = await prepare_research(
            _make_state(output_dir=str(out_dir), final_report_md="some report"),
            {},
        )

    assert result["research_plan_md_path"] is not None
    assert Path(result["research_plan_md_path"]).exists()


# ---------------------------------------------------------------------------
# no output dir — paths are None
# ---------------------------------------------------------------------------


async def test_no_output_dir_paths_are_none():
    async def _mock_acall_llm(*args, **kwargs):
        return json.dumps([])

    with patch("src.graph.nodes.prepare_research.acall_llm", side_effect=_mock_acall_llm):
        result = await prepare_research(_make_state(), {})

    assert result["research_plan_md_path"] is None
    assert isinstance(result["research_plan"], list)


# ---------------------------------------------------------------------------
# plan items have required fields
# ---------------------------------------------------------------------------


async def test_plan_items_have_required_fields():
    analyst_report = {
        "threads": [{
            "id": "S1",
            "deep_research_needed": [
                {"question": "Q1?", "why": "reason", "expected_impact": "big"}
            ],
            "gaps": [],
        }]
    }

    _review_response = json.dumps([{"id": "RQ-000", "status": "keep"}])

    async def _mock_acall_llm(*args, **kwargs):
        return _review_response

    with patch("src.graph.nodes.prepare_research.acall_llm", side_effect=_mock_acall_llm):
        result = await prepare_research(_make_state(analyst_report=analyst_report), {})

    for item in result["research_plan"]:
        assert "id" in item
        assert "query" in item
        assert "type" in item
        assert item["id"].startswith("RQ-")
