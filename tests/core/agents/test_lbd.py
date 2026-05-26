"""Unit tests for src/core/agents/lbd.py."""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from src.core.agents.lbd import LBDResult, _deduplicate, _parse_terms, run


_MOCK_PAPERS = [
    {"paperId": "P1", "title": "Malaria and iron deficiency", "year": 2020, "authors": "A", "abstract": "Iron deficiency affects malaria."},
    {"paperId": "P2", "title": "Iron deficiency and anaemia", "year": 2019, "authors": "B", "abstract": "Iron links to anaemia."},
]


# ---------------------------------------------------------------------------
# run() — happy path
# ---------------------------------------------------------------------------


async def test_run_returns_thesis_and_concepts():
    with patch("src.core.agents.lbd.AstaClient") as MockAsta, \
         patch("src.core.agents.lbd.acall_llm") as mock_llm:
        MockAsta.return_value.search = AsyncMock(return_value=_MOCK_PAPERS)
        mock_llm.side_effect = [
            "iron deficiency, nutritional status",  # concept extraction
            "anaemia, haemoglobin",                  # B-term extraction
            "Indirect pathway thesis.",              # synthesis
        ]

        result = await run(task_id="T1", query="malaria anaemia connection", linked_scope="S1")

    assert result["task_id"] == "T1"
    assert result["task_type"] == "lbd"
    assert isinstance(result["thesis"], str)
    assert len(result["concepts"]) > 0
    assert result["success"] is True


async def test_run_timeout_returns_failure():
    import asyncio

    async def _slow(*a, **kw):
        await asyncio.sleep(9999)

    with patch("src.core.agents.lbd._run_lbd", new=_slow):
        result = await run(task_id="T1", query="q", timeout=1)

    assert result["success"] is False
    assert "timed out" in (result["error_message"] or "")


async def test_run_exception_returns_failure():
    with patch("src.core.agents.lbd._run_lbd", new=AsyncMock(side_effect=RuntimeError("lbd failed"))):
        result = await run(task_id="T1", query="q")

    assert result["success"] is False
    assert "lbd failed" in (result["error_message"] or "")


# ---------------------------------------------------------------------------
# _parse_terms helper
# ---------------------------------------------------------------------------


def test_parse_terms_comma_separated():
    terms = _parse_terms("iron, anaemia, malaria")
    assert "iron" in terms
    assert "anaemia" in terms


def test_parse_terms_json_list():
    terms = _parse_terms('["iron deficiency", "anaemia", "malaria parasitaemia"]')
    assert len(terms) == 3
    assert "anaemia" in terms


def test_parse_terms_empty_returns_empty():
    terms = _parse_terms("")
    assert terms == []


# ---------------------------------------------------------------------------
# _deduplicate helper
# ---------------------------------------------------------------------------


def test_deduplicate_removes_same_title():
    papers = [
        {"title": "Malaria and iron", "paperId": "P1"},
        {"title": "Malaria and iron", "paperId": "P2"},  # duplicate title
        {"title": "Other paper", "paperId": "P3"},
    ]
    result = _deduplicate(papers)
    assert len(result) == 2
    assert result[0]["paperId"] == "P1"  # first wins


def test_deduplicate_preserves_order():
    papers = [
        {"title": "Alpha", "paperId": "A"},
        {"title": "Beta", "paperId": "B"},
        {"title": "Gamma", "paperId": "C"},
    ]
    result = _deduplicate(papers)
    assert [p["paperId"] for p in result] == ["A", "B", "C"]
