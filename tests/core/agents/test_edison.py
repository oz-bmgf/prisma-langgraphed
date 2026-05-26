"""Unit tests for src/core/agents/edison.py."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.agents.edison import EdisonQueryResult, _parse_edison_result, run


# ---------------------------------------------------------------------------
# run() — missing API key
# ---------------------------------------------------------------------------


async def test_run_no_api_key_returns_no_api_key_status():
    with patch("src.core.agents.edison.EDISON_API_KEY", ""):
        result = await run(task_id="T1", query="malaria vaccines", linked_scope="S1")

    assert result["status"] == "no_api_key"
    assert result["task_id"] == "T1"
    assert result["papers"] == []


# ---------------------------------------------------------------------------
# run() — SDK not installed
# ---------------------------------------------------------------------------


async def test_run_sdk_missing_returns_no_api_key():
    import sys

    with patch("src.core.agents.edison.EDISON_API_KEY", "test-key"), \
         patch.dict(sys.modules, {"edison_client": None}):
        result = await run(task_id="T1", query="q")

    assert result["status"] in ("no_api_key", "error")


# ---------------------------------------------------------------------------
# run() — timeout
# ---------------------------------------------------------------------------


async def test_run_timeout_returns_error():
    import asyncio

    async def _slow(*a, **kw):
        await asyncio.sleep(9999)

    with patch("src.core.agents.edison.EDISON_API_KEY", "test-key"), \
         patch("src.core.agents.edison._query_edison", new=_slow):
        result = await run(task_id="T1", query="q", timeout=1)

    assert result["status"] == "error"
    assert "timed out" in (result["error_message"] or "")


# ---------------------------------------------------------------------------
# _parse_edison_result — result parsing
# ---------------------------------------------------------------------------


def test_parse_result_none_returns_no_response():
    result = _parse_edison_result("T1", "q", None)
    assert result.status == "no_response"


def test_parse_result_empty_papers_returns_no_evidence():
    raw = MagicMock()
    raw.papers = []
    raw.synthesis = "Some synthesis."
    result = _parse_edison_result("T1", "q", raw)
    assert result.status == "no_evidence"
    assert result.thesis == "Some synthesis."


def test_parse_result_with_papers_returns_ok():
    paper = MagicMock()
    paper.paperId = "P1"
    paper.title = "Vaccine efficacy"
    paper.year = 2022
    paper.authors = "Smith J"
    paper.abstract = "RCT study showing 87% efficacy."
    paper.url = "https://example.com/paper"

    raw = MagicMock()
    raw.papers = [paper]
    raw.synthesis = "Strong evidence for vaccine."
    del raw.results  # ensure only .papers is used

    result = _parse_edison_result("T1", "q", raw)
    assert result.status == "ok"
    assert len(result.papers) == 1
    assert result.papers[0].title == "Vaccine efficacy"


def test_parse_result_dict_format():
    raw = {
        "papers": [{"paperId": "P1", "title": "Study", "year": 2020}],
        "synthesis": "Synthesis text.",
    }
    result = _parse_edison_result("T1", "q", raw)
    assert result.status == "ok"
    assert result.papers[0].paperId == "P1"
    assert result.thesis == "Synthesis text."
