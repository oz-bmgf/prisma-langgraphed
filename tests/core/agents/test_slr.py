"""Unit tests for src/core/agents/slr.py."""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from src.core.agents.slr import SLRResult, run


_MOCK_PAPERS = [
    {"paperId": "P1", "title": "Malaria vaccine RCT", "year": 2022, "authors": "Smith J", "abstract": "Efficacy 85%."},
    {"paperId": "P2", "title": "Insecticide resistance", "year": 2021, "authors": "Jones K", "abstract": "IR spreading."},
]


# ---------------------------------------------------------------------------
# run() — happy path
# ---------------------------------------------------------------------------


async def test_run_returns_thesis_and_results():
    with patch("src.tools.literature_tools.OpenAlexClient") as MockOpenAlex, \
         patch("src.tools.literature_tools.AstaClient") as MockAsta, \
         patch("src.core.agents.slr.acall_llm", new=AsyncMock(return_value="Synthesised thesis.")):
        MockOpenAlex.return_value.search = AsyncMock(return_value=_MOCK_PAPERS)
        MockAsta.return_value.search = AsyncMock(return_value=[])
        result = await run(task_id="T1", query="malaria vaccines", linked_scope="S1")

    assert result["task_id"] == "T1"
    assert result["task_type"] == "slr"
    assert result["thesis"] == "Synthesised thesis."
    assert len(result["results"]) == 2
    assert result["success"] is True


async def test_run_deduplicates_papers():
    duplicate = dict(_MOCK_PAPERS[0])  # same title as P1
    with patch("src.tools.literature_tools.OpenAlexClient") as MockOpenAlex, \
         patch("src.tools.literature_tools.AstaClient") as MockAsta, \
         patch("src.core.agents.slr.acall_llm", new=AsyncMock(return_value="Thesis.")):
        MockOpenAlex.return_value.search = AsyncMock(return_value=_MOCK_PAPERS)
        MockAsta.return_value.search = AsyncMock(return_value=[duplicate])
        result = await run(task_id="T1", query="malaria")

    # duplicate title should be filtered out
    titles = [p["title"] for p in result["results"]]
    assert titles.count("Malaria vaccine RCT") == 1


async def test_run_no_papers_returns_empty_thesis():
    with patch("src.tools.literature_tools.OpenAlexClient") as MockOpenAlex, \
         patch("src.tools.literature_tools.AstaClient") as MockAsta:
        MockOpenAlex.return_value.search = AsyncMock(return_value=[])
        MockAsta.return_value.search = AsyncMock(return_value=[])
        result = await run(task_id="T1", query="obscure query")

    assert result["success"] is True
    assert "No papers found" in result["thesis"]
    assert result["results"] == []


# ---------------------------------------------------------------------------
# run() — error / timeout handling
# ---------------------------------------------------------------------------


async def test_run_search_exception_still_succeeds_via_other_source():
    with patch("src.tools.literature_tools.OpenAlexClient") as MockOpenAlex, \
         patch("src.tools.literature_tools.AstaClient") as MockAsta, \
         patch("src.core.agents.slr.acall_llm", new=AsyncMock(return_value="Thesis from Asta only.")):
        MockOpenAlex.return_value.search = AsyncMock(side_effect=RuntimeError("OpenAlex down"))
        MockAsta.return_value.search = AsyncMock(return_value=_MOCK_PAPERS)
        result = await run(task_id="T1", query="q")

    assert result["success"] is True
    assert result["thesis"] == "Thesis from Asta only."


async def test_run_timeout_returns_failure():
    import asyncio

    async def _slow(*a, **kw):
        await asyncio.sleep(9999)

    with patch("src.core.agents.slr._run_slr", new=_slow):
        result = await run(task_id="T1", query="q", timeout=1)

    assert result["success"] is False
    assert "timed out" in (result["error_message"] or "")


async def test_run_unexpected_exception_returns_failure():
    with patch("src.core.agents.slr._run_slr", new=AsyncMock(side_effect=ValueError("oops"))):
        result = await run(task_id="T1", query="q")

    assert result["success"] is False
    assert "oops" in (result["error_message"] or "")
