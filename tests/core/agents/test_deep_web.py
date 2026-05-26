"""Unit tests for src/core/agents/deep_web.py."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.agents.deep_web import DeepWebResult, deep_web_research, run


# ---------------------------------------------------------------------------
# deep_web_research — primary path
# ---------------------------------------------------------------------------


async def test_deep_web_research_primary_success():
    with patch(
        "src.core.agents.deep_web._primary_research",
        new=AsyncMock(
            return_value=DeepWebResult(
                question="What is malaria vaccine efficacy?",
                answer="Vaccine efficacy is 87% in RCTs.",
                success=True,
                model_used="o3-deep-research",
            )
        ),
    ):
        result = await deep_web_research("What is malaria vaccine efficacy?")

    assert result.success is True
    assert "87%" in result.answer
    assert result.model_used == "o3-deep-research"


async def test_deep_web_research_primary_falls_back_on_error():
    with patch(
        "src.core.agents.deep_web._primary_research",
        new=AsyncMock(side_effect=RuntimeError("primary fail")),
    ), patch(
        "src.core.agents.deep_web._fallback_research",
        new=AsyncMock(
            return_value=DeepWebResult(
                question="q",
                answer="fallback answer",
                success=True,
                model_used="gpt-4o",
            )
        ),
    ):
        result = await deep_web_research("What is malaria vaccine efficacy?")

    assert result.success is True
    assert result.answer == "fallback answer"


async def test_deep_web_research_both_fail_returns_failure():
    with patch("src.core.agents.deep_web._primary_research", new=AsyncMock(side_effect=RuntimeError("primary fail"))):
        with patch("src.core.agents.deep_web._fallback_research", new=AsyncMock(side_effect=RuntimeError("fallback fail"))):
            result = await deep_web_research("test question")

    assert result.success is False
    assert result.error_message is not None
    assert result.answer == ""


async def test_deep_web_research_timeout_falls_back():
    import asyncio

    async def _slow(*a, **kw):
        await asyncio.sleep(9999)
        return DeepWebResult(question="q", answer="never", success=True)

    with patch("src.core.agents.deep_web._primary_research", new=_slow):
        with patch(
            "src.core.agents.deep_web._fallback_research",
            new=AsyncMock(
                return_value=DeepWebResult(question="q", answer="fallback ok", success=True)
            ),
        ):
            result = await deep_web_research("q", timeout=1)

    assert result.success is True
    assert result.answer == "fallback ok"


# ---------------------------------------------------------------------------
# run() — worker entry point
# ---------------------------------------------------------------------------


async def test_run_returns_expected_keys():
    with patch(
        "src.core.agents.deep_web.deep_web_research",
        new=AsyncMock(
            return_value=DeepWebResult(
                question="q",
                answer="the answer",
                sources=["https://example.com"],
                success=True,
            )
        ),
    ):
        result = await run(task_id="T1", query="q", linked_scope="S1")

    assert result["task_id"] == "T1"
    assert result["task_type"] == "deep_web"
    assert result["linked_scope"] == "S1"
    assert result["result"] == "the answer"
    assert result["content"] == "the answer"
    assert result["success"] is True


async def test_run_error_result_propagated():
    with patch(
        "src.core.agents.deep_web.deep_web_research",
        new=AsyncMock(
            return_value=DeepWebResult(
                question="q",
                answer="",
                success=False,
                error_message="API unreachable",
            )
        ),
    ):
        result = await run(task_id="T2", query="q")

    assert result["success"] is False
    assert result["error_message"] == "API unreachable"
