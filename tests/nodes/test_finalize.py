"""Unit tests for src/graph/nodes/finalize.py."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from src.graph.nodes.finalize import finalize


def _make_state(**overrides) -> dict:
    base = {
        "synthesis_model": "claude-sonnet-4-6",
        "final_report_md": None,
        "research_results": [],
        "threads_dir": None,
        "output_dir": None,
    }
    base.update(overrides)
    return base


_SAMPLE_REPORT = """\
## Executive Summary

This is the executive summary.

## Key Insights

Key insight 1. Key insight 2.

## Scope A: Vaccines

Content about vaccines.

### Recommended Research

- Need more data.
"""

_SAMPLE_RESULTS = [
    {
        "task_id": "RQ-001",
        "status": "ok",
        "query": "vaccine efficacy studies",
        "channel": "slr",
        "linked_scope": "Scope A",
        "thesis": "Strong evidence found for vaccine efficacy.",
    }
]


# ---------------------------------------------------------------------------
# no research results — report unchanged
# ---------------------------------------------------------------------------


async def test_no_research_skips_enrichment():
    result = await finalize(
        _make_state(final_report_md=_SAMPLE_REPORT, research_results=[]),
        {},
    )
    assert result["final_report_wresearch_md"] == _SAMPLE_REPORT


# ---------------------------------------------------------------------------
# enriches report with research results
# ---------------------------------------------------------------------------


async def test_enriches_with_research(tmp_path):
    call_count = 0
    llm_responses = [
        "External evidence confirms strong efficacy.",  # scope enrichment
        "Key insight supplement based on research.",    # key findings
        "Updated executive summary with research.",     # exec summary
    ]

    async def _mock_acall_llm(*args, **kwargs):
        nonlocal call_count
        r = llm_responses[call_count % len(llm_responses)]
        call_count += 1
        return r

    with patch("src.graph.nodes.finalize.acall_llm", side_effect=_mock_acall_llm):
        result = await finalize(
            _make_state(
                final_report_md=_SAMPLE_REPORT,
                research_results=_SAMPLE_RESULTS,
                threads_dir=str(tmp_path),
            ),
            {},
        )

    enriched = result["final_report_wresearch_md"]
    assert enriched is not None
    assert len(enriched) > len(_SAMPLE_REPORT)


# ---------------------------------------------------------------------------
# writes file to disk
# ---------------------------------------------------------------------------


async def test_writes_file_to_disk(tmp_path):
    async def _mock_acall_llm(*args, **kwargs):
        return "Generated content " * 20  # > 200 chars for exec summary threshold

    with patch("src.graph.nodes.finalize.acall_llm", side_effect=_mock_acall_llm):
        result = await finalize(
            _make_state(
                final_report_md=_SAMPLE_REPORT,
                research_results=_SAMPLE_RESULTS,
                threads_dir=str(tmp_path),
            ),
            {},
        )

    assert result["final_report_wresearch_md_path"] is not None
    assert Path(result["final_report_wresearch_md_path"]).exists()


# ---------------------------------------------------------------------------
# no output dir — path is None
# ---------------------------------------------------------------------------


async def test_no_threads_dir_path_is_none():
    result = await finalize(
        _make_state(final_report_md=_SAMPLE_REPORT, research_results=[]),
        {},
    )
    assert result["final_report_wresearch_md_path"] is None


# ---------------------------------------------------------------------------
# returns expected keys
# ---------------------------------------------------------------------------


async def test_returns_expected_keys():
    result = await finalize(_make_state(), {})
    assert set(result.keys()) == {"final_report_wresearch_md_path", "final_report_wresearch_md"}
