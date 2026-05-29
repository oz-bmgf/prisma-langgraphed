"""Unit tests for src/core/investigation.py."""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from src.core.investigation import (
    _TERMINAL_STATUSES,
    _dedup_chunks,
    _validate_coverage_audit,
    run_investigation,
)
from src.core.output_schemas import InvestigationAction, InvestigationActionsOutput


# ---------------------------------------------------------------------------
# _dedup_chunks
# ---------------------------------------------------------------------------


def test_dedup_chunks_removes_existing_by_file_id():
    existing = [{"file_id": "f1", "text": "a"}]
    new = [{"file_id": "f1", "text": "a"}, {"file_id": "f2", "text": "b"}]
    result = _dedup_chunks(new, existing)
    assert len(result) == 1
    assert result[0]["file_id"] == "f2"


def test_dedup_chunks_removes_by_chunk_id():
    existing = [{"chunk_id": "c1", "text": "x"}]
    new = [{"chunk_id": "c1", "text": "x"}, {"chunk_id": "c2", "text": "y"}]
    result = _dedup_chunks(new, existing)
    assert len(result) == 1
    assert result[0]["chunk_id"] == "c2"


def test_dedup_chunks_empty_existing():
    new = [{"file_id": "f1", "text": "a"}, {"file_id": "f2", "text": "b"}]
    result = _dedup_chunks(new, [])
    assert len(result) == 2


def test_dedup_chunks_removes_version_group_duplicates():
    existing = [{"intelligence_version_group": "grp1", "file_id": "f-existing", "text": "old"}]
    new = [
        {"intelligence_version_group": "grp1", "file_id": "f-new1", "text": "new-same-group"},
        {"intelligence_version_group": "grp2", "file_id": "f-new2", "text": "new-different"},
    ]
    result = _dedup_chunks(new, existing)
    assert len(result) == 1
    assert result[0]["intelligence_version_group"] == "grp2"


# ---------------------------------------------------------------------------
# _validate_coverage_audit
# ---------------------------------------------------------------------------


def test_validate_coverage_audit_passes_when_l4_disabled():
    with patch("src.core.investigation.INVESTIGATION_L4_COVERAGE_AUDIT", False):
        output = InvestigationActionsOutput(status="answered", confidence="high", answer="")
        assert _validate_coverage_audit(output) is True


def test_validate_coverage_audit_fails_when_l4_enabled_missing_topics():
    with patch("src.core.investigation.INVESTIGATION_L4_COVERAGE_AUDIT", True):
        output = InvestigationActionsOutput(
            status="answered", confidence="high", answer="The link looks fine."
        )
        assert _validate_coverage_audit(output) is False


def test_validate_coverage_audit_passes_when_all_topics_covered():
    with patch("src.core.investigation.INVESTIGATION_L4_COVERAGE_AUDIT", True):
        answer = (
            "disbursement rate is 60%. "
            "milestone completion on track. "
            "grantee capacity strong. "
            "external policy context stable. "
            "evidence quality: reporting submitted and no contradictions."
        )
        output = InvestigationActionsOutput(status="answered", confidence="high", answer=answer)
        assert _validate_coverage_audit(output) is True


# ---------------------------------------------------------------------------
# run_investigation
# ---------------------------------------------------------------------------


async def test_run_investigation_returns_investigation_result():
    mock_output = InvestigationActionsOutput(
        status="answered",
        confidence="high",
        answer="The link is well supported.",
        evidence_refs=["§0001"],
        next_actions=[],
    )
    with patch("src.core.investigation.acall_structured", new=AsyncMock(return_value=mock_output)):
        result = await run_investigation(
            link_id="link-001",
            inv_id="INV-001",
            bow_id="B001",
            scope_id="S-01",
            claim={"name": "Training→Capacity", "from_stage": "ACTIVITIES", "to_stage": "OUTPUTS"},
            model="claude-haiku-4-5-20251001",
        )
    assert result.terminal_status == "answered"
    assert result.model == "claude-haiku-4-5-20251001"
    assert result.iterations >= 1


async def test_run_investigation_stops_on_terminal_status():
    call_count = 0

    async def mock_llm(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return InvestigationActionsOutput(
            status="answered",
            confidence="high",
            answer="Done.",
            next_actions=[],
        )

    with patch("src.core.investigation.acall_structured", new=mock_llm):
        result = await run_investigation(
            link_id="link-002",
            inv_id="INV-002",
            bow_id="B002",
            scope_id="S-02",
            claim={"name": "X→Y"},
            model="claude-haiku-4-5-20251001",
            max_iterations=10,
        )
    assert call_count == 1
    assert result.terminal_status == "answered"


async def test_run_investigation_saturation_stops_loop():
    iteration_count = 0

    async def mock_llm(*args, **kwargs):
        nonlocal iteration_count
        iteration_count += 1
        return InvestigationActionsOutput(
            status="partially_answered",
            confidence="low",
            answer="Still looking.",
            next_actions=[InvestigationAction(tool="search_investment", query="test query")],
        )

    with patch("src.core.investigation.acall_structured", new=mock_llm), \
         patch("src.core.investigation._execute_actions", new=AsyncMock(return_value=([], 0))):
        result = await run_investigation(
            link_id="link-003",
            inv_id="INV-003",
            bow_id="B003",
            scope_id="S-03",
            claim={"name": "A→B"},
            model="claude-haiku-4-5-20251001",
            max_iterations=10,
        )
    # Each call returns actions → execute → empty chunks; loop breaks after threshold
    from src.config import CONSECUTIVE_EMPTY_THRESHOLD
    assert iteration_count == CONSECUTIVE_EMPTY_THRESHOLD


async def test_run_investigation_no_tools_returns_empty_evidence():
    mock_output = InvestigationActionsOutput(
        status="not_answerable",
        confidence="insufficient",
        answer="No documents available.",
        next_actions=[],
    )
    with patch("src.core.investigation.acall_structured", new=AsyncMock(return_value=mock_output)):
        result = await run_investigation(
            link_id="link-004",
            inv_id="INV-004",
            bow_id="B004",
            scope_id="S-04",
            claim={"name": "Z→W"},
            model="claude-haiku-4-5-20251001",
        )
    assert result.total_chunks_retrieved == 0
    assert result.web_searches == 0


async def test_run_investigation_result_has_to_dict():
    mock_output = InvestigationActionsOutput(
        status="answered", confidence="high", answer="Done.", next_actions=[]
    )
    with patch("src.core.investigation.acall_structured", new=AsyncMock(return_value=mock_output)):
        result = await run_investigation(
            link_id="link-005",
            inv_id="INV-005",
            bow_id="B005",
            scope_id="S-05",
            claim={"name": "M→N"},
            model="claude-haiku-4-5-20251001",
        )
    d = result.to_dict()
    assert isinstance(d, dict)
    assert "terminal_status" in d
