"""Unit tests for src/core/science_investigator.py."""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from src.core.science_investigator import (
    _build_gate_note,
    _has_text_evidence,
    investigate_science_question,
)
from src.core.output_schemas import ScienceAction, ScienceActionsOutput


# ---------------------------------------------------------------------------
# _has_text_evidence
# ---------------------------------------------------------------------------


def test_has_text_evidence_confirms_keyword():
    from src.core.science_investigator import _CONFIRM_KEYWORDS
    assert _has_text_evidence("The study confirms the hypothesis.", _CONFIRM_KEYWORDS) is True


def test_has_text_evidence_disconfirms_keyword():
    from src.core.science_investigator import _DISCONFIRM_KEYWORDS
    assert _has_text_evidence("No evidence was found in the trial.", _DISCONFIRM_KEYWORDS) is True


def test_has_text_evidence_no_match():
    from src.core.science_investigator import _CONFIRM_KEYWORDS
    assert _has_text_evidence("The study was conducted in 2020.", _CONFIRM_KEYWORDS) is False


# ---------------------------------------------------------------------------
# _build_gate_note
# ---------------------------------------------------------------------------


def test_gate_note_asta_not_called():
    note = _build_gate_note(asta_called=False, confirming=False, disconfirming=False)
    assert "NOT been called" in note


def test_gate_note_asta_satisfied_confirming():
    note = _build_gate_note(asta_called=True, confirming=True, disconfirming=False)
    assert "Confirming evidence found" in note
    assert "ASTA gate SATISFIED" in note


def test_gate_note_asta_satisfied_both():
    note = _build_gate_note(asta_called=True, confirming=True, disconfirming=True)
    assert "Both confirming and disconfirming" in note


# ---------------------------------------------------------------------------
# investigate_science_question
# ---------------------------------------------------------------------------


async def test_asta_gate_blocks_early_evidence_gathered():
    """evidence_gathered without ASTA call should force another iteration."""
    call_count = 0

    async def mock_llm(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # First call: try to finish without ASTA
            return ScienceActionsOutput(
                status="evidence_gathered",
                confirming_evidence_found=True,
                answer="Found it.",
                next_actions=[],
            )
        else:
            # Second call: proper terminal
            return ScienceActionsOutput(
                status="insufficient_evidence",
                answer="No ASTA results found.",
                next_actions=[],
            )

    with patch("src.core.science_investigator.acall_structured", new=mock_llm):
        result = await investigate_science_question(
            assumption_id="A-001",
            inv_id="INV-001",
            bow_id="B001",
            scope_id="S-01",
            question="Is the vaccine effective?",
            model="claude-haiku-4-5-20251001",
        )

    # Gate should have blocked the first evidence_gathered, requiring a second call
    assert call_count >= 2


async def test_asta_soft_cap_skips_excess_calls():
    """ASTA actions beyond soft cap are silently skipped."""
    from src.config import ASTA_SOFT_CAP

    call_count = 0

    async def mock_llm(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count <= ASTA_SOFT_CAP + 2:
            return ScienceActionsOutput(
                status="continue",
                next_actions=[ScienceAction(tool="search_asta", query="vaccine efficacy")],
            )
        return ScienceActionsOutput(status="evidence_gathered", answer="Done.", next_actions=[])

    from unittest.mock import MagicMock
    mock_search_asta = MagicMock()
    mock_search_asta.ainvoke = AsyncMock(return_value="Paper title. Abstract text.")

    with patch("src.core.science_investigator.acall_structured", new=mock_llm), \
         patch("src.core.science_investigator.search_asta", new=mock_search_asta):
        result = await investigate_science_question(
            assumption_id="A-002",
            inv_id="INV-002",
            bow_id="B002",
            scope_id="S-02",
            question="Does the treatment work?",
            model="claude-haiku-4-5-20251001",
        )

    assert result.asta_calls <= ASTA_SOFT_CAP


async def test_consecutive_empty_forces_insufficient_evidence():
    """3 consecutive empty rounds should force insufficient_evidence."""
    mock_llm = AsyncMock(return_value=ScienceActionsOutput(
        status="continue",
        next_actions=[ScienceAction(tool="search_investment", query="test")],
    ))

    with patch("src.core.science_investigator.acall_structured", new=mock_llm), \
         patch("src.core.science_investigator._execute_actions", new=AsyncMock(return_value=([], 0))):
        result = await investigate_science_question(
            assumption_id="A-003",
            inv_id="INV-003",
            bow_id="B003",
            scope_id="S-03",
            question="Is efficacy proven?",
            model="claude-haiku-4-5-20251001",
        )

    assert result.terminal_status == "insufficient_evidence"


async def test_investigate_science_returns_result_with_to_dict():
    mock_llm = AsyncMock(return_value=ScienceActionsOutput(
        status="insufficient_evidence",
        answer="Nothing found.",
        next_actions=[],
    ))

    with patch("src.core.science_investigator.acall_structured", new=mock_llm):
        result = await investigate_science_question(
            assumption_id="A-004",
            inv_id="INV-004",
            bow_id="B004",
            scope_id="S-04",
            question="What does the literature say?",
            model="claude-haiku-4-5-20251001",
        )

    d = result.to_dict()
    assert isinstance(d, dict)
    assert "terminal_status" in d
    assert result.terminal_status == "insufficient_evidence"


async def test_asta_client_called_for_search_asta_action():
    from unittest.mock import MagicMock
    mock_asta = MagicMock()
    mock_asta.search = AsyncMock(return_value=[
        {"title": "Vaccine Trial", "abstract": "Shows 90% efficacy.", "paperId": "V1"}
    ])
    call_count = 0

    async def mock_llm(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return ScienceActionsOutput(
                status="continue",
                next_actions=[ScienceAction(tool="search_asta", query="vaccine efficacy RCT")],
            )
        return ScienceActionsOutput(
            status="evidence_gathered",
            confirming_evidence_found=True,
            answer="Literature supports the assumption.",
            next_actions=[],
        )

    with patch("src.core.science_investigator.acall_structured", new=mock_llm):
        result = await investigate_science_question(
            assumption_id="A-005",
            inv_id="INV-005",
            bow_id="B005",
            scope_id="S-05",
            question="Is the vaccine at least 70% efficacious?",
            asta_client=mock_asta,
            model="claude-haiku-4-5-20251001",
        )

    assert result.asta_calls >= 1
    mock_asta.search.assert_called()


async def test_no_asta_client_does_not_crash():
    mock_llm = AsyncMock(return_value=ScienceActionsOutput(
        status="insufficient_evidence",
        answer="No results.",
        next_actions=[],
    ))

    with patch("src.core.science_investigator.acall_structured", new=mock_llm):
        result = await investigate_science_question(
            assumption_id="A-006",
            inv_id="INV-006",
            bow_id="B006",
            scope_id="S-06",
            question="Any evidence?",
            asta_client=None,
            model="claude-haiku-4-5-20251001",
        )

    assert result.asta_calls == 0
