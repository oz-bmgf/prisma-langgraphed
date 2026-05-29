"""Unit tests for src/core/causal_model.py."""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from src.core.causal_model import (
    _RISK_MATRIX,
    _SCIENCE_KEYWORDS,
    forecast_consequences,
    make_investigation_claims,
    rank_assumptions,
)
from src.core.evidence_model import CausalLink, CausalModel, ScoredAssumption
from src.core.output_schemas import (
    CausalLinkSchema,
    CausalModelExtraction,
    ConsequenceForecast,
    ForecastOutput,
    RankedAssumptionsOutput,
    AssumptionRisk,
)


# ---------------------------------------------------------------------------
# _RISK_MATRIX
# ---------------------------------------------------------------------------


def test_risk_matrix_has_nine_entries():
    assert len(_RISK_MATRIX) == 9


def test_risk_matrix_terminal_high_is_rank_1():
    _label, sort_key = _RISK_MATRIX[("terminal", "high")]
    assert sort_key == 1


def test_risk_matrix_minor_low_is_rank_9():
    _label, sort_key = _RISK_MATRIX[("minor", "low")]
    assert sort_key == 9


def test_risk_matrix_terminal_low_is_medium():
    label, _sort_key = _RISK_MATRIX[("terminal", "low")]
    assert label == "medium"


def test_risk_matrix_all_sort_keys_unique():
    sort_keys = [sk for _, sk in _RISK_MATRIX.values()]
    assert sorted(sort_keys) == list(range(1, 10))


# ---------------------------------------------------------------------------
# rank_assumptions
# ---------------------------------------------------------------------------


async def test_rank_assumptions_sorts_by_risk_rank():
    cm = CausalModel(
        inv_id="INV-001",
        theory_of_change="A causes B causes C.",
        links=[
            CausalLink(
                name="A→B",
                from_stage="ACTIVITIES",
                to_stage="OUTPUTS",
                mechanism="A enables B",
                assumptions=["A happens", "B follows"],
            )
        ],
    )

    mock_output = RankedAssumptionsOutput(
        assumptions=[
            AssumptionRisk(
                assumption="B follows",
                causal_link="A→B",
                consequence="minor",
                uncertainty="low",
                if_wrong="Small delay",
                investigation_question="Does B happen after A?",
            ),
            AssumptionRisk(
                assumption="A happens",
                causal_link="A→B",
                consequence="terminal",
                uncertainty="high",
                if_wrong="Investment fails entirely",
                investigation_question="Is A being executed?",
            ),
        ]
    )

    with patch("src.core.causal_model.acall_structured", new=AsyncMock(return_value=mock_output)):
        result = await rank_assumptions(cm, model="claude-haiku-4-5-20251001")

    assert len(result.assumptions) == 2
    assert result.assumptions[0].risk_rank < result.assumptions[1].risk_rank
    assert result.assumptions[0].assumption == "A happens"


async def test_rank_assumptions_sets_risk_rank_from_matrix():
    cm = CausalModel(
        inv_id="INV-002",
        links=[
            CausalLink(name="X→Y", assumptions=["Assumption X"], from_stage="FUNDING", to_stage="ACTIVITIES")
        ],
    )
    mock_output = RankedAssumptionsOutput(
        assumptions=[
            AssumptionRisk(
                assumption="Assumption X",
                causal_link="X→Y",
                consequence="major",
                uncertainty="moderate",
                if_wrong="Major setback",
                investigation_question="Is X valid?",
            )
        ]
    )
    with patch("src.core.causal_model.acall_structured", new=AsyncMock(return_value=mock_output)):
        result = await rank_assumptions(cm, model="claude-haiku-4-5-20251001")

    expected_rank = _RISK_MATRIX[("major", "moderate")][1]
    assert result.assumptions[0].risk_rank == expected_rank


async def test_rank_assumptions_empty_links_returns_unchanged():
    cm = CausalModel(inv_id="INV-003", links=[])
    result = await rank_assumptions(cm, model="claude-haiku-4-5-20251001")
    assert result.assumptions == []


# ---------------------------------------------------------------------------
# forecast_consequences
# ---------------------------------------------------------------------------


async def test_forecast_consequences_clips_at_approved_amount():
    cm = CausalModel(
        inv_id="INV-004",
        links=[CausalLink(name="A→B", from_stage="ACTIVITIES", to_stage="OUTPUTS")],
    )
    mock_output = ForecastOutput(
        forecasts=[
            ConsequenceForecast(
                link_name="A→B",
                dollars_at_risk=99_000_000.0,
                months_at_risk=12.0,
                rationale="High exposure",
            )
        ]
    )
    scoring = {"approved_amount": 5_000_000.0}
    timeline = {"duration_months": 24}

    with patch("src.core.causal_model.acall_structured", new=AsyncMock(return_value=mock_output)):
        result = await forecast_consequences(
            cm, scoring=scoring, timeline=timeline, model="claude-haiku-4-5-20251001"
        )

    assert result.links[0].dollars_at_risk == pytest.approx(5_000_000.0)


async def test_forecast_consequences_uses_max_not_sum():
    cm = CausalModel(
        inv_id="INV-005",
        links=[CausalLink(name="Link1", from_stage="OUTPUTS", to_stage="OUTCOMES")],
    )
    mock_output = ForecastOutput(
        forecasts=[
            ConsequenceForecast(link_name="Link1", dollars_at_risk=100_000.0, months_at_risk=3.0, rationale="a"),
            ConsequenceForecast(link_name="Link1", dollars_at_risk=200_000.0, months_at_risk=6.0, rationale="b"),
        ]
    )
    scoring = {"approved_amount": 1_000_000.0}
    timeline = {"duration_months": 36}

    with patch("src.core.causal_model.acall_structured", new=AsyncMock(return_value=mock_output)):
        result = await forecast_consequences(
            cm, scoring=scoring, timeline=timeline, model="claude-haiku-4-5-20251001"
        )

    assert result.links[0].dollars_at_risk == pytest.approx(200_000.0)
    assert result.links[0].months_at_risk == pytest.approx(6.0)


async def test_forecast_consequences_handles_llm_failure_gracefully():
    cm = CausalModel(
        inv_id="INV-006",
        links=[CausalLink(name="X→Z", from_stage="ACTIVITIES", to_stage="IMPACT")],
    )
    with patch("src.core.causal_model.acall_structured", new=AsyncMock(side_effect=RuntimeError("LLM down"))):
        result = await forecast_consequences(
            cm, scoring={"approved_amount": 500_000.0}, timeline={"duration_months": 12},
            model="claude-haiku-4-5-20251001",
        )

    assert result.links[0].dollars_at_risk == 0.0


# ---------------------------------------------------------------------------
# make_investigation_claims
# ---------------------------------------------------------------------------


def test_make_investigation_claims_task_id_format():
    cm = CausalModel(
        inv_id="INV-075775",
        links=[CausalLink(name="A→B", assumptions=["assumption one"])],
        assumptions=[
            ScoredAssumption(
                assumption="assumption one",
                causal_link="A→B",
                investigation_question="Is A→B valid?",
                risk_rank=3,
            )
        ],
    )
    claims = make_investigation_claims(cm, bow_id="B02878", inv_id="INV-075775")
    assert len(claims) == 1
    assert claims[0]["task_id"] == "INV-075775-assumption-001"


def test_make_investigation_claims_science_keywords_add_hint():
    cm = CausalModel(
        inv_id="INV-777",
        assumptions=[
            ScoredAssumption(
                assumption="Vaccine efficacy is at least 70%",
                causal_link="Vaccination→Immunity",
                investigation_question="What is the vaccine efficacy?",
                risk_rank=2,
            )
        ],
    )
    claims = make_investigation_claims(cm, bow_id="B111", inv_id="INV-777")
    assert "web_search_hint" in claims[0]


def test_make_investigation_claims_non_science_no_hint():
    cm = CausalModel(
        inv_id="INV-888",
        assumptions=[
            ScoredAssumption(
                assumption="Finance reporting submitted on time",
                causal_link="Finance→Accountability",
                investigation_question="Are reports submitted on schedule?",
                risk_rank=7,
            )
        ],
    )
    claims = make_investigation_claims(cm, bow_id="B222", inv_id="INV-888")
    assert "web_search_hint" not in claims[0]
