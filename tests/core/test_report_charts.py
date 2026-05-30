"""Unit tests for src/core/report_charts.py."""
from __future__ import annotations

import base64

import pytest

from src.core.report_charts import render_confusion_matrix, render_scatter_plot

# ---------------------------------------------------------------------------
# Shared minimal fixtures
# ---------------------------------------------------------------------------

_SCOPE_A = {
    "scope_id": "s1",
    "inv_ids": ["INV-001"],
    "bow_ids": ["BOW-01"],
    "investment_report": {"severity": "pathway_altering"},
    "investment_facts": {
        "risk_severity": "aligned",
        "execution_rate": 0.65,
        "approved_amount": 5_000_000.0,
    },
}

_SCOPE_B = {
    "scope_id": "s2",
    "inv_ids": ["INV-002"],
    "bow_ids": ["BOW-01"],
    "investment_report": {"severity": "aligned"},
    "investment_facts": {
        "risk_severity": "aligned",
        "execution_rate": 0.85,
        "approved_amount": 8_000_000.0,
    },
}

_SCOPE_C = {
    "scope_id": "s3",
    "inv_ids": ["INV-003"],
    "bow_ids": ["BOW-02"],
    "investment_report": {"severity": "program_critical"},
    "investment_facts": {
        "risk_severity": "program_critical",
        "execution_rate": 0.30,
        "approved_amount": 12_000_000.0,
    },
}


# ---------------------------------------------------------------------------
# render_confusion_matrix
# ---------------------------------------------------------------------------


def test_pnd_matrix_returns_base64():
    result = render_confusion_matrix([_SCOPE_A, _SCOPE_B], {})
    assert result is not None
    assert isinstance(result, str)
    assert len(result) > 100
    # Verify it is valid base64
    decoded = base64.b64decode(result)
    assert decoded[:4] == b"\x89PNG"  # PNG magic bytes


def test_pnd_matrix_returns_none_on_empty():
    result = render_confusion_matrix([], {})
    assert result is None


def test_pnd_matrix_single_scope_returns_png():
    # 1 data point is enough to render the matrix
    result = render_confusion_matrix([_SCOPE_A], {})
    assert result is not None
    assert base64.b64decode(result)[:4] == b"\x89PNG"


def test_pnd_matrix_investment_scoring_arg_is_accepted():
    # investment_scoring is accepted even when non-empty (not used for severity matrix)
    scoring = {"INV-001": {"team_exec_inv": "Meets Expectations"}}
    result = render_confusion_matrix([_SCOPE_A, _SCOPE_B], scoring)
    assert result is not None


def test_pnd_matrix_three_scopes():
    result = render_confusion_matrix([_SCOPE_A, _SCOPE_B, _SCOPE_C], {})
    assert result is not None
    assert base64.b64decode(result)[:4] == b"\x89PNG"


# ---------------------------------------------------------------------------
# render_scatter_plot
# ---------------------------------------------------------------------------


def test_scatter_plot_returns_base64():
    result = render_scatter_plot([_SCOPE_A, _SCOPE_B], {})
    assert result is not None
    assert isinstance(result, str)
    decoded = base64.b64decode(result)
    assert decoded[:4] == b"\x89PNG"


def test_scatter_plot_single_point_returns_png():
    # One valid scope → 1 plotable point → still returns a PNG
    result = render_scatter_plot([_SCOPE_A], {})
    assert result is not None
    assert base64.b64decode(result)[:4] == b"\x89PNG"


def test_scatter_plot_returns_none_on_empty():
    result = render_scatter_plot([], {})
    assert result is None


def test_scatter_plot_all_bows_in_one_plot():
    # Scopes from different BOWs — all rendered in a single plot
    result = render_scatter_plot([_SCOPE_A, _SCOPE_B, _SCOPE_C], {})
    assert result is not None
    assert base64.b64decode(result)[:4] == b"\x89PNG"


def test_scatter_plot_custom_axes():
    result = render_scatter_plot(
        [_SCOPE_A, _SCOPE_B],
        {},
        x_axis="execution_rate",
        y_axis="approved_amount",
    )
    assert result is not None


def test_scatter_plot_falls_back_to_investment_scoring():
    # Scopes with no investment_facts; values come from investment_scoring dict
    scope_no_facts_a = {
        "scope_id": "sx1",
        "inv_ids": ["INV-010"],
        "bow_ids": ["BOW-99"],
        "investment_report": {"severity": "aligned"},
        "investment_facts": {},
    }
    scope_no_facts_b = {
        "scope_id": "sx2",
        "inv_ids": ["INV-011"],
        "bow_ids": ["BOW-99"],
        "investment_report": {"severity": "efficiency_reducing"},
        "investment_facts": {},
    }
    scoring = {
        "INV-010": {"execution_rate": 0.5, "approved_amount": 3_000_000.0},
        "INV-011": {"execution_rate": 0.7, "approved_amount": 6_000_000.0},
    }
    result = render_scatter_plot([scope_no_facts_a, scope_no_facts_b], scoring)
    assert result is not None
    assert base64.b64decode(result)[:4] == b"\x89PNG"
