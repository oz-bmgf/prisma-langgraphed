"""Unit tests for src/graph/nodes/deliver.py."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.graph.nodes.deliver import deliver


def _make_state(**overrides) -> dict:
    base = {
        "program": "Malaria",
        "run_name": "crimson-falcon",
        "base_dir": "/tmp",
        "threads_dir": None,  # None → falls back to {base_dir}/{program}-experiments/run-{run_name}
        "final_report_wresearch_md_path": None,
        "final_report_pdf_path": None,
        "run_meta": None,
    }
    base.update(overrides)
    return base


def _delivery_dir(tmp_path: Path) -> Path:
    """Canonical delivery path when threads_dir is not provided."""
    return tmp_path / "Malaria-experiments" / "run-crimson-falcon"


# ---------------------------------------------------------------------------
# creates delivery dir and run_meta.json
# ---------------------------------------------------------------------------


async def test_creates_delivery_dir_and_meta(tmp_path):
    result = await deliver(
        _make_state(base_dir=str(tmp_path)),
        {},
    )

    delivery_dir = _delivery_dir(tmp_path)
    assert delivery_dir.is_dir()
    meta_path = delivery_dir / "run_meta.json"
    assert meta_path.exists()
    meta = json.loads(meta_path.read_text())
    assert meta["program"] == "Malaria"
    assert meta["run_name"] == "crimson-falcon"


# ---------------------------------------------------------------------------
# copies report markdown
# ---------------------------------------------------------------------------


async def test_copies_markdown(tmp_path):
    report = tmp_path / "report.md"
    report.write_text("# Final Report")

    await deliver(
        _make_state(
            base_dir=str(tmp_path),
            final_report_wresearch_md_path=str(report),
        ),
        {},
    )

    delivery_dir = _delivery_dir(tmp_path)
    assert (delivery_dir / "final_report_wresearch.md").exists()
    content = (delivery_dir / "final_report_wresearch.md").read_text()
    assert content == "# Final Report"


# ---------------------------------------------------------------------------
# copies PDF if it exists
# ---------------------------------------------------------------------------


async def test_copies_pdf(tmp_path):
    pdf_src = tmp_path / "final_report.pdf"
    pdf_src.write_bytes(b"%PDF-1.4 fake")

    await deliver(
        _make_state(
            base_dir=str(tmp_path),
            final_report_pdf_path=str(pdf_src),
        ),
        {},
    )

    delivery_dir = _delivery_dir(tmp_path)
    assert (delivery_dir / "final_report.pdf").exists()


# ---------------------------------------------------------------------------
# missing files are silently skipped
# ---------------------------------------------------------------------------


async def test_missing_files_skipped(tmp_path):
    result = await deliver(
        _make_state(
            base_dir=str(tmp_path),
            final_report_wresearch_md_path="/nonexistent/report.md",
            final_report_pdf_path="/nonexistent/report.pdf",
        ),
        {},
    )
    # no exception; run_meta.json created with no delivered files
    delivery_dir = _delivery_dir(tmp_path)
    meta = json.loads((delivery_dir / "run_meta.json").read_text())
    assert meta["delivered_files"] == []


# ---------------------------------------------------------------------------
# returns empty dict (terminal node)
# ---------------------------------------------------------------------------


async def test_returns_empty_dict(tmp_path):
    result = await deliver(_make_state(base_dir=str(tmp_path)), {})
    assert result == {
        "excerpts_csv_path": None,
        "numerical_verification_path": None,
        "allocation_verification_path": None,
    }
