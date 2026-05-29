"""Unit tests for src/graph/nodes/deliver.py."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from src.graph.nodes.deliver import deliver

_EXCERPT_FIELDS = [
    "excerpt_id", "scope_id", "scope_label", "inv_id", "link_name",
    "source_file", "page", "source_type", "type", "quote",
    "significance", "context_needed", "numerical_facts",
]


def _sample_excerpt(**overrides) -> dict:
    base = {
        "excerpt_id": "EX-inv1-link1-001",
        "scope_id": "scope-1",
        "scope_label": "Malaria Prevention",
        "inv_id": "inv1",
        "link_name": "Bed Net Coverage",
        "source_file": "doc/report.pdf",
        "page": 3,
        "source_type": "tier1_primary",
        "type": "evidence",
        "quote": "Net coverage reached 85% in study districts.",
        "significance": "supporting",
        "context_needed": "False",
        "numerical_facts": "85%",
        # new-schema fields also present (real excerpts carry both)
        "text": "Net coverage reached 85% in study districts.",
        "source": "doc/report.pdf",
        "credibility_tier": "tier1_primary",
    }
    base.update(overrides)
    return base


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


# ---------------------------------------------------------------------------
# excerpts CSV — 13-column old-schema output
# ---------------------------------------------------------------------------


async def test_excerpts_csv_written_with_13_columns(tmp_path):
    excerpts = [_sample_excerpt(), _sample_excerpt(excerpt_id="EX-inv1-link1-002", quote="A second finding.")]

    result = await deliver(
        _make_state(base_dir=str(tmp_path), all_excerpts=excerpts),
        {},
    )

    delivery_dir = _delivery_dir(tmp_path)
    csv_path = delivery_dir / "excerpts.csv"
    assert csv_path.exists(), "excerpts.csv was not written"

    rows = list(csv.DictReader(csv_path.read_text().splitlines()))
    assert len(rows) == 2, f"expected 2 data rows, got {len(rows)}"
    assert set(rows[0].keys()) == set(_EXCERPT_FIELDS), (
        f"column mismatch: {set(rows[0].keys()) ^ set(_EXCERPT_FIELDS)}"
    )
    assert rows[0]["excerpt_id"] == "EX-inv1-link1-001"
    assert rows[1]["quote"] == "A second finding."


async def test_excerpts_csv_path_returned(tmp_path):
    result = await deliver(
        _make_state(base_dir=str(tmp_path), all_excerpts=[_sample_excerpt()]),
        {},
    )

    delivery_dir = _delivery_dir(tmp_path)
    assert result["excerpts_csv_path"] == str(delivery_dir / "excerpts.csv")


async def test_excerpts_csv_new_schema_fallback(tmp_path):
    """Excerpt with only new-schema keys (no old-schema keys) still writes valid CSV."""
    new_schema_excerpt = {
        "text": "Key finding about malaria nets.",
        "source": "doc/nets.pdf",
        "credibility_tier": "tier1_primary",
        "inv_id": "inv2",
        "scope_id": "scope-2",
        "link_id": "link-nets",
        "page": 5,
        "significance": "supporting",
        "numerical_facts": [],
    }

    result = await deliver(
        _make_state(base_dir=str(tmp_path), all_excerpts=[new_schema_excerpt]),
        {},
    )

    delivery_dir = _delivery_dir(tmp_path)
    csv_path = delivery_dir / "excerpts.csv"
    assert csv_path.exists()
    rows = list(csv.DictReader(csv_path.read_text().splitlines()))
    assert len(rows) == 1
    assert rows[0]["quote"] == "Key finding about malaria nets."
    assert rows[0]["source_file"] == "doc/nets.pdf"


async def test_no_excerpts_no_csv(tmp_path):
    result = await deliver(
        _make_state(base_dir=str(tmp_path), all_excerpts=[]),
        {},
    )

    delivery_dir = _delivery_dir(tmp_path)
    assert not (delivery_dir / "excerpts.csv").exists()
    assert result["excerpts_csv_path"] is None
