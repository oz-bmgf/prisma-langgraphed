"""Unit tests for src/graph/nodes/precheck.py."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from src.graph.nodes.precheck import precheck


def _make_state(**overrides) -> dict:
    base = {
        "ingested_dir": "/nonexistent",
        "focus_bows": None,
    }
    base.update(overrides)
    return base


def _populate_ingested_dir(d: Path) -> None:
    """Write minimal valid artifacts into tmp ingested dir."""
    (d / "embedding_index").mkdir(parents=True)
    (d / "pages").mkdir()
    (d / "doc_list.json").write_text(json.dumps([{"file_id": "f1"}]))
    (d / "investment_scoring.json").write_text(json.dumps({"INV-01": 0.9}))
    (d / "bow_investment_map.json").write_text(json.dumps({"BOW-A": ["INV-01"]}))
    (d / "embedding_index" / "chunks.json").write_text(json.dumps([{"chunk_id": "c1"}]))


# ---------------------------------------------------------------------------
# passes-clean
# ---------------------------------------------------------------------------


async def test_passes_clean(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    ingested_dir = tmp_path / "ingested"
    ingested_dir.mkdir()
    _populate_ingested_dir(ingested_dir)

    result = await precheck(_make_state(ingested_dir=str(ingested_dir)), {})

    assert result["precheck_passed"] is True
    assert isinstance(result["precheck_report"], str)


# ---------------------------------------------------------------------------
# fails-missing-artifacts
# ---------------------------------------------------------------------------


async def test_fails_missing_artifacts(tmp_path):
    ingested_dir = tmp_path / "empty_ingested"
    ingested_dir.mkdir()

    result = await precheck(_make_state(ingested_dir=str(ingested_dir)), {})

    assert result["precheck_passed"] is False
    assert isinstance(result["precheck_report"], str)


# ---------------------------------------------------------------------------
# report-is-string
# ---------------------------------------------------------------------------


async def test_report_is_string(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    ingested_dir = tmp_path / "ingested"
    ingested_dir.mkdir()
    _populate_ingested_dir(ingested_dir)

    result = await precheck(_make_state(ingested_dir=str(ingested_dir)), {})

    assert isinstance(result["precheck_report"], str)
    assert len(result["precheck_report"]) > 0
    assert "OVERALL" in result["precheck_report"]


# ---------------------------------------------------------------------------
# focus_bows unknown fails
# ---------------------------------------------------------------------------


async def test_fails_unknown_focus_bows(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    ingested_dir = tmp_path / "ingested"
    ingested_dir.mkdir()
    _populate_ingested_dir(ingested_dir)

    result = await precheck(
        _make_state(ingested_dir=str(ingested_dir), focus_bows=["NONEXISTENT-BOW"]),
        {},
    )

    assert result["precheck_passed"] is False
    assert "NONEXISTENT-BOW" in result["precheck_report"]


# ---------------------------------------------------------------------------
# returns only expected keys
# ---------------------------------------------------------------------------


async def test_returns_only_expected_keys(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    ingested_dir = tmp_path / "ingested"
    ingested_dir.mkdir()
    _populate_ingested_dir(ingested_dir)

    result = await precheck(_make_state(ingested_dir=str(ingested_dir)), {})

    assert set(result.keys()) == {"precheck_passed", "precheck_report"}
