"""Unit tests for src/graph/nodes/load_collection.py."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from src.graph.nodes.load_collection import load_collection


def _make_state(**overrides) -> dict:
    base = {
        "ingested_dir": "/fake/ingested",
        "doc_list": None,
    }
    base.update(overrides)
    return base


def _fake_read_json(path: Path) -> object:
    name = path.name
    if name == "doc_list.json":
        return [{"file_id": "f1"}]
    if name == "investment_scoring.json":
        return {"INV-01": 0.9}
    if name == "bow_investment_map.json":
        return {"BOW-A": ["INV-01"]}
    if name == "investment_bow_rows.json":
        return [{"inv_id": "INV-01", "bow_id": "BOW-A"}]
    if name == "investment_intelligence.json":
        return {"INV-01": {"summary": "..."}}
    raise FileNotFoundError(path)


# ---------------------------------------------------------------------------
# skip-if-already-loaded
# ---------------------------------------------------------------------------


async def test_skip_if_already_loaded():
    state = _make_state(doc_list=[{"file_id": "existing"}])
    result = await load_collection(state, {})
    assert result == {}


# ---------------------------------------------------------------------------
# loads all fields
# ---------------------------------------------------------------------------


async def test_loads_all_fields(tmp_path):
    ingested_dir = tmp_path / "ingested"
    ingested_dir.mkdir()
    (ingested_dir / "embedding_index").mkdir()
    (ingested_dir / "pages").mkdir()

    (ingested_dir / "doc_list.json").write_text(json.dumps([{"file_id": "f1"}]))
    (ingested_dir / "investment_scoring.json").write_text(json.dumps({"INV-01": 0.9}))
    (ingested_dir / "bow_investment_map.json").write_text(json.dumps({"BOW-A": ["INV-01"]}))
    (ingested_dir / "investment_bow_rows.json").write_text(json.dumps([{"inv_id": "INV-01"}]))
    (ingested_dir / "investment_intelligence.json").write_text(json.dumps({"INV-01": {}}))

    state = _make_state(ingested_dir=str(ingested_dir))
    result = await load_collection(state, {})

    assert result["doc_list"] == [{"file_id": "f1"}]
    assert result["investment_scoring"] == {"INV-01": 0.9}
    assert result["bow_investment_map"] == {"BOW-A": ["INV-01"]}
    assert result["investment_bow_rows"] == [{"inv_id": "INV-01"}]
    assert result["investment_intelligence"] == {"INV-01": {}}


# ---------------------------------------------------------------------------
# sets path fields
# ---------------------------------------------------------------------------


async def test_sets_path_fields(tmp_path):
    ingested_dir = tmp_path / "ingested"
    ingested_dir.mkdir()
    (ingested_dir / "embedding_index").mkdir()
    (ingested_dir / "pages").mkdir()

    for fname in [
        "doc_list.json",
        "investment_scoring.json",
        "bow_investment_map.json",
        "investment_bow_rows.json",
        "investment_intelligence.json",
    ]:
        (ingested_dir / fname).write_text("[]")

    state = _make_state(ingested_dir=str(ingested_dir))
    result = await load_collection(state, {})

    assert result["chunks_json_path"] == str(ingested_dir / "embedding_index" / "chunks.json")
    assert result["pages_dir"] == str(ingested_dir / "pages")
