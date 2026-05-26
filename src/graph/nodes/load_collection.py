"""load_collection node — reads {program}-ingested/ artifacts into WorkflowState."""
from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from langchain_core.runnables import RunnableConfig

from src.graph.state import WorkflowState

logger = logging.getLogger(__name__)


def _read_json(path: Path) -> object:
    with open(path) as fh:
        return json.load(fh)


def _augment_bow_map(bow_investment_map: dict, bow_rows_path: Path) -> dict:
    """Secondary BoW augmentation from investment_bow_rows.json.

    Matches OLD repo collection_loader.py secondary-augmentation pass:
    every (inv_id, bow_id) row in investment_bow_rows.json is added to
    bow_investment_map when the link is absent. New bow_ids are created.
    This covers collections built before the Step-3 fold-in patch was applied.
    """
    if not bow_rows_path.exists():
        return bow_investment_map

    try:
        rows = json.loads(bow_rows_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning(
            "investment_bow_rows.json unreadable (%s) — skipping secondary-BoW augmentation", exc
        )
        return bow_investment_map

    if not isinstance(rows, list):
        return bow_investment_map

    if not isinstance(bow_investment_map, dict):
        bow_investment_map = {}
    result = {k: dict(v) if isinstance(v, dict) else list(v) for k, v in bow_investment_map.items()}
    added_links = 0
    new_bows = 0
    for row in rows:
        if not isinstance(row, dict):
            continue
        bid = (row.get("bow_id") or "").strip()
        iid = (row.get("inv_id") or "").strip()
        if not bid or not iid:
            continue
        if bid not in result:
            result[bid] = {
                "bow_label": (row.get("bow_name") or "").strip() or bid,
                "inv_ids": [],
            }
            new_bows += 1
        entry = result[bid]
        inv_ids = entry.get("inv_ids", []) if isinstance(entry, dict) else list(entry)
        if iid not in inv_ids:
            if isinstance(entry, dict):
                result[bid].setdefault("inv_ids", []).append(iid)
            else:
                result[bid] = list(result[bid]) + [iid]
            added_links += 1

    if added_links or new_bows:
        logger.info(
            "load_collection: augmented BoW map from investment_bow_rows.json: "
            "+%d (inv, bow) links, +%d new BoWs",
            added_links, new_bows,
        )
    return result


async def load_collection(state: WorkflowState, config: RunnableConfig) -> dict:
    if state.get("doc_list") is not None:
        return {}  # already loaded; checkpoint restored fields

    raw = state.get("ingested_dir") or str(
        Path(state["base_dir"]) / f"{state['program']}-ingested"
    )
    ingested_dir = Path(raw)
    # asyncio-APPROVED-1: to_thread wraps blocking JSON file read
    doc_list = await asyncio.to_thread(_read_json, ingested_dir / "doc_list.json")
    # asyncio-APPROVED-1: to_thread wraps blocking JSON file read
    investment_scoring = await asyncio.to_thread(_read_json, ingested_dir / "investment_scoring.json")
    # asyncio-APPROVED-1: to_thread wraps blocking JSON file read
    bow_investment_map_raw = await asyncio.to_thread(_read_json, ingested_dir / "bow_investment_map.json")
    # asyncio-APPROVED-1: to_thread wraps blocking JSON file read
    investment_intelligence = await asyncio.to_thread(_read_json, ingested_dir / "investment_intelligence.json")

    # Load investment_bow_rows.json for secondary augmentation and state passthrough
    bow_rows_path = ingested_dir / "investment_bow_rows.json"
    investment_bow_rows: list = []
    if bow_rows_path.exists():
        try:
            # asyncio-APPROVED-1: to_thread wraps blocking JSON file read
            raw_rows = await asyncio.to_thread(_read_json, bow_rows_path)
            investment_bow_rows = raw_rows if isinstance(raw_rows, list) else []
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("load_collection: investment_bow_rows.json unreadable: %s", exc)

    # Secondary BoW augmentation — matches OLD repo collection_loader.py behaviour
    # asyncio-APPROVED-1: to_thread wraps blocking dict mutation
    bow_investment_map = await asyncio.to_thread(
        _augment_bow_map,
        bow_investment_map_raw,
        bow_rows_path,
    )

    return {
        "doc_list": doc_list,
        "investment_scoring": investment_scoring,
        "bow_investment_map": bow_investment_map,
        "investment_bow_rows": investment_bow_rows,
        "investment_intelligence": investment_intelligence,
        "chunks_json_path": str(ingested_dir / "embedding_index" / "chunks.json"),
        "pages_dir": str(ingested_dir / "pages"),
    }
