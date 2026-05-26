"""End-to-end smoke test — runs load_collection + precheck only.

Usage:
    python scripts/smoke_test.py

Stops at the interrupt_before="analyze" gate. Validates that:
  - load_collection populated doc_list (non-empty list)
  - precheck_passed is True or False (not None — it ran)
  - precheck_report is a non-empty string
  - No exceptions raised
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

# Add the project root to sys.path so imports work when run from any cwd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# src.config loads .env at import time
from src.config import COLLECTIONS_BASE_PATH, DEFAULT_RESEARCH_MODEL, DEFAULT_SYNTHESIS_MODEL
from src.core.checkpointer import build_checkpointer
from src.graph.workflow import compile_graph, create_initial_state, make_thread_id

# ---------------------------------------------------------------------------
# Configuration — VDEV collection
# ---------------------------------------------------------------------------

BASE_DIR = str(COLLECTIONS_BASE_PATH)
INGESTED_DIR = str(COLLECTIONS_BASE_PATH / "VDEV-ingested")
PROGRAM = "VDEV"
COLLECTION_NAME = "vdev"
RUN_NAME = "smoke-test-01"


def _ensure_investment_bow_rows(ingested_dir: str) -> None:
    """Derive investment_bow_rows.json from bow_investment_map.json if missing."""
    path = Path(ingested_dir) / "investment_bow_rows.json"
    if path.exists():
        return
    bow_map_path = Path(ingested_dir) / "bow_investment_map.json"
    if not bow_map_path.exists():
        return
    bow_map = json.loads(bow_map_path.read_text(encoding="utf-8"))
    rows = []
    for bow_id, bow_data in bow_map.items():
        if isinstance(bow_data, dict):
            inv_ids = bow_data.get("inv_ids", [])
        elif isinstance(bow_data, list):
            inv_ids = bow_data
        else:
            inv_ids = []
        for inv_id in inv_ids:
            rows.append({"bow_id": bow_id, "inv_id": inv_id})
    path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"  [setup] Derived investment_bow_rows.json → {len(rows)} rows")


async def run_smoke_test() -> None:
    print("=" * 60)
    print("NQPR smoke test")
    print(f"  program      : {PROGRAM}")
    print(f"  ingested_dir : {INGESTED_DIR}")
    print("=" * 60)

    # Derive missing investment_bow_rows.json if needed
    _ensure_investment_bow_rows(INGESTED_DIR)

    initial_state = create_initial_state(
        program=PROGRAM,
        run_name=RUN_NAME,
        collection_name=COLLECTION_NAME,
        base_dir=BASE_DIR,
        ingested_dir=INGESTED_DIR,
        research_model=DEFAULT_RESEARCH_MODEL,
        synthesis_model=DEFAULT_SYNTHESIS_MODEL,
    )
    thread_id = make_thread_id(PROGRAM, RUN_NAME)
    config = {"configurable": {"thread_id": thread_id}}

    final_event: dict = {}

    print("\nStreaming graph (will stop after precheck)...")
    async with build_checkpointer() as checkpointer:
        graph = compile_graph(checkpointer)
        async for event in graph.astream(
            initial_state,
            config=config,
            stream_mode="values",
        ):
            stage = event.get("current_stage")
            precheck_passed = event.get("precheck_passed")
            doc_list = event.get("doc_list")
            n_docs = len(doc_list) if isinstance(doc_list, list) else None
            print(f"  event: current_stage={stage!r}  doc_list={n_docs} docs  precheck_passed={precheck_passed!r}")
            final_event = event

            # Stop as soon as precheck has run (precheck_passed is no longer None)
            if precheck_passed is not None:
                break

    # ---------------------------------------------------------------------------
    # Assertions
    # ---------------------------------------------------------------------------
    print("\nAsserting results...")

    doc_list = final_event.get("doc_list")
    assert isinstance(doc_list, list) and len(doc_list) > 0, (
        f"doc_list should be a non-empty list, got: {type(doc_list)} len={len(doc_list) if isinstance(doc_list, list) else 'N/A'}"
    )

    precheck_passed = final_event.get("precheck_passed")
    assert precheck_passed is not None, "precheck_passed should not be None — precheck node did not run"
    assert isinstance(precheck_passed, bool), f"precheck_passed should be bool, got {type(precheck_passed)}"

    precheck_report = final_event.get("precheck_report")
    assert isinstance(precheck_report, str) and len(precheck_report) > 0, (
        f"precheck_report should be a non-empty string, got: {precheck_report!r}"
    )

    # ---------------------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Smoke test PASSED")
    print(f"  doc_list        : {len(doc_list)} documents loaded")
    print(f"  precheck_passed : {precheck_passed}")
    print(f"  precheck_report :\n{precheck_report}")
    print("=" * 60)


if __name__ == "__main__":
    # asyncio-APPROVED-4: asyncio.run in CLI entry point — top-level event loop
    asyncio.run(run_smoke_test())
