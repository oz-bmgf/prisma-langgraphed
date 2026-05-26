"""Run the pipeline on MOCK data through the precheck node only.

Usage:
    python scripts/run_mock_pipeline.py

Stops at interrupt_before="analyze" (no LLM calls). Prints:
  - Documents loaded and count
  - Investments and BOWs
  - precheck_passed and summary of precheck_report
  - thread_id for later resumption

Prerequisites:
    python scripts/create_mock_data.py   (run once to generate MOCK-ingested/)
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# safety fallback — src.config loads .env at import time; this handles direct
# script invocation before any src import has triggered that load.
from dotenv import load_dotenv

_env_path = Path(__file__).resolve().parents[1] / ".env"
if _env_path.exists():
    load_dotenv(_env_path, override=False)

from src.config import (
    COLLECTIONS_BASE_PATH,
    DEFAULT_RESEARCH_MODEL,
    DEFAULT_SYNTHESIS_MODEL,
)
from src.core.checkpointer import build_checkpointer
from src.graph.workflow import compile_graph, create_initial_state, make_thread_id

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = str(COLLECTIONS_BASE_PATH)
INGESTED_DIR = str(COLLECTIONS_BASE_PATH / "MOCK-ingested")
PROGRAM = "MOCK"
COLLECTION_NAME = "mock"
RUN_NAME = "mock-run-01"


async def run() -> None:
    ingested = Path(INGESTED_DIR)
    if not ingested.exists():
        print("ERROR: MOCK-ingested/ not found.")
        print("Run first: python scripts/create_mock_data.py")
        sys.exit(1)

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

    print("=" * 60)
    print("NQPR pipeline — MOCK program (precheck only)")
    print(f"  ingested_dir : {INGESTED_DIR}")
    print(f"  thread_id    : {thread_id}")
    print("=" * 60)

    final_event: dict = {}

    print("\nStreaming (stops after precheck)...")
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
            print(f"  stage={stage!r}  docs={n_docs}  precheck_passed={precheck_passed!r}")
            final_event = event

            if precheck_passed is not None:
                break

    # ---------------------------------------------------------------------------
    # Results
    # ---------------------------------------------------------------------------
    doc_list = final_event.get("doc_list") or []
    investment_scoring = final_event.get("investment_scoring") or {}
    bow_investment_map = final_event.get("bow_investment_map") or {}
    precheck_passed = final_event.get("precheck_passed")
    precheck_report = final_event.get("precheck_report") or ""

    print()
    print("=" * 60)
    print("Results")
    print("=" * 60)
    print(f"  Documents loaded  : {len(doc_list)}")
    for doc in doc_list:
        print(f"    {doc.get('file_id')}  {doc.get('doc_type')}  inv={doc.get('inv_id')}")

    print(f"  Investments       : {list(investment_scoring.keys())}")
    print(f"  BOWs              : {list(bow_investment_map.keys())}")
    print(f"  precheck_passed   : {precheck_passed}")
    print()
    print("  precheck_report (first 800 chars):")
    print("  " + precheck_report[:800].replace("\n", "\n  "))

    print()
    print(f"  thread_id: {thread_id}")
    print()
    print("To resume through analyze (WARNING: real LLM calls, costs money):")
    print(
        "  # async for event in graph.astream(None, config=config, stream_mode='values'):\n"
        "  #     ...  # process events after analyze node completes"
    )
    print("=" * 60)

    # Assertions
    assert isinstance(doc_list, list) and len(doc_list) > 0, "doc_list is empty"
    assert precheck_passed is not None, "precheck did not run"
    assert isinstance(precheck_report, str) and len(precheck_report) > 0, "precheck_report empty"

    print("\nAll assertions passed.")


if __name__ == "__main__":
    # asyncio-APPROVED-4: asyncio.run in CLI entry point — top-level event loop
    asyncio.run(run())
