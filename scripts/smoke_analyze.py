"""Smoke test: run only the orientation node with a real LLM call.

Validates the full LLM plumbing end-to-end without running the whole pipeline.

Usage:
    ANTHROPIC_API_KEY=... python scripts/smoke_analyze.py
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

async def main() -> None:
    # src.config loads .env at import time
    from src.config import COLLECTIONS_BASE_PATH, DEFAULT_RESEARCH_MODEL, DEFAULT_SYNTHESIS_MODEL
    from src.graph.subgraphs.analyze import orientation
    from src.core.telemetry import setup_telemetry

    setup_telemetry("nqpr-smoke-analyze")

    ingested = COLLECTIONS_BASE_PATH / "MOCK-ingested"

    doc_list = json.loads((ingested / "doc_list.json").read_text())
    investment_scoring = json.loads((ingested / "investment_scoring.json").read_text())
    bow_investment_map = json.loads((ingested / "bow_investment_map.json").read_text())
    investment_intelligence = json.loads((ingested / "investment_intelligence.json").read_text())

    state = {
        "program": "MOCK",
        "collection_name": "mock",
        "base_dir": str(COLLECTIONS_BASE_PATH),
        "ingested_dir": str(ingested),
        "doc_list": doc_list,
        "investment_scoring": investment_scoring,
        "bow_investment_map": bow_investment_map,
        "investment_intelligence": investment_intelligence,
        "chunks_json_path": str(ingested / "embedding_index" / "chunks.json"),
        "pages_dir": str(ingested / "pages"),
        "focus": None,
        "focus_bows": None,
        "aux_collections": None,
        "threads_dir": "/tmp/nqpr_smoke_analyze",
        "research_model": DEFAULT_RESEARCH_MODEL,
        "synthesis_model": DEFAULT_SYNTHESIS_MODEL,
        "orientation_summary": None,
        "scopes": None,
        "scope_timelines": None,
        "clusters": None,
        "scope_outputs": None,
        "analyst_report": None,
        "final_report_md": None,
        "excerpts_csv_path": None,
        "numerical_provenance": None,
        "verification_sources": None,
        "allocation_verification_path": None,
        "numerical_verification_path": None,
        "run_meta": None,
        "evidence_packs": [],
        "link_assessments": [],
        "science_results": [],
        "scope_decisions": [],
        "timeline_narrative_results": [],
        "errors": [],
    }

    print("Calling orientation node with real LLM...")
    result = await orientation(state)

    assert "orientation_summary" in result, "orientation_summary missing from result"
    assert isinstance(result["orientation_summary"], str), "orientation_summary must be a string"
    assert len(result["orientation_summary"]) > 50, "orientation_summary too short"

    print("orientation_summary (first 300 chars):")
    print(result["orientation_summary"][:300])
    print()
    print("Smoke test PASSED — orientation node works end-to-end")


if __name__ == "__main__":
    # asyncio-APPROVED-4: asyncio.run in CLI entry point — top-level event loop
    asyncio.run(main())
