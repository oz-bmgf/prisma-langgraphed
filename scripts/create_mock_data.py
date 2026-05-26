"""Create a minimal but valid MOCK-ingested/ directory for local testing.

Usage:
    python scripts/create_mock_data.py

Idempotent — safe to run multiple times. Creates:
    ~/qpr-collections/MOCK-ingested/
        doc_list.json
        investment_scoring.json
        bow_investment_map.json
        investment_bow_rows.json
        investment_intelligence.json
        embedding_index/
            chunks.json
            vectors.npy
            chunks.sqlite
            config.json
        pages/
            file-001/p1.txt
            file-002/p1.txt
            file-003/p1.txt
"""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.config import COLLECTIONS_BASE_PATH, DEFAULT_EMBEDDING_MODEL

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_DIR = COLLECTIONS_BASE_PATH
INGESTED_DIR = BASE_DIR / "MOCK-ingested"
EMBEDDING_DIR = INGESTED_DIR / "embedding_index"
PAGES_DIR = INGESTED_DIR / "pages"
COLLECTION = "mock"
DIM = 1536

# ---------------------------------------------------------------------------
# Catalog data
# ---------------------------------------------------------------------------

DOC_LIST = [
    {
        "file_id": "file-001",
        "filename": "proposal_inv001_2023.pdf",
        "doc_type": "proposal",
        "inv_id": "INV-001",
        "bow_id": "BOW-01",
        "total_pages": 12,
        "doc_date": "2023-01-15",
    },
    {
        "file_id": "file-002",
        "filename": "progress_report_inv001_2024.pdf",
        "doc_type": "progress_report",
        "inv_id": "INV-001",
        "bow_id": "BOW-01",
        "total_pages": 8,
        "doc_date": "2024-06-01",
    },
    {
        "file_id": "file-003",
        "filename": "annual_review_inv002_2024.pdf",
        "doc_type": "progress_report",
        "inv_id": "INV-002",
        "bow_id": "BOW-01",
        "total_pages": 15,
        "doc_date": "2024-09-30",
    },
]

INVESTMENT_SCORING = {
    "INV-001": {
        "inv_id": "INV-001",
        "title": "Mock Malaria Net Distribution Program",
        "score": 3.8,
        "allocation_usd": 2500000,
        "bow_id": "BOW-01",
        "maturity_stage": "implementation",
        "grantee": "Mock NGO Alpha",
        "geography": "Sub-Saharan Africa",
        "start_year": 2022,
        "end_year": 2026,
        "start": "2022-01",
        "end": "2026-12",
        "status": "active",
        "approved_amount": 2500000,
        "paid_amount": 1200000,
    },
    "INV-002": {
        "inv_id": "INV-002",
        "title": "Mock Vector Control Pilot",
        "score": 3.2,
        "allocation_usd": 800000,
        "bow_id": "BOW-01",
        "maturity_stage": "pilot",
        "grantee": "Mock Research Institute",
        "geography": "East Africa",
        "start_year": 2023,
        "end_year": 2025,
        "start": "2023-06",
        "end": "2025-12",
        "status": "active",
        "approved_amount": 800000,
        "paid_amount": 350000,
    },
}

BOW_INVESTMENT_MAP = {
    "BOW-01": {
        "bow_id": "BOW-01",
        "bow_label": "Malaria Prevention & Control",
        "inv_ids": ["INV-001", "INV-002"],
    }
}

INVESTMENT_BOW_ROWS = [
    {"bow_id": "BOW-01", "inv_id": "INV-001"},
    {"bow_id": "BOW-01", "inv_id": "INV-002"},
]

INVESTMENT_INTELLIGENCE = {
    "INV-001": {
        "inv_id": "INV-001",
        "theory_of_change": (
            "Distributing insecticide-treated nets at scale reduces malaria incidence "
            "by limiting mosquito-human contact in high-burden communities."
        ),
        "key_decisions": [
            "Targeting criteria shifted to all-age distribution in 2023 based on RCT evidence.",
            "Partnership with local health ministry signed Q2 2023.",
        ],
        "timeline_summary": "Phase 1 (2022-2023): needs assessment and partner onboarding. "
                            "Phase 2 (2024-2026): distribution and monitoring at scale.",
    },
    "INV-002": {
        "inv_id": "INV-002",
        "theory_of_change": (
            "Indoor residual spraying as complement to net distribution reduces vector "
            "breeding in settings where net coverage alone is insufficient."
        ),
        "key_decisions": [
            "Pilot restricted to 3 districts pending safety review.",
        ],
        "timeline_summary": "Pilot (2023-2025): efficacy comparison across 3 districts.",
    },
}

# ---------------------------------------------------------------------------
# Chunks — 2 per document, 6 total
# ---------------------------------------------------------------------------

CHUNKS = [
    {
        "chunk_id": "file-001-c0",
        "file_id": "file-001",
        "filename": "proposal_inv001_2023.pdf",
        "collection": COLLECTION,
        "doc_type": "proposal",
        "inv_id": "INV-001",
        "bow_id": "BOW-01",
        "page_start": 1,
        "page_end": 3,
        "text": (
            "This proposal requests USD 2.5M to distribute insecticide-treated bed nets "
            "to 500,000 households across three high-burden districts in Sub-Saharan Africa. "
            "Evidence from randomised controlled trials demonstrates 30% reduction in malaria incidence."
        ),
        "context": "Executive summary",
        "doc_date": "2023-01-15",
        "intelligence_role": "",
        "intelligence_action": "",
        "intelligence_is_best": False,
        "intelligence_version_group": "",
        "intelligence_pass_status": "ok",
        "content_tags": [],
        "supports_bow_ids": [],
        "topic_tags": ["malaria", "nets", "distribution"],
        "carve_out_metadata": None,
        "score": 0.0,
    },
    {
        "chunk_id": "file-001-c1",
        "file_id": "file-001",
        "filename": "proposal_inv001_2023.pdf",
        "collection": COLLECTION,
        "doc_type": "proposal",
        "inv_id": "INV-001",
        "bow_id": "BOW-01",
        "page_start": 4,
        "page_end": 6,
        "text": (
            "Budget breakdown: procurement USD 1.8M, logistics USD 0.4M, monitoring USD 0.3M. "
            "Implementation partner is Mock NGO Alpha, operational since 2010 with presence in 8 countries."
        ),
        "context": "Budget and implementation",
        "doc_date": "2023-01-15",
        "intelligence_role": "",
        "intelligence_action": "",
        "intelligence_is_best": False,
        "intelligence_version_group": "",
        "intelligence_pass_status": "ok",
        "content_tags": [],
        "supports_bow_ids": [],
        "topic_tags": ["budget", "procurement"],
        "carve_out_metadata": None,
        "score": 0.0,
    },
    {
        "chunk_id": "file-002-c0",
        "file_id": "file-002",
        "filename": "progress_report_inv001_2024.pdf",
        "collection": COLLECTION,
        "doc_type": "progress_report",
        "inv_id": "INV-001",
        "bow_id": "BOW-01",
        "page_start": 1,
        "page_end": 3,
        "text": (
            "By June 2024, 320,000 of 500,000 targeted households received nets. "
            "Preliminary monitoring data shows 22% reduction in confirmed malaria cases year-on-year "
            "in intervention districts versus 4% in control districts."
        ),
        "context": "Progress to date",
        "doc_date": "2024-06-01",
        "intelligence_role": "",
        "intelligence_action": "",
        "intelligence_is_best": False,
        "intelligence_version_group": "",
        "intelligence_pass_status": "ok",
        "content_tags": [],
        "supports_bow_ids": [],
        "topic_tags": ["malaria", "impact", "monitoring"],
        "carve_out_metadata": None,
        "score": 0.0,
    },
    {
        "chunk_id": "file-002-c1",
        "file_id": "file-002",
        "filename": "progress_report_inv001_2024.pdf",
        "collection": COLLECTION,
        "doc_type": "progress_report",
        "inv_id": "INV-001",
        "bow_id": "BOW-01",
        "page_start": 4,
        "page_end": 5,
        "text": (
            "Challenges: supply chain delays in Q1 2024 due to port congestion reduced net delivery by 15%. "
            "Mitigation: diversified supplier list and pre-positioned buffer stock introduced in Q2."
        ),
        "context": "Challenges and mitigation",
        "doc_date": "2024-06-01",
        "intelligence_role": "",
        "intelligence_action": "",
        "intelligence_is_best": False,
        "intelligence_version_group": "",
        "intelligence_pass_status": "ok",
        "content_tags": [],
        "supports_bow_ids": [],
        "topic_tags": ["supply_chain", "risk"],
        "carve_out_metadata": None,
        "score": 0.0,
    },
    {
        "chunk_id": "file-003-c0",
        "file_id": "file-003",
        "filename": "annual_review_inv002_2024.pdf",
        "collection": COLLECTION,
        "doc_type": "progress_report",
        "inv_id": "INV-002",
        "bow_id": "BOW-01",
        "page_start": 1,
        "page_end": 4,
        "text": (
            "The indoor residual spraying pilot covered 45,000 structures across 3 districts. "
            "Entomological monitoring shows 60% reduction in Anopheles mosquito density in treated areas. "
            "No adverse health events reported in the first year of operations."
        ),
        "context": "Annual review findings",
        "doc_date": "2024-09-30",
        "intelligence_role": "",
        "intelligence_action": "",
        "intelligence_is_best": False,
        "intelligence_version_group": "",
        "intelligence_pass_status": "ok",
        "content_tags": [],
        "supports_bow_ids": [],
        "topic_tags": ["IRS", "vector_control", "entomology"],
        "carve_out_metadata": None,
        "score": 0.0,
    },
    {
        "chunk_id": "file-003-c1",
        "file_id": "file-003",
        "filename": "annual_review_inv002_2024.pdf",
        "collection": COLLECTION,
        "doc_type": "progress_report",
        "inv_id": "INV-002",
        "bow_id": "BOW-01",
        "page_start": 5,
        "page_end": 8,
        "text": (
            "Cost-effectiveness analysis: USD 18 per DALY averted at current coverage levels, "
            "compared to USD 25 benchmark for similar programmes. Recommend scale-up to 6 additional "
            "districts subject to safety review completion by December 2024."
        ),
        "context": "Cost-effectiveness and recommendation",
        "doc_date": "2024-09-30",
        "intelligence_role": "",
        "intelligence_action": "",
        "intelligence_is_best": False,
        "intelligence_version_group": "",
        "intelligence_pass_status": "ok",
        "content_tags": [],
        "supports_bow_ids": [],
        "topic_tags": ["cost_effectiveness", "scale_up"],
        "carve_out_metadata": None,
        "score": 0.0,
    },
]

PAGE_CONTENT = {
    "file-001": (
        "Proposal: Mock Malaria Net Distribution Program. "
        "This programme targets 500,000 households in high-burden districts. "
        "Bed nets have proven efficacy in reducing malaria transmission by limiting mosquito contact at night."
    ),
    "file-002": (
        "Progress Report — June 2024. "
        "Distribution is 64% complete with 320,000 households reached. "
        "Early monitoring data indicates a 22% reduction in confirmed malaria cases in intervention areas."
    ),
    "file-003": (
        "Annual Review — Vector Control Pilot, September 2024. "
        "Indoor residual spraying achieved 60% reduction in mosquito density across 45,000 structures. "
        "The programme is cost-effective at USD 18 per DALY averted and is recommended for scale-up."
    ),
}

# SQLite schema — must match local.py _SCHEMA_SQL
_CHUNK_COLUMNS = [
    "idx", "chunk_id", "file_id", "filename", "collection",
    "doc_type", "raw_doc_type", "inv_id", "bow_id",
    "section_id", "section_label", "page_start", "page_end",
    "text", "context", "doc_date",
    "intelligence_role", "intelligence_action", "intelligence_is_best",
    "intelligence_version_group", "intelligence_pass_status",
    "content_tags", "supports_bow_ids", "topic_tags", "carve_out_metadata",
]

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS chunks (
    idx                          INTEGER PRIMARY KEY,
    chunk_id                     TEXT UNIQUE NOT NULL,
    file_id                      TEXT NOT NULL,
    filename                     TEXT NOT NULL,
    collection                   TEXT NOT NULL,
    doc_type                     TEXT,
    raw_doc_type                 TEXT,
    inv_id                       TEXT,
    bow_id                       TEXT,
    section_id                   TEXT,
    section_label                TEXT,
    page_start                   INTEGER,
    page_end                     INTEGER,
    text                         TEXT,
    context                      TEXT,
    doc_date                     TEXT,
    intelligence_role            TEXT,
    intelligence_action          TEXT,
    intelligence_is_best         INTEGER,
    intelligence_version_group   TEXT,
    intelligence_pass_status     TEXT,
    content_tags                 TEXT,
    supports_bow_ids             TEXT,
    topic_tags                   TEXT,
    carve_out_metadata           TEXT
);
CREATE INDEX IF NOT EXISTS idx_chunks_inv        ON chunks(inv_id);
CREATE INDEX IF NOT EXISTS idx_chunks_bow        ON chunks(bow_id);
CREATE INDEX IF NOT EXISTS idx_chunks_doc_type   ON chunks(doc_type);
CREATE INDEX IF NOT EXISTS idx_chunks_collection ON chunks(collection);
CREATE INDEX IF NOT EXISTS idx_chunks_compound   ON chunks(collection, inv_id, doc_type);
CREATE INDEX IF NOT EXISTS idx_chunks_file_id    ON chunks(file_id);
"""


def _write_json(path: Path, obj: object) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def _build_sqlite(chunks: list[dict], sqlite_path: Path) -> None:
    if sqlite_path.exists():
        sqlite_path.unlink()
    conn = sqlite3.connect(str(sqlite_path))
    conn.executescript(_SCHEMA_SQL)
    for idx, chunk in enumerate(chunks):
        row = (
            idx,
            chunk["chunk_id"],
            chunk["file_id"],
            chunk["filename"],
            chunk["collection"],
            chunk.get("doc_type", ""),
            chunk.get("doc_type", ""),
            chunk.get("inv_id", ""),
            chunk.get("bow_id", ""),
            chunk.get("section_id", ""),
            chunk.get("section_label", ""),
            chunk.get("page_start", 0),
            chunk.get("page_end", 0),
            chunk.get("text", ""),
            chunk.get("context", ""),
            chunk.get("doc_date", ""),
            chunk.get("intelligence_role", ""),
            chunk.get("intelligence_action", ""),
            int(chunk.get("intelligence_is_best", False)),
            chunk.get("intelligence_version_group", ""),
            chunk.get("intelligence_pass_status", "ok"),
            json.dumps(chunk.get("content_tags") or []),
            json.dumps(chunk.get("supports_bow_ids") or []),
            json.dumps(chunk.get("topic_tags") or []),
            json.dumps(chunk.get("carve_out_metadata")) if chunk.get("carve_out_metadata") else "",
        )
        cols = ", ".join(_CHUNK_COLUMNS)
        placeholders = ", ".join("?" * len(_CHUNK_COLUMNS))
        conn.execute(f"INSERT OR IGNORE INTO chunks ({cols}) VALUES ({placeholders})", row)
    conn.commit()
    conn.close()


def create_mock_data() -> None:
    print("=" * 60)
    print("create_mock_data — MOCK program")
    print(f"  target: {INGESTED_DIR}")
    print("=" * 60)

    # Directory scaffolding
    INGESTED_DIR.mkdir(parents=True, exist_ok=True)
    EMBEDDING_DIR.mkdir(parents=True, exist_ok=True)
    PAGES_DIR.mkdir(parents=True, exist_ok=True)

    # Catalog files
    _write_json(INGESTED_DIR / "doc_list.json", DOC_LIST)
    print(f"  [ok] doc_list.json            ({len(DOC_LIST)} documents)")

    _write_json(INGESTED_DIR / "investment_scoring.json", INVESTMENT_SCORING)
    print(f"  [ok] investment_scoring.json  ({len(INVESTMENT_SCORING)} investments)")

    _write_json(INGESTED_DIR / "bow_investment_map.json", BOW_INVESTMENT_MAP)
    print(f"  [ok] bow_investment_map.json  ({len(BOW_INVESTMENT_MAP)} BOWs)")

    _write_json(INGESTED_DIR / "investment_bow_rows.json", INVESTMENT_BOW_ROWS)
    print(f"  [ok] investment_bow_rows.json ({len(INVESTMENT_BOW_ROWS)} rows)")

    _write_json(INGESTED_DIR / "investment_intelligence.json", INVESTMENT_INTELLIGENCE)
    print(f"  [ok] investment_intelligence.json")

    # Embedding index — chunks.json
    _write_json(EMBEDDING_DIR / "chunks.json", CHUNKS)
    print(f"  [ok] embedding_index/chunks.json ({len(CHUNKS)} chunks)")

    # Embedding index — vectors.npy (random float32, shape (N, DIM))
    rng = np.random.default_rng(seed=42)
    vectors = rng.standard_normal((len(CHUNKS), DIM)).astype(np.float32)
    np.save(EMBEDDING_DIR / "vectors.npy", vectors)
    print(f"  [ok] embedding_index/vectors.npy  (shape {vectors.shape})")

    # Embedding index — config.json
    _write_json(
        EMBEDDING_DIR / "config.json",
        {"model": DEFAULT_EMBEDDING_MODEL, "n_chunks": len(CHUNKS), "dim": DIM},
    )
    print(f"  [ok] embedding_index/config.json")

    # Embedding index — chunks.sqlite (built from chunks.json)
    _build_sqlite(CHUNKS, EMBEDDING_DIR / "chunks.sqlite")
    print(f"  [ok] embedding_index/chunks.sqlite ({len(CHUNKS)} rows)")

    # Pages
    for doc in DOC_LIST:
        fid = doc["file_id"]
        page_dir = PAGES_DIR / fid
        page_dir.mkdir(parents=True, exist_ok=True)
        (page_dir / "p1.txt").write_text(PAGE_CONTENT[fid], encoding="utf-8")
    print(f"  [ok] pages/ ({len(DOC_LIST)} document subdirs, 1 page each)")

    print()
    print("Done. MOCK-ingested/ is ready.")
    print(f"  Path: {INGESTED_DIR}")
    print()
    print("Next step:")
    print("  python scripts/run_mock_pipeline.py")


if __name__ == "__main__":
    create_mock_data()
