"""precheck node — fast integrity gate; validates pre-existing artifacts."""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import sqlite3
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Iterable, Optional

from langchain_core.runnables import RunnableConfig

from src.graph.state import WorkflowState


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


class _Status(Enum):
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"
    SKIP = "SKIP"


@dataclass
class _CheckResult:
    name: str
    status: _Status
    message: str = ""
    detail: dict = field(default_factory=dict)


@dataclass
class _PrecheckResult:
    results: list[_CheckResult]

    @property
    def n_pass(self) -> int:
        return sum(1 for r in self.results if r.status is _Status.PASS)

    @property
    def n_warn(self) -> int:
        return sum(1 for r in self.results if r.status is _Status.WARN)

    @property
    def n_fail(self) -> int:
        return sum(1 for r in self.results if r.status is _Status.FAIL)

    @property
    def n_skip(self) -> int:
        return sum(1 for r in self.results if r.status is _Status.SKIP)

    @property
    def overall(self) -> _Status:
        if self.n_fail:
            return _Status.FAIL
        if self.n_warn:
            return _Status.WARN
        return _Status.PASS


# ---------------------------------------------------------------------------
# Individual check functions (synchronous — disk I/O only, no LLM)
# ---------------------------------------------------------------------------


def _check_doc_list(ingested_dir: Path) -> _CheckResult:
    doc_list_path = ingested_dir / "doc_list.json"
    if not doc_list_path.exists():
        return _CheckResult("doc_list.json", _Status.FAIL, "doc_list.json missing")
    try:
        data = json.loads(doc_list_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        return _CheckResult("doc_list.json", _Status.FAIL, f"read error: {exc}")
    if not isinstance(data, list) or len(data) == 0:
        return _CheckResult("doc_list.json", _Status.FAIL, "doc_list.json is empty or not a list")
    return _CheckResult("doc_list.json", _Status.PASS, f"{len(data)} documents")


def _check_investment_scoring(ingested_dir: Path) -> _CheckResult:
    path = ingested_dir / "investment_scoring.json"
    if not path.exists():
        return _CheckResult("investment_scoring.json", _Status.FAIL, "investment_scoring.json missing")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        return _CheckResult("investment_scoring.json", _Status.FAIL, f"read error: {exc}")
    if not isinstance(data, dict) or len(data) == 0:
        return _CheckResult("investment_scoring.json", _Status.WARN, "investment_scoring.json is empty")
    return _CheckResult("investment_scoring.json", _Status.PASS, f"{len(data)} investments")


def _check_bow_investment_map(ingested_dir: Path) -> _CheckResult:
    path = ingested_dir / "bow_investment_map.json"
    if not path.exists():
        return _CheckResult("bow_investment_map.json", _Status.FAIL, "bow_investment_map.json missing")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        return _CheckResult("bow_investment_map.json", _Status.FAIL, f"read error: {exc}")
    if not isinstance(data, dict):
        return _CheckResult("bow_investment_map.json", _Status.FAIL, "expected dict")
    return _CheckResult("bow_investment_map.json", _Status.PASS, f"{len(data)} BoWs")


def _check_embedding_index(ingested_dir: Path) -> _CheckResult:
    index_dir = ingested_dir / "embedding_index"
    chunks_json = index_dir / "chunks.json"
    if not index_dir.is_dir():
        return _CheckResult("embedding_index/", _Status.FAIL, "embedding_index/ dir missing")
    if not chunks_json.exists():
        return _CheckResult("embedding_index/", _Status.FAIL, "chunks.json missing")
    try:
        data = json.loads(chunks_json.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        return _CheckResult("embedding_index/", _Status.FAIL, f"chunks.json read error: {exc}")
    n = len(data) if isinstance(data, list) else 0
    if n == 0:
        return _CheckResult("embedding_index/", _Status.FAIL, "chunks.json is empty")
    return _CheckResult("embedding_index/", _Status.PASS, f"{n} chunks in index")


def _check_pages_dir(ingested_dir: Path) -> _CheckResult:
    pages_dir = ingested_dir / "pages"
    if not pages_dir.is_dir():
        return _CheckResult("pages/", _Status.WARN, "pages/ dir missing — image reads will fail")
    n_dirs = sum(1 for d in pages_dir.iterdir() if d.is_dir())
    return _CheckResult("pages/", _Status.PASS, f"{n_dirs} per-file dirs")


def _check_focus_bows(ingested_dir: Path, focus_bows: Optional[list[str]]) -> _CheckResult:
    if not focus_bows:
        return _CheckResult("focus_bows", _Status.SKIP, "no focus_bows specified")
    bow_map_path = ingested_dir / "bow_investment_map.json"
    if not bow_map_path.exists():
        return _CheckResult("focus_bows", _Status.FAIL, "bow_investment_map.json missing")
    bow_map = json.loads(bow_map_path.read_text(encoding="utf-8"))
    known = set(bow_map) if isinstance(bow_map, dict) else set()
    unknown = [b for b in focus_bows if b not in known]
    if unknown:
        return _CheckResult("focus_bows", _Status.FAIL, f"unknown BoW IDs: {unknown}", detail={"unknown": unknown})
    return _CheckResult("focus_bows", _Status.PASS, f"all {len(focus_bows)} BoW IDs valid")


def _check_api_keys() -> _CheckResult:
    missing = []
    if not os.environ.get("ANTHROPIC_API_KEY"):
        missing.append("ANTHROPIC_API_KEY")
    if missing:
        return _CheckResult("API keys", _Status.FAIL, f"missing: {missing}")
    return _CheckResult("API keys", _Status.PASS, "ANTHROPIC_API_KEY set")


def _check_disk_space(ingested_dir: Path, min_free_gb: float = 20.0) -> _CheckResult:
    try:
        _, _, free = shutil.disk_usage(str(ingested_dir))
    except OSError as exc:
        return _CheckResult("disk space", _Status.SKIP, f"statvfs failed: {exc}")
    free_gb = free / (1024 ** 3)
    if free_gb >= min_free_gb:
        return _CheckResult("disk space", _Status.PASS, f"{free_gb:.1f} GB free")
    return _CheckResult("disk space", _Status.WARN, f"only {free_gb:.1f} GB free (recommended ≥{min_free_gb:.0f} GB)")


def _run_checks(ingested_dir: Path, focus_bows: Optional[list[str]]) -> _PrecheckResult:
    return _PrecheckResult(results=[
        _check_doc_list(ingested_dir),
        _check_investment_scoring(ingested_dir),
        _check_bow_investment_map(ingested_dir),
        _check_embedding_index(ingested_dir),
        _check_pages_dir(ingested_dir),
        _check_focus_bows(ingested_dir, focus_bows),
        _check_api_keys(),
        _check_disk_space(ingested_dir),
    ])


_GLYPH = {
    _Status.PASS: "✓",
    _Status.WARN: "⚠",
    _Status.FAIL: "✗",
    _Status.SKIP: "—",
}


def _format_report(result: _PrecheckResult) -> str:
    name_w = max((len(r.name) for r in result.results), default=10)
    sep = "=" * (name_w + 56)
    lines = [sep, f"{'CHECK'.ljust(name_w)}  STATUS  DETAIL", sep]
    for r in result.results:
        lines.append(f"{r.name.ljust(name_w)}  {_GLYPH[r.status]} {r.status.value:<5}  {r.message}")
    lines.append(sep)
    g = _GLYPH[result.overall]
    lines.append(
        f"OVERALL: {g} {result.overall.value}  "
        f"(pass={result.n_pass}, warn={result.n_warn}, fail={result.n_fail}, skip={result.n_skip})"
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------


async def precheck(state: WorkflowState, config: RunnableConfig) -> dict:
    ingested_dir = Path(state["ingested_dir"])
    focus_bows: Optional[list[str]] = state.get("focus_bows")

    # asyncio-APPROVED-1: to_thread wraps blocking filesystem validation checks
    result = await asyncio.to_thread(_run_checks, ingested_dir, focus_bows)
    report = _format_report(result)
    passed = result.overall is not _Status.FAIL

    return {"precheck_passed": passed, "precheck_report": report}
