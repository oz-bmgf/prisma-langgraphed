"""deliver node — writes final artifacts to the experiments run folder.

Files written into {base_dir}/{program}-experiments/run-{run_name}/:
  final_report.md         — already written by assemble_report; copy here
  final_report.pdf        — rendered by pandoc/weasyprint if available (graceful skip)
  excerpts.csv            — all annotated excerpts from link investigations
  run_meta.json           — run metadata: coverage_pct, grade, model, collection, etc.
  numerical_provenance.json  — InvestmentFacts per investment
  allocation_verification.json
  numerical_verification.json
"""
from __future__ import annotations

import asyncio
import csv
import io
import json
import logging
import shutil
from pathlib import Path
from typing import Optional

from langchain_core.runnables import RunnableConfig

from src.core.report_renderer import render_pdf as _render_pdf
from src.graph.state import WorkflowState

logger = logging.getLogger(__name__)


def _copy_file(src: Path, dst: Path) -> None:
    if src.resolve() == dst.resolve():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(src), str(dst))


def _write_json_file(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    path.write_text(buf.getvalue(), encoding="utf-8")


async def deliver(state: WorkflowState, config: RunnableConfig) -> dict:
    program: str = state["program"]
    run_name: str = state["run_name"]
    base_dir: str = state["base_dir"]

    threads_dir: str = state.get("threads_dir") or str(
        Path(base_dir) / f"{program}-experiments" / f"run-{run_name}"
    )
    delivery_dir = Path(threads_dir)
    # asyncio-APPROVED-1: to_thread wraps blocking mkdir
    await asyncio.to_thread(delivery_dir.mkdir, parents=True, exist_ok=True)

    delivered: list[str] = []

    # ── final_report.md ───────────────────────────────────────────────────────
    md_path_str: Optional[str] = state.get("final_report_md_path")
    if md_path_str:
        md_src = Path(md_path_str)
        if md_src.exists():
            md_dst = delivery_dir / "final_report.md"
            # asyncio-APPROVED-1: to_thread wraps blocking shutil.copy2
            await asyncio.to_thread(_copy_file, md_src, md_dst)
            delivered.append(str(md_dst))

            # ── final_report.pdf (skip if rerender already produced one) ──
            if not state.get("final_report_pdf_path"):
                pdf_dst = delivery_dir / "final_report.pdf"
                success = await asyncio.to_thread(_render_pdf, str(md_src), str(pdf_dst))
                if success:
                    delivered.append(str(pdf_dst))
                else:
                    logger.info("deliver: PDF render skipped (weasyprint unavailable or error)")

    # ── final_report.pdf (copy pre-rendered PDF if provided) ─────────────────
    pdf_path_str: Optional[str] = state.get("final_report_pdf_path")
    if pdf_path_str:
        pdf_src = Path(pdf_path_str)
        if pdf_src.exists():
            pdf_dst = delivery_dir / "final_report.pdf"
            # asyncio-APPROVED-1: to_thread wraps blocking shutil.copy2
            await asyncio.to_thread(_copy_file, pdf_src, pdf_dst)
            delivered.append(str(pdf_dst))

    # Also handle wresearch report if present
    wresearch_str: Optional[str] = state.get("final_report_wresearch_md_path")
    if wresearch_str:
        wr_src = Path(wresearch_str)
        if wr_src.exists():
            wr_dst = delivery_dir / "final_report_wresearch.md"
            await asyncio.to_thread(_copy_file, wr_src, wr_dst)
            delivered.append(str(wr_dst))

    # ── excerpts.csv ──────────────────────────────────────────────────────────
    all_excerpts: list[dict] = state.get("all_excerpts") or []  # type: ignore[assignment]
    if all_excerpts:
        excerpts_dst = delivery_dir / "excerpts.csv"
        fieldnames = ["inv_id", "scope_id", "link_id", "text", "source", "page",
                      "significance", "numerical_facts", "credibility_tier"]
        # Normalise numerical_facts to string
        normalised = [
            {**ex, "numerical_facts": ", ".join(str(n) for n in (ex.get("numerical_facts") or []))}
            for ex in all_excerpts
        ]
        try:
            # asyncio-APPROVED-1: to_thread wraps blocking CSV write
            await asyncio.to_thread(_write_csv, excerpts_dst, normalised, fieldnames)
            delivered.append(str(excerpts_dst))
        except Exception as exc:
            logger.warning("deliver: excerpts.csv write failed: %s", exc)

    # ── numerical_provenance.json ─────────────────────────────────────────────
    numerical_provenance: list[dict] = state.get("numerical_provenance") or []  # type: ignore[assignment]
    if numerical_provenance:
        prov_dst = delivery_dir / "numerical_provenance.json"
        try:
            await asyncio.to_thread(_write_json_file, prov_dst, numerical_provenance)
            delivered.append(str(prov_dst))
        except Exception as exc:
            logger.warning("deliver: numerical_provenance.json write failed: %s", exc)

    # ── allocation_verification.json ──────────────────────────────────────────
    alloc_verif: list[dict] = state.get("allocation_verification") or []  # type: ignore[assignment]
    if alloc_verif:
        alloc_dst = delivery_dir / "allocation_verification.json"
        try:
            await asyncio.to_thread(_write_json_file, alloc_dst, alloc_verif)
            delivered.append(str(alloc_dst))
        except Exception as exc:
            logger.warning("deliver: allocation_verification.json write failed: %s", exc)

    # ── numerical_verification.json ───────────────────────────────────────────
    num_verif: list[dict] = state.get("numerical_verification") or []  # type: ignore[assignment]
    if num_verif:
        num_dst = delivery_dir / "numerical_verification.json"
        try:
            await asyncio.to_thread(_write_json_file, num_dst, num_verif)
            delivered.append(str(num_dst))
        except Exception as exc:
            logger.warning("deliver: numerical_verification.json write failed: %s", exc)

    # ── run_meta.json ─────────────────────────────────────────────────────────
    run_meta: Optional[dict] = state.get("run_meta") or {}
    run_meta_out = {
        **(run_meta or {}),
        "program": program,
        "run_name": run_name,
        "collection": state.get("collection_name", ""),
        "model": state.get("synthesis_model", ""),
        "total_findings": sum(
            len((s.get("section_draft") or {}).get("ranked_deviations", []))
            for s in (state.get("scope_outputs") or [])
        ),
        "documents_read": len({
            ex.get("source", "")
            for ex in all_excerpts
            if ex.get("source")
        }),
        "coverage_pct": float(state.get("coverage_pct") or 0.0),
        "grade": state.get("grade") or "D",
        "delivered_files": delivered,
    }
    meta_dst = delivery_dir / "run_meta.json"
    # asyncio-APPROVED-1: to_thread wraps blocking JSON write
    await asyncio.to_thread(_write_json_file, meta_dst, run_meta_out)

    return {
        "excerpts_csv_path": str(delivery_dir / "excerpts.csv") if all_excerpts else None,
        "numerical_verification_path": str(delivery_dir / "numerical_verification.json") if num_verif else None,
        "allocation_verification_path": str(delivery_dir / "allocation_verification.json") if alloc_verif else None,
    }
