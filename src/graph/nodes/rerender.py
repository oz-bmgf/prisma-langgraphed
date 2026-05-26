"""rerender node — writes figure PNGs to disk and renders focused.pdf from final_report_wresearch.md.

Mirrors old repo's reassemble + render_pdfs approach:
  1. Call report_charts functions → decode base64 → save PNG files to threads_dir/
  2. Strip existing inline base64 image refs from final_report_wresearch.md
  3. Re-embed figures by basename — weasyprint resolves via base_url=threads_dir/
  4. Split at the per-BOW deep-dive section → keep only the leadership-facing portion
  5. Strip the Table of Contents (focused doc is too short to need navigation)
  6. Write focused.md alongside the PNGs, render focused.pdf (no auto-TOC)

focused.md contains (mirrors old repo's focused PDF):
  Executive Summary  — portfolio risk narrative
  Portfolio Dashboard — calibration metrics, team-vs-AI confusion matrix
  (stops before ## Investment Analysis by Bundle of Work)

PNGs written to threads_dir/:
  team_vs_ai_confusion.png      — portfolio-wide team-vs-AI severity matrix
  scatter_{safe_bow_id}.png     — per-BOW execution-rate vs approved-amount scatter
"""
from __future__ import annotations

import asyncio
import base64
import logging
import re
from pathlib import Path
from typing import Optional

from langchain_core.runnables import RunnableConfig

from src.core.report_charts import render_confusion_matrix, render_scatter_plot
from src.core.report_renderer import render_pdf as _render_pdf
from src.graph.state import WorkflowState

logger = logging.getLogger(__name__)

_BASE64_IMG_RE = re.compile(r'!\[[^\]]*\]\(data:image/png;base64,[^)]+\)\n*')


def _group_single_bow_scopes(scope_outputs: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {}
    for s in scope_outputs:
        bow_ids = s.get("bow_ids") or []
        if len(bow_ids) == 1:
            groups.setdefault(bow_ids[0], []).append(s)
    return groups


def _split_focused(md: str) -> str:
    """Keep only the leadership-facing sections (Executive Summary + Portfolio Dashboard).

    Splits before ## Investment Analysis by Bundle of Work, which is where the
    per-BOW deep dive begins. Falls back to the full report if the marker is absent.
    Also strips the auto-generated Table of Contents — the focused doc is short
    enough that in-document navigation is noise, not signal.
    """
    split_marker = "## Investment Analysis by Bundle of Work"
    idx = md.find(split_marker)
    focused = md[:idx].rstrip() + "\n" if idx >= 0 else md

    # Strip ## Table of Contents block (runs until the next ## heading)
    focused = re.sub(
        r"(?ms)^## Table of Contents\n.*?(?=^##\s|\Z)",
        "",
        focused,
        count=1,
    )
    return focused


def _strip_base64_images(md: str) -> str:
    """Remove inline base64 PNG refs and their preceding bold label lines."""
    md = re.sub(r'\*\*Team vs AI Risk Severity\*\*\n\n', '', md)
    return _BASE64_IMG_RE.sub('', md)


def _embed_confusion_matrix(md: str, png_name: str) -> str:
    """Insert confusion matrix at the end of the Portfolio Dashboard section.

    The dashboard is the calibration section — the matrix belongs there.
    Falls back to appending before the next major section if the dashboard
    heading is absent.
    """
    snippet = (
        f"\n**Team vs AI Risk Severity**\n\n"
        f"![Team vs AI severity confusion matrix]({png_name})\n\n"
    )
    # Insert before the first section that follows the dashboard
    for marker in (
        "## Investment Analysis by Bundle of Work",
        "## Cross-Cutting Analysis",
    ):
        idx = md.find(marker)
        if idx >= 0:
            return md[:idx] + snippet + md[idx:]
    return md + snippet


def _embed_bow_scatter(md: str, bow_id: str, png_name: str) -> str:
    """Insert scatter plot on the line after the BOW section heading."""
    lines = md.split("\n")
    for i, line in enumerate(lines):
        if line.startswith("#") and bow_id.lower() in line.lower():
            lines.insert(i + 1, f"\n![{bow_id} scatter]({png_name})\n")
            return "\n".join(lines)
    return md


def _generate_figures_sync(
    scope_outputs: list[dict],
    investment_scoring: dict,
    out_dir: Path,
    bow_groups: dict[str, list[dict]],
) -> tuple[Optional[str], dict[str, str]]:
    """Generate all figure PNGs synchronously (matplotlib is not async-safe).

    Returns (matrix_png_name | None, {bow_id: png_name}).
    """
    b64_matrix = render_confusion_matrix(scope_outputs, investment_scoring)
    matrix_png_name: Optional[str] = None
    if b64_matrix:
        matrix_png_name = "team_vs_ai_confusion.png"
        (out_dir / matrix_png_name).write_bytes(base64.b64decode(b64_matrix))
        logger.info("rerender: wrote %s", matrix_png_name)

    bow_png_names: dict[str, str] = {}
    for bow_id, scopes in bow_groups.items():
        b64 = render_scatter_plot(bow_id, scopes, investment_scoring)
        if b64:
            safe_id = re.sub(r'[^a-z0-9]', '_', bow_id.lower())
            fname = f"scatter_{safe_id}.png"
            (out_dir / fname).write_bytes(base64.b64decode(b64))
            bow_png_names[bow_id] = fname
            logger.info("rerender: wrote %s", fname)

    return matrix_png_name, bow_png_names


async def rerender(state: WorkflowState, config: RunnableConfig) -> dict:
    """Write PNG figures to disk and render focused.pdf from final_report.md."""
    threads_dir = state.get("threads_dir") or ""
    scope_outputs: list[dict] = state.get("scope_outputs") or []
    investment_scoring: dict = state.get("investment_scoring") or {}

    # Source: final_report_wresearch.md — finalize rewrites the Executive Summary
    # using research results, so the focused PDF must reflect those enrichments.
    md_path_str = state.get("final_report_wresearch_md_path")
    if not md_path_str:
        logger.warning("rerender: final_report_wresearch_md_path not in state, skipping")
        return {}

    md_src = Path(md_path_str)
    if not md_src.exists():
        logger.warning("rerender: final_report_wresearch.md not found at %s, skipping", md_src)
        return {}

    out_dir = Path(threads_dir)
    # asyncio-APPROVED-1: to_thread wraps blocking mkdir
    await asyncio.to_thread(out_dir.mkdir, parents=True, exist_ok=True)

    # asyncio-APPROVED-1: to_thread wraps blocking file read
    report_md = await asyncio.to_thread(md_src.read_text, "utf-8")

    bow_groups = _group_single_bow_scopes(scope_outputs)

    # asyncio-APPROVED-1: to_thread wraps blocking figure generation (matplotlib + PNG write)
    matrix_png, bow_pngs = await asyncio.to_thread(
        _generate_figures_sync, scope_outputs, investment_scoring, out_dir, bow_groups
    )

    # Strip existing inline base64 refs, re-embed as filesystem basename refs
    report_md = _strip_base64_images(report_md)
    if matrix_png:
        report_md = _embed_confusion_matrix(report_md, matrix_png)
    for bow_id, png_name in bow_pngs.items():
        report_md = _embed_bow_scatter(report_md, bow_id, png_name)

    # Keep only Executive Summary + Portfolio Dashboard (calibration); drop per-BOW deep dive
    focused_md = _split_focused(report_md)

    # Write focused.md in threads_dir/ — weasyprint resolves PNG basenames
    # against this directory via base_url=str(focused_md_path.parent)
    focused_md_path = out_dir / "focused.md"
    # asyncio-APPROVED-1: to_thread wraps blocking Path.write_text
    await asyncio.to_thread(focused_md_path.write_text, focused_md, "utf-8")

    # Render focused.pdf — no auto-TOC (executive-facing doc, short enough to navigate inline)
    focused_pdf_path = out_dir / "focused.pdf"
    # asyncio-APPROVED-1: to_thread wraps blocking render_pdf
    success = await asyncio.to_thread(
        _render_pdf, str(focused_md_path), str(focused_pdf_path), toc=False
    )

    if success:
        logger.info("rerender: focused.pdf written to %s", focused_pdf_path)
        return {"final_report_pdf_path": str(focused_pdf_path)}

    logger.warning("rerender: PDF render failed (weasyprint unavailable or error)")
    return {}
