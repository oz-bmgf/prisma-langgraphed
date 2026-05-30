"""rerender node — renders focused.pdf and appendix.pdf from final_report.md.

Mirrors old repo's render_pdfs.py split approach:

  focused.pdf  — leadership-facing (~30-40 pages, no Table of Contents):
    # Portfolio Analysis Report (title)
    ## Executive Summary
    ## Portfolio Dashboard  (calibration metrics + confusion matrix + scatter)

  appendix.pdf — full audit trail (2-level auto-TOC prepended):
    ## Investment Analysis by Bundle of Work  (per-BOW deep dives)
    ## Cross-Cutting Analysis
    ## Bibliography

Split point: first occurrence of ``## Investment Analysis by Bundle of Work``.
If the heading is absent the full report is treated as focused and no appendix
is written (graceful fallback, mirrors old repo's behaviour at the ``# Appendix:``
marker).

PNGs written to threads_dir/ before rendering:
  team_vs_ai_confusion.png  — team-vs-AI severity matrix
  scatter_portfolio.png     — execution-rate vs approved-amount scatter
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

# ---------------------------------------------------------------------------
# Split helpers
# ---------------------------------------------------------------------------

_SPLIT_MARKER = "## Investment Analysis by Bundle of Work"


def _split_focused_and_appendix(md: str) -> tuple[str, str]:
    """Split markdown into (focused, appendix) at the Investment Analysis heading.

    Focused  = title + Executive Summary + Portfolio Dashboard sections.
    Appendix = Investment Analysis by BOW + Cross-Cutting Analysis + Bibliography.

    If the split marker is absent the entire document is returned as focused
    with an empty appendix string (same fallback as old render_pdfs.py).
    """
    idx = md.find(_SPLIT_MARKER)
    if idx < 0:
        return md, ""
    focused = md[:idx].rstrip() + "\n"
    appendix = md[idx:]
    return focused, appendix


def _strip_top_level_toc(md: str) -> str:
    """Remove the ``## Table of Contents`` block from the focused section.

    The focused PDF has only two content sections so in-document navigation
    is noise, not signal. Mirrors old render_pdfs._strip_top_level_toc().
    """
    return re.sub(
        r"(?ms)^## Table of Contents\n.*?(?=^##\s|\Z)",
        "",
        md,
        count=1,
    )


def _prepend_appendix_toc(program: str, appendix_md: str) -> str:
    """Build a 2-level Table of Contents and prepend a title page.

    Includes ## and ### headings only — deeper levels would produce a
    100-row ToC that obscures rather than helps navigation.
    Mirrors old render_pdfs._prepend_appendix_toc().
    """
    entries: list[str] = []
    for line in appendix_md.splitlines():
        m = re.match(r"^(#{2,3})\s+(.+)$", line)
        if not m:
            continue
        level = len(m.group(1))
        title = m.group(2).strip()
        slug = re.sub(r"[^a-z0-9 -]", "", title.lower()).replace(" ", "-")
        indent = "  " * (level - 2)  # ## → no indent, ### → 2 spaces
        entries.append(f"{indent}- [{title}](#{slug})")
    title_page = (
        f"# {program} — Appendix\n\n"
        "*Portfolio Risk Assessment — detailed analysis*\n\n"
        "## Table of Contents\n\n"
        + "\n".join(entries)
        + "\n\n"
    )
    return title_page + appendix_md


# ---------------------------------------------------------------------------
# Figure helpers
# ---------------------------------------------------------------------------

def _strip_base64_images(md: str) -> str:
    """Remove inline base64 PNG refs and their preceding bold label lines."""
    md = re.sub(r'\*\*Team vs AI Risk Severity\*\*\n\n', '', md)
    md = re.sub(r'\*\*Execution Rate vs Approved Amount\*\*\n\n', '', md)
    return _BASE64_IMG_RE.sub('', md)


def _embed_confusion_matrix(md: str, png_name: str) -> str:
    """Insert confusion matrix just before the Investment Analysis heading.

    Falls back to inserting before Cross-Cutting Analysis, then appending.
    """
    snippet = (
        f"\n**Team vs AI Risk Severity**\n\n"
        f"![Team vs AI severity confusion matrix]({png_name})\n\n"
    )
    for marker in (_SPLIT_MARKER, "## Cross-Cutting Analysis"):
        idx = md.find(marker)
        if idx >= 0:
            return md[:idx] + snippet + md[idx:]
    return md + snippet


def _embed_portfolio_scatter(md: str, png_name: str) -> str:
    """Insert scatter plot just before the Investment Analysis heading.

    Falls back to inserting before Cross-Cutting Analysis, then appending.
    """
    snippet = (
        f"\n**Execution Rate vs Approved Amount**\n\n"
        f"![Portfolio scatter]({png_name})\n\n"
    )
    for marker in (_SPLIT_MARKER, "## Cross-Cutting Analysis"):
        idx = md.find(marker)
        if idx >= 0:
            return md[:idx] + snippet + md[idx:]
    return md + snippet


def _generate_figures_sync(
    scope_outputs: list[dict],
    investment_scoring: dict,
    out_dir: Path,
) -> tuple[Optional[str], Optional[str]]:
    """Generate all figure PNGs synchronously (matplotlib is not async-safe).

    Returns (matrix_png_name | None, scatter_png_name | None).
    """
    b64_matrix = render_confusion_matrix(scope_outputs, investment_scoring)
    matrix_png_name: Optional[str] = None
    if b64_matrix:
        matrix_png_name = "team_vs_ai_confusion.png"
        (out_dir / matrix_png_name).write_bytes(base64.b64decode(b64_matrix))
        logger.info("rerender: wrote %s", matrix_png_name)

    b64_scatter = render_scatter_plot(scope_outputs, investment_scoring)
    scatter_png_name: Optional[str] = None
    if b64_scatter:
        scatter_png_name = "scatter_portfolio.png"
        (out_dir / scatter_png_name).write_bytes(base64.b64decode(b64_scatter))
        logger.info("rerender: wrote %s", scatter_png_name)

    return matrix_png_name, scatter_png_name


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

async def rerender(state: WorkflowState, config: RunnableConfig) -> dict:
    """Write PNG figures and render focused.pdf + appendix.pdf.

    focused.pdf  — leadership-facing: Executive Summary + Portfolio Dashboard.
    appendix.pdf — audit trail: Investment Analysis by BOW + Cross-Cutting + Bibliography,
                   with a 2-level Table of Contents prepended.
    """
    threads_dir: str = state.get("threads_dir") or ""
    scope_outputs: list[dict] = state.get("scope_outputs") or []
    investment_scoring: dict = state.get("investment_scoring") or {}
    program: str = state.get("program") or "Portfolio"

    # Prefer research-enriched report when available; fall back to analyze-only report.
    md_path_str = state.get("final_report_wresearch_md_path") or state.get("final_report_md_path")
    if not md_path_str:
        logger.warning(
            "rerender: no report md path in state "
            "(tried final_report_wresearch_md_path, final_report_md_path), skipping"
        )
        return {}

    md_src = Path(md_path_str)
    if not md_src.exists():
        logger.warning("rerender: report not found at %s, skipping", md_src)
        return {}

    out_dir = Path(threads_dir)
    # asyncio-APPROVED-1: to_thread wraps blocking mkdir
    await asyncio.to_thread(out_dir.mkdir, parents=True, exist_ok=True)

    # asyncio-APPROVED-1: to_thread wraps blocking file read
    report_md = await asyncio.to_thread(md_src.read_text, "utf-8")

    # ── 1. Generate figure PNGs ───────────────────────────────────────────────
    # asyncio-APPROVED-1: to_thread wraps blocking figure generation (matplotlib)
    matrix_png, scatter_png = await asyncio.to_thread(
        _generate_figures_sync, scope_outputs, investment_scoring, out_dir
    )

    # ── 2. Strip existing inline base64 refs, re-embed as basename refs ───────
    report_md = _strip_base64_images(report_md)
    if matrix_png:
        report_md = _embed_confusion_matrix(report_md, matrix_png)
    if scatter_png:
        report_md = _embed_portfolio_scatter(report_md, scatter_png)

    # ── 3. Split into focused + appendix ─────────────────────────────────────
    focused_md, appendix_md = _split_focused_and_appendix(report_md)
    focused_md = _strip_top_level_toc(focused_md)

    # ── 4. Write focused.md and render focused.pdf ───────────────────────────
    focused_md_path = out_dir / "focused.md"
    # asyncio-APPROVED-1: to_thread wraps blocking Path.write_text
    await asyncio.to_thread(focused_md_path.write_text, focused_md, "utf-8")

    focused_pdf_path = out_dir / "focused.pdf"
    # asyncio-APPROVED-1: to_thread wraps blocking render_pdf (weasyprint)
    focused_ok = await asyncio.to_thread(
        _render_pdf, str(focused_md_path), str(focused_pdf_path), toc=False
    )
    if focused_ok:
        logger.info("rerender: focused.pdf written to %s", focused_pdf_path)
    else:
        logger.warning("rerender: focused PDF render failed (weasyprint unavailable or error)")

    # ── 5. Write appendix.md (with prepended TOC) and render appendix.pdf ────
    appendix_pdf_path_str: Optional[str] = None
    if appendix_md:
        appendix_md_with_toc = _prepend_appendix_toc(program, appendix_md)
        appendix_md_path = out_dir / "appendix.md"
        # asyncio-APPROVED-1: to_thread wraps blocking Path.write_text
        await asyncio.to_thread(appendix_md_path.write_text, appendix_md_with_toc, "utf-8")

        appendix_pdf_path = out_dir / "appendix.pdf"
        # asyncio-APPROVED-1: to_thread wraps blocking render_pdf (weasyprint)
        appendix_ok = await asyncio.to_thread(
            _render_pdf, str(appendix_md_path), str(appendix_pdf_path), toc=True
        )
        if appendix_ok:
            logger.info("rerender: appendix.pdf written to %s", appendix_pdf_path)
            appendix_pdf_path_str = str(appendix_pdf_path)
        else:
            logger.warning("rerender: appendix PDF render failed (weasyprint unavailable or error)")
    else:
        logger.info("rerender: no appendix section found — focused PDF only")

    result: dict = {}
    if focused_ok:
        result["final_report_pdf_path"] = str(focused_pdf_path)
    if appendix_pdf_path_str:
        result["appendix_pdf_path"] = appendix_pdf_path_str
    return result
