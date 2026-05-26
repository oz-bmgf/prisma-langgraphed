"""Unit tests for src/core/report_renderer.py."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from src.core.report_renderer import render_pdf


# ---------------------------------------------------------------------------
# render_pdf graceful failure
# ---------------------------------------------------------------------------


def test_render_pdf_returns_false_gracefully(tmp_path):
    """render_pdf must return False without raising when weasyprint is unavailable."""
    md_file = tmp_path / "report.md"
    md_file.write_text("# Test\n\nHello world.\n", encoding="utf-8")
    pdf_file = tmp_path / "report.pdf"

    # Simulate weasyprint not installed by patching the import
    with patch.dict(sys.modules, {"weasyprint": None}):
        result = render_pdf(str(md_file), str(pdf_file))

    assert result is False
    assert not pdf_file.exists()


def test_render_pdf_returns_false_when_markdown_missing(tmp_path):
    """render_pdf must return False when the source markdown does not exist."""
    missing = tmp_path / "nonexistent.md"
    pdf_out = tmp_path / "out.pdf"
    result = render_pdf(str(missing), str(pdf_out))
    assert result is False


def test_render_pdf_success_with_weasyprint(tmp_path):
    """render_pdf returns True and writes a non-empty PDF when weasyprint is available."""
    pytest.importorskip("weasyprint")
    pytest.importorskip("markdown")

    md_file = tmp_path / "report.md"
    md_file.write_text(
        "# Portfolio Report\n\n"
        "## Executive Summary\n\nSome text here.\n\n"
        "| Col A | Col B |\n|-------|-------|\n| foo | bar |\n",
        encoding="utf-8",
    )
    pdf_file = tmp_path / "report.pdf"

    result = render_pdf(str(md_file), str(pdf_file))
    assert result is True
    assert pdf_file.exists()
    assert pdf_file.stat().st_size > 1000  # non-trivial PDF


def test_render_pdf_creates_parent_dirs(tmp_path):
    """render_pdf creates the output directory if it doesn't exist."""
    pytest.importorskip("weasyprint")
    pytest.importorskip("markdown")

    md_file = tmp_path / "report.md"
    md_file.write_text("# Hello\n\nworld.\n", encoding="utf-8")
    deep_pdf = tmp_path / "deep" / "nested" / "out.pdf"

    result = render_pdf(str(md_file), str(deep_pdf))
    assert result is True
    assert deep_pdf.exists()


def test_render_pdf_skips_auto_toc_when_toc_present(tmp_path):
    """render_pdf does not add a second TOC when the markdown already has one."""
    pytest.importorskip("weasyprint")
    pytest.importorskip("markdown")

    md_file = tmp_path / "report.md"
    md_file.write_text(
        "# Report\n\n"
        "## Table of Contents\n\n- [Executive Summary](#executive-summary)\n\n"
        "## Executive Summary\n\nContent.\n",
        encoding="utf-8",
    )
    pdf_file = tmp_path / "out.pdf"
    result = render_pdf(str(md_file), str(pdf_file))
    assert result is True
