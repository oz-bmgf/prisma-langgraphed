"""Markdown → PDF renderer using weasyprint.

Pipeline: markdown → HTML (python-markdown + custom table pre-processor)
          → PDF (weasyprint with professional CSS).

CSS and table-width optimizer ported from
prisma-ai-review/src/qpr/md_to_pdf.py (commit history: render_pdfs refactor).

Public API:
    render_pdf(markdown_path, output_pdf_path) -> bool

The function never raises — it logs a warning and returns False when the
PDF tool is absent or fails, so the pipeline always has the markdown
fallback.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CSS — professional A4 layout
# Ported from prisma-ai-review/src/qpr/md_to_pdf.py
# ---------------------------------------------------------------------------

_CSS = """
@page {
    size: A4;
    margin: 2.5cm 2cm 2.5cm 2cm;
    @bottom-center {
        content: counter(page);
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        font-size: 9pt;
        color: #888;
    }
}

body {
    font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    font-size: 10pt;
    line-height: 1.55;
    color: #222;
    max-width: 100%;
}

/* Embedded figures (base64 PNG charts) scale to content width. */
img {
    width: 100%;
    max-width: 100%;
    height: auto;
    display: block;
    margin: 12pt auto;
}

h1 {
    font-size: 22pt;
    color: #1a3a5c;
    border-bottom: 3pt solid #1a3a5c;
    padding-bottom: 6pt;
    margin-top: 0;
    margin-bottom: 16pt;
}

h2 {
    font-size: 16pt;
    color: #1a3a5c;
    border-bottom: 1.5pt solid #ccd6e0;
    padding-bottom: 4pt;
    margin-top: 28pt;
    margin-bottom: 12pt;
    page-break-after: avoid;
}

h3 {
    font-size: 13pt;
    color: #2a5a8c;
    margin-top: 20pt;
    margin-bottom: 8pt;
    page-break-after: avoid;
}

h4 {
    font-size: 11pt;
    color: #444;
    margin-top: 12pt;
    margin-bottom: 6pt;
}

h5 {
    font-size: 10pt;
    color: #555;
    margin-top: 10pt;
    margin-bottom: 4pt;
}

p { margin-bottom: 8pt; text-align: justify; }
strong { color: #1a3a5c; }
em { color: #555; }

ul, ol { margin-bottom: 10pt; padding-left: 24pt; }
li { margin-bottom: 4pt; }

hr {
    border: none;
    border-top: 1pt solid #ddd;
    margin: 20pt 0;
}

a { color: #2563eb; text-decoration: none; }

/* Table of Contents */
nav#TOC {
    background-color: #f8f9fb;
    border: 1pt solid #ddd;
    border-radius: 4pt;
    padding: 16pt 20pt;
    margin-bottom: 24pt;
    page-break-after: always;
}
nav#TOC ul { list-style-type: none; padding-left: 0; }
nav#TOC > ul > li { font-weight: bold; font-size: 11pt; margin-top: 6pt; }
nav#TOC > ul > li > ul > li { font-weight: normal; font-size: 10pt; padding-left: 20pt; }
nav#TOC a { color: #2a5a8c; }
nav#TOC .toc-h3 { padding-left: 20pt; font-size: 10pt; font-weight: normal; }
nav#TOC .toc-h4 { padding-left: 40pt; font-size: 9.5pt; font-weight: normal; }

/* Tables */
table {
    table-layout: fixed;
    width: 100%;
    border-collapse: collapse;
    font-size: 9pt;
    line-height: 1.35;
    margin: 8pt 0;
    page-break-inside: auto;
}
table tr {
    page-break-inside: avoid;
    break-inside: avoid;
}
table th, table td {
    border: 0.5pt solid #ccd6e0;
    padding: 4pt 6pt;
    vertical-align: top;
    word-wrap: break-word;
    overflow-wrap: anywhere;
}
table th {
    background-color: #f1f5f9;
    color: #1a3a5c;
    text-align: left;
    font-weight: 600;
}
table tr:nth-child(even) td { background-color: #fafbfc; }
"""

# ---------------------------------------------------------------------------
# Table-width optimizer (ported from md_to_pdf.py)
# ---------------------------------------------------------------------------

_TABLE_BLOCK_RE = re.compile(
    r"(^\|.+\|[ \t]*\n^\|[ \t]*[-:][ \t\-:|]*\|[ \t]*\n(?:^\|.+\|[ \t]*\n)+)",
    re.MULTILINE,
)
_TABLE_TOTAL_CHARS = 100
_MIN_COL_PCT = 8
_MAX_COL_PCT = 75


def _strip_md_inline(s: str) -> str:
    s = re.sub(r"\*\*(.+?)\*\*", r"\1", s)
    s = re.sub(r"\*(.+?)\*", r"\1", s)
    s = re.sub(r"`([^`]+)`", r"\1", s)
    s = re.sub(r"\[(.+?)\]\([^)]+\)", r"\1", s)
    return s


def _wrap_lines(text: str, width: int) -> int:
    if width <= 0:
        return 10**6
    if not text:
        return 1
    text = _strip_md_inline(text)
    lines = 1
    col = 0
    for tok in text.split(" "):
        while len(tok) > width:
            tok = tok[width:]
            if col > 0:
                lines += 1
                col = 0
            lines += 1
        need = len(tok) + (1 if col > 0 else 0)
        if col + need > width:
            lines += 1
            col = len(tok)
        else:
            col += need
    return max(1, lines)


def _table_height(rows: list[list[str]], widths_pct: list[int]) -> int:
    widths_chars = [max(1, int(round(w * _TABLE_TOTAL_CHARS / 100))) for w in widths_pct]
    total = 0
    for row in rows:
        h = 1
        for i, cell in enumerate(row):
            if i < len(widths_chars):
                h = max(h, _wrap_lines(cell, widths_chars[i]))
        total += h
    return total


def _initial_widths(rows: list[list[str]], n: int, min_pct: int) -> list[int]:
    max_lens = [0] * n
    for r in rows:
        for i in range(min(len(r), n)):
            max_lens[i] = max(max_lens[i], len(_strip_md_inline(r[i])))
    s = sum(max_lens) or 1
    raw = [max_lens[i] * 100 / s for i in range(n)]
    widths = [max(min_pct, min(_MAX_COL_PCT, int(round(r)))) for r in raw]

    def _adjust(direction: int) -> bool:
        for idx in range(n):
            if direction > 0 and widths[idx] < _MAX_COL_PCT:
                widths[idx] += 1
                return True
            if direction < 0 and widths[idx] > min_pct:
                widths[idx] -= 1
                return True
        return False

    for _ in range(500):
        diff = 100 - sum(widths)
        if diff == 0:
            break
        if not _adjust(1 if diff > 0 else -1):
            break
    widths = [max(min_pct, min(_MAX_COL_PCT, w)) for w in widths]
    return widths


def _optimize_widths(rows: list[list[str]]) -> list[int]:
    n = max(len(r) for r in rows)
    if n <= 1:
        return [100]
    min_pct = _MIN_COL_PCT
    if min_pct * n > 100:
        min_pct = max(1, 100 // n)
    _RENDER_CHARS = 60

    def _longest_word(cell: str) -> int:
        text = _strip_md_inline(cell)
        return max((len(w) for w in text.split()), default=1)

    per_col_min = []
    for i in range(n):
        longest = 1
        for r in rows:
            if i < len(r):
                longest = max(longest, _longest_word(r[i]))
        h_pct = int(round((longest + 2) * 100 / _RENDER_CHARS))
        per_col_min.append(max(min_pct, min(_MAX_COL_PCT, h_pct)))
    if sum(per_col_min) > 100:
        scale = 100 / sum(per_col_min)
        per_col_min = [max(1, int(p * scale)) for p in per_col_min]

    widths = _initial_widths(rows, n, min_pct)
    for i in range(n):
        if widths[i] < per_col_min[i]:
            deficit = per_col_min[i] - widths[i]
            for j in sorted(range(n), key=lambda k: -widths[k]):
                if j == i:
                    continue
                avail = widths[j] - per_col_min[j]
                if avail <= 0:
                    continue
                take = min(avail, deficit)
                widths[j] -= take
                widths[i] += take
                deficit -= take
                if deficit == 0:
                    break

    best_h = _table_height(rows, widths)
    for _ in range(200):
        improved = False
        for src in range(n):
            for dst in range(n):
                if src == dst:
                    continue
                if widths[src] - 1 < per_col_min[src]:
                    continue
                if widths[dst] + 1 > _MAX_COL_PCT:
                    continue
                widths[src] -= 1
                widths[dst] += 1
                h = _table_height(rows, widths)
                if h < best_h:
                    best_h = h
                    improved = True
                else:
                    widths[src] += 1
                    widths[dst] -= 1
        if not improved:
            break
    return widths


def _md_inline_to_html(s: str) -> str:
    import html as _html
    s = _html.escape(s, quote=False)
    s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"<em>\1</em>", s)
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    s = re.sub(r"\[(.+?)\]\(([^)]+)\)", r'<a href="\2">\1</a>', s)
    s = (s.replace("&lt;br&gt;", "<br>")
          .replace("&lt;br/&gt;", "<br>")
          .replace("&lt;br /&gt;", "<br>"))
    return s


def _table_to_html(md_block: str) -> str:
    lines = [l.rstrip() for l in md_block.strip("\n").split("\n") if l.strip()]
    if len(lines) < 2:
        return md_block
    header = [c.strip() for c in lines[0].strip().strip("|").split("|")]
    body_lines = lines[2:]
    body = [[c.strip() for c in l.strip().strip("|").split("|")] for l in body_lines]
    all_rows = [header] + body
    n_cols = max(len(r) for r in all_rows)
    all_rows = [r + [""] * (n_cols - len(r)) for r in all_rows]

    widths = _optimize_widths(all_rows)
    parts = ["<table>", "<colgroup>"]
    for w in widths:
        parts.append(f'<col style="width: {w}%">')
    parts.append("</colgroup>")
    parts.append("<thead><tr>")
    for cell in header:
        parts.append(f"<th>{_md_inline_to_html(cell)}</th>")
    parts.append("</tr></thead>")
    parts.append("<tbody>")
    for row in body:
        parts.append("<tr>")
        for cell in row + [""] * (n_cols - len(row)):
            parts.append(f"<td>{_md_inline_to_html(cell)}</td>")
        parts.append("</tr>")
    parts.append("</tbody></table>")
    return "\n".join(parts)


def _preprocess_tables(md_text: str) -> str:
    return _TABLE_BLOCK_RE.sub(lambda m: _table_to_html(m.group(1)), md_text)


# ---------------------------------------------------------------------------
# TOC helpers (ported from md_to_pdf.py)
# ---------------------------------------------------------------------------

def _slugify(text: str) -> str:
    slug = text.lower().strip()
    slug = re.sub(r"[^\w\s-]", "", slug)
    slug = re.sub(r"[-\s]+", "-", slug).strip("-")
    return slug


def _add_heading_ids(html: str) -> str:
    seen: dict[str, int] = {}

    def replacer(match):
        tag = match.group(1)
        attrs = match.group(2) or ""
        content = match.group(3)
        if "id=" in attrs:
            return match.group(0)
        text = re.sub(r"<[^>]+>", "", content).strip()
        slug = _slugify(text)
        if slug in seen:
            seen[slug] += 1
            slug = f"{slug}-{seen[slug]}"
        else:
            seen[slug] = 0
        return f'<{tag}{attrs} id="{slug}">{content}</{tag}>'

    return re.sub(r"<(h[234])((?:\s[^>]*)?)>(.*?)</\1>", replacer, html, flags=re.DOTALL)


def _build_toc(html: str, toc_depth: int = 4) -> str:
    heading_re = re.compile(r'<(h[234])\b[^>]*id="([^"]*)"[^>]*>(.*?)</\1>', re.DOTALL)
    entries: list[str] = []
    for match in heading_re.finditer(html):
        level_num = int(match.group(1)[1])
        if level_num > toc_depth:
            continue
        anchor = match.group(2)
        text = re.sub(r"<[^>]+>", "", match.group(3)).strip()
        css_class = f"toc-h{level_num}"
        entries.append(f'<li class="{css_class}"><a href="#{anchor}">{text}</a></li>')
    if not entries:
        return ""
    return (
        '<nav id="TOC">\n<h2>Table of Contents</h2>\n'
        "<ul>\n" + "\n".join(entries) + "\n</ul>\n</nav>\n"
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def render_pdf(markdown_path: str, output_pdf_path: str, *, toc: bool = True) -> bool:
    """Render the markdown report at markdown_path to a PDF at output_pdf_path.

    Pipeline: markdown → HTML (python-markdown + optimized tables) → PDF
    (weasyprint with professional A4 CSS).

    Returns True on success, False on failure. Never raises — a missing
    weasyprint install or any render error logs a warning and returns False
    so the pipeline always has the markdown fallback.

    Image src paths are resolved relative to markdown_path's parent directory
    via weasyprint's base_url — both data: URIs and relative filenames work.

    toc: set False to suppress the auto-generated table of contents (use for
    focused/executive PDFs that are short enough not to need navigation).
    """
    md_path = Path(markdown_path)
    pdf_path = Path(output_pdf_path)

    try:
        from weasyprint import HTML as _WP_HTML
    except ImportError:
        logger.warning("PDF rendering skipped: weasyprint not found")
        return False

    try:
        import markdown as _md_lib
    except ImportError:
        logger.warning("PDF rendering skipped: markdown package not found")
        return False

    try:
        md_text = md_path.read_text(encoding="utf-8")
        md_text = _preprocess_tables(md_text)

        # Skip auto-TOC if caller disabled it or if markdown already has one
        generate_toc = toc and not bool(
            re.search(r"^##\s+Table of Contents\b", md_text, re.MULTILINE)
        )

        html_body = _md_lib.markdown(md_text, extensions=["extra", "toc", "sane_lists"])
        html_body = _add_heading_ids(html_body)
        toc_html = _build_toc(html_body) if generate_toc else ""

        full_html = (
            '<!DOCTYPE html>\n<html lang="en">\n<head>\n'
            '    <meta charset="utf-8">\n'
            f"    <style>{_CSS}</style>\n"
            "</head>\n<body>\n"
            f"{toc_html}\n{html_body}\n"
            "</body>\n</html>"
        )

        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        _WP_HTML(string=full_html, base_url=str(md_path.parent)).write_pdf(str(pdf_path))
        return True

    except Exception as exc:
        logger.warning("PDF rendering failed: %s", exc)
        return False
