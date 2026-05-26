"""Chart generation for portfolio review reports.

All functions are pure: no file I/O, no plt.show(), no global state.
Each returns a base64-encoded PNG string or None when data is insufficient.

Ported from prisma-ai-review/src/qpr/score_comparison_figures.py, adapted
to the NEW repo's severity-based data model (investment_report.severity /
investment_facts.risk_severity) instead of the OLD repo's 7-level scoring
ladder.

Rendering pipeline: matplotlib → PNG bytes → base64 string.
Caller embeds the result in markdown as:
    ![alt](data:image/png;base64,<string>)
"""
from __future__ import annotations

import base64
import io
import logging
from typing import Any

logger = logging.getLogger(__name__)

# Must be called once before any matplotlib.pyplot import.
# Python 3.14 raises AttributeError on matplotlib.backends if use() is called
# after pyplot has already been imported in the same process.
try:
    import matplotlib
    matplotlib.use("Agg")
except Exception:
    pass

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

_SEVERITY_ORDER = [
    "program_critical",
    "pathway_altering",
    "efficiency_reducing",
    "aligned",
    "",
]
_SEVERITY_SHORT = {
    "program_critical": "Prog.\nCritical",
    "pathway_altering": "Pathway\nAltering",
    "efficiency_reducing": "Effic.\nReducing",
    "aligned": "Aligned",
    "": "Unknown",
}
_SEVERITY_HEX = {
    "program_critical": "#cf222e",
    "pathway_altering": "#fb8500",
    "efficiency_reducing": "#1a7f37",
    "aligned": "#656d76",
    "": "#bbbbbb",
}


def _norm_severity(raw: str | None) -> str:
    v = (raw or "").lower().replace("-", "_")
    return v if v in _SEVERITY_HEX else ""


def _scope_ai_severity(scope: dict) -> str:
    return _norm_severity((scope.get("investment_report") or {}).get("severity"))


def _scope_team_severity(scope: dict) -> str:
    return _norm_severity((scope.get("investment_facts") or {}).get("risk_severity"))


def _png_bytes_to_b64(buf: io.BytesIO) -> str:
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ---------------------------------------------------------------------------
# render_confusion_matrix
# ---------------------------------------------------------------------------


def render_confusion_matrix(
    scope_outputs: list[dict],
    investment_scoring: dict,
) -> str | None:
    """Team-vs-AI risk severity confusion matrix as inline base64 PNG.

    Rows = team risk_severity; columns = AI-assessed severity.
    Diagonal cells (green border) = agreement.
    Color intensity ∝ investment count per cell.

    ``investment_scoring`` is accepted for signature compatibility with
    render_scatter_plot but is not used here — severity comes from the
    investment_report / investment_facts keys already in scope_outputs.

    Returns base64 PNG string or None when matplotlib is unavailable or
    there are fewer than 2 data points.
    """
    if not scope_outputs:
        logger.warning("render_confusion_matrix: empty scope_outputs")
        return None

    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap
    except Exception as exc:
        logger.warning("render_confusion_matrix: matplotlib unavailable: %s", exc)
        return None

    LABELS = _SEVERITY_ORDER
    n = len(LABELS)
    matrix = [[0] * n for _ in range(n)]
    for s in scope_outputs:
        ai_idx = LABELS.index(_scope_ai_severity(s))
        tm_idx = LABELS.index(_scope_team_severity(s))
        matrix[tm_idx][ai_idx] += 1

    total = sum(matrix[i][j] for i in range(n) for j in range(n))
    if total < 2:
        logger.warning("render_confusion_matrix: insufficient data (total=%d)", total)
        return None

    import numpy as np
    m = np.array(matrix, dtype=int)

    cmap = LinearSegmentedColormap.from_list("blues", ["#f7fbff", "#08306b"])
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(m, cmap=cmap, vmin=0, vmax=max(m.max(), 1), origin="upper", aspect="equal")
    ax.set_box_aspect(1)

    short_labels = [_SEVERITY_SHORT.get(l, l) for l in LABELS]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(short_labels, fontsize=9)
    ax.set_yticklabels(short_labels, fontsize=9)
    ax.set_xlabel("AI severity", fontsize=10, fontweight="bold")
    ax.set_ylabel("Team severity", fontsize=10, fontweight="bold")
    ax.set_title(f"Team vs AI Risk Severity\n(n={total} investments)", fontsize=12, fontweight="bold", pad=8)

    # Diagonal (agreement) cells — green border
    for i in range(n):
        ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False,
                                    edgecolor="#2ca02c", linewidth=2, alpha=0.7))

    threshold = m.max() / 2 if m.max() else 0
    for i in range(n):
        for j in range(n):
            v = int(m[i, j])
            if v == 0:
                continue
            color = "white" if v > threshold else "#222"
            ax.text(j, i, str(v), ha="center", va="center", color=color,
                    fontsize=11, fontweight="bold")

    fig.subplots_adjust(left=0.18, right=0.97, top=0.90, bottom=0.12)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return _png_bytes_to_b64(buf)


# ---------------------------------------------------------------------------
# render_scatter_plot
# ---------------------------------------------------------------------------


def render_scatter_plot(
    bow_id: str,
    scope_outputs: list[dict],
    investment_scoring: dict,
    x_axis: str = "execution_rate",
    y_axis: str = "approved_amount",
) -> str | None:
    """Per-BOW X-Y scatter of investment-level metrics as inline base64 PNG.

    Points are colored by AI-assessed severity. Labels show inv_id.

    ``investment_scoring`` is consulted for approved_amount when
    investment_facts is missing or incomplete.

    Returns base64 PNG string or None when fewer than 2 data points exist.
    """
    if not scope_outputs:
        logger.warning("render_scatter_plot(%s): empty scope_outputs", bow_id)
        return None

    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
    except Exception as exc:
        logger.warning("render_scatter_plot: matplotlib unavailable: %s", exc)
        return None

    points: list[dict[str, Any]] = []
    for s in scope_outputs:
        facts = s.get("investment_facts") or {}
        inv_ids = s.get("inv_ids") or [s.get("inv_id", "")]
        inv_id = inv_ids[0] if inv_ids else ""

        # x value
        raw_x = facts.get(x_axis)
        if raw_x is None and investment_scoring:
            inv_detail = investment_scoring.get(inv_id) or {}
            raw_x = (inv_detail.get(x_axis) if isinstance(inv_detail, dict)
                     else getattr(inv_detail, x_axis, None))

        # y value
        raw_y = facts.get(y_axis)
        if raw_y is None and investment_scoring:
            inv_detail = investment_scoring.get(inv_id) or {}
            raw_y = (inv_detail.get(y_axis) if isinstance(inv_detail, dict)
                     else getattr(inv_detail, y_axis, None))

        try:
            x_val = float(raw_x)
            y_val = float(raw_y)
        except (TypeError, ValueError):
            continue

        sev = _scope_ai_severity(s)
        short_id = ", ".join(inv_ids)[:18]
        points.append({
            "x": x_val,
            "y": y_val,
            "label": short_id,
            "color": _SEVERITY_HEX.get(sev, "#bbbbbb"),
            "sev": sev,
        })

    if len(points) < 2:
        logger.warning("render_scatter_plot(%s): insufficient points (%d)", bow_id, len(points))
        return None

    # Scale y if it looks like raw dollars (>1000 → display in $M)
    y_divisor = 1.0
    y_label = y_axis.replace("_", " ").title()
    if y_axis in ("approved_amount", "paid_amount") or (points and points[0]["y"] > 1000):
        y_divisor = 1e6
        y_label += " ($M)"

    x_label = x_axis.replace("_", " ").title()
    if x_axis == "execution_rate":
        x_label = "Execution Rate"

    fig, ax = plt.subplots(figsize=(7, 5), dpi=120)

    seen_sev: set[str] = set()
    for p in points:
        ax.scatter(p["x"], p["y"] / y_divisor,
                   c=p["color"], s=60, alpha=0.8, edgecolors="white", linewidths=0.5,
                   zorder=3)
        ax.annotate(f"  {p['label']}", (p["x"], p["y"] / y_divisor),
                    fontsize=7, color="#333", ha="left", va="center")
        seen_sev.add(p["sev"])

    ax.set_xlabel(x_label, fontsize=10)
    ax.set_ylabel(y_label, fontsize=10)
    ax.set_title(f"{bow_id} — {x_label} vs {y_label}", fontsize=11, fontweight="bold")
    ax.grid(linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    legend_handles = [
        Patch(color=_SEVERITY_HEX[s], label=(s or "unknown").replace("_", " ").title())
        for s in _SEVERITY_ORDER if s in seen_sev
    ]
    if legend_handles:
        ax.legend(handles=legend_handles, fontsize=8, frameon=False, loc="best")

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return _png_bytes_to_b64(buf)
