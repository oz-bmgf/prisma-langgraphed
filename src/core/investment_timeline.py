"""Investment timeline construction — Phase 2.5 / 2.6.

Phase 2.5 (build_scope_timeline): pure-local, no LLM — builds ScopeTimeline
    objects with date-ordered documents, score history, staleness flags.

Phase 2.6 (build_timeline_narratives_async): async LLM — generates rich
    chronological narratives per investment + a scope-level synthesis.

Narrative persistence (save_narratives_async / load_narratives): saves to
    {ingested_dir}/timeline_narratives.json so subsequent runs reuse cached
    text instead of regenerating.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
from dataclasses import asdict, dataclass, field
from datetime import date as _dt_date
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Date extraction helpers
# ---------------------------------------------------------------------------

_MONTH_MAP: dict[str, str] = {
    "jan": "01", "feb": "02", "mar": "03", "apr": "04",
    "may": "05", "jun": "06", "jul": "07", "aug": "08",
    "sep": "09", "oct": "10", "nov": "11", "dec": "12",
}


def _extract_date_from_filename(filename: str) -> tuple[str, str]:
    """Return (YYYY-MM-DD, confidence). confidence: high | medium | low | none."""
    fn = filename

    m = re.search(r"(20[12]\d)[-_](\d{2})[-_](\d{2})", fn)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}", "high"

    m = re.search(r"(?<!\d)(\d{2})[._](\d{2})[._](20[12]\d)", fn)
    if m:
        return f"{m.group(3)}-{m.group(1)}-{m.group(2)}", "high"

    m = re.search(r"(?<!\d)(20[12]\d)(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])(?!\d)", fn)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}", "high"

    m = re.search(
        r"(\d{1,2})(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)(\d{4})",
        fn, re.IGNORECASE,
    )
    if m:
        day = m.group(1).zfill(2)
        mon = _MONTH_MAP[m.group(2).lower()]
        return f"{m.group(3)}-{mon}-{day}", "high"

    m = re.search(
        r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[\s_-]?(20[12]\d)",
        fn, re.IGNORECASE,
    )
    if m:
        mon = _MONTH_MAP[m.group(1).lower()]
        return f"{m.group(2)}-{mon}-01", "medium"

    m = re.search(r"(?<!\d)(20[12]\d)(?!\d)", fn)
    if m:
        return f"{m.group(1)}-01-01", "low"

    return "", "none"


def _extract_date_from_text(text: str) -> tuple[str, str]:
    m = re.search(r"Date Created[:\s]*(\d{1,2})/(\d{1,2})/(\d{4})", text)
    if m:
        return f"{m.group(3)}-{m.group(1).zfill(2)}-{m.group(2).zfill(2)}", "high"

    m = re.search(
        r"Date[:\s]*(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+(\d{1,2}),?\s+(20\d{2})",
        text, re.IGNORECASE,
    )
    if m:
        mon = _MONTH_MAP[m.group(1)[:3].lower()]
        return f"{m.group(3)}-{mon}-{m.group(2).zfill(2)}", "high"

    return "", "none"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class DocumentEvent:
    """One document in an investment's timeline."""
    file_id: str
    filename: str
    doc_type: str
    date: str                 # YYYY-MM-DD or "" if unknown
    date_confidence: str      # high | medium | low | none
    date_source: str          # filename | text | none
    summary: str = ""
    page_count: int = 0


@dataclass
class InvestmentTimeline:
    """Temporal context for one investment."""
    inv_id: str
    org: str = ""
    title: str = ""
    bow_id: str = ""
    bow_name: str = ""

    # Score trajectory
    impact_current: str = ""
    execution_current: str = ""
    score_history: list[dict] = field(default_factory=list)
    score_trend: str = ""     # improving | stable | deteriorating | insufficient_data

    # Key narrative fields from scoring
    key_results: str = ""
    team_rationale: str = ""

    # Document inventory (date-ordered)
    documents: list[DocumentEvent] = field(default_factory=list)

    # Flags (structural = prompt-safe; rating = opt-in only)
    flags: list[str] = field(default_factory=list)
    rating_flags: list[str] = field(default_factory=list)

    # Coverage stats
    total_docs: int = 0
    doc_types_present: list[str] = field(default_factory=list)
    doc_types_missing: list[str] = field(default_factory=list)
    latest_doc_date: str = ""
    months_since_latest: int = 0

    # LLM-generated (Phase 2.6)
    narrative: str = ""
    key_events: list[dict] = field(default_factory=list)

    def to_summary(self, *, include_team_rating: bool = False) -> str:
        lines = [f"### {self.inv_id} ({self.org})"]
        if self.title:
            lines.append(f"**{self.title}**")
        if self.bow_id:
            lines.append(f"BOW: {self.bow_name} ({self.bow_id})")

        if include_team_rating:
            if self.score_history:
                scores = " → ".join(
                    f"{s.get('year', '?')}: {s.get('impact', '?')}/{s.get('execution', '?')}"
                    for s in self.score_history
                )
                lines.append(f"Score trajectory: {scores} [{self.score_trend}]")
            elif self.impact_current or self.execution_current:
                lines.append(f"Current: impact={self.impact_current}, execution={self.execution_current}")

        if self.key_results:
            lines.append(f"Key results: {self.key_results[:300]}")
        if self.team_rationale:
            lines.append(f"Team rationale: {self.team_rationale[:300]}")

        lines.append(f"Documents: {self.total_docs} files")
        if self.documents:
            lines.append(f"  Types present: {', '.join(self.doc_types_present)}")
            if self.doc_types_missing:
                lines.append(f"  Types missing: {', '.join(self.doc_types_missing)}")
            lines.append(
                f"  Latest: {self.latest_doc_date or 'unknown'} ({self.months_since_latest}mo ago)"
            )
            for doc in self.documents[:5]:
                date_str = doc.date or "undated"
                lines.append(f"  - [{date_str}] {doc.doc_type}: {doc.filename[:60]}")
                if doc.summary:
                    lines.append(f"    {doc.summary[:150]}...")
            if len(self.documents) > 5:
                lines.append(f"  ... and {len(self.documents) - 5} more")

        if self.flags:
            lines.append(f"FLAGS: {', '.join(self.flags)}")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ScopeTimeline:
    """Aggregated timeline for a scope (one or more investments × one or more BOWs)."""
    scope_id: str
    label: str
    bow_ids: list[str]
    investments: list[InvestmentTimeline]
    scope_flags: list[str] = field(default_factory=list)
    narrative: str = ""

    def to_context(self) -> str:
        """Format all investment timelines as shared LLM context."""
        parts = [
            f"## Scope: {self.label} ({', '.join(self.bow_ids)})",
            f"Investments: {len(self.investments)}",
        ]
        if self.scope_flags:
            parts.append(f"Scope-level flags: {', '.join(self.scope_flags)}")
        parts.append("")
        for inv in self.investments:
            parts.append(inv.to_summary())
            parts.append("")
        return "\n".join(parts)

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Expected document types
# ---------------------------------------------------------------------------

_EXPECTED_DOC_TYPES = ["proposal", "investment_document", "budget", "progress_report"]


# ---------------------------------------------------------------------------
# Phase 2.5 — pure-local timeline construction
# ---------------------------------------------------------------------------

def _compute_score_trend(score_history: list[dict]) -> str:
    if len(score_history) < 2:
        return "insufficient_data"
    color_val = {"green": 3, "yellow": 2, "red": 1, "blue": 0, "": 0}
    exec_scores = [color_val.get(s.get("execution", ""), 0) for s in score_history]
    exec_scores = [s for s in exec_scores if s > 0]
    if len(exec_scores) < 2:
        return "insufficient_data"
    delta = exec_scores[-1] - exec_scores[0]
    if delta > 0:
        return "improving"
    elif delta < 0:
        return "deteriorating"
    return "stable"


def build_investment_timeline(
    inv_id: str,
    doc_entries: list[dict],
    scoring_data: dict | None = None,
    pages_dir: Path | None = None,
    intelligence: dict | None = None,
) -> InvestmentTimeline:
    """Build timeline for one investment from catalog + scoring + intelligence.
    Pure-local — no LLM calls.
    """
    timeline = InvestmentTimeline(inv_id=inv_id)

    if scoring_data:
        timeline.org = scoring_data.get("org", "")
        timeline.title = scoring_data.get("title", "")
        timeline.bow_id = scoring_data.get("bow_id", "")
        timeline.bow_name = scoring_data.get("bow_name", "")
        timeline.impact_current = scoring_data.get("impact", "")
        timeline.execution_current = scoring_data.get("execution", "")
        timeline.key_results = scoring_data.get("key_results", "")
        timeline.team_rationale = scoring_data.get("team_rationale", "")

        sh = scoring_data.get("score_history", [])
        if isinstance(sh, list):
            timeline.score_history = [
                {
                    "year": s.get("year", ""),
                    "impact": s.get("impact", ""),
                    "execution": s.get("execution", ""),
                    "reinvestment": s.get("reinvestment", ""),
                }
                for s in sh
            ]
        timeline.score_trend = _compute_score_trend(timeline.score_history)

    # Build document events
    doc_events: list[DocumentEvent] = []
    for d in doc_entries:
        fid = d.get("file_id", d.get("paper_id", ""))
        filename = d.get("filename", "")
        doc_type = d.get("doc_type", "other")

        date, confidence = _extract_date_from_filename(filename)
        date_source = "filename" if date else "none"

        if not date and pages_dir:
            idx_path = pages_dir / fid / "index.json"
            if idx_path.exists():
                try:
                    idx = json.loads(idx_path.read_text())
                    text_date, text_conf = _extract_date_from_text(idx.get("summary", ""))
                    if text_date:
                        date, confidence, date_source = text_date, text_conf, "text"
                except (json.JSONDecodeError, OSError):
                    pass

        summary = d.get("summary", "")
        if not summary and pages_dir:
            idx_path = pages_dir / fid / "index.json"
            if idx_path.exists():
                try:
                    summary = json.loads(idx_path.read_text()).get("summary", "")
                except (json.JSONDecodeError, OSError):
                    pass

        doc_events.append(DocumentEvent(
            file_id=fid,
            filename=filename,
            doc_type=doc_type,
            date=date,
            date_confidence=confidence,
            date_source=date_source,
            summary=summary[:500] if summary else "",
            page_count=d.get("total_pages", 0),
        ))

    doc_events.sort(key=lambda e: e.date or "9999")
    timeline.documents = doc_events
    timeline.total_docs = len(doc_events)

    present_types = set(e.doc_type for e in doc_events)
    timeline.doc_types_present = sorted(present_types)
    timeline.doc_types_missing = [t for t in _EXPECTED_DOC_TYPES if t not in present_types]

    dated_docs = [e for e in doc_events if e.date and e.date_confidence != "low"]
    if dated_docs:
        timeline.latest_doc_date = dated_docs[-1].date

    # Structural flags (prompt-safe)
    flags: list[str] = []
    if timeline.total_docs <= 2:
        flags.append("SPARSE")
    if "progress_report" not in present_types and "investment_document" in present_types:
        flags.append("NO_PROGRESS_REPORT")
    if timeline.key_results and len(timeline.key_results.strip()) < 50:
        flags.append("THIN_KEY_RESULTS")
    if timeline.team_rationale and len(timeline.team_rationale.strip()) < 50:
        flags.append("THIN_RATIONALE")
    if "termination" in present_types:
        flags.append("TERMINATED")
    timeline.flags = flags

    # Rating flags (opt-in — don't include in standard LLM prompts)
    rating_flags: list[str] = []
    if timeline.score_trend == "deteriorating":
        rating_flags.append("SCORE_DETERIORATING")
    if timeline.execution_current == "red":
        rating_flags.append("RED_EXECUTION")
    if timeline.impact_current == "red":
        rating_flags.append("RED_IMPACT")
    timeline.rating_flags = rating_flags

    # Enrich from intelligence layer
    if intelligence:
        if intelligence.get("timeline_narrative"):
            timeline.narrative = intelligence["timeline_narrative"]
        for sig in ("missing_information", "risk_signals", "strength_signals"):
            val = intelligence.get(sig, "")
            if val:
                label = sig.replace("_", " ").upper()
                timeline.narrative += f"\n\n{label}: {val}"
        if intelligence.get("summary") and not timeline.key_results:
            timeline.key_results = intelligence["summary"]
        kd = intelligence.get("key_dates") or {}
        intel_latest = (kd.get("latest_report") or "").strip()
        if intel_latest and intel_latest > (timeline.latest_doc_date or ""):
            timeline.latest_doc_date = intel_latest
        if intelligence.get("stage"):
            timeline.flags.append(f"STAGE:{intelligence['stage'].upper()}")

    # Compute months_since_latest after intelligence merge
    if timeline.latest_doc_date:
        try:
            today = _dt_date.today()
            parts = timeline.latest_doc_date.split("-")
            y = int(parts[0])
            mo = int(parts[1]) if len(parts) >= 2 else 1
            d = int(parts[2]) if len(parts) >= 3 else 1
            if 1 <= mo <= 12 and 1 <= d <= 31:
                latest = _dt_date(y, mo, d)
                timeline.months_since_latest = (
                    (today.year - latest.year) * 12 + (today.month - latest.month)
                )
        except (ValueError, TypeError, IndexError):
            pass

    if timeline.months_since_latest > 12 and "STALE" not in timeline.flags:
        timeline.flags.append("STALE")

    return timeline


def build_scope_timeline(
    scope: dict,
    doc_list: list[dict],
    scoring: dict,
    pages_dir: Path | None = None,
    investment_intelligence: dict | None = None,
) -> ScopeTimeline:
    """Build timelines for all investments in a scope. Pure-local — no LLM calls.

    Supports both ``inv_id`` (singular, new repo) and ``inv_ids`` (plural, old repo)
    scope formats.
    """
    inv_ids: list[str] = scope.get("inv_ids") or (
        [scope["inv_id"]] if scope.get("inv_id") else []
    )

    # Group docs by investment
    inv_docs: dict[str, list[dict]] = {}
    for d in doc_list:
        iid = d.get("inv_id", "")
        if iid in inv_ids:
            inv_docs.setdefault(iid, []).append(d)

    # Build per-investment timelines
    timelines: list[InvestmentTimeline] = []
    for inv_id in inv_ids:
        sd = scoring.get(inv_id, {})
        if hasattr(sd, "__dataclass_fields__"):
            sd = asdict(sd)
        inv_intel = (investment_intelligence or {}).get(inv_id)
        tl = build_investment_timeline(
            inv_id=inv_id,
            doc_entries=inv_docs.get(inv_id, []),
            scoring_data=sd,
            pages_dir=pages_dir,
            intelligence=inv_intel,
        )
        timelines.append(tl)

    # Sort: most flagged first, then by recency
    timelines.sort(
        key=lambda t: (-(len(t.flags) + len(t.rating_flags)), t.latest_doc_date or "0000"),
    )

    # Scope-level aggregate flags
    scope_flags: list[str] = []
    red = sum(1 for t in timelines if "RED_EXECUTION" in t.rating_flags or "RED_IMPACT" in t.rating_flags)
    if red:
        scope_flags.append(f"{red}_RED_INVESTMENTS")
    stale = sum(1 for t in timelines if "STALE" in t.flags)
    if stale:
        scope_flags.append(f"{stale}_STALE_INVESTMENTS")
    terminated = sum(1 for t in timelines if "TERMINATED" in t.flags)
    if terminated:
        scope_flags.append(f"{terminated}_TERMINATED")
    deteriorating = sum(1 for t in timelines if "SCORE_DETERIORATING" in t.rating_flags)
    if deteriorating:
        scope_flags.append(f"{deteriorating}_DETERIORATING_SCORES")

    return ScopeTimeline(
        scope_id=scope.get("scope_id", ""),
        label=scope.get("label", ""),
        bow_ids=scope.get("bow_ids", []),
        investments=timelines,
        scope_flags=scope_flags,
    )


# ---------------------------------------------------------------------------
# Phase 2.6 — async LLM narrative generation
# ---------------------------------------------------------------------------

_INVESTMENT_NARRATIVE_SYSTEM = """\
You are a program analyst constructing a factual chronological narrative \
from document summaries for a single grant/investment.

Write a detailed chronological account (500-1000 words) covering:
1. Origin and purpose: When the grant started, what it aimed to do, grantee identity
2. Key milestones and events: Progress, setbacks, partner changes, amendments, pivots
3. Current status: Where things stand as of the most recent document
4. Document gaps: What periods or topics have no documentation

Rules: Be specific with dates, amounts, names. Distinguish what documents SAY vs INFER.
Note contradictions explicitly. Write in past tense for completed events, \
present tense for current status. Do NOT wrap in JSON."""

_SCOPE_SYNTHESIS_SYSTEM = """\
You are a program analyst writing a scope-level synthesis (200-400 words) connecting \
multiple investments. Explain how they relate, shared dependencies, timeline overlaps, \
and the overall trajectory of this body of work. Do NOT wrap in JSON."""


async def _build_single_investment_narrative(
    inv: InvestmentTimeline,
    scope_id: str,
    scope_label: str,
    model: str,
) -> None:
    """Generate narrative for one investment. Mutates inv.narrative in place."""
    from src.core.llm_utils import acall_llm  # lazy import to avoid circular

    lines = [f"# Investment: {inv.inv_id} — {inv.org}: {inv.title}"]
    lines.append(f"Scope: {scope_label}")
    if inv.bow_id:
        lines.append(f"BOW: {inv.bow_name} ({inv.bow_id})")
    if inv.key_results:
        lines.append(f"Team's key results: {inv.key_results}")
    if inv.team_rationale:
        lines.append(f"Team's rationale: {inv.team_rationale}")

    lines.append(f"\nDocuments ({inv.total_docs} files, chronologically):")
    for doc in inv.documents:
        date_str = doc.date or "undated"
        conf = (
            f" ({doc.date_confidence} confidence)"
            if doc.date and doc.date_confidence != "high" else ""
        )
        lines.append(f"\n### [{date_str}{conf}] {doc.doc_type}: {doc.filename}")
        lines.append(f"Pages: {doc.page_count}")
        lines.append(doc.summary if doc.summary else "(no summary available)")

    if inv.flags:
        lines.append(f"\nFlags: {', '.join(inv.flags)}")

    try:
        raw = await acall_llm(
            "\n".join(lines),
            system_msg=_INVESTMENT_NARRATIVE_SYSTEM,
            model=model,
        )
        narrative = str(raw).strip() if raw else ""
        if len(narrative) > 100:
            inv.narrative = narrative
    except Exception as exc:
        logger.warning("[%s] Narrative generation failed: %s", inv.inv_id, str(exc)[:120])


async def build_timeline_narratives_async(
    scope_timeline: ScopeTimeline,
    model: str,
) -> None:
    """Build LLM-generated narratives for a scope's investments. Mutates in place.

    One async LLM call per investment (parallel via asyncio.gather), then one
    scope-level synthesis call.
    """
    from src.core.llm_utils import acall_llm  # lazy import

    if not scope_timeline.investments:
        return

    # asyncio-APPROVED-2: concurrent LLM — one narrative per investment, inside core library called
    # via asyncio.to_thread; not convertible to Send() without restructuring the library API.
    results = await asyncio.gather(
        *[
            _build_single_investment_narrative(
                inv, scope_timeline.scope_id, scope_timeline.label, model
            )
            for inv in scope_timeline.investments
        ],
        return_exceptions=True,
    )
    for i, r in enumerate(results):
        if isinstance(r, Exception):
            logger.warning(
                "[%s] Narrative task %d raised: %s",
                scope_timeline.scope_id, i, r,
            )

    narrated = [inv for inv in scope_timeline.investments if inv.narrative]
    if not narrated:
        logger.warning("[%s] No investment narratives generated; skipping scope synthesis",
                       scope_timeline.scope_id)
        return

    try:
        summaries = "\n\n---\n\n".join(
            f"**{inv.inv_id}:**\n{inv.narrative[:2000]}"
            for inv in narrated
        )
        scope_raw = await acall_llm(
            (
                f"# Scope: {scope_timeline.label}\n"
                f"# {len(narrated)} investments\n\n"
                f"Per-investment narratives:\n\n{summaries}\n\n"
                "Write the scope-level synthesis."
            ),
            system_msg=_SCOPE_SYNTHESIS_SYSTEM,
            model=model,
        )
        if len(str(scope_raw)) > 50:
            scope_timeline.narrative = str(scope_raw)
    except Exception as exc:
        logger.warning(
            "[%s] Scope synthesis failed: %s", scope_timeline.scope_id, str(exc)[:120]
        )

    logger.info(
        "[%s] Timeline narratives: %d/%d investments, scope narrative %d chars",
        scope_timeline.scope_id,
        len(narrated),
        len(scope_timeline.investments),
        len(scope_timeline.narrative),
    )


# ---------------------------------------------------------------------------
# Narrative persistence
# ---------------------------------------------------------------------------

def _compute_input_hash(scope_timelines: dict[str, Any]) -> str:
    """Hash document filenames + summaries to detect collection changes."""
    h = hashlib.sha256()
    for sid in sorted(scope_timelines):
        st = scope_timelines[sid]
        invs = st.investments if isinstance(st, ScopeTimeline) else st.get("investments", [])
        for inv in invs:
            inv_id = inv.inv_id if isinstance(inv, InvestmentTimeline) else inv.get("inv_id", "")
            docs = inv.documents if isinstance(inv, InvestmentTimeline) else inv.get("documents", [])
            h.update(inv_id.encode())
            for doc in docs:
                fn = doc.filename if isinstance(doc, DocumentEvent) else doc.get("filename", "")
                sm = doc.summary if isinstance(doc, DocumentEvent) else doc.get("summary", "")
                h.update((fn or "").encode())
                h.update((sm or "").encode())
    return h.hexdigest()[:16]


async def save_narratives_async(
    scope_timelines: dict[str, ScopeTimeline],
    path: Path,
) -> None:
    """Persist narrative cache to disk. Skips if all narratives are empty."""
    data: dict[str, Any] = {"_input_hash": _compute_input_hash(scope_timelines)}
    for sid, st in scope_timelines.items():
        inv_data: dict[str, dict] = {}
        for inv in st.investments:
            if inv.narrative:
                inv_data[inv.inv_id] = {"narrative": inv.narrative, "key_events": inv.key_events}
        data[sid] = {
            "scope_id": st.scope_id,
            "label": st.label,
            "scope_narrative": st.narrative,
            "investments": inv_data,
        }

    total_inv = sum(len(s["investments"]) for s in data.values() if isinstance(s, dict))
    if total_inv == 0:
        logger.warning("NOT saving narrative cache — all narratives empty (generation likely failed)")
        return

    payload = json.dumps(data, indent=2, ensure_ascii=False)
    # asyncio-APPROVED-1: to_thread wraps blocking Path.write_text
    await asyncio.to_thread(path.write_text, payload, "utf-8")
    logger.info("Saved timeline narratives: %d scopes, %d investments → %s",
                len(data) - 1, total_inv, path)


def load_narratives(
    scope_timelines: dict[str, ScopeTimeline],
    path: Path,
) -> bool:
    """Load cached narratives into scope_timelines in place. Returns True if loaded."""
    if not path.exists():
        return False
    try:
        data: dict = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to load narrative cache %s: %s", path, exc)
        return False

    cached_hash = data.get("_input_hash")
    if not cached_hash:
        logger.info("Narrative cache has no input hash (legacy); regenerating")
        return False

    current_hash = _compute_input_hash(scope_timelines)
    if cached_hash != current_hash:
        logger.info("Narrative cache stale (%s → %s); regenerating", cached_hash, current_hash)
        return False

    loaded = 0
    for sid, st in scope_timelines.items():
        scope_data = data.get(sid, {})
        if not isinstance(scope_data, dict):
            continue
        st.narrative = scope_data.get("scope_narrative", "")
        inv_data = scope_data.get("investments", {})
        for inv in st.investments:
            cached_inv = inv_data.get(inv.inv_id, {})
            if cached_inv:
                inv.narrative = cached_inv.get("narrative", "")
                inv.key_events = cached_inv.get("key_events", [])
                loaded += 1

    if loaded > 0:
        logger.info("Loaded timeline narratives from %s: %d investments", path, loaded)
        return True
    return False
