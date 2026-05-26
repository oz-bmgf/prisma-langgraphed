"""gs_verifier graph — dual LLM verifier for gold standard findings (§13).

Topology:
  START → load_gold → dispatch_findings [Send() per finding]
        → verify_finding (fan-out worker)
        → collect_verdicts → reconcile_output
        → build_tiered_gold → apply_verdicts → END

verify_finding runs verifier_a and verifier_b in parallel via asyncio.gather.
Reconciliation is deterministic: exact agreement → locked; coarse agreement →
locked on Opus label; genuine disagree → needs_review.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from pathlib import Path
from typing import Optional

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from src.config import DEFAULT_STRONG_MODEL, DEFAULT_SYNTHESIS_MODEL, VERIFIER_MAX_TOKENS
from src.core.llm_utils import acall_llm
from src.graph.state import FindingVerificationState, GsVerifierState
from src.prompts.analyze_prompts import (
    VERDICT_SCHEMA_DESCRIPTION,
    VERIFIER_SYSTEM,
    VERIFIER_TASK_TEMPLATE,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Status ordering and coarse-category maps for reconciliation
# ---------------------------------------------------------------------------

_STATUS_STRICTNESS: dict[str, int] = {
    "retain": 0,
    "modify": 1,
    "reclassify": 1,
    "demote": 2,
    "reject": 3,
}

_COARSE: dict[str, str] = {
    "retain": "keep",
    "modify": "revise",
    "demote": "revise",
    "reclassify": "revise",
    "reject": "drop",
}

_VALID_STATUSES = frozenset(_STATUS_STRICTNESS.keys())


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def _extract_json(text: str) -> dict:
    """Extract the first JSON object from LLM prose (with or without fences)."""
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        return json.loads(fence.group(1))
    brace = re.search(r"\{.*\}", text, re.DOTALL)
    if brace:
        return json.loads(brace.group(0))
    raise ValueError("No JSON object found in LLM response")


def _reconcile(a: dict, b: dict) -> dict:
    """Deterministic reconciliation of two verdicts.

    Returns a reconciliation dict:
      {agreement, locked_status, priority, reconcile_note}
    """
    status_a = (a.get("overall_status") or "").lower()
    status_b = (b.get("overall_status") or "").lower()

    # Incomplete / error verdicts → flag for human review
    if status_a not in _VALID_STATUSES or status_b not in _VALID_STATUSES:
        return {
            "agreement": "incomplete",
            "locked_status": None,
            "priority": "critical",
            "reconcile_note": (
                f"One or both verdicts malformed: a={status_a!r} b={status_b!r}"
            ),
        }

    coarse_a = _COARSE[status_a]
    coarse_b = _COARSE[status_b]

    # Exact agreement
    if status_a == status_b:
        return {
            "agreement": "agree",
            "locked_status": status_a,
            "priority": "low",
            "reconcile_note": f"Both verifiers agree: {status_a}",
        }

    # Coarse agreement → lock on the stricter (Opus = verifier_a) label
    if coarse_a == coarse_b:
        strict_a = _STATUS_STRICTNESS[status_a]
        strict_b = _STATUS_STRICTNESS[status_b]
        chosen = status_a if strict_a >= strict_b else status_b
        return {
            "agreement": "coarse_agree",
            "locked_status": chosen,
            "priority": "medium",
            "reconcile_note": (
                f"Coarse agreement ({coarse_a}); locked on stricter label: {chosen}"
            ),
        }

    # keep vs drop — genuine disagreement, high priority
    if {coarse_a, coarse_b} == {"keep", "drop"}:
        return {
            "agreement": "disagree",
            "locked_status": None,
            "priority": "high",
            "reconcile_note": (
                f"Fundamental disagreement: a={status_a} ({coarse_a}), "
                f"b={status_b} ({coarse_b}) — needs human review"
            ),
        }

    # Other coarse disagreements (keep vs revise, revise vs drop)
    strict_a = _STATUS_STRICTNESS[status_a]
    strict_b = _STATUS_STRICTNESS[status_b]
    stricter = status_a if strict_a >= strict_b else status_b
    return {
        "agreement": "disagree",
        "locked_status": stricter,
        "priority": "medium",
        "reconcile_note": (
            f"Coarse disagreement: a={status_a}, b={status_b}; "
            f"defaulting to stricter label: {stricter}"
        ),
    }


def _build_evidence_bundle(finding: dict, doc_list: list[dict]) -> str:
    """Build a plain-text evidence bundle from a finding's cited evidence."""
    doc_map = {d.get("file_id", ""): d for d in (doc_list or [])}
    lines: list[str] = []

    for polarity in ("evidence_for", "evidence_against"):
        items = finding.get(polarity) or []
        if not items:
            continue
        label = "SUPPORTING EVIDENCE" if polarity == "evidence_for" else "COUNTER EVIDENCE"
        lines.append(f"### {label}")
        for ev in items:
            fid = ev.get("file_id", "")
            doc = doc_map.get(fid, {})
            fname = doc.get("filename", fid)
            quote = ev.get("quote") or ev.get("excerpt") or ""
            lines.append(f"- [{fname}] {quote[:400]}")

    return "\n".join(lines) or "(no evidence attached)"


def _classify_finding_type(finding: dict) -> str:
    return finding.get("finding_type") or finding.get("type") or "observation"


# ---------------------------------------------------------------------------
# Node: load_gold
# ---------------------------------------------------------------------------

async def load_gold(state: GsVerifierState) -> dict:
    gold_path = Path(state["gold_path"])
    if not gold_path.exists():
        return {"errors": [f"gold_path not found: {gold_path}"]}

    def _read(p: Path) -> dict:
        with open(p, encoding="utf-8") as fh:
            return json.load(fh)

    # asyncio-APPROVED-1: to_thread wraps blocking JSON file read
    gold_data = await asyncio.to_thread(_read, gold_path)

    # Locate doc_list and investment_scoring relative to gold_path
    ingested_dir = gold_path.parent.parent / f"{state['program']}-ingested"
    doc_list: list[dict] = []
    investment_scoring: dict = {}

    dl_path = ingested_dir / "doc_list.json"
    is_path = ingested_dir / "investment_scoring.json"

    if dl_path.exists():
        # asyncio-APPROVED-1: to_thread wraps blocking JSON file read
        doc_list = await asyncio.to_thread(_read, dl_path)  # type: ignore[assignment]
    if is_path.exists():
        # asyncio-APPROVED-1: to_thread wraps blocking JSON file read
        investment_scoring = await asyncio.to_thread(_read, is_path)

    return {
        "gold_data": gold_data,
        "doc_list": doc_list,
        "investment_scoring": investment_scoring,
    }


# ---------------------------------------------------------------------------
# Node: dispatch_findings  (returns Send() per finding)
# ---------------------------------------------------------------------------

def dispatch_findings(state: GsVerifierState) -> list[Send]:
    gold_data = state.get("gold_data") or {}
    doc_list = state.get("doc_list") or []
    findings: list[dict] = gold_data.get("findings") or []
    sends: list[Send] = []

    for finding in findings:
        scope_label = (
            finding.get("scope_label")
            or finding.get("scope")
            or finding.get("thread_title")
            or "Unknown scope"
        )
        finding_type = _classify_finding_type(finding)
        evidence_bundle = _build_evidence_bundle(finding, doc_list)

        payload = FindingVerificationState(
            finding=finding,
            scope_label=scope_label,
            finding_type=finding_type,
            evidence_bundle=evidence_bundle,
            program=state["program"],
            as_of_date=state["as_of_date"],
            verifier_a_model=state["verifier_a_model"],
            verifier_b_model=state["verifier_b_model"],
            result=None,
        )
        sends.append(Send("verify_finding", payload))

    return sends


# ---------------------------------------------------------------------------
# Node: verify_finding  (fan-out worker)
# ---------------------------------------------------------------------------

async def verify_finding(state: FindingVerificationState) -> dict:
    finding = state["finding"]
    finding_text = finding.get("finding") or finding.get("text") or json.dumps(finding)

    task_prompt = VERIFIER_TASK_TEMPLATE.format(
        scope_label=state["scope_label"],
        finding_type=state["finding_type"],
        program=state["program"],
        as_of_date=state["as_of_date"],
        finding_text=finding_text,
        evidence_bundle=state["evidence_bundle"],
        schema=VERDICT_SCHEMA_DESCRIPTION,
    )

    # asyncio-APPROVED-2: concurrent LLM — fixed arity-2 (verifier_a and verifier_b), not a
    # variable fan-out over N items. Send() would require 2 worker nodes + a reconcile node
    # for a pattern that is always exactly 2 calls; gather is the right primitive here.
    verdict_a_raw, verdict_b_raw = await asyncio.gather(
        acall_llm(task_prompt, VERIFIER_SYSTEM, model=state["verifier_a_model"], max_tokens=VERIFIER_MAX_TOKENS),
        acall_llm(task_prompt, VERIFIER_SYSTEM, model=state["verifier_b_model"], max_tokens=VERIFIER_MAX_TOKENS),
    )

    verdict_a: dict = {}
    verdict_b: dict = {}
    errors: list[str] = []

    try:
        verdict_a = _extract_json(verdict_a_raw)
    except Exception as exc:
        errors.append(f"verifier_a parse error for finding {finding.get('id', '?')}: {exc}")
        verdict_a = {"overall_status": "error", "rationale": str(exc)}

    try:
        verdict_b = _extract_json(verdict_b_raw)
    except Exception as exc:
        errors.append(f"verifier_b parse error for finding {finding.get('id', '?')}: {exc}")
        verdict_b = {"overall_status": "error", "rationale": str(exc)}

    reconciliation = _reconcile(verdict_a, verdict_b)

    result = {
        "finding_id": finding.get("id") or finding.get("finding_id") or "",
        "scope_label": state["scope_label"],
        "finding_type": state["finding_type"],
        "verdict_a": verdict_a,
        "verdict_b": verdict_b,
        "reconciliation": reconciliation,
    }

    return {"verdicts": [result], "errors": errors}


# ---------------------------------------------------------------------------
# Node: collect_verdicts  (aggregation — verdicts already accumulated by reducer)
# ---------------------------------------------------------------------------

async def collect_verdicts(state: GsVerifierState) -> dict:
    verdicts = state.get("verdicts") or []
    logger.info("collect_verdicts: %d verdicts received", len(verdicts))
    return {}


# ---------------------------------------------------------------------------
# Node: reconcile_output  (writes gold_v3_reverified.json)
# ---------------------------------------------------------------------------

async def reconcile_output(state: GsVerifierState) -> dict:
    verdicts = state.get("verdicts") or []
    gold_data = state.get("gold_data") or {}
    out_path = Path(state["out_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Build verdict lookup by finding_id
    verdict_map = {v["finding_id"]: v for v in verdicts if v.get("finding_id")}

    status_counts: dict[str, int] = {
        "agree_retain": 0,
        "agree_reject": 0,
        "coarse_agree": 0,
        "disagree_high": 0,
        "disagree_medium": 0,
        "incomplete": 0,
    }

    annotated_findings: list[dict] = []
    for finding in gold_data.get("findings") or []:
        fid = finding.get("id") or finding.get("finding_id") or ""
        v = verdict_map.get(fid, {})
        rec = v.get("reconciliation", {})
        agreement = rec.get("agreement", "incomplete")
        locked_status = rec.get("locked_status")

        if agreement == "agree":
            if locked_status == "reject":
                status_counts["agree_reject"] += 1
            else:
                status_counts["agree_retain"] += 1
        elif agreement == "coarse_agree":
            status_counts["coarse_agree"] += 1
        elif agreement == "disagree":
            priority = rec.get("priority", "medium")
            if priority == "high":
                status_counts["disagree_high"] += 1
            else:
                status_counts["disagree_medium"] += 1
        else:
            status_counts["incomplete"] += 1

        annotated_findings.append({
            **finding,
            "_verification": v,
        })

    output = {
        **gold_data,
        "findings": annotated_findings,
        "verification_meta": {
            "verifier_a_model": state["verifier_a_model"],
            "verifier_b_model": state["verifier_b_model"],
            "as_of_date": state["as_of_date"],
            "status_counts": status_counts,
            "total_findings": len(annotated_findings),
        },
    }

    def _write(path: Path, data: dict) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)

    # asyncio-APPROVED-1: to_thread wraps blocking JSON file write
    await asyncio.to_thread(_write, out_path, output)
    logger.info("reconcile_output: wrote %s", out_path)

    return {
        "reconciled_path": str(out_path),
        "status_counts": status_counts,
    }


# ---------------------------------------------------------------------------
# Node: build_tiered_gold
# ---------------------------------------------------------------------------

async def build_tiered_gold(state: GsVerifierState) -> dict:
    verdicts = state.get("verdicts") or []
    gold_data = state.get("gold_data") or {}
    reconciled_path = state.get("reconciled_path")
    if not reconciled_path:
        return {"errors": ["build_tiered_gold: reconciled_path not set"]}

    base_dir = Path(reconciled_path).parent / "tiered"
    base_dir.mkdir(parents=True, exist_ok=True)

    verdict_map = {v["finding_id"]: v for v in verdicts if v.get("finding_id")}

    tier1: list[dict] = []  # agree + non-reject
    tier2: list[dict] = []  # disagreement / needs review
    tier3: list[dict] = []  # agree + reject

    for finding in gold_data.get("findings") or []:
        fid = finding.get("id") or finding.get("finding_id") or ""
        v = verdict_map.get(fid, {})
        rec = v.get("reconciliation", {})
        agreement = rec.get("agreement", "incomplete")
        locked_status = rec.get("locked_status")

        annotated = {**finding, "_verification": v}

        if agreement == "agree" and locked_status != "reject":
            tier1.append(annotated)
        elif agreement == "agree" and locked_status == "reject":
            tier3.append(annotated)
        else:
            tier2.append(annotated)

    def _write_tier(path: Path, items: list[dict]) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(items, fh, indent=2, ensure_ascii=False)

    # asyncio-APPROVED-2: concurrent file I/O — three tier JSON writes in parallel
    await asyncio.gather(
        asyncio.to_thread(_write_tier, base_dir / "tier1_retain.json", tier1),
        asyncio.to_thread(_write_tier, base_dir / "tier2_review.json", tier2),
        asyncio.to_thread(_write_tier, base_dir / "tier3_reject.json", tier3),
    )
    logger.info(
        "build_tiered_gold: tier1=%d tier2=%d tier3=%d",
        len(tier1), len(tier2), len(tier3),
    )

    return {"tiered_gold_dir": str(base_dir)}


# ---------------------------------------------------------------------------
# Node: apply_verdicts  (writes gold_v4.json, rejected.jsonl, flagged.jsonl)
# ---------------------------------------------------------------------------

async def apply_verdicts(state: GsVerifierState) -> dict:
    tiered_gold_dir = state.get("tiered_gold_dir")
    if not tiered_gold_dir:
        return {"errors": ["apply_verdicts: tiered_gold_dir not set"]}

    tiered_dir = Path(tiered_gold_dir)
    out_dir = tiered_dir.parent
    gold_v4_path = out_dir / "gold_v4.json"
    rejected_path = out_dir / "rejected.jsonl"
    flagged_path = out_dir / "flagged_for_review.jsonl"

    def _read_json(p: Path) -> list[dict]:
        if not p.exists():
            return []
        with open(p, encoding="utf-8") as fh:
            return json.load(fh)

    # asyncio-APPROVED-2: concurrent file I/O — three tier JSON reads in parallel
    tier1, tier2, tier3 = await asyncio.gather(
        asyncio.to_thread(_read_json, tiered_dir / "tier1_retain.json"),
        asyncio.to_thread(_read_json, tiered_dir / "tier2_review.json"),
        asyncio.to_thread(_read_json, tiered_dir / "tier3_reject.json"),
    )

    # gold_v4 = tier1 (kept findings, cleaned of internal _verification key)
    gold_v4_findings = [{k: v for k, v in f.items() if k != "_verification"} for f in tier1]

    def _write_json(path: Path, data: object) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)

    def _write_jsonl(path: Path, items: list[dict]) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            for item in items:
                fh.write(json.dumps(item, ensure_ascii=False) + "\n")

    # asyncio-APPROVED-2: concurrent file I/O — gold_v4 + rejected + flagged writes in parallel
    await asyncio.gather(
        asyncio.to_thread(_write_json, gold_v4_path, {"findings": gold_v4_findings}),
        asyncio.to_thread(_write_jsonl, rejected_path, tier3),
        asyncio.to_thread(_write_jsonl, flagged_path, tier2),
    )
    logger.info("apply_verdicts: gold_v4=%d retained, rejected=%d, flagged=%d",
                len(gold_v4_findings), len(tier3), len(tier2))

    return {"gold_v4_path": str(gold_v4_path)}


# ---------------------------------------------------------------------------
# Graph compilation
# ---------------------------------------------------------------------------

_builder: StateGraph = StateGraph(GsVerifierState)

_builder.add_node("load_gold", load_gold)
_builder.add_node("dispatch_findings", dispatch_findings)
_builder.add_node("verify_finding", verify_finding)
_builder.add_node("collect_verdicts", collect_verdicts)
_builder.add_node("reconcile_output", reconcile_output)
_builder.add_node("build_tiered_gold", build_tiered_gold)
_builder.add_node("apply_verdicts", apply_verdicts)

_builder.add_edge(START, "load_gold")
_builder.add_edge("load_gold", "dispatch_findings")
_builder.add_conditional_edges("dispatch_findings", lambda s: s, ["verify_finding"])
_builder.add_edge("verify_finding", "collect_verdicts")
_builder.add_edge("collect_verdicts", "reconcile_output")
_builder.add_edge("reconcile_output", "build_tiered_gold")
_builder.add_edge("build_tiered_gold", "apply_verdicts")
_builder.add_edge("apply_verdicts", END)

gs_verifier_graph = _builder.compile()


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def create_verifier_state(
    program: str,
    gold_path: str,
    out_path: str,
    as_of_date: str,
    verifier_a_model: str = DEFAULT_STRONG_MODEL,
    verifier_b_model: str = DEFAULT_SYNTHESIS_MODEL,
    skip_causal: bool = False,
) -> GsVerifierState:
    return GsVerifierState(
        program=program,
        gold_path=gold_path,
        out_path=out_path,
        as_of_date=as_of_date,
        verifier_a_model=verifier_a_model,
        verifier_b_model=verifier_b_model,
        skip_causal=skip_causal,
        gold_data=None,
        doc_list=None,
        investment_scoring=None,
        verdicts=[],
        reconciled_path=None,
        tiered_gold_dir=None,
        gold_v4_path=None,
        status_counts=None,
        errors=[],
    )
