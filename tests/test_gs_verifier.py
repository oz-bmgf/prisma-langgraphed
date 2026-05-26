"""Tests for the gs_verifier dual-verifier graph (§13)."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_FINDING = {
    "id": "f-001",
    "finding": "Malaria nets reduced incidence by 30% in the study region.",
    "finding_type": "impact",
    "evidence_for": [{"file_id": "file-001", "quote": "30% reduction observed in RCT"}],
    "evidence_against": [],
    "severity": "high",
}

SAMPLE_GOLD_DATA = {
    "program": "VDEV",
    "findings": [SAMPLE_FINDING],
    "meta": {"version": "v2"},
}

SAMPLE_DOC_LIST = [
    {"file_id": "file-001", "filename": "rct_nets.pdf", "doc_type": "research", "inv_id": "inv-A"},
    {"file_id": "file-002", "filename": "progress.pdf", "doc_type": "progress_report", "inv_id": "inv-A"},
]

VALID_VERDICT = {
    "overall_status": "retain",
    "confidence": "high",
    "rationale": "Evidence directly supports the claim.",
    "verdict_decomposition": {
        "claim_accuracy": "supported",
        "evidence_quality": "strong",
        "evidence_relevance": "directly_relevant",
        "magnitude_accuracy": "accurate",
        "attribution_accuracy": "accurate",
        "temporal_validity": "current",
        "causal_validity": "supported",
        "scope_accuracy": "accurate",
    },
    "pipeline_evidence_ledger": [
        {"file_id": "file-001", "quote": "30% reduction observed in RCT", "supports": True}
    ],
    "overrule_analysis": {"would_overrule": False, "overrule_reason": None},
}


def _import_module():
    import src.graph.gs_verifier as m
    return m


# ---------------------------------------------------------------------------
# Unit tests for pure helpers
# ---------------------------------------------------------------------------

class TestExtractJson:
    def test_extracts_bare_json(self):
        m = _import_module()
        text = '{"overall_status": "retain"}'
        result = m._extract_json(text)
        assert result["overall_status"] == "retain"

    def test_extracts_from_fenced_block(self):
        m = _import_module()
        text = 'Some prose.\n```json\n{"overall_status": "reject"}\n```'
        result = m._extract_json(text)
        assert result["overall_status"] == "reject"

    def test_raises_on_no_json(self):
        m = _import_module()
        with pytest.raises((ValueError, Exception)):
            m._extract_json("No JSON here at all.")


class TestReconcile:
    def test_exact_agree_retain(self):
        m = _import_module()
        a = {"overall_status": "retain"}
        b = {"overall_status": "retain"}
        rec = m._reconcile(a, b)
        assert rec["agreement"] == "agree"
        assert rec["locked_status"] == "retain"
        assert rec["priority"] == "low"

    def test_exact_agree_reject(self):
        m = _import_module()
        a = {"overall_status": "reject"}
        b = {"overall_status": "reject"}
        rec = m._reconcile(a, b)
        assert rec["agreement"] == "agree"
        assert rec["locked_status"] == "reject"

    def test_coarse_agree_modify_demote(self):
        m = _import_module()
        a = {"overall_status": "modify"}
        b = {"overall_status": "demote"}
        rec = m._reconcile(a, b)
        assert rec["agreement"] == "coarse_agree"
        # locked on the stricter label
        assert rec["locked_status"] == "demote"
        assert rec["priority"] == "medium"

    def test_keep_vs_drop_high_priority(self):
        m = _import_module()
        a = {"overall_status": "retain"}
        b = {"overall_status": "reject"}
        rec = m._reconcile(a, b)
        assert rec["agreement"] == "disagree"
        assert rec["priority"] == "high"
        assert rec["locked_status"] is None

    def test_revise_vs_drop_medium_priority(self):
        m = _import_module()
        a = {"overall_status": "modify"}
        b = {"overall_status": "reject"}
        rec = m._reconcile(a, b)
        assert rec["agreement"] == "disagree"
        assert rec["priority"] == "medium"

    def test_malformed_status_incomplete(self):
        m = _import_module()
        a = {"overall_status": "error"}
        b = {"overall_status": "retain"}
        rec = m._reconcile(a, b)
        assert rec["agreement"] == "incomplete"
        assert rec["priority"] == "critical"


class TestBuildEvidenceBundle:
    def test_includes_evidence_for(self):
        m = _import_module()
        bundle = m._build_evidence_bundle(SAMPLE_FINDING, SAMPLE_DOC_LIST)
        assert "SUPPORTING EVIDENCE" in bundle
        assert "30% reduction" in bundle

    def test_missing_doc_list_uses_file_id(self):
        m = _import_module()
        bundle = m._build_evidence_bundle(SAMPLE_FINDING, [])
        assert "file-001" in bundle

    def test_no_evidence_returns_fallback(self):
        m = _import_module()
        bundle = m._build_evidence_bundle({"evidence_for": [], "evidence_against": []}, [])
        assert "no evidence" in bundle.lower()


class TestClassifyFindingType:
    def test_uses_finding_type_field(self):
        m = _import_module()
        finding = {"finding_type": "impact"}
        assert m._classify_finding_type(finding) == "impact"

    def test_falls_back_to_type(self):
        m = _import_module()
        finding = {"type": "gap"}
        assert m._classify_finding_type(finding) == "gap"

    def test_falls_back_to_observation(self):
        m = _import_module()
        assert m._classify_finding_type({}) == "observation"


# ---------------------------------------------------------------------------
# Node: load_gold
# ---------------------------------------------------------------------------

class TestLoadGold:
    @pytest.mark.asyncio
    async def test_loads_from_disk(self, tmp_path):
        m = _import_module()

        program = "VDEV"
        # load_gold looks for ingested_dir at gold_path.parent.parent / f"{program}-ingested"
        # So place gold_path one level deeper than data_root
        data_root = tmp_path
        gold_dir = data_root / "runs"
        gold_dir.mkdir()
        gold_path = gold_dir / "gold_standard_verified.json"
        gold_path.write_text(json.dumps(SAMPLE_GOLD_DATA))

        ingested_dir = data_root / f"{program}-ingested"
        ingested_dir.mkdir()
        (ingested_dir / "doc_list.json").write_text(json.dumps(SAMPLE_DOC_LIST))
        (ingested_dir / "investment_scoring.json").write_text(json.dumps({"inv-A": {}}))

        state = {
            "program": program,
            "gold_path": str(gold_path),
            "out_path": str(data_root / "out" / "gold_v3.json"),
            "as_of_date": "2025-01-01",
            "verifier_a_model": "claude-opus-4-7",
            "verifier_b_model": "claude-sonnet-4-6",
            "skip_causal": False,
            "gold_data": None,
            "doc_list": None,
            "investment_scoring": None,
            "verdicts": [],
            "errors": [],
        }
        result = await m.load_gold(state)
        assert result["gold_data"] == SAMPLE_GOLD_DATA
        assert result["doc_list"] == SAMPLE_DOC_LIST

    @pytest.mark.asyncio
    async def test_returns_error_on_missing_file(self, tmp_path):
        m = _import_module()
        state = {
            "program": "VDEV",
            "gold_path": str(tmp_path / "nonexistent.json"),
            "out_path": "/tmp/out.json",
            "as_of_date": "2025-01-01",
            "verifier_a_model": "claude-opus-4-7",
            "verifier_b_model": "claude-sonnet-4-6",
            "skip_causal": False,
            "gold_data": None,
            "doc_list": None,
            "investment_scoring": None,
            "verdicts": [],
            "errors": [],
        }
        result = await m.load_gold(state)
        assert len(result.get("errors", [])) > 0


# ---------------------------------------------------------------------------
# Node: dispatch_findings
# ---------------------------------------------------------------------------

class TestDispatchFindings:
    def test_returns_one_send_per_finding(self):
        from langgraph.types import Send
        m = _import_module()
        state = {
            "program": "VDEV",
            "gold_path": "/tmp/gold.json",
            "out_path": "/tmp/out.json",
            "as_of_date": "2025-01-01",
            "verifier_a_model": "claude-opus-4-7",
            "verifier_b_model": "claude-sonnet-4-6",
            "skip_causal": False,
            "gold_data": SAMPLE_GOLD_DATA,
            "doc_list": SAMPLE_DOC_LIST,
            "investment_scoring": {},
            "verdicts": [],
            "errors": [],
        }
        sends = m.dispatch_findings(state)
        assert len(sends) == 1
        assert isinstance(sends[0], Send)
        assert sends[0].node == "verify_finding"

    def test_returns_empty_list_when_no_findings(self):
        m = _import_module()
        state = {
            "program": "VDEV",
            "gold_path": "/tmp/gold.json",
            "out_path": "/tmp/out.json",
            "as_of_date": "2025-01-01",
            "verifier_a_model": "claude-opus-4-7",
            "verifier_b_model": "claude-sonnet-4-6",
            "skip_causal": False,
            "gold_data": {"findings": []},
            "doc_list": [],
            "investment_scoring": {},
            "verdicts": [],
            "errors": [],
        }
        sends = m.dispatch_findings(state)
        assert sends == []


# ---------------------------------------------------------------------------
# Node: verify_finding
# ---------------------------------------------------------------------------

class TestVerifyFinding:
    @pytest.mark.asyncio
    async def test_calls_both_verifiers_in_parallel(self):
        m = _import_module()

        verdict_json = json.dumps(VALID_VERDICT)

        with patch("src.graph.gs_verifier.acall_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = verdict_json

            state = {
                "finding": SAMPLE_FINDING,
                "scope_label": "Malaria Prevention",
                "finding_type": "impact",
                "evidence_bundle": "SUPPORTING EVIDENCE\n- [rct_nets.pdf] 30% reduction",
                "program": "VDEV",
                "as_of_date": "2025-01-01",
                "verifier_a_model": "claude-opus-4-7",
                "verifier_b_model": "claude-sonnet-4-6",
                "result": None,
            }
            result = await m.verify_finding(state)

        assert mock_llm.call_count == 2
        verdicts = result["verdicts"]
        assert len(verdicts) == 1
        v = verdicts[0]
        assert v["finding_id"] == "f-001"
        assert "verdict_a" in v
        assert "verdict_b" in v
        assert "reconciliation" in v

    @pytest.mark.asyncio
    async def test_handles_malformed_verdict_gracefully(self):
        m = _import_module()

        with patch("src.graph.gs_verifier.acall_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = "Sorry, I cannot help with that."

            state = {
                "finding": SAMPLE_FINDING,
                "scope_label": "Malaria Prevention",
                "finding_type": "impact",
                "evidence_bundle": "No evidence.",
                "program": "VDEV",
                "as_of_date": "2025-01-01",
                "verifier_a_model": "claude-opus-4-7",
                "verifier_b_model": "claude-sonnet-4-6",
                "result": None,
            }
            result = await m.verify_finding(state)

        # Should not raise; errors captured
        assert len(result.get("errors", [])) > 0
        verdicts = result["verdicts"]
        assert len(verdicts) == 1
        rec = verdicts[0]["reconciliation"]
        assert rec["agreement"] == "incomplete"

    @pytest.mark.asyncio
    async def test_exact_agreement_produces_low_priority(self):
        m = _import_module()

        verdict_json = json.dumps(VALID_VERDICT)  # both verifiers return "retain"

        with patch("src.graph.gs_verifier.acall_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = verdict_json

            state = {
                "finding": SAMPLE_FINDING,
                "scope_label": "Malaria Prevention",
                "finding_type": "impact",
                "evidence_bundle": "SUPPORTING EVIDENCE\n- [rct_nets.pdf] 30% reduction",
                "program": "VDEV",
                "as_of_date": "2025-01-01",
                "verifier_a_model": "claude-opus-4-7",
                "verifier_b_model": "claude-sonnet-4-6",
                "result": None,
            }
            result = await m.verify_finding(state)

        rec = result["verdicts"][0]["reconciliation"]
        assert rec["agreement"] == "agree"
        assert rec["priority"] == "low"


# ---------------------------------------------------------------------------
# Node: reconcile_output
# ---------------------------------------------------------------------------

class TestReconcileOutput:
    @pytest.mark.asyncio
    async def test_writes_annotated_findings_to_disk(self, tmp_path):
        m = _import_module()

        verdict = {
            "finding_id": "f-001",
            "scope_label": "Malaria Prevention",
            "finding_type": "impact",
            "verdict_a": VALID_VERDICT,
            "verdict_b": VALID_VERDICT,
            "reconciliation": {
                "agreement": "agree",
                "locked_status": "retain",
                "priority": "low",
                "reconcile_note": "Both agree: retain",
            },
        }
        out_path = tmp_path / "gold_v3_reverified.json"

        state = {
            "program": "VDEV",
            "gold_path": str(tmp_path / "gold.json"),
            "out_path": str(out_path),
            "as_of_date": "2025-01-01",
            "verifier_a_model": "claude-opus-4-7",
            "verifier_b_model": "claude-sonnet-4-6",
            "skip_causal": False,
            "gold_data": SAMPLE_GOLD_DATA,
            "doc_list": SAMPLE_DOC_LIST,
            "investment_scoring": {},
            "verdicts": [verdict],
            "reconciled_path": None,
            "errors": [],
        }

        result = await m.reconcile_output(state)
        assert result["reconciled_path"] == str(out_path)
        assert out_path.exists()

        written = json.loads(out_path.read_text())
        assert "findings" in written
        assert "verification_meta" in written
        assert written["verification_meta"]["status_counts"]["agree_retain"] == 1


# ---------------------------------------------------------------------------
# Node: build_tiered_gold
# ---------------------------------------------------------------------------

class TestBuildTieredGold:
    @pytest.mark.asyncio
    async def test_classifies_tiers_correctly(self, tmp_path):
        m = _import_module()

        reconciled_path = tmp_path / "gold_v3_reverified.json"
        reconciled_path.write_text(json.dumps({}))

        verdicts = [
            {
                "finding_id": "f-001",
                "reconciliation": {"agreement": "agree", "locked_status": "retain"},
            },
            {
                "finding_id": "f-002",
                "reconciliation": {"agreement": "disagree", "locked_status": None},
            },
            {
                "finding_id": "f-003",
                "reconciliation": {"agreement": "agree", "locked_status": "reject"},
            },
        ]
        gold_data = {
            "findings": [
                {"id": "f-001", "finding": "Finding 1"},
                {"id": "f-002", "finding": "Finding 2"},
                {"id": "f-003", "finding": "Finding 3"},
            ]
        }

        state = {
            "program": "VDEV",
            "gold_path": str(tmp_path / "gold.json"),
            "out_path": str(reconciled_path),
            "as_of_date": "2025-01-01",
            "verifier_a_model": "claude-opus-4-7",
            "verifier_b_model": "claude-sonnet-4-6",
            "skip_causal": False,
            "gold_data": gold_data,
            "doc_list": [],
            "investment_scoring": {},
            "verdicts": verdicts,
            "reconciled_path": str(reconciled_path),
            "tiered_gold_dir": None,
            "errors": [],
        }

        result = await m.build_tiered_gold(state)
        tiered_dir = Path(result["tiered_gold_dir"])
        tier1 = json.loads((tiered_dir / "tier1_retain.json").read_text())
        tier2 = json.loads((tiered_dir / "tier2_review.json").read_text())
        tier3 = json.loads((tiered_dir / "tier3_reject.json").read_text())

        assert len(tier1) == 1 and tier1[0]["id"] == "f-001"
        assert len(tier2) == 1 and tier2[0]["id"] == "f-002"
        assert len(tier3) == 1 and tier3[0]["id"] == "f-003"


# ---------------------------------------------------------------------------
# Node: apply_verdicts
# ---------------------------------------------------------------------------

class TestApplyVerdicts:
    @pytest.mark.asyncio
    async def test_writes_gold_v4_and_jsonl_files(self, tmp_path):
        m = _import_module()

        tiered_dir = tmp_path / "tiered"
        tiered_dir.mkdir()

        tier1 = [{"id": "f-001", "finding": "Retained finding"}]
        tier2 = [{"id": "f-002", "finding": "Flagged finding"}]
        tier3 = [{"id": "f-003", "finding": "Rejected finding"}]

        (tiered_dir / "tier1_retain.json").write_text(json.dumps(tier1))
        (tiered_dir / "tier2_review.json").write_text(json.dumps(tier2))
        (tiered_dir / "tier3_reject.json").write_text(json.dumps(tier3))

        state = {
            "program": "VDEV",
            "gold_path": str(tmp_path / "gold.json"),
            "out_path": str(tmp_path / "gold_v3.json"),
            "as_of_date": "2025-01-01",
            "verifier_a_model": "claude-opus-4-7",
            "verifier_b_model": "claude-sonnet-4-6",
            "skip_causal": False,
            "gold_data": {},
            "doc_list": [],
            "investment_scoring": {},
            "verdicts": [],
            "reconciled_path": str(tmp_path / "gold_v3.json"),
            "tiered_gold_dir": str(tiered_dir),
            "gold_v4_path": None,
            "errors": [],
        }

        result = await m.apply_verdicts(state)
        gold_v4_path = Path(result["gold_v4_path"])
        assert gold_v4_path.exists()

        gold_v4 = json.loads(gold_v4_path.read_text())
        assert len(gold_v4["findings"]) == 1
        assert gold_v4["findings"][0]["id"] == "f-001"

        rejected_path = gold_v4_path.parent / "rejected.jsonl"
        flagged_path = gold_v4_path.parent / "flagged_for_review.jsonl"
        assert rejected_path.exists()
        assert flagged_path.exists()


# ---------------------------------------------------------------------------
# Graph compilation smoke test
# ---------------------------------------------------------------------------

class TestGsVerifierGraph:
    def test_graph_compiles(self):
        from src.graph.gs_verifier import gs_verifier_graph
        assert gs_verifier_graph is not None

    def test_create_verifier_state_factory(self):
        from src.graph.gs_verifier import create_verifier_state
        state = create_verifier_state(
            program="VDEV",
            gold_path="/tmp/gold.json",
            out_path="/tmp/out.json",
            as_of_date="2025-01-01",
        )
        assert state["program"] == "VDEV"
        assert state["verifier_a_model"] == "claude-opus-4-7"
        assert state["verifier_b_model"] == "claude-sonnet-4-6"
        assert state["verdicts"] == []
        assert state["errors"] == []
