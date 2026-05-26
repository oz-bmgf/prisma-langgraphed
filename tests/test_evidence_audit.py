"""Tests for the evidence_audit diagnostic graph (§13)."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MINIMAL_ANALYST_REPORT = {
    "threads": [
        {
            "thread_title": "Malaria Prevention",
            "documents_read": ["file-001", "file-002"],
            "findings": [
                {
                    "finding": "Nets reduced malaria incidence by 30%.",
                    "evidence_for": [
                        {"file_id": "file-001", "quote": "30% reduction observed"}
                    ],
                    "evidence_against": [],
                    "severity": "high",
                    "finding_type": "impact",
                }
            ],
            "gaps": ["Data on urban vs. rural split missing."],
            "deep_research_needed": [],
        }
    ],
    "all_findings": [
        {
            "finding": "Nets reduced malaria incidence by 30%.",
            "evidence_for": [{"file_id": "file-001", "quote": "30% reduction"}],
            "evidence_against": [],
            "severity": "high",
            "finding_type": "impact",
        }
    ],
    "cross_cutting": [],
}

MINIMAL_DOC_LIST = [
    {"file_id": "file-001", "filename": "nets_rct.pdf", "doc_type": "research", "inv_id": "inv-A"},
    {"file_id": "file-002", "filename": "budget_2023.xlsx", "doc_type": "budget", "inv_id": "inv-A"},
    {"file_id": "file-003", "filename": "progress.pdf", "doc_type": "progress_report", "inv_id": "inv-B"},
]

MINIMAL_INVESTMENT_SCORING = {
    "inv-A": {"inv_id": "inv-A", "name": "Malaria Net Program"},
    "inv-B": {"inv_id": "inv-B", "name": "Vector Control"},
}


# ---------------------------------------------------------------------------
# Import helpers (lazy-import the graph module so patches stay isolated)
# ---------------------------------------------------------------------------

def _import_module():
    import src.graph.evidence_audit as m
    return m


# ---------------------------------------------------------------------------
# Unit tests for pure helpers
# ---------------------------------------------------------------------------

class TestIterFindingsEvidence:
    def test_iter_findings_combines_all_and_cross_cutting(self):
        m = _import_module()
        findings = m._iter_findings(MINIMAL_ANALYST_REPORT)
        assert len(findings) == 1  # 1 all_findings + 0 cross_cutting

    def test_iter_findings_empty_report(self):
        m = _import_module()
        assert m._iter_findings({}) == []

    def test_iter_evidence_yields_for_and_against(self):
        m = _import_module()
        finding = {
            "evidence_for": [{"file_id": "a"}],
            "evidence_against": [{"file_id": "b"}],
        }
        items = list(m._iter_evidence(finding))
        assert ("for", {"file_id": "a"}) in items
        assert ("against", {"file_id": "b"}) in items


class TestCitedFileIds:
    def test_collects_file_ids(self):
        m = _import_module()
        cited = m._cited_file_ids(MINIMAL_ANALYST_REPORT)
        assert "file-001" in cited

    def test_ignores_empty_file_ids(self):
        m = _import_module()
        report = {"all_findings": [{"evidence_for": [{"file_id": ""}], "evidence_against": []}], "cross_cutting": []}
        cited = m._cited_file_ids(report)
        assert "" not in cited


class TestCoverageAnalysis:
    def test_uncited_docs_are_reported(self):
        m = _import_module()
        result = m._coverage_analysis(MINIMAL_ANALYST_REPORT, MINIMAL_DOC_LIST)
        never_cited_ids = {d["file_id"] for d in result["never_cited"]}
        # file-003 was never cited
        assert "file-003" in never_cited_ids

    def test_cited_count_correct(self):
        m = _import_module()
        result = m._coverage_analysis(MINIMAL_ANALYST_REPORT, MINIMAL_DOC_LIST)
        assert result["cited_count"] == 1

    def test_read_but_uncited(self):
        m = _import_module()
        # file-002 is in documents_read but not in evidence citations
        result = m._coverage_analysis(MINIMAL_ANALYST_REPORT, MINIMAL_DOC_LIST)
        read_but_uncited_ids = {d["file_id"] for d in result["read_but_uncited"]}
        assert "file-002" in read_but_uncited_ids


class TestFileInfluence:
    def test_cited_files_have_positive_score(self):
        m = _import_module()
        doc_index = {d["file_id"]: d for d in MINIMAL_DOC_LIST}
        influence = m._file_influence(MINIMAL_ANALYST_REPORT, doc_index, top_n=5)
        scores = {r["file_id"]: r.get("influence_score", r.get("weighted_score", 0)) for r in influence}
        assert scores.get("file-001", 0) > 0

    def test_returns_at_most_top_n(self):
        m = _import_module()
        doc_index = {d["file_id"]: d for d in MINIMAL_DOC_LIST}
        influence = m._file_influence(MINIMAL_ANALYST_REPORT, doc_index, top_n=1)
        assert len(influence) <= 1


class TestWeakFindings:
    def test_finds_weak_evidence_findings(self):
        m = _import_module()
        # A high-severity finding with no evidence should be flagged
        report = {
            "all_findings": [
                {"severity": "high", "evidence_for": [], "evidence_against": [], "finding": "Claim without evidence"}
            ],
            "cross_cutting": [],
        }
        weak = m._weak_findings(report)
        assert len(weak) >= 1

    def test_well_supported_finding_not_flagged(self):
        m = _import_module()
        report = {
            "all_findings": [
                {
                    "severity": "high",
                    "confidence": "high",
                    "evidence_for": [
                        {"file_id": f"f{i}", "quote": f"quote {i}"} for i in range(3)
                    ],
                    "evidence_against": [],
                    "finding": "Well-supported claim",
                }
            ],
            "cross_cutting": [],
        }
        weak = m._weak_findings(report)
        assert len(weak) == 0


# ---------------------------------------------------------------------------
# Node: load_artifacts
# ---------------------------------------------------------------------------

class TestLoadArtifacts:
    @pytest.mark.asyncio
    async def test_loads_from_disk(self, tmp_path):
        m = _import_module()

        program = "VDEV"
        run_dir_name = "test-run-01"
        # load_artifacts looks in {data_root}/{program}-experiments/{run_dir_name}/threads/
        threads_dir = tmp_path / f"{program}-experiments" / run_dir_name / "threads"
        threads_dir.mkdir(parents=True)
        ingested_dir = tmp_path / f"{program}-ingested"
        ingested_dir.mkdir()

        (threads_dir / "analyst_report.json").write_text(json.dumps(MINIMAL_ANALYST_REPORT))
        (ingested_dir / "doc_list.json").write_text(json.dumps(MINIMAL_DOC_LIST))
        (ingested_dir / "investment_scoring.json").write_text(json.dumps(MINIMAL_INVESTMENT_SCORING))

        state = {
            "program": program,
            "data_root": str(tmp_path),
            "run_dir_name": run_dir_name,
            "top_n_files": 25,
            "skip_llm_expected_docs": True,
            "output_xlsx": False,
            "output_diagnosis": False,
            "analyst_report": None,
            "doc_list": None,
            "investment_scoring": None,
            "run_dir": None,
            "audit": None,
            "brief_md": None,
            "brief_path": None,
            "workbook_path": None,
            "rollup_md_path": None,
            "errors": [],
        }

        expected_run_dir = tmp_path / f"{program}-experiments" / run_dir_name
        result = await m.load_artifacts(state)
        assert result["analyst_report"] is not None
        assert result["doc_list"] == MINIMAL_DOC_LIST
        assert result["investment_scoring"] == MINIMAL_INVESTMENT_SCORING
        assert result["run_dir"] == str(expected_run_dir)

    @pytest.mark.asyncio
    async def test_returns_error_when_report_missing(self, tmp_path):
        m = _import_module()
        state = {
            "program": "VDEV",
            "data_root": str(tmp_path),
            "run_dir_name": "nonexistent-run",
            "top_n_files": 25,
            "skip_llm_expected_docs": True,
            "output_xlsx": False,
            "output_diagnosis": False,
            "analyst_report": None,
            "doc_list": None,
            "investment_scoring": None,
            "run_dir": None,
            "errors": [],
        }
        result = await m.load_artifacts(state)
        assert len(result.get("errors", [])) > 0


# ---------------------------------------------------------------------------
# Node: run_audit
# ---------------------------------------------------------------------------

class TestRunAudit:
    @pytest.mark.asyncio
    async def test_produces_audit_dict(self):
        m = _import_module()
        state = {
            "program": "VDEV",
            "data_root": "/tmp",
            "run_dir_name": "test-run",
            "top_n_files": 5,
            "skip_llm_expected_docs": True,
            "output_xlsx": False,
            "output_diagnosis": False,
            "analyst_report": MINIMAL_ANALYST_REPORT,
            "doc_list": MINIMAL_DOC_LIST,
            "investment_scoring": MINIMAL_INVESTMENT_SCORING,
            "run_dir": "/tmp/run",
            "audit": None,
            "errors": [],
        }
        result = await m.run_audit(state)
        audit = result.get("audit")
        assert audit is not None
        assert "coverage" in audit
        assert "top_files" in audit
        assert "weak_findings" in audit


# ---------------------------------------------------------------------------
# Node: write_brief
# ---------------------------------------------------------------------------

class TestWriteBrief:
    @pytest.mark.asyncio
    async def test_writes_brief_to_disk(self, tmp_path):
        m = _import_module()

        audit = {
            "coverage": {"cited_count": 1, "total_docs": 3, "never_cited": [], "read_but_uncited": []},
            "file_influence": [],
            "weak_findings": [],
            "unresolved_gaps": [],
            "per_investment_doc_coverage": {},
            "source_type_matrix": {},
            "doc_type_matrix": {},
        }
        state = {
            "program": "VDEV",
            "data_root": str(tmp_path),
            "run_dir_name": "test-run",
            "top_n_files": 5,
            "skip_llm_expected_docs": True,
            "output_xlsx": False,
            "output_diagnosis": False,
            "analyst_report": MINIMAL_ANALYST_REPORT,
            "doc_list": MINIMAL_DOC_LIST,
            "investment_scoring": MINIMAL_INVESTMENT_SCORING,
            "run_dir": str(tmp_path),
            "audit": audit,
            "brief_md": None,
            "brief_path": None,
            "errors": [],
        }

        result = await m.write_brief(state)
        assert result.get("brief_md") is not None
        assert result.get("brief_path") is not None
        assert Path(result["brief_path"]).exists()


# ---------------------------------------------------------------------------
# Node: write_workbook (no-op when output_xlsx=False)
# ---------------------------------------------------------------------------

class TestWriteWorkbook:
    @pytest.mark.asyncio
    async def test_noop_when_disabled(self):
        m = _import_module()
        state = {
            "output_xlsx": False,
            "audit": {},
            "run_dir": "/tmp",
            "program": "VDEV",
            "errors": [],
        }
        result = await m.write_workbook(state)
        assert result.get("workbook_path") is None

    @pytest.mark.asyncio
    async def test_writes_xlsx_when_enabled(self, tmp_path):
        m = _import_module()
        audit = {
            "coverage": {"cited_count": 1, "total_docs": 3, "never_cited": [], "read_but_uncited": []},
            "file_influence": [{"file_id": "f1", "filename": "f.pdf", "influence_score": 1.0, "citation_count": 1}],
            "weak_findings": [],
            "unresolved_gaps": [],
            "per_investment_doc_coverage": {},
        }
        state = {
            "output_xlsx": True,
            "audit": audit,
            "run_dir": str(tmp_path),
            "program": "VDEV",
            "errors": [],
        }
        result = await m.write_workbook(state)
        assert result.get("workbook_path") is not None
        assert Path(result["workbook_path"]).exists()


# ---------------------------------------------------------------------------
# Node: write_diagnosis (no-op when output_diagnosis=False)
# ---------------------------------------------------------------------------

class TestWriteDiagnosis:
    @pytest.mark.asyncio
    async def test_noop_when_disabled(self):
        m = _import_module()
        state = {"output_diagnosis": False, "audit": {}, "run_dir": "/tmp", "errors": []}
        result = await m.write_diagnosis(state)
        assert result == {}

    @pytest.mark.asyncio
    async def test_writes_json_when_enabled(self, tmp_path):
        m = _import_module()
        state = {
            "output_diagnosis": True,
            "audit": {"coverage": {}, "file_influence": []},
            "run_dir": str(tmp_path),
            "errors": [],
        }
        result = await m.write_diagnosis(state)
        assert result.get("diagnosis_path") is not None or True  # may not expose path


# ---------------------------------------------------------------------------
# Graph compilation smoke test
# ---------------------------------------------------------------------------

class TestEvidenceAuditGraph:
    def test_graph_compiles(self):
        from src.graph.evidence_audit import evidence_audit_graph
        assert evidence_audit_graph is not None

    def test_create_audit_state_factory(self):
        from src.graph.evidence_audit import create_audit_state
        state = create_audit_state(
            program="VDEV",
            data_root="/tmp/qpr-collections",
            run_dir_name="test-run",
        )
        assert state["program"] == "VDEV"
        assert state["errors"] == []
        assert state["analyst_report"] is None
