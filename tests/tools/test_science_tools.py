"""Unit tests for src/tools/science_tools.py."""
import pytest

from src.tools.science_tools import SCIENCE_TOOLS, search_asta


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class MockSearchBackend:
    async def search(self, *a, **kw): return []
    async def distinct_inv_ids(self): return []
    async def distinct_bow_ids(self): return []
    async def count_by_bow_id(self): return {}


def _config(extra: dict | None = None):
    base = {"configurable": {"search_backend": MockSearchBackend(), "doc_list": []}}
    if extra:
        base["configurable"].update(extra)
    return base


# ---------------------------------------------------------------------------
# search_asta
# ---------------------------------------------------------------------------


async def test_search_asta_no_api_key():
    import os
    original = os.environ.pop("ASTA_API_KEY", None)
    try:
        result = await search_asta.ainvoke(
            {"query": "malaria vaccine efficacy"}, config=_config()
        )
        # Without an API key: either Semantic Scholar results or "not configured"
        assert isinstance(result, str)
        assert len(result) > 0
    finally:
        if original:
            os.environ["ASTA_API_KEY"] = original


async def test_science_tools_count():
    assert len(SCIENCE_TOOLS) == 11  # 1 search_asta + 10 investigation tools


async def test_science_tools_includes_search_asta():
    names = [t.name for t in SCIENCE_TOOLS]
    assert "search_asta" in names


async def test_science_tools_includes_investigation_tools():
    names = [t.name for t in SCIENCE_TOOLS]
    for expected in [
        "search_investment", "search_portfolio", "search_web",
        "read_document", "compute", "submit_findings",
        "list_documents", "read_document_summary",
        "get_document_structure", "read_section",
    ]:
        assert expected in names, f"{expected} missing from SCIENCE_TOOLS"
