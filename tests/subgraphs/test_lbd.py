"""Tests for src/graph/subgraphs/lbd.py — canonical LBD subgraph location."""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from langgraph.types import Send

from src.graph.subgraphs.lbd import (
    _parse_concepts,
    lbd_collect_concept_papers,
    lbd_dispatch_concepts,
    lbd_discover_connections,
    lbd_fetch_concept_papers,
    lbd_finalise,
    lbd_graph,
    lbd_synthesise,
)
from src.graph.state import LBDAgentState, LBDConceptFetchState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _state(**overrides) -> dict:
    base: dict = {
        "task_id": "L1",
        "query": "malaria iron deficiency anemia connection",
        "context": "",
        "seed_concepts": None,
        "concept_papers": [],
        "broad_search_results": None,
        "merged_papers": None,
        "discovered_concepts": None,
        "narrative": None,
        "paper_count": 0,
        "tool_traces": [],
        "result": None,
        "success": False,
        "error_message": None,
        "errors": [],
    }
    base.update(overrides)
    return base


_PAPERS = [
    {"title": "Malaria iron link", "abstract": "Iron deficiency reduces immunity.", "year": 2022},
    {"title": "Anemia tropics", "abstract": "Anemia prevalent in tropics.", "year": 2021},
]


# ---------------------------------------------------------------------------
# test_lbd_dispatch_skips_completed
# Task spec: fan-out uses seed_concepts; worker returns early when result pre-set
# ---------------------------------------------------------------------------


async def test_lbd_dispatch_sends_per_concept():
    """lbd_dispatch_concepts emits one Send per seed concept."""
    state = _state(seed_concepts=["malaria", "anemia", "iron"])
    result = await lbd_dispatch_concepts(state)
    assert isinstance(result, list)
    assert len(result) == 3
    assert all(s.node == "lbd_fetch_concept_papers" for s in result)
    concepts_sent = {s.arg["concept"] for s in result}
    assert concepts_sent == {"malaria", "anemia", "iron"}


async def test_lbd_dispatch_falls_back_to_query_when_no_concepts():
    """lbd_dispatch_concepts uses original query when seed_concepts is None."""
    state = _state(seed_concepts=None)
    result = await lbd_dispatch_concepts(state)
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0].arg["concept"] == state["query"]


async def test_lbd_dispatch_caps_at_max_concepts():
    """lbd_dispatch_concepts caps at _MAX_CONCEPTS_FOR_FETCH (5)."""
    state = _state(seed_concepts=["a", "b", "c", "d", "e", "f", "g"])
    result = await lbd_dispatch_concepts(state)
    assert isinstance(result, list)
    assert len(result) == 5


# ---------------------------------------------------------------------------
# test_lbd_worker_pure
# Task spec: worker returns plain dict, no side effects
# ---------------------------------------------------------------------------


async def test_lbd_worker_pure_returns_concept_papers():
    """lbd_fetch_concept_papers returns concept_papers and trace, no file I/O."""
    fake_papers = [{"title": "Paper A", "abstract": "abc", "year": 2020}]
    worker_state: LBDConceptFetchState = {
        "concept": "malaria",
        "query": "malaria iron link",
        "top_k": 10,
        "result": None,
    }
    with patch("src.graph.subgraphs.lbd.search_asta") as mock_tool:
        mock_tool.ainvoke = AsyncMock(return_value=fake_papers)
        result = await lbd_fetch_concept_papers(worker_state, {})

    assert isinstance(result, dict)
    assert "concept_papers" in result
    assert result["concept_papers"] == fake_papers
    assert result["tool_traces"][0]["success"] is True
    assert result["tool_traces"][0]["concept"] == "malaria"


async def test_lbd_worker_error_surfaces_to_errors_field():
    """lbd_fetch_concept_papers network failure goes into errors[], not raises."""
    worker_state: LBDConceptFetchState = {
        "concept": "anemia",
        "query": "malaria iron link",
        "top_k": 10,
        "result": None,
    }
    with patch("src.graph.subgraphs.lbd.search_asta") as mock_tool:
        mock_tool.ainvoke = AsyncMock(side_effect=RuntimeError("network timeout"))
        result = await lbd_fetch_concept_papers(worker_state, {})

    assert "errors" in result
    assert any("anemia" in e for e in result["errors"])
    assert result["tool_traces"][0]["success"] is False


async def test_lbd_worker_skips_when_result_pre_populated():
    """lbd_fetch_concept_papers returns early when result already set (idempotency)."""
    prepopulated = [{"title": "Cached paper"}]
    worker_state: LBDConceptFetchState = {
        "concept": "malaria",
        "query": "malaria iron link",
        "top_k": 10,
        "result": prepopulated,
    }
    result = await lbd_fetch_concept_papers(worker_state, {})
    assert result == {"concept_papers": prepopulated}


# ---------------------------------------------------------------------------
# test_lbd_graph_has_all_nodes
# ---------------------------------------------------------------------------


def test_lbd_graph_has_all_nodes():
    """Compiled graph nodes match topology declared in module docstring."""
    graph_nodes = set(lbd_graph.get_graph().nodes.keys()) - {"__start__", "__end__"}
    expected = {
        "lbd_start",
        "lbd_broad_search",
        "lbd_extract_concepts",
        "lbd_fetch_concept_papers",
        "lbd_collect_concept_papers",
        "lbd_discover_connections",
        "lbd_synthesise",
        "lbd_finalise",
    }
    assert graph_nodes == expected


def test_lbd_graph_compiles():
    assert lbd_graph is not None


# ---------------------------------------------------------------------------
# test_lbd_handles_empty_results
# Task spec: discover_connections and synthesise handle empty merged_papers gracefully
# ---------------------------------------------------------------------------


async def test_lbd_discover_connections_returns_empty_when_no_papers():
    """lbd_discover_connections sets merged_papers=[] and discovered_concepts=[] gracefully."""
    state = _state(concept_papers=[], broad_search_results=[])
    result = await lbd_discover_connections(state, {})
    assert result["merged_papers"] == []
    assert result["discovered_concepts"] == []


async def test_lbd_discover_connections_writes_merged_papers():
    """lbd_discover_connections deduplicates concept_papers + broad_search_results once."""
    paper_a = {"title": "Paper A", "abstract": "abc"}
    paper_b = {"title": "Paper B", "abstract": "def"}
    paper_b_dup = {"title": "Paper B", "abstract": "def (dup)"}
    state = _state(
        concept_papers=[paper_a, paper_b],
        broad_search_results=[paper_b_dup],
    )
    with patch("src.graph.subgraphs.lbd.acall_llm", new=AsyncMock(return_value='["bridge_concept"]')):
        result = await lbd_discover_connections(state, {})

    assert "merged_papers" in result
    titles = [p["title"] for p in result["merged_papers"]]
    assert titles.count("Paper B") == 1
    assert len(result["merged_papers"]) == 2


async def test_lbd_synthesise_skips_when_already_done():
    """lbd_synthesise returns {} when narrative already set (idempotency)."""
    state = _state(narrative="Existing narrative.")
    result = await lbd_synthesise(state, {})
    assert result == {}


async def test_lbd_synthesise_llm_failure_surfaces_error():
    """lbd_synthesise LLM exception surfaces to errors field, narrative=None."""
    state = _state(merged_papers=_PAPERS)
    with patch("src.graph.subgraphs.lbd.acall_llm", new=AsyncMock(side_effect=RuntimeError("LLM down"))):
        result = await lbd_synthesise(state, {})
    assert result["narrative"] is None
    assert any("lbd_synthesise" in e for e in result.get("errors", []))


async def test_lbd_finalise_adds_status_field():
    """lbd_finalise adds status: ok/error for finalize.py enrichment gate."""
    state = _state(merged_papers=_PAPERS, narrative="connections found", errors=[])
    result = await lbd_finalise(state, {})
    assert result["result"]["status"] == "ok"

    state_err = _state(errors=["lbd_fetch_x: timeout"])
    result_err = await lbd_finalise(state_err, {})
    assert result_err["result"]["status"] == "error"


async def test_lbd_finalise_reads_merged_papers():
    """lbd_finalise uses merged_papers (not raw concept_papers) for papers list."""
    state = _state(
        merged_papers=_PAPERS,
        concept_papers=[{"title": "Stale"}],
        narrative="connections found",
    )
    result = await lbd_finalise(state, {})
    assert result["result"]["papers"] == _PAPERS
    assert result["result"]["paper_count"] == 2


# ---------------------------------------------------------------------------
# config threading
# ---------------------------------------------------------------------------


async def test_lbd_discover_connections_passes_config_to_acall_llm():
    """acall_llm must receive config= so LLM calls appear as child spans."""
    captured: list[dict] = []

    async def _mock_acall(*args, config=None, **kwargs):
        captured.append({"config": config})
        return '["bridge_concept"]'

    state = _state(concept_papers=_PAPERS, broad_search_results=[])
    with patch("src.graph.subgraphs.lbd.acall_llm", side_effect=_mock_acall):
        await lbd_discover_connections(state, {"configurable": {"research_model": "test-model"}})

    assert len(captured) == 1
    assert captured[0]["config"] is not None


# ---------------------------------------------------------------------------
# Full dry-run through compiled graph
# ---------------------------------------------------------------------------


async def test_lbd_graph_dry_run():
    """End-to-end dry run through compiled lbd_graph with mocked I/O."""
    fake_papers = [
        {"title": "Study A", "abstract": "malaria and iron link", "year": 2020},
        {"title": "Study B", "abstract": "anemia and nets", "year": 2021},
    ]
    with (
        patch("src.graph.subgraphs.lbd.search_asta") as mock_asta,
        patch("src.graph.subgraphs.lbd.acall_llm", new=AsyncMock(return_value='["malaria", "anemia"]')),
    ):
        mock_asta.ainvoke = AsyncMock(return_value=fake_papers)
        final = await lbd_graph.ainvoke(_state())

    assert final["result"] is not None
    assert final["result"]["success"] is True
    assert final["merged_papers"] is not None
    assert len(final["merged_papers"]) > 0
