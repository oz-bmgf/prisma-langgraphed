from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, patch

from src.graph.agents.lbd_graph import (
    lbd_graph,
    lbd_extract_concepts,
    lbd_dispatch_concepts,  # conditional edge function
    lbd_fetch_concept_papers,
    lbd_collect_concept_papers,
    lbd_finalise,
    _parse_concepts,
)
from src.graph.state import LBDAgentState, LBDConceptFetchState


def minimal_lbd_state(**overrides) -> LBDAgentState:
    base: LBDAgentState = {
        "task_id": "lbd-001",
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


def test_lbd_graph_compiles():
    assert lbd_graph is not None
    nodes = list(lbd_graph.nodes.keys())
    assert "lbd_extract_concepts" in nodes
    # lbd_dispatch_concepts is a conditional edge function, not a node
    assert "lbd_fetch_concept_papers" in nodes
    assert "lbd_collect_concept_papers" in nodes
    assert "lbd_discover_connections" in nodes
    assert "lbd_synthesise" in nodes
    assert "lbd_finalise" in nodes


def test_parse_concepts_from_list():
    result = _parse_concepts('["malaria", "iron deficiency", "anemia"]')
    assert "malaria" in result
    assert "anemia" in result


def test_parse_concepts_from_csv():
    result = _parse_concepts("malaria, iron deficiency, anemia, immune response")
    assert len(result) >= 3


@pytest.mark.asyncio
async def test_lbd_extract_concepts_returns_list():
    with patch("src.graph.subgraphs.lbd.acall_llm", new=AsyncMock(return_value="malaria, anemia, iron")):
        state = minimal_lbd_state()
        result = await lbd_extract_concepts(state, {})
    assert "seed_concepts" in result
    assert isinstance(result["seed_concepts"], list)
    assert len(result["seed_concepts"]) >= 1


@pytest.mark.asyncio
async def test_lbd_extract_concepts_falls_back_on_error():
    with patch("src.graph.subgraphs.lbd.acall_llm", new=AsyncMock(side_effect=Exception("LLM error"))):
        state = minimal_lbd_state()
        result = await lbd_extract_concepts(state, {})
    assert result["seed_concepts"] == [state["query"]]


@pytest.mark.asyncio
async def test_lbd_dispatch_concepts_sends_per_concept():
    from langgraph.types import Send
    state = minimal_lbd_state(seed_concepts=["malaria", "anemia", "iron deficiency"])
    result = await lbd_dispatch_concepts(state)
    assert isinstance(result, list)
    assert len(result) == 3
    for s in result:
        assert isinstance(s, Send)
        assert s.node == "lbd_fetch_concept_papers"


@pytest.mark.asyncio
async def test_lbd_dispatch_concepts_uses_query_as_fallback():
    from langgraph.types import Send
    state = minimal_lbd_state(seed_concepts=None)
    result = await lbd_dispatch_concepts(state)
    assert len(result) == 1
    assert result[0].arg["concept"] == state["query"]


@pytest.mark.asyncio
async def test_lbd_fetch_concept_papers_success():
    fake_papers = [{"title": "Paper A"}, {"title": "Paper B"}]
    with patch("src.graph.subgraphs.lbd.search_asta") as mock_tool:
        mock_tool.ainvoke = AsyncMock(return_value=fake_papers)
        state: LBDConceptFetchState = {"concept": "malaria", "query": "test", "top_k": 10, "result": None}
        result = await lbd_fetch_concept_papers(state, {})
    assert "concept_papers" in result
    assert len(result["concept_papers"]) == 2
    assert result["tool_traces"][0]["success"] is True


@pytest.mark.asyncio
async def test_lbd_fetch_concept_papers_error():
    with patch("src.graph.subgraphs.lbd.search_asta") as mock_tool:
        mock_tool.ainvoke = AsyncMock(side_effect=Exception("network error"))
        state: LBDConceptFetchState = {"concept": "malaria", "query": "test", "top_k": 10, "result": None}
        result = await lbd_fetch_concept_papers(state, {})
    assert "errors" in result
    assert result["tool_traces"][0]["success"] is False


@pytest.mark.asyncio
async def test_lbd_collect_concept_papers_deduplicates():
    papers = [
        {"title": "Paper A"},
        {"title": "Paper B"},
        {"title": "Paper A"},  # duplicate
    ]
    state = minimal_lbd_state(concept_papers=papers)
    result = await lbd_collect_concept_papers(state, {})
    assert result["paper_count"] == 2


@pytest.mark.asyncio
async def test_lbd_finalise_success():
    state = minimal_lbd_state(
        narrative="Found indirect connections via bridging concepts.",
        concept_papers=[{"title": "A"}, {"title": "B"}],
        seed_concepts=["malaria", "anemia"],
        discovered_concepts=[{"term": "iron", "type": "bridge"}],
    )
    result = await lbd_finalise(state, {})
    assert result["result"]["success"] is True
    assert result["result"]["thesis"] == "Found indirect connections via bridging concepts."
    assert "malaria" in result["result"]["concepts"]


@pytest.mark.asyncio
async def test_lbd_graph_dry_run():
    fake_papers = [
        {"title": "Paper A", "abstract": "malaria and iron deficiency", "year": 2020},
        {"title": "Paper B", "abstract": "anemia in sub-saharan africa", "year": 2021},
    ]
    with (
        patch("src.graph.subgraphs.lbd.acall_llm", new=AsyncMock(return_value="malaria, anemia, iron")),
        patch("src.graph.subgraphs.lbd.search_asta") as mock_asta,
    ):
        mock_asta.ainvoke = AsyncMock(return_value=fake_papers)
        result_state = await lbd_graph.ainvoke(minimal_lbd_state())

    assert result_state["result"] is not None
    assert result_state["result"]["success"] is True
