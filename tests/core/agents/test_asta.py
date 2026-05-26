"""Unit tests for src/core/agents/asta.py."""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.agents.asta import AstaClient, _normalize_ss_paper


# ---------------------------------------------------------------------------
# AstaClient.search() — with API key, primary path
# ---------------------------------------------------------------------------


async def test_search_calls_asta_when_api_key_set():
    parsed_paper = {
        "paperId": "P1",
        "title": "Malaria vaccine",
        "year": 2022,
        "authors": [{"name": "Smith J"}],
        "abstract": "Efficacy 85%.",
        "externalIds": {},
        "url": "",
    }

    # SSE response: code uses resp.text (not resp.json()) and parses each `data:` line
    sse_event = {
        "result": {
            "content": [
                {"type": "text", "text": json.dumps(parsed_paper)}
            ]
        }
    }
    sse_body = f"data: {json.dumps(sse_event)}\n\n"

    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.text = sse_body

    mock_async_client = MagicMock()
    mock_async_client.__aenter__ = AsyncMock(return_value=mock_async_client)
    mock_async_client.__aexit__ = AsyncMock(return_value=False)
    mock_async_client.post = AsyncMock(return_value=mock_response)

    enriched = [{
        "paperId": "P1",
        "title": "Malaria vaccine",
        "year": 2022,
        "authors": [{"name": "Smith J"}],
        "abstract": "Efficacy 85%.",
    }]

    with patch("src.core.agents.asta.ASTA_API_KEY", "test-key"), \
         patch("src.core.agents.asta.httpx.AsyncClient", return_value=mock_async_client), \
         patch.object(AstaClient, "_enrich_by_ids", new_callable=lambda: lambda self, *a, **k: __import__("asyncio").coroutine(lambda: enriched)()):
        client = AstaClient(api_key="test-key")
        with patch.object(client, "_enrich_by_ids", AsyncMock(return_value=enriched)):
            results = await client.search("malaria vaccine")

    assert len(results) == 1
    assert results[0]["title"] == "Malaria vaccine"
    assert results[0]["paperId"] == "P1"


async def test_search_falls_back_to_semantic_scholar_on_asta_failure():
    ss_paper = {
        "paperId": "SS1",
        "title": "Semantic Scholar Paper",
        "year": 2021,
        "authors": [{"name": "Jones K"}],
        "abstract": "Abstract text.",
        "externalIds": {},
        "url": "https://semanticscholar.org/SS1",
    }

    with patch("src.core.agents.asta.ASTA_API_KEY", "test-key"), \
         patch.object(AstaClient, "_search_asta", new=AsyncMock(side_effect=RuntimeError("Asta down"))), \
         patch.object(AstaClient, "_search_semantic_scholar", new=AsyncMock(return_value=[_normalize_ss_paper(ss_paper)])):
        client = AstaClient(api_key="test-key")
        results = await client.search("any query")

    assert results[0]["paperId"] == "SS1"


async def test_search_uses_semantic_scholar_when_no_api_key():
    ss_paper = {
        "paperId": "SS2",
        "title": "Public Paper",
        "year": 2020,
        "authors": [],
        "abstract": "Some abstract.",
        "externalIds": {},
        "url": "",
    }

    with patch("src.core.agents.asta.ASTA_API_KEY", ""), \
         patch.object(AstaClient, "_search_semantic_scholar", new=AsyncMock(return_value=[_normalize_ss_paper(ss_paper)])):
        client = AstaClient()
        results = await client.search("q")

    assert results[0]["paperId"] == "SS2"


# ---------------------------------------------------------------------------
# _normalize_ss_paper
# ---------------------------------------------------------------------------


def test_normalize_ss_paper_all_fields():
    raw = {
        "paperId": "P1",
        "title": "Test paper",
        "year": 2023,
        "authors": [{"name": "Alice"}, {"name": "Bob"}],
        "abstract": "Some abstract text.",
        "externalIds": {"DOI": "10.1000/xyz"},
        "url": "https://example.com",
    }
    result = _normalize_ss_paper(raw)
    assert result["paperId"] == "P1"
    assert result["authors"] == "Alice, Bob"
    assert result["abstract"] == "Some abstract text."
    assert result["url"] == "https://example.com"
    assert result["source"] == "https://example.com"


def test_normalize_ss_paper_missing_optional_fields():
    raw = {"paperId": "P2"}
    result = _normalize_ss_paper(raw)
    assert result["paperId"] == "P2"
    assert result["title"] == ""
    assert result["year"] is None
    assert result["abstract"] == ""
    assert result["externalIds"] == {}
