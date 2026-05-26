"""build_search_backend — factory for SearchBackend instances.

Reads NQPR_SEARCH_BACKEND from the environment to select the concrete
backend. Defaults to "local".

Usage at graph startup:
    backend = build_search_backend(ingested_dir, collection_name)
    config = {"configurable": {"search_backend": backend}}
    await graph.ainvoke(initial_state, config)

Inside any @tool function:
    backend: SearchBackend = config["configurable"]["search_backend"]
    results = await backend.search(query, top_k=20, bow_id_filter=bow_id)
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

from src.backends.base import SearchBackend
from src.config import DEFAULT_EMBEDDING_MODEL, SEARCH_BACKEND as _DEFAULT_BACKEND

logger = logging.getLogger(__name__)


def build_search_backend(
    ingested_dir: str,
    collection_name: str,
    *,
    aux_team: Optional[str] = None,
) -> SearchBackend:
    """Return a SearchBackend for the given collection.

    Args:
        ingested_dir:     Absolute path to {program}-ingested/ directory.
        collection_name:  Logical collection name (used for Qdrant/Azure routing).
        aux_team:         When building an aux backend, pass the short or
                          canonical team name. Used by Azure only; ignored for
                          local/qdrant.
    """
    backend_type = os.environ.get("NQPR_SEARCH_BACKEND", _DEFAULT_BACKEND).lower()
    logger.info("Building SearchBackend: type=%s collection=%s", backend_type, collection_name)

    if backend_type == "local":
        from src.backends.local import LocalSearchIndex

        index_dir = Path(ingested_dir) / "embedding_index"
        return LocalSearchIndex(index_dir)

    if backend_type == "qdrant":
        from src.backends.qdrant import QdrantSearchBackend

        coll = os.environ.get("QDRANT_COLLECTION_NAME", collection_name)
        url = os.environ.get("QDRANT_URL") or None
        host = os.environ.get("QDRANT_HOST", "localhost")
        port = int(os.environ.get("QDRANT_PORT", "6333"))
        api_key = os.environ.get("QDRANT_API_KEY") or None
        model = os.environ.get("QDRANT_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
        return QdrantSearchBackend(
            coll, url=url, host=host, port=port, api_key=api_key, embedding_model=model,
        )

    if backend_type == "azure":
        from src.backends.azure import AzureSearchBackend

        team = aux_team or os.environ.get("AZURE_SEARCH_TEAM", collection_name)
        index_name = os.environ.get("AZURE_SEARCH_INDEX", "edp-idm-index")
        endpoint_host = os.environ.get(
            "AZURE_SEARCH_ENDPOINT_HOST",
            AzureSearchBackend.DEFAULT_ENDPOINT_HOST,
        )
        return AzureSearchBackend(
            team, index_name=index_name, endpoint_host=endpoint_host,
        )

    raise ValueError(
        f"Unknown NQPR_SEARCH_BACKEND={backend_type!r}. "
        "Valid values: 'local' (default), 'qdrant', 'azure'."
    )
