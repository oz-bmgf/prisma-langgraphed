"""Backward-compatibility shim — implementation moved to src/graph/subgraphs/lbd.py.

Import from src.graph.subgraphs.lbd for new code.
"""
from src.graph.subgraphs.lbd import (  # noqa: F401
    _parse_concepts,
    build_lbd_graph,
    lbd_broad_search,
    lbd_collect_concept_papers,
    lbd_dispatch_concepts,
    lbd_discover_connections,
    lbd_extract_concepts,
    lbd_fetch_concept_papers,
    lbd_finalise,
    lbd_graph,
    lbd_start,
    lbd_synthesise,
)

__all__ = [
    "_parse_concepts",
    "build_lbd_graph",
    "lbd_broad_search",
    "lbd_collect_concept_papers",
    "lbd_dispatch_concepts",
    "lbd_discover_connections",
    "lbd_extract_concepts",
    "lbd_fetch_concept_papers",
    "lbd_finalise",
    "lbd_graph",
    "lbd_start",
    "lbd_synthesise",
]
