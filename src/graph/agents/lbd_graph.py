"""Backward-compatibility shim — implementation moved to src/graph/subgraphs/lbd.py.

Import from src.graph.subgraphs.lbd for new code.
"""
from src.graph.subgraphs.lbd import (  # noqa: F401
    _parse_concepts,
    _parse_papers_from_content,
    _route_lbd,
    build_lbd_graph,
    lbd_agent,
    lbd_collect_papers,
    lbd_discover_connections,
    lbd_finalise,
    lbd_graph,
    lbd_synthesise,
    lbd_tools,
)

__all__ = [
    "_parse_concepts",
    "_parse_papers_from_content",
    "_route_lbd",
    "build_lbd_graph",
    "lbd_agent",
    "lbd_collect_papers",
    "lbd_discover_connections",
    "lbd_finalise",
    "lbd_graph",
    "lbd_synthesise",
    "lbd_tools",
]
