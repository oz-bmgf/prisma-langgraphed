"""Backward-compatibility shim — implementation moved to src/graph/subgraphs/edison.py.

Import from src.graph.subgraphs.edison for new code.
"""
from src.graph.subgraphs.edison import (  # noqa: F401
    build_edison_graph,
    edison_finalise,
    edison_graph,
    edison_rewrite_query,
    edison_search,
    route_rewrite_entry,
)

__all__ = [
    "build_edison_graph",
    "edison_finalise",
    "edison_graph",
    "edison_rewrite_query",
    "edison_search",
    "route_rewrite_entry",
]
