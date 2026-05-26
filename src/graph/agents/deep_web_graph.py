"""Backward-compatibility shim — implementation moved to src/graph/subgraphs/deep_web.py.

Import from src.graph.subgraphs.deep_web for new code.
"""
from src.graph.subgraphs.deep_web import (  # noqa: F401
    build_deep_web_graph,
    deep_web_collect_rounds,
    deep_web_dispatch_rounds,
    deep_web_finalise,
    deep_web_graph,
    deep_web_route_after_primary,
    deep_web_search_round,
    deep_web_synthesise_fallback,
    deep_web_try_primary,
)

__all__ = [
    "build_deep_web_graph",
    "deep_web_collect_rounds",
    "deep_web_dispatch_rounds",
    "deep_web_finalise",
    "deep_web_graph",
    "deep_web_route_after_primary",
    "deep_web_search_round",
    "deep_web_synthesise_fallback",
    "deep_web_try_primary",
]
