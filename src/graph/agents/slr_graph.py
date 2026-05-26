"""Backward-compatibility shim — implementation moved to src/graph/subgraphs/slr.py.

Import from src.graph.subgraphs.slr for new code.
"""
from src.graph.subgraphs.slr import (  # noqa: F401
    build_slr_graph,
    slr_collect_papers,
    slr_expand_queries,
    slr_fetch_source,
    slr_finalise,
    slr_graph,
    slr_plan_sources,
    slr_start,
    slr_synthesise,
)

__all__ = [
    "build_slr_graph",
    "slr_collect_papers",
    "slr_expand_queries",
    "slr_fetch_source",
    "slr_finalise",
    "slr_graph",
    "slr_plan_sources",
    "slr_start",
    "slr_synthesise",
]
