"""Backward-compatibility shim — implementation moved to src/graph/subgraphs/slr.py.

Import from src.graph.subgraphs.slr for new code.
"""
from src.graph.subgraphs.slr import (  # noqa: F401
    _parse_papers_from_content,
    _route_slr,
    build_slr_graph,
    slr_agent,
    slr_collect_papers,
    slr_finalise,
    slr_graph,
    slr_synthesise,
    slr_tools,
)

__all__ = [
    "_parse_papers_from_content",
    "_route_slr",
    "build_slr_graph",
    "slr_agent",
    "slr_collect_papers",
    "slr_finalise",
    "slr_graph",
    "slr_synthesise",
    "slr_tools",
]
