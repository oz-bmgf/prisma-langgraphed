"""Shared utilities for research subgraph nodes.

Extracted from LBD and SLR subgraphs where identical logic was duplicated.
"""
from __future__ import annotations


def deduplicate_papers(papers: list[dict]) -> list[dict]:
    """Deduplicate papers by title (case-insensitive). Preserves first-seen order."""
    seen: set[str] = set()
    unique: list[dict] = []
    for p in papers:
        key = (p.get("title") or "").lower().strip()
        if key and key not in seen:
            seen.add(key)
            unique.append(p)
    return unique
