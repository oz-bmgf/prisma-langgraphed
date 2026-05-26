from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, runtime_checkable


@dataclass
class SearchResult:
    chunk_id: str
    text: str
    score: float
    file_id: str
    inv_id: Optional[str]
    bow_id: Optional[str]
    page_start: Optional[int]
    page_end: Optional[int]
    doc_type: Optional[str]


@runtime_checkable
class SearchBackend(Protocol):
    async def search(
        self,
        query: str,
        *,
        top_k: int = 20,
        collection_filter: Optional[str] = None,
        bow_id_filter: Optional[str] = None,
        inv_id_filter: Optional[str] = None,
        doc_type_filter: Optional[str] = None,
    ) -> list[SearchResult]: ...

    async def distinct_inv_ids(self) -> list[str]: ...
    async def distinct_bow_ids(self) -> list[str]: ...
    async def count_by_bow_id(self) -> dict[str, int]: ...


# Doc types where recency matters — shared constant so all backends apply the
# same boost heuristic.
RECENCY_BOOST_DOC_TYPES: frozenset[str] = frozenset(
    {"progress_report", "final_report", "amendment", "milestone"}
)
