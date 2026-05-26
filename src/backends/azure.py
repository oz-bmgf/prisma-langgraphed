"""AzureSearchBackend — async SearchBackend backed by Azure AI Search.

Targets the foundation-wide edp-idm-index. Each instance is a team view
scoped to managingTeamL1/L2. Pass team="" to query foundation-wide.

Env vars (read by factory.py):
    AZURE_SEARCH_INDEX   — override default index name (edp-idm-index)
    AZURE_SEARCH_TEAM    — override team filter (defaults to collection_name)

DNS note: the WUS2 private endpoint sometimes resolves via the public DNS
on macOS when Warp/VPN races. We pin known private IPs at module import.
VPN must be active for live queries; tests mock the SDK and bypass this.
"""
from __future__ import annotations

import asyncio
import logging
import socket
from typing import Optional

from src.backends.base import SearchBackend, SearchResult
from src.config import DEFAULT_EMBEDDING_MODEL

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DNS pin — applied once at import time
# ---------------------------------------------------------------------------

_PRIVATE_IPS: dict[str, str] = {
    "ais-eds-edp-idm-prd-wus2.search.windows.net": "10.66.149.68",
    "aisedpidmprodwus201.blob.core.windows.net": "10.66.149.71",
    "dlsedpprdwus202.blob.core.windows.net": "10.66.89.80",
}
_real_getaddrinfo = socket.getaddrinfo


def _pinned_getaddrinfo(host, *args, **kwargs):
    pinned = _PRIVATE_IPS.get(host)
    if pinned:
        return _real_getaddrinfo(pinned, *args, **kwargs)
    return _real_getaddrinfo(host, *args, **kwargs)


if socket.getaddrinfo is not _pinned_getaddrinfo:
    socket.getaddrinfo = _pinned_getaddrinfo

# ---------------------------------------------------------------------------
# Module-level singletons
# ---------------------------------------------------------------------------

_CREDENTIAL = None
_SEARCH_CLIENTS: dict[str, object] = {}


def _credential():
    global _CREDENTIAL
    if _CREDENTIAL is None:
        from azure.identity import DefaultAzureCredential
        _CREDENTIAL = DefaultAzureCredential()
    return _CREDENTIAL


def _search_client(endpoint: str, index_name: str):
    key = f"{endpoint}|{index_name}"
    client = _SEARCH_CLIENTS.get(key)
    if client is None:
        from azure.search.documents import SearchClient
        client = SearchClient(endpoint=endpoint, index_name=index_name, credential=_credential())
        _SEARCH_CLIENTS[key] = client
    return client


# ---------------------------------------------------------------------------
# OData helpers
# ---------------------------------------------------------------------------

def _odata_escape(s: str) -> str:
    return s.replace("'", "''")


# ---------------------------------------------------------------------------
# Team aliases: short name → Azure canonical managingTeamL1 value
# ---------------------------------------------------------------------------

_TEAM_ALIASES: dict[str, str] = {
    "dts": "Discovery and Translational Sciences",
    "ntd": "Neglected Tropical Diseases",
    "ppp": "Pneumonia & Pandemic Preparedness",
    "egh": "Exemplars in Global Health",
    "vdev": "Vaccine Development",
    "vhi": "Vaccines & Human Immunobiology",
    "rht": "Reproductive Health Technologies",
    "whi": "Women's Health Innovations",
    "mncnh": "Maternal, Newborn, Child Nutrition and Health",
    "edge": "Enterics, Diagnostics, Genomics & Epidemiology",
    "fp": "Family Planning",
    "wash": "Water, Sanitation, and Hygiene",
    "ghoop": "GH Office of the President",
    "ggoop": "GGO Office of the President",
    "oop": "Office of the President",
    "ifs": "Inclusive Financial Systems",
    "dpf": "Development Policy & Finance",
    "pac": "Program Advocacy & Comms",
    "dpi": "Digital Public Infrastructure",
    "usp": "USP Data",
    "usemo": "U.S. Economic Mobility & Opportunity",
    "pss": "Postsecondary Success",
    "el": "Early Learning",
    "geo": "Global Education",
    "ai": "Assessment Initiative",
    "idev": "Integrated Development",
}


def _resolve_team_name(team: str) -> str:
    if not team:
        return team
    short_match = _TEAM_ALIASES.get(team.lower())
    if short_match:
        return short_match
    canonical_by_lower = {v.lower(): v for v in _TEAM_ALIASES.values()}
    canonical = canonical_by_lower.get(team.lower())
    if canonical and canonical != team:
        logger.info("AzureSearchBackend: normalized team casing %r → %r", team, canonical)
        return canonical
    return team


# ---------------------------------------------------------------------------
# Fields to request from Azure AI Search
# ---------------------------------------------------------------------------

_SELECT_FIELDS = [
    "chunk_id", "parent_id", "fileName", "sourceUrl",
    "investmentId", "investmentName", "managingTeamL1", "managingTeamL2",
    "lastModified", "chunk",
]
_VECTOR_K_NEAREST = 50


# ---------------------------------------------------------------------------
# AzureSearchBackend
# ---------------------------------------------------------------------------

class AzureSearchBackend:
    """Async SearchBackend wrapping Azure AI Search via asyncio.to_thread.

    Each instance is a team-scoped view. Pass strict=True (default) for the
    primary backend — errors raise rather than returning empty. Pass
    strict=False for aux-corpus fanout where a transient failure should not
    abort the run.
    """

    DEFAULT_ENDPOINT_HOST = "ais-eds-edp-idm-prd-wus2.search.windows.net"
    DEFAULT_INDEX = "edp-idm-index"

    def __init__(
        self,
        team: str = "",
        *,
        index_name: str = DEFAULT_INDEX,
        endpoint_host: str = DEFAULT_ENDPOINT_HOST,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        strict: bool = True,
    ) -> None:
        self._strict = strict
        self._team = _resolve_team_name(team or "")
        self._index_name = index_name
        self._endpoint = f"https://{endpoint_host}"
        self._embedding_model = embedding_model
        logger.info(
            "AzureSearchBackend: index=%s team=%s endpoint=%s",
            index_name, self._team or "(all teams)", endpoint_host,
        )

    # ------------------------------------------------------------------
    # OData filter builders
    # ------------------------------------------------------------------

    def _team_filter(self) -> Optional[str]:
        """OR of L1 and L2 to handle the dual-taxonomy indexing bug."""
        if not self._team:
            return None
        escaped = _odata_escape(self._team)
        return f"(managingTeamL1 eq '{escaped}' or managingTeamL2 eq '{escaped}')"

    def _build_filter(
        self,
        *,
        collection: Optional[str],
        inv_id: Optional[str],
    ) -> Optional[str]:
        parts: list[str] = []
        team_clause = self._team_filter()
        if team_clause:
            parts.append(team_clause)
        if inv_id:
            parts.append(f"investmentId eq '{_odata_escape(inv_id)}'")
        if collection == "investment":
            parts.append("search.ismatch('splibraw/invest/*', 'sourceUrl')")
        elif collection == "strategy":
            parts.append("search.ismatch('splibraw/insight/*', 'sourceUrl')")
        return " and ".join(parts) if parts else None

    # ------------------------------------------------------------------
    # Result mapping
    # ------------------------------------------------------------------

    @staticmethod
    def _derive_collection(source_url: Optional[str]) -> str:
        if not source_url:
            return ""
        if "splibraw/insight/" in source_url:
            return "strategy"
        if "splibraw/invest/" in source_url:
            return "investment"
        return ""

    @staticmethod
    def _row_to_result(row: dict) -> SearchResult:
        score = row.get("@search.reranker_score")
        if score is None:
            score = row.get("@search.rerankerScore")
        if score is None:
            score = row.get("@search.score") or 0.0
        return SearchResult(
            chunk_id=row.get("chunk_id") or "",
            text=row.get("chunk") or "",
            score=float(score),
            file_id=row.get("parent_id") or "",
            inv_id=row.get("investmentId") or None,
            bow_id=None,
            page_start=None,
            page_end=None,
            doc_type=None,
        )

    # ------------------------------------------------------------------
    # Sync search core
    # ------------------------------------------------------------------

    def _search_sync(
        self,
        query: str,
        *,
        top_k: int,
        collection_filter: Optional[str],
        inv_id_filter: Optional[str],
        bow_id_filter: Optional[str],
        doc_type_filter: Optional[str],
    ) -> list[SearchResult]:
        if bow_id_filter or doc_type_filter:
            logger.debug(
                "AzureSearchBackend: bow_id_filter and doc_type_filter not supported; ignored"
            )
        filter_expr = self._build_filter(collection=collection_filter, inv_id=inv_id_filter)
        effective_filter = filter_expr if filter_expr is not None else self._team_filter()
        try:
            from azure.search.documents.models import QueryType, VectorizableTextQuery

            client = _search_client(self._endpoint, self._index_name)
            results = client.search(
                search_text=query,
                vector_queries=[
                    VectorizableTextQuery(
                        text=query,
                        k_nearest_neighbors=_VECTOR_K_NEAREST,
                        fields="chunkVector",
                    )
                ],
                query_type=QueryType.SEMANTIC,
                semantic_configuration_name="default",
                top=top_k,
                select=_SELECT_FIELDS,
                filter=effective_filter,
            )
            return [self._row_to_result(row) for row in results]
        except Exception as exc:
            if self._strict:
                raise
            logger.warning(
                "AzureSearchBackend: search failed (%s) — returning empty (strict=False)",
                exc,
            )
            return []

    # ------------------------------------------------------------------
    # Async public interface (SearchBackend protocol)
    # ------------------------------------------------------------------

    async def search(
        self,
        query: str,
        *,
        top_k: int = 20,
        collection_filter: Optional[str] = None,
        bow_id_filter: Optional[str] = None,
        inv_id_filter: Optional[str] = None,
        doc_type_filter: Optional[str] = None,
    ) -> list[SearchResult]:
        # asyncio-APPROVED-1: to_thread wraps blocking Azure AI Search SDK call
        return await asyncio.to_thread(
            self._search_sync,
            query,
            top_k=top_k,
            collection_filter=collection_filter,
            inv_id_filter=inv_id_filter,
            bow_id_filter=bow_id_filter,
            doc_type_filter=doc_type_filter,
        )

    async def distinct_inv_ids(self) -> list[str]:
        self._deferred_check("distinct_inv_ids")
        return []

    async def distinct_bow_ids(self) -> list[str]:
        self._deferred_check("distinct_bow_ids")
        return []

    async def count_by_bow_id(self) -> dict[str, int]:
        self._deferred_check("count_by_bow_id")
        return {}

    def _deferred_check(self, method: str) -> None:
        if self._strict:
            raise NotImplementedError(
                f"AzureSearchBackend.{method} is not implemented for the remote backend. "
                "Use LocalSearchIndex as the primary backend, or implement via Azure facets."
            )
        logger.warning(
            "AzureSearchBackend.%s: not implemented; returning empty (strict=False)",
            method,
        )
