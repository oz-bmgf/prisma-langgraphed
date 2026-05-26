"""QdrantSearchBackend — async SearchBackend backed by a Qdrant vector database.

Requires a running Qdrant server. Supports named dense + BM25 sparse vectors
uploaded via the old scripts/upload_to_qdrant.py (or equivalent). When both
are present, uses Qdrant's native server-side RRF fusion. Falls back to
dense-only retrieval when sparse vectors are absent.

Env vars (read by factory.py):
    QDRANT_HOST, QDRANT_PORT, QDRANT_URL, QDRANT_API_KEY
    QDRANT_COLLECTION_NAME   — override the collection name
    QDRANT_EMBEDDING_MODEL   — override the embedding model
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional

from src.backends.base import RECENCY_BOOST_DOC_TYPES, SearchBackend, SearchResult
from src.config import (
    DEFAULT_EMBEDDING_MODEL,
    EMBED_DIM,
    RECENCY_BOOST_PER_YEAR as _RECENCY_BOOST_PER_YEAR,
    RECENCY_BASELINE_YEAR as _RECENCY_BASELINE_YEAR,
)

logger = logging.getLogger(__name__)


class QdrantSearchBackend:
    """Async SearchBackend wrapping the sync qdrant-client via asyncio.to_thread."""

    def __init__(
        self,
        collection_name: str,
        *,
        host: str = "localhost",
        port: int = 6333,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    ) -> None:
        self._collection_name = collection_name
        self._embedding_model = embedding_model

        from qdrant_client import QdrantClient
        if url:
            self._client = QdrantClient(url=url, api_key=api_key)
        else:
            self._client = QdrantClient(host=host, port=port, api_key=api_key)

        # Probe collection for capabilities
        self._dim: int = EMBED_DIM
        self._uses_named_vectors: bool = False
        self._has_sparse: bool = False
        try:
            info = self._client.get_collection(collection_name)
            params = info.config.params
            vc = params.vectors
            sparse_vc = getattr(params, "sparse_vectors", None)

            if isinstance(vc, dict) and "dense" in vc:
                self._dim = vc["dense"].size
                self._uses_named_vectors = True
            elif isinstance(vc, dict) and vc:
                self._dim = getattr(next(iter(vc.values())), "size", EMBED_DIM)
                self._uses_named_vectors = True
            elif hasattr(vc, "size"):
                self._dim = vc.size
                self._uses_named_vectors = False

            if isinstance(sparse_vc, dict):
                self._has_sparse = "bm25" in sparse_vc
            elif sparse_vc is not None:
                logger.warning(
                    "Unrecognized sparse_vectors shape (%s) for %s — dense-only",
                    type(sparse_vc).__name__, collection_name,
                )
        except Exception as exc:
            logger.warning("Qdrant collection probe failed (%s) — using defaults", exc)

        logger.info(
            "QdrantSearchBackend: collection=%s dim=%d named=%s sparse=%s",
            collection_name, self._dim, self._uses_named_vectors, self._has_sparse,
        )

        # Lazy cache for distinct-value methods
        self._payload_cache: Optional[dict] = None

        # Lazy OpenAI + sparse model clients
        self._openai_client: Any = None
        self._sparse_model: Any = None

    # ------------------------------------------------------------------
    # Embedding helpers
    # ------------------------------------------------------------------

    def _embed_query(self, query: str) -> list[float]:
        import time as _time
        from openai import RateLimitError
        if self._openai_client is None:
            from openai import OpenAI
            self._openai_client = OpenAI()
        for attempt in range(5):
            try:
                resp = self._openai_client.embeddings.create(
                    input=[query[:30000]], model=self._embedding_model
                )
                return resp.data[0].embedding
            except RateLimitError:
                if attempt == 4:
                    raise
                wait = 2 ** attempt
                logger.warning("Rate limited on embedding, retrying in %ds", wait)
                _time.sleep(wait)
        raise RuntimeError("unreachable")

    def _embed_query_sparse(self, query: str):
        if self._sparse_model is None:
            from fastembed import SparseTextEmbedding
            self._sparse_model = SparseTextEmbedding("Qdrant/bm25")
        return next(iter(self._sparse_model.embed([query])))

    # ------------------------------------------------------------------
    # Filter construction
    # ------------------------------------------------------------------

    def _build_filter(
        self,
        *,
        collection: Optional[str],
        bow_id: Optional[str],
        inv_id: Optional[str],
        doc_type: Optional[str],
    ):
        from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue

        must: list = []
        if collection and collection != "all":
            must.append(FieldCondition(key="collection", match=MatchValue(value=collection)))
        if bow_id:
            must.append(Filter(should=[
                FieldCondition(key="bow_id", match=MatchValue(value=bow_id)),
                FieldCondition(key="supports_bow_ids", match=MatchAny(any=[bow_id])),
            ]))
        if inv_id:
            must.append(FieldCondition(key="inv_id", match=MatchValue(value=inv_id)))
        if doc_type:
            must.append(FieldCondition(key="doc_type", match=MatchValue(value=doc_type)))
        return Filter(must=must) if must else None

    # ------------------------------------------------------------------
    # Result construction
    # ------------------------------------------------------------------

    @staticmethod
    def _point_to_result(point, score: float) -> SearchResult:
        pl = point.payload or {}
        return SearchResult(
            chunk_id=pl.get("chunk_id", ""),
            text=pl.get("text", ""),
            score=score,
            file_id=pl.get("file_id", ""),
            inv_id=pl.get("inv_id") or None,
            bow_id=pl.get("bow_id") or None,
            page_start=pl.get("page_start"),
            page_end=pl.get("page_end"),
            doc_type=pl.get("doc_type") or None,
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
        bow_id_filter: Optional[str],
        inv_id_filter: Optional[str],
        doc_type_filter: Optional[str],
    ) -> list[SearchResult]:
        qfilter = self._build_filter(
            collection=collection_filter,
            bow_id=bow_id_filter,
            inv_id=inv_id_filter,
            doc_type=doc_type_filter,
        )

        if self._has_sparse:
            from qdrant_client.models import Fusion, FusionQuery, Prefetch, SparseVector

            dense_vec = self._embed_query(query)
            sparse_emb = self._embed_query_sparse(query)
            sparse_vec = SparseVector(
                indices=list(sparse_emb.indices),
                values=list(sparse_emb.values),
            )
            response = self._client.query_points(
                collection_name=self._collection_name,
                prefetch=[
                    Prefetch(query=dense_vec, using="dense", limit=top_k * 4, filter=qfilter),
                    Prefetch(query=sparse_vec, using="bm25",  limit=top_k * 4, filter=qfilter),
                ],
                query=FusionQuery(fusion=Fusion.RRF),
                limit=top_k,
                with_payload=True,
            )
            candidates = [self._point_to_result(h, h.score) for h in response.points]
        else:
            dense_vec = self._embed_query(query)
            response = self._client.query_points(
                collection_name=self._collection_name,
                query=dense_vec,
                using="dense" if self._uses_named_vectors else None,
                query_filter=qfilter,
                limit=top_k,
                with_payload=True,
            )
            candidates = [self._point_to_result(h, h.score) for h in response.points]

        # Recency boost (additive, keeps parity with LocalSearchIndex)
        scored: list[tuple[float, SearchResult]] = []
        for r in candidates:
            score = r.score
            if r.doc_type and r.doc_type in RECENCY_BOOST_DOC_TYPES:
                year = _extract_year(r.file_id)
                if year:
                    score += _RECENCY_BOOST_PER_YEAR * max(0, year - _RECENCY_BASELINE_YEAR)
            scored.append((score, r))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            SearchResult(
                chunk_id=r.chunk_id, text=r.text, score=s, file_id=r.file_id,
                inv_id=r.inv_id, bow_id=r.bow_id, page_start=r.page_start,
                page_end=r.page_end, doc_type=r.doc_type,
            )
            for s, r in scored
        ]

    # ------------------------------------------------------------------
    # Payload cache for distinct-value queries
    # ------------------------------------------------------------------

    def _ensure_cache_sync(self) -> None:
        if self._payload_cache is not None:
            return
        inv_ids: set[str] = set()
        bow_ids: set[str] = set()
        bow_counts: dict[str, int] = {}
        offset = None
        while True:
            results, next_offset = self._client.scroll(
                self._collection_name,
                limit=1000,
                with_payload=["inv_id", "bow_id", "supports_bow_ids"],
                with_vectors=False,
                offset=offset,
            )
            for point in results:
                pl = point.payload or {}
                inv_id = pl.get("inv_id") or ""
                bow_id = pl.get("bow_id") or ""
                supports = pl.get("supports_bow_ids") or []
                if inv_id:
                    inv_ids.add(inv_id)
                if bow_id:
                    bow_ids.add(bow_id)
                    bow_counts[bow_id] = bow_counts.get(bow_id, 0) + 1
                if isinstance(supports, list):
                    for bid in supports:
                        if isinstance(bid, str) and bid:
                            bow_counts[bid] = bow_counts.get(bid, 0) + 1
            if next_offset is None:
                break
            offset = next_offset
        self._payload_cache = {
            "inv_ids": sorted(inv_ids),
            "bow_ids": sorted(bow_ids),
            "bow_counts": bow_counts,
        }

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
        # asyncio-APPROVED-1: to_thread wraps blocking qdrant-client search call
        return await asyncio.to_thread(
            self._search_sync,
            query,
            top_k=top_k,
            collection_filter=collection_filter,
            bow_id_filter=bow_id_filter,
            inv_id_filter=inv_id_filter,
            doc_type_filter=doc_type_filter,
        )

    async def distinct_inv_ids(self) -> list[str]:
        # asyncio-APPROVED-1: to_thread wraps blocking qdrant-client payload cache build
        await asyncio.to_thread(self._ensure_cache_sync)
        return self._payload_cache["inv_ids"]  # type: ignore[index]

    async def distinct_bow_ids(self) -> list[str]:
        # asyncio-APPROVED-1: to_thread wraps blocking qdrant-client payload cache build
        await asyncio.to_thread(self._ensure_cache_sync)
        return self._payload_cache["bow_ids"]  # type: ignore[index]

    async def count_by_bow_id(self) -> dict[str, int]:
        # asyncio-APPROVED-1: to_thread wraps blocking qdrant-client payload cache build
        await asyncio.to_thread(self._ensure_cache_sync)
        return self._payload_cache["bow_counts"]  # type: ignore[index]


def _extract_year(s: str) -> Optional[int]:
    import datetime
    import re
    max_year = datetime.datetime.now().year + 1
    for m in re.finditer(r"\b(2\d{3})\b", s):
        y = int(m.group(1))
        if 2000 <= y <= max_year:
            return y
    return None
