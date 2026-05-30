"""LocalSearchIndex — async wrapper over SQLite metadata + memmap'd numpy vectors.

On-disk layout (under index_dir/):
    chunks.sqlite  — per-chunk metadata with B-tree indexes for fast filter
    vectors.npy    — numpy float32 (N, dim), memmap'd at load time
    norms.npy      — pre-computed L2 norms, memmap'd at load time
    config.json    — model name, n_chunks, dim
    chunks.json    — canonical JSON export (kept for forensics and migration)

All public methods are async and wrap sync numpy/SQLite operations via
asyncio.to_thread. The index is lazy-loaded on first use; __init__ is
instantaneous and does no I/O.
"""
from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import threading
from pathlib import Path
from typing import Any, Optional

from src.backends.base import RECENCY_BOOST_DOC_TYPES, SearchBackend, SearchResult
from src.config import (
    DEFAULT_EMBEDDING_MODEL,
    RECENCY_BOOST_PER_YEAR as _RECENCY_BOOST_PER_YEAR,
    RECENCY_BASELINE_YEAR as _RECENCY_BASELINE_YEAR,
)

logger = logging.getLogger(__name__)

# SQLite column order — must match the schema; we index positionally for speed.
_CHUNK_COLUMNS = [
    "idx", "chunk_id", "file_id", "filename", "collection",
    "doc_type", "raw_doc_type", "inv_id", "bow_id",
    "section_id", "section_label", "page_start", "page_end",
    "text", "context", "doc_date",
    "intelligence_role", "intelligence_action", "intelligence_is_best",
    "intelligence_version_group", "intelligence_pass_status",
    "content_tags", "supports_bow_ids", "topic_tags", "carve_out_metadata",
]
_COL_LIST = ", ".join(_CHUNK_COLUMNS)

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS chunks (
    idx                          INTEGER PRIMARY KEY,
    chunk_id                     TEXT UNIQUE NOT NULL,
    file_id                      TEXT NOT NULL,
    filename                     TEXT NOT NULL,
    collection                   TEXT NOT NULL,
    doc_type                     TEXT,
    raw_doc_type                 TEXT,
    inv_id                       TEXT,
    bow_id                       TEXT,
    section_id                   TEXT,
    section_label                TEXT,
    page_start                   INTEGER,
    page_end                     INTEGER,
    text                         TEXT,
    context                      TEXT,
    doc_date                     TEXT,
    intelligence_role            TEXT,
    intelligence_action          TEXT,
    intelligence_is_best         INTEGER,
    intelligence_version_group   TEXT,
    intelligence_pass_status     TEXT,
    content_tags                 TEXT,
    supports_bow_ids             TEXT,
    topic_tags                   TEXT,
    carve_out_metadata           TEXT
);
CREATE INDEX IF NOT EXISTS idx_chunks_inv        ON chunks(inv_id);
CREATE INDEX IF NOT EXISTS idx_chunks_bow        ON chunks(bow_id);
CREATE INDEX IF NOT EXISTS idx_chunks_doc_type   ON chunks(doc_type);
CREATE INDEX IF NOT EXISTS idx_chunks_collection ON chunks(collection);
CREATE INDEX IF NOT EXISTS idx_chunks_compound   ON chunks(collection, inv_id, doc_type);
CREATE INDEX IF NOT EXISTS idx_chunks_file_id    ON chunks(file_id);
"""


# ---------------------------------------------------------------------------
# Schema migrations (idempotent; run once on first load of older indexes)
# ---------------------------------------------------------------------------

def _migrate_intelligence_label_to_role(sqlite_path: Path) -> None:
    """Rename intelligence_label → intelligence_role if still present."""
    if not sqlite_path.exists():
        return
    conn = sqlite3.connect(str(sqlite_path))
    try:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(chunks)").fetchall()}
        if "intelligence_role" in cols:
            return
        if "intelligence_label" in cols:
            conn.execute("ALTER TABLE chunks RENAME COLUMN intelligence_label TO intelligence_role")
            conn.commit()
            logger.info("Migrated intelligence_label → intelligence_role at %s", sqlite_path)
    finally:
        conn.close()


def _migrate_add_strategy_columns(sqlite_path: Path) -> None:
    """Add supports_bow_ids + topic_tags columns if absent (v3 PR2)."""
    if not sqlite_path.exists():
        return
    conn = sqlite3.connect(str(sqlite_path))
    try:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(chunks)").fetchall()}
        added = []
        for col in ("supports_bow_ids", "topic_tags"):
            if col not in cols:
                conn.execute(f"ALTER TABLE chunks ADD COLUMN {col} TEXT")
                added.append(col)
        if added:
            conn.commit()
            logger.info("Added columns %s to %s", added, sqlite_path)
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Row → SearchResult
# ---------------------------------------------------------------------------

def _row_to_result(row: tuple, score: float) -> SearchResult:
    """Map a SQLite row (in _CHUNK_COLUMNS order) to a SearchResult."""
    # positional indices per _CHUNK_COLUMNS
    return SearchResult(
        chunk_id=row[1],
        text=row[13] or "",
        score=score,
        file_id=row[2],
        inv_id=row[7] or None,
        bow_id=row[8] or None,
        page_start=row[11],
        page_end=row[12],
        doc_type=row[5] or None,
    )


# ---------------------------------------------------------------------------
# LocalSearchIndex
# ---------------------------------------------------------------------------

class LocalSearchIndex:
    """Async SearchBackend backed by SQLite + memmap'd numpy vectors.

    Lazy-loaded: __init__ does no I/O. The index is loaded on the first
    search call, inside asyncio.to_thread.
    """

    def __init__(self, index_dir: Path) -> None:
        self._index_dir = Path(index_dir)
        self._loaded = False
        self._load_lock = threading.Lock()

        # Populated by _do_load()
        self._sqlite_path: Path | None = None
        self._vectors: Any = None   # numpy memmap or ndarray
        self._norms: Any = None     # numpy memmap or ndarray
        self._n_chunks: int = 0
        self._dim: int = 0
        self._model: str = DEFAULT_EMBEDDING_MODEL
        self._tfidf_vectorizer: Any = None
        self._tfidf_matrix: Any = None

        # Per-thread SQLite connections (sqlite3 is not thread-safe)
        self._conn_lock = threading.Lock()
        self._conns: dict[int, sqlite3.Connection] = {}

        # Lazy OpenAI client for query embedding
        self._openai_client: Any = None

    # ------------------------------------------------------------------
    # Lazy load
    # ------------------------------------------------------------------

    def _do_load(self) -> None:
        """Blocking load — called once from inside asyncio.to_thread."""
        import numpy as np

        path = self._index_dir
        sqlite_path = path / "chunks.sqlite"
        chunks_json = path / "chunks.json"

        if not sqlite_path.exists():
            if not chunks_json.exists():
                raise FileNotFoundError(
                    f"Neither {sqlite_path} nor {chunks_json} found; cannot load index."
                )
            _build_sqlite_from_json(chunks_json, sqlite_path)

        _migrate_intelligence_label_to_role(sqlite_path)
        _migrate_add_strategy_columns(sqlite_path)

        vectors_path = path / "vectors.npy"
        vectors = np.load(vectors_path, mmap_mode="r")

        # Integrity check: sqlite row count must match vectors shape.
        # After a rebuild, n_sqlite may still be < vectors.shape[0] due to duplicate
        # chunk_ids in chunks.json being silently dropped by INSERT OR IGNORE.  We
        # only re-trigger the rebuild when n_sqlite == 0 (completely empty) or when
        # n_sqlite > vectors.shape[0] (impossible for a clean index — rebuild to fix).
        con = sqlite3.connect(str(sqlite_path))
        n_sqlite = con.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        con.close()
        needs_rebuild = (n_sqlite == 0) or (n_sqlite > vectors.shape[0])
        if not needs_rebuild and n_sqlite != vectors.shape[0]:
            # Partial mismatch — likely duplicate chunk_ids in chunks.json were deduped.
            logger.warning(
                "chunks.sqlite has %d rows vs %d vectors — %d duplicate chunk_ids in "
                "chunks.json were deduped; proceeding with %d-row index",
                n_sqlite, vectors.shape[0], vectors.shape[0] - n_sqlite, n_sqlite,
            )
        if needs_rebuild:
            if not chunks_json.exists():
                raise RuntimeError(
                    f"chunks.sqlite has {n_sqlite} rows but vectors.npy has "
                    f"{vectors.shape[0]} and chunks.json is missing — cannot reconcile"
                )
            logger.warning(
                "chunks.sqlite/vectors.npy size mismatch (%d vs %d); rebuilding sqlite",
                n_sqlite, vectors.shape[0],
            )
            backup = sqlite_path.with_suffix(".sqlite.stale.bak")
            sqlite_path.rename(backup)
            _build_sqlite_from_json(chunks_json, sqlite_path)
            _migrate_intelligence_label_to_role(sqlite_path)
            _migrate_add_strategy_columns(sqlite_path)
            # Verify the rebuild produced rows — if still 0, chunks.json is likely corrupt.
            con2 = sqlite3.connect(str(sqlite_path))
            n_rebuilt = con2.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            con2.close()
            if n_rebuilt == 0:
                raise RuntimeError(
                    f"chunks.sqlite rebuild produced 0 rows — {chunks_json} may be corrupt"
                )

        # Norms — load or recompute
        norms_path = path / "norms.npy"
        norms = None
        if norms_path.exists():
            candidate = np.load(norms_path, mmap_mode="r")
            if candidate.shape[0] == vectors.shape[0]:
                norms = candidate
            else:
                logger.warning("norms.npy shape mismatch; recomputing")
        if norms is None:
            norms = np.linalg.norm(np.asarray(vectors), axis=1)
            norms[norms == 0] = 1.0
            np.save(norms_path, norms)
            norms = np.load(norms_path, mmap_mode="r")

        config_path = path / "config.json"
        config = json.loads(config_path.read_text()) if config_path.exists() else {}

        self._sqlite_path = sqlite_path
        self._vectors = vectors
        self._norms = norms
        self._n_chunks = int(config.get("n_chunks") or vectors.shape[0])
        self._dim = int(config.get("dim") or (vectors.shape[1] if vectors.ndim == 2 else 0))
        self._model = config.get("model", DEFAULT_EMBEDDING_MODEL)

        # Optional TF-IDF sidecar for hybrid search
        tfidf_dir = path.parent / "tfidf_index"
        if tfidf_dir.is_dir():
            self._load_tfidf(tfidf_dir)

        logger.info(
            "LocalSearchIndex loaded: %d chunks, dim=%d from %s",
            self._n_chunks, self._dim, path,
        )

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        with self._load_lock:
            if self._loaded:
                return
            self._do_load()
            self._loaded = True

    def _load_tfidf(self, tfidf_dir: Path) -> None:
        try:
            import joblib
            from scipy import sparse

            vec_path = tfidf_dir / "vectorizer.joblib"
            mat_path = tfidf_dir / "matrix.npz"
            if not vec_path.exists() or not mat_path.exists():
                return
            self._tfidf_vectorizer = joblib.load(vec_path)
            self._tfidf_matrix = sparse.load_npz(mat_path)
            if self._tfidf_matrix.shape[0] != self._n_chunks:
                logger.warning(
                    "TF-IDF matrix rows (%d) != chunks (%d), disabling",
                    self._tfidf_matrix.shape[0], self._n_chunks,
                )
                self._tfidf_vectorizer = self._tfidf_matrix = None
        except Exception as exc:
            logger.warning("TF-IDF load failed (hybrid disabled): %s", exc)

    # ------------------------------------------------------------------
    # SQLite connection (per-thread)
    # ------------------------------------------------------------------

    def _conn(self) -> sqlite3.Connection:
        tid = threading.get_ident()
        with self._conn_lock:
            conn = self._conns.get(tid)
            if conn is None:
                conn = sqlite3.connect(str(self._sqlite_path))
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA synchronous=NORMAL")
                self._conns[tid] = conn
            return conn

    # ------------------------------------------------------------------
    # Filter → numpy index array
    # ------------------------------------------------------------------

    def _filter_indices(
        self,
        *,
        collection: Optional[str],
        bow_id: Optional[str],
        inv_id: Optional[str],
        doc_type: Optional[str],
    ):
        import numpy as np

        clauses: list[str] = []
        params: list[Any] = []

        if collection and collection != "all":
            clauses.append("collection = ?")
            params.append(collection)

        if bow_id:
            # Inclusive: matches strict bow_id OR cross-cutting supports_bow_ids
            clauses.append("(bow_id = ? OR supports_bow_ids LIKE ?)")
            params.append(bow_id)
            params.append(f'%"{bow_id}"%')

        if inv_id:
            clauses.append("inv_id = ?")
            params.append(inv_id)

        if doc_type:
            clauses.append("doc_type = ?")
            params.append(doc_type)

        if not clauses:
            return np.arange(self._n_chunks, dtype=np.int64)

        sql = "SELECT idx FROM chunks WHERE " + " AND ".join(clauses)
        rows = self._conn().execute(sql, params).fetchall()
        result = np.fromiter((r[0] for r in rows), dtype=np.int64, count=len(rows))
        if result.size == 0:
            logger.debug(
                "Filter returned 0 chunks (inv=%s, bow=%s, doc=%s, coll=%s)",
                inv_id, bow_id, doc_type, collection,
            )
        return result

    # ------------------------------------------------------------------
    # Chunk hydration
    # ------------------------------------------------------------------

    def _fetch_rows(self, indices) -> dict[int, tuple]:
        if len(indices) == 0:
            return {}
        out: dict[int, tuple] = {}
        for start in range(0, len(indices), 800):
            batch = indices[start:start + 800]
            placeholders = ",".join("?" * len(batch))
            sql = f"SELECT {_COL_LIST} FROM chunks WHERE idx IN ({placeholders})"
            for row in self._conn().execute(sql, [int(i) for i in batch]).fetchall():
                out[row[0]] = row
        return out

    # ------------------------------------------------------------------
    # Query embedding
    # ------------------------------------------------------------------

    def _embed_query(self, query: str):
        import numpy as np
        if self._openai_client is None:
            from openai import OpenAI
            self._openai_client = OpenAI()
        resp = self._openai_client.embeddings.create(
            input=[query[:30000]], model=self._model
        )
        return np.array(resp.data[0].embedding, dtype=np.float32)

    # ------------------------------------------------------------------
    # Sync search core (called inside asyncio.to_thread)
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
        import numpy as np

        self._ensure_loaded()

        indices = self._filter_indices(
            collection=collection_filter,
            bow_id=bow_id_filter,
            inv_id=inv_id_filter,
            doc_type=doc_type_filter,
        )
        if indices.size == 0:
            return []

        q_vec = self._embed_query(query)
        q_norm = float(np.linalg.norm(q_vec))
        if q_norm == 0:
            return []

        # Vector scores — batched to avoid large heap allocations
        sub_norms = self._norms[indices]
        scores = np.empty(indices.size, dtype=np.float32)
        _BATCH = 1024
        for start in range(0, indices.size, _BATCH):
            stop = min(start + _BATCH, indices.size)
            batch_idx = indices[start:stop]
            scores[start:stop] = (self._vectors[batch_idx] @ q_vec) / (
                sub_norms[start:stop] * q_norm
            )

        # Hybrid RRF fusion when TF-IDF is available
        if self._tfidf_vectorizer is not None:
            return self._hybrid_rrf(query, indices, scores, top_k)

        # Pure vector top-k
        if top_k >= scores.size:
            top_local = np.argsort(scores)[::-1]
        else:
            top_local = np.argpartition(scores, -top_k)[-top_k:]
            top_local = top_local[np.argsort(scores[top_local])[::-1]]

        top_global = indices[top_local]
        rows = self._fetch_rows(top_global.tolist())
        results = []
        for i, score in zip(top_global, scores[top_local]):
            row = rows.get(int(i))
            if row is not None:
                results.append(_row_to_result(row, float(score)))
        return results

    def _hybrid_rrf(self, query: str, indices, vec_scores, top_k: int) -> list[SearchResult]:
        """RRF fusion of vector + TF-IDF keyword rankings with doc-type recency boost."""
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity

        # TF-IDF scores on the same filtered subset
        q_tfidf = self._tfidf_vectorizer.transform([query])
        sub_matrix = self._tfidf_matrix[indices]
        kw_scores = cosine_similarity(q_tfidf, sub_matrix).flatten()

        k, pool = 60, top_k * 2
        # Vector ranking
        if pool >= vec_scores.size:
            emb_order = np.argsort(vec_scores)[::-1]
        else:
            emb_order = np.argpartition(vec_scores, -pool)[-pool:]
            emb_order = emb_order[np.argsort(vec_scores[emb_order])[::-1]]

        # Keyword ranking (non-zero only)
        kw_nonzero = np.where(kw_scores > 0)[0]
        if kw_nonzero.size == 0:
            kw_order = np.array([], dtype=np.int64)
        elif pool >= kw_nonzero.size:
            kw_order = kw_nonzero[np.argsort(kw_scores[kw_nonzero])[::-1]]
        else:
            kw_order = np.argpartition(kw_scores[kw_nonzero], -pool)[-pool:]
            kw_order = kw_nonzero[kw_order[np.argsort(kw_scores[kw_nonzero[kw_order]])[::-1]]]

        # RRF
        rrf: dict[int, float] = {}
        w = 0.5
        for rank, local_i in enumerate(emb_order):
            global_i = int(indices[local_i])
            rrf[global_i] = rrf.get(global_i, 0.0) + w / (k + rank)
        for rank, local_i in enumerate(kw_order):
            global_i = int(indices[local_i])
            rrf[global_i] = rrf.get(global_i, 0.0) + (1 - w) / (k + rank)

        # Recency boost
        rows = self._fetch_rows(list(rrf.keys()))
        for global_i, row in rows.items():
            if row[5] in RECENCY_BOOST_DOC_TYPES:  # doc_type column
                year = _extract_year(row[3] or "")  # filename column
                if year:
                    rrf[global_i] += _RECENCY_BOOST_PER_YEAR * max(0, year - _RECENCY_BASELINE_YEAR)

        sorted_ids = sorted(rrf, key=rrf.get, reverse=True)[:top_k]  # type: ignore[arg-type]
        results = []
        for global_i in sorted_ids:
            row = rows.get(global_i)
            if row is not None:
                results.append(_row_to_result(row, rrf[global_i]))
        return results

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
        # asyncio-APPROVED-1: to_thread wraps blocking local FAISS/numpy search
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
        # asyncio-APPROVED-1: to_thread wraps blocking index scan
        return await asyncio.to_thread(self._distinct_inv_ids_sync)

    async def distinct_bow_ids(self) -> list[str]:
        # asyncio-APPROVED-1: to_thread wraps blocking index scan
        return await asyncio.to_thread(self._distinct_bow_ids_sync)

    async def count_by_bow_id(self) -> dict[str, int]:
        # asyncio-APPROVED-1: to_thread wraps blocking index scan
        return await asyncio.to_thread(self._count_by_bow_id_sync)

    # ------------------------------------------------------------------
    # Sync implementations for distinct/count queries
    # ------------------------------------------------------------------

    def _distinct_inv_ids_sync(self) -> list[str]:
        self._ensure_loaded()
        rows = self._conn().execute(
            "SELECT DISTINCT inv_id FROM chunks WHERE inv_id IS NOT NULL AND inv_id != ''"
        ).fetchall()
        return [r[0] for r in rows]

    def _distinct_bow_ids_sync(self) -> list[str]:
        self._ensure_loaded()
        rows = self._conn().execute(
            "SELECT DISTINCT bow_id FROM chunks WHERE bow_id IS NOT NULL AND bow_id != ''"
        ).fetchall()
        return [r[0] for r in rows]

    def _count_by_bow_id_sync(self) -> dict[str, int]:
        self._ensure_loaded()
        counts: dict[str, int] = {}
        for bow_id, n in self._conn().execute(
            "SELECT bow_id, COUNT(*) FROM chunks "
            "WHERE bow_id IS NOT NULL AND bow_id != '' GROUP BY bow_id"
        ).fetchall():
            counts[bow_id] = n
        # Cross-cutting chunks via supports_bow_ids
        for (sb_json,) in self._conn().execute(
            "SELECT supports_bow_ids FROM chunks "
            "WHERE supports_bow_ids IS NOT NULL AND supports_bow_ids != ''"
        ).fetchall():
            try:
                supports = json.loads(sb_json)
            except (json.JSONDecodeError, TypeError):
                continue
            for bid in supports:
                if isinstance(bid, str) and bid:
                    counts[bid] = counts.get(bid, 0) + 1
        return counts


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_year(filename: str) -> Optional[int]:
    """Extract a 4-digit year [2000, current+1] from a filename."""
    import re
    import datetime
    max_year = datetime.datetime.now().year + 1
    for m in re.finditer(r"\b(2\d{3})\b", filename):
        y = int(m.group(1))
        if 2000 <= y <= max_year:
            return y
    return None


def _build_sqlite_from_json(chunks_json: Path, sqlite_path: Path) -> None:
    """One-time migration: build chunks.sqlite from chunks.json."""
    chunks = json.loads(chunks_json.read_text(encoding="utf-8"))
    conn = sqlite3.connect(str(sqlite_path))
    conn.executescript(_SCHEMA_SQL)
    for idx, chunk in enumerate(chunks):
        canon_doc_type = chunk.get("doc_type", "")
        row = (
            idx,
            chunk.get("chunk_id", ""),
            chunk.get("file_id", ""),
            chunk.get("filename", ""),
            chunk.get("collection", ""),
            canon_doc_type,
            chunk.get("doc_type", ""),
            chunk.get("inv_id") or "",
            chunk.get("bow_id") or "",
            chunk.get("section_id", ""),
            chunk.get("section_label", ""),
            chunk.get("page_start", 0),
            chunk.get("page_end", 0),
            chunk.get("text", ""),
            chunk.get("context", ""),
            chunk.get("doc_date", ""),
            chunk.get("intelligence_role", ""),
            chunk.get("intelligence_action", ""),
            int(chunk.get("intelligence_is_best", False)),
            chunk.get("intelligence_version_group", ""),
            chunk.get("intelligence_pass_status", "ok"),
            json.dumps(chunk.get("content_tags") or []),
            json.dumps(chunk.get("supports_bow_ids") or []),
            json.dumps(chunk.get("topic_tags") or []),
            json.dumps(chunk.get("carve_out_metadata")) if chunk.get("carve_out_metadata") else "",
        )
        cols = ", ".join(_CHUNK_COLUMNS)
        placeholders = ", ".join("?" * len(_CHUNK_COLUMNS))
        conn.execute(f"INSERT OR IGNORE INTO chunks ({cols}) VALUES ({placeholders})", row)
    conn.commit()
    conn.close()
    logger.info("Built chunks.sqlite from %s (%d chunks)", chunks_json, len(chunks))
