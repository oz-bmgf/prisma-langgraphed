"""Checkpointer factory for the NQPR pipeline.

Provides async context managers for both SQLite (dev) and PostgreSQL
(production) checkpointers. All graph compilation goes through
build_checkpointer().

Usage:
    async with build_checkpointer() as checkpointer:
        graph = compile_graph(checkpointer)
        await graph.ainvoke(state, config)

No other file may import AsyncSqliteSaver or AsyncPostgresSaver directly
(AGENTS.md §4 migration rule).
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from src.config import (
    CHECKPOINT_DB_PATH,
    CHECKPOINT_PG_MAX_CONNECTIONS,
    CHECKPOINT_POSTGRES_DSN,
    CHECKPOINTER_BACKEND,
)


@asynccontextmanager
async def build_checkpointer():
    """Async context manager that yields an initialised checkpointer.

    Backend is selected by NQPR_CHECKPOINTER_BACKEND:
      sqlite   — file auto-created, no extra dependencies (default)
      postgres — requires langgraph-checkpoint-postgres + psycopg[binary]
    """
    if CHECKPOINTER_BACKEND == "postgres":
        async with _build_postgres_checkpointer() as cp:
            yield cp
    else:
        async with _build_sqlite_checkpointer() as cp:
            yield cp


@asynccontextmanager
async def _build_sqlite_checkpointer():
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    # asyncio-APPROVED-1: to_thread wraps blocking mkdir for SQLite checkpoint directory
    await asyncio.to_thread(CHECKPOINT_DB_PATH.parent.mkdir, parents=True, exist_ok=True)
    async with AsyncSqliteSaver.from_conn_string(str(CHECKPOINT_DB_PATH)) as cp:
        yield cp


@asynccontextmanager
async def _build_postgres_checkpointer():
    if not CHECKPOINT_POSTGRES_DSN:
        raise ValueError(
            "NQPR_CHECKPOINT_POSTGRES_DSN must be set when "
            "NQPR_CHECKPOINTER_BACKEND=postgres"
        )

    try:
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    except ImportError:
        raise ImportError(
            "langgraph-checkpoint-postgres is not installed. "
            "Run: pip install langgraph-checkpoint-postgres"
        )

    async with AsyncPostgresSaver.from_conn_string(
        CHECKPOINT_POSTGRES_DSN,
        max_size=CHECKPOINT_PG_MAX_CONNECTIONS,
    ) as cp:
        await cp.setup()
        yield cp


async def get_checkpoint_state(thread_id: str, checkpointer) -> dict | None:
    """Return the most recent WorkflowState dict for a thread_id.

    Returns None if no checkpoint exists for that thread.
    """
    config = {"configurable": {"thread_id": thread_id}}
    cp_tuple = await checkpointer.aget_tuple(config)
    if cp_tuple is None:
        return None
    return cp_tuple.checkpoint.get("channel_values", {})


async def list_checkpoints(thread_id: str, checkpointer) -> list[dict]:
    """Return all checkpoint metadata for a thread_id, most recent first.

    Each entry has: checkpoint_id, node_name, step, created_at.
    """
    config = {"configurable": {"thread_id": thread_id}}
    results: list[dict] = []
    async for cp_tuple in checkpointer.alist(config):
        meta = cp_tuple.metadata or {}
        results.append({
            "checkpoint_id": cp_tuple.checkpoint["id"],
            "node_name": meta.get("source", "unknown"),
            "step": meta.get("step", -1),
            "created_at": cp_tuple.checkpoint.get("ts", ""),
        })
    return results


async def delete_checkpoint(thread_id: str, checkpointer) -> bool:
    """Delete all checkpoints for a thread_id.

    Returns True if any checkpoints were deleted.
    Uses adelete_thread which removes all checkpoints for the thread atomically.
    """
    config = {"configurable": {"thread_id": thread_id}}
    # Check whether any checkpoints exist first
    has_any = False
    async for _ in checkpointer.alist(config):
        has_any = True
        break
    if has_any:
        await checkpointer.adelete_thread(thread_id)
    return has_any
