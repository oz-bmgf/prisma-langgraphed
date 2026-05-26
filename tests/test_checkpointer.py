"""Tests for src/core/checkpointer.py and compile_graph() interrupt behaviour."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langgraph.graph import END, START, StateGraph
from typing import Annotated
import operator


# ---------------------------------------------------------------------------
# Helpers — minimal 2-node graph for integration tests
# ---------------------------------------------------------------------------


class _MinState(dict):
    """Minimal state dict for stub graph tests."""


def _make_stub_graph(counter_field: str = "counter"):
    """Build a tiny 2-node StateGraph that increments a counter field."""
    from typing import TypedDict

    class StubState(TypedDict):
        counter: Annotated[int, operator.add]

    async def node_a(state: StubState) -> dict:
        return {"counter": 1}

    async def node_b(state: StubState) -> dict:
        return {"counter": 1}

    builder = StateGraph(StubState)
    builder.add_node("node_a", node_a)
    builder.add_node("node_b", node_b)
    builder.add_edge(START, "node_a")
    builder.add_edge("node_a", "node_b")
    builder.add_edge("node_b", END)
    return builder


# ---------------------------------------------------------------------------
# test_build_checkpointer_sqlite
# ---------------------------------------------------------------------------


async def test_build_checkpointer_sqlite(tmp_path: Path):
    db_file = tmp_path / "test_checkpoints.db"

    with patch("src.core.checkpointer.CHECKPOINTER_BACKEND", "sqlite"), \
         patch("src.core.checkpointer.CHECKPOINT_DB_PATH", db_file):
        from src.core.checkpointer import build_checkpointer
        async with build_checkpointer() as cp:
            assert hasattr(cp, "aget_tuple"), "checkpointer missing aget_tuple"
            assert hasattr(cp, "alist"), "checkpointer missing alist"
            assert hasattr(cp, "adelete_thread"), "checkpointer missing adelete_thread"
            assert db_file.exists(), "SQLite file was not created"


# ---------------------------------------------------------------------------
# test_build_checkpointer_postgres_missing_dsn
# ---------------------------------------------------------------------------


async def test_build_checkpointer_postgres_missing_dsn():
    with patch("src.core.checkpointer.CHECKPOINTER_BACKEND", "postgres"), \
         patch("src.core.checkpointer.CHECKPOINT_POSTGRES_DSN", ""):
        from src.core.checkpointer import build_checkpointer
        with pytest.raises(ValueError, match="NQPR_CHECKPOINT_POSTGRES_DSN"):
            async with build_checkpointer():
                pass


# ---------------------------------------------------------------------------
# test_build_checkpointer_postgres_missing_package
# ---------------------------------------------------------------------------


async def test_build_checkpointer_postgres_missing_package():
    import sys

    with patch("src.core.checkpointer.CHECKPOINTER_BACKEND", "postgres"), \
         patch("src.core.checkpointer.CHECKPOINT_POSTGRES_DSN", "postgresql://localhost/test"), \
         patch.dict(sys.modules, {"langgraph.checkpoint.postgres.aio": None}):
        from src.core.checkpointer import build_checkpointer
        with pytest.raises((ImportError, TypeError)):
            async with build_checkpointer():
                pass


# ---------------------------------------------------------------------------
# test_get_checkpoint_state_missing
# ---------------------------------------------------------------------------


async def test_get_checkpoint_state_missing(tmp_path: Path):
    db_file = tmp_path / "cp_missing.db"

    with patch("src.core.checkpointer.CHECKPOINTER_BACKEND", "sqlite"), \
         patch("src.core.checkpointer.CHECKPOINT_DB_PATH", db_file):
        from src.core.checkpointer import build_checkpointer, get_checkpoint_state
        async with build_checkpointer() as cp:
            result = await get_checkpoint_state("nonexistent::thread", cp)
            assert result is None


# ---------------------------------------------------------------------------
# test_get_checkpoint_state_present
# ---------------------------------------------------------------------------


async def test_get_checkpoint_state_present(tmp_path: Path):
    db_file = tmp_path / "cp_present.db"
    builder = _make_stub_graph()

    with patch("src.core.checkpointer.CHECKPOINTER_BACKEND", "sqlite"), \
         patch("src.core.checkpointer.CHECKPOINT_DB_PATH", db_file):
        from src.core.checkpointer import build_checkpointer, get_checkpoint_state
        async with build_checkpointer() as cp:
            graph = builder.compile(checkpointer=cp)
            config = {"configurable": {"thread_id": "test::get-state"}}
            await graph.ainvoke({"counter": 0}, config=config)

            state = await get_checkpoint_state("test::get-state", cp)
            assert state is not None
            assert isinstance(state, dict)
            # counter was incremented twice (node_a + node_b each add 1)
            assert state.get("counter") == 2


# ---------------------------------------------------------------------------
# test_list_checkpoints_empty
# ---------------------------------------------------------------------------


async def test_list_checkpoints_empty(tmp_path: Path):
    db_file = tmp_path / "cp_list_empty.db"

    with patch("src.core.checkpointer.CHECKPOINTER_BACKEND", "sqlite"), \
         patch("src.core.checkpointer.CHECKPOINT_DB_PATH", db_file):
        from src.core.checkpointer import build_checkpointer, list_checkpoints
        async with build_checkpointer() as cp:
            result = await list_checkpoints("never::ran", cp)
            assert result == []


# ---------------------------------------------------------------------------
# test_list_checkpoints_after_run
# ---------------------------------------------------------------------------


async def test_list_checkpoints_after_run(tmp_path: Path):
    db_file = tmp_path / "cp_list_run.db"
    builder = _make_stub_graph()

    with patch("src.core.checkpointer.CHECKPOINTER_BACKEND", "sqlite"), \
         patch("src.core.checkpointer.CHECKPOINT_DB_PATH", db_file):
        from src.core.checkpointer import build_checkpointer, list_checkpoints
        async with build_checkpointer() as cp:
            graph = builder.compile(checkpointer=cp)
            config = {"configurable": {"thread_id": "test::list"}}
            await graph.ainvoke({"counter": 0}, config=config)

            checkpoints = await list_checkpoints("test::list", cp)
            # At minimum: one checkpoint per node + initial
            assert len(checkpoints) >= 2
            for entry in checkpoints:
                assert "checkpoint_id" in entry
                assert "node_name" in entry
                assert "step" in entry


# ---------------------------------------------------------------------------
# test_delete_checkpoint
# ---------------------------------------------------------------------------


async def test_delete_checkpoint(tmp_path: Path):
    db_file = tmp_path / "cp_delete.db"
    builder = _make_stub_graph()

    with patch("src.core.checkpointer.CHECKPOINTER_BACKEND", "sqlite"), \
         patch("src.core.checkpointer.CHECKPOINT_DB_PATH", db_file):
        from src.core.checkpointer import (
            build_checkpointer,
            delete_checkpoint,
            get_checkpoint_state,
        )
        async with build_checkpointer() as cp:
            graph = builder.compile(checkpointer=cp)
            config = {"configurable": {"thread_id": "test::delete"}}
            await graph.ainvoke({"counter": 0}, config=config)

            deleted = await delete_checkpoint("test::delete", cp)
            assert deleted is True

            state = await get_checkpoint_state("test::delete", cp)
            assert state is None


# ---------------------------------------------------------------------------
# test_compile_graph_no_interrupts_by_default
# ---------------------------------------------------------------------------


def test_compile_graph_no_interrupts_by_default():
    from src.graph.workflow import compile_graph
    graph = compile_graph(human_interrupts=False)
    interrupts = graph.interrupt_before_nodes
    assert interrupts == [], f"Expected no interrupts, got: {interrupts}"


# ---------------------------------------------------------------------------
# test_compile_graph_with_interrupts
# ---------------------------------------------------------------------------


def test_compile_graph_with_interrupts():
    from src.graph.workflow import compile_graph, _HUMAN_INTERRUPT_NODES
    graph = compile_graph(human_interrupts=True)
    interrupts = graph.interrupt_before_nodes
    assert "analyze" in interrupts
    assert len(interrupts) == len(_HUMAN_INTERRUPT_NODES)


# ---------------------------------------------------------------------------
# test_resume_from_checkpoint
# ---------------------------------------------------------------------------


async def test_resume_from_checkpoint(tmp_path: Path):
    db_file = tmp_path / "cp_resume.db"

    from typing import TypedDict

    class ResState(TypedDict):
        counter: Annotated[int, operator.add]
        visited: Annotated[list[str], operator.add]

    async def n1(state: ResState) -> dict:
        return {"counter": 1, "visited": ["n1"]}

    async def n2(state: ResState) -> dict:
        return {"counter": 1, "visited": ["n2"]}

    async def n3(state: ResState) -> dict:
        return {"counter": 1, "visited": ["n3"]}

    builder = StateGraph(ResState)
    builder.add_node("n1", n1)
    builder.add_node("n2", n2)
    builder.add_node("n3", n3)
    builder.add_edge(START, "n1")
    builder.add_edge("n1", "n2")
    builder.add_edge("n2", "n3")
    builder.add_edge("n3", END)

    with patch("src.core.checkpointer.CHECKPOINTER_BACKEND", "sqlite"), \
         patch("src.core.checkpointer.CHECKPOINT_DB_PATH", db_file):
        from src.core.checkpointer import build_checkpointer, get_checkpoint_state
        thread_id = "test::resume"
        config = {"configurable": {"thread_id": thread_id}}

        async with build_checkpointer() as cp:
            # First pass: run through n1 only, interrupted before n2
            graph = builder.compile(
                checkpointer=cp,
                interrupt_before=["n2"],
            )
            await graph.ainvoke({"counter": 0, "visited": []}, config=config)

            state_after_n1 = await get_checkpoint_state(thread_id, cp)
            assert "n1" in (state_after_n1.get("visited") or [])
            assert "n2" not in (state_after_n1.get("visited") or [])

            # Resume: pass None — LangGraph restores from checkpoint
            result = await graph.ainvoke(None, config=config)
            assert "n2" in result.get("visited", [])
            assert "n3" in result.get("visited", [])
            assert result.get("counter") == 3


# ---------------------------------------------------------------------------
# test_resume_rebuilds_backend_from_state
# ---------------------------------------------------------------------------


def test_resume_rebuilds_backend_from_state(tmp_path: Path):
    from main import _build_backend
    from src.backends.base import SearchBackend

    state = {
        "ingested_dir": str(tmp_path),
        "collection_name": "test-collection",
    }
    # build_search_backend will instantiate LocalSearchIndex
    # (NQPR_SEARCH_BACKEND defaults to "local")
    backend = _build_backend(state)
    assert isinstance(backend, SearchBackend)
