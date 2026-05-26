"""Tests for src/api.py — FastAPI pipeline API."""
from __future__ import annotations

import asyncio
import json
import socket
import threading
import time
from contextlib import asynccontextmanager
from unittest.mock import MagicMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient

from src.api import app


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _make_mock_graph():
    graph = MagicMock()

    async def _astream(*args, **kwargs):
        # Empty async generator — no nodes fire
        if False:
            yield {}

    async def _aget_state(*args, **kwargs):
        snap = MagicMock()
        snap.values = {}
        snap.next = ()
        return snap

    graph.astream = _astream
    graph.aget_state = _aget_state
    return graph


def _make_lifespan_patches(mock_graph):
    """Return a stack of patches that prevent real I/O during lifespan startup."""
    @asynccontextmanager
    async def _mock_build_checkpointer():
        yield MagicMock()

    return [
        patch("src.api.compile_graph", return_value=mock_graph),
        patch("src.api.build_checkpointer", _mock_build_checkpointer),
        patch("src.api.setup_telemetry"),
    ]


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def client():
    mock_graph = _make_mock_graph()
    patches = _make_lifespan_patches(mock_graph)
    with patches[0], patches[1], patches[2]:
        with TestClient(app) as c:
            yield c


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_start_run_returns_202(client):
    resp = client.post(
        "/runs",
        json={
            "program": "TEST",
            "run_name": "run-01",
            "collection_name": "TEST-ingested",
            "base_dir": "/tmp/test",
            "ingested_dir": "/tmp/test/TEST-ingested",
        },
    )
    assert resp.status_code == 202
    body = resp.json()
    assert body["thread_id"] == "TEST::run-01"
    assert body["status"] == "started"


def test_get_status_returns_404_for_unknown(client):
    resp = client.get("/runs/NO::SUCH-RUN")
    assert resp.status_code == 404


def test_resume_returns_404_for_unknown(client):
    resp = client.post("/runs/NO::SUCH-RUN/resume", json={"value": "approved"})
    assert resp.status_code == 404


def test_get_report_returns_404_when_not_ready(client):
    resp = client.get("/runs/NO::SUCH-RUN/report")
    assert resp.status_code == 404


def test_stream_returns_404_for_unknown(client):
    resp = client.get("/runs/NO::SUCH-RUN/stream")
    assert resp.status_code == 404


def test_cancel_unknown_run_returns_204(client):
    resp = client.delete("/runs/NO::SUCH-RUN")
    assert resp.status_code == 204


def test_uvicorn_startup():
    import uvicorn

    port = _free_port()
    config = uvicorn.Config(
        app, host="127.0.0.1", port=port, lifespan="off", loop="asyncio"
    )
    server = uvicorn.Server(config)

    thread = threading.Thread(target=asyncio.run, args=(server.serve(),), daemon=True)
    thread.start()

    started = False
    for _ in range(40):
        time.sleep(0.1)
        try:
            resp = httpx.get(f"http://127.0.0.1:{port}/docs", timeout=1.0)
            if resp.status_code == 200:
                started = True
                break
        except Exception:
            pass

    server.should_exit = True
    thread.join(timeout=5)

    assert started, "Uvicorn server did not start in time"
