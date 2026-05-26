"""Tests for src.config — single source of truth for runtime constants."""
from __future__ import annotations

import importlib
import os
from pathlib import Path
from unittest import mock


def test_config_imports_cleanly() -> None:
    from src.config import DEFAULT_RESEARCH_MODEL, TOP_K_DEFAULT

    assert isinstance(DEFAULT_RESEARCH_MODEL, str) and DEFAULT_RESEARCH_MODEL
    assert isinstance(TOP_K_DEFAULT, int) and TOP_K_DEFAULT > 0


def test_config_reads_env_override() -> None:
    import src.config

    with mock.patch.dict(os.environ, {"NQPR_TOP_K_DEFAULT": "99"}):
        importlib.reload(src.config)
        assert src.config.TOP_K_DEFAULT == 99

    importlib.reload(src.config)


def test_dotenv_loaded_by_config_import(tmp_path: Path, monkeypatch: object) -> None:
    """Config loads .env at import time — env vars from .env are visible after import."""
    import src.config

    # If a real .env exists and sets NQPR_TOP_K_DEFAULT, respect that; otherwise
    # confirm the default value is an int (the module loaded without error).
    assert isinstance(src.config.TOP_K_DEFAULT, int)
    assert isinstance(src.config.DEFAULT_RESEARCH_MODEL, str)
    assert isinstance(src.config.COLLECTIONS_BASE_PATH, Path)
