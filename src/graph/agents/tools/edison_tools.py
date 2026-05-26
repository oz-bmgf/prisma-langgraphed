from __future__ import annotations

from langchain_core.tools import tool

from src.config import EDISON_TIMEOUT_SECONDS


@tool
async def search_edison(query: str, top_k: int = 10, timeout: int = EDISON_TIMEOUT_SECONDS) -> list[dict]:
    """Search Edison literature platform. Requires EDISON_PLATFORM_API_KEY. Returns paper dicts."""
    from src.core.agents.edison import run as edison_run
    result = await edison_run(
        task_id="tool-call",
        query=query,
        timeout=timeout,
    )
    papers = result.get("papers", [])
    return papers if isinstance(papers, list) else []
