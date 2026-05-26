"""Deep web research agent using OpenAI o3-deep-research with web_search_preview.

Primary path  : ChatOpenAI(use_responses_api=True).bind_tools([web_search_preview])
                Auto-instrumented via LangchainInstrumentor; config threads trace context.
Fallback path : iterative acall_llm rounds (FALLBACK_MODEL) when primary is unavailable.

Never raises — returns DeepWebResult(success=False) on any error.
"""
from __future__ import annotations

import asyncio
import logging

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from src.config import (
    DEEP_WEB_FALLBACK_MODEL,
    DEEP_WEB_MAX_ROUNDS,
    DEEP_WEB_PRIMARY_MODEL,
    DEEP_WEB_TIMEOUT_SECONDS,
)
from src.core.llm_utils import acall_llm

logger = logging.getLogger(__name__)


class DeepWebResult(BaseModel):
    question: str
    answer: str
    sources: list[str] = []
    model_used: str = ""
    search_rounds: int = 0
    success: bool = True
    error_message: str | None = None


async def deep_web_research(
    question: str,
    context: str = "",
    *,
    model: str | None = None,
    fallback_model: str | None = None,
    timeout: int | None = None,
    config: RunnableConfig | None = None,
) -> DeepWebResult:
    """Run deep-web research for a question. Never raises."""
    primary = model or DEEP_WEB_PRIMARY_MODEL
    fallback = fallback_model or DEEP_WEB_FALLBACK_MODEL
    max_secs = timeout or DEEP_WEB_TIMEOUT_SECONDS

    try:
        # asyncio-APPROVED-3: wait_for wraps single external primary model call with timeout
        return await asyncio.wait_for(
            _primary_research(question, context, model=primary, config=config),
            timeout=max_secs,
        )
    except Exception as exc:
        logger.warning(
            "deep_web primary path failed (%s): %s — trying fallback", primary, exc
        )

    try:
        # asyncio-APPROVED-3: wait_for wraps single external fallback model call with timeout
        return await asyncio.wait_for(
            _fallback_research(question, context, model=fallback, config=config),
            timeout=max_secs,
        )
    except Exception as exc:
        logger.error("deep_web fallback also failed: %s", exc)
        return DeepWebResult(
            question=question,
            answer="",
            success=False,
            error_message=str(exc),
        )


async def _primary_research(
    question: str,
    context: str,
    *,
    model: str,
    config: RunnableConfig | None = None,
) -> DeepWebResult:
    prompt = f"{context}\n\nQuestion: {question}" if context else question

    llm = ChatOpenAI(
        model=model,
        use_responses_api=True,
        output_version="responses/v1",
    ).bind_tools([{"type": "web_search_preview"}])

    response = await llm.ainvoke([HumanMessage(content=prompt)], config=config)

    answer = ""
    sources: list[str] = []
    for block in response.content or []:
        if isinstance(block, dict) and block.get("type") == "text":
            answer = block.get("text", "")
            sources = [
                a.get("url", "")
                for a in block.get("annotations", [])
                if a.get("type") == "url_citation" and a.get("url")
            ]
            break

    return DeepWebResult(
        question=question,
        answer=answer,
        sources=sources,
        model_used=model,
        search_rounds=1,
        success=True,
    )


async def _fallback_research(
    question: str,
    context: str,
    *,
    model: str,
    config: RunnableConfig | None = None,
) -> DeepWebResult:
    from src.prompts.research_prompts import (
        DEEP_WEB_FALLBACK_ROUND_TEMPLATE,
        DEEP_WEB_FALLBACK_SYSTEM,
    )

    rounds = DEEP_WEB_MAX_ROUNDS
    accumulated: list[str] = []

    for round_i in range(rounds):
        prompt = DEEP_WEB_FALLBACK_ROUND_TEMPLATE.format(
            question=question,
            context=context,
            round=round_i + 1,
            rounds=rounds,
            prior="\n---\n".join(accumulated[-2:]),
        )
        answer = await acall_llm(
            prompt,
            system_msg=DEEP_WEB_FALLBACK_SYSTEM,
            model=model,
            config=config,
        )
        accumulated.append(answer)

    synthesis = accumulated[-1] if accumulated else ""
    return DeepWebResult(
        question=question,
        answer=synthesis,
        sources=[],
        model_used=model,
        search_rounds=rounds,
        success=True,
    )


async def run(
    task_id: str,
    query: str,
    linked_scope: str = "",
    priority: str = "",
    *,
    config: RunnableConfig | None = None,
) -> dict:
    """Entry point called by deep_web_worker in research.py."""
    result = await deep_web_research(question=query, config=config)
    return {
        "task_id": task_id,
        "task_type": "deep_web",
        "linked_scope": linked_scope,
        "query": query,
        "result": result.answer,
        "content": result.answer,
        "sources": result.sources,
        "model_used": result.model_used,
        "search_rounds": result.search_rounds,
        "success": result.success,
        "error_message": result.error_message,
    }
