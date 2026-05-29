"""Async LLM utilities for NQPR pipeline.

acall_llm — unified async LLM gateway (LangChain-native; Anthropic or OpenAI routed by
             model prefix). New signature: acall_llm(prompt, system_msg="", *, model, ...)

parse_json_and_prose, safe_parse_json — JSON extraction from LLM output.
trace_llm_context, log_trace_event — optional JSONL trace logging via LLM_TRACE_FILE.

No LangGraph imports — pure business logic callable from both node bodies and tests.
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Generator, TypeVar

from pydantic import BaseModel

_SchemaT = TypeVar("_SchemaT", bound=BaseModel)

# Eager imports — must happen before the event loop starts so LangChain's
# first-time initialization (tokenizer/cache dirs via os.mkdir) does not
# trigger a blockbuster BlockingError during request handling.
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Trace write lock + failure tracking
# ---------------------------------------------------------------------------

_TRACE_WRITE_LOCK = threading.Lock()
_TRACE_EMIT_FAILURES: set[str] = set()
_TRACE_EMIT_WARN_LIMIT = 5

_TRACE_CTX: ContextVar[dict[str, str] | None] = ContextVar("_llm_trace_ctx", default=None)
_TRACE_CTX_KEYS = ("agent", "scope_id", "scope_label", "investment_id", "step", "call_type")

from src.config import (
    DEFAULT_RESEARCH_MODEL,
    DEFAULT_MAX_TOKENS,
    THINKING_BUDGET_TOKENS,
    THINKING_MIN_MAX_TOKENS,
)

_ANTHROPIC_PREFIXES = ("claude-",)
_OPENAI_PREFIXES = ("gpt-", "o1-", "o3-", "o4-")

DEFAULT_MODEL = DEFAULT_RESEARCH_MODEL


# ---------------------------------------------------------------------------
# Trace context manager
# ---------------------------------------------------------------------------


@contextmanager
def trace_llm_context(**kwargs: str) -> Generator[None, None, None]:
    """Set trace metadata for all acall_llm calls in this scope."""
    prev = _TRACE_CTX.get() or {}
    merged = {**prev, **{k: v for k, v in kwargs.items() if v}}
    token = _TRACE_CTX.set(merged)
    try:
        yield
    finally:
        _TRACE_CTX.reset(token)


def log_trace_event(entry_type: str, **fields: Any) -> None:
    """Write a structured event to the trace file."""
    trace_file = os.environ.get("LLM_TRACE_FILE")
    if not trace_file:
        return
    try:
        ctx = _TRACE_CTX.get() or {}
        entry = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "entry_type": entry_type,
            **{k: ctx.get(k, "") for k in _TRACE_CTX_KEYS},
            **fields,
        }
        for key, value in ctx.items():
            if key not in entry and value not in ("", None):
                entry[key] = value
        payload = json.dumps(entry, default=str) + "\n"
        with _TRACE_WRITE_LOCK:
            with open(trace_file, "a") as f:
                f.write(payload)
    except Exception as exc:
        _warn_trace_fail(exc)


def log_inv_decision(
    *,
    decision: str,
    inv_id: str = "",
    bow_id: str = "",
    **details: Any,
) -> None:
    """Emit an inv_decision trace event."""
    log_trace_event("inv_decision", decision=decision, inv_id=inv_id, bow_id=bow_id, **details)


def _warn_trace_fail(exc: BaseException) -> None:
    key = f"{type(exc).__name__}:{exc}"
    with _TRACE_WRITE_LOCK:
        if key in _TRACE_EMIT_FAILURES or len(_TRACE_EMIT_FAILURES) >= _TRACE_EMIT_WARN_LIMIT:
            return
        _TRACE_EMIT_FAILURES.add(key)
        count = len(_TRACE_EMIT_FAILURES)
    logger.warning("trace.jsonl emit failed (%d/%d): %s", count, _TRACE_EMIT_WARN_LIMIT, exc)


# ---------------------------------------------------------------------------
# LLM client builder
# ---------------------------------------------------------------------------


def _build_llm(
    model: str,
    *,
    max_tokens: int,
    thinking: bool = False,
    temperature: float = 0.0,
    **extra,
) -> Any:
    """Build a LangChain chat model instance for the given model identifier."""
    if any(model.startswith(p) for p in _ANTHROPIC_PREFIXES):
        params: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if thinking:
            params["thinking"] = {"type": "enabled", "budget_tokens": THINKING_BUDGET_TOKENS}
            params["max_tokens"] = max(max_tokens, THINKING_MIN_MAX_TOKENS)
        params.update(extra)
        return ChatAnthropic(**params)
    else:
        params = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        params.update(extra)
        return ChatOpenAI(**params)


# ---------------------------------------------------------------------------
# Unified async LLM gateway
# ---------------------------------------------------------------------------


async def acall_llm(
    prompt: str,
    system_msg: str = "",
    *,
    model: str,
    json_mode: bool = False,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    images: list[dict] | None = None,
    thinking: bool = False,
    config: RunnableConfig | None = None,
    **kwargs,
) -> str:
    """Async unified LLM call — routes to Anthropic or OpenAI by model prefix.

    Returns plain text.  For structured (Pydantic) output use acall_structured().

    Args:
        prompt:     User-turn text.
        system_msg: Optional system prompt.
        model:      Model identifier (e.g. "claude-sonnet-4-6", "gpt-4o").
        json_mode:  Return parsed JSON dict instead of raw string (deprecated —
                    prefer acall_structured with an explicit schema).
        max_tokens: Maximum tokens in the response.
        images:     Optional list of image content dicts appended to the user message.
        thinking:   Enable extended thinking (Anthropic models only).
        config:     LangGraph RunnableConfig for trace context propagation.
        **kwargs:   Passed through to the underlying LangChain chat model.

    Returns:
        str (or dict when json_mode=True).
    """
    if json_mode:
        logger.warning(
            "acall_llm json_mode=True is deprecated; use acall_structured() with an explicit schema"
        )

    messages: list[Any] = []
    if system_msg:
        messages.append(SystemMessage(content=system_msg))

    if images:
        content: Any = [{"type": "text", "text": prompt}] + images
    else:
        content = prompt
    messages.append(HumanMessage(content=content))

    t0 = time.time()
    error_msg: str | None = None

    try:
        llm = _build_llm(model, max_tokens=max_tokens, thinking=thinking, **kwargs)
        response = await llm.ainvoke(messages, config=config)
        text: str = response.content if hasattr(response, "content") else str(response)

        if json_mode:
            return safe_parse_json(text)

        return text if isinstance(text, str) else str(text)

    except Exception as exc:
        error_msg = f"{type(exc).__name__}: {exc}"
        logger.warning("acall_llm failed (%s): %s", model, error_msg[:200])
        raise
    finally:
        elapsed_ms = (time.time() - t0) * 1000
        _log_trace_call(model=model, latency_ms=elapsed_ms, error=error_msg)


async def acall_structured(
    prompt: str,
    system_msg: str = "",
    *,
    model: str,
    schema: type[_SchemaT],
    max_tokens: int = DEFAULT_MAX_TOKENS,
    images: list[dict] | None = None,
    thinking: bool = False,
    config: RunnableConfig | None = None,
    **kwargs: Any,
) -> _SchemaT:
    """Async structured LLM call — returns a validated Pydantic model instance.

    Routes to Anthropic or OpenAI by model prefix and delegates structured output
    to LangChain's with_structured_output, letting each provider use its own
    optimal strategy.

    Args:
        prompt:     User-turn text.
        system_msg: Optional system prompt.
        model:      Model identifier (e.g. "claude-sonnet-4-6", "gpt-4o").
        schema:     Pydantic BaseModel subclass that defines the output shape.
        max_tokens: Maximum tokens in the response.
        images:     Optional list of image content dicts appended to the user message.
        thinking:   Enable extended thinking (Anthropic only).
        config:     LangGraph RunnableConfig for trace context propagation.
        **kwargs:   Passed through to the underlying LangChain chat model.

    Returns:
        A validated instance of *schema*.
    """
    messages: list[Any] = []
    if system_msg:
        messages.append(SystemMessage(content=system_msg))

    if images:
        content: Any = [{"type": "text", "text": prompt}] + images
    else:
        content = prompt
    messages.append(HumanMessage(content=content))

    t0 = time.time()
    error_msg: str | None = None

    try:
        llm = _build_llm(model, max_tokens=max_tokens, thinking=thinking, **kwargs)
        # OpenAI's json_schema path (default for ChatOpenAI) wraps parsed responses
        # in ParsedChatCompletion whose `parsed: None` field produces
        # PydanticSerializationUnexpectedValue warnings during checkpointing.
        # function_calling uses the tool-call path (PydanticToolsParser) which has
        # no such wrapper.  Anthropic already defaults to function_calling.
        so_kwargs: dict[str, Any] = {}
        if any(model.startswith(p) for p in _OPENAI_PREFIXES):
            so_kwargs["method"] = "function_calling"
        structured = llm.with_structured_output(schema, **so_kwargs)
        result: _SchemaT = await structured.ainvoke(messages, config=config)
        return result

    except Exception as exc:
        error_msg = f"{type(exc).__name__}: {exc}"
        logger.warning("acall_structured failed (%s): %s", model, error_msg[:200])
        raise
    finally:
        elapsed_ms = (time.time() - t0) * 1000
        _log_trace_call(model=model, latency_ms=elapsed_ms, error=error_msg)


def _log_trace_call(model: str, latency_ms: float, error: str | None) -> None:
    trace_file = os.environ.get("LLM_TRACE_FILE")
    if not trace_file:
        return
    try:
        ctx = _TRACE_CTX.get() or {}
        entry = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "entry_type": "llm_call",
            "model": model,
            "latency_ms": round(latency_ms),
            **{k: ctx.get(k, "") for k in _TRACE_CTX_KEYS},
        }
        if error:
            entry["error"] = error
        payload = json.dumps(entry) + "\n"
        with _TRACE_WRITE_LOCK:
            with open(trace_file, "a") as f:
                f.write(payload)
    except Exception as exc:
        _warn_trace_fail(exc)


# ---------------------------------------------------------------------------
# JSON parsing utilities
# ---------------------------------------------------------------------------


def parse_json_and_prose(response: str | dict | list) -> tuple[dict, str]:
    """Parse a response that has a JSON block followed by free prose.

    Returns (structured_data, prose_text).
    """
    if isinstance(response, dict):
        return response, ""
    if isinstance(response, list):
        return {"items": response}, ""
    if not isinstance(response, str):
        return {}, str(response)

    text = response.strip()

    # ```json ... ``` block
    json_start = text.find("```json")
    if json_start >= 0:
        json_body_start = text.find("\n", json_start) + 1
        json_end = text.find("```", json_body_start)
        if json_end > json_body_start:
            json_str = text[json_body_start:json_end].strip()
            prose = text[json_end + 3:].strip()
            try:
                structured = json.loads(json_str)
                if isinstance(structured, dict):
                    return structured, prose
            except json.JSONDecodeError:
                pass

    # ``` ... ``` (no json tag)
    fence_start = text.find("```\n")
    if fence_start >= 0:
        body_start = fence_start + 4
        fence_end = text.find("```", body_start)
        if fence_end > body_start:
            json_str = text[body_start:fence_end].strip()
            prose = text[fence_end + 3:].strip()
            try:
                structured = json.loads(json_str)
                if isinstance(structured, dict):
                    return structured, prose
            except json.JSONDecodeError:
                pass

    structured = safe_parse_json(text)
    if structured:
        return structured, ""

    return {}, text


def safe_parse_json(raw: str | dict, fallback: dict | None = None) -> dict:
    """Robustly parse LLM output as JSON.

    Tries: direct parse, markdown fence extraction, largest balanced {…},
    JSON array wrapping, truncated key-value salvage.
    """
    if isinstance(raw, dict):
        return raw
    if not raw or not isinstance(raw, str):
        return fallback or {}

    text = raw.strip()

    # Strategy 1: direct parse
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
        if isinstance(result, list):
            return {"items": result}
    except json.JSONDecodeError:
        pass

    # Strategy 2: markdown fence extraction
    for pattern in [r"```json\s*\n(.*?)\n\s*```", r"```\s*\n(.*?)\n\s*```"]:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                result = json.loads(match.group(1))
                if isinstance(result, dict):
                    return result
                if isinstance(result, list):
                    return {"items": result}
            except json.JSONDecodeError:
                pass

    # Strategy 3: largest balanced { … } block
    best_json: dict | None = None
    best_len = 0
    for m in re.finditer(r"\{", text):
        start = m.start()
        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(text)):
            ch = text[i]
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == '"' and not escape:
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start:i + 1]
                    if len(candidate) > best_len:
                        try:
                            result = json.loads(candidate)
                            if isinstance(result, dict):
                                best_json = result
                                best_len = len(candidate)
                        except json.JSONDecodeError:
                            pass
                    break

    if best_json is not None:
        return best_json

    logger.warning("Failed to parse LLM JSON (%d chars), using fallback", len(text))
    out = fallback or {}
    out["_parse_failed"] = True
    return out
