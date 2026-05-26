"""Tool call tracing — OTel spans + context-local trace buffer.

Usage in Send() workers (return trace in result dict):
    result, base_trace = await traced_tool("slr_worker")(fn)(*args, **kwargs)
    trace = {**base_trace, "query": query, ...}
    return {"research_results": [result], "slr_traces": [trace]}

Usage in investigation loop tools (append to buffer, flush on worker return):
    init_trace_buffer()      # at start of investigate_link
    # ... tool calls inside the loop append via append_to_buffer() ...
    flushed = flush_trace_buffer()   # before returning from investigate_link
    return {"link_assessments": [result], **flushed}
"""
from __future__ import annotations

import time
from contextvars import ContextVar
from datetime import datetime, timezone
from functools import wraps

from opentelemetry import trace as otel_trace

_tracer = otel_trace.get_tracer("nqpr.tools")

# ---------------------------------------------------------------------------
# traced_tool decorator
# ---------------------------------------------------------------------------


def traced_tool(tool_name: str):
    """Decorator for async tool functions.

    Wraps the function in an OTel span and returns (result, base_trace_dict)
    instead of just result. The caller enriches base_trace_dict with
    tool-specific metadata and appends to the appropriate state reducer field.
    """
    def decorator(fn):
        @wraps(fn)
        async def wrapper(*args, **kwargs):
            start = time.monotonic()
            started_at = datetime.now(timezone.utc).isoformat()
            with _tracer.start_as_current_span(
                f"tool.{tool_name}",
                attributes={"tool.name": tool_name},
            ) as span:
                try:
                    result = await fn(*args, **kwargs)
                    duration_ms = int((time.monotonic() - start) * 1000)
                    span.set_attribute("tool.success", True)
                    span.set_attribute("tool.duration_ms", duration_ms)
                    return result, {
                        "tool_name": tool_name,
                        "called_at": started_at,
                        "duration_ms": duration_ms,
                        "success": True,
                    }
                except Exception as exc:
                    duration_ms = int((time.monotonic() - start) * 1000)
                    span.set_attribute("tool.success", False)
                    span.record_exception(exc)
                    return None, {
                        "tool_name": tool_name,
                        "called_at": started_at,
                        "duration_ms": duration_ms,
                        "success": False,
                        "error_message": str(exc),
                    }
        return wrapper
    return decorator


# ---------------------------------------------------------------------------
# Context-local trace buffer (for tools called inside investigation loops)
# ---------------------------------------------------------------------------

_BUFFER_KEYS = (
    "web_search_traces",
    "compute_traces",
    "collection_search_traces",
    "asta_traces",
)

_trace_buffer: ContextVar[dict[str, list]] = ContextVar(
    "_trace_buffer", default={}
)


def init_trace_buffer() -> None:
    """Initialise an empty trace buffer for the current async context.

    Call at the start of each investigate_link / investigate_science_assumption worker.
    """
    _trace_buffer.set({k: [] for k in _BUFFER_KEYS})


def append_to_buffer(field: str, trace: dict) -> None:
    """Append a trace dict to the named buffer field.

    Called from within @tool functions during an investigation loop.
    No-ops if the buffer has not been initialised (e.g. tool called outside a loop).
    """
    buf = _trace_buffer.get()
    if field in buf:
        buf[field].append(trace)


def flush_trace_buffer() -> dict[str, list]:
    """Return all buffered traces and reset the buffer.

    Call just before returning from investigate_link / investigate_science_assumption.
    The returned dict can be merged directly into the worker's return dict.
    """
    buf = _trace_buffer.get()
    result = {k: list(v) for k, v in buf.items()}
    _trace_buffer.set({k: [] for k in _BUFFER_KEYS})
    return result


# ---------------------------------------------------------------------------
# Trace summary helper
# ---------------------------------------------------------------------------


def summarise_traces(state: dict) -> dict:
    """Compute per-tool-type summary from a WorkflowState dict.

    Returns a dict of {tool_short_name: {count, avg_duration_ms, error_count}}.
    """
    field_map = {
        "asta_traces": "asta",
        "slr_traces": "slr",
        "lbd_traces": "lbd",
        "deep_web_traces": "deep_web",
        "edison_traces": "edison",
        "web_search_traces": "web_search",
        "compute_traces": "compute",
        "collection_search_traces": "collection_search",
        "investigation_traces": "investigation",
    }
    summary: dict[str, dict] = {}
    for field, short in field_map.items():
        traces: list[dict] = state.get(field) or []
        if not traces:
            continue
        count = len(traces)
        durations = [t.get("duration_ms", 0) for t in traces if t.get("duration_ms") is not None]
        avg_ms = int(sum(durations) / len(durations)) if durations else 0
        errors = sum(1 for t in traces if not t.get("success", True))
        entry: dict = {"count": count, "avg_duration_ms": avg_ms, "error_count": errors}
        if field == "investigation_traces":
            iters = [t.get("iterations_used", 0) for t in traces]
            entry["avg_iterations"] = round(sum(iters) / len(iters), 1) if iters else 0
        summary[short] = entry
    return summary
