#!/usr/bin/env python3
"""Smoke-test the OTEL tracing setup.

Calls init_tracing(), emits one test span with LLM attributes, force-flushes
all three BatchSpanProcessors, and prints the per-backend UI URLs for the trace.

Run with real or dummy keys to verify there are no import errors:

    python scripts/verify_tracing.py

Dummy keys are used automatically when the real ones are absent from the
environment, so this script is safe to run in CI with TRACING_ENABLED=false
to verify the import path only.

Exit codes
----------
0   tracing initialised and spans flushed (or tracing disabled — no error)
1   TracerProvider not set, or force_flush() reported a failure
"""
from __future__ import annotations

import sys
import os
from pathlib import Path

# ── Bootstrap: project root on sys.path, .env loaded ─────────────────────────

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv  # noqa: E402 — needs sys.path set first
load_dotenv(_ROOT / ".env", override=False)

# Fill in dummy values so init_tracing() can construct exporter objects even
# without real keys.  A failed HTTP flush is expected with dummies — we only
# care that setup runs without import/config errors.
os.environ.setdefault("LANGSMITH_API_KEY", "lsv2_dummy_key_verify_script")
os.environ.setdefault("LANGFUSE_PUBLIC_KEY", "pk-lf-dummy-verify")
os.environ.setdefault("LANGFUSE_SECRET_KEY", "sk-lf-dummy-verify")
os.environ.setdefault("OTEL_SERVICE_NAME", "verify-tracing")

# ── Logging ───────────────────────────────────────────────────────────────────

import logging  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)-8s %(name)s — %(message)s",
)
logger = logging.getLogger("verify_tracing")

# ── Main ──────────────────────────────────────────────────────────────────────

from observability.tracing import init_tracing, get_tracer_provider, shutdown  # noqa: E402


def _trace_url_langsmith(trace_id: str) -> str:
    project = (
        os.getenv("LANGSMITH_PROJECT")
        or os.getenv("LANGCHAIN_PROJECT")
        or "default"
    )
    return f"https://smith.langchain.com/o/-/projects/{project}/traces/{trace_id}"


def _trace_url_langfuse(trace_id: str) -> str:
    host = os.getenv("LANGFUSE_HOST", "http://localhost:3000").rstrip("/")
    return f"{host}/trace/{trace_id}"


def _trace_url_phoenix(trace_id: str) -> str:
    host = os.getenv("PHOENIX_HOST", "http://localhost:6006").rstrip("/")
    return f"{host}/traces/{trace_id}"


def main() -> int:
    tracing_enabled = os.getenv("TRACING_ENABLED", "true").lower() not in ("false", "0", "no")

    print("=" * 60)
    print("OTEL tracing verification")
    print("=" * 60)
    print(f"  TRACING_ENABLED : {tracing_enabled}")
    print(f"  OTEL_SERVICE_NAME: {os.getenv('OTEL_SERVICE_NAME', 'prisma-langgraphed')}")
    print(f"  LANGFUSE_HOST   : {os.getenv('LANGFUSE_HOST', 'http://localhost:3000')}")
    print(f"  PHOENIX_HOST    : {os.getenv('PHOENIX_HOST', 'http://localhost:6006')}")
    print()

    if not tracing_enabled:
        print("TRACING_ENABLED=false — skipping span emission (import path OK).")
        return 0

    # ── 1. Initialise ─────────────────────────────────────────────────────────
    print("Initialising OTEL tracing...")
    init_tracing(service_name="verify-tracing")

    provider = get_tracer_provider()
    if provider is None:
        print(
            "ERROR: TracerProvider is None after init_tracing() — "
            "tracing may have been suppressed by OTEL_SDK_DISABLED or an exception."
        )
        return 1

    print("TracerProvider: OK")

    # ── 2. Emit a test span ───────────────────────────────────────────────────
    from opentelemetry import trace

    tracer = trace.get_tracer("verify-tracing")
    with tracer.start_as_current_span("test.verify_tracing") as span:
        span.set_attribute("llm.model", "claude-sonnet-4-6")
        span.set_attribute("llm.input_tokens", 123)
        span.set_attribute("llm.output_tokens", 456)
        span.set_attribute("verify.script", "scripts/verify_tracing.py")
        trace_id_int = span.get_span_context().trace_id

    trace_id_hex = format(trace_id_int, "032x")
    print(f"\nTest span emitted")
    print(f"  Trace ID : {trace_id_hex}")

    # ── 3. Print deep-link URLs ───────────────────────────────────────────────
    print("\nView this trace in each backend (once spans are flushed):")
    print(f"  LangSmith : {_trace_url_langsmith(trace_id_hex)}")
    print(f"  Langfuse  : {_trace_url_langfuse(trace_id_hex)}")
    print(f"  Phoenix   : {_trace_url_phoenix(trace_id_hex)}")

    # ── 4. Flush ──────────────────────────────────────────────────────────────
    print("\nFlushing spans (timeout: 10 s)...")
    try:
        ok = provider.force_flush(timeout_millis=10_000)
    except Exception as exc:
        print(f"ERROR: force_flush() raised {exc!r}")
        return 1

    if not ok:
        # With dummy keys the HTTP POST will be rejected (401/connection refused),
        # so force_flush() returns False.  That's expected in a dry-run.
        dummy_mode = (
            "lsv2_dummy" in os.getenv("LANGSMITH_API_KEY", "")
            or "dummy" in os.getenv("LANGFUSE_PUBLIC_KEY", "")
        )
        if dummy_mode:
            print(
                "WARN: force_flush() returned False — expected with dummy keys "
                "(backends rejected the HTTP POST). Import path and wiring are OK."
            )
            shutdown()
            print("\n✓ OTEL tracing setup verified (import / wiring pass; "
                  "real keys needed for live export).")
            return 0
        else:
            print(
                "ERROR: force_flush() returned False — one or more exporters "
                "failed to deliver spans within the timeout. "
                "Check backend connectivity and credentials."
            )
            shutdown()
            return 1

    shutdown()
    print("\n✓ OTEL tracing setup fully verified — spans delivered to all backends.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
