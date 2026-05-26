"""OpenTelemetry setup for NQPR pipeline observability.

The canonical initializer is `src.observability.init_tracing` — call that once
at process startup. `setup_telemetry()` here delegates to it for backward
compatibility with any existing callers.

Tracing destinations are controlled by the OTEL collector (see otel/ dir):
  - LangSmith: set LANGSMITH_API_KEY + LANGCHAIN_TRACING_V2=true
  - Langfuse:  set LANGFUSE_OTEL_BASIC_AUTH (base64 of pk:sk)
  - Collector: OTEL_EXPORTER_OTLP_ENDPOINT (default http://localhost:4317)
"""
from __future__ import annotations


def setup_telemetry(service_name: str = "nqpr-pipeline") -> None:
    """Configure OTEL tracing. Delegates to init_tracing(); service_name is
    applied via the OTEL_SERVICE_NAME env var if not already set."""
    import os
    os.environ.setdefault("OTEL_SERVICE_NAME", service_name)
    from src.observability import init_tracing
    init_tracing()


def get_tracer(name: str = "nqpr"):
    """Return a named tracer from the global TracerProvider."""
    from opentelemetry import trace
    return trace.get_tracer(name)
