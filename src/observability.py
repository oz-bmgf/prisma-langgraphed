"""Single OTEL provider initialization — call once at process startup.

The graph process exports one OTEL stream (gRPC) to a collector. The collector
fans it out to LangSmith and Langfuse — no application code changes are needed
to add or remove a tracing destination.

Architecture:
    Graph process
        │
        │  OTEL SDK (grpc:4317)
        ▼
    OTEL Collector (otel-collector container)
        │
        ├──► LangSmith  (https://api.smith.langchain.com/otel)
        └──► Langfuse   (http://langfuse-server:3000/api/public/otel)
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


def init_tracing() -> None:
    """Initialize OTEL tracing. Safe to call multiple times (idempotent).

    Fails silently when the collector endpoint is unreachable so the graph
    always runs regardless of observability infrastructure availability.
    """
    if os.getenv("OTEL_SDK_DISABLED", "").lower() in ("true", "1", "yes"):
        return

    try:
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.instrumentation.langchain import LangchainInstrumentor

        # Idempotency: ProxyTracerProvider is the SDK default before any set_tracer_provider call.
        if type(trace.get_tracer_provider()).__name__ != "ProxyTracerProvider":
            return

        otel_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
        service_name = os.getenv("OTEL_SERVICE_NAME", "prisma-langgraphed")

        from opentelemetry.sdk.resources import SERVICE_NAME, Resource
        resource = Resource({SERVICE_NAME: service_name})

        exporter = OTLPSpanExporter(endpoint=otel_endpoint, insecure=True)
        provider = TracerProvider(resource=resource)
        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)

        # Instrument all LangChain / LangGraph calls automatically.
        # Every acall_llm invocation and ChatAnthropic/ChatOpenAI call is captured
        # as an OTEL span with correct parent-child nesting via RunnableConfig.
        LangchainInstrumentor().instrument()

    except Exception as exc:
        logger.warning("OTEL tracing initialization failed (tracing disabled): %s", exc)
