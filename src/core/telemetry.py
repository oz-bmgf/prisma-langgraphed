"""OpenTelemetry setup for NQPR pipeline observability.

Call setup_telemetry() once at startup (e.g. in main.py) to configure a
TracerProvider that emits spans to the console. In production, replace
ConsoleSpanExporter with an OTLP exporter pointing at your collector.

LangSmith tracing is zero-config — set these env vars to activate it:
  LANGCHAIN_TRACING_V2=true
  LANGCHAIN_API_KEY=<your-key>
  LANGCHAIN_PROJECT=nqpr-pipeline  (optional, defaults to "default")
"""
from __future__ import annotations


def setup_telemetry(service_name: str = "nqpr-pipeline") -> None:
    """Configure a console-backed OTel TracerProvider."""
    from opentelemetry import trace
    from opentelemetry.sdk.resources import SERVICE_NAME, Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

    provider = TracerProvider(resource=Resource({SERVICE_NAME: service_name}))
    provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
    trace.set_tracer_provider(provider)


def get_tracer(name: str = "nqpr"):
    """Return a named tracer from the global TracerProvider."""
    from opentelemetry import trace
    return trace.get_tracer(name)
