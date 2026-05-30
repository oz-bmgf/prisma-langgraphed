"""Single-collector OTEL exporter: app → local otel-collector → Phoenix + Langfuse.

Architecture:

    Application process
        │
        └──► BatchSpanProcessor ──► OTLPSpanExporter (gRPC) ──► otel-collector:4317
               (background thread)                               (Docker service)
                                                                         │
                                                                   ┌─────┴─────┐
                                                                   ▼           ▼
                                                               Phoenix     Langfuse
                                    (when PHOENIX_TRACING_ENABLED=true in collector env)
                                    (when LANGFUSE_TRACING_ENABLED=true in collector env)

LangSmith stays on its own native SDK path via LANGCHAIN_TRACING_V2 — no OTEL
exporter for LangSmith is registered here.

The collector config is generated at container startup from
otel/collector.config.yaml.j2 by otel/generate_config.py, driven by the
PHOENIX_* and LANGFUSE_* env vars passed via env_file in docker-compose.yml.
To toggle backends or change endpoints, edit .env and rebuild:
  docker compose build otel-collector && docker compose up -d otel-collector

App-side env vars
-----------------
TRACING_ENABLED              true (default) | false  — master on/off; false registers
                             a NoOpTracerProvider and skips all OTEL export
OTEL_SDK_DISABLED            true to disable (legacy kill-switch; same as TRACING_ENABLED=false)
OTEL_SERVICE_NAME            service name on all spans (default: prisma-langgraphed)
OTEL_EXPORTER_OTLP_ENDPOINT  HTTP/Protobuf base URL for local collector (SDK appends /v1/traces)
                             (default: http://localhost:4318)
OTEL_BSP_MAX_QUEUE_SIZE        4096  max spans buffered before dropping
OTEL_BSP_SCHEDULE_DELAY_MILLIS 5000  export interval (ms)
OTEL_BSP_MAX_EXPORT_BATCH_SIZE 1024  spans per export request
OTEL_BSP_EXPORT_TIMEOUT_MILLIS 30000 per-request timeout (ms)
OTEL_TRACES_SAMPLER_ARG        1.0   TraceIdRatioBased sample rate (0.0–1.0); 1.0 = keep all

Collector-side env vars (read by otel/generate_config.py, not by this module)
--------------
PHOENIX_TRACING_ENABLED      include Phoenix exporter in collector pipeline
PHOENIX_ENDPOINT             full OTLP/HTTP endpoint for Phoenix
LANGFUSE_TRACING_ENABLED     include Langfuse exporter in collector pipeline
LANGFUSE_ENDPOINT            full OTLP/HTTP endpoint for Langfuse
LANGFUSE_PUBLIC_KEY          used to build Basic auth header in the collector
LANGFUSE_SECRET_KEY          used to build Basic auth header in the collector

LangSmith (native SDK — do not touch)
LANGCHAIN_TRACING_V2         enables the LangChain native SDK integration
LANGCHAIN_API_KEY            LangSmith API key
"""
from __future__ import annotations

import atexit
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

# Module-level singleton — set once by init_tracing(), cleared by shutdown().
_provider = None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def init_tracing(service_name: Optional[str] = None) -> None:
    """Initialise single OTEL gRPC exporter → local collector.

    Safe to call multiple times — idempotent after the first successful call.
    Fails silently (logs a warning) so the application always starts regardless
    of observability infrastructure availability.

    Parameters
    ----------
    service_name:
        Service name embedded in every span's resource attributes.
        Falls back to OTEL_SERVICE_NAME env var, then ``"prisma-langgraphed"``.
    """
    global _provider

    # ── Master kill-switches ─────────────────────────────────────────────────
    if os.getenv("TRACING_ENABLED", "true").lower() in ("false", "0", "no"):
        logger.debug("TRACING_ENABLED=false — registering NoOpTracerProvider")
        _register_noop()
        return

    if os.getenv("OTEL_SDK_DISABLED", "").lower() in ("true", "1", "yes"):
        logger.debug("OTEL_SDK_DISABLED=true — registering NoOpTracerProvider")
        _register_noop()
        return

    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import SERVICE_NAME, Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.sdk.trace.sampling import ParentBased, TraceIdRatioBased
        from opentelemetry.exporter.otlp.proto.http import Compression

        # Idempotency: ProxyTracerProvider is the SDK default before the first
        # set_tracer_provider() call; any other type means we already initialised.
        if type(trace.get_tracer_provider()).__name__ != "ProxyTracerProvider":
            logger.debug("TracerProvider already set — skipping init_tracing()")
            return

        svc = service_name or os.getenv("OTEL_SERVICE_NAME", "prisma-langgraphed")
        project_name = os.getenv("PHOENIX_PROJECT_NAME", svc)

        resource = Resource.create(
            {
                SERVICE_NAME: svc,
                "openinference.project.name": project_name,
            }
        )

        # Optional head-based sampling — set OTEL_TRACES_SAMPLER_ARG=0.25 to keep
        # 25 % of traces (drop the rest at the root span).  ParentBased ensures
        # child spans follow the root decision so traces are never split.
        sampler_ratio = float(os.getenv("OTEL_TRACES_SAMPLER_ARG", "1.0"))
        sampler = ParentBased(TraceIdRatioBased(sampler_ratio)) if sampler_ratio < 1.0 else None
        provider = TracerProvider(
            resource=resource,
            **({} if sampler is None else {"sampler": sampler}),
        )
        if sampler is not None:
            logger.info("OTEL head-based sampling active — ratio=%.4f", sampler_ratio)

        bsp_kwargs = dict(
            max_queue_size=int(
                os.getenv("OTEL_BSP_MAX_QUEUE_SIZE", "4096")
            ),
            schedule_delay_millis=int(
                os.getenv("OTEL_BSP_SCHEDULE_DELAY_MILLIS", "5000")
            ),
            max_export_batch_size=int(
                os.getenv("OTEL_BSP_MAX_EXPORT_BATCH_SIZE", "1024")
            ),
            export_timeout_millis=int(
                os.getenv("OTEL_BSP_EXPORT_TIMEOUT_MILLIS", "30000")
            ),
        )

        # Gzip compression cuts LLM prompt/completion span sizes by 70-90 %, reducing
        # both network transfer and otel-collector memory pressure.
        # Endpoint is read from OTEL_EXPORTER_OTLP_ENDPOINT; the SDK appends /v1/traces.
        provider.add_span_processor(
            BatchSpanProcessor(
                OTLPSpanExporter(compression=Compression.Gzip),
                **bsp_kwargs,
            )
        )

        trace.set_tracer_provider(provider)
        _provider = provider

        # Graceful shutdown on process exit: force_flush delivers in-flight
        # spans, then shutdown() stops the background export thread.
        atexit.register(_shutdown_on_exit)

        logger.info(
            "OTEL tracing initialised — collector: %s",
            os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318"),
        )
        _auto_instrument()

    except Exception as exc:
        logger.warning(
            "OTEL tracing initialisation failed (tracing disabled): %s", exc
        )


def shutdown() -> None:
    """Flush and shut down the active TracerProvider.

    Performs a ``force_flush`` first so any in-flight batch is delivered before
    the background export thread is stopped.  Safe to call even if
    ``init_tracing()`` was never called or already shut down.
    """
    global _provider
    if _provider is not None:
        try:
            _provider.force_flush(timeout_millis=5000)
        except Exception as exc:
            logger.debug("OTEL force_flush error: %s", exc)
        try:
            _provider.shutdown()
            logger.debug("OTEL TracerProvider shut down cleanly")
        except Exception as exc:
            logger.warning("OTEL shutdown error: %s", exc)
        finally:
            _provider = None


def get_tracer_provider():
    """Return the active ``TracerProvider``, or ``None`` if not yet initialised."""
    return _provider


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _register_noop() -> None:
    """Set a NoOpTracerProvider as the global provider."""
    try:
        from opentelemetry import trace
        from opentelemetry.trace import NoOpTracerProvider
        trace.set_tracer_provider(NoOpTracerProvider())
    except Exception:
        pass


def _shutdown_on_exit() -> None:
    """atexit handler — delegates to shutdown()."""
    shutdown()


def _auto_instrument() -> None:
    """Activate available OpenInference instrumentors.

    Each instrumentor is guarded with a separate try/except so a missing
    optional package never prevents the others from running.
    """
    try:
        from openinference.instrumentation.langchain import LangChainInstrumentor
        LangChainInstrumentor().instrument()
        logger.debug("openinference: LangChainInstrumentor activated")
    except ImportError:
        pass

    try:
        from openinference.instrumentation.openai import OpenAIInstrumentor
        OpenAIInstrumentor().instrument()
        logger.debug("openinference: OpenAIInstrumentor activated")
    except ImportError:
        pass

    try:
        from openinference.instrumentation.anthropic import AnthropicInstrumentor
        AnthropicInstrumentor().instrument()
        logger.debug("openinference: AnthropicInstrumentor activated")
    except ImportError:
        pass
