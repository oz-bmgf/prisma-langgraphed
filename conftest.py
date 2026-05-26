"""Root conftest — applied to all tests.

Sets OTEL_SDK_DISABLED before any test module is imported so that
init_tracing() returns early and no spans are emitted to a real collector.
"""
import os

os.environ.setdefault("OTEL_SDK_DISABLED", "true")
