#!/usr/bin/env python3
"""
Reads env vars, renders collector.config.yaml.j2 → collector.config.yaml,
then exec's the collector process so it becomes PID 1.
"""
import base64
import os
import sys

from jinja2 import Environment, FileSystemLoader


def required(key: str) -> str:
    val = os.environ.get(key, "").strip()
    if not val:
        print(f"[generate_config] WARNING: {key} is not set", flush=True)
    return val


def bool_env(key: str, default: bool) -> bool:
    return os.environ.get(key, str(default)).strip().lower() not in ("false", "0", "no")


phoenix_enabled  = bool_env("PHOENIX_TRACING_ENABLED", True)
langfuse_enabled = bool_env("LANGFUSE_TRACING_ENABLED", True)

# Auth: prefer pre-built LANGFUSE_OTEL_BASIC_AUTH; fall back to building from key pair.
langfuse_auth = ""
if langfuse_enabled:
    langfuse_auth = os.environ.get("LANGFUSE_OTEL_BASIC_AUTH", "").strip()
    if not langfuse_auth:
        pub = required("LANGFUSE_PUBLIC_KEY")
        sec = required("LANGFUSE_SECRET_KEY")
        langfuse_auth = base64.b64encode(f"{pub}:{sec}".encode()).decode()

env  = Environment(loader=FileSystemLoader("/etc/otel"))
tmpl = env.get_template("collector.config.yaml.j2")

rendered = tmpl.render(
    phoenix_enabled      = phoenix_enabled,
    phoenix_endpoint     = required("PHOENIX_ENDPOINT") if phoenix_enabled else "",
    langfuse_enabled     = langfuse_enabled,
    langfuse_endpoint    = required("LANGFUSE_ENDPOINT") if langfuse_enabled else "",
    langfuse_auth_header = langfuse_auth,
)

out_path = "/etc/otel/collector.config.yaml"
with open(out_path, "w") as f:
    f.write(rendered)

print(
    f"[generate_config] Wrote {out_path} "
    f"(phoenix={phoenix_enabled}, langfuse={langfuse_enabled})",
    flush=True,
)

os.execv("/otelcol-contrib", ["/otelcol-contrib", f"--config={out_path}"])
