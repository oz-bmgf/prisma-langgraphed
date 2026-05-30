# Python Alpine provides a stable pip environment for Jinja2.
# The OTel collector binary is a statically-linked Go binary — it runs on Alpine without issues.
FROM python:3.14-alpine
RUN pip install --no-cache-dir jinja2

COPY --from=otel/opentelemetry-collector-contrib:latest /otelcol-contrib /otelcol-contrib

COPY collector.config.yaml.j2 /etc/otel/collector.config.yaml.j2
COPY generate_config.py        /etc/otel/generate_config.py

ENTRYPOINT ["python3", "/etc/otel/generate_config.py"]
