FROM otel/opentelemetry-collector-contrib:latest

RUN apk add --no-cache python3 py3-pip && \
    pip3 install jinja2 --break-system-packages

COPY collector.config.yaml.j2 /etc/otel/collector.config.yaml.j2
COPY generate_config.py        /etc/otel/generate_config.py

ENTRYPOINT ["python3", "/etc/otel/generate_config.py"]
