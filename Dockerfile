FROM python:3.14-alpine

WORKDIR /app

# Build deps for psycopg (no 3.14 binary wheel — compiled from source).
# Runtime deps for weasyprint (cairo/pango) are also included; omit them
# if PDF rendering is unused in this environment.
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    libffi-dev \
    libcairo2 \
    libpango-1.0-0 \
    libpangocairo-1.0-0 \
    libgdk-pixbuf-2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY . .

# Install main deps + postgres checkpointer + dev CLI (langgraph-cli[inmem]).
# postgres extra: langgraph-checkpoint-postgres + psycopg[binary,pool].
RUN pip install --no-cache-dir -e ".[dev,postgres]"

# NQPR_CHECKPOINTER_BACKEND is read by src/config.py (main.py CLI runs).
# DATABASE_URL is read by `langgraph dev` as the --postgres-uri value
# (langgraph-cli uses DATABASE_URL as the envvar for that option).
ENV NQPR_CHECKPOINTER_BACKEND=postgres
ENV PYTHONUNBUFFERED=1

CMD ["langgraph", "dev", "--host", "0.0.0.0", "--port", "8125", "--no-browser"]
