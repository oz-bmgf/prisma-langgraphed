# prisma-langgraphed

LangGraph rewrite of the PRISMA investment-portfolio analysis pipeline.

---

## 1. Prerequisites

- Python 3.11+
- Git
- `ANTHROPIC_API_KEY` — required for all LLM nodes
- `OPENAI_API_KEY` — optional; needed only for embedding queries against the local search index
- `qdrant-client` / `azure-search-documents` — only if using those search backends

---

## 2. Setup

### Clone and install

```bash
git clone <repo>
cd prisma-langgraphed
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

### Environment variables

```bash
cp .env.example .env
# Edit .env — at minimum set ANTHROPIC_API_KEY
```

All supported variables are documented in `.env.example`.

---

## 3. Run Tests

```bash
pytest tests/ -v --tb=short
# Expected: 250 tests, 0 failures
```

---

## 4. Create Mock Program Data

Generates a minimal `MOCK-ingested/` collection under `~/qpr-collections/`. Safe to run multiple times — idempotent.

```bash
python scripts/create_mock_data.py
```

Creates:
- `doc_list.json`, `investment_scoring.json`, `bow_investment_map.json`, `investment_bow_rows.json`, `investment_intelligence.json`
- `embedding_index/` — `chunks.json`, `vectors.npy`, `chunks.sqlite`, `config.json`
- `pages/` — one subdirectory per document with a `p1.txt` page file

---

## 5. Run the Pipeline on Mock Data

**Quick smoke test (no LLM calls):**

```bash
python scripts/run_mock_pipeline.py
```

Prints: documents loaded, investments, BOWs, `precheck_passed`, and a `thread_id` for resuming.

**Using the CLI (real LLM calls — incurs API costs):**

```bash
# Start a run — will interrupt before analyze and prompt for confirmation
python main.py run --program MOCK --run-name test-01

# Resume after an interrupt
python main.py resume --program MOCK --run-name test-01

# Check checkpoint state
python main.py status --program MOCK --run-name test-01

# Run without interactive prompts (e.g. in CI — aborts at first interrupt)
python main.py run --program MOCK --run-name ci-01 --abort-on-interrupt analyze
```

The default SQLite checkpoint DB is `/tmp/nqpr_checkpoints.db`. Override with `--db /path/to/db`.

---

## 6. Project Structure

```
prisma-langgraphed/
├── main.py                # CLI entry point (run / resume / status)
├── langgraph.json         # Graph registry for langgraph dev Studio
├── src/
│   ├── backends/          # SearchBackend implementations (local, qdrant, azure)
│   ├── core/
│   │   ├── llm_utils.py       # acall_llm — LangChain-native LLM gateway
│   │   ├── output_schemas.py  # Pydantic v2 models for structured LLM output
│   │   ├── telemetry.py       # OTel setup (setup_telemetry, get_tracer)
│   │   └── evidence_model.py
│   ├── graph/
│   │   ├── state.py       # WorkflowState, AnalyzeState, and all sub-states
│   │   ├── workflow.py    # Top-level graph compilation
│   │   ├── evidence_audit.py  # Diagnostic graph (§13)
│   │   ├── gs_verifier.py     # Gold-standard dual-verifier graph (§13)
│   │   ├── nodes/         # One file per pipeline stage
│   │   └── subgraphs/     # analyze, causal, research
│   ├── prompts/           # All LLM prompt string constants (no logic)
│   │   ├── analyze_prompts.py
│   │   ├── causal_prompts.py
│   │   ├── report_prompts.py
│   │   ├── research_prompts.py
│   │   └── tool_prompts.py
│   └── tools/             # LangGraph ToolNode tools
├── tests/
│   ├── core/
│   ├── nodes/
│   ├── subgraphs/
│   ├── tools/
│   └── test_workflow.py
├── scripts/
│   ├── create_mock_data.py
│   ├── run_mock_pipeline.py
│   └── smoke_test.py
├── AGENTS.md              # Claude Code instructions — read every session
├── ARCHITECTURE.md        # LangGraph design — read before any node work
└── CODEBASE_AUDIT.md      # Original system map — read before porting
```

---

## 7. Key Documents

| File | Purpose |
|---|---|
| [AGENTS.md](AGENTS.md) | Claude Code instructions — read every session |
| [ARCHITECTURE.md](ARCHITECTURE.md) | LangGraph design — read before any node work |
| [CODEBASE_AUDIT.md](CODEBASE_AUDIT.md) | Original system map — read before porting |

---

## 8. Adding a New Graph Node

1. Add output fields to `WorkflowState` in [src/graph/state.py](src/graph/state.py)
2. Create `src/graph/nodes/<name>.py` with `async def <name>(state, config) -> dict:`
3. Add node and edges to [src/graph/workflow.py](src/graph/workflow.py)
4. Write `tests/nodes/test_<name>.py`
5. Run `pytest tests/` to confirm nothing broken

---

## 9. Running the Graph

### Local (in-memory checkpointer)

```bash
source .env && langgraph dev
```

Studio opens at https://smith.langchain.com — connect to `http://localhost:8125`.

### Docker (Postgres checkpointer)

```bash
docker compose up --build
```

This starts two services defined in `docker-compose.yml`:
- **postgres** — Postgres 16, data persisted in the `postgres_data` named volume
- **graph** — the LangGraph dev server, serving all graphs from `langgraph.json`

Studio opens at https://smith.langchain.com — connect to `http://localhost:8125`.

Checkpoints survive container restarts because they are stored in Postgres (not in-memory).

To wipe all checkpoints and start fresh:

```bash
docker compose down -v   # removes the postgres_data volume
docker compose up --build
```

---

## 10. Observability

### LangSmith (zero-config tracing)

Set these env vars in `.env`:

```bash
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=<your-langsmith-key>
LANGCHAIN_PROJECT=nqpr-pipeline
```

Every LangChain/LangGraph call is automatically traced — no code changes required.

### LangGraph Studio (local graph visualization)

```bash
pip install -e ".[dev]"   # installs langgraph-cli[inmem]
langgraph dev             # serves Studio UI at http://localhost:8125
```

All 6 graphs are registered in `langgraph.json` and appear in the Studio sidebar.

### OpenTelemetry (console spans)

```python
from src.core.telemetry import setup_telemetry
setup_telemetry()  # call once at startup
```

### LLM call traces (JSONL)

Set `LLM_TRACE_FILE=/path/to/trace.jsonl` to log every `acall_llm` invocation as a structured JSONL line.

---

## 11. REST API

`src/api.py` exposes the pipeline over HTTP using FastAPI.

### Start the server

```bash
uvicorn src.api:app --reload
# API docs at http://localhost:8000/docs
```

### Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/runs` | Start a run — returns 202 with `thread_id` |
| `GET` | `/runs/{thread_id}` | Status snapshot (running / interrupted / done) |
| `POST` | `/runs/{thread_id}/resume` | Send `{"value": "approved"}` to resume an interrupt |
| `GET` | `/runs/{thread_id}/report` | Fetch the analyst report (`text/markdown`) |
| `GET` | `/runs/{thread_id}/stream` | SSE stream of node-completion events |
| `DELETE` | `/runs/{thread_id}` | Cancel a run (204) |

### Example: start a run

```bash
curl -s -X POST http://localhost:8000/runs \
  -H "Content-Type: application/json" \
  -d '{
    "program": "MOCK",
    "run_name": "api-test-01",
    "collection_name": "MOCK-ingested",
    "base_dir": "~/qpr-collections",
    "ingested_dir": "~/qpr-collections/MOCK-ingested"
  }' | jq .
# {"thread_id": "MOCK::api-test-01", "status": "started"}
```

### Example: stream events

```bash
curl -N http://localhost:8000/runs/MOCK::api-test-01/stream
# data: {"event": "node_complete", "node": "load_collection"}
# data: {"event": "interrupt", "node": "analyze"}
```

### Example: resume after interrupt

```bash
curl -s -X POST http://localhost:8000/runs/MOCK::api-test-01/resume \
  -H "Content-Type: application/json" \
  -d '{"value": "approved"}'
```

### Checkpoint DB

Checkpoints are stored in `~/.nqpr_checkpoints/checkpoints.db` by default. Override with `NQPR_CHECKPOINT_DB=/path/to/db`.
