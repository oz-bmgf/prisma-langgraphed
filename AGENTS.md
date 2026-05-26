# AGENTS.md — Claude Code Implementation Guide

Read ARCHITECTURE.md and CODEBASE_AUDIT.md before writing any code. The rules in this file take precedence over general intuition when there is a conflict.

---

## 1. PROJECT SUMMARY

NQPR is a quarterly portfolio review pipeline for a global health foundation. Each run analyses an investment portfolio — typically 20–60 investments across several thematic "bundles of work" (BOWs) — and produces a structured analyst report, causal assessment, and final PDF for human review and delivery.

This repo is a full LangGraph-native rewrite. The original system lives in `../prisma-ai-review/` (referred to as `old-repo` throughout this file) — it is read-only reference material. Never import from it in production code. Never copy code verbatim from it; re-implement for the new patterns.

**Scope:** Precheck through Finalize. The vector store and collection artifacts (`{program}-ingested/`) are assumed to exist on disk before the pipeline runs. Ingestion is out of scope.

> **⚠ Required fix to ARCHITECTURE.md §1 — model fields missing from WorkflowState**
>
> `AnalyzeState` (§3) and `CausalState` (§4) both carry `research_model: str` and `synthesis_model: str`. These fields are absent from `WorkflowState` in the current ARCHITECTURE.md §1 TypedDict listing.
>
> **Resolution:** Add both fields to `WorkflowState` under the "Run identity" section:
>
> ```python
> # ── Run identity ──────────────────────────────────────────────────────────
> program: str
> run_name: str
> collection_name: str
> base_dir: str
> output_dir: Optional[str]
> focus: Optional[str]
> focus_bows: Optional[list[str]]
> aux_collections: Optional[list[str]]
> research_model: str      # ← ADD: model used for investigation/research LLM calls
> synthesis_model: str     # ← ADD: model used for synthesis and report-writing LLM calls
> ```
>
> **Why `WorkflowState`, not `RunnableConfig`:** These fields must be checkpointed — if the graph is interrupted and resumed, model selection must survive the checkpoint round-trip. `RunnableConfig` values are ephemeral and must be re-supplied on every `.invoke`/`.astream` call. They also need to flow into subgraphs via explicit input field mapping, which is not possible for configurable values. Add them to the graph invocation's initial state dict alongside `program`, `run_name`, etc.
>
> **Do not read model names from `RunnableConfig` inside node bodies.** Always read from `state["research_model"]` and `state["synthesis_model"]`.

---

## 2. SKILLS IN USE

- **langgraph-fundamentals:** `StateGraph`, nodes, edges, conditional edges, `TypedDict` state reducers, subgraph compilation, `max_concurrency`
- **langgraph-persistence:** `AsyncSqliteSaver`, `thread_id = f"{program}::{run_name}"`, checkpoint resumability, idempotent disk artifacts
- **langgraph-human-in-the-loop:** `interrupt()`, `Command(resume=...)`, `interrupt_before`, three interrupt points (pre-analyze, review-research-plan, approve-report)
- **langchain-fundamentals:** `@tool`, `ToolNode`, async tool patterns, `RunnableConfig`, `config["configurable"]` dependency injection

---

## 3. REPOSITORY LAYOUT

```
src/
  config.py              ← all runtime constants, reads from env
  graph/
    state.py             ← ALL TypedDicts: WorkflowState, AnalyzeState,
                           CausalState, ResearchDispatchState, and all
                           five Send() sub-states (InvestmentRubricState,
                           LinkInvestigationState, ScienceAssumptionState,
                           ResearchTaskState, ScopeDecisionState)
    workflow.py          ← top-level graph compilation and CLI entrypoint
    nodes/               ← one file per top-level node
      load_collection.py
      precheck.py
      analyze.py
      prepare_research.py
      review_research_plan.py
      research.py
      finalize.py
      approve_report.py
      deliver.py
    subgraphs/
      analyze.py         ← 6-node analyze subgraph
      causal.py          ← 18-node causal pipeline subgraph
      research.py        ← 6-node research dispatch subgraph
  tools/
    collection_tools.py  ← CollectionToolNode (6 tools: search + retrieval only)
    investigation_tools.py  ← InvestigationToolNode (10 tools)
    science_tools.py     ← ScienceToolNode (extends Investigation + search_asta)
    narration_tools.py   ← NarrationToolNode (6 tools)
  backends/
    base.py              ← SearchBackend protocol + SearchResult dataclass
    local.py             ← LocalSearchIndex (numpy + SQLite)
    qdrant.py            ← QdrantCollectionSearchIndex
    azure.py             ← AzureSearchIndex
    factory.py           ← build_search_backend() factory
  core/
    checkpointer.py      ← build_checkpointer() factory; get/list/delete helpers
                           — no other file may import AsyncSqliteSaver/AsyncPostgresSaver directly
    asta_client.py       ← thin re-export of AstaClient for science_tools.py compat
    agents/
      __init__.py
      asta.py            ← async AstaClient (JSON-RPC 2.0 → Semantic Scholar fallback)
      openalex.py        ← async OpenAlexClient (cursor-based, rate-limited)
      deep_web.py        ← deep_web_research() + run(); o3-deep-research primary, GPT-4o fallback
      edison_rewriter.py ← rewrite_query() + rewrite_batch() via acall_llm
      slr.py             ← SLR agent: search OpenAlex+Asta → synthesise via acall_llm
      lbd.py             ← LBD agent: Swanson ABC model via Asta + acall_llm
      edison.py          ← EdisonLiteratureClient; lazy import of edison_client SDK
tests/
  nodes/
  subgraphs/
  tools/
  backends/
  core/
    agents/              ← unit tests for each agent in src/core/agents/
```

---

## 4. MIGRATION RULES — NEVER VIOLATE

These rules exist because the old codebase used patterns that do not translate cleanly to LangGraph. Violating any of them produces state-merging bugs, checkpointing failures, or non-resumable graphs.

| Rule | Rationale |
|---|---|
| Never hardcode model names, paths, limits, or thresholds in node or core files — import from `src.config` | Centralised config allows runtime overrides via env vars and prevents scattered magic values |
| All node functions must be `async def` | LangGraph graph execution is async-native; sync nodes block the event loop |
| Never use `ThreadPoolExecutor` for parallelism | Use `Send()` for all graph-level parallelism; `asyncio.to_thread` for blocking I/O only |
| All inter-node data flows through `WorkflowState` | No globals, no side-channel files as primary data flow; checkpointer can only snapshot state |
| Every node returns a `dict` of only the fields it modifies | Never return the full state object — LangGraph merges partial dicts via reducers |
| Disk writes are for caching and resume only | Not primary data flow; a node must be correct even if its disk artifact is deleted |
| All LLM calls go through `acall_llm` | Never call Anthropic or OpenAI SDKs directly from node or tool code |
| Search always goes through `search_collection` tool | Never instantiate a `SearchBackend` directly in node or tool code; backends are DI-injected via `config["configurable"]` |
| Dispatch nodes return `list[Send]` only | They are pure routers; they must never write to state |
| Worker nodes return `{"reducer_field": [single_result]}` only | One item per worker; `operator.add` reducer accumulates across all parallel branches |
| Collect/reducer nodes do aggregation only | No LLM calls; they run once after all parallel branches complete |
| `load_collection` resumability via early return | First line of the node body: `if state.get("doc_list") is not None: return {}` — do not use a conditional edge before the node |
| `interrupt()` is synchronous — never `await` it | LangGraph intercepts the call as a sentinel; `await interrupt(...)` will raise or hang |
| `interrupt()` pattern: inline state mutation | Call `interrupt(payload)`, read the resume value, mutate state if needed (e.g. prune `research_plan`), return the state update dict. Never implement interrupt logic as a conditional edge |
| `from __future__ import annotations` is the first line of `state.py` | Python evaluates `TypedDict` field annotations at class-definition time in 3.9–3.11; without this, `Annotated[...]` and forward references raise `NameError` |
| Fan-out reducer fields must be initialised to `[]`, not `None` | `operator.add` has no base case for `None`; initialise all `Annotated[list, operator.add]` fields to `[]` in the initial state dict |
| State fields are the primary data carrier — never write intermediate data to disk and pass the path through state | Intermediate JSON artifacts on disk are not checkpointed. If a node writes `foo.json` and returns `foo_path`, the resume path skips that node and the data is gone. Write data to state directly. Only disk-write human-readable deliverables (`.md`, `.pdf`, `.csv`) that are the intended output of the pipeline, never intermediate machine-readable data. |
| No direct imports of `AsyncSqliteSaver` or `AsyncPostgresSaver` outside `src/core/checkpointer.py` | All checkpointer construction goes through `build_checkpointer()`. This keeps backend selection centralised and makes it trivial to swap backends without touching callers. |
| `compile_graph()` runs unattended by default — never enable `interrupt_before` nodes in production code paths | Pass `human_interrupts=True` (or `--interactive` CLI flag) only for manual sessions. The graph must complete end-to-end without human intervention for CI, API, and scheduled runs. |
| Research agent workers in `research.py` import from `src.core.agents.<module>` — never use the old `from src.core import slr_agent` lazy-import pattern | The agents are now real modules; the old pattern was a placeholder for missing implementations |
| All new model names and timeouts for research agents come from `src.config` | `DEEP_WEB_PRIMARY_MODEL`, `DEEP_WEB_TIMEOUT_SECONDS`, `SLR_TIMEOUT_SECONDS`, `LBD_TIMEOUT_SECONDS`, `EDISON_TIMEOUT_SECONDS`, `ASTA_API_KEY`, `OPENALEX_EMAIL` etc. |
| Never use `asyncio.Semaphore`, `asyncio.Lock`, `asyncio.Event`, or `asyncio.BoundedSemaphore` anywhere in `src/` | Rate-limiting is handled by `max_concurrency` on `compile()`; intra-node synchronisation primitives hide concurrency bugs from LangGraph's scheduler |
| Every `asyncio.to_thread`, `asyncio.gather`, `asyncio.wait_for`, `asyncio.sleep`, `asyncio.run`, `asyncio.Queue`, `asyncio.Task`, `asyncio.create_task`, `asyncio.TimeoutError`, and `asyncio.CancelledError` usage must have an `# asyncio-APPROVED-N` comment on the line immediately before it | Enables static grep audits; see §9 for the full policy |
| `asyncio.create_task` and `asyncio.Queue` are restricted to `src/api.py` only | Bare task creation and queue usage outside the SSE infrastructure layer bypasses LangGraph's scheduler and breaks checkpoint resumability |
| Every `except asyncio.CancelledError` block must end with a bare `raise` | Swallowing `CancelledError` prevents task cancellation from propagating, causing resource leaks and hang-on-shutdown |

---

## 5. IMPLEMENTATION ORDER — follow exactly, do not skip phases

### Phase A: `src/graph/state.py`

Define all TypedDicts in a single file in this order. No implementation logic — types only.

**Required order within the file:**

```
from __future__ import annotations   ← absolute first line, no exceptions

import operator
from typing import Annotated, Optional, TypedDict

# 1. Sub-state slices (Send() payloads)
InvestmentRubricState
LinkInvestigationState
ScienceAssumptionState
ResearchTaskState
ScopeDecisionState

# 2. Top-level WorkflowState  ← include research_model and synthesis_model (see §1 gap note)
WorkflowState

# 3. Subgraph states (standalone TypedDicts — not subclasses)
AnalyzeState
CausalState
ResearchDispatchState
```

`WorkflowState` must include `research_model: str` and `synthesis_model: str` under the Run identity block (see §1 gap note above). This is a correction to ARCHITECTURE.md §1 and must be applied here before any node is written.

`CausalState` formal definition:

```python
class CausalState(TypedDict):
    # Inputs (passed in from AnalyzeState)
    scopes: list[dict]
    scope_timelines: dict
    research_model: str
    synthesis_model: str
    cache_dir: str
    # Fan-out reducer fields
    evidence_packs: Annotated[list[dict], operator.add]
    link_assessments: Annotated[list[dict], operator.add]
    science_results: Annotated[list[dict], operator.add]
    scope_decisions: Annotated[list[dict], operator.add]
    # Progressive outputs
    scope_outputs: list[dict]
    # Error accumulator
    errors: Annotated[list[str], operator.add]
```

`ResearchDispatchState` formal definition:

```python
class ResearchDispatchState(TypedDict):
    # Inputs (passed in from WorkflowState)
    research_plan: list[dict]
    research_dir: str
    # Fan-out reducer field
    research_results: Annotated[list[dict], operator.add]
    # Aggregate outputs
    dispatch_results: Optional[list[dict]]
    edison_results: Optional[list[dict]]
    # Error accumulator
    errors: Annotated[list[str], operator.add]
```

Do not proceed to Phase B until `python -c "from src.graph.state import WorkflowState, AnalyzeState, CausalState, ResearchDispatchState"` exits with code 0.

---

### Phase B: `src/backends/`

Write `base.py` first (the `SearchBackend` protocol and `SearchResult` dataclass), then the three concrete backends, then the factory. Every backend must implement the `SearchBackend` protocol exactly — the `@runtime_checkable` decorator will catch missing methods at startup.

`base.py` must define:

```python
from typing import Optional, Protocol, runtime_checkable
from dataclasses import dataclass

@dataclass
class SearchResult:
    chunk_id: str
    text: str
    score: float
    file_id: str
    inv_id: Optional[str]
    bow_id: Optional[str]
    page_start: Optional[int]
    page_end: Optional[int]
    doc_type: Optional[str]

@runtime_checkable
class SearchBackend(Protocol):
    async def search(
        self,
        query: str,
        *,
        top_k: int = 20,
        collection_filter: Optional[str] = None,
        bow_id_filter: Optional[str] = None,
        inv_id_filter: Optional[str] = None,
        doc_type_filter: Optional[str] = None,
    ) -> list[SearchResult]: ...

    async def distinct_inv_ids(self) -> list[str]: ...
    async def distinct_bow_ids(self) -> list[str]: ...
    async def count_by_bow_id(self) -> dict[str, int]: ...
```

`factory.py` must read `NQPR_SEARCH_BACKEND` from the environment (`"local"` | `"qdrant"` | `"azure"`) and return the appropriate backend. Default to `"local"`. Store the result in `config["configurable"]["search_backend"]` at graph startup — never pass it as a function argument.

---

### Phase C: `src/tools/`

All four ToolNode groups. Every tool is `async def` decorated with `@tool`. All search calls go through `search_collection`. Unit tests use a mock `SearchBackend` fixture that returns fixed `list[SearchResult]`.

Read `../prisma-ai-review/src/qpr/investigation_loop.py`, `collection_api.py`, `science_investigator.py`, and `narration_tools.py` to understand the tool contracts before implementing. Re-implement — do not copy verbatim.

`CollectionToolNode` exposes exactly 6 tools: `search_collection`, `read_section`, `read_pages`, `read_key_docs`, `read_page_image`, `get_page_images_for_section`. No scoring lookups, no navigation helpers, no boolean flags.

---

### Phase D: `src/core/`

Port non-orchestration logic from `../prisma-ai-review/`. This includes rubric evaluation, investigation loop, science investigator, decision projection, report assembler, exec summary passes, and narration logic.

**No LangGraph imports in `core/`.** These are pure business-logic functions. They receive typed inputs and return typed outputs. Nodes in Phase E–I will call them via `asyncio.to_thread` or directly if they are already async.

Read the corresponding original file first. Understand the algorithm. Then re-implement against the new state types and async conventions.

#### `src/core/science_investigator.py` — `search_asta` hard requirement

The investigation loop must track three boolean flags at loop scope across all iterations:

```python
asta_called_ever: bool = False     # True as soon as any search_asta action executes
confirming_found: bool = False     # True when LLM reports confirming_evidence_found=true
                                   # OR when a new chunk contains confirming language
disconfirming_found: bool = False  # same pattern for disconfirming evidence
```

On every iteration prompt, these three flags must be surfaced as **TERMINATION GATES**. If `asta_called_ever` is `False`, the gate note must be:

```
- search_asta has NOT been called yet — you must call it before terminating
```

The `search_asta` tool itself may have a fallback (Semantic Scholar public API, or a graceful "unavailable" message) — that is correct and fine. The loop-level enforcement is what matters: the model must be told in every iteration prompt whether the ASTA gate has been satisfied, so it cannot emit `status=evidence_gathered` without having attempted at least one ASTA call.

**This enforcement is prompt-based, not a hard server-side code gate.** The old code trusts the model to honour the system prompt constraint ("HARD REQUIREMENTS before emitting status=evidence_gathered: You MUST have called search_asta at least once"). Do not add a code override that rejects `evidence_gathered` if `asta_called_ever` is False — the model's structured output is trusted. The gate notes are sufficient.

Also preserve:
- `asta_soft_cap: int = 5` — maximum ASTA calls per question (skip further ASTA actions once reached, but do not block termination)
- `consecutive_empty: int` counter — 3 rounds with zero new chunks → force `terminal_status = "insufficient_evidence"` without waiting for the model to declare it
- Per-round `asta_called` flag (int 1/0) distinct from the cumulative `asta_called_ever` — the executor increments `asta_calls` only when an ASTA action actually executes (not merely when the model says `asta_called=true`)

---

### Phase E: `src/graph/subgraphs/causal.py`

All 18 nodes per ARCHITECTURE.md §4. Compile with:

```python
causal_graph.compile(
    checkpointer=...,
    max_concurrency=16,   # for investigate_link
)
```

Use separate `max_concurrency=8` for the rubric evaluation and science investigation fan-outs if LangGraph supports per-node caps at the time of implementation; otherwise apply the tightest global cap that keeps the event loop stable.

> **Note — LangGraph 1.2.1:** `max_concurrency` is not a valid `compile()` kwarg in this version and raises `TypeError`. Concurrency is managed by the event loop. If upgrading LangGraph, check whether `max_concurrency` support has been added and wire it then.

Each worker node must check for a cached result in `cache_dir` before calling the underlying async function. The cache key is `f"{cache_dir}/{scope_id}/{inv_id}.json"` (or equivalent per worker type).

---

### Phase F: `src/graph/subgraphs/research.py`

6 nodes per ARCHITECTURE.md §5. `fan_out_research_tasks` routes by `task_type` to four typed worker nodes. Each worker caches its result to `research_dir/{linked_scope}/{task_id}/result.json`.

---

### Phase G: `src/graph/subgraphs/analyze.py`

6 nodes per ARCHITECTURE.md §3. `run_causal_pipeline` invokes the causal subgraph via `await causal_subgraph.ainvoke(sub_state, config)`. The input projection maps `AnalyzeState` fields to `CausalState` fields by name — confirm field name alignment before wiring.

**Optional → required projection note:** `WorkflowState` declares `doc_list: Optional[list[dict]]` (and similarly for the other collection input fields) because they are `None` before `load_collection` runs. `AnalyzeState` declares them non-optional (`doc_list: list[dict]`) because by the time the analyze subgraph is invoked, `load_collection` is guaranteed to have populated them. This asymmetry is intentional. The `analyze` node in `src/graph/nodes/analyze.py` (Phase I) is responsible for the projection; it must assert non-`None` before constructing the `AnalyzeState` input dict:

```python
assert state["doc_list"] is not None, "load_collection must run before analyze"
assert state["investment_scoring"] is not None
assert state["bow_investment_map"] is not None
assert state["investment_intelligence"] is not None
assert state["chunks_json_path"] is not None
assert state["pages_dir"] is not None
```

This makes the contract explicit and produces a clear error if the graph topology is wired incorrectly.

---

### Phase H: `src/graph/workflow.py`

Top-level graph with `AsyncPostgresSaver` checkpointer (Postgres, not SQLite).

Requires: `langgraph-checkpoint-postgres` + `psycopg[binary]`

```python
thread_id = f"{program}::{run_name}"

graph = StateGraph(WorkflowState)
# ... add nodes and edges per ARCHITECTURE.md §2 ...
compiled = graph.compile(
    checkpointer=checkpointer,  # AsyncPostgresSaver — see make_checkpointer() below
    interrupt_before=["analyze", "review_research_plan", "research", "approve_report"],
)
```

`AsyncPostgresSaver.from_conn_string` is an **async context manager** (not a direct
call like SQLite). Use the `make_checkpointer()` helper in `workflow.py`:

```python
async with make_checkpointer(conn_string) as cp:   # calls cp.setup() automatically
    graph = compile_graph(cp)
    await graph.ainvoke(initial_state, {"configurable": {"thread_id": thread_id}})
```

Connection string is read from env var `NQPR_CHECKPOINT_DB` (default:
`postgresql://localhost/nqpr_checkpoints`).

Initial state dict must initialise all `Annotated[list, operator.add]` fields to `[]`:

```python
initial_state = {
    "program": program,
    "run_name": run_name,
    "collection_name": collection_name,
    "base_dir": base_dir,
    "ingested_dir": ingested_dir,
    "research_model": research_model,
    "synthesis_model": synthesis_model,
    # fan-out reducers — must be [] not None
    "evidence_packs": [],
    "link_assessments": [],
    "science_results": [],
    "scope_decisions": [],
    "research_results": [],
    "errors": [],
    # all Optional fields default to None implicitly
}
```

---

### Phase I: `src/graph/nodes/`

One file per node. Start with `load_collection.py` and `precheck.py` as they are the simplest to validate end-to-end.

**`load_collection` canonical pattern:**

```python
async def load_collection(state: WorkflowState, config: RunnableConfig) -> dict:
    if state.get("doc_list") is not None:
        return {}   # already loaded in a prior run; checkpoint restored the fields

    ingested_dir = Path(state["ingested_dir"])
    doc_list = await asyncio.to_thread(_read_json, ingested_dir / "doc_list.json")
    investment_scoring = await asyncio.to_thread(_read_json, ingested_dir / "investment_scoring.json")
    bow_investment_map = await asyncio.to_thread(_read_json, ingested_dir / "bow_investment_map.json")
    investment_bow_rows = await asyncio.to_thread(_read_json, ingested_dir / "investment_bow_rows.json")
    investment_intelligence = await asyncio.to_thread(_read_json, ingested_dir / "investment_intelligence.json")
    return {
        "doc_list": doc_list,
        "investment_scoring": investment_scoring,
        "bow_investment_map": bow_investment_map,
        "investment_bow_rows": investment_bow_rows,
        "investment_intelligence": investment_intelligence,
        "chunks_json_path": str(ingested_dir / "embedding_index" / "chunks.json"),
        "pages_dir": str(ingested_dir / "pages"),
    }
```

> **Note — `investment_bow_rows.json` may be absent from older collections.** If the file is missing, derive it from `bow_investment_map.json` (which is always present) before the graph runs:
>
> ```python
> rows = []
> for bow_id, bow_data in bow_map.items():
>     inv_ids = bow_data.get("inv_ids", []) if isinstance(bow_data, dict) else bow_data
>     for inv_id in inv_ids:
>         rows.append({"bow_id": bow_id, "inv_id": inv_id})
> (ingested_dir / "investment_bow_rows.json").write_text(json.dumps(rows, indent=2))
> ```
>
> This derivation is idempotent — running it twice produces the same file. For a 35-BOW collection the expected row count is ~740. The `scripts/smoke_test.py` performs this derivation automatically in its setup step.

**`review_research_plan` canonical pattern:**

```python
from langgraph.types import interrupt

async def review_research_plan(state: WorkflowState, config: RunnableConfig) -> dict:
    resume_value = interrupt({          # NOT awaited — synchronous sentinel
        "stage": "review_research_plan",
        "program": state["program"],
        "research_plan_md_path": state["research_plan_md_path"],
        "task_count": len(state["research_plan"]),
        "question": (
            "Review and edit research_plan.md, then reply:\n"
            "  'approve'               — dispatch all tasks\n"
            "  'regenerate'            — rebuild the plan\n"
            "  {'prune': [task_id...]} — remove specific tasks, then dispatch"
        ),
    })

    if resume_value == "approve":
        return {"research_plan_approved": True}

    if resume_value == "regenerate":
        return {"research_plan_approved": False}

    if isinstance(resume_value, dict) and "prune" in resume_value:
        prune_ids = set(resume_value["prune"])
        return {
            "research_plan": [t for t in state["research_plan"] if t["id"] not in prune_ids],
            "research_plan_approved": True,
        }

    return {"research_plan_approved": True}   # unrecognised input → treat as approve
```

---

## 6. TESTING CONVENTIONS

- `pytest` + `pytest-asyncio` for all async tests — set `asyncio_mode = "auto"` in `pyproject.toml` (`[tool.pytest.ini_options]`) or `pytest.ini`, otherwise async tests are silently collected but never actually run
- Mock LLM calls with a fixture that returns a fixed canned string — never make real API calls in tests
- Mock `SearchBackend` with a fixture that returns a fixed `list[SearchResult]` — never hit a real vector store in tests
- `tests/` mirrors `src/` structure exactly: `tests/nodes/` covers `src/graph/nodes/`, etc.
- Every node: unit test with a mock `WorkflowState` dict; assert that the returned dict contains exactly the expected keys and no others
- Every `Send()` dispatch node: assert the correct `Send()` count, correct node name, and correct sub-state field values
- Every tool: unit test with a mock `SearchBackend`; assert that `search_collection` calls `backend.search` with the expected parameters
- No real LLM, no real filesystem I/O, no real network calls in any test

---

## 7. KEY FILES — read at the start of every session

| File | When to read |
|---|---|
| `AGENTS.md` (this file) | Every session |
| `src/config.py` | All runtime constants — import from here, never hardcode |
| `ARCHITECTURE.md` | Before implementing any node, subgraph, or state type |
| `CODEBASE_AUDIT.md` | Before porting any logic from old-repo; covers every module's inputs, outputs, and dependencies |
| `src/graph/state.py` | Before writing any node or tool that reads state |
| `src/core/llm_utils.py` | Before making any LLM call — use `acall_llm(prompt, system_msg="", *, model, output_schema=None, ...)` |
| `src/core/output_schemas.py` | When a node needs structured LLM output — pass a Pydantic model as `output_schema` |
| `src/prompts/` | Before writing any inline prompt string — add new prompts to the appropriate domain module |
| `main.py` | CLI entry point — `run`, `resume`, `status` subcommands |
| `langgraph.json` | Graph registry for `langgraph dev` Studio UI |
| `src/core/telemetry.py` | Call `setup_telemetry()` once at startup for OTel tracing |
| `src/api.py` | FastAPI app — 6 REST endpoints; background `asyncio.Task` per run; SSE stream; lifespan manages shared checkpointer + graph |

---

## 8. REFERENCE CODEBASE

`../prisma-ai-review/` is read-only. Read it to understand existing behaviour and data contracts. Never import from it. Never copy verbatim.

When implementing any node or core function, read the corresponding original file first, then re-implement against the new async, state-typed patterns. Key mappings (old file → new location):

| Old file | New location |
|---|---|
| `src/qpr/research_analyst_agent.py` | `src/graph/subgraphs/analyze.py` + `src/graph/nodes/analyze.py` |
| `src/qpr/causal_pipeline.py` | `src/graph/subgraphs/causal.py` |
| `src/qpr/investigation_loop.py` | `src/tools/investigation_tools.py` + `src/core/investigation.py` |
| `src/qpr/rubric_evaluator.py` | `src/core/rubric_evaluator.py` |
| `src/qpr/science_investigator.py` | `src/tools/science_tools.py` + `src/core/science_investigator.py` |
| `src/qpr/decision_projection.py` | `src/core/decision_projection.py` |
| `src/qpr/report_assembler.py` | `src/core/report_assembler.py` |
| `src/qpr/narration_tools.py` | `src/tools/narration_tools.py` |
| `src/qpr/llm_utils.py` | `src/core/llm_utils.py` (expose `acall_llm`) |
| `src/qpr/search/` | `src/backends/` |
| `src/qpr/deep_web_research.py` | `src/core/agents/deep_web.py` |
| `src/qpr/edison_query_rewriter.py` | `src/core/agents/edison_rewriter.py` |
| `src/search/paper_sources/asta_client.py` | `src/core/agents/asta.py` |
| `src/search/paper_sources/openalex_client.py` | `src/core/agents/openalex.py` |
| `src/search/paper_sources/edison_client.py` | `src/core/agents/edison.py` |
| `src/search/research_agent.py` | `src/core/agents/slr.py` |
| `src/search/lbd_agent.py` | `src/core/agents/lbd.py` |

### Research agent rules

| Rule | Rationale |
|---|---|
| All research agents live in `src/core/agents/` | The `research.py` subgraph workers import from `src.core.agents`; never import from `src.core` directly (old lazy-import pattern) |
| `deep_web.py` primary path uses `ChatOpenAI(use_responses_api=True, output_version="responses/v1").bind_tools([{"type": "web_search_preview"}])` — does NOT go through `acall_llm` | The OpenAI Responses API requires `use_responses_api=True`; `acall_llm` uses `ChatOpenAI` without that flag. The call is still LangChain-native and auto-instrumented. |
| All other agent LLM calls go through `acall_llm` | SLR synthesis, LBD concept extraction/synthesis, and query rewriting all use `acall_llm` |
| Agent `run()` functions never raise | Return a dict with `success=False` and `error_message` on any failure; the worker's try/except is the last safety net |
| `AstaClient` in `src/core/agents/asta.py` is the authoritative implementation | `src/core/asta_client.py` is a thin re-export for `science_tools.py` backward compatibility only |
| Test research workers by patching `src.core.agents.<module>.run` | Patching `sys.modules` does not work because the real modules are loaded at import time and their attributes are cached on the package object |

### Phase D ports — completed

The five core modules below have been ported from `../prisma-ai-review/src/qpr/` and are fully implemented. Read the corresponding file before modifying any of them.

#### `src/core/causal_model.py`

| Symbol | Contract |
|---|---|
| `extract_causal_model(scope, model)` | Pulls evidence chunks from `scope["evidence_packs"]`, calls LLM via `CausalModelExtraction` schema, then chains `rank_assumptions` and `forecast_consequences`. Returns `CausalModel`. |
| `rank_assumptions(cm, model)` | Classifies each assumption's consequence+uncertainty via LLM, maps through `_RISK_MATRIX` deterministically, sorts ascending by `risk_rank`. Returns updated `CausalModel`. |
| `forecast_consequences(cm, scoring, timeline, model)` | Calls LLM via `ForecastOutput` schema. `dollars_at_risk` is clipped at `approved_amount` (Basel-EAD). Aggregates with `max()` per link, never `sum()`. |
| `make_investigation_claims(cm, bow_id, inv_id)` | Returns list of claim dicts with `task_id="{inv_id}-assumption-{i+1:03d}"`. Adds `web_search_hint=True` when a science keyword is found in the assumption text. |
| `_RISK_MATRIX` | `dict[(consequence, uncertainty) → (label, sort_key)]`. 9 entries, sort_key 1–9 (lower = higher priority). Never bypass with hardcoded strings. |

Prompts live in `src/prompts/causal_prompts.py`: `CAUSAL_EXTRACTION_SYSTEM`, `CAUSAL_EXTRACTION_PROMPT`, `ASSUMPTION_RANKING_SYSTEM`, `ASSUMPTION_RANKING_PROMPT`, `CONSEQUENCE_FORECAST_SYSTEM`, `CONSEQUENCE_FORECAST_PROMPT`, `BOW_ENRICHMENT_SYSTEM`.

#### `src/core/investigation.py`

| Symbol | Contract |
|---|---|
| `run_investigation(link_id, inv_id, bow_id, scope_id, claim, model, *, tools, max_iterations)` | Multi-turn tool-calling loop. Calls `acall_llm` with `InvestigationActionsOutput` schema. L4 coverage audit gate (enabled via `NQPR_L4_COVERAGE_AUDIT=true`) validates 5-item checklist before accepting terminal status. `CONSECUTIVE_EMPTY_THRESHOLD` (3) consecutive empty rounds → force `insufficient_evidence`. Returns `InvestigationResult`. |
| `_execute_actions(actions, tools, inv_id, bow_id, model, facts)` | Fans out all tool actions concurrently via `asyncio.gather` (`# asyncio-APPROVED-2`). Each blocking search uses `asyncio.to_thread` (`# asyncio-APPROVED-1`). Returns `(new_chunks, new_asta_calls)`. |
| `_dedup_chunks(new_chunks, existing)` | Deduplicates by `file_id + chunk_id` (falls back to `page_start` then index). Used by both `investigation.py` and `science_investigator.py`. |
| `_SUPPORTED_TOOLS` | `frozenset` of 7 valid tool names. Actions with tools outside this set are silently skipped. |

Prompts live in `src/prompts/tool_prompts.py`: `INVESTIGATION_TOOL_DESCRIPTIONS`, `INVESTIGATION_SYSTEM`, `L4_COVERAGE_AUDIT_ITEMS`, `L4_COVERAGE_AUDIT_INSTRUCTION`.

Config constants: `INVESTIGATION_L4_COVERAGE_AUDIT`, `INVESTIGATION_L1_REASONING`, `CONSECUTIVE_EMPTY_THRESHOLD`.

#### `src/core/rubric_evaluator.py`

| Symbol | Contract |
|---|---|
| `build_evidence_pack(inv_id, scope_id, timeline, *, top_k, tools, model)` | 4-strategy evidence retrieval. Strategy 1: LLM generates 10 queries (`StrategyQueryList` schema). Strategy 2: 4 hardcoded fallback queries. Strategy 3: 3 doc-type-specific queries. Strategy 4: 5 strategy collection queries (`collection="strategy"`). All strategies fan out via `asyncio.gather` (`# asyncio-APPROVED-2`). Returns `InvestmentEvidencePack` with at least `_MIN_STRATEGY_CHUNKS = 20` strategy chunks. |
| `_detect_fact_contradictions(chunks, facts)` | Checks `approved_amount` with 10% tolerance across sources. |
| `_score_disbursement_velocity(facts, timeline)` | green (0.70≤ratio≤2.0) / yellow (underspend early or overspend mid) / red (ratio<0.40 past 25%). |
| `_compute_local_scores(chunks)` | `document_freshness` (≤6mo=green, ≤12mo=yellow, >12mo=red), `reporting_completeness` (progress_report present=green), `rationale_adequacy` (>500chars=green). |

#### `src/core/science_investigator.py`

| Symbol | Contract |
|---|---|
| `investigate_science_question(assumption_id, inv_id, bow_id, scope_id, question, *, asta_client, tools, model, max_iterations, asta_soft_cap)` | Calls `acall_llm` with `ScienceActionsOutput` schema. ASTA gate: if `status=evidence_gathered` with `asta_called_ever=False`, gate injects `search_asta` action and continues the loop. Soft cap (`ASTA_SOFT_CAP`, default 5): silently skips ASTA actions once reached. Consecutive empty (`CONSECUTIVE_EMPTY_THRESHOLD`, default 3): forces `insufficient_evidence`. Returns `ScienceInvestigationResult`. |
| `_call_asta(asta_client, query)` | Uses `inspect.iscoroutinefunction` (not the deprecated `asyncio.iscoroutinefunction`) to detect async `search` method. |
| `_build_gate_note(asta_called, confirming, disconfirming)` | Returns a TERMINATION GATES string injected into every iteration prompt. |

Imports `_dedup_chunks` and `_execute_actions` from `src.core.investigation`.

Prompt lives in `src/prompts/tool_prompts.py`: `SCIENCE_INVESTIGATE_SYSTEM`.

Config constants: `ASTA_SOFT_CAP`, `SCIENCE_MAX_ITERATIONS`, `CONSECUTIVE_EMPTY_THRESHOLD`.

> **ASTA gate is prompt-based, not a hard code gate.** The model's structured output is trusted. The gate note informs the model; the code only injects a forced action when `evidence_gathered` is returned with empty `next_actions` and `asta_called_ever=False`. Never add a code block that rejects `evidence_gathered` outright.

#### `src/core/decision_projection.py`

| Symbol | Contract |
|---|---|
| `project_decisions(scope_id, scope_output, *, model)` | Calls `acall_llm` with `DecisionProjectionOutput` schema. Sanitizes → §1a gates → ranks → caps. Returns `{"scope_id": scope_id, "decisions": [d.to_dict() for d in decisions]}`. |
| `_sanitize_candidate(candidate)` | Rejects if `decision_type` not in `DECISION_TYPE_VOCABULARY`, `recommended_action` is empty, or `triggering_link_ids` is empty. |
| `_section1a_gate(candidate)` | Passes if type in `_THIN_EVIDENCE_DECISION_TYPES` OR confidence ≠ "low" OR corroboration_count ≥ 2. |
| `_compute_rank_score(candidate)` | `cor × mat × max(log10(dollars+10), 1) × urg × evd`. Scalar maps: materiality high=3/med=2/low=1; urgency immediate=4/near_term=3/medium_term=2/long_term=1; confidence high=3/med=2/low=1. |
| `_apply_caps(decisions, inv_id)` | Max `DECISION_MAX_PER_INV` (3) per non-empty `inv_id`. Max `DECISION_MAX_PER_SCOPE` (8) total. Decisions with empty `inv_id` do not count against the per-INV quota. |
| `DECISION_TYPE_VOCABULARY` | `frozenset` of 15 valid type strings. |
| `_THIN_EVIDENCE_DECISION_TYPES` | `frozenset({"request_progress_packet", "validate_assumption"})` — bypass §1a gate. |

Config constants: `DECISION_MAX_PER_INV`, `DECISION_MAX_PER_SCOPE`.

#### Pydantic schemas added to `src/core/output_schemas.py`

`CausalLinkSchema`, `CausalModelExtraction`, `AssumptionRisk`, `RankedAssumptionsOutput`, `ConsequenceForecast`, `ForecastOutput`, `InvestigationAction`, `InvestigationActionsOutput`, `ScienceAction`, `ScienceActionsOutput`, `DecisionCandidate`, `DecisionProjectionOutput`, `StrategyQueryList`.

#### Config constants added to `src/config.py`

`INVESTIGATION_L4_COVERAGE_AUDIT`, `INVESTIGATION_L1_REASONING`, `ASTA_SOFT_CAP`, `SCIENCE_MAX_ITERATIONS`, `CONSECUTIVE_EMPTY_THRESHOLD`, `DECISION_MAX_PER_INV`, `DECISION_MAX_PER_SCOPE`.

#### `CausalModel` serialization rule

`CausalModel` (and `CausalLink`) are `@dataclass` types, not Pydantic models. They have **no `to_dict()` method**. When a causal.py node stores a `CausalModel` in `scope["causal_model"]` for downstream routing, it must convert it with `dataclasses.asdict()`:

```python
import dataclasses

if dataclasses.is_dataclass(cm) and not isinstance(cm, type):
    scope["causal_model"] = dataclasses.asdict(cm)
elif hasattr(cm, "to_dict"):
    scope["causal_model"] = cm.to_dict()
else:
    scope["causal_model"] = cm
```

Failing to do this causes `_route_link_investigations` to receive a dataclass object, fail the `isinstance(causal_model, dict)` check, and silently produce zero link investigation Send()s.

#### Test patching rule for `src/graph/subgraphs/causal.py` nodes

Node bodies use local imports (`from src.core import rubric_evaluator` inside the function body). Patching `sys.modules["src.core.rubric_evaluator"]` does **not** work because Python's `from package import module` uses `getattr(package, name)` first, which returns the already-bound attribute on `src.core`, bypassing `sys.modules`. Always patch the function directly:

```python
# CORRECT
with patch("src.core.rubric_evaluator.build_evidence_pack", new=AsyncMock(...)):
    ...

# WRONG — sys.modules patching bypassed by getattr on already-imported package
with patch.dict(sys.modules, {"src.core.rubric_evaluator": mock_mod}):
    ...
```

For integration tests that run `run_causal_pipeline` end-to-end, also patch each port module's local `acall_llm` binding (they are imported at module level with `from src.core.llm_utils import acall_llm`, so `patch("src.core.llm_utils.acall_llm", ...)` does **not** intercept them):

```python
with patch("src.graph.subgraphs.causal.acall_llm", side_effect=mock_fn), \
     patch("src.core.causal_model.acall_llm", side_effect=mock_fn), \
     patch("src.core.investigation.acall_llm", side_effect=mock_fn), \
     patch("src.core.rubric_evaluator.acall_llm", side_effect=mock_fn), \
     patch("src.core.science_investigator.acall_llm", side_effect=mock_fn), \
     patch("src.core.decision_projection.acall_llm", side_effect=mock_fn):
    result = await run_causal_pipeline(state)
```

---

## 9. ASYNCIO POLICY

This project enforces a strict asyncio usage policy. Every asyncio call must be annotated with a comment immediately before the call on its own line. Unannotated asyncio calls are treated as policy violations and will fail CI.

### Approved patterns

| Pattern | Annotation | When to use |
|---|---|---|
| `asyncio.to_thread(blocking_fn, *args)` | `# asyncio-APPROVED-1: to_thread wraps blocking <description>` | Any blocking I/O or CPU-bound call that does not have a native async API (file reads/writes, sync SDK clients, numpy/pandas, urllib) |
| `asyncio.gather(*coroutines)` | `# asyncio-APPROVED-2: concurrent <HTTP\|LLM\|file I/O> — <description>` | Concurrent HTTP calls or LLM calls within a single node; concurrent file I/O. Must be non-reducible to LangGraph `Send()` (i.e., called from inside a helper function, not a top-level node) |
| `asyncio.wait_for(single_call, timeout=N)` | `# asyncio-APPROVED-3: wait_for wraps single external <description> call with timeout` | When the underlying SDK has no native async timeout parameter. Always wraps exactly one external call. |
| Infrastructure only: `asyncio.run`, `asyncio.Task`, `asyncio.Queue`, `asyncio.create_task`, `asyncio.sleep` (throttle), `asyncio.wait_for` (SSE heartbeat), `asyncio.TimeoutError`/`asyncio.CancelledError` handlers | `# asyncio-APPROVED-4: <description>` | `asyncio.run` in CLI entry points; `Task`/`Queue`/`create_task` in `src/api.py` SSE infrastructure only; `asyncio.sleep` in OpenAlex rate throttle; `wait_for` on queue.get() for SSE heartbeat |

### Banned patterns

| Pattern | Reason | Replacement |
|---|---|---|
| `asyncio.Semaphore` | Hides rate-limiting from LangGraph scheduler; causes deadlock risk on resume | Set `max_concurrency=N` on `compile()` |
| `asyncio.Lock` | Same as Semaphore — bypasses scheduler | Redesign as separate nodes or use LangGraph state |
| `asyncio.Event` | Same as Semaphore | Not needed in LangGraph's node-edge model |
| `asyncio.create_task` outside `src/api.py` | Spawns untracked tasks that bypass checkpointing | Use `Send()` for parallel node execution |
| `asyncio.Queue` outside `src/api.py` | Same as create_task | Use LangGraph state fields with `operator.add` reducers |
| Swallowing `CancelledError` | Prevents task cancellation from propagating | Always re-raise: `except asyncio.CancelledError: ...; raise` |

### Annotation placement rule

The annotation comment goes on the line **immediately before** the asyncio call, not inline with it:

```python
# CORRECT
# asyncio-APPROVED-1: to_thread wraps blocking JSON file read
result = await asyncio.to_thread(_read_json, path)

# WRONG — inline comment is not detected by the static scanner
result = await asyncio.to_thread(_read_json, path)  # asyncio-APPROVED-1: ...

# WRONG — blank line between comment and call
# asyncio-APPROVED-1: to_thread wraps blocking JSON file read

result = await asyncio.to_thread(_read_json, path)
```

### Static analysis

`tests/test_asyncio_policy.py` enforces this policy with 8 AST/regex tests:

1. `test_no_unapproved_asyncio_calls` — no Semaphore/Lock/Event/BoundedSemaphore anywhere
2. `test_no_asyncio_semaphore` — belt-and-suspenders Semaphore check
3. `test_no_asyncio_create_task_outside_api` — create_task restricted to api.py
4. `test_no_asyncio_queue_outside_api` — Queue restricted to api.py
5. `test_cancelled_error_is_reraised_in_api` — all CancelledError handlers re-raise
6. `test_deep_web_primary_timeout_pattern` — deep_web_graph uses wait_for + annotation
7. `test_all_llm_nodes_have_state_guards` — all HTTP worker nodes have skip-if-already-computed guards
8. `test_mock_inputs_trigger_correct_guards` — fixture JSON files exist with the guard-triggering keys

---

## 10. TRACING

### Auto-instrumentation coverage

Every LLM call in this codebase goes through `acall_llm` in `src/core/llm_utils.py`, which delegates to LangChain's `ChatAnthropic` or `ChatOpenAI`. When `LangchainInstrumentor` is active, every `.ainvoke` call is automatically captured as an OTEL span — no per-call instrumentation code is required.

`LangchainInstrumentor` is initialised once at startup from `src/core/telemetry.py`:

```python
from opentelemetry.instrumentation.langchain import LangchainInstrumentor
LangchainInstrumentor().instrument()
```

This covers all calls made via `acall_llm`, including structured-output calls (`llm.with_structured_output(schema).ainvoke(...)`).

### Why `config: RunnableConfig` matters for span nesting

`acall_llm` accepts an optional `config: RunnableConfig` parameter and forwards it to the underlying LangChain call:

```python
response = await llm.ainvoke(messages, config=config)
```

The `config` object carries the active trace context (run ID, parent run ID) injected by LangGraph when the node is executed. Passing it through is what causes LangChain to attach the LLM span as a child of the current node span in LangSmith and Langfuse. **Without `config`, every LLM call appears as a root-level span with no parent.**

Rule: every node that calls `acall_llm` must pass its own `config` argument through:

```python
async def my_node(state: WorkflowState, config: RunnableConfig) -> dict:
    result = await acall_llm(prompt, system_msg, model=model, config=config)
```

### Calls that are NOT auto-instrumented

Two categories of raw SDK calls bypass `acall_llm` and are not captured by `LangchainInstrumentor`:

**1. Embedding calls in search backends**

`src/backends/local.py` and `src/backends/qdrant.py` call `OpenAI().embeddings.create()` directly for query-time embedding. These are retrieval operations, not agent LLM calls, and have negligible cost per call. They can be wrapped with a manual span if embedding latency becomes a tracing concern:

```python
with tracer.start_as_current_span("embed_query") as span:
    span.set_attribute("model", self._embed_model)
    resp = self._openai_client.embeddings.create(input=[query], model=self._embed_model)
```

### Summary

| Call site | Goes through LangChain | Auto-instrumented | Action needed |
|---|---|---|---|
| All `acall_llm` callers (every node, every agent) | Yes | Yes — via `LangchainInstrumentor` | Pass `config` through |
| `deep_web.py` — `ChatOpenAI(use_responses_api=True)` | Yes | Yes — via `LangchainInstrumentor` | Pass `config` through (already done) |
| `local.py`, `qdrant.py` — `embeddings.create()` | No | No | Add manual span if needed |
