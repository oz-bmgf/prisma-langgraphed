# Analyze Subgraph — Node Contracts and Verification

Source of truth: old-repo `src/qpr/research_analyst_agent.py`, `src/qpr/investment_timeline.py`, `src/qpr/report_assembler.py`, `src/qpr/report_writer.py`, `CODEBASE_AUDIT.md`, `ARCHITECTURE.md §3`.
Implementation: `src/graph/subgraphs/analyze.py` (1 506 lines), `src/graph/nodes/analyze.py`.

Verdict key: **verified** = all contract points met; **partial** = core logic present but deviations exist; **missing** = node absent or entirely wrong contract.

---

## Node: `load_catalog` — Phase 0

### A. Expected Contract

> New node — no old-repo equivalent. Old code always called `load_collection()` before `run_research_analyst()`.
> ARCHITECTURE.md §3: "Load JSON artifacts from ingested_dir when running standalone (no-op if already populated)."

| | |
|---|---|
| **Reads** | `investment_scoring` (guard), `ingested_dir` |
| **Writes** | `doc_list`, `investment_scoring`, `bow_investment_map`, `investment_intelligence`, `chunks_json_path`, `pages_dir` |
| **Guard** | Return `{}` if `investment_scoring` already populated |
| **Side effects** | Disk reads via `asyncio.to_thread` |
| **Search calls** | None |
| **LLM calls** | None |

Expected files read: `doc_list.json`, `investment_scoring.json`, `bow_investment_map.json`, `investment_intelligence.json`. Paths: `{ingested_dir}/{file}`. `chunks_json_path = {ingested_dir}/embedding_index/chunks.json`. `pages_dir = {ingested_dir}/pages/`.

ARCHITECTURE.md does not specify `investment_bow_rows.json`; `load_collection` node (which runs in the top-level graph) loads it into `WorkflowState` but `load_catalog` is the standalone fallback.

### B. Implementation Check

Lines 64–92 of `analyze.py`:
- Guard: `if state.get("investment_scoring"): return {}` ✓
- Reads `doc_list.json`, `investment_scoring.json`, `bow_investment_map.json`, `investment_intelligence.json` via `asyncio.to_thread` ✓
- Sets `chunks_json_path = str(base / "embedding_index" / "chunks.json")` ✓
- Sets `pages_dir = str(base / "pages")` — **no `etl/` fallback** ✗
- Does **not** load `investment_bow_rows.json` (consistent with ARCHITECTURE.md, but see F-016)
- Error handling: returns `{"errors": [f"load_catalog:{exc}"]}` ✓

### C. Verdict: **partial**

| Finding | Severity |
|---|---|
| F-016 (pages_dir `etl/` fallback absent) | MEDIUM |
| F-016 (`investment_bow_rows.json` not loaded in standalone path) | LOW |

---

## Node: `orientation` — Phase 1

### A. Expected Contract

Old-repo: `_phase1_orient(tools, call_llm, *, model: str = ANALYSIS_MODEL)`

| | |
|---|---|
| **Reads** | `tools.list_bows()`, `tools.get_scoring(inv_id)` per investment, strategy-corpus doc summaries (date-desc sorted, ≤ 1.6 M chars, ≤ 400 chars/doc) |
| **Writes** | `program_context` (7-field dict) |
| **LLM call** | Model: `orientation_model` (default `ANALYSIS_MODEL = "gpt-5.5"`). Params: `json_mode=True, thinking=False, max_tokens=8000`. |
| **Side effects** | None |
| **Search calls** | None — uses structured metadata from `tools.list_bows()` / `tools.get_scoring()` only |

**Exact system prompt** (`_ORIENTATION_SYSTEM`, lines 59–87 of old `research_analyst_agent.py`):
```
{SAFETY_PREAMBLE}
You are a senior research analyst preparing to assess an investment program.
You will be given summaries of the top portfolio-level documents and investment
scoring data.

Build a mental model of the program. Return JSON:
{
  "theory_of_change": "The program's causal chain from investment to impact (3-5 sentences)",
  "major_bets": [
    {"bet": "Description of a major strategic bet", "bows": ["B001"], "amount_approx": "$50M"}
  ],
  "stated_priorities": ["Priority 1 from docs", "Priority 2"],
  "key_timelines": [
    {"milestone": "What", "target_date": "When", "status": "on_track|at_risk|missed"}
  ],
  "portfolio_health_signals": [
    "Signal from scoring data or docs (e.g., '5 of 19 investments are red')"
  ],
  "initial_concerns": [
    "Things that already seem problematic from the high-level view"
  ]
}

Read carefully and note contradictions between documents. The initial concerns
will drive the research threads in the next phase.
```

**Expected output schema:**
- `theory_of_change`: `str`
- `major_bets`: `list[{bet: str, bows: list[str], amount_approx: str}]` — **list of structured dicts**
- `stated_priorities`: `list[str]`
- `key_timelines`: `list[{milestone: str, target_date: str, status: str}]` — **list of structured dicts**
- `portfolio_health_signals`: `list[str]`
- `initial_concerns`: `list[str]`

Old code also read strategy-corpus documents: date-desc sorted, ≤ 400 chars/doc, ≤ 1.6 M chars total. These were included as raw text in the user prompt (not via embedding search).

### B. Implementation Check

Lines 100–239 of `analyze.py`:

1. **Model**: reads `state.get("synthesis_model")` — uses `synthesis_model` (`claude-opus-4-7`), **not** `orientation_model` / `research_model` ✗ (F-002)

2. **Data sources**: reads `investment_scoring`, `bow_investment_map`, `investment_intelligence`, `doc_list` from state — replaces old `tools.list_bows()` / `tools.get_scoring()` calls with state field reads ✓ (equivalent data)

3. **Document summaries**: does **not** read strategy-corpus documents with date-desc sort. Instead, optionally calls `backend.search("program theory of change goals outcomes investments", top_k=30)` and injects up to 20 results as "Representative Document Excerpts" ✗ (F-003, F-020)

4. **Prompt structure**: new inline prompt (lines 195–212) differs from old `_ORIENTATION_SYSTEM`. No `SAFETY_PREAMBLE`. No `thinking=False, max_tokens=8000` params visible — uses `acall_llm` defaults ✗ (F-021)

5. **Output schema**:
   - `major_bets`: new prompt asks for `"list of strings"` — old expected list of `{bet, bows, amount_approx}` dicts ✗ (F-019)
   - `key_timelines`: new asks for `"list of strings"` — old expected list of `{milestone, target_date, status}` dicts ✗ (F-019)
   - `bow_summaries`: new field, **not** in old schema — new prompt adds `"dict mapping bow_id to a 1-sentence summary"` ✗ (new key, old consumers may not expect it)
   - `initial_concerns`, `stated_priorities`, `portfolio_health_signals`, `theory_of_change` ✓

6. **Error handling**: returns `{"errors": [...], "program_context": {}}` on LLM failure ✓

### C. Verdict: **partial**

| Finding | Severity |
|---|---|
| F-002 (uses `synthesis_model` not `orientation_model`) | HIGH |
| F-003 (embedding search replaces document-summary ingestion) | HIGH |
| F-019 (`major_bets` and `key_timelines` schema changed to flat strings) | HIGH |
| F-020 (strategy corpus date-desc sorted summary read absent) | HIGH |
| F-021 (system prompt differs; `SAFETY_PREAMBLE` absent; `max_tokens=8000` not set) | MEDIUM |

---

## Node: `compute_scopes` — Phase 2

### A. Expected Contract

Old-repo: `_compute_thread_scopes(tools: CollectionTools) -> list[dict]`

| | |
|---|---|
| **Reads** | `tools._embedding_index.count_by_bow_id()`, `tools._embedding_index.distinct_inv_ids()`, `tools.ctx.bow_investment_map`, `focus_bows` |
| **Writes** | `scopes: list[dict]` |
| **LLM calls** | None |
| **Search calls** | `tools._embedding_index.count_by_bow_id()` — chunk counts per BOW. `tools._embedding_index.distinct_inv_ids()` — filters phantom investments (no indexed docs). |
| **Side effects** | `threads/scope_set.json` written to `cache_dir` (resume artifact). `threads/checkpoint_initial.json` written to `cache_dir`. |

**Old grouping key**: `"{topic} > {sub_topic}"` — requires `topic` and `sub_topic` keys in BOW metadata.

**Old split thresholds**:
- Skip: `chunk_count < 200`
- Split: `(total_chunks > 12 000 and len(group_bows) > 1)` OR `total_invs > 8` OR `(is_catch_all and len(group_bows) > 1)`

**Old phantom-investment filter**: `distinct_inv_ids()` call — drops investments with zero indexed chunks from scope `inv_ids`.

**Old scope dict keys**: `{scope_id, label, topic, sub_topic, bow_ids, inv_ids, chunk_count}` — includes `topic` and `sub_topic`.

### B. Implementation Check

Lines 247–387 of `analyze.py`:

1. **Chunk counts**: calls `backend.count_by_bow_id()` from `config["configurable"]["search_backend"]` — equivalent to old `tools._embedding_index.count_by_bow_id()` ✓

2. **Phantom filter**: **absent** — new code does not call `distinct_inv_ids()` to filter investments with zero indexed docs. It filters by presence in `investment_scoring` only ✗ (F-023)

3. **Grouping key**: uses `bow_data.get("sub_topic") or bow_data.get("focus_area") or bow_id` — reads `sub_topic` key if present but does not require `topic`. Groups on `sub_topic` only, not `"{topic} > {sub_topic}"` ✗ (F-022)

4. **Skip threshold**: `MIN_CHUNKS = int(os.environ.get("NQPR_MIN_CHUNKS_PER_BOW", "5"))` — default **5**, old was **200** ✗ (F-024)

5. **Split thresholds**: `SPLIT_CHUNKS = 12_000`, `SPLIT_INVS = 8` ✓ (match old). But old `is_catch_all` split condition is absent ✗ (F-024)

6. **Scope dict keys**: new code produces `{scope_id, inv_id, inv_ids, bow_ids, label, chunk_count}` — adds `inv_id` (primary), missing `topic` and `sub_topic` ✗ (F-022)

7. **Disk artifacts**: `scope_set.json` and `checkpoint_initial.json` **not written** — replaced by LangGraph checkpointing (intentional design change; LOW impact)

### C. Verdict: **partial**

| Finding | Severity |
|---|---|
| F-022 (scope dict missing `topic`/`sub_topic`; grouping key differs) | MEDIUM |
| F-023 (phantom-investment filter via `distinct_inv_ids()` absent) | MEDIUM |
| F-024 (MIN_CHUNKS default 5 vs old 200; `is_catch_all` split logic absent) | HIGH |

---

## Node: `build_timelines` — Phase 2.5

### A. Expected Contract

Old-repo: loop over scopes calling `build_scope_timeline(scope, doc_list, scoring_data, pages_dir, investment_intelligence)`, then attempt `load_narratives()` from `{ingested_dir}/timeline_narratives.json`.

| | |
|---|---|
| **Reads** | `scopes`, `doc_list`, `investment_scoring`, `investment_intelligence`, `pages_dir` |
| **Writes** | `scope_timelines: dict[scope_id, ScopeTimeline.to_dict()]` |
| **LLM calls** | None |
| **Search calls** | None |
| **Side effects** | Attempts to pre-populate narratives from `timeline_narratives.json` SHA256-keyed cache (if hash matches, narratives are loaded and scope skips LLM generation in Phase 2.6) |
| **Branching** | If `load_narratives()` returns `True` (cache hit), Phase 2.6 fan-out is smaller |

`build_scope_timeline` signature: `(scope, doc_list, scoring, pages_dir, investment_intelligence) -> ScopeTimeline`. Sorts per-investment timelines by `(-(len(flags) + len(rating_flags)), latest_doc_date)` — most flags first, then most recent.

### B. Implementation Check

Lines 395–441 of `analyze.py`:

1. **Reads**: `scopes`, `doc_list`, `investment_scoring` (as `scoring`), `investment_intelligence`, `pages_dir` ✓
2. **Calls**: `build_scope_timeline(scope, doc_list, scoring, pages_dir, investment_intelligence=intelligence)` per scope via `asyncio.to_thread` ✓
3. **Returns**: `{"scope_timelines": {sid: st.to_dict() ...}}` ✓
4. **Cache load**: **absent** — no `load_narratives()` call. Old code pre-populated narratives here when the SHA256 cache was valid. New code relies entirely on `timeline_narrative_results` set-membership in state for resume. ✗ (F-025)

### C. Verdict: **partial**

| Finding | Severity |
|---|---|
| F-025 (SHA256 `timeline_narratives.json` cache not loaded; resume behavior differs for old-format collections) | MEDIUM |

---

## Router: `dispatch_investment_narratives` — Phase 2.6 Level-1 fan-out (conditional edge, not a node)

### A. Expected Contract

Old-repo: `build_timeline_narratives(scope_timeline, call_llm, *, model)` called once per scope inside `ThreadPoolExecutor(max_workers=min(len(investments), 6))`. Internally, it runs:

- Step 1: one `_build_single_investment_narrative` call **per investment** (parallel, up to 6 workers)
- Step 2: one scope-synthesis LLM call (sequentially after all investment narratives)

Cache guard: skips scopes whose SHA256 hash matches `timeline_narratives.json`.

### B. Implementation Check

Lines 449–509 of `analyze.py` (conditional edge function `dispatch_investment_narratives`):

- Reads `scopes`, `scope_timelines`, `investment_scoring`, `synthesis_model` ✓
- Resume guard: skips `"{scope_id}:{inv_id}"` already in `investment_narrative_results` ✓
- Emits `Send("generate_investment_narrative", InvestmentNarrativeState)` per investment ✓ — matches old Step 1 parallelism
- Falls back to `"collect_investment_narratives"` when all done ✓
- Old guard: SHA256 cache check before generating — **absent** here; resume relies on `investment_narrative_results` membership ✗ (F-025 — same issue)

### C. Verdict: **partial**

---

## Node: `generate_investment_narrative` — Phase 2.6 Level-1 worker

### A. Expected Contract

Old-repo: `_build_single_investment_narrative(inv: InvestmentTimeline, scope_id, scope_label, model)` — one LLM prose narrative per investment (500–1000 words), `json_mode=False`, system prompt `_INVESTMENT_NARRATIVE_SYSTEM` (emphasizes chronological, prose, no JSON). Deliberately omits `score_history`/`score_trend`.

| | |
|---|---|
| **Reads** | `InvestmentNarrativeState`: `scope_id`, `inv_id`, `inv_data`, `model` |
| **Writes** | `investment_narrative_results: [{scope_id, inv_id, narrative, inv_data}]` |
| **LLM call** | model from state, prose narrative, no json_mode |
| **Search calls** | None |

### B. Implementation Check

Lines 517–574 of `analyze.py`:

1. **Rich path** (when `inv_data` has `documents` key): reconstructs `InvestmentTimeline` + `DocumentEvent` dataclasses, calls `_build_single_investment_narrative(inv_obj, scope_id, scope_label, model)` — delegates to same old-repo function ✓
2. **Fallback path** (minimal financial dict): short prompt, 2-3 sentence summary via `acall_llm` ✓ (new behavior for incomplete data)
3. **Return**: `{"investment_narrative_results": [{scope_id, inv_id, narrative, inv_data}]}` ✓

### C. Verdict: **verified** (delegates to old-repo implementation on rich path)

---

## Node: `collect_investment_narratives` — Phase 2.6 sync point

### A. Expected Contract

Trivial reducer: waits for all `generate_investment_narrative` workers to complete before routing to `dispatch_scope_syntheses`.

### B. Implementation Check

Lines 582–586: logs count, returns `{}`. ✓

### C. Verdict: **verified**

---

## Router: `dispatch_scope_syntheses` — Phase 2.6 Level-2 fan-out (conditional edge, not a node)

### A. Expected Contract

Old-repo: after per-investment narratives, one scope-synthesis LLM call per scope (sequential in old code, `_SCOPE_SYNTHESIS_SYSTEM` prompt).

### B. Implementation Check

Lines 594–628: groups `investment_narrative_results` by `scope_id`, emits `Send("generate_scope_synthesis", ScopeSynthesisState)` per scope needing synthesis. Resume guard via `timeline_narrative_results` set. ✓

### C. Verdict: **verified**

---

## Node: `generate_scope_synthesis` — Phase 2.6 Level-2 worker

### A. Expected Contract

Old-repo: `_SCOPE_SYNTHESIS_SYSTEM` prompt, one LLM call per scope synthesising all investment narratives into a 200–400 word scope-level paragraph.

### B. Implementation Check

Lines 636–676 of `analyze.py`:

1. Reads `investment_narratives` (per-investment narratives), `scope_timeline_dict`, `model` ✓
2. Merges per-investment narratives back into `scope_tl_out["investments"]` ✓
3. Calls `acall_llm(...)` with `system_msg=SCOPE_SYNTHESIS_SYSTEM` imported from `src.core.investment_timeline` ✓ — uses same system prompt constant
4. Returns `{"timeline_narrative_results": [scope_tl_out]}` ✓

### C. Verdict: **verified**

---

## Node: `collect_timeline_narratives` — Phase 2.7

### A. Expected Contract

Old-repo: merge narrative results back into `scope_timelines`, then call `save_narratives()` to write `{ingested_dir}/timeline_narratives.json` (SHA256-keyed cache for future runs).

| | |
|---|---|
| **Reads** | `timeline_narrative_results`, `scope_timelines` |
| **Writes** | `scope_timelines` (updated with narratives) |
| **Side effects (old)** | Writes `timeline_narratives.json` SHA256 cache to disk |

### B. Implementation Check

Lines 684–714 of `analyze.py`:

1. Merges `timeline_narrative_results` into `scope_timelines` dict — updates narrative, per-investment narratives, and key_events ✓
2. **No `save_narratives()` call** — no disk write ✗
3. Comment: "LangGraph checkpointing handles resume — no file-based cache read or write." (intentional design choice)
4. Returns `{"scope_timelines": updated}` ✓

### C. Verdict: **partial**

| Finding | Severity |
|---|---|
| F-025 (`timeline_narratives.json` not written; old collections cannot share cache with new pipeline) | LOW (design intent; new pipeline uses state for resume) |

---

## Node: `run_causal_pipeline` — Phase 3

### A. Expected Contract

Old-repo: `run_causal_pipeline(scopes, scope_timelines, tools, call_llm, context, cache_dir, research_model, synthesis_model) -> list[ScopeOutput]`

| | |
|---|---|
| **Reads** | `scopes`, `scope_timelines`, `research_model`, `synthesis_model`, `program_context` (as `context` positional arg in old code) |
| **Writes** | `scope_outputs`, `all_excerpts`, all 9 trace fields, `errors` |
| **Delegates** | `causal_graph.ainvoke(causal_input, config)` |
| **Key projection** | Must forward `program_context` to causal subgraph as `context` |

### B. Implementation Check

Lines 722–777 of `analyze.py`:

1. Builds `causal_input` with `scopes`, `scope_timelines`, `research_model`, `synthesis_model`, all reducer fields ✓
2. **`program_context` NOT included in `causal_input`** — the key forwarding the portfolio context to causal sub-stages is missing ✗ (F-001 / F-009)
3. Projects causal result back to `AnalyzeState`: `scope_outputs`, `all_excerpts`, all traces, `errors` ✓
4. Evidence packs / link_assessments / science_results / scope_decisions **not forwarded** — comment explains they are embedded in scope_outputs ✓ (intentional)
5. Error handling: returns `{"errors": [f"run_causal_pipeline:{exc}"]}` on failure ✓

### C. Verdict: **partial**

| Finding | Severity |
|---|---|
| F-001 (`program_context` / `context` not forwarded to causal subgraph) | CRITICAL |

---

## Node: `build_investment_report_worker` — Phase 3.5

### A. Expected Contract

ARCHITECTURE.md §3: "Blind AI-vs-team verdict; Step 1 = LLM verdict without scores; Step 2 = load team scores; Step 3 = compute `divergence_severity`; returns `{"scope_outputs": [scope]}`."

Old-repo: this phase was implemented as part of the causal pipeline's synthesis pass and `evaluation_comparison.py`. `ScopeOutput.investment_report` was built from `InvestmentReport` dataclass.

| | |
|---|---|
| **Reads** | `InvestmentReportWorkerState`: `scope_id`, `scope` (includes `link_assessments`), `investment_scoring`, `model` |
| **Writes** | `{"scope_outputs": [scope]}` with `scope["investment_report"]` added |
| **LLM call** | `model` (synthesis_model), blind verdict prompt, json_mode implicit |
| **Search calls** | None |

**Expected `investment_report` keys**: `overall_status`, `severity`, `ai_execution_verdict`, `ai_impact_verdict`, `key_risks`, `key_strengths`, `executive_summary`, `team_execution_score`, `team_impact_score`, `divergence_severity`.

### B. Implementation Check

Lines 812–903 of `analyze.py`:

1. Step 1 — blind LLM prompt (lines 840–855): correct structure, prompts for `{overall_status, severity, ai_execution_verdict, ai_impact_verdict, key_risks, key_strengths, executive_summary}` ✓
2. Step 2 — loads team scores from `investment_scoring` dict ✓
3. Step 3 — divergence calculation (lines 869–887): rank-difference approach using `_severity_rank` and `_team_rank` maps ✓ (reasonable, equivalent to old divergence logic)
4. Output: writes `scope["investment_report"]`, `scope["team_execution"]`, `scope["team_impact"]` ✓
5. Falls back gracefully when no `link_assessments` ✓

Minor divergence: old `evaluation_comparison.py` used `VERDICT_LEVELS`, `SEVERITY_RANK` maps from the module; new uses inline rank maps. Logic is equivalent but not identical.

### C. Verdict: **verified**

---

## Node: `collect_investment_reports` — Phase 3.5 reducer

### A. Expected Contract

Trivial join: all `build_investment_report_worker` branches complete; `merge_scope_outputs` accumulates results.

### B. Implementation Check

Line 906–908: returns `{}`. ✓

### C. Verdict: **verified**

---

## Node: `synthesize_scope_section_worker` — Phase 3.6

### A. Expected Contract

Old-repo: `report_writer.write_section_draft(thread, findings, source_index, call_llm, *, model, max_findings=15) -> SectionDraft`

| | |
|---|---|
| **Reads** | `SectionDraftWorkerState`: `scope_id`, `scope` (investment_report, link_assessments, investment_facts), `model` |
| **Writes** | `{"scope_outputs": [scope]}` with `scope["section_draft"]` added |
| **LLM call** | `model` (synthesis_model), essay format with 4-part structure |
| **Word budget** | `WORD_BUDGETS = {"program_critical": 1500, "pathway_altering": 1000, "efficiency_reducing": 500}` (default 300) — keyed by top severity |
| **Source refs** | §-reference canonicalization; retry once if §-refs missing |

Old `_SECTION_WRITER_SYSTEM` required 4-part essay structure (What/Evidence/Why/Counter/Assessment) with inline `§[source_id]` citations. `write_section_draft` would retry once if §-refs were missing from output.

### B. Implementation Check

Lines 951–1018 of `analyze.py`:

1. Computes `ranked_deviations` sorted by `dollars_at_risk × severity_weight` ✓ (ARCHITECTURE.md §3 requirement met)
2. LLM prompt (lines 990–999): asks for 3-4 paragraph narrative covering key findings, execution, financial, risks — different structure from old 4-part essay ✗ (F-036)
3. **No word budget** — old `WORD_BUDGETS` keyed by severity absent ✗ (F-036)
4. **No §-reference canonicalization** — old code required inline `§[source_id]` citations and would retry if missing ✗ (F-036)
5. **No retry** on missing citations ✗ (F-036)
6. Output: `scope["section_draft"] = {scope_id, heading, summary, key_findings, ranked_deviations}` — adds new `ranked_deviations` key not in old `SectionDraft` ✓ (ARCHITECTURE.md specifies this)

### C. Verdict: **partial**

| Finding | Severity |
|---|---|
| F-036 (word budgets absent; no §-ref canonicalization; no retry on missing citations) | MEDIUM |

---

## Node: `collect_scope_sections` — Phase 3.6 reducer

### A. Expected Contract

Trivial join.

### B. Implementation Check

Lines 1021–1023: returns `{}`. ✓

### C. Verdict: **verified**

---

## Node: `cross_cutting_analysis` — Phase 5

### A. Expected Contract

Old-repo: `_phase4_crosscut(threads, context, call_llm, *, model=SYNTHESIS_MODEL, portfolio_metrics=...)`

| | |
|---|---|
| **Reads** | `threads` (all research threads), `context` (ProgramContext), `portfolio_metrics` (pre-computed string, from `code_interpreter` in old code) |
| **Writes** | Returns `(list[Finding], list[dict])` — cross-cutting findings + emergent decisions |
| **LLM call** | `model=synthesis_model`, `_CROSSCUT_SYSTEM` (detailed system prompt), `json_mode=True` |
| **Pre-computation** | `portfolio_metrics` was pre-computed by `numerical_analyst._call_code_interpreter()` using OpenAI `code_interpreter` tool |
| **Side effects (old)** | `cluster_identification.identify_clusters()` called separately → writes `threads/clusters.json` |

**Expected output**: `list[Finding]` with a `.narrative` attribute (essay) + `list[dict]` (emergent decisions). NOT a single dict.

### B. Implementation Check

Lines 1031–1137 of `analyze.py`:

1. **Portfolio metrics**: computed via pure Python (lines 1047–1077): `total_approved`, `total_paid`, `at_risk_count`, `concentration_by_bow` — **no `code_interpreter`** ✗ (F-026). Note: new approach is deterministic and less error-prone.
2. **`cluster_identification.identify_clusters()`**: **absent** — new code does not identify thematic clusters ✗ (F-027)
3. **`context: ProgramContext`**: not available to this node (program_context not in AnalyzeState after run_causal_pipeline) ✗ (F-001)
4. **Output schema**: returns `{"cross_cutting_analysis": {patterns, contradictions, shared_dependencies, emergent_decisions, essay, portfolio_metrics}}` as a dict — old returned `(list[Finding], list[dict])` ✗ (F-028)
5. **LLM call**: `synthesis_model`, inline prompt (not `_CROSSCUT_SYSTEM`), returns JSON dict ✓ (different prompt, similar structure)
6. **`threads/clusters.json`**: **not written** ✗ (F-027)
7. Error handling: partial result on LLM failure ✓

### C. Verdict: **partial**

| Finding | Severity |
|---|---|
| F-027 (`cluster_identification.identify_clusters()` absent; clusters never populated) | HIGH |
| F-028 (output schema changed from `list[Finding]` to `dict`; downstream `assemble_report` and `prepare_research` may expect old shape) | HIGH |
| F-026 (no `code_interpreter`; pure Python portfolio metrics — less flexible but deterministic) | MEDIUM |

---

## Node: `quality_assessment` — Phase 6a

### A. Expected Contract

Old-repo: `_phase5_quality(threads, tools, focus_bows) -> dict`

| | |
|---|---|
| **Reads** | Research threads, `tools` (for doc counts) |
| **Writes** | Returns `{documents_available, documents_read, coverage_pct, confidence_map, low_confidence_threads, all_gaps, grade}` |
| **LLM calls** | None — fully deterministic |
| **Grade thresholds** | A: coverage > 0.5 and low_confidence ≤ 2; B: > 0.3, ≤ 4; C: > 0.15; D: else |

### B. Implementation Check

Lines 1145–1231 of `analyze.py`:

1. **Reads**: `doc_list`, `scope_outputs`, `scopes`, `all_excerpts`, `investigation_traces` ✓
2. **Coverage computation**: unique doc_ids from `all_excerpts` (source field) + `link_assessments.documents_read` + `investigation_traces.documents_read`; denominator = `len(active_inv_ids)` (from scopes) or `len(doc_list)` ✓ (reasonable equivalent)
3. **Grade thresholds**: A: coverage > 0.5 and low_confidence_scopes ≤ 2; B: > 0.3, ≤ 4; C: > 0.15; D: else ✓ (exact match)
4. **Confidence map**: per-scope based on `link_assessments.confidence` distribution ✓
5. **Writes**: `coverage_pct`, `grade`, `confidence_map`, `run_meta` (quality_meta) ✓
6. `run_meta` contains quality metrics, **not timing data** (F-011 already logged)

### C. Verdict: **verified** (F-011 already logged)

---

## Node: `assemble_report` — Phase 6b

### A. Expected Contract

Old-repo: `report_assembler.assemble_report(context, section_drafts, cross_cutting, decisions, bow_hierarchy, bow_nodes, source_index_all, call_llm, *, model, documents_read, documents_available, coverage_pct, thread_stats, focus, focus_bows, investments, document_catalog, scope_outputs, figures_dir, inv_team_scores, embedding_index) -> str`

| | |
|---|---|
| **Reads** | Pre-written `section_drafts` (one per scope, from `write_section_draft`), `context: ProgramContext`, `cross_cutting: list[Finding]`, `investments: dict`, `embedding_index` |
| **Writes** | `final_report_md` (Markdown string) |
| **Writes to disk** | `threads/final_report.md`, `threads/analyst_report.json`, `threads/scope_outputs.json`, `{output_dir}/excerpts.csv`, working notes per section |
| **LLM calls** | Multiple (executive summary, key insights narration with `NarrationToolbox`, `exec_summary/premise_investigator` ASTA cascade) |
| **Retry logic** | `_ASSEMBLY_MAX_RETRIES = 5`, `_ASSEMBLY_RETRY_DELAY = 60` seconds |
| **Side effects** | Writes to `run_dir/threads/`; renders PNG chart figures; outputs CSV bibliography |

### B. Implementation Check

Lines 1239–1280 of `analyze.py` + `src/core/report_assembler.py` (511 lines):

1. **Signature called**: `report_assembler.assemble_report(scope_outputs, cross_cutting_analysis, all_excerpts, confidence_map, coverage_pct, grade, model, config)` — completely different from old signature ✗ (F-029, F-030, F-031)
2. **`section_drafts`**: **not passed** — new `report_assembler` generates sections internally from `scope_outputs` ✗ (F-030)
3. **`context: ProgramContext`**: **not passed** ✗ (F-031)
4. **`cross_cutting`**: passed as `cross_cutting_analysis` (a dict); old expected `list[Finding]` ✗ (F-028)
5. **`investments` / `embedding_index`**: **not passed** ✗ (F-031)
6. **Retry loop**: new `assemble_report` node has **no retry** (`_ASSEMBLY_MAX_RETRIES=5` absent) ✗ (F-029)
7. **`NarrationToolbox`**: not invoked by new `report_assembler` call chain ✗ (F-032)
8. **`exec_summary/premise_investigator`**: absent ✗ (F-033)
9. **`excerpts.csv`**: not written by `assemble_report` or the analyze node ✗ (F-005)
10. **Disk writes in node**: writes `threads/final_report.md` ✓; sets `final_report_md_path` ✓
11. **Returns**: `{"final_report_md", "analyst_report", "final_report_md_path", "bibliography"}` ✓

### C. Verdict: **partial**

| Finding | Severity |
|---|---|
| F-029 (no retry loop; `_ASSEMBLY_MAX_RETRIES=5` absent) | HIGH |
| F-030 (section_drafts not pre-generated; `write_section_draft` logic bypassed) | HIGH |
| F-031 (`context: ProgramContext`, `investments`, `embedding_index` not passed) | HIGH |
| F-032 (`NarrationToolbox` with 6 agentic tools not invoked) | HIGH |
| F-033 (`exec_summary/premise_investigator` ASTA cascade absent) | HIGH |
| F-005 (`excerpts_csv_path` not written) | MEDIUM |
| F-028 (`cross_cutting` type changed: list[Finding] → dict; old assemble_report expected Finding objects) | HIGH |

---

## Node: `verify_report` — Phase 6b verifier

### A. Expected Contract

Old-repo: called `allocation_verifier.verify_and_rewrite(report_path, scoring_path, output_path, scorecard_path, auto_rewrite=True)` + `numerical_verifier.verify_report(report_path, collection_name, *, base_dir, call_llm, output_path)`.

| | |
|---|---|
| **Reads** | `final_report_md` (via disk path), `investment_scoring`, `all_excerpts`, `scope_outputs` |
| **Writes** | `allocation_verification`, `numerical_verification`, `numerical_provenance` |
| **LLM calls** | Optional: `synthesis_model` for unmatched figure verification |
| **Search calls** | Old `numerical_verifier` loaded `chunks.json` and cross-referenced figures against chunk text |
| **Report rewrite** | Old `allocation_verifier.apply_allocation_edits()` rewrote `final_report_md` in-place for approved mismatches |
| **Disk writes (old)** | `threads/allocation_verification.json`, `threads/allocation_issues.md`, `threads/numerical_provenance.json`, `threads/verification_sources.json`, `threads/numerical_verification.json` |

### B. Implementation Check

Lines 1288–1419 of `analyze.py`:

1. **Dollar extraction**: `$(\d+(?:\.\d+)?)\s*([MBK]?)` regex from `final_report_md` ✓
2. **Cross-reference**: ±10% tolerance against `investment_scoring` approved/paid amounts ✓
3. **LLM pass for unmatched**: prompt asks for `[{figure, verdict, explanation}]` JSON array ✓
4. **`numerical_provenance`**: built from `scope_outputs[*].investment_facts` ✓
5. **Report rewrite**: **absent** — old `apply_allocation_edits` rewrote `final_report_md` in-place ✗ (F-034)
6. **`verification_sources`**: **not written** ✗ (F-004)
7. **`chunks.json` lookup**: **absent** — old `numerical_verifier` loaded chunks.json for figure cross-reference. New code uses `all_excerpts` from state instead ✗ / different approach
8. **Disk writes**: **none** — all results go to state ✗ (F-006; path fields remain `None`)
9. **`allocation_issues.md`**: **not written** ✗ (F-035)

### C. Verdict: **partial**

| Finding | Severity |
|---|---|
| F-034 (report NOT rewritten; mismatch figures not corrected) | HIGH |
| F-004 (`verification_sources` not written) | MEDIUM |
| F-035 (`allocation_issues.md` not written) | LOW |

---

## Summary Table

| Node | Phase | Verdict | Key Findings |
|---|---|---|---|
| `load_catalog` | 0 | **partial** | F-016 |
| `orientation` | 1 | **partial** | F-002, F-003, F-019, F-020, F-021 |
| `compute_scopes` | 2 | **partial** | ~~F-022~~ fixed 2026-05-29, F-023, F-024 |
| `build_timelines` | 2.5 | **partial** | F-025 |
| `dispatch_investment_narratives` | 2.6 router | **partial** | F-025 |
| `generate_investment_narrative` | 2.6 L1 worker | **verified** | — |
| `collect_investment_narratives` | 2.6 sync | **verified** | — |
| `dispatch_scope_syntheses` | 2.6 router | **verified** | — |
| `generate_scope_synthesis` | 2.6 L2 worker | **verified** | — |
| `collect_timeline_narratives` | 2.7 | **partial** | F-025 |
| `run_causal_pipeline` | 3 | **partial** | ~~F-001~~ fixed 2026-05-29 (program_context added to CausalState + causal_input), F-014 |
| `build_investment_report_worker` | 3.5 | **verified** | — |
| `collect_investment_reports` | 3.5 join | **verified** | — |
| `synthesize_scope_section_worker` | 3.6 | **verified** | ~~F-036~~ fixed 2026-05-29, ~~F-030~~ fixed 2026-05-29 |
| `collect_scope_sections` | 3.6 join | **verified** | — |
| `cross_cutting_analysis` | 5 | **partial** | F-026, F-027, F-028 |
| `quality_assessment` | 6a | **verified** | (F-011 pre-logged) |
| `assemble_report` | 6b | **partial** | F-028 (N/A within scope), F-029, ~~F-030~~ fixed in synthesize_scope_section_worker 2026-05-29, F-031, F-032, F-033, F-005 |
| `verify_report` | 6b verifier | **partial** | F-004, ~~F-034~~ fixed 2026-05-29, F-035 |
