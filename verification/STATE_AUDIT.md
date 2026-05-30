# State Audit — Analyze Pipeline

Source: `src/graph/state.py`, `src/graph/subgraphs/analyze.py`, `src/graph/nodes/analyze.py`.
Reference: `ARCHITECTURE.md §1 / §3 / §4`, `CODEBASE_AUDIT.md §3`.
Scope: analyze pipeline fields only (WorkflowState analyze-relevant slice, full AnalyzeState, full CausalState, all Send() sub-states for the analyze pipeline).

Flags appended to `verification/FINDINGS.md`.

---

## Key to columns

| Column | Meaning |
|---|---|
| **Type** | Python annotation from `state.py` |
| **Reducer** | `operator.add` / `merge_scope_outputs` / `_take_update` / `add_messages` / `overwrite` (no annotation = last-write wins) |
| **Old data-flow value** | What old-repo entity this field carries; "—" = no old-repo equivalent |
| **Written by** | Nodes in the analyze subgraph that assign this field |
| **Read by** | Nodes in the analyze subgraph that read this field |

`WS` = field lives in `WorkflowState`.  `AS` = `AnalyzeState`.  `CS` = `CausalState`.

---

## 1. WorkflowState — Analyze-pipeline slice

Only fields meaningfully touched by the analyze pipeline are listed here.
Fields that belong exclusively to other stages (research, finalize, etc.) are omitted.

### 1a. Run-identity / config fields

| Field | Type | Reducer | Old data-flow value | Written by | Read by | Flags |
|---|---|---|---|---|---|---|
| `program` | `str` | overwrite | Collection/program name, e.g. `"VDEV"` | `load_collection` node (initial state) | `analyze` bridge | — |
| `run_name` | `str` | overwrite | Auto-generated via `generate_run_name()` in old `_cmd_analyze` | initial state | `analyze` bridge (derives `threads_dir`) | — |
| `collection_name` | `str` | overwrite | `args.collection` from CLI | initial state | `analyze` bridge → `AnalyzeState` | — |
| `base_dir` | `str` | overwrite | `args.base_dir` (default `~/qpr-collections`) | initial state | `analyze` bridge | — |
| `threads_dir` | `Optional[str]` | overwrite | `output_dir / "threads"` (old `cache_dir` kwarg) | `analyze` bridge (derives if absent) | `analyze` bridge only | — |
| `focus` | `Optional[str]` | overwrite | `args.focus` | initial state | `analyze` bridge → `AnalyzeState` | — |
| `focus_bows` | `Optional[list[str]]` | overwrite | `args.focus_bows` | initial state | `analyze` bridge → `AnalyzeState` | — |
| `aux_collections` | `Optional[list[str]]` | overwrite | `args.aux_collections` | initial state | `analyze` bridge → `AnalyzeState` | — |
| `research_model` | `str` | overwrite | `args.research_model` (default `ANALYSIS_MODEL`) | initial state | `analyze` bridge → `AnalyzeState` | ⚑ F-002 |
| `synthesis_model` | `str` | overwrite | `args.model` / `synthesis_model` kwarg (default `SYNTHESIS_MODEL`) | initial state | `analyze` bridge → `AnalyzeState` | — |

### 1b. Collection-input fields (written by `load_collection` node; read-only thereafter)

| Field | Type | Reducer | Old data-flow value | Written by | Read by | Flags |
|---|---|---|---|---|---|---|
| `ingested_dir` | `Optional[str]` | overwrite | Derived: `{base_dir}/{program}-ingested/` | `load_collection` node | `analyze` bridge → `AnalyzeState` | — |
| `doc_list` | `Optional[list[dict]]` | overwrite | `doc_list.json` content | `load_collection` node | `analyze` bridge → `AnalyzeState` | — |
| `investment_scoring` | `Optional[dict]` | overwrite | `investment_scoring.json` as `{inv_id: InvestmentDetail}` | `load_collection` node | `analyze` bridge → `AnalyzeState` | — |
| `bow_investment_map` | `Optional[dict]` | overwrite | `bow_investment_map.json` as `{bow_id: [inv_id,...]}` | `load_collection` node | `analyze` bridge → `AnalyzeState` | — |
| `investment_bow_rows` | `Optional[list]` | overwrite | `investment_bow_rows.json` raw rows | `load_collection` node | not forwarded to `AnalyzeState` | ⚑ F-016 |
| `investment_intelligence` | `Optional[dict]` | overwrite | `investment_intelligence.json` as `{inv_id: InvestmentIntelligence}` | `load_collection` node | `analyze` bridge → `AnalyzeState` | — |
| `chunks_json_path` | `Optional[str]` | overwrite | `{ingested_dir}/embedding_index/chunks.json` | `load_collection` node | `analyze` bridge → `AnalyzeState` | — |
| `pages_dir` | `Optional[str]` | overwrite | `{ingested_dir}/pages/` (with `etl/` fallback in old code) | `load_collection` node | `analyze` bridge → `AnalyzeState` | ⚑ F-017 |

### 1c. Stage-2 outputs — written back from `analyze` bridge

| Field | Type | Reducer | Old data-flow value | Written by | Read by (post-analyze) | Flags |
|---|---|---|---|---|---|---|
| `final_report_md_path` | `Optional[str]` | overwrite | `threads/final_report.md` path | `analyze` bridge (from `assemble_report`) | `deliver` node | — |
| `final_report_md` | `Optional[str]` | overwrite | Markdown content of `threads/final_report.md` | `analyze` bridge | `finalize` node | — |
| `analyst_report` | `Optional[dict]` | overwrite | `threads/analyst_report.json` as serialised `AnalystReport` | `analyze` bridge | downstream stages | — |
| `scope_outputs` | `Optional[list[dict]]` | overwrite | `threads/scope_outputs.json` as `list[ScopeOutput]` | `analyze` bridge | downstream stages | — |
| `excerpts_csv_path` | `Optional[str]` | overwrite | Path to `{output_dir}/excerpts.csv` | **never written** by analyze subgraph or bridge | `deliver` node | ⚑ F-005 |
| `numerical_provenance` | `Optional[list[dict]]` | overwrite | `threads/numerical_provenance.json` | `analyze` bridge (from `verify_report`) | downstream | — |
| `verification_sources` | `Optional[list[dict]]` | overwrite | `threads/verification_sources.json` (from old `numerical_verifier`) | **never written** | downstream | ⚑ F-004 |
| `allocation_verification` | `Optional[list[dict]]` | overwrite | `threads/allocation_verification.json` (from old `allocation_verifier`) | `analyze` bridge (from `verify_report`) | downstream | — |
| `numerical_verification` | `Optional[list[dict]]` | overwrite | `threads/numerical_verification.json` (from old `numerical_verifier`) | `analyze` bridge (from `verify_report`) | downstream | — |
| `bibliography` | `Optional[list[dict]]` | overwrite | Deduplicated cited sources list built during assembly | `analyze` bridge (from `assemble_report`) | downstream | — |
| `run_meta` | `Optional[dict]` | overwrite | Old: timing + findings counts in `run_meta.json`; **new: quality metrics** (documents_available, grade, etc.) | `analyze` bridge (from `quality_assessment`) | `deliver` node | ⚑ F-011 |
| `coverage_pct` | `Optional[float]` | overwrite | `coverage_pct` from quality assessment | `analyze` bridge | downstream | — |
| `grade` | `Optional[str]` | overwrite | `"A"/"B"/"C"/"D"` grade from quality assessment | `analyze` bridge | downstream | — |
| `confidence_map` | `Optional[dict]` | overwrite | `{scope_id: "high"/"medium"/"low"}` | `analyze` bridge | downstream | — |
| `program_context` | `Optional[dict]` | overwrite | `ProgramContext` (7 fields from orientation) | `analyze` bridge | downstream | — |
| `scopes` | `Optional[list[dict]]` | overwrite | `list[Scope]` from `_compute_thread_scopes` | `analyze` bridge | downstream | — |
| `scope_timelines` | `Optional[dict]` | overwrite | `{scope_id: ScopeTimeline}` | `analyze` bridge | downstream | — |
| `cross_cutting_analysis` | `Optional[dict]` | overwrite | `CrossCuttingFindings` from `_phase4_crosscut` | `analyze` bridge | downstream | — |
| `allocation_verification_path` | `Optional[str]` | overwrite | Path to `allocation_verification.json` on disk | **never written** by analyze pipeline | downstream | ⚑ F-006 |
| `numerical_verification_path` | `Optional[str]` | overwrite | Path to `numerical_verification.json` on disk | **never written** by analyze pipeline | downstream | ⚑ F-006 |

### 1d. Fan-out accumulator fields at WorkflowState level

**These use `_take_update` reducer (last-writer-wins) — not `operator.add`.**
The `analyze` bridge clears all four to `[]` in its return dict after analysis, since accumulated data is embedded in `scope_outputs`.

| Field | Type | Reducer | Old data-flow value | Written by (WS level) | Notes |
|---|---|---|---|---|---|
| `evidence_packs` | `Annotated[list[dict], _take_update]` | **_take_update** | Per-investment `InvestmentEvidencePack` list, accumulated during Phase 3.1 | `analyze` bridge (clears to `[]`) | ⚑ F-007: Old code accumulated and retained; new WorkflowState discards after analyze |
| `link_assessments` | `Annotated[list[dict], _take_update]` | **_take_update** | Per-link `LinkAssessment` list, accumulated during Phase 3.4 | `analyze` bridge (clears to `[]`) | ⚑ F-007 |
| `science_results` | `Annotated[list[dict], _take_update]` | **_take_update** | Per-assumption `ScienceInvestigationResult` list, accumulated during Phase 3.5d | `analyze` bridge (clears to `[]`) | ⚑ F-007 |
| `scope_decisions` | `Annotated[list[dict], _take_update]` | **_take_update** | Per-scope `Decision` lists, accumulated during Phase 3.8 | `analyze` bridge (clears to `[]`) | ⚑ F-007 |
| `all_excerpts` | `Annotated[list[dict], operator.add]` | operator.add | Annotated evidence excerpts (bibliography source); old: `excerpts.csv` rows | `analyze` bridge (writes from subgraph result) | — |

### 1e. Trace fields at WorkflowState level

All use `operator.add`. Written by the `analyze` bridge (bulk-forwards from AnalyzeState result). Never read within the analyze subgraph itself — terminal outputs only.

| Field | Old data-flow value | Notes |
|---|---|---|
| `asta_traces` | Semantic Scholar search trace dicts | ⚑ F-010 |
| `slr_traces` | SLR search trace dicts | ⚑ F-010 |
| `lbd_traces` | LBD search trace dicts | ⚑ F-010 |
| `deep_web_traces` | Deep web search trace dicts | ⚑ F-010 |
| `edison_traces` | Edison search trace dicts | ⚑ F-010 |
| `web_search_traces` | General web search trace dicts | ⚑ F-010 |
| `compute_traces` | Code interpreter trace dicts | ⚑ F-010 |
| `collection_search_traces` | Collection embedding search trace dicts | ⚑ F-010 |
| `investigation_traces` | Investigation tool call trace dicts | read by `quality_assessment` via bridge |

### 1f. Error / status

| Field | Type | Reducer | Notes |
|---|---|---|---|
| `errors` | `Annotated[list[str], operator.add]` | operator.add | Accumulated from all nodes and bridge |
| `current_stage` | `Optional[str]` | overwrite | Not written by analyze subgraph; written by top-level workflow nodes only |

---

## 2. AnalyzeState — All fields

The analyze subgraph's internal TypedDict. Not a subclass of `WorkflowState`.

### 2a. Collection-input fields (read-only within subgraph)

| Field | Type | Reducer | Old data-flow value | Written by (within AS) | Read by |
|---|---|---|---|---|---|
| `program` | `str` | overwrite | Collection/program name | `load_catalog` (standalone only) | `orientation` |
| `collection_name` | `str` | overwrite | `args.collection` | `load_catalog` (standalone only) | backend routing via `config["configurable"]` |
| `base_dir` | `str` | overwrite | `args.base_dir` | `load_catalog` (standalone only) | `load_catalog` |
| `ingested_dir` | `str` | overwrite | `{base_dir}/{program}-ingested/` | `load_catalog` (standalone only) | `load_catalog`; forwarded to `InvestmentRubricState`, `LinkInvestigationState`, `ScienceAssumptionState` |
| `doc_list` | `list[dict]` | overwrite | `doc_list.json` | `load_catalog` (standalone only) | `orientation`, `build_timelines`, `quality_assessment`, `verify_report` |
| `investment_scoring` | `dict` | overwrite | `investment_scoring.json` | `load_catalog` (standalone only) | `orientation`, `build_timelines`, `dispatch_investment_reports` (passes to worker), `cross_cutting_analysis`, `verify_report` |
| `bow_investment_map` | `dict` | overwrite | `bow_investment_map.json` | `load_catalog` (standalone only) | `orientation`, `compute_scopes` |
| `investment_intelligence` | `dict` | overwrite | `investment_intelligence.json` | `load_catalog` (standalone only) | `orientation`, `build_timelines` |
| `chunks_json_path` | `str` | overwrite | `{ingested_dir}/embedding_index/chunks.json` | `load_catalog` (standalone only) | backend (via config) |
| `pages_dir` | `str` | overwrite | `{ingested_dir}/pages/` | `load_catalog` (standalone only) | `build_timelines`, forwarded to workers |
| `focus` | `Optional[str]` | overwrite | `args.focus` free-text | never (pass-through) | `orientation`, `compute_scopes` |
| `focus_bows` | `Optional[list[str]]` | overwrite | `args.focus_bows` | never (pass-through) | `compute_scopes` |
| `aux_collections` | `Optional[list[str]]` | overwrite | `args.aux_collections` | never (pass-through) | backend routing |

### 2b. Run-context fields

| Field | Type | Reducer | Old data-flow value | Written by | Read by |
|---|---|---|---|---|---|
| `threads_dir` | `Optional[str]` | overwrite | `output_dir / "threads"` (old `cache_dir`) | never (pass-through from bridge) | `assemble_report` (write path for `final_report.md`) |
| `research_model` | `str` | overwrite | `args.research_model` (old default `"gpt-5.5"`) | never (pass-through) | `run_causal_pipeline` (causal_input), `dispatch_investment_narratives` (actually reads `synthesis_model`), forwarded to causal workers |
| `synthesis_model` | `str` | overwrite | `args.model` / `synthesis_model` kwarg (old default `"claude-opus-4-7"`) | never (pass-through) | `orientation` ⚑F-002, `dispatch_investment_narratives`, `dispatch_scope_syntheses`, `build_investment_report_worker`, `synthesize_scope_section_worker`, `cross_cutting_analysis`, `assemble_report`, `verify_report` |

### 2c. Phase outputs (progressive)

| Field | Type | Reducer | Old data-flow value | Written by | Read by | Flags |
|---|---|---|---|---|---|---|
| `program_context` | `Optional[dict]` | overwrite | `ProgramContext` dict (7 fields) from `_phase1_orient` | `orientation` | **not read by any subsequent analyze node** | ⚑ F-001, F-009 |
| `scopes` | `Optional[list[dict]]` | overwrite | `list[Scope]` from `_compute_thread_scopes` | `compute_scopes` | `build_timelines`, `dispatch_investment_narratives`, `quality_assessment`, `run_causal_pipeline` | — |
| `scope_timelines` | `Optional[dict]` | overwrite | `{scope_id: ScopeTimeline}` from `build_scope_timeline` | `build_timelines`, `collect_timeline_narratives` | `dispatch_investment_narratives`, `dispatch_scope_syntheses`, `collect_timeline_narratives`, `run_causal_pipeline` | — |
| `cross_cutting_analysis` | `Optional[dict]` | overwrite | `CrossCuttingFindings` (patterns, contradictions, emergent_decisions, essay, portfolio_metrics) from `_phase4_crosscut` | `cross_cutting_analysis` | `assemble_report` | — |
| `scope_outputs` | `Annotated[list[dict], merge_scope_outputs]` | **merge_scope_outputs** | `list[ScopeOutput]` from `run_causal_pipeline`; augmented by Phase 3.5/3.6 workers | `run_causal_pipeline`, `build_investment_report_worker`, `synthesize_scope_section_worker` | `dispatch_investment_reports`, `dispatch_scope_sections`, `cross_cutting_analysis`, `quality_assessment`, `assemble_report`, `verify_report` | ⚑ F-012 |
| `analyst_report` | `Optional[dict]` | overwrite | `AnalystReport` dataclass; old: `threads/analyst_report.json` | `assemble_report` | **not read within analyze subgraph** | ⚑ F-008 |
| `final_report_md` | `Optional[str]` | overwrite | `threads/final_report.md` content | `assemble_report` | `verify_report` | — |
| `final_report_md_path` | `Optional[str]` | overwrite | `threads/final_report.md` path | `assemble_report` | **not read within analyze subgraph** | ⚑ F-008 |
| `bibliography` | `Optional[list[dict]]` | overwrite | Deduplicated cited-sources list (new); old: `excerpts.csv` partially | `assemble_report` | **not read within analyze subgraph** | ⚑ F-008 |
| `excerpts_csv_path` | `Optional[str]` | overwrite | Path to `{output_dir}/excerpts.csv` (old: written by `run_research_analyst`) | **never written** within analyze subgraph | — | ⚑ F-005 |
| `numerical_provenance` | `Optional[list[dict]]` | overwrite | `threads/numerical_provenance.json` (per-investment `InvestmentFacts`) | `verify_report` | **not read within analyze subgraph** | ⚑ F-008 |
| `verification_sources` | `Optional[list[dict]]` | overwrite | `threads/verification_sources.json` (old `numerical_verifier` output) | **never written** | — | ⚑ F-004 |
| `allocation_verification` | `Optional[list[dict]]` | overwrite | `threads/allocation_verification.json` records | `verify_report` | **not read within analyze subgraph** | ⚑ F-008 |
| `allocation_verification_path` | `Optional[str]` | overwrite | Path to `allocation_verification.json` on disk | **never written** | — | ⚑ F-006 |
| `numerical_verification` | `Optional[list[dict]]` | overwrite | `threads/numerical_verification.json` records | `verify_report` | **not read within analyze subgraph** | ⚑ F-008 |
| `numerical_verification_path` | `Optional[str]` | overwrite | Path to `numerical_verification.json` on disk | **never written** | — | ⚑ F-006 |
| `run_meta` | `Optional[dict]` | overwrite | Old: `run_meta.json` with timings + coverage counts written by `_cmd_analyze`; **new: quality_meta dict (documents_available, grade, etc.) from quality_assessment** | `quality_assessment` | **not read within analyze subgraph** | ⚑ F-011 |
| `coverage_pct` | `Optional[float]` | overwrite | Phase 6a quality metric | `quality_assessment` | `assemble_report` | — |
| `grade` | `Optional[str]` | overwrite | `"A"/"B"/"C"/"D"` from Phase 6a | `quality_assessment` | `assemble_report` | — |
| `confidence_map` | `Optional[dict]` | overwrite | `{scope_id: confidence_level}` from Phase 6a | `quality_assessment` | `assemble_report` | — |

### 2d. Fan-out reducer fields (AnalyzeState level — `operator.add`)

| Field | Type | Reducer | Old data-flow value | Written by | Read by | Flags |
|---|---|---|---|---|---|---|
| `evidence_packs` | `Annotated[list[dict], operator.add]` | operator.add | Per-investment `InvestmentEvidencePack`s accumulated via ThreadPoolExecutor in old Phase 3.1 | causal worker nodes (via `run_causal_pipeline` projection) | **not directly read within analyze subgraph** (data in scope_outputs) | — |
| `link_assessments` | `Annotated[list[dict], operator.add]` | operator.add | Per-link `LinkAssessment`s from Phase 3.4 | causal worker nodes | **not directly read within analyze subgraph** (data in scope_outputs) | — |
| `science_results` | `Annotated[list[dict], operator.add]` | operator.add | Per-assumption `ScienceInvestigationResult`s from Phase 3.5d | causal worker nodes | **not directly read within analyze subgraph** (data in scope_outputs) | — |
| `scope_decisions` | `Annotated[list[dict], operator.add]` | operator.add | Per-scope `Decision` lists from Phase 3.8 | causal worker nodes | **not directly read within analyze subgraph** (data in scope_outputs) | — |
| `investment_narrative_results` | `Annotated[list[dict], operator.add]` | operator.add | **No old-repo equivalent** — per-investment intermediate for two-level narrative fan-out | `generate_investment_narrative` | `collect_investment_narratives` (count), `dispatch_scope_syntheses` (groups by scope_id), `dispatch_investment_narratives` (resume check) | ⚑ F-013 |
| `timeline_narrative_results` | `Annotated[list[dict], operator.add]` | operator.add | Old: `{ingested_dir}/timeline_narratives.json` SHA256-keyed disk cache. **New: in-state accumulator replacing the disk cache.** | `generate_scope_synthesis` | `dispatch_scope_syntheses` (resume check), `collect_timeline_narratives` (merge) | — |
| `all_excerpts` | `Annotated[list[dict], operator.add]` | operator.add | Old: `excerpts.csv` bibliography rows; old generated as a file, new accumulated in state | causal workers → forwarded by `run_causal_pipeline` | `quality_assessment`, `assemble_report`, `verify_report` | — |

### 2e. Trace fields (AnalyzeState level — all `operator.add`)

All trace fields use `operator.add`. Within the analyze subgraph, only `investigation_traces` is read (by `quality_assessment`). All others are accumulated by causal/science workers and propagated to WorkflowState by the bridge — they are terminal outputs from the analyze subgraph's perspective.

| Field | Old data-flow value | Read within AS? |
|---|---|---|
| `asta_traces` | ASTA search traces from `science_investigator`; no old-repo trace format | No |
| `slr_traces` | SLR search traces; no old-repo equivalent | No |
| `lbd_traces` | LBD search traces; no old-repo equivalent | No |
| `deep_web_traces` | Deep web search traces | No |
| `edison_traces` | Edison search traces | No |
| `web_search_traces` | Web search traces from `search_web` tool | No |
| `compute_traces` | Code interpreter traces | No |
| `collection_search_traces` | Embedding search traces from `search_collection` tool | No |
| `investigation_traces` | Investigation tool call traces | **Yes** — `quality_assessment` reads `documents_read` from each trace |

### 2f. Error accumulator

| Field | Type | Reducer |
|---|---|---|
| `errors` | `Annotated[list[str], operator.add]` | operator.add |

---

## 3. CausalState — All fields

Internal state of the causal pipeline subgraph. Not exposed to `AnalyzeState` or `WorkflowState` directly; projected in/out by `run_causal_pipeline` node.

### 3a. Input fields (projected from AnalyzeState)

| Field | Type | Reducer | Old data-flow value | Notes |
|---|---|---|---|---|
| `scopes` | `list[dict]` | overwrite | `list[Scope]` from Phase 2 | Non-optional in CS vs Optional in AS |
| `scope_timelines` | `dict` | overwrite | `{scope_id: ScopeTimeline.to_dict()}` from Phase 2.5/2.7 | Non-optional in CS |
| `research_model` | `str` | overwrite | `args.research_model` | Forwarded to all causal workers |
| `synthesis_model` | `str` | overwrite | `args.model` | Forwarded to synthesis nodes |

### 3b. Missing field — `cache_dir`

> **⚑ F-014**: `ARCHITECTURE.md §4` lists `cache_dir: str` as a `CausalState` field, but the actual `CausalState` definition in `state.py` does **not** include it. Workers receive `ingested_dir` and `collection_name` via their Send() sub-state slices (`InvestmentRubricState`, `LinkInvestigationState`, `ScienceAssumptionState`) for fallback tool construction. Per-scope disk caching (old `_save_checkpoint`/`_load_checkpoint`) is handled by LangGraph checkpointing, but the `cache_dir` parameter to the old `run_causal_pipeline` has no direct equivalent in `CausalState`.

### 3c. Fan-out reducer fields

| Field | Type | Reducer | Old data-flow value | Written by | Read by |
|---|---|---|---|---|---|
| `evidence_packs` | `Annotated[list[dict], operator.add]` | operator.add | Accumulated by ThreadPoolExecutor workers in Phase 3.1 | `evaluate_investment_rubric` worker | `collect_evidence_packs` reducer; data embedded into `scope_outputs` |
| `link_assessments` | `Annotated[list[dict], operator.add]` | operator.add | Accumulated by ThreadPoolExecutor workers in Phase 3.4 | `investigate_link` worker | `collect_link_assessments` reducer; data embedded into `scope_outputs` |
| `science_results` | `Annotated[list[dict], operator.add]` | operator.add | Accumulated by ThreadPoolExecutor workers in Phase 3.5d | `investigate_science_assumption` worker | `collect_science_results` reducer; data embedded into `scope_outputs` |
| `scope_decisions` | `Annotated[list[dict], operator.add]` | operator.add | Accumulated by ThreadPoolExecutor(4) in Phase 3.8 (called from run_research_analyst, not causal_pipeline in old code) | `project_scope_decisions` worker | `collect_decisions` reducer; data embedded into `scope_outputs` |

### 3d. Progressive output

| Field | Type | Reducer | Old data-flow value | Written by | Read by |
|---|---|---|---|---|---|
| `scope_outputs` | `Annotated[list[dict], merge_scope_outputs]` | **merge_scope_outputs** | `list[ScopeOutput]` built progressively across causal sub-stages | `collect_evidence_packs`, `collect_bow_enrichment`, `collect_link_assessments`, `synthesize_findings`, `collect_science_results`, `necessity_check`, `collect_decisions`, `enrich_bow_context_worker` | all causal reducer nodes that use scope context; projected back to `AnalyzeState` | ⚑ F-012 |
| `all_excerpts` | `Annotated[list[dict], operator.add]` | operator.add | Old: `excerpts.csv` rows (accumulated per link); new: top-10 per link by credibility tier | `investigate_link` worker, `investigate_science_assumption` worker | projected to `AnalyzeState` |

### 3e. Trace fields (all `operator.add`)

Same set as `AnalyzeState` traces. Not read within the causal subgraph; projected to `AnalyzeState` by `run_causal_pipeline`. Same flag applies: ⚑ F-010.

### 3f. Error accumulator

| Field | Type | Reducer |
|---|---|---|
| `errors` | `Annotated[list[str], operator.add]` | operator.add |

---

## 4. Send() Sub-state Slices — Analyze Pipeline

Each slice is a TypedDict sent to a worker node. Fields shown as those directly relevant to collection-API search or data flow.

### `InvestmentRubricState` (→ `evaluate_investment_rubric`, Stage 3.1)

| Field | Type | Old data-flow value | Notes |
|---|---|---|---|
| `inv_id` | `str` | investment ID | — |
| `scope_id` | `str` | scope ID | — |
| `scope_label` | `Optional[str]` | human-readable label for scope-fit classification | **New field** — no old-repo equivalent |
| `timeline` | `dict` | serialised `InvestmentTimeline` | Old: `InvestmentTimeline` object passed by value |
| `result` | `Optional[dict]` | `InvestmentEvidencePack` filled by worker | — |
| `research_model` | `str` | model for LLM calls | — |
| `ingested_dir` | `str` | fallback when `search_backend` absent from config | **New field** — needed for `_get_tools` / `_EmbeddingIndexAdapter` bridge |
| `collection_name` | `str` | fallback for backend routing | **New field** — needed for `_get_tools` bridge |

### `LinkInvestigationState` (→ `investigate_link`, Stage 3.4)

| Field | Type | Old data-flow value | Notes |
|---|---|---|---|
| `link_id` | `str` | causal link ID | — |
| `inv_id` | `str` | investment ID | — |
| `bow_id` | `str` | BOW ID | — |
| `scope_id` | `str` | scope ID | — |
| `scope_label` | `str` | human-readable label for excerpt CSV | **New field** |
| `claim` | `dict` | serialised `InvestigationClaim` | Old: `InvestigationClaim` object |
| `model` | `str` | model for LLM calls | Old: `research_model` kwarg to `run_investigation` |
| `result` | `Optional[dict]` | `LinkAssessment` filled by worker | — |
| `ingested_dir` | `str` | fallback for `_get_tools` bridge | **New field** |
| `collection_name` | `str` | fallback for backend routing | **New field** |

### `ScienceAssumptionState` (→ `investigate_science_assumption`, Stage 3.5d)

| Field | Type | Old data-flow value | Notes |
|---|---|---|---|
| `assumption_id` | `str` | assumption ID | — |
| `inv_id` | `str` | investment ID | — |
| `bow_id` | `str` | BOW ID | — |
| `scope_id` | `str` | scope ID | — |
| `question` | `str` | science question text | Old: `ScienceQuestion` object |
| `result` | `Optional[dict]` | `ScienceInvestigationResult` | — |
| `research_model` | `str` | model for LLM calls | — |
| `ingested_dir` | `str` | fallback for `_get_tools` bridge | **New field** |
| `collection_name` | `str` | fallback for backend routing | **New field** |

### `InvestmentNarrativeState` (→ `generate_investment_narrative`, Phase 2.6 Level 1)

**Entirely new — no old-repo equivalent. Part of the two-level narrative fan-out.**

| Field | Type | Notes |
|---|---|---|
| `scope_id` | `str` | — |
| `scope_label` | `str` | — |
| `inv_id` | `str` | — |
| `inv_data` | `dict` | `InvestmentTimeline.to_dict()` or minimal financial dict |
| `model` | `str` | — |

### `ScopeSynthesisState` (→ `generate_scope_synthesis`, Phase 2.6 Level 2)

**Entirely new — no old-repo equivalent.**

| Field | Type | Notes |
|---|---|---|
| `scope_id` | `str` | — |
| `scope_label` | `str` | — |
| `investment_narratives` | `list` | Accumulated from Level 1 |
| `scope_timeline_dict` | `dict` | — |
| `model` | `str` | — |

### `BowEnrichmentWorkerState` (→ `enrich_bow_context_worker`, Stage 3.1.5)

| Field | Type | Old data-flow value | Notes |
|---|---|---|---|
| `scope_id` | `str` | scope ID | — |
| `scope` | `dict` | full scope dict | — |
| `model` | `str` | synthesis_model | — |
| `result` | `Optional[dict]` | unused — worker writes directly to `scope_outputs` | — |

### `InvestmentReportWorkerState` (→ `build_investment_report_worker`, Phase 3.5)

| Field | Type | Old data-flow value | Notes |
|---|---|---|---|
| `scope_id` | `str` | scope ID | — |
| `scope` | `dict` | full scope dict including link_assessments | — |
| `investment_scoring` | `dict` | `{inv_id: InvestmentDetail}` — for team scores | — |
| `model` | `str` | synthesis_model | — |
| `result` | `Optional[dict]` | unused — worker writes directly to `scope_outputs` | — |

### `SectionDraftWorkerState` (→ `synthesize_scope_section_worker`, Phase 3.6)

| Field | Type | Old data-flow value | Notes |
|---|---|---|---|
| `scope_id` | `str` | scope ID | — |
| `scope` | `dict` | full scope dict including investment_report, link_assessments, investment_facts | — |
| `model` | `str` | synthesis_model | — |
| `result` | `Optional[dict]` | unused — worker writes directly to `scope_outputs` | — |

### `ScopeDecisionState` (→ `project_scope_decisions`, Stage 3.8)

| Field | Type | Old data-flow value | Notes |
|---|---|---|---|
| `scope_id` | `str` | scope ID | — |
| `scope_output` | `dict` | serialised `ScopeOutput` | Old: `ScopeOutput` object passed to `run_phase38` workers |
| `decisions` | `Optional[list[dict]]` | list of `Decision` objects | — |
| `synthesis_model` | `str` | model for projection LLM call | Old: `run_phase38(model=research_model)` — note old used `research_model`; new uses `synthesis_model` | ⚑ F-018 |

---

## 5. Collection-API Search Calls Carried Through State

The table below maps every collection-API search call in the analyze pipeline to the state field that carries its inputs and the state field that carries its results.

| Phase | Node / Core function | Search call | Input from state | Result to state |
|---|---|---|---|---|
| Phase 1 | `orientation` | `backend.search("program theory of change...", top_k=30)` | `search_backend` from `config["configurable"]` (not from AS fields) | LLM prompt only; no state field for raw results | ⚑ F-003 |
| Phase 2 | `compute_scopes` | `backend.count_by_bow_id()` | `search_backend` from `config["configurable"]` | `bow_chunk_counts` local variable → drives `scopes` |
| Phase 3.1 | `evaluate_investment_rubric` → `rubric_evaluator.build_evidence_pack` | S1–S4 via `_EmbeddingIndexAdapter` (`search_with_filter`, `hybrid_search`) | `inv_id`, `ingested_dir`, `collection_name` from `InvestmentRubricState` | chunks accumulated in `InvestmentEvidencePack` → embedded into `CausalState.scope_outputs` via `collect_evidence_packs` |
| Phase 3.4 | `investigate_link` → `investigation_loop.run_investigation` | `search_investment`, `search_portfolio` tools | `inv_id`, `bow_id`, `ingested_dir`, `collection_name` from `LinkInvestigationState` + `search_backend` from config | `LinkAssessment` → `CausalState.link_assessments` → `CausalState.scope_outputs` |
| Phase 3.4 | `investigate_link` | All 10 investigation tool calls | `claim` dict from `LinkInvestigationState` | `investigation_traces` accumulated in `CausalState` |
| Phase 3.5d | `investigate_science_assumption` → `science_investigator` | `search_asta`, `search_bow`, `search_science`, `search_policy`, `search_all` tools | `ingested_dir`, `collection_name` from `ScienceAssumptionState` | `ScienceInvestigationResult` → `CausalState.science_results` → `CausalState.scope_outputs`; `asta_traces` |
| Phase 6b | `assemble_report` | none directly | `all_excerpts` from `AnalyzeState` (pre-accumulated) | `final_report_md` |
| Phase 6b | `verify_report` | none | `all_excerpts`, `investment_scoring`, `final_report_md` from `AnalyzeState` | `allocation_verification`, `numerical_verification`, `numerical_provenance` |

---

## 6. Summary: Fields by Flag Category

### 6a. Old-repo values with no state field in the new design

| Old value | Expected state field | Gap |
|---|---|---|
| `orientation_model` kwarg (old `run_research_analyst`) | None — `orientation` uses `synthesis_model` instead | ⚑ F-002 |
| `context: ProgramContext` passed to `run_causal_pipeline` | `CausalState` has no `program_context` field | ⚑ F-009 |
| `cache_dir` passed to `run_causal_pipeline` | `CausalState` has no `cache_dir` field | ⚑ F-014 |
| `pages_dir` fallback to `etl/` layout | `load_collection` node may not implement `etl/` fallback | ⚑ F-017 |
| `document_catalog.json` (tried before `doc_list.json` in old loader) | `load_catalog` reads only `doc_list.json` | ⚑ F-016 |
| `verification_sources.json` from old `numerical_verifier` | `verification_sources` field declared but never written | ⚑ F-004 |
| `excerpts_csv_path` (old: `{output_dir}/excerpts.csv`) | Field declared in AS and WS; never written by any analyze node | ⚑ F-005 |
| `allocation_verification_path` / `numerical_verification_path` (disk paths) | Fields declared; never written by any analyze node | ⚑ F-006 |
| `investment_bow_rows` (loaded by `load_collection` node → WS) | Not forwarded into `AnalyzeState` | ⚑ F-016 |

### 6b. Old-code accumulator fields that use an overwrite-style reducer in the new design

| Field | Old behavior | New WorkflowState reducer | Impact |
|---|---|---|---|
| `WS.evidence_packs` | Remained populated in `run_research_analyst` return value | `_take_update` — bridge clears to `[]` | Any post-analyze code reading `WS.evidence_packs` will see `[]` | ⚑ F-007 |
| `WS.link_assessments` | Same | `_take_update` — bridge clears to `[]` | Same | ⚑ F-007 |
| `WS.science_results` | Same | `_take_update` — bridge clears to `[]` | Same | ⚑ F-007 |
| `WS.scope_decisions` | Same | `_take_update` — bridge clears to `[]` | Same | ⚑ F-007 |

### 6c. Fields written but never read within the analyze subgraph (terminal outputs)

| Field | Written by | Why not read internally |
|---|---|---|
| `program_context` | `orientation` | `run_causal_pipeline` does not include it in `causal_input`; no other AS node reads it | ⚑ F-009 |
| `analyst_report` | `assemble_report` | Output field; read by downstream stages outside analyze | ⚑ F-008 |
| `final_report_md_path` | `assemble_report` | Output field | ⚑ F-008 |
| `bibliography` | `assemble_report` | Output field | ⚑ F-008 |
| `numerical_provenance` | `verify_report` | Output field | ⚑ F-008 |
| `allocation_verification` | `verify_report` | Output field | ⚑ F-008 |
| `numerical_verification` | `verify_report` | Output field | ⚑ F-008 |
| `run_meta` | `quality_assessment` | Output field | ⚑ F-011 |
| `asta_traces` … `collection_search_traces` (8 trace fields) | causal workers (via `run_causal_pipeline`) | Terminal trace output fields | ⚑ F-010 |

### 6d. Fields declared in AnalyzeState but never written within the analyze subgraph

| Field | Notes |
|---|---|
| `verification_sources` | Declared in AS; no node writes it; old `numerical_verifier` produced this | ⚑ F-004 |
| `excerpts_csv_path` | Declared in AS; `assemble_report` does not write it | ⚑ F-005 |
| `allocation_verification_path` | Declared in AS; no node writes it | ⚑ F-006 |
| `numerical_verification_path` | Declared in AS; no node writes it | ⚑ F-006 |
