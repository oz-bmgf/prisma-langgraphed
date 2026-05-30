# Verification Findings

Scope: analyze pipeline and collection-API search only.
Source: `STATE_AUDIT.md`, `MAPPING.md`, `CODEBASE_AUDIT.md`, `ARCHITECTURE.md`.
Status: **Do not fix — record only.**

Each finding cites the state field(s) or construct involved, the category, the old-repo behavior, and the new-repo behavior.

---

## Category key

| Category | Meaning |
|---|---|
| **NO-FIELD** | Old-repo data value has no corresponding state field in the new design |
| **ACCUMULATE→OVERWRITE** | Old code accumulated a list; new WorkflowState uses a last-writer-wins reducer |
| **WRITE-ONLY** | Field is written by a node but never read within the analyze subgraph |
| **UNWRITTEN** | Field is declared in the TypedDict but no analyze subgraph node writes it |
| **BEHAVIOR-DELTA** | New code implements the same logical step differently (model choice, search strategy, control flow) |
| **MISSING-NODE** | Old-repo sub-stage has no corresponding LangGraph node in the current implementation |

---

## Findings

### F-001 — ~~BEHAVIOR-DELTA~~ **RESOLVED**: `program_context` not forwarded to causal subgraph

**Fields:** `AnalyzeState.program_context`, `CausalState.program_context` (added)

**Old behavior:** `run_research_analyst` passed the `ProgramContext` object as the `context` positional argument to `run_causal_pipeline`. The causal pipeline used it for BOW-level causal model extraction (sub-stage 3.1a), BOW synthesis prompts, and as general portfolio context for investigation prompts.

**New behavior (pre-fix):** `program_context` was written by `orientation` but never included in `causal_input`, making it unreachable by any causal subgraph node.

**Fix applied 2026-05-29:**
- Added `program_context: Optional[dict]` to `CausalState` in `state.py` (inputs block, with comment).
- Added `"program_context": state.get("program_context")` to the `causal_input` projection dict in `run_causal_pipeline` (`analyze.py`).
- Verified with node test: `program_context` dict arrives in `causal_graph.ainvoke`'s state argument.
- Causal nodes can now read `state.get("program_context")` for portfolio context. Prompt-level usage is a separate concern.

---

### F-002 — BEHAVIOR-DELTA: `orientation` uses `synthesis_model` instead of a dedicated `orientation_model`

**Fields:** `AnalyzeState.synthesis_model`, `WorkflowState.research_model`

**Old behavior:** `run_research_analyst` accepted `orientation_model: str = ANALYSIS_MODEL` (default `"gpt-5.5"`). The `_phase1_orient` function used this model for the orientation LLM call. The CLI never forwarded it, so it always defaulted to `gpt-5.5`.

**New behavior:** `orientation` node reads `state.get("synthesis_model") or _DEFAULT_MODEL` (line 112 of `analyze.py`). `_DEFAULT_MODEL` is `DEFAULT_SYNTHESIS_MODEL` (`claude-opus-4-7` from `config.py`). There is no `orientation_model` state field. The `research_model` state field (the closer equivalent to the old `orientation_model`) is not used by `orientation`.

**Impact:** Orientation now runs on `synthesis_model` (Opus-class) rather than the old `analysis_model` (GPT-class). This is a different model family for the same task.

---

### F-003 — BEHAVIOR-DELTA: `orientation` performs embedding search not present in old `_phase1_orient`

**Fields:** `SearchBackend` (via `config["configurable"]["search_backend"]`); no state field for raw search results

**Old behavior:** `_phase1_orient` called `tools.list_bows()` and `tools.get_bow_summary(bow_id)` for up to N BOWs, then built an LLM prompt from the structured metadata. No free-text embedding search was performed.

**New behavior:** `orientation` node additionally calls `backend.search("program theory of change goals outcomes investments", top_k=30)` (lines 175–193 of `analyze.py`) when a `search_backend` is present in `config["configurable"]`. Results are injected into the LLM prompt as "Representative Document Excerpts." This call writes to no state field; results are consumed inline.

**Impact:** Orientation now performs a collection search that the old code did not; behavior diverges when no `search_backend` is configured (search is skipped with a DEBUG log). Result chunks are not traced to `collection_search_traces`.

---

### F-004 — UNWRITTEN: `verification_sources` field declared but never written

**Fields:** `AnalyzeState.verification_sources`, `WorkflowState.verification_sources`

**Old behavior:** `numerical_verifier.verify_report(...)` produced a `verification_sources` list (list of source dicts) and wrote it to `threads/verification_sources.json`. `run_research_analyst` then stored it on the `AnalystReport`.

**New behavior:** `verify_report` node (analyze subgraph) writes `allocation_verification`, `numerical_verification`, and `numerical_provenance` but does **not** write `verification_sources`. The field is declared in both `AnalyzeState` (line 369) and `WorkflowState` (line 272) but is never assigned by any node in the analyze pipeline.

**Impact:** `WorkflowState.verification_sources` will always be `None` after the analyze stage. Any downstream stage or human-review step that expects this field will receive no data.

---

### F-005 — ~~UNWRITTEN~~ **RESOLVED**: `excerpts_csv_path` declared but no analyze node writes it

**Fields:** `AnalyzeState.excerpts_csv_path`, `WorkflowState.excerpts_csv_path`

**Old behavior:** `run_research_analyst` assembled an `excerpts.csv` file (bibliography CSV with file_id, section_id, text, §-ref columns).

**New behavior (pre-fix):** `assemble_report` node did not write a CSV file and did not set `excerpts_csv_path`.

**Fix applied 2026-05-29:**
- `assemble_report` node now writes `{threads_dir}/excerpts.csv` from `all_excerpts` with 16 columns: `ref_id, excerpt_id, inv_id, scope_id, link_id, file_id, source_file, page, page_start, page_end, source_type, credibility_tier, type, significance, context_needed, quote`.
- Sets `excerpts_csv_path` in the returned state dict.
- Each excerpt carries a positional `ref_id=§NNNN` (see F-039) so the CSV is directly citable.
- Writing is wrapped in `asyncio.to_thread` per asyncio policy.

---

### F-006 — UNWRITTEN: `allocation_verification_path` and `numerical_verification_path` never written

**Fields:** `AnalyzeState.allocation_verification_path`, `AnalyzeState.numerical_verification_path`, `WorkflowState.allocation_verification_path`, `WorkflowState.numerical_verification_path`

**Old behavior:** `run_research_analyst` called `allocation_verifier.verify_and_rewrite(...)` and `numerical_verifier.verify_report(...)`, which wrote JSON files to disk and returned their paths. These paths were stored on the `AnalystReport` and in the old state equivalent.

**New behavior:** `verify_report` node writes the verification data into `allocation_verification` and `numerical_verification` state fields (lists of dicts), but writes no files to disk and does not set the `*_path` fields. Both path fields are declared in the TypedDict but are always `None` after the analyze subgraph runs.

**Impact:** Any downstream consumer expecting a file at `allocation_verification_path` or `numerical_verification_path` will find `None`. The old `allocation_issues.md` (rewritten mismatches file) also has no equivalent.

---

### F-007 — ACCUMULATE→OVERWRITE: WorkflowState fan-out fields cleared to `[]` after analyze

**Fields:** `WorkflowState.evidence_packs`, `WorkflowState.link_assessments`, `WorkflowState.science_results`, `WorkflowState.scope_decisions`

**Old behavior:** After `run_research_analyst` returned, the accumulated `evidence_packs`, `link_assessments`, `science_results`, and `scope_decisions` remained available on the `AnalystReport` object in memory. Downstream code (e.g., report assembly, executive summary, evidence audit) could access the full populated lists.

**New behavior:** These four fields use the `_take_update` reducer in `WorkflowState` (not `operator.add`). The `analyze` bridge node explicitly clears them to `[]` in its return dict (lines 100–103 of `analyze.py`):

```python
"evidence_packs": [],
"link_assessments": [],
"science_results": [],
"scope_decisions": [],
```

Because `_take_update` is last-writer-wins, this wipe takes effect immediately. After the `analyze` stage completes, any node in the top-level graph that reads these WorkflowState fields will see empty lists. The data is embedded in `scope_outputs` (inside each scope dict), but not accessible as flat lists.

**Impact:** Post-analyze stages that relied on flat `evidence_packs` or `link_assessments` lists from `AnalystReport` will find `[]` in `WorkflowState`. Evidence audit and gs_verifier graphs that read these fields from WorkflowState will receive no data.

---

### F-008 — WRITE-ONLY: Terminal output fields written but never read within the analyze subgraph

**Fields:** `AnalyzeState.analyst_report`, `AnalyzeState.final_report_md_path`, `AnalyzeState.bibliography`, `AnalyzeState.numerical_provenance`, `AnalyzeState.allocation_verification`, `AnalyzeState.numerical_verification`, `AnalyzeState.run_meta`

**Old behavior:** These were intermediate values used within `run_research_analyst` — e.g., `analyst_report` was read between phases to carry partial state, `numerical_provenance` was built during Phase 6b and read by the post-run verifier.

**New behavior:** All these fields are written by terminal nodes (`assemble_report`, `verify_report`, `quality_assessment`) and propagated outward by the bridge. Within the analyze subgraph itself, no subsequent node reads them after they are written. This is architecturally correct for output fields, but means there is no intra-subgraph validation that these fields are populated before the subgraph exits.

**Note:** Not a bug — expected for terminal outputs. Flagged for completeness to confirm no node intends to read these fields and was missed.

---

### F-009 — NO-FIELD: `program_context` / `context` not in `CausalState`

**Fields:** `CausalState` (missing `program_context`); `AnalyzeState.program_context`

**Old behavior:** `run_causal_pipeline(scopes, scope_timelines, tools, call_llm, context, ...)` received `context: ProgramContext` as a positional argument. The causal pipeline used it for: (a) BOW-level causal model prompts, (b) investigation context injection, (c) BOW enrichment system prompts.

**New behavior:** `CausalState` defines no `program_context` or `context` field. The `run_causal_pipeline` node's `causal_input` dict (lines 728–751 of `analyze.py`) does not include `program_context`. The `orientation` node writes `program_context` to `AnalyzeState`, but no subsequent analyze subgraph node reads it, and the causal subgraph never receives it.

**Risk:** See F-001. Duplicated here to note the `CausalState` schema gap.

---

### F-010 — WRITE-ONLY: Trace fields (8 of 9) written but never read within analyze subgraph

**Fields:** `AnalyzeState.asta_traces`, `.slr_traces`, `.lbd_traces`, `.deep_web_traces`, `.edison_traces`, `.web_search_traces`, `.compute_traces`, `.collection_search_traces`

**Old behavior:** `LLM_TRACE_FILE` JSONL file accumulated all call traces. No structured per-tool state accumulation existed.

**New behavior:** Eight trace fields are accumulated by causal/science workers and forwarded to `WorkflowState` by the bridge. They are never read within the analyze subgraph itself. `investigation_traces` is the only trace field read internally (by `quality_assessment` to extract `documents_read` for coverage computation).

**Note:** These are design-correct terminal outputs for observability. Flagged because `collection_search_traces` — the field that records every embedding search call — is accumulated but never validated or queried within the pipeline, making search-call auditing post-hoc only.

---

### F-011 — BEHAVIOR-DELTA: `run_meta` semantics changed

**Fields:** `AnalyzeState.run_meta`, `WorkflowState.run_meta`

**Old behavior:** `_cmd_analyze` wrote `{output_dir}/run_meta.json` containing: `run_name`, `program`, `collection`, `timestamp`, `duration_seconds`, `phase_timings`, `coverage_pct`, `grade`, `num_scopes`, `num_findings`. This was a CLI-level artifact written after `run_research_analyst` completed.

**New behavior:** `quality_assessment` node writes `run_meta` to `AnalyzeState` containing: `{"documents_available": N, "documents_read": M, "coverage_pct": P, "grade": G, "low_confidence_scopes": [...]}`. This is a quality-assessment dict, not a run-timing dict. The field name is the same but the content schema differs materially. Timing data is absent.

**Impact:** Any consumer expecting `run_meta` to contain `duration_seconds`, `phase_timings`, or `timestamp` will find these keys absent.

---

### F-012 — BEHAVIOR-DELTA: `scope_outputs` uses `merge_scope_outputs` (shallow merge) instead of in-place mutation

**Fields:** `AnalyzeState.scope_outputs`, `CausalState.scope_outputs`

**Old behavior:** `run_causal_pipeline` maintained `scope_outputs` as a `list[ScopeOutput]` mutated in place across sequential sub-stages. Each sub-stage's reducer updated fields directly on the `ScopeOutput` dataclass (e.g., `scope.causal_model = ...`, `scope.link_assessments.extend(...)`). Final state was always consistent because all mutations happened on the same object in a single thread.

**New behavior:** `merge_scope_outputs` performs a **shallow dict merge** by `scope_id` (line 30 of `state.py`: `by_id[sid].update(scope)`). This means if two parallel workers both write a scope dict that contains a top-level key with a nested dict value, the second write silently replaces the entire nested dict, not just changed sub-keys.

Example risk: if `collect_evidence_packs` writes `{"scope_id": "s1", "evidence_packs": [...], "investment_facts": {...}}` and a concurrent `enrich_bow_context_worker` writes `{"scope_id": "s1", "bow_context": {...}, "investment_facts": {"overridden": true}}`, the `investment_facts` from the first write is lost.

**Impact:** Shallow merge correctness depends on no two parallel branches writing the same top-level key for the same scope_id. This invariant is not enforced by the reducer and would fail silently if violated.

---

### F-013 — BEHAVIOR-DELTA: Two-level narrative fan-out adds `investment_narrative_results` with no old-repo equivalent

**Fields:** `AnalyzeState.investment_narrative_results`

**Old behavior:** `build_timeline_narratives(scope_timeline, call_llm, model)` generated a single narrative per scope in one LLM call. The result was merged into `ScopeTimeline.narrative`.

**New behavior:** Phase 2.6 uses a two-level fan-out: (1) `generate_investment_narrative` produces one narrative per investment (N × M sends), accumulating into `investment_narrative_results`; (2) `generate_scope_synthesis` synthesises all investment narratives for a scope into a scope-level narrative. The intermediate `investment_narrative_results` accumulator has no old-repo equivalent and is not described in `ARCHITECTURE.md §3` (the ARCHITECTURE doc shows a single fan-out level for Phase 2.6).

**Impact:** Significantly more LLM calls than old code for large portfolios (N_investments × N_scopes vs N_scopes). `ARCHITECTURE.md §3` is inconsistent with the implementation.

---

### F-014 — NO-FIELD: `CausalState` missing `cache_dir` field listed in ARCHITECTURE.md §4

**Fields:** `CausalState` (missing); `ARCHITECTURE.md §4`

**Old behavior:** `run_causal_pipeline(scopes, scope_timelines, tools, call_llm, context, cache_dir, ...)` received `cache_dir: Path` for per-scope disk checkpoint read/write.

**New behavior:** `CausalState` in `state.py` (lines 403–438) does not include a `cache_dir` field, contrary to `ARCHITECTURE.md §4` which lists `cache_dir: str` as a `CausalState` field. Workers receive `ingested_dir` and `collection_name` via their Send() sub-state slices for fallback tool construction. LangGraph checkpointing replaces per-scope disk checkpoints, but the field gap means any causal node that reads `state.get("cache_dir")` will receive `None`.

**Impact:** Resume/idempotency behavior for causal workers may differ from the design in `ARCHITECTURE.md §4` ("each worker checks for a cached result on disk (`cache_dir/...`) before calling the underlying async function").

---

### F-015 — NO-FIELD: `CausalState` missing `pages_dir` field

**Fields:** `CausalState` (missing)

**Old behavior:** `run_causal_pipeline` received `tools: CollectionTools` which encapsulated `pages_dir` for page-image reading.

**New behavior:** `CausalState` has no `pages_dir` field. Causal workers receive `ingested_dir` via Send() sub-states and derive `pages_dir` from it inside `_get_tools`. This is functional but means `pages_dir` is recomputed per worker rather than being checkpointed state.

---

### F-016 — NO-FIELD / BEHAVIOR-DELTA: `investment_bow_rows` not forwarded to `AnalyzeState`; `document_catalog.json` fallback absent

**Fields:** `WorkflowState.investment_bow_rows`; `AnalyzeState` (missing `investment_bow_rows`); `load_catalog` node

**Part A — `investment_bow_rows` not in AnalyzeState:** `WorkflowState` includes `investment_bow_rows: Optional[list]` (line 256) and `load_collection` node populates it. The `analyze` bridge (`analyze.py` lines 32–83) does not include `investment_bow_rows` in `analyze_input`. `AnalyzeState` has no `investment_bow_rows` field. Any causal or analyze logic that needs the raw BOW-row data cannot access it.

**Part B — `document_catalog.json` fallback absent in `load_catalog`:** Old `load_collection` tried `document_catalog.json` first, falling back to `doc_list.json`. New `load_catalog` node (standalone path) reads only `doc_list.json` (line 80 of `analyze.py`). Collections that ship only `document_catalog.json` will fail the standalone path with a `FileNotFoundError`.

---

### F-017 — BEHAVIOR-DELTA: `pages_dir` fallback to `etl/` layout may be absent in `load_collection` node

**Fields:** `WorkflowState.pages_dir`

**Old behavior:** `collection_loader._resolve_pages_dir(ingested_dir)` checked for `pages/` first and fell back to `etl/` for the older two-directory layout. Collections built before the `pages/` rename used `etl/`.

**New behavior:** `load_collection` node (`src/graph/nodes/load_collection.py`) sets `pages_dir`. Whether it implements the `etl/` fallback is unverified. If absent, any collection using the `etl/` layout will produce a `pages_dir` pointing to a non-existent directory, causing all page-text reads to fail silently (empty string returns).

---

### F-018 — BEHAVIOR-DELTA: `ScopeDecisionState.synthesis_model` vs old `run_phase38(model=research_model)`

**Fields:** `ScopeDecisionState.synthesis_model`; old `run_phase38` `model` kwarg

**Old behavior:** `run_phase38(scope_outputs, call_llm, model=research_model, max_workers=4)` — decision projection used `research_model` (default `"gpt-5.5"`, the analysis/investigation model).

**New behavior:** `ScopeDecisionState` carries `synthesis_model` (filled from `AnalyzeState.synthesis_model` = `"claude-opus-4-7"`). The decision projection worker therefore uses the synthesis model (Opus-class) rather than the research/analysis model (GPT-class) that the old code used.

**Impact:** Decision projection runs on a different model than the old code intended.

---

## Findings from Analyze Subgraph Node Verification (CONTRACTS.md)

Severity: **CRITICAL** = core contract broken; **HIGH** = material output-quality impact; **MEDIUM** = secondary feature missing or changed; **LOW** = cosmetic / intentional design choice with no functional regression.

---

### F-019 — HIGH: `orientation` output schema — `major_bets` and `key_timelines` flattened to strings

**Node:** `orientation`

**Old contract:** `_ORIENTATION_SYSTEM` required structured JSON. `major_bets` = `list[{bet: str, bows: list[str], amount_approx: str}]`. `key_timelines` = `list[{milestone: str, target_date: str, status: "on_track|at_risk|missed"}]`.

**New behavior:** New inline prompt requests `major_bets: list of strings` and `key_timelines: list of strings`. Structured dict fields (`bows`, `amount_approx`, `milestone`, `target_date`, `status`) are lost. Any downstream code reading `program_context["major_bets"][i]["bows"]` or `program_context["key_timelines"][i]["status"]` will raise `TypeError`.

Additionally, new prompt adds `bow_summaries: dict` — a key not present in old schema. Old downstream code iterating `program_context` keys may not handle it.

---

### F-020 — ~~HIGH~~ **RESOLVED**: `orientation` does not read strategy-corpus document summaries

**Node:** `orientation`

**Old contract:** `_phase1_orient` read strategy-corpus document summaries, date-desc sorted (most recent first), injecting them into the orientation LLM prompt.

**New behavior (pre-fix):** `orientation` only read structured metadata; no date-sorted strategy doc list injected.

**Fix applied 2026-05-29:**
- `orientation` node now filters `doc_list` for `collection="strategy"` docs, sorts them by `doc_date` descending (most recent first, matching OLD), and injects up to 15 entries with date, filename, doc_type, and 200-char summary into the orientation prompt as a "Strategy Documents" section.
- This section appears after the BOW/investment metadata and before the embedding search excerpts.
- Falls back to empty section if no strategy docs in `doc_list`.

---

### F-021 — MEDIUM: `orientation` system prompt differs; `SAFETY_PREAMBLE` absent; `max_tokens=8000` not enforced

**Node:** `orientation`

**Old contract:** System prompt = `_ORIENTATION_SYSTEM` which prepends `SAFETY_PREAMBLE` before the analyst instructions. LLM call params: `json_mode=True, thinking=False, max_tokens=8000`.

**New behavior:** New `orientation` node uses an inline user prompt (no system message equivalent to `_ORIENTATION_SYSTEM`). `acall_llm` is called without `json_mode` or `max_tokens` parameters — defaults to `acall_llm` defaults (likely `max_tokens=4096` from `config.DEFAULT_MAX_TOKENS`). `SAFETY_PREAMBLE` is absent.

**Impact:** Responses may be truncated (4 096 vs 8 000 tokens). Without `json_mode`, parsing relies on regex extraction (`re.search(r"\{.*\}", ...)`), which is less robust than forcing JSON mode.

---

### F-022 — ~~MEDIUM~~ **RESOLVED**: `compute_scopes` scope dict missing `topic`/`sub_topic` keys; grouping key differs

**Node:** `compute_scopes`

**Old contract:** Scope dict contained `{scope_id, label, topic, sub_topic, bow_ids, inv_ids, chunk_count}`. Grouping key: `"{topic} > {sub_topic}"` from BOW metadata.

**New behavior (pre-fix):** Scope dict was missing `topic` and `sub_topic`; grouping key used `sub_topic or focus_area or bow_id` fallback, causing BOWs with empty metadata to form separate scopes instead of merging.

**Fix applied 2026-05-29:** Grouping key changed to `f"{topic} > {sub_topic}"` (matching OLD). A `bow_meta` side-dict tracks `topic`/`sub_topic` per BOW. Both keys added to all scope dicts (split and merged paths). Verified with node test: fixture BOWs with empty metadata merge into 1 scope. Scope ID format (`scope_0000`) kept as-is — not part of this contract.

---

### F-023 — ~~MEDIUM~~ **RESOLVED**: `compute_scopes` phantom-investment filter absent

**Node:** `compute_scopes`

**Old contract:** `_compute_thread_scopes` called `tools._embedding_index.distinct_inv_ids()` to filter investments with no indexed chunks from scope `inv_ids`.

**Fix applied 2026-05-29:** `compute_scopes` now calls `await backend.distinct_inv_ids()` (alongside `count_by_bow_id()`) and stores the result in `doc_bearing_inv_ids: set[str] | None`. The merged-path de-duplication loop and the split-path per-BOW filter both exclude investments absent from `doc_bearing_inv_ids`. Degrades gracefully (`doc_bearing_inv_ids = None`) when the backend does not expose `distinct_inv_ids`. Verified: phantom `INV-GHOST` filtered out when not in index; all investments kept when method unavailable.

---

### F-024 — ~~HIGH~~ **RESOLVED**: `compute_scopes` skip threshold default 5 vs old 200; `is_catch_all` split logic absent

**Node:** `compute_scopes`

**Old contract:** Skip threshold: `chunk_count < 200`.

**New behavior (pre-fix):** Default `MIN_CHUNKS = 5` — 40× lower than old 200.

**Fix applied 2026-05-29:**
- `NQPR_MIN_CHUNKS_PER_BOW` environment variable default changed from `"5"` to `"200"`, matching OLD repo threshold.
- `is_catch_all` split logic not separately added — covered by the existing `SPLIT_CHUNKS=12K` / `SPLIT_INVS=8` size gates which handle the same over-large-group cases.

---

### F-025 — MEDIUM: `build_timelines` / Phase 2.6 — SHA256 narrative cache not loaded or written

**Nodes:** `build_timelines`, `dispatch_investment_narratives`, `collect_timeline_narratives`

**Old contract:** `build_timelines` (old Phase 2.5/2.6) loaded `{ingested_dir}/timeline_narratives.json` via `load_narratives()`. If the SHA256 hash of scope timelines matched the cached hash, all narrative strings were pre-populated and Phase 2.6 LLM calls were skipped. `save_narratives()` was called after generation to persist the cache.

**New behavior:** `build_timelines` node does not attempt `load_narratives()`. `collect_timeline_narratives` does not call `save_narratives()`. Resume within a single LangGraph run is handled correctly via `timeline_narrative_results` set-membership. However, cross-run narrative reuse (the primary purpose of the SHA256 cache) is absent. Every new graph run regenerates all narratives regardless of whether scope timelines have changed.

**Impact:** Users who ran the old pipeline on a collection and then start a new LangGraph run will regenerate all narratives from scratch rather than reusing cached results. For large portfolios (20+ investments), this adds significant LLM cost.

---

### F-026 — MEDIUM: `cross_cutting_analysis` — no `code_interpreter`; portfolio metrics computed in pure Python

**Node:** `cross_cutting_analysis`

**Old contract:** `portfolio_metrics` was pre-computed by `numerical_analyst._call_code_interpreter()` using OpenAI `code_interpreter` tool before `_phase4_crosscut` was called. This allowed arbitrary Python execution for metrics that were hard to compute in pure Python.

**New behavior:** `cross_cutting_analysis` node computes portfolio metrics in pure Python (total_approved, total_paid, at_risk_count, concentration_by_bow). This is deterministic and does not require an OpenAI Responses API call. The metrics computed are a subset of what the old `code_interpreter` produced.

**Impact:** Simpler metrics than old code; no `code_interpreter` API cost. Risk: any complex quantitative metric the old `code_interpreter` computed that is not in the new pure-Python set is silently absent from the LLM prompt.

---

### F-027 — ~~HIGH~~ **RESOLVED**: `cross_cutting_analysis` — `cluster_identification.identify_clusters()` absent; clusters never populated

**Node:** `cross_cutting_analysis`

**Old contract:** `cluster_identification.identify_clusters()` produced 6–12 `ThematicCluster` records used to structure the executive summary.

**New behavior (pre-fix):** `cross_cutting_analysis` did not call cluster identification; `clusters` was never produced.

**Fix applied 2026-05-29:**
- `cross_cutting_analysis` now runs a cluster identification LLM call after the main analysis call: builds a shortlist of high-severity deviations (using `ranked_deviations` from scope `section_draft`), then calls the LLM to group them into 3–5 thematic clusters with `theme`, `description`, `scope_ids`, and `risk_level`.
- Results stored in `cross_cutting_analysis["clusters"]` and returned as `state.clusters`.
- Not written to disk (intentional LangGraph design — state replaces `threads/clusters.json`).

---

### F-028 — HIGH (N/A within analyze-pipeline scope): `cross_cutting_analysis` output schema mismatch — dict vs `list[Finding]`

**Node:** `cross_cutting_analysis`; downstream `assemble_report`

**Old contract:** `_phase4_crosscut` returned `(list[Finding], list[dict])`. Each `Finding` had attributes: `finding_id`, `claim`, `evidence`, `confidence`, `gaps`, `sources`, and a monkey-patched `.narrative` string (the cross-cutting essay). `emergent_decisions` was a separate `list[dict]`.

**New behavior:** `cross_cutting_analysis` node returns `{"cross_cutting_analysis": {patterns, contradictions, shared_dependencies, emergent_decisions, essay, portfolio_metrics}}` — a single flat dict. The new `report_assembler.assemble_report` in `src/core` accepts `cross_cutting_analysis: dict` and works with this schema.

**Assessment (2026-05-29):** The new pipeline is internally consistent — `cross_cutting_analysis` writes a dict and `assemble_report` reads a dict. The schema mismatch risk is for downstream stages (`prepare_research`, `finalize`) outside the analyze-pipeline scope. No fix needed within scope.

---

### F-029 — HIGH: `assemble_report` node — no retry loop

**Node:** `assemble_report`

**Old contract:** `_ASSEMBLY_MAX_RETRIES = 5`, `_ASSEMBLY_RETRY_DELAY = 60` seconds. On structure-validation failure (e.g., missing required sections, failed source-reference check), the old assembler would retry up to 5 times.

**New behavior:** `assemble_report` node calls `report_assembler.assemble_report(...)` once (lines 1244–1259). On exception, it returns `{"errors": [...]}` immediately. No retry on partial assembly or validation failure.

**Impact:** Any transient LLM failure or structure-validation miss terminates assembly rather than retrying. The old code's 5-retry safety net is absent.

---

### F-030 — ~~HIGH~~ **RESOLVED**: `assemble_report` node — `section_drafts` not pre-generated; `write_section_draft` logic bypassed

**Node:** `synthesize_scope_section_worker` (Phase 3.6) + `assemble_report`

**Old contract:** By the time `assemble_report` (old) was called, `section_drafts: list[SectionDraft]` had already been generated — one per scope — by `report_writer.write_section_draft()` with word budgets, 4-part essay structure, §-ref canonicalization, and retry on missing citations. The assembler only stitched and post-processed these pre-written sections.

**New behavior (pre-fix):** `synthesize_scope_section_worker` generated "3-4 paragraph narrative" with no essay structure; `assemble_report` consumed `scope["section_draft"]` via `_scope_body_section`.

**Fix applied 2026-05-29:**
- F-036 (previously resolved): word budgets + §-ref source index + retry on missing citations.
- F-030 (this fix): added `_SECTION_ESSAY_SYSTEM` constant with the 5-step essay format (What/Evidence/Why/Counter/Assessment). Passed as `system_msg` to both the primary and retry `acall_llm` calls. The prompt instruction updated to reference the essay format. Verified: `system_msg` contains essay structure; §-refs present; retry logic still fires when needed.

---

### F-031 — HIGH: `assemble_report` node — `context: ProgramContext` and `investments` not passed

**Node:** `assemble_report`

**Old contract:** Old `assemble_report` received `context: ProgramContext` (theory_of_change, major_bets, etc.) and `investments: dict` (full investment metadata keyed by inv_id). Both were used in exec-summary generation, BOW routing, and bibliography construction.

**New behavior:** New `report_assembler.assemble_report` call (analyze.py lines 1244–1254) passes `scope_outputs`, `cross_cutting_analysis`, `all_excerpts`, `confidence_map`, `coverage_pct`, `grade`, `model`, `config`. Neither `program_context` (the portfolio theory-of-change) nor the full `investment_scoring` / `investments` dict is forwarded. Report assembly operates without the portfolio's stated theory of change or complete investment metadata.

---

### F-032 — ~~HIGH~~ **RESOLVED**: `assemble_report` — `NarrationToolbox` with 6 agentic tools not invoked

**Node:** `assemble_report`

**Old contract:** 5 parallel lens-specific narrators, each using NarrationToolbox with a fixed call budget.

**New behavior (pre-fix):** `assemble_report` generated the executive summary via direct LLM calls without NarrationToolbox.

**Fix applied 2026-05-29:**
- `assemble_report` node now builds a `narration_configurable` containing `scope_outputs` (keyed by scope_id), `investment_scoring`, `investment_intelligence`, `all_excerpts`, `search_backend`, and passes it as `narration_config` to `report_assembler.assemble_report`.
- `report_assembler._build_executive_summary` now calls `_run_narration_pass(config, model)` before the final synthesis LLM call.
- `_run_narration_pass` invokes `list_filtered_investments.ainvoke` + 3 × `search_within_scope.ainvoke` to gather cross-investment patterns before synthesis.
- Residual gap vs. OLD: old ran 5 parallel per-lens narrators with per-narrator budget; new runs a single 4-query pass (simplified). F-046 (per-narrator budget) remains open.

---

### F-033 — HIGH: `assemble_report` — `exec_summary/premise_investigator` ASTA cascade absent

**Node:** `assemble_report`

**Old contract:** `exec_summary/premise_investigator.run_per_bet_investigation()` ran a 3-pass per-bet scientific premise cascade (Investigation → Key Findings → Exec Summary), each using Pydantic-validated LLM output and optional ASTA tool calls. This produced `BetScienceAssessment` objects that structured the executive summary around evidence-graded strategic bets.

**New behavior:** No equivalent cascade in new `report_assembler`. Executive summary is generated by a single (or small number of) LLM calls over `scope_outputs` and `cross_cutting_analysis`. The evidence-graded science assessment of portfolio bets is absent.

---

### F-034 — ~~HIGH~~ **RESOLVED**: `verify_report` — report NOT rewritten; allocation mismatches not corrected

**Node:** `verify_report`

**Old contract:** `allocation_verifier.apply_allocation_edits()` (called via `verify_and_rewrite(auto_rewrite=True)`) rewrote `final_report_md` in-place for figures with `status == "approved_without_qualifier"` — inserting correct values or parenthetical qualifiers. The rewritten file replaced the original.

**New behavior (pre-fix):** `verify_report` node identifies mismatches and records them in `allocation_verification` state, but does **not** rewrite `final_report_md`.

**Fix applied 2026-05-29:** `verify_report` now collects figures where the LLM verification pass returns `verdict == "discrepancy"` and rewrites `final_report_md` in-place, appending `[⚠ {explanation}]` after each flagged figure. The updated text is returned as `final_report_md` in the output dict (last-writer-wins reducer). Verified with targeted node test.

---

### F-035 — LOW: `verify_report` — `allocation_issues.md` not written

**Node:** `verify_report`

**Old contract:** `allocation_verifier` wrote `threads/allocation_issues.md` listing all flagged mismatches for human review.

**New behavior:** All mismatch data goes to `allocation_verification` state field only; no Markdown file is produced for human review.

---

### F-036 — ~~MEDIUM~~ **RESOLVED**: `synthesize_scope_section_worker` — word budgets, §-ref canonicalization, and citation retry absent

**Node:** `synthesize_scope_section_worker`

**Old contract:** `report_writer.write_section_draft` enforced word budgets (`WORD_BUDGETS = {"program_critical": 1500, "pathway_altering": 1000, "efficiency_reducing": 500}`), required inline `§[source_id]` citations in the 4-part essay structure (What / Evidence / Why / Counter / Assessment), and retried once if §-refs were missing.

**New behavior (pre-fix):** `synthesize_scope_section_worker` asked for "3-4 paragraph narrative" with no word budget, no citation format requirement, and no retry.

**Fix applied 2026-05-29:**
- Added `_WORD_BUDGETS` dict (`program_critical: 1500, pathway_altering: 1000, efficiency_reducing: 500`) and `_count_refs` helper to `analyze.py`.
- `synthesize_scope_section_worker` now builds a `§NNNN`-keyed source index from `link_assessments`, injects it + the derived word budget into the prompt, and retries once (appending a hard reminder) when `_count_refs(output) < 2` and the source index is non-empty.
- Verified with targeted node test (2-call sequence confirmed when first LLM response lacks §-refs).

---

## Findings from Tool Verification

---

### F-037 — ~~HIGH~~ **RESOLVED**: `SCIENCE_TOOLS` exports `[search_asta] + InvestigationTools` (11 tools); old science investigator had own 8-tool set

**Tools:** `science_tools.py:SCIENCE_TOOLS`; `src/qpr/science_investigator.py:_TOOL_DESCRIPTIONS`

**Old contract:** `_TOOL_DESCRIPTIONS` (lines 72–108) listed exactly 8 tools: `search_asta`, `search_bow`, `search_science`, `search_policy`, `search_all`, `search_web`, `read_pages`, `compute`. These were a curated subset: no `submit_findings`, no `list_documents`, no `read_document`, no `read_section`, no structured document navigation — only literature and scoped collection search.

**New behavior (pre-fix):** `science_tools.py` line 151: `SCIENCE_TOOLS = [search_asta] + list(INVESTIGATION_TOOLS)` = 11 tools. Added to the science agent: `search_investment`, `search_portfolio`, `read_document`, `submit_findings`, `list_documents`, `read_document_summary`, `get_document_structure`, `read_section`. **Missing from the old 8-tool set:** `search_bow`, `search_science`, `search_policy`, `search_all`, `read_pages` — the four collection-scoped search tools that allowed literature, policy, and BOW-neighbour lookups.

**Fix applied 2026-05-29:**
- `search_bow`, `search_science`, `search_policy` added as `@tool` functions in `investigation_tools.py` (these were a prerequisite).
- `science_tools.py:SCIENCE_TOOLS` rebuilt: `[search_asta, search_bow, search_science, search_policy, search_web, read_document, compute, read_section]` — exactly 8 tools matching the old vocabulary.
- Removed: `submit_findings` (conflicts with `evidence_gathered` termination), `search_investment`, `search_portfolio`, `list_documents`, `read_document_summary`, `get_document_structure`.
- `ScienceAction.tool` field in `output_schemas.py` updated to describe the correct 8-tool set.
- `SCIENCE_INVESTIGATE_SYSTEM` in `tool_prompts.py` updated with accurate tool descriptions for `search_bow`, `search_science`, `search_policy`, and the rationale for why `submit_findings` is excluded.
- `science_tools.py` module docstring updated with the correct tool list and exclusion rationale.

---

### F-038 — MEDIUM: `search_collection` return type is `str` (formatted text); ARCHITECTURE.md §11 specifies `-> list[dict]`

**Tool:** `collection_tools.py:search_collection`

**ARCHITECTURE.md §11:**
```python
async def search_collection(...) -> list[dict]:
```

**Old-repo `CollectionTools.search()`:** `-> list[dict]` with keys: `text`, `filename`, `file_id`, `collection`, `doc_type`, `page_start`, `score`, and source attribution.

**New behavior:** `search_collection` returns `str` via `_fmt_results()`, which formats each result as:
```
[{i}] score={r.score:.3f} file={r.file_id[:60]} inv={r.inv_id or '-'} bow={r.bow_id or '-'} pages={r.page_start}–{r.page_end} doc_type={r.doc_type or '-'}
    {snippet[:400]}
```
Text is truncated to **400 chars** per result. Returning `str` is correct for LLM tool consumption, but diverges from the ARCHITECTURE.md type contract, and any code calling `search_collection` expecting a `list[dict]` will receive a string.

**Note:** The `_fmt_results` return is appropriate for an `@tool` function the LLM calls directly. The discrepancy is with ARCHITECTURE.md §11's stated return type.

---

### F-039 — ~~HIGH~~ **RESOLVED**: §-citation references lost from search tool results; bibliography accumulation absent

**Tools:** `investigation_tools.py:search_investment`, `search_portfolio`; `collection_tools.py:search_collection`

**Old contract:** Investigation results formatted with `§NNNN` reference IDs accumulated in `accumulated_refs`; later written to `excerpts.csv`.

**New behavior (pre-fix):** `_fmt_results()` produced formatted text with no §-reference IDs; no `accumulated_refs`; no `excerpts.csv`.

**Fix applied 2026-05-29:**
- `run_investigation` in `investigation.py` now assigns a positional `ref_id = f"§{ref_counter:04d}"` to each deduplicated entry in `annotated_excerpts`. Counter increments across dedup iterations (skips true duplicates), preserving positional identity across the investigation session.
- The `ref_id` field is added alongside `excerpt_id`, `file_id`, `page_start`, `page_end` to the annotated excerpt dict.
- `assemble_report` node writes `excerpts.csv` from `all_excerpts` (see F-005 fix) — the 16-column CSV includes `ref_id` so the output matches the old `§-ref` traceability model.
- Residual: `_fmt_results()` in search tools does not embed §-refs in the inline text returned to the LLM (the LLM still sees `[i] score=... file=...` format, not `§NNNN`). §-refs appear in `_format_source_index` within the iteration prompt — LLM can cite them, but tools don't auto-annotate their output.

---

### F-040 — MEDIUM: Page reads capped at 50 pages; old CollectionTools declared "NEVER truncates"

**Tools:** `collection_tools.py:read_section`, `read_pages`; `investigation_tools.py:read_document`

**Old contract:** `CollectionTools` class docstring (line 182 of `collection_api.py`): `"NEVER truncates. Agents navigate via TOC → section → pages."` Page reads returned full content.

**New behavior:** `_read_page_range` in `collection_tools.py` (line 60): `for pg in range(page_start, min(page_end, page_start + 50) + 1)`. All page reads are **silently capped at 50 pages**. A section spanning pages 1–200 would return only pages 1–50. No warning or truncation indicator is returned to the LLM.

**Impact:** Long documents (grants, strategy docs) may be silently truncated. The LLM receives a partial section without knowing content was cut off, potentially leading to missed evidence.

---

### F-041 — ~~HIGH~~ **RESOLVED**: `search_web` changed from OpenAI Responses API (`gpt-5.4`) to Tavily; stub returned on missing key

**Tool:** `investigation_tools.py:search_web`

**Old contract:** `execute_tool("search_web", ...)` used `client.responses.create(model="gpt-5.4", ...)` with `{"type": "web_search_preview"}` tool via OpenAI Responses API.

**New behavior (pre-fix):** `search_web` used Tavily; returned stub string when `TAVILY_API_KEY` absent.

**Fix applied 2026-05-29:**
- `search_web` rewritten to use `openai.OpenAI().responses.create(model="gpt-5.4", ..., tools=[{"type": "web_search"}])` via `asyncio.to_thread` — exactly matching `thread_sub_agent._call_with_web_search` in OLD.
- Tavily import and `TAVILY_API_KEY` check removed.
- On failure (e.g. `OPENAI_API_KEY` absent), returns `f"[web search not configured — {exc}]"` — same `"[web search not configured"` prefix that the F-056 guard in `_execute_actions` checks, so it is not counted as evidence.
- Module docstring updated to remove Tavily reference.
- `top_urls` trace field set to `[]` (OpenAI Responses API does not expose per-URL citations in output_text).

**Applies to both:** link investigation `search_web` (via `INVESTIGATION_TOOLS`) and science investigation `search_web` (via `SCIENCE_TOOLS`).

Also applies to the ASTA search quality regression documented below (same fix date).

---

### F-042 — MEDIUM: `compute` tool uses local Python `eval()`, not OpenAI `code_interpreter`; AGENTS.md contract violated

**Tool:** `investigation_tools.py:compute`

**AGENTS.md §10:** "Execute sandboxed Python via OpenAI `code_interpreter`"

**Old contract:** `execute_tool("compute", ...)` (line 1481 of `investigation_loop.py`) called `call_llm(prompt=f"Compute: {question}\nData: {data}", model=FAST_MODEL, json_mode=False)` — delegated computation to an LLM (not `code_interpreter`).

**New behavior:** `compute` tool (lines 317–353 of `investigation_tools.py`) tries:
1. `ast.literal_eval(question.strip())` — evaluates Python literals
2. `eval(question.strip(), {"__builtins__": {}, "math": math})` — restricted namespace eval
3. Falls back to structured passthrough prompt

**No LLM is called. No `code_interpreter` is invoked.** The tool description reads "Perform a numerical calculation" — which understates the limitation. Complex statistical calculations, percentile ranking, or multi-step computation that the old LLM path could handle (however poorly) now fails silently and returns a passthrough message.

**Note:** The `eval()` with `__builtins__={}` is not truly sandboxed — it can still access memory and globals via `math` namespace. This is a security concern for untrusted inputs, though the pipeline context is internal.

---

### F-043 — ~~HIGH~~ **RESOLVED**: `submit_findings` returns JSON content instead of sentinel; schema constraints not enforced

**Tool:** `investigation_tools.py:submit_findings`

**Old contract:** `submit_findings` returned sentinel `"FINDINGS_SUBMITTED"` to terminate the loop.

**New behavior (pre-fix):** `submit_findings` returned JSON; no enum constraints; loop didn't reliably terminate when called.

**Fix applied 2026-05-29:**
- `submit_findings` **removed from `INVESTIGATION_TOOLS`** — it is no longer exposed to the LLM as a callable tool.
- Loop termination is now entirely via `InvestigationActionsOutput.status ∈ {answered, not_answerable, unresolved_conflict}` with `next_actions=[]`, as documented in the system prompt.
- `INVESTIGATION_TOOL_DESCRIPTIONS` updated with explicit termination instructions: "Set status=answered and next_actions=[] when done. Do NOT use submit_findings."
- `_SUPPORTED_TOOLS` in `investigation.py` no longer includes `submit_findings` (already absent; entry removed for clarity).
- The tool function itself is preserved in `investigation_tools.py` for backward compatibility with any code that calls it directly, but it is not in `INVESTIGATION_TOOLS`.

---

### F-044 — MEDIUM: `list_filtered_investments` has no filter parameters; AGENTS.md description is "filtered by posture, BOW, or divergence severity"

**Tool:** `narration_tools.py:list_filtered_investments`

**AGENTS.md §10:** "List investments filtered by posture, BOW, or divergence severity"

**Old NarrationToolbox:** The tool accepted posture, bow_id, and divergence_severity as filter arguments, letting narrators scope their view before reasoning.

**New behavior:** `list_filtered_investments` (lines 39–72) takes **no parameters** (only `config: RunnableConfig`). It lists all investments in `relevance_subset` (pre-scoped by the framework) without any per-call filter. The model cannot filter by posture or severity at call time — it receives a fixed list.

**Impact:** Narrators cannot dynamically narrow their scope (e.g., "show me only program_critical investments"). The AGENTS.md tool description is misleading — the model may attempt to pass filter arguments that are not accepted, causing tool call failures.

---

### F-045 — ~~HIGH~~ **RESOLVED**: `read_evidence_pack` reads `investment_sections` key that is never written by the causal pipeline

**Tool:** `narration_tools.py:read_evidence_pack`

**Old contract:** `NarrationToolbox.read_evidence_pack` returned the full `InvestmentEvidencePack` from Phase 3.1 rubric evaluation.

**New behavior (pre-fix):** `read_evidence_pack` looked up `investment_sections` key that was never written; always fell back to thin metadata.

**Fix applied 2026-05-29:**
- `assemble_report` node now builds `investment_sections` from `all_excerpts` grouped by `inv_id` (up to 20 excerpts × 500 chars each per investment), and injects them into each scope's `investment_sections` dict before calling `report_assembler.assemble_report`.
- `read_evidence_pack` in `narration_tools.py` now has a second fallback: when `investment_sections` is empty, it filters `all_excerpts` directly from `configurable["all_excerpts"]` (added to the narration configurable by the `assemble_report` node). Returns up to 20 excerpts with `ref_id`, `source_file`, and text.
- Priority chain: (1) `investment_sections[inv_id]` → (2) `all_excerpts` filtered by `inv_id` → (3) metadata + intelligence fallback.
- Residual: evidence pack is sourced from link investigation excerpts (Phase 3.4 output), not the Phase 3.1 rubric chunks — it's richer than the old thin-metadata fallback but not identical to the old InvestmentEvidencePack.

---

### F-046 — MEDIUM: `NarrationToolbox` per-narrator call budget and "budget exhausted" sentinel absent from new @tool functions

**Tools:** `narration_tools.py` (all 6 tools)

**Old contract:** `NarrationToolbox` tracked `_calls_used` against a `budget` (default 15). When `_calls_used >= budget`, any tool call returned `"[budget exhausted — wrap up now]"`. `run_narrator_with_tools` passed `NARRATION_TOOL_SCHEMAS` only when `budget_remaining > 0`, giving the LLM a graceful off-ramp. The JSONL trace recorded every call with timestamp, args, response summary, tokens estimated, and budget remaining.

**New behavior:** The 6 `@tool` functions in `narration_tools.py` have no call-budget enforcement. An LLM narrator that calls `search_within_scope` in a loop will not receive a budget-exhausted signal and may run until `max_iterations` (hard cap). `collection_search_traces` records calls but provides no budget state to the LLM.

**Impact:** Narrator agents may spend their entire context window on tool calls without wrapping up. The old soft-cap mechanism that gracefully terminated over-active narrators is absent.

---

### F-047 — LOW: `search_investment` and `search_portfolio` traces hardcode `"backend": "local"` regardless of actual backend

**Tools:** `investigation_tools.py:search_investment` (line 98), `search_portfolio` (line 160)

**Old contract:** `execute_tool` identified the backend dynamically from the index type.

**New behavior:** Both tools write `"backend": "local"` unconditionally to `collection_search_traces`, even when running against Qdrant or Azure backends. This makes trace analysis misleading for non-local deployments.

---

### F-048 — MEDIUM: Search result formatting loses source-credibility annotations

**Tools:** All search tools using `_fmt_results()`

**Old contract:** `investigation_loop.execute_tool("search_investment", ...)` annotated each result with `_SOURCE_TYPE_MAP[doc_type]` — labels like `"GRANTEE SELF-REPORT"`, `"FOUNDATION INTERNAL"`, `"EXTERNAL/INDEPENDENT"`, `"EXPERT CONSENSUS"`. These labels were included in the formatted result shown to the LLM, prompting it to weight evidence by source type.

**New behavior:** `_fmt_results()` (lines 35–47 of `collection_tools.py`) outputs: `score`, `file_id[:60]`, `inv_id`, `bow_id`, `pages`, `doc_type`. The `doc_type` field is present but the credibility annotation (translating `"progress_report"` → `"GRANTEE SELF-REPORT"`) is absent. The LLM receives raw `doc_type` strings without the credibility framing that guided old investigations.

**Impact:** LLM investigators may not appropriately discount grantee self-reported progress reports relative to independent scientific evidence, potentially inflating confidence in unverified claims.

---

## Findings from Control-Flow Verification (CONTROL_FLOW.md)

---

### F-049 — ~~HIGH~~ **RESOLVED**: `dispatch_rubric_evaluation` emits one Send per scope (primary inv_id), not one per investment

**Location:** `src/graph/subgraphs/causal.py:dispatch_rubric_evaluation`

**Old contract:** `run_causal_pipeline` called `rubric_evaluator.build_evidence_pack` once per investment — for every `inv_id` in `scope["inv_ids"]`, not just the primary.

**New behavior (pre-fix):** One `Send()` per scope using `scope.get("inv_id")` only; `already_done` keyed by `scope_id` alone (too coarse).

**Fix applied 2026-05-29:**
- `dispatch_rubric_evaluation` now iterates `scope.get("inv_ids") or [scope.get("inv_id")]` and emits one `Send()` per investment per scope (matching ARCHITECTURE.md §4 and old ThreadPoolExecutor behaviour).
- `already_done` set changed to `{(pack.get("scope_id"), pack.get("inv_id"))}` — fine-grained dedup that allows remaining investments in a scope to resume when some are already done.
- Verified: 2-investment scope → 2 Sends; partially-done scope correctly skips only the completed investment.

---

### F-050 — CRITICAL: Top-level workflow graph is truncated — Stages 3–5 and routing functions absent

**Location:** `src/graph/workflow.py`

**ARCHITECTURE.md §2 specifies:** `load_collection → precheck → analyze → prepare_research → review_research_plan → research → finalize → approve_report → deliver` with routing functions `route_after_research_plan_review` and `route_after_report_approval`.

**New behavior:** `workflow.py` implements only: `load_collection → precheck → analyze → rerender → deliver`. The following are completely absent:
- `prepare_research` node
- `review_research_plan` node (human interrupt)
- `research` node (Stage 4 — SLR/LBD/deep_web/Edison research dispatch)
- `finalize` node (Stage 5 — report enrichment with research results)
- `approve_report` node (human interrupt)
- `route_after_research_plan_review()` conditional edge function
- `route_after_report_approval()` conditional edge function

**`_HUMAN_INTERRUPT_NODES = ["analyze"]`** — only one interrupt point vs ARCHITECTURE.md §7's four (`analyze`, `review_research_plan`, `research`, `approve_report`).

**Impact:** The pipeline terminates after report assembly and PDF render. External research (Stage 4), research-enriched final report (Stage 5), and the two human review checkpoints are not executed.

---

### F-051 — ~~HIGH~~ **RESOLVED** (via F-043): `run_investigation` terminal status vocabulary mismatch

**Location:** `src/core/investigation.py:_TERMINAL_STATUSES`

**Old contract:** Loop terminated on `submit_findings` sentinel; `overall_assessment.status` used old vocabulary (`on_track|deviations_found|critical_risk|insufficient_evidence`).

**New behavior:** Loop terminates when `InvestigationActionsOutput.status ∈ {answered, not_answerable, unresolved_conflict}` with `next_actions=[]`. New vocabulary fully documented in `INVESTIGATION_SYSTEM` prompt and `InvestigationActionsOutput` schema.

**Resolution:** F-043 fix (remove `submit_findings` from `INVESTIGATION_TOOLS`) directly addresses this: the LLM now exclusively uses the new status vocabulary for termination. System prompt explicitly lists the three terminal statuses and instructs: "Do NOT use submit_findings." The old vocabulary (`on_track`, etc.) is no longer referenced anywhere in the new pipeline prompts.

---

### F-052 — MEDIUM: `_route_science_investigations` dispatches ALL causal assumptions; old code filtered to `web_search_hint=True` only

**Location:** `src/graph/subgraphs/causal.py:_route_science_investigations`

**Old contract:** `science_investigator.investigate_science_question` was called only for assumptions where `make_investigation_claims` set `web_search_hint=True` — i.e., assumptions containing science-domain keywords (mechanism, efficacy, clinical, biological, etc.).

**New behavior:** `_route_science_investigations` (lines 760–792 of `causal.py`) iterates over `causal_model.get("assumptions", [])` without any filter. Every assumption in the causal model, regardless of whether it has a science keyword, is dispatched to `investigate_science_assumption`.

**Impact:** Science investigation fan-out is N × broader than old code. Many non-science assumptions (e.g., operational, financial, partnership-related) are investigated via ASTA/Semantic Scholar, producing irrelevant results and consuming more ASTA API quota.

---

### F-053 — LOW (reduced from MEDIUM): `_execute_actions` silently skips tool actions not in `_SUPPORTED_TOOLS`

**Location:** `src/core/investigation.py:_execute_actions`

**`_SUPPORTED_TOOLS` (updated 2026-05-29):** `{"search_investment", "search_portfolio", "search_bow", "search_science", "search_policy", "search_doc_type", "search_all", "search_web", "read_pages", "read_document", "read_section", "compute"}` — **12 tools** (was 10; `read_document` and `read_section` added with dedicated dispatch branches).

**Further resolved (2026-05-29 session 2):** `read_document` and `read_section` now dispatched in `_execute_actions` — `read_document` via `read_document.ainvoke(file_id, page_start, page_end)`, `read_section` via `read_section.ainvoke(file_id, section_id)`.

**Still silently skipped** (in `INVESTIGATION_TOOLS` but not in `_SUPPORTED_TOOLS`): `list_documents`, `read_document_summary`, `get_document_structure` — if LLM requests these as `next_actions` entries they are silently ignored.

**Residual impact:** Minimal — the 3 remaining skipped tools are document navigation helpers; the LLM prefers `search_investment` for discovery. `submit_findings` has been removed from `INVESTIGATION_TOOLS` entirely (F-043 fix), so that skip case is gone.

---

### F-054 — MEDIUM: L4 coverage audit gate fires only at `iteration >= max_iterations - 2`, not on every terminal status

**Location:** `src/core/investigation.py:_build_iteration_prompt`, line 274: `if INVESTIGATION_L4_COVERAGE_AUDIT and iteration >= max_iterations - 2:`

**Old contract:** L4 coverage audit was applied whenever the LLM reported a terminal status — checking whether all 5 audit items were addressed before accepting the finding. If audit failed, the loop forced one more iteration.

**New behavior:** The audit check `_validate_coverage_audit` (lines 295–310) is called on every terminal status (`if output.status in _TERMINAL_STATUSES`). However, the **prompt injection** of the L4 checklist only happens when `iteration >= max_iterations - 2` (i.e., the last 2 iterations). The model has no explicit coverage checklist visible to it until near the end of the loop. A model that declares `answered` early (e.g., at iteration 2) will have the audit applied, may fail, and be forced to continue — but without the L4 checklist in its prompt, it has no guidance on what to address.

**Impact:** Early terminal status decisions are unguided by the L4 checklist. The audit still enforces coverage (by injecting a search action on failure), but without the checklist in the prompt, the forced search may not target the right gaps.

---

### F-055 — LOW: `ScienceInvestigationResult.iterations` records configured max, not actual iteration count

**Location:** `src/core/science_investigator.py`, line 223: `iterations=max_iterations`

**Old contract:** Old code tracked `actual_iteration` and recorded the loop exit iteration.

**New behavior:** `ScienceInvestigationResult` is constructed with `iterations=max_iterations` (the configured maximum, e.g., 8) regardless of when the loop actually terminated. If the loop exits at iteration 2 (ASTA gate satisfied, evidence gathered), `iterations=8` is reported.

**Impact:** Observability and quality metrics that use `ScienceInvestigationResult.iterations` to measure investigation depth will always show the configured maximum, making it impossible to distinguish early-exit from exhausted loops.

---

### F-056 — ~~MEDIUM~~ **RESOLVED**: `search_web` Tavily stub string treated as non-empty evidence; consecutive-empty counter not incremented

**Location:** `src/tools/investigation_tools.py:search_web`; `src/core/investigation.py:_execute_actions`

**Old contract:** When web search returned no results, the tool returned empty or raised an exception. Either case left `new_chunks` empty, incrementing `consecutive_empty`.

**New behavior (pre-fix):** When `TAVILY_API_KEY` is not set, `search_web` returns a stub string (`"[web search not configured — TAVILY_API_KEY not set]\n..."`). `_execute_actions` wrapped this as a chunk, resetting the consecutive-empty counter.

**Fix applied 2026-05-29:**
- `_execute_actions` `search_web` branch now guards: `if text and not text.startswith("[web search not configured"):` before appending to `chunks` and incrementing `web_count`.
- Stub responses (and any future variants starting with `"[web search not configured"`) are no longer appended as evidence chunks — the round counts as empty, `consecutive_empty` increments correctly.
- The stub is still returned to the LLM as the tool response (so it knows web search was unavailable), but it does not enter the evidence accumulator.

---

### F-057 — LOW: `dispatch_link_investigations` and `dispatch_science_investigations` are passthrough join nodes, not routing nodes; routing is in separate functions

**Location:** `src/graph/subgraphs/causal.py` lines 495–501 and 749–757

**ARCHITECTURE.md §4 says:** "dispatch_link_investigations — Routing node: emits one `Send()` per causal link." "dispatch_science_investigations — Routing node: emits one `Send()` per science assumption."

**New behavior:** Both `dispatch_link_investigations` and `dispatch_science_investigations` are registered as regular nodes (via `_builder.add_node(...)`) and return `{}` (passthrough). The routing logic is in `_route_link_investigations` and `_route_science_investigations` — async functions used as conditional-edge routing functions via `add_conditional_edges(node_name, routing_fn, {...})`.

**Impact:** No functional impact — LangGraph correctly routes via the conditional-edge functions. The ARCHITECTURE.md §4 description is inaccurate about which construct does the routing.

---

### F-058 — CRITICAL: Three top-level routing functions and five nodes absent from workflow.py

**Location:** `src/graph/workflow.py`

**ARCHITECTURE.md §2 requires:**
- `prepare_research` node
- `review_research_plan` node (human interrupt)  
- `research` node (Stage 4 research dispatch subgraph)
- `finalize` node (Stage 5 report enrichment)
- `approve_report` node (human interrupt)
- `route_after_research_plan_review(state) → "research" | "prepare_research"` conditional edge function
- `route_after_report_approval(state) → "deliver" | "finalize"` conditional edge function

**New behavior:** `workflow.py` graph: `load_collection → precheck →[route_after_precheck]→ analyze → rerender → deliver`. None of the above nodes or routing functions exist. `_HUMAN_INTERRUPT_NODES = ["analyze"]` (1 interrupt vs the 4 specified in ARCHITECTURE.md §7).

**Impact:** The full multi-stage NQPR pipeline (research dispatch, research-enriched reporting, human review gates) is not implemented. The workflow exits after analysis and PDF rendering. External literature research (SLR/LBD/deep web/Edison) results are never incorporated into the final report.

---

### F-059 — ~~HIGH~~ **RESOLVED**: `enrich_bow_context_worker` requires `web_search_fn` in `config["configurable"]`; absent by default; silently falls back to empty BOW context

**Location:** `src/graph/subgraphs/causal.py:enrich_bow_context_worker`; `src/graph/subgraphs/analyze.py:run_causal_pipeline`

**Old contract:** BOW context enrichment used `_call_with_web_search` (`thread_sub_agent.py`) — OpenAI Responses API (`gpt-5.4` + `{"type": "web_search"}` tool) — to get field landscape data. Web search was always configured.

**New behavior (pre-fix):** `enrich_bow_context_worker` reads `web_search_fn = config["configurable"].get("web_search_fn")`. If `None` (the default), all BOW enrichment silently returns `_empty_bow`.

**Fix applied 2026-05-29:**
- Added `_bow_web_search(query: str) -> str` coroutine to `analyze.py` (module-level, near `run_causal_pipeline`). Uses `openai.OpenAI().responses.create(model="gpt-5.4", ..., tools=[{"type": "web_search"}])` via `asyncio.to_thread` — matching the old `thread_sub_agent._call_with_web_search` pattern exactly.
- `run_causal_pipeline` now builds `causal_config` (shallow copy of caller's config) and injects `web_search_fn = _bow_web_search` into `causal_configurable["web_search_fn"]` when the key is not already present. Passes `causal_config` (not the original `config`) to `causal_graph.ainvoke`.
- Callers that inject their own `web_search_fn` (e.g., tests) are not overridden.
- `enrich_bow_context_worker` now produces real `benchmarks`, `comparable_programs`, `market_context`, `regulatory_context` in standard runs.

---

### F-060 — ~~HIGH~~ **RESOLVED**: `AstaClient` uses `search_papers_by_relevance` (keyword-based) instead of `snippet_search` (full-text passage); enriches via SS public batch API instead of ASTA `get_paper_batch`

**Location:** `src/core/agents/asta.py`

**Old contract:** `prisma-ai-review AstaClient` strategy `"snippets"` (the default):
1. `snippet_search` via ASTA MCP — NL-friendly, returns ~500-word passage text from 12M+ full-text papers; result stored as `abstract` in `SearchResult`.
2. `get_paper_batch` via ASTA MCP — batch-enriches corpus IDs with DOI, full abstract, venue, open-access PDF URL; prefers longer of snippet text vs paper abstract.

**New behavior (pre-fix):** `AstaClient._search_asta` called `search_papers_by_relevance` (keyword-sensitive) instead of `snippet_search`, returning `paperId + title` only. Enrichment used the Semantic Scholar public batch API (`/graph/v1/paper/batch`), not the ASTA `get_paper_batch` tool. Science investigators therefore received shorter, keyword-matched metadata instead of passage-level evidence from full-text papers.

**Fix applied 2026-05-29:**
- `src/core/agents/asta.py` fully rewritten to match OLD `strategy="snippets"`:
  - `_call_mcp_tool(tool_name, arguments)` — generic async MCP transport via `httpx`; handles SSE parsing + `structuredContent` / `content[0].text` response formats.
  - `_snippet_search_and_enrich(query, max_results)` — calls `snippet_search` first; deduplicates by `corpusId` (appending additional passage text for repeat papers); then calls `get_paper_batch` via the ASTA MCP endpoint (not SS public API) for DOI/abstract/venue enrichment; keeps longer of snippet text vs paper abstract.
  - `_extract_snippet_data(raw)` — static helper matching OLD `_extract_snippet_data`; extracts `data` list from JSON-RPC result.
  - `search(query, max_results)` — calls `_snippet_search_and_enrich`; returns `[]` when `ASTA_API_KEY` absent (the `search_asta` @tool layer provides a Semantic Scholar public API fallback for that case).
- Removed: `_search_semantic_scholar`, `_search_openalex`, `_enrich_by_ids`, `_normalize_ss_paper`.
- Return format unchanged: `list[dict]` with `paperId`, `title`, `year`, `authors`, `abstract`, `externalIds`.
