# Control-Flow Verification — Analyze Pipeline

Scope: every branch, loop, and early-exit in the old analyze pipeline and its LangGraph equivalent.
Sources: old-repo `research_analyst_agent.py`, `causal_pipeline.py`, `investigation_loop.py`, `science_investigator.py`; new-repo `workflow.py`, `subgraphs/analyze.py`, `subgraphs/causal.py`, `core/investigation.py`, `core/science_investigator.py`.

Verdict key: ✓ = matches old behavior; ✗ = mismatch (see FINDINGS.md); ⚠ = partial or nuanced.

---

## A. Top-Level Graph (workflow.py)

### A1. After precheck — route to analyze or terminate

| | Old | New |
|---|---|---|
| **Condition** | `precheck_result.passed` | `state.get("precheck_passed")` |
| **True → next** | `_cmd_analyze` | `"analyze"` node |
| **False → next** | `sys.exit(1)` | `END` |
| **Routing fn** | N/A (CLI check) | `route_after_precheck(state) -> "analyze" if state.get("precheck_passed") else END` |
| **Verdict** | ✓ | condition and both branches correct |

### A2. After research-plan review — research or regenerate

| | Old | New |
|---|---|---|
| **Condition** | Human approves/regenerates plan via CLI | `route_after_research_plan_review` |
| **New routing fn** | — | **ABSENT** — workflow.py has no `prepare_research`, `review_research_plan`, or `research` nodes |
| **Verdict** | ✗ F-058 | Node and routing function completely absent |

### A3. After report approval — deliver or re-finalize

| | Old | New |
|---|---|---|
| **Condition** | Human approves/revises final report | `route_after_report_approval` |
| **New routing fn** | — | **ABSENT** — `finalize`, `approve_report` nodes absent |
| **Verdict** | ✗ F-058 | Node and routing function completely absent |

### A4. Human-in-the-loop interrupt points

| ARCHITECTURE.md §7 interrupt | workflow.py `_HUMAN_INTERRUPT_NODES` |
|---|---|
| `analyze` (interrupt_before) | `"analyze"` ✓ |
| `review_research_plan` (interrupt_before) | **absent** ✗ F-058 |
| `research` (interrupt_before) | **absent** ✗ F-058 |
| `approve_report` (interrupt_before) | **absent** ✗ F-058 |

---

## B. Analyze Subgraph Conditional Edges (analyze.py)

### B1. Phase 0 — `load_catalog` early-exit guard

| | Old | New |
|---|---|---|
| **Condition** | Data always pre-loaded before `run_research_analyst` | `if state.get("investment_scoring"): return {}` |
| **Guard reads** | N/A | `AnalyzeState.investment_scoring` |
| **Exit when satisfied** | N/A | Returns `{}` (no-op) |
| **Exit when absent** | N/A | Reads JSON from disk → fills state fields |
| **Verdict** | ✓ (new node; guard is correct) | |

### B2. Phase 2 — `compute_scopes` focus_bows branch

| | Old | New |
|---|---|---|
| **Condition** | `if focus_bows and bow_id not in focus_bows: continue` | `if focus_bows and bow_id not in focus_bows: continue` |
| **Effect** | BOW excluded from scopes list | BOW excluded from `active_bow_ids` set |
| **Verdict** | ✓ equivalent | |

### B3. Phase 2 — `compute_scopes` phantom-BOW skip

| | Old | New |
|---|---|---|
| **Condition** | `if nc < 200: skip` | `if bow_chunk_counts and bow_chunk_counts.get(bow_id, 0) < MIN_CHUNKS: continue` |
| **Threshold** | 200 chunks | `MIN_CHUNKS = int(os.environ.get("NQPR_MIN_CHUNKS_PER_BOW", "5"))` — default **5** |
| **Verdict** | ✗ F-024 | Default threshold 40× lower; many near-empty BOWs pass that old code would skip |

### B4. Phase 2 — `compute_scopes` split-scope branch

| | Old | New |
|---|---|---|
| **Condition** | `total_chunks > 12_000 and len(group_bows) > 1 OR total_invs > 8 OR is_catch_all` | `(group_chunk_count > SPLIT_CHUNKS) OR (len(unique_inv_ids) > SPLIT_INVS)` |
| **SPLIT_CHUNKS** | 12 000 | 12 000 ✓ |
| **SPLIT_INVS** | 8 | 8 ✓ |
| **`is_catch_all` trigger** | Yes — extra split trigger for catch-all groupings | **Absent** ✗ F-024 |
| **Verdict** | ⚠ | Main thresholds match; `is_catch_all` missing |

### B5. Phase 2.6 — `dispatch_investment_narratives` fan-out routing

| | Old | New (conditional edge function) |
|---|---|---|
| **Guard: no scopes** | N/A | `if not scopes: return "collect_investment_narratives"` |
| **Resume skip** | SHA256 cache check on disk | `already_done = {f"{r['scope_id']}:{r['inv_id']}" for r in investment_narrative_results}` |
| **Fan-out** | Per-investment `_build_single_investment_narrative` | `Send("generate_investment_narrative", ...)` per investment |
| **Passthrough edge** | N/A | `return sends or "collect_investment_narratives"` |
| **Default when all done** | Skip narrative phase | Routes to `"collect_investment_narratives"` passthrough ✓ |
| **Verdict** | ✓ structure; ⚠ cache mechanism | SHA256 disk cache replaced by set-membership (F-025) |

### B6. Phase 2.6 — `dispatch_scope_syntheses` fan-out routing

| | Old | New |
|---|---|---|
| **Guard: no narratives** | N/A | `if not inv_results: return "collect_timeline_narratives"` |
| **Resume skip** | SHA256 cache | `already_done = {r.get("scope_id") for r in timeline_narrative_results}` |
| **Passthrough edge** | N/A | `return sends or "collect_timeline_narratives"` ✓ |
| **Verdict** | ✓ | |

### B7. Phase 3.5 — `dispatch_investment_reports` fan-out routing

| | Old | New |
|---|---|---|
| **Guard: no scope_outputs** | N/A | `if not scope_outputs: return "collect_investment_reports"` |
| **Resume skip condition** | N/A | `"investment_report" in s` (scope dict key presence) |
| **Passthrough edge** | N/A | `return sends or "collect_investment_reports"` ✓ |
| **Verdict** | ✓ | New phase not in old pipeline; routing logic is correct |

### B8. Phase 3.6 — `dispatch_scope_sections` fan-out routing

| | Old | New |
|---|---|---|
| **Guard: no scope_outputs** | N/A | `if not scope_outputs: return "collect_scope_sections"` |
| **Resume skip condition** | N/A | `"section_draft" in s` (scope dict key presence) |
| **Passthrough edge** | N/A | `return sends or "collect_scope_sections"` ✓ |
| **Verdict** | ✓ | |

### B9. Phase 5 — `cross_cutting_analysis` early exit

| | Old | New |
|---|---|---|
| **Condition** | N/A (always ran) | `if not scope_outputs: return {"cross_cutting_analysis": {}}` |
| **Verdict** | ✓ (safe guard) | |

### B10. Phase 6b — `verify_report` early exit

| | Old | New |
|---|---|---|
| **Condition** | N/A (always ran on report path) | `if not final_report_md: return {}` |
| **Verdict** | ✓ (safe guard) | |

### B11. Phase 6b — `assemble_report` retry loop

| | Old | New |
|---|---|---|
| **Loop** | `for attempt in range(_ASSEMBLY_MAX_RETRIES)` — up to 5 retries | **No loop** — single call to `report_assembler.assemble_report()` |
| **Retry condition** | Structure validation failure | N/A |
| **Max retries** | 5 (`_ASSEMBLY_MAX_RETRIES=5`) | 0 (no retry) |
| **Verdict** | ✗ F-029 | Retry loop absent |

---

## C. Causal Subgraph Conditional Edges (causal.py)

### C1. Stage 3.1 — `dispatch_rubric_evaluation` fan-out routing

| | Old | New |
|---|---|---|
| **Source node** | N/A (node IS the router) | `START` conditional edge — `dispatch_rubric_evaluation` IS the routing function |
| **What it fans out on** | One `build_evidence_pack` call per investment per scope | One `Send("evaluate_investment_rubric", ...)` per **scope** (primary `inv_id` only) |
| **ARCHITECTURE.md spec** | "one Send() per investment" | One Send per scope ✗ F-049 |
| **Resume skip condition** | Per-scope checkpoint file on disk | `already_done = {pack.get("scope_id") for pack in evidence_packs}` — keyed by scope_id |
| **Passthrough edge** | N/A | `return sends or "collect_evidence_packs"` |
| **Fallback when empty** | No scopes → skip | Routes to `"collect_evidence_packs"` directly ✓ |
| **Verdict** | ✗ F-049 | Per-scope not per-investment; multi-investment scopes under-evaluated |

### C2. Stage 3.1.5 — `dispatch_bow_enrichment` fan-out routing

| | Old | New |
|---|---|---|
| **Guard: no scope_outputs** | N/A | `if not scope_outputs: return "collect_bow_enrichment"` ✓ |
| **Resume skip condition** | N/A | `"bow_context" in s` key presence ✓ |
| **Passthrough edge** | N/A | `return sends or "collect_bow_enrichment"` ✓ |
| **Worker early exits** | N/A | `if not bow_ids → _empty_bow` ✓; `if not web_snippets → _empty_bow` (silently fails without web_search_fn) ✗ F-059 |
| **Verdict** | ⚠ | Routing ✓; worker silently fails without web_search_fn (F-059) |

### C3. Stage 3.4 — `_route_link_investigations` routing function

| | Old | New |
|---|---|---|
| **ARCHITECTURE.md says** | `dispatch_link_investigations` is a "Routing node" emitting Sends | `dispatch_link_investigations` is a passthrough node returning `{}`; routing is in `_route_link_investigations` conditional edge fn ✗ F-057 |
| **Source of links** | `InvestigationClaim` objects from `make_investigation_claims()` | `causal_model.get("links", [])` dict from CausalState scope |
| **Resume skip condition** | Per-link checkpoint file on disk | `already_done = {(a.get("scope_id"), a.get("link_id")) for a in link_assessments}` — tuple key ✓ |
| **Passthrough edge** | N/A | `return sends or "collect_link_assessments"` ✓ |
| **Default when no links** | Skip link investigation | Routes to `"collect_link_assessments"` ✓ |
| **Verdict** | ⚠ F-057 | Routing logic correct; node/function split differs from ARCHITECTURE.md spec |

### C4. Stage 3.5d — `_route_science_investigations` routing function

| | Old | New |
|---|---|---|
| **ARCHITECTURE.md says** | `dispatch_science_investigations` is a routing node | Same pattern as above: passthrough node + conditional edge fn ✗ F-057 |
| **Source of assumptions** | `InvestigationClaim` objects with `web_search_hint=True` filter | `causal_model.get("assumptions", [])` — ALL assumptions, no filter ✗ F-052 |
| **Resume skip condition** | N/A | `already_done = {r.get("assumption_id") for r in science_results}` ✓ |
| **assumption_id format** | Per-claim unique ID | `f"{scope_id}_{i}"` — index-based ✓ |
| **Passthrough edge** | N/A | `return sends or "collect_science_results"` ✓ |
| **Verdict** | ✗ F-052, ✗ F-057 | No web_search_hint filter; all assumptions dispatched |

### C5. Stage 3.7→3.8 — `dispatch_decision_projections` routing function

| | Old | New |
|---|---|---|
| **Source** | Called from `run_research_analyst` AFTER `run_causal_pipeline` | Inside causal subgraph, called from `necessity_check` via conditional edge |
| **Condition** | Per-scope decision projection | `already_done = {d.get("scope_id") for d in scope_decisions}` ✓ |
| **Passthrough edge** | N/A | `return sends or "collect_decisions"` ✓ |
| **Verdict** | ⚠ | Location moved (old: post-causal; new: in-causal) — covered by F-018/F-035; routing logic itself ✓ |

### C6. Stage 3.5 synthesis chain ∥ Stage 3.5d science fan-out — parallel join at necessity_check

| | ARCHITECTURE.md §4 | New |
|---|---|---|
| **Spec** | "synthesize_findings → critique_synthesis → identify_gaps" ∥ "dispatch_science_investigations → investigate_science_assumption → collect_science_results" → both join at necessity_check | `_builder.add_edge("identify_gaps", "necessity_check")` + `_builder.add_edge("collect_science_results", "necessity_check")` |
| **Join semantics** | LangGraph waits for ALL incoming edges | LangGraph fan-in: `necessity_check` executes only after both branches complete ✓ |
| **Verdict** | ✓ | |

---

## D. Investigation Loop (core/investigation.py)

### D1. Main loop — `run_investigation` iteration and termination

| Termination condition | Old | New |
|---|---|---|
| **Max iterations** | `max_iterations=20` (default `NQPR_LINK_REASONING_EFFORT` env) | `for iteration in range(max_iterations)` where `max_iterations=MAX_INVESTIGATION_ITERATIONS=40` |
| **Default max** | 20 | 40 (differs!) ✗ |
| **Terminal status detected** | `status in {"on_track","deviations_found","critical_risk","insufficient_evidence"}` (from `submit_findings`) | `status in _TERMINAL_STATUSES = {"answered","not_answerable","unresolved_conflict"}` ✗ F-051 |
| **Terminal via tool call** | `"FINDINGS_SUBMITTED"` sentinel returned by `execute_tool("submit_findings")` | No sentinel — `InvestigationActionsOutput.status` field checked ✗ F-043 |
| **No actions returned** | `if not actions: break` | `if not actions: break` ✓ |
| **LLM error** | Raises (propagates up) | `except Exception: break` — swallows LLM failure, exits loop ⚠ |

### D2. Empty-round consecutive counter

| Condition | Old | New |
|---|---|---|
| **"Empty round" definition** | No new chunks returned by tool actions | `not new_chunks` after dedup ✓ |
| **Threshold** | 3 | `CONSECUTIVE_EMPTY_THRESHOLD=3` ✓ |
| **Action on hit** | `break` | `break` ✓ |
| **Reset on non-empty** | `consecutive_empty = 0` | `consecutive_empty = 0` ✓ |

### D3. L4 coverage audit gate

| | Old | New |
|---|---|---|
| **Enabled by** | `NQPR_LINK_COVERAGE_AUDIT=true` | `INVESTIGATION_L4_COVERAGE_AUDIT=true` (config constant) |
| **When applied** | Whenever `status in terminal_statuses` | Only when `iteration >= max_iterations - 2` ✗ F-054 |
| **Effect on failure** | Continue loop | `actions = [{"tool": "search_investment", "query": "...evidence quality..."}]` ✓ |
| **Verdict** | ✗ F-054 | Gate fires too late; early terminal status escapes audit |

### D4. Tool action dispatch — `_execute_actions`

| | Old `execute_tool` | New `_execute_actions` |
|---|---|---|
| **Supported tools** | All 10 RESEARCH_TOOLS | `_SUPPORTED_TOOLS = {"search_investment","search_bow","search_doc_type","search_all","search_web","read_pages","compute"}` (7 tools) ✗ F-053 |
| **Unsupported tool action** | Logged as "unknown tool" | Silently skipped (`if tool_name not in _SUPPORTED_TOOLS: return [], 0`) ✗ F-053 |
| **`submit_findings` action** | Returns `"FINDINGS_SUBMITTED"` sentinel | **Not in `_SUPPORTED_TOOLS`** — silently skipped ✗ F-043 |
| **`search_portfolio`** | `execute_tool("search_portfolio")` — calls `_idx.search_with_filter()` without inv_id filter | Not in `_SUPPORTED_TOOLS` — silently skipped ✗ F-053 |
| **Parallelism** | `asyncio.gather` (APPROVED-2) | `asyncio.gather` (APPROVED-2) ✓ |
| **Error per action** | Logged; continues | `except Exception: logger.warning()` — swallowed ✓ |
| **`search_bow`** | `_idx.search_with_filter(query, bow_id=bow_id)` | `search_investment.ainvoke({"query":query}, config=_bow_config)` where `_bow_config = {inv_id=None, bow_id=bow_id}` ✓ |
| **`search_doc_type`** | N/A in old tool set | NEW — maps to `search_investment` with `doc_type` filter ✓ (but not in INVESTIGATION_TOOLS @tool set) |
| **`read_pages` → `read_document`** | Reads page files directly | `read_document.ainvoke({page_start, page_end})` — applies 50-page cap ✗ F-040 |

### D5. Empty result from `search_web` (Tavily stub)

| | Old | New |
|---|---|---|
| **When Tavily not configured** | OpenAI Responses API — always attempted | Returns stub string `"[web search not configured...]"` with 130+ chars |
| **Effect on `new_chunks`** | Actual web results or error | Stub text appended as a chunk — **non-empty** |
| **Effect on `consecutive_empty`** | Counter increments if no results | Counter does **NOT** increment — stub counts as a result ✗ F-056 |

---

## E. Science Investigation Loop (core/science_investigator.py)

### E1. Main loop — `investigate_science_question` termination

| Termination condition | Old | New |
|---|---|---|
| **Max iterations** | `max_iterations=8` (`SCIENCE_MAX_ITERATIONS`) | `for iteration in range(max_iterations)` where `max_iterations=SCIENCE_MAX_ITERATIONS=8` ✓ |
| **Terminal statuses** | `{"evidence_gathered","insufficient_evidence","blocked"}` | `_SCIENCE_TERMINAL = frozenset({"evidence_gathered","insufficient_evidence","blocked"})` ✓ |
| **ASTA gate blocking** | `status=="evidence_gathered" and not asta_called_ever` → continue | `gate_blocked = output.status == "evidence_gathered" and not asta_called_ever` → not break ✓ |
| **Gate blocked + no actions** | Inject forced `search_asta` action | `if not actions and gate_blocked: actions = [{"tool":"search_asta","query":question}]` ✓ |
| **Gate blocked + has actions** | Continue with those actions | `if output.status in _SCIENCE_TERMINAL and not gate_blocked: break` → continues ✓ |
| **LLM error** | Propagates | `except Exception: logger.warning(); break` — swallowed ⚠ |

### E2. Empty-round consecutive counter

| | Old | New |
|---|---|---|
| **Threshold** | `CONSECUTIVE_EMPTY_THRESHOLD=3` | `CONSECUTIVE_EMPTY_THRESHOLD=3` ✓ |
| **Action on hit** | Force `status="insufficient_evidence"` | Forces `ScienceActionsOutput(status="insufficient_evidence", ...)` ✓ |
| **Break after force** | Yes | `break` ✓ |

### E3. ASTA soft cap

| | Old | New |
|---|---|---|
| **Cap value** | `asta_soft_cap=5` | `asta_soft_cap=ASTA_SOFT_CAP=5` ✓ |
| **When capped** | Skip ASTA actions | `if asta_actions and asta_calls >= asta_soft_cap: asta_called_ever = True` (marks as called) ✓ |
| **Effect on gate** | Gate satisfied by cap | `asta_called_ever = True` when capped ✓ |

### E4. Result `iterations` field

| | Old | New |
|---|---|---|
| **Records** | Actual loop exit iteration | `iterations=max_iterations` — always records configured maximum ✗ F-055 |

### E5. Non-ASTA tool dispatch delegation

| | Old | New |
|---|---|---|
| **Delegate to** | `claim_investigator._execute_actions` | `investigation._execute_actions` (different module, same logic) ✓ |
| **Tools covered** | `search_bow, search_science, search_policy, search_all, search_web, read_pages, compute` | `_SUPPORTED_TOOLS = {"search_investment","search_bow","search_doc_type","search_all","search_web","read_pages","compute"}` — `search_science` and `search_policy` are NOT in `_SUPPORTED_TOOLS` ✗ |
| **Verdict** | ✗ | `search_science` and `search_policy` actions silently skipped (no tool maps to them) — same as F-053 |

---

## F. Collection-API Search Retry / Empty-Result Handling

### F1. `build_evidence_pack` — 4 retrieval strategies

| Strategy | Condition | Old | New |
|---|---|---|---|
| S1 | Always run | 10 LLM queries via `search_with_filter` | 10 LLM queries via `_EmbeddingIndexAdapter.search_with_filter` ✓ |
| S2 | Always run | 3/4 hardcoded queries | 4 hardcoded queries (count discrepancy minor) ⚠ |
| S3 | Always run | 5 strategy queries | 5 strategy queries ✓ |
| S4 | Always run | Rerank via `hybrid_search` | Rerank via adapter ✓ |
| **Min strategy chunks** | Post-processing | `_MIN_STRATEGY_CHUNKS = 20` guarantee | 20-chunk minimum in new rubric_evaluator ✓ |
| **Retry on empty** | None — all strategies run regardless | None — same ✓ |

### F2. Investigation loop search empty-result handling

| Condition | Old | New |
|---|---|---|
| **Empty search result** | Chunk not appended → `consecutive_empty++` | Same via `_dedup_chunks` → `not new_chunks` → `consecutive_empty++` ✓ |
| **3 empty rounds** | `break` | `break` ✓ |
| **Search failure exception** | Logged; action skipped | `except Exception: logger.warning()` — swallowed; `chunks=[]`, `web_count=0` ✓ |

### F3. Stub web search counted as non-empty (Tavily missing)

As documented in D5 — when Tavily is not configured, the stub string is appended as a chunk. Empty-round counter does not increment. ✗ F-056.

---

## G. Summary — All Branches/Loops with Verdict

| ID | Location | Type | Old condition | New condition | Verdict |
|---|---|---|---|---|---|
| A1 | `route_after_precheck` | Branch | precheck passed | `state.get("precheck_passed")` | ✓ |
| A2 | `route_after_research_plan_review` | Branch | human approval | **ABSENT** | ✗ F-058 |
| A3 | `route_after_report_approval` | Branch | human approval | **ABSENT** | ✗ F-058 |
| A4 | Human interrupt nodes | Branch | 4 nodes | 1 node only (`analyze`) | ✗ F-058 |
| B2 | `compute_scopes` focus_bows | Branch | `bow_id not in focus_bows` | same | ✓ |
| B3 | `compute_scopes` skip phantom | Branch | `< 200 chunks` | `< 5 chunks` (default) | ✗ F-024 |
| B4 | `compute_scopes` split | Branch | `> 12K or > 8 invs or is_catch_all` | `> 12K or > 8 invs` (no catch-all) | ⚠ F-024 |
| B5 | `dispatch_investment_narratives` | Conditional edge | SHA256 cache + per-investment | set-membership + per-investment | ⚠ F-025 |
| B6 | `dispatch_scope_syntheses` | Conditional edge | SHA256 cache + per-scope | set-membership + per-scope | ✓ |
| B7 | `dispatch_investment_reports` | Conditional edge | N/A | `"investment_report" in s` | ✓ (new phase) |
| B8 | `dispatch_scope_sections` | Conditional edge | N/A | `"section_draft" in s` | ✓ (new phase) |
| B11 | `assemble_report` retry | Loop | `_ASSEMBLY_MAX_RETRIES=5` | **no retry** | ✗ F-029 |
| C1 | `dispatch_rubric_evaluation` | Conditional edge | one per investment | one per scope (primary inv only) | ✗ F-049 |
| C2 | `dispatch_bow_enrichment` | Conditional edge | one per scope | one per scope | ✓ |
| C2b | `enrich_bow_context_worker` | Early exit | N/A | no `web_search_fn` → `_empty_bow` | ✗ F-059 |
| C3 | `_route_link_investigations` | Conditional edge | one per link | one per link | ✓ |
| C4 | `_route_science_investigations` | Conditional edge | one per flagged assumption | one per ALL assumptions | ✗ F-052 |
| C5 | `dispatch_decision_projections` | Conditional edge | one per scope | one per scope | ✓ |
| C6 | synthesis ∥ science join | Parallel fan-in | both branches → necessity_check | both edges → necessity_check | ✓ |
| D1a | `run_investigation` max-iter | Loop limit | `max_iterations=20` default | `MAX_INVESTIGATION_ITERATIONS=40` | ✗ (doubled) |
| D1b | `run_investigation` terminal | Loop exit | `{"on_track","deviations_found","critical_risk","insufficient_evidence"}` | `{"answered","not_answerable","unresolved_conflict"}` | ✗ F-051 |
| D2 | Investigation consecutive empty | Loop exit | 3 empty rounds | 3 empty rounds | ✓ |
| D3 | L4 coverage audit gate | Branch | any terminal status | only `iteration >= max_iterations - 2` | ✗ F-054 |
| D4 | `_execute_actions` tool dispatch | Branch | all 10 RESEARCH_TOOLS | 7 tools; others silently skipped | ✗ F-053 |
| D5 | Tavily stub ≠ empty | Branch | real results or empty | stub = non-empty | ✗ F-056 |
| E1 | `investigate_science_question` terminal | Loop exit | `{"evidence_gathered","insufficient_evidence","blocked"}` | same | ✓ |
| E2 | Science consecutive empty | Loop exit | 3 empty rounds → force insufficient | same | ✓ |
| E3 | ASTA soft cap | Branch | `asta_calls >= 5` | same | ✓ |
| E4 | Science `iterations` field | Result | actual iteration count | `max_iterations` always | ✗ F-055 |
| E5 | `search_science`/`search_policy` in science | Branch | dispatched to `_execute_actions` | not in `_SUPPORTED_TOOLS` → silently skipped | ✗ F-053 |
| F1 | `build_evidence_pack` strategies | Loop (4 strategies) | all 4 always run | all 4 always run | ✓ |
| F3 | Tavily stub counts as evidence | Branch | N/A | stub appended as chunk | ✗ F-056 |
