# CODEBASE AUDIT — `src/qpr`

Generated from source-read of every Python file in `src/qpr/` (excluding `_archive/`).

---

## Table of Contents

1. [Entry Point](#1-entry-point)
2. [Stage 1 — Collect](#2-stage-1--collect)
3. [Stage 2 — Analyze](#3-stage-2--analyze)
4. [Stage 3 — Prepare-Research](#4-stage-3--prepare-research)
5. [Stage 4 — Research](#5-stage-4--research)
6. [Stage 5 — Finalize](#6-stage-5--finalize)
7. [Report Assembly](#7-report-assembly)
8. [Evidence Audit](#8-evidence-audit)
9. [Gold-Standard Verifier (gs_verifier)](#9-gold-standard-verifier-gs_verifier)
10. [Diagnostics](#10-diagnostics)
11. [LLM Infrastructure & Shared Utilities](#11-llm-infrastructure--shared-utilities)
12. [Search Backends](#12-search-backends)
13. [Dependency Graph](#13-dependency-graph)
14. [CLI Commands → Pipeline Stages](#14-cli-commands--pipeline-stages)
15. [Data Passed Between Stages](#15-data-passed-between-stages)

---

## 1. Entry Point

### `cli.py`

| Field | Detail |
|---|---|
| **What it does** | Unified CLI entry point for the NQPR pipeline. Registers 8 subcommands, loads `.env`, initialises Phoenix/OTEL tracing, delegates to handler functions. |
| **Inputs** | `argv` (CLI args). Each subcommand accepts typed flags (see §14). |
| **Outputs** | Calls handler functions; no direct file writes except via delegated modules. |
| **External deps** | `dotenv`, `src.tracing.bootstrap.init_tracing` (optional); `argparse`, `logging` (stdlib). |
| **How invoked** | `python -m src.qpr <subcommand>` or `nqpr <subcommand>`. |
| **State between calls** | None. |

### `__main__.py`

| Field | Detail |
|---|---|
| **What it does** | Package runner: calls `cli.main()` so `python -m src.qpr` works. |
| **How invoked** | `python -m src.qpr`. |

---

## 2. Stage 1 — Collect

### `collection_prep.py`

| Field | Detail |
|---|---|
| **What it does** | Orchestrates the 8-step ETL pipeline for one program: file registration → scoring parse → BOW mapping → investment intelligence → multimodal ingest → embedding index → chunk enrichment → validation. |
| **Inputs** | `collect_program(program, data_root, *, output_dir, scoring_pdfs_dir, invest_base, skip_ingestion, skip_embedding, force, force_steps, sync_vision, adaptive_vision, collection_base_dir, intelligence_workers, ingest_workers) → Path` |
| **Outputs** | Returns `Path` to `{program}-ingested/`. Writes: `doc_list.json`, `investment_scoring.json`, `bow_investment_map.json`, `investment_intelligence.json`, `chunks.json`, `chunks.sqlite`, `vectors.npy`, `tfidf_matrix.npz`, `tfidf_vocab.json`. |
| **External deps** | `fitz` (PyMuPDF), `numpy`, `sklearn`; inherits env vars from sub-modules. |
| **How invoked** | `python -m src.qpr collect --program <P> --data-root <D>` → `_cmd_collect` → `collect_program(...)`. |
| **State between calls** | Module-level logger only. |

### `collection_setup.py`

| Field | Detail |
|---|---|
| **What it does** | Creates an empty collection stub directory; optionally renders PDF pages to PNG for vision extraction. |
| **Inputs** | `ensure_collection(program, data_root, *, skip_ingestion, force, collection_base_dir) → (str, CollectionManager)`; `ensure_pages(collection_dir, *, model, dpi, max_workers, ...) → Path|None` |
| **Outputs** | Writes collection stub on disk; renders per-page PNGs under `pages/`. |
| **External deps** | `fitz` (PyMuPDF); `vision_extraction`, `document_index`, `program_config`. |
| **How invoked** | Called from `collection_prep.collect_program` at step 1. |
| **State between calls** | None. |

### `collection_health.py`

| Field | Detail |
|---|---|
| **What it does** | Validates and optionally repairs all collection artifacts (scoring JSON, BOW map, doc_list, pages, embedding index, TF-IDF, intelligence, BOW accuracy). |
| **Inputs** | `check_collection(program, output_dir, invest_dir, scoring_dir, *, with_llm) → HealthReport`; `repair_collection(report, *, with_llm) → int` |
| **Outputs** | Returns `HealthReport`; `print_report()` writes summary to stdout; repair functions may rewrite artifacts. |
| **External deps** | `fitz`, `numpy`; `llm_utils.call_llm` (only on `--with-llm` path). |
| **How invoked** | `python -m src.qpr collect --check|--repair`; also called from `collection_prep` after each major step. |
| **State between calls** | None. |

### `collection_loader.py`

| Field | Detail |
|---|---|
| **What it does** | Loads a fully prepared collection from disk (embedding index, doc catalog, scoring, BOW hierarchy, intelligence) into a ready-to-query `CollectionTools` + `CollectionContext` pair. |
| **Inputs** | `load_collection(collection_name, base_dir, *, aux_collections) → (CollectionTools, CollectionContext)` |
| **Outputs** | Returns `(CollectionTools, CollectionContext)`; no files written (read-only). |
| **External deps** | `numpy`, `sqlite3`, `sklearn`; env vars `NQPR_SEARCH_BACKEND`, `QDRANT_*`, `AZURE_SEARCH_*`. |
| **How invoked** | Imported and called before any analysis (from `_cmd_analyze`, `_cmd_investigate`, etc.). |
| **State between calls** | None (returns fresh objects each call). |

### `collection_builder.py`

| Field | Detail |
|---|---|
| **What it does** | Fast, LLM-free file registration — walks source directories, adds files to a `CollectionManager`, returns the collection ID. |
| **Inputs** | `build_collection_from_directories(source_dirs, collection_name, description, content_mode, base_dir) → str` |
| **Outputs** | Writes collection manifest via `CollectionManager`. |
| **External deps** | `dotenv`; `_inv_id.extract_inv_id`. |
| **How invoked** | Imported by bootstrap scripts; not a top-level CLI command. |
| **State between calls** | None. |

### `collection_api.py`

| Field | Detail |
|---|---|
| **What it does** | Unified tool API for NQPR agents — exposes BOW navigation, document reading, section lookup, semantic/hybrid search, scoring retrieval, and image access. |
| **Inputs** | Constructor: `CollectionTools(ctx: CollectionContext)`. Methods: `search(query, *, top_k, bow_filter, inv_filter)`, `read_section(file_id, section_id)`, `read_pages(file_id, pages)`, `get_scoring(inv_id)`, etc. |
| **Outputs** | Methods return dicts, lists, strings, or base64 image bytes. No files written. |
| **External deps** | `openai` (dense search fallback), `sklearn`, `numpy`. |
| **How invoked** | Instantiated by `load_collection`; methods called by agent investigation loops. |
| **State between calls** | Instance: `_strategy_chunks`, `_doc_index`, `_chunk_embeddings`, `_openai_client`, `_lock`, `_embedding_index`, `_aux_indexes`, three lookup dicts — all cached on the instance. |

### `program_config.py`

| Field | Detail |
|---|---|
| **What it does** | Maps program names (HIV, TB, Malaria, MNCNH, …) to canonical document root paths. |
| **Inputs** | None — loaded at import via `src.config.settings.settings`. |
| **Outputs** | Exports `PROGRAM_CONFIG: dict` at module level. |
| **External deps** | `src.config.settings`. |
| **How invoked** | `from src.qpr.program_config import PROGRAM_CONFIG`. |
| **State between calls** | `PROGRAM_CONFIG` module-level dict, populated once at import. |

### `multimodal_ingest.py`

| Field | Detail |
|---|---|
| **What it does** | Unified multimodal ingest — extracts text + vision from PDFs/DOCX/PPTX/XLSX; applies adaptive vision (PyMuPDF first, GPT-4o for table/figure pages); produces `IngestedDocument` with chunks and page texts. |
| **Inputs** | `ingest_document(filepath, *, file_id, collection, doc_type, inv_id, bow_id, vision_model, output_dir, call_llm, force, ...) → IngestedDocument`; `ingest_collection(docs, output_dir, *, ...) → list[IngestedDocument]` |
| **Outputs** | `IngestedDocument` dataclass; per-document JSON caches under `output_dir`. |
| **External deps** | `fitz`, `python-docx`, `python-pptx`, `openpyxl`, `openai`; env var `NQPR_INCLUDE_SKIP`. |
| **How invoked** | `ingest_collection(...)` called from `collection_prep` step 5. |
| **State between calls** | `_FITZ_FALLBACK_LOCK` (threading.Lock), `PIPELINE_VERSIONS` dict, `_TRIAGE_MODEL = "gpt-4o-mini"` (module-level). |

### `batch_vision.py`

| Field | Detail |
|---|---|
| **What it does** | OpenAI Batch API driver for bulk page vision extraction — renders pages to PNG, submits batches (≤180 MB), polls, and collects per-page markdown results with content-filter sidecars. |
| **Inputs** | `render_all_pages`, `submit_vision_batch`, `poll_batch`, `collect_results`; CLI subcommands: `submit`, `poll`, `collect`, `status`. |
| **Outputs** | Writes `batch_input_NNN.jsonl`, `batch_output.jsonl`, `batch_info.json`, per-page `p{N}.txt` / `p{N}.blocked.json`. |
| **External deps** | `openai`, `fitz`. |
| **How invoked** | Imported by `multimodal_ingest` (`use_batch=True`); also `python -m src.qpr.batch_vision submit|poll|collect|status`. |
| **State between calls** | None. |

### `document_index.py`

| Field | Detail |
|---|---|
| **What it does** | Segments document pages into labeled sections via LLM, generates a summary, and persists a `DocumentIndex` JSON — the TOC/section-navigation backbone for all downstream stages. |
| **Inputs** | `build_document_index(pages_dir, paper_id, call_llm, *, collection, doc_type, inv_id, bow_id, model, force) → DocumentIndex`; `index_all_documents(...)`, `build_document_catalog(...)`, `load_document_catalog(...)` |
| **Outputs** | Writes `{pages_dir}/{paper_id}/index.json`; `build_document_catalog` writes catalog JSON. Returns `DocumentIndex` dataclass. |
| **External deps** | LLM callable; `model_defaults.ANALYSIS_MODEL`. |
| **How invoked** | Called from `collection_prep` step 3; `build_document_catalog` from `collection_setup.ensure_pages`. |
| **State between calls** | `_SECTION_SYSTEM` prompt string (module-level). |

### `vision_extraction.py`

| Field | Detail |
|---|---|
| **What it does** | Single-page and full-document vision extraction via multimodal LLM — converts PDF pages to PNG, sends to GPT-4o vision API, returns structured markdown with table/figure detection flags; retries on rate limits. |
| **Inputs** | `extract_page_vision(img_bytes, page_num, *, model, api_key) → PageExtraction`; `extract_document_vision(filepath, ...) → DocumentExtraction`; `extract_collection_vision(...)` |
| **Outputs** | `PageExtraction` / `DocumentExtraction` dataclasses; may write page PNG files. |
| **External deps** | `openai`, `fitz`. |
| **How invoked** | `extract_page_vision` called per-page from `multimodal_ingest` adaptive path. |
| **State between calls** | `_RETRY_DELAYS = [2, 8, 30]` (module-level). |

### `investment_intelligence.py`

| Field | Detail |
|---|---|
| **What it does** | Five-tier LLM pipeline per investment (raw extraction → cheap cleaning → version diffing → synthesis/decisions → timeline narrative); also classifies strategy docs in batches; applies decisions to doc lists and enriches chunks. |
| **Inputs** | `analyze_investment(inv_id, invest_dir, scoring_data, bow_list, cache_dir) → InvestmentIntelligence`; `analyze_all_investments(program, invest_base, output_dir, scoring, bow_map, max_workers)`; `analyze_strategy_docs(...) → StrategyIntelligence` |
| **Outputs** | `InvestmentIntelligence` / `StrategyIntelligence` dataclasses; caches to `intelligence/{inv_id}.json`. `apply_intelligence_to_doc_list` and `enrich_chunks_with_intelligence` mutate in-place. |
| **External deps** | `fitz`, `python-docx`, `python-pptx`, `openpyxl`, `llm_utils`, `model_defaults.SYNTHESIS_MODEL`. |
| **How invoked** | `analyze_all_investments(...)` called from `collection_prep` step 4. |
| **State between calls** | `CONTENT_TAG_VOCABULARY`, `STRATEGY_ROLE_VOCABULARY`, `_STRATEGY_BATCH_SIZE=100`, `_STRATEGY_BATCH_WORKERS=8`, `CHUNK_INTELLIGENCE_FIELDS` (module-level). |

### `investment_scoring.py`

| Field | Detail |
|---|---|
| **What it does** | Parses QPR scoring PDFs into structured `InvestmentDetail` objects with financial fields, score history, and maturity model; provides performance assessment, stage-aware staking, and LLM-based classification. |
| **Inputs** | `parse_scoring_detail_pages(fulltext) → dict[str, InvestmentDetail]`; `load_enriched_investments(collection_path, ...) → dict`; `classify_investments_llm(investments, call_llm, *, model) → dict` |
| **Outputs** | `InvestmentDetail` dataclasses; no files written directly. |
| **External deps** | LLM callable; `model_defaults.ANALYSIS_MODEL`, `FAST_MODEL`. |
| **How invoked** | `parse_scoring_detail_pages` called from `collection_prep` step 2; `load_enriched_investments` from `collection_loader`. |
| **State between calls** | `MATURITY_BENCHMARKS`, `MATURITY_*` tier constants (module-level). |

### `investment_docs.py`

| Field | Detail |
|---|---|
| **What it does** | Three-layer document architecture (`DocumentMap`, `InvestmentDocStore`, `ProgramInvestmentCollection`) for document classification, text extraction, and budget analysis. Largely superseded by `investment_intelligence.py` but retained for compatibility. |
| **Inputs** | `InvestmentDocStore(inv_dir, inv_id, *, cache_dir)` → `.ingest(call_llm)`, `.get_key_documents()`, `.budget_summary(call_llm)`, `.search(query)` |
| **Outputs** | List of ingested document dicts; cache files under `cache_dir`. |
| **External deps** | `python-docx`, `python-pptx`, `fitz`, `openpyxl`; LLM callable. |
| **How invoked** | Imported by older analysis paths; `ProgramInvestmentCollection` used for compatibility. |
| **State between calls** | Instance: `docs`, `inv_dir`, `inv_id`, `cache_dir`, `doc_map`, `stores`. |

### `investment_timeline.py`

| Field | Detail |
|---|---|
| **What it does** | Constructs chronological `InvestmentTimeline` and `ScopeTimeline` objects from doc_list + scoring + intelligence data; optionally generates LLM narrative summaries with SHA256-based cache invalidation. |
| **Inputs** | `build_investment_timeline(inv_id, doc_entries, scoring_data, pages_dir, intelligence) → InvestmentTimeline`; `build_scope_timeline(scope, doc_list, scoring, pages_dir, intelligence) → ScopeTimeline`; `build_timeline_narratives(scope_timeline, call_llm, *, model)` |
| **Outputs** | `InvestmentTimeline` / `ScopeTimeline` dataclasses; `save_narratives` writes JSON to disk. |
| **External deps** | LLM callable; `model_defaults.SYNTHESIS_MODEL`. |
| **How invoked** | Called from analysis modules after intelligence analysis completes. |
| **State between calls** | None. |

### `precheck.py`

| Field | Detail |
|---|---|
| **What it does** | Runs fast pre-flight integrity checks on a prepared collection (chunk sizes, config alignment, orphan pages, BOW coverage, disk space, API key availability) before an expensive agent run. |
| **Inputs** | `run_checks(ingested_dir, *, invest_dir, focus_bows, min_free_gb) → PrecheckResult`; `format_report(precheck) → str` |
| **Outputs** | Returns `PrecheckResult`; prints formatted table; exits non-zero on FAIL (unless `--force`). |
| **External deps** | `numpy`, `sqlite3`, `shutil`; env vars `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`. |
| **How invoked** | `python -m src.qpr precheck --program <P>` → `_cmd_precheck`. |
| **State between calls** | None. |

### `doc_types.py`

| Field | Detail |
|---|---|
| **What it does** | Canonical document type normalization — maps raw upstream type strings to a fixed 31-member `CANONICAL_DOC_TYPES` frozenset, with filename-pattern overrides. |
| **Inputs** | `canonicalize_doc_type(doc_type, *, filename, folder) → str` |
| **Outputs** | Returns canonical doc type string; pure function, no files written. |
| **External deps** | `re` (stdlib only). |
| **How invoked** | Imported throughout the pipeline wherever raw doc_type strings need normalisation. |
| **State between calls** | `CANONICAL_DOC_TYPES` frozenset, `_IP_REPORT_RE` compiled regex (module-level). |

### `doc_expectations.py`

| Field | Detail |
|---|---|
| **What it does** | Document type expectations registry — maps each canonical doc type to extraction guidance, completeness checklists, and risk scores; provides two-tier screening (rule-based Tier 0, LLM Tier 1) for intelligence triage. |
| **Inputs** | `tier0_screen(files, *, bow_inv_ids, agent_type) → (list, list)`; `tier1_batch_screen(files, bow_context, call_llm, ...) → list[dict]`; `get_extraction_guidance(doc_type, agent_type) → str` |
| **Outputs** | Filtered file lists and screening decision dicts; no files written. |
| **External deps** | LLM callable for `tier1_batch_screen`. |
| **How invoked** | Called from `investment_intelligence` triage step. |
| **State between calls** | `DOC_EXPECTATIONS` dict (module-level constant). |

### `semantic_chunker.py`

| Field | Detail |
|---|---|
| **What it does** | Section-aware semantic chunking — splits document page texts at sentence boundaries into overlapping `SemanticChunk` objects with full provenance metadata for embedding and retrieval. |
| **Inputs** | `chunk_document(page_texts, sections, *, file_id, filename, collection, doc_type, inv_id, bow_id, summary, doc_date, target_chunk_size=1000, max_chunk_size=2000, ...) → list[SemanticChunk]` |
| **Outputs** | List of `SemanticChunk` dataclasses; no files written. |
| **External deps** | None (pure Python, `re` only). |
| **How invoked** | Called from `multimodal_ingest` after page text extraction. |
| **State between calls** | `_SENTENCE_BOUNDARY`, `_VISION_PREAMBLES`, `_PAGE_NUMBER_RE` compiled patterns (module-level). |

### `asset_registry.py`

| Field | Detail |
|---|---|
| **What it does** | Builds a run-time asset name registry from intelligence prose using regex; enables cross-corpus retrieval fanout by matching query mentions to registered asset names. |
| **Inputs** | `build_asset_registry(team, *, investment_intelligence, ingested_dir, seed_dir) → AssetRegistry`; `write_run_assets(registry, run_dir) → Path`; `cross_corpus_fanout(query, *, registry, aux_indexes, top_k) → list[SearchResult]` |
| **Outputs** | `AssetRegistry` dataclass; writes `assets.json` to `run_dir`. |
| **External deps** | `yaml` (optional); reads optional seed file `config/asset_seeds/{team}.yml`. |
| **How invoked** | `build_asset_registry` called from `collection_loader` when `aux_collections` is provided. |
| **State between calls** | `_MENTION_PATTERN_CACHE` dict (module-level, survives process lifetime). |

### `_inv_id.py`

| Field | Detail |
|---|---|
| **What it does** | Extracts an investment ID from a filesystem path by returning the first path component matching `INV-\d+`. |
| **Inputs** | `extract_inv_id(rel_path: Path | str) → str` |
| **Outputs** | Matching component string or empty string; pure function. |
| **External deps** | None. |
| **How invoked** | Called from `collection_builder` and `collection_prep.build_investment_doc_list`. |
| **State between calls** | `_INV_RE` compiled regex (module-level). |

---

## 3. Stage 2 — Analyze

### `research_analyst_agent.py`

| Field | Detail |
|---|---|
| **What it does** | Top-level V4 pipeline orchestrator running 6 sequential phases: orientation → scope/thread design → timelines + narratives → causal pipeline (per-thread parallel) → cross-cutting analysis → quality assessment + report assembly. |
| **Inputs** | `run_research_analyst(tools, call_llm, *, orientation_model, research_model, synthesis_model, focus, focus_bows, cache_dir) → AnalystReport` |
| **Outputs** | Returns `AnalystReport`. Writes: `cache_dir/final_report.md`, `analyst_report.json`, `scope_outputs.json`, `numerical_provenance.json`, `verification_sources.json`, `excerpts.csv`, `allocation_verification.json`, `numerical_verification.json`. |
| **External deps** | All sub-modules below; no direct API calls. |
| **How invoked** | `python -m src.qpr analyze --collection <C>` → `_cmd_analyze` → `run_research_analyst(...)`. |
| **State between calls** | None; all state in returned `AnalystReport` and files in `cache_dir`. |

### `causal_pipeline.py`

| Field | Detail |
|---|---|
| **What it does** | Core Phase 3 orchestrator — 8 sub-stages (3.1–3.8): causal model extraction → consequence forecasts → BOW context → per-link tool-calling investigation (parallel) → synthesis → critique → gap analysis → science validation → necessity check → decision projection. |
| **Inputs** | `run_causal_pipeline(scopes, scope_timelines, tools, call_llm, context, cache_dir, research_model, synthesis_model) → list[ScopeOutput]` |
| **Outputs** | Returns `list[ScopeOutput]`; writes per-scope JSON checkpoints to `cache_dir`. |
| **External deps** | `causal_model`, `rubric_evaluator`, `investigation_loop`, `claim_investigator`, `science_investigator`, `decision_projection`. |
| **How invoked** | Called from `research_analyst_agent._phase4_causal()`. |
| **State between calls** | Constants: `_MAX_LINK_WORKERS=16`, `_MAX_INVESTIGATION_ITERATIONS=40` (module-level). |

### `causal_model.py`

| Field | Detail |
|---|---|
| **What it does** | Extracts theory-of-change causal models (FUNDING→ACTIVITIES→OUTPUTS→OUTCOMES→IMPACT) from proposal chunks via LLM; ranks assumptions by consequence×uncertainty; forecasts quantified consequences; converts to `InvestigationClaim` objects. |
| **Inputs** | `extract_causal_model(inv_id, proposal_chunks, call_llm, *, model, program, bow_model) → CausalModel`; `rank_assumptions(model) → list[ScoredAssumption]`; `forecast_consequences(model, facts_by_inv, call_llm, ...) → list[ScoredAssumption]`; `make_investigation_claims(ranked, inv_id) → list[InvestigationClaim]` |
| **Outputs** | `CausalModel`, `list[ScoredAssumption]`, `list[InvestigationClaim]`; no files written. `forecast_consequences` mutates `model.dollars_by_link` and `model.months_by_link` in-place. |
| **External deps** | `anthropic` (direct client for biosafety-defang path); env var `ANTHROPIC_API_KEY`. |
| **How invoked** | Called from `causal_pipeline` stages 3.1 and 3.2. |
| **State between calls** | None. |

### `investigation_loop.py`

| Field | Detail |
|---|---|
| **What it does** | Provider-neutral tool-calling investigation loop (OpenAI GPT-5.x and Anthropic); the LLM autonomously calls 9 tools to investigate a hypothesis; controlled by three env-var levers (reasoning effort, coverage audit, auditor). |
| **Inputs** | `run_investigation(claim, tools, call_llm, inv_id, bow_id, *, model, max_iterations) → InvestigationResult`; tool set: `search_investment`, `search_portfolio`, `search_web`, `read_document`, `compute`, `submit_findings`, `list_documents`, `read_document_summary`, `get_document_structure`, `read_section` |
| **Outputs** | Returns `InvestigationResult` dataclass; no files written. |
| **External deps** | `openai` (Responses API), `anthropic` (Messages API); env vars `NQPR_TOPK_CAP`, `NQPR_LINK_REASONING_EFFORT`, `NQPR_LINK_COVERAGE_AUDIT`, `NQPR_LINK_AUDITOR`. |
| **How invoked** | Called from `causal_pipeline` stage 3.4 per causal link (parallel). |
| **State between calls** | `accumulated_refs` list (local per call); no module-level mutable globals. |

### `thread_sub_agent.py`

| Field | Detail |
|---|---|
| **What it does** | Five-round iterative investigation sub-agent for a `ResearchThread`; also exposes `_call_with_web_search()` utility used by other modules. Being replaced by `investigation_loop`. |
| **Inputs** | `run_thread_sub_agent(thread, tools, call_llm, *, model, num_rounds, cache_dir) → ResearchThread`; `_call_with_web_search(prompt, *, model, context) → str` |
| **Outputs** | Returns mutated `ResearchThread`; writes `cache_dir/thread_{id}_r{n}.json` checkpoints. |
| **External deps** | `openai` (Responses API web search, model `gpt-5.4`). |
| **How invoked** | Called from `research_analyst_agent` phase 3; `_call_with_web_search` imported by other modules. |
| **State between calls** | Cache file at `cache_dir/thread_{id}_r{n}.json` persists across restarts (resume on cache hit). |

### `rubric_evaluator.py`

| Field | Detail |
|---|---|
| **What it does** | Builds `InvestmentEvidencePack` for one investment using 4-strategy retrieval (LLM-generated queries, hardcoded fallback, doc-type-specific, strategy doc queries); reranks combined pool; computes deterministic local scores; detects fact contradictions. |
| **Inputs** | `build_evidence_pack(inv_id, timeline, tools, *, top_k, call_llm, model) → InvestmentEvidencePack` |
| **Outputs** | Returns `InvestmentEvidencePack`; no files written. |
| **External deps** | `call_llm`, `InvestmentFacts.from_scoring_and_timeline`, `llm_utils.trace_llm_context`. |
| **How invoked** | Called from `causal_pipeline` stage 3.1 (parallel, one call per investment). |
| **State between calls** | `_EXPECTED_DOC_TYPES` set (module-level). |

### `claim_investigator.py`

| Field | Detail |
|---|---|
| **What it does** | Iterative, direction-neutral claim investigator (9 tools, up to `max_iterations=12`); dispatches tool actions in parallel via `ThreadPoolExecutor(8)`; stops on terminal status or 3 consecutive empty rounds. |
| **Inputs** | `investigate_claim(claim, evidence_packs, tools, call_llm, *, model, max_iterations=12) → ClaimResult`; `run_claim_investigation(claims, evidence_packs, tools, call_llm, ...) → list[ClaimResult]` |
| **Outputs** | Returns `ClaimResult` / `list[ClaimResult]`; no files written. |
| **External deps** | `openai`/`anthropic` via `call_llm`; `contextvars.ContextVar` (`_TRACE_CTX`). |
| **How invoked** | `run_claim_investigation` called from `causal_pipeline` stage 3.4; `_execute_actions` imported by `science_investigator`. |
| **State between calls** | `_TRACE_CTX` (module-level `ContextVar`, reset per claim). |

### `science_investigator.py`

| Field | Detail |
|---|---|
| **What it does** | Iterative science assumption investigator (Phase 3.5d Step 2); uses 8 tools including `search_asta` (Semantic Scholar 225M+ papers); enforces hard requirements before terminal states (must call ASTA, must attempt both confirming and disconfirming searches). |
| **Inputs** | `investigate_science_question(question, question_index, tools, asta_client, call_llm, *, model, inv_id, org, title, bow_id, scope_id, max_iterations, ...) → ScienceInvestigationResult` |
| **Outputs** | Returns `ScienceInvestigationResult`; no files written. |
| **External deps** | `AstaClient` (Semantic Scholar); `claim_investigator._execute_actions`. |
| **How invoked** | Called from `causal_pipeline` stage 3.5d per science assumption (parallel). |
| **State between calls** | None (all local per call). |

### `decision_projection.py`

| Field | Detail |
|---|---|
| **What it does** | Phase 3.8 — aggregates per-link `LinkAssessment.leadership_options` into structured `Decision` objects per scope; applies evidence-weighted gating, ranks by composite score, caps at 3 per INV and 8 per scope. |
| **Inputs** | `run_phase38(scope_outputs, call_llm, *, model, max_workers) → int` (mutates `ScopeOutput.decisions` in place) |
| **Outputs** | Returns int (total decisions); mutates passed-in `ScopeOutput` objects. |
| **External deps** | `call_llm`; `evidence_model._THIN_EVIDENCE_DECISION_TYPES`; `concurrent.futures`. |
| **How invoked** | Called from `research_analyst_agent` after Phase 3. |
| **State between calls** | `DECISION_TYPE_VOCABULARY` (15 types), module-level (immutable). |

### `cluster_identification.py`

| Field | Detail |
|---|---|
| **What it does** | Identifies 6–12 thematic investment clusters for the exec summary via deterministic shortlisting → single LLM grouping call → `ThematicCluster` records; caches to `threads/clusters.json`. |
| **Inputs** | `identify_clusters(reports, investments_by_id, cross_cutting_findings, bow_nodes, *, call_llm, model, run_dir, use_cache=True) → list[ThematicCluster]`; `load_clusters(run_dir) → list[ThematicCluster]` |
| **Outputs** | Returns `list[ThematicCluster]`; writes/reads `run_dir/threads/clusters.json` cache. |
| **External deps** | `call_llm`; no other 3rd-party libraries. |
| **How invoked** | Called from `research_analyst_agent` cross-cutting phase; `load_clusters` from report assembler. |
| **State between calls** | `clusters.json` disk cache. |

### `numerical_analyst.py`

| Field | Detail |
|---|---|
| **What it does** | Deterministic computation via OpenAI `code_interpreter` tool (sandboxed Python); provides entry points for pre-computed metrics, ad-hoc questions, and scenario analysis; coerces non-OpenAI models to `ANALYSIS_MODEL`. |
| **Inputs** | `compute_metrics(facts, model) → QuantitativeResult`; `analyze_question(facts, question, ...) → QuantitativeResult`; `compute_scope_metrics(packs, model) → dict[str, QuantitativeResult]` |
| **Outputs** | Returns `QuantitativeResult` dataclass; appends JSONL to `LLM_TRACE_FILE` (if set). |
| **External deps** | `openai` (Responses API, `code_interpreter`; OpenAI-only); env var `LLM_TRACE_FILE`. |
| **How invoked** | `compute_scope_metrics` and `_call_code_interpreter` called from `research_analyst_agent`. |
| **State between calls** | None. |

### `numerical_verifier.py`

| Field | Detail |
|---|---|
| **What it does** | Post-generation numerical verification audit — extracts all numerical claims (5 pattern types), runs 5-tier verification (provenance → scoring → aggregation → investigation trace → chunk matching), produces `VerificationScorecard`. |
| **Inputs** | `verify_report(report_path, collection_name, *, base_dir, call_llm, output_path) → VerificationScorecard` |
| **Outputs** | Returns `VerificationScorecard`; optionally writes JSON scorecard. |
| **External deps** | `call_llm` (optional, for LLM derivation passes). |
| **How invoked** | `python -m src.qpr.numerical_verifier` or imported; called from `research_analyst_agent` at pipeline end. |
| **State between calls** | None. |

### `allocation_verifier.py`

| Field | Detail |
|---|---|
| **What it does** | Verifies dollar allocations cited in reports against investment scoring data; auto-rewrites mismatches with replacement figures or parenthetical qualifiers. |
| **Inputs** | `verify_and_rewrite(report_path, scoring_path, output_path, scorecard_path, auto_rewrite, label) → AllocationScorecard` |
| **Outputs** | Writes rewritten report and scorecard JSON. |
| **External deps** | `numerical_verifier.extract_figures`. |
| **How invoked** | `python -m src.qpr.allocation_verifier` or imported; called from `research_analyst_agent`. |
| **State between calls** | Constants `_MATCH_TOLERANCE_M=0.5`, `_QUALIFIER_CONTEXT_WINDOW=300` (module-level). |

### `evaluation_comparison.py`

| Field | Detail |
|---|---|
| **What it does** | Pure-function aggregation over Phase 3 AI-vs-team verdicts — per-investment rows, per-BoW rollups, portfolio divergence dashboard, partner relationship matrix, `build_risk_disagreement_ranking`. |
| **Inputs** | `aggregate_bow_verdicts(...)`, `build_bow_comparison_table(...)`, `build_divergence_dashboard(...)`, `build_risk_disagreement_ranking(...) → list[RiskDisagreementItem]` |
| **Outputs** | Dataclass instances and markdown strings; no files written. |
| **External deps** | `freshness` (imported lazily). |
| **How invoked** | Imported by report assembly. |
| **State between calls** | `VERDICT_LEVELS`, `SEVERITY_RANK`, `SHORT_VERDICT`, `VERDICT_LETTER`, `VERDICT_STEP` dicts (module-level). |

### `scenario_calculator.py`

| Field | Detail |
|---|---|
| **What it does** | Pure-Python deterministic scenario calculators covering burn rate/runway, funding gap, timeline cascade, enrollment projections, statistical power, manufacturing timeline, policy adoption — each returns `ScenarioResult` with best/base/worst cases. |
| **Inputs** | `burn_rate_runway(facts: InvestmentFacts) → ScenarioResult`; `enrollment_projection(...)`, etc. |
| **Outputs** | `ScenarioResult` instances; no files written. |
| **External deps** | `evidence_model.InvestmentFacts`; stdlib `math`, `datetime` only. |
| **How invoked** | Imported by analysis pipeline scenario steps. |
| **State between calls** | None. |

---

## 4. Stage 3 — Prepare-Research

### `research_planner.py`

| Field | Detail |
|---|---|
| **What it does** | Extracts research recommendations from analyst run artifacts; generates de-novo questions in parallel from two LLMs; deduplicates semantically via `text-embedding-3-small` cosine similarity (threshold 0.85); writes a research plan. |
| **Inputs** | `build_research_plan(run_dir: Path, call_llm) → Path`; also CLI `python -m src.qpr.research_planner --run-dir <path>`. |
| **Outputs** | Writes `run_dir/research_plan.json` and `research_plan.md`; returns `Path`. |
| **External deps** | `openai` (embeddings via `text-embedding-3-small`); `numpy`. |
| **How invoked** | `python -m src.qpr prepare-research --run-dir <D>` → `_cmd_prepare_research` → `build_research_plan(...)`. |
| **State between calls** | None. |

---

## 5. Stage 4 — Research

### `research_dispatch.py`

| Field | Detail |
|---|---|
| **What it does** | Dispatches SLR/LBD/deep_web/Edison research tasks in parallel (`ThreadPoolExecutor`, `max_workers=16`); caches per-task results; Edison batch path uses `EdisonLiteratureClient.query_batch()` with optional query rewriting. |
| **Inputs** | `dispatch_research(research_tasks, output_dir, *, max_workers=16) → list[dict]`; `dispatch_edison_research(research_tasks, output_dir, *, max_concurrent, timeout, priority_filter, skip_rewrite) → list[dict]` |
| **Outputs** | Returns `list[dict]`; writes per-task `research/{scope}/{task}/result.json`, `research_tasks.json`, `dispatch_results.json`, `edison_results.json`, `edison_rewritten_queries.json`. |
| **External deps** | `ResearchAgent` (SLR), `run_lbd_agent` (LBD), `OpenAlexClient`, `AstaClient`, `EdisonLiteratureClient`; env var `EDISON_PLATFORM_API_KEY`. |
| **How invoked** | `python -m src.qpr research --run-dir <D>` → `_cmd_research` → `dispatch_research(...)` or `dispatch_edison_research(...)`. |
| **State between calls** | Per-task `result.json` cache files on disk. |

### `deep_web_research.py`

| Field | Detail |
|---|---|
| **What it does** | Performs deep web research via OpenAI Responses API; primary path uses `o3-deep-research` with `web_search_preview`; fallback uses `gpt-5.2` with 3 rounds of iterative web search. |
| **Inputs** | `deep_web_research(question, context, *, model="o3-deep-research", fallback_model, timeout) → DeepWebResult`; `run_deep_web_claim(claim_text, evidence_question, decision_context) → dict` |
| **Outputs** | Returns `DeepWebResult` dataclass; no files written. |
| **External deps** | `openai` (Responses API). |
| **How invoked** | Called from `research_dispatch.dispatch_research` for `deep_web` task type; from `thread_sub_agent` for web-augmented synthesis. |
| **State between calls** | None. |

### `edison_query_rewriter.py`

| Field | Detail |
|---|---|
| **What it does** | Rewrites internal investment-review assumptions into universally searchable scientific questions for Edison LITERATURE_HIGH; removes internal identifiers (INV-XXXXXX, BOW, BMGF) while preserving scientific terms. |
| **Inputs** | `rewrite_query(assumption, context, *, call_llm, model=VERIFICATION_MODEL) → str`; `rewrite_batch(items, *, call_llm, model, max_workers) → list[dict]` |
| **Outputs** | Returns rewritten query strings; no files written. |
| **External deps** | `call_llm`. |
| **How invoked** | Called from `research_dispatch.dispatch_edison_research` unless `skip_rewrite=True`. |
| **State between calls** | None. |

---

## 6. Stage 5 — Finalize

### `report_enrichment.py`

| Field | Detail |
|---|---|
| **What it does** | Step 5 finalizer — matches external research results to open-scope placeholders in the assembled report; rewrites executive summary, key findings, and research-needed sections. |
| **Inputs** | `finalize_report(run_dir: Path, research_dir: Path, call_llm) → Path`; reads `final_report.md`, `research/*.json`. |
| **Outputs** | Writes `final_report_wresearch.md` under `run_dir/threads/`; optionally triggers PDF render. Returns `Path`. |
| **External deps** | No third-party libraries; no env vars specific to this module. |
| **How invoked** | `python -m src.qpr finalize --run-dir <D>` → `_cmd_finalize` → `finalize_report(...)`. |
| **State between calls** | None. |

---

## 7. Report Assembly

### `report_assembler.py`

| Field | Detail |
|---|---|
| **What it does** | Central orchestrator for report assembly — drives section writing, executive-summary generation (v2 typology-driven), key-insights narration, and final Markdown stitching. |
| **Inputs** | `assemble_report(run_dir, program, investments_by_id, scope_outputs, clusters, findings, call_llm, *, model, figures_dir, lens_ids, use_verification, parallel) → str` |
| **Outputs** | Returns final Markdown string; writes `final_report.md`, `key_insights.md`, `exec_summary.md`, `excerpts.csv` bibliography under `run_dir/threads/`. |
| **External deps** | `evidence_model`, `llm_utils`, `model_defaults`, `narrator_style`, `report_writer`, `key_insights_passes`, `exec_summary_passes`, `exec_summary_critic`. |
| **How invoked** | Called from `research_analyst_agent` phase 6. No direct CLI. |
| **State between calls** | `EXEC_SUMMARY_V2=True`, `_ASSEMBLY_MAX_RETRIES=5`, `_ASSEMBLY_RETRY_DELAY=60` (module-level). |

### `report_writer.py`

| Field | Detail |
|---|---|
| **What it does** | Writes individual section drafts via LLM, managing per-section word budgets and source-reference canonicalisation. |
| **Inputs** | `write_section_draft(thread, findings, source_index, call_llm, *, model, max_findings) → SectionDraft` |
| **Outputs** | Returns `SectionDraft`; writes working notes to `run_dir/threads/working_notes/<section_id>.md`. |
| **External deps** | None beyond project modules. |
| **How invoked** | Called from `report_assembler` and `thread_sub_agent`. |
| **State between calls** | `WORD_BUDGETS: dict[str, int]`, `_SECTION_WRITER_SYSTEM` prompt (module-level). |

### `exec_summary_passes.py`

| Field | Detail |
|---|---|
| **What it does** | Pre-pass utilities for executive-summary generation — selects material deviations, detects emergent portfolio patterns, builds strategy alignment text, validates topic/phrase coverage. |
| **Inputs** | `select_material_deviations_for_exec(...)`, `detect_emergent_portfolio_patterns(...)`, `build_strategy_alignment(...)`, `check_topic_coverage(text) → list[str]`, `strip_forbidden_phrases(text) → str` |
| **Outputs** | Python objects/strings; no files written. |
| **External deps** | Re-exports from `exec_summary._shared`. |
| **How invoked** | Imported by `report_assembler`. |
| **State between calls** | `_DEVIATION_PATTERNS`, `_REQUIRED_TOPIC_PHRASES`, `_FORBIDDEN_PHRASES` (module-level). |

### `exec_summary_critic.py`

| Field | Detail |
|---|---|
| **What it does** | Post-draft critic/revise loop for exec-summary sections — enforces structural entity coverage (≥0.70 threshold) and soft quality rubric via LLM call. |
| **Inputs** | `critic_revise(*, draft_text, section_kind, strategy_block, payload_block, cluster_block, call_llm, model, run_dir, max_tokens) → tuple[str, dict]` |
| **Outputs** | Returns `(revised_text, trace_dict)`; writes critic trace JSON to `run_dir/critic_trace/<section_kind>.json`. |
| **External deps** | None beyond project modules. |
| **How invoked** | Called from `report_assembler` exec-summary generation. |
| **State between calls** | `_CRITIC_RUBRIC` prompt string (module-level). |

### `exec_summary/bet_assessment_cache.py`

| Field | Detail |
|---|---|
| **What it does** | Read/write cache for `BetScienceAssessment` objects keyed by bet fingerprint and cascade-prompt hash; atomic save via `os.replace`. |
| **Inputs** | `load_cache(cache_path)`, `save_cache(cache_path, cache)`, `partition_bet_inputs(bets, cache, cache_path) → tuple[list, list]` |
| **Outputs** | Writes `bet_assessments.json` atomically. |
| **External deps** | env var `NQPR_SKIP_CASCADE_CACHE`. |
| **How invoked** | Called from `premise_investigator`. |
| **State between calls** | `CACHE_VERSION="1.0"`; `@lru_cache(maxsize=1)` on `compute_cascade_prompts_hash()` (module-level). |

### `exec_summary/bet_premise_inventory.py`

| Field | Detail |
|---|---|
| **What it does** | Deterministically assembles per-bet premise inventories from `science_validation.flags` (no LLM calls), producing frozen dataclass records used by the cascade passes. |
| **Inputs** | `gather_all_bet_inventories(*, major_bets, investments_by_id, reports_by_id) → list[BetPremiseInventory]` |
| **Outputs** | Returns `list[BetPremiseInventory]`; no files written. |
| **External deps** | None. |
| **How invoked** | Called from exec-summary orchestration in `report_assembler`. |
| **State between calls** | None. |

### `exec_summary/premise_investigator.py`

| Field | Detail |
|---|---|
| **What it does** | Runs the three-pass per-bet scientific-premise cascade (Investigation → Key Findings → Exec Summary), each pass using Pydantic-validated LLM output and optional agentic tool calls; emits `BetScienceAssessment`. |
| **Inputs** | `run_per_bet_investigation(*, bet_id, bet_description, aggregate_exposure_usd, n_investments, premises, reports_by_id, investments_by_id, asta_client, tools, portfolio_lookup, call_llm, model, max_iterations, ...) → BetScienceAssessment` |
| **Outputs** | Returns `BetScienceAssessment`; writes OpenTelemetry spans. |
| **External deps** | `opentelemetry`, `pydantic` v2. |
| **How invoked** | Called from exec-summary orchestration. |
| **State between calls** | `_tracer = trace.get_tracer(...)` module-level. |

### `exec_summary/_shared.py`

| Field | Detail |
|---|---|
| **What it does** | Pure utility helpers for funding-posture classification, dollar-exposure formatting, and Markdown escaping shared across exec-summary modules. |
| **External deps** | None. |
| **How invoked** | Imported by sibling exec-summary modules. |

### `score_comparison_figures.py`

| Field | Detail |
|---|---|
| **What it does** | Renders PNG comparison figures: team-vs-AI confusion matrix and reallocatable-capital timeline scatter. |
| **Inputs** | `render_team_vs_ai_confusion(scope_outputs, investments_by_id, inv_scores, output_path, ...) → tuple[Path, str] | None`; `render_reallocatable_timeline(...) → tuple[Path, str] | None` |
| **Outputs** | Writes `.png` files; returns `(Path, alt_text)`. |
| **External deps** | `matplotlib`, `numpy`. |
| **How invoked** | Called from `report_assembler`. |
| **State between calls** | `SCORE_LADDER`, `SHORT_LABEL`, `VERDICT_LETTER` constants (module-level). |

### `key_insights_passes.py`

| Field | Detail |
|---|---|
| **What it does** | Generates the Key Insights section by running five parallel lens-specific narrators (optionally agentic with `NarrationToolbox`), then applies a Stage 2 verification pass per lens before stitching. |
| **Inputs** | `generate_key_insights(clusters, exec_inputs, investments_by_id, scope_outputs, call_llm, *, model, figures_dir, ctx_for_toolbox, lens_ids, use_verification, parallel) → str` |
| **Outputs** | Returns Markdown string; no files written directly. |
| **External deps** | `concurrent.futures`; `narration_tools.NarrationToolbox`. |
| **How invoked** | Called from `report_assembler`. |
| **State between calls** | `LENS_SPECS: dict[str, LensSpec]` (5 entries; module-level). |

### `narration_tools.py`

| Field | Detail |
|---|---|
| **What it does** | Provides the `NarrationToolbox` agentic tool-calling harness for narrator LLMs — exposes 6 portfolio-query tools with budget tracking and JSONL tracing; `run_narrator_with_tools` drives the agentic loop. |
| **Inputs** | `NarrationToolbox(scope_outputs, investments_by_id, excerpts_csv_path, budget, trace_path)`; `run_narrator_with_tools(toolbox, system_prompt, user_prompt, *, model, max_iterations, max_tokens_per_call) → tuple[str, list[dict]]` |
| **Outputs** | Returns `(narrative_text, tool_call_log)`; appends JSONL to `trace_path`. |
| **External deps** | `anthropic`. |
| **How invoked** | Instantiated by `key_insights_passes` lens narrators. |
| **State between calls** | Instance: `_calls_used`, `budget`, `_excerpt_index` (lazy). |

### `narrator_style.py`

| Field | Detail |
|---|---|
| **What it does** | Single-constant module exporting `NARRATOR_STYLE_RULES: str` — stylistic rules injected into narrator system prompts. |
| **External deps** | None. |
| **How invoked** | `from src.qpr.narrator_style import NARRATOR_STYLE_RULES`. |

### `section_validator.py`

| Field | Detail |
|---|---|
| **What it does** | Validates a written section draft for filler phrases, missing source references, and BoW-ID coverage; returns a `ValidationResult`. |
| **Inputs** | `validate_section(draft, source_ids, bow_ids) → ValidationResult` |
| **External deps** | None. |
| **How invoked** | Imported by report assembly passes. |

### `md_to_pdf.py`

| Field | Detail |
|---|---|
| **What it does** | Converts Markdown to PDF using weasyprint (primary) with pandoc/wkhtmltopdf fallback; hill-climbing optimizer for table column widths. |
| **Inputs** | `md_to_pdf(md_path, pdf_path, toc=True, toc_depth=4) → Path`; CLI `python -m src.qpr.md_to_pdf input.md [output.pdf] [--no-toc]`. |
| **Outputs** | Writes PDF file; returns `Path`. |
| **External deps** | `weasyprint`, `markdown` (python-markdown); `pandoc`, `wkhtmltopdf` (optional CLI). |
| **How invoked** | `python -m src.qpr.md_to_pdf` or imported; called from `report_assembler.render_pdf`. |

---

## 8. Evidence Audit

### `evidence_audit/audit.py`

| Field | Detail |
|---|---|
| **What it does** | Top-level evidence-audit orchestrator; sequences all sub-audits (coverage, citations, weak evidence, expected docs, usage diff), merges results, and writes `evidence_audit.json`. |
| **Inputs** | `run_audit(program, data_root, run_dir_name, top_n_files, skip_llm_expected_docs, inv_limit, model, force_llm, expected_docs_scope, ...) → dict` |
| **Outputs** | Returns audit dict; writes `<run_dir>/audit/evidence_audit.json`. |
| **External deps** | All `evidence_audit/` sub-modules. |
| **How invoked** | `python -m src.qpr evidence-audit --program <P>` → `_cmd_report_evidence_audit` → `run_audit(...)`. |

### `evidence_audit/loaders.py`

| Field | Detail |
|---|---|
| **What it does** | Loads and caches all program artifacts (report JSON, doc list, excerpts CSV, sidecars) into the `ProgramArtifacts` dataclass — the single data carrier throughout the audit module. |
| **Inputs** | `load_program(program, data_root, run_dir_name) → ProgramArtifacts` |
| **Outputs** | Returns `ProgramArtifacts`; no files written. |
| **External deps** | None. |
| **How invoked** | Called from `run_audit` and `_cmd_report_evidence_audit`. |
| **State between calls** | `cached_property` descriptors persist `doc_index`, `filename_to_doc`, `views` on the instance. |

### `evidence_audit/citation_views.py`

| Field | Detail |
|---|---|
| **What it does** | Provides the single canonical `CitationViews` frozen dataclass merging all three citation layers (structured, inline `[EX-...]`, investment-level) — authoritative answer to "is this file cited?". |
| **Inputs** | `CitationViews.build(art: ProgramArtifacts) → CitationViews`; `resolve_inline_ex_file_ids(art) → set[str]` |
| **Outputs** | Returns `CitationViews` (frozen); no files written. |
| **External deps** | None. |

### `evidence_audit/expected_docs.py`

| Field | Detail |
|---|---|
| **What it does** | Three-stage expected-document matcher: Stage 1 lexical (Jaccard + synonym normalization), Stage 2 LLM rescore, Stage 3 LLM substance-check; writes per-investment JSON results and inferred-labels sidecars. |
| **Inputs** | `run_expected_docs(art, inv_limit, model, force, scope, max_workers, ...) → list[dict]`; `audit_investment(art, inv_id, intel, cache_dir, ...) → dict` |
| **Outputs** | Writes `<run_dir>/audit/expected_docs/{INV}.json` per investment; writes `{Program}-ingested/inferred_labels/{INV}.json` sidecars. |
| **External deps** | `yaml` (reads `lifecycle_taxonomy.yaml`); `llm_utils.call_llm`; `doc_types.canonicalize_doc_type`. |
| **State between calls** | `_TAXONOMY_CACHE` dict, `_MATCH_STOPWORDS`, `_MATCH_SYNONYMS` (module-level). |

### `evidence_audit/nqpr_usage_diff.py`

| Field | Detail |
|---|---|
| **What it does** | Computes the diff between audit-confirmed document usage and NQPR pipeline usage — identifies high-value files confirmed read but never cited, and findings with unverified evidence. |
| **Inputs** | `compute_nqpr_usage_diff(art, audit_results, sidecars) → dict` |
| **Outputs** | Returns dict; no files written. |

### `evidence_audit/coverage.py`

| Field | Detail |
|---|---|
| **What it does** | Tallies ingested-but-unused documents and checks per-investment core-document presence against a required-doc taxonomy. |
| **Inputs** | `ingested_but_unused(art) → dict`; `per_investment_doc_coverage(art) → dict` |

### `evidence_audit/citations.py`

| Field | Detail |
|---|---|
| **What it does** | Computes top-N most-cited files by weighted influence and source-type × finding-type cross-tabulation matrices. |
| **Inputs** | `file_influence(art, top_n) → list[dict]`; `source_type_x_finding_type(art) → dict`; `doc_type_x_finding_type(art) → dict` |

### `evidence_audit/weak_evidence.py`

| Field | Detail |
|---|---|
| **What it does** | Identifies four categories of weak-evidence signals: low-confidence findings, external-research-leaning findings, unresolved gaps, team-flagged investments the analyst never engaged with. |

### `evidence_audit/brief.py`

| Field | Detail |
|---|---|
| **What it does** | Renders and writes the human-facing team brief Markdown from the audit dict. |
| **Inputs** | `render_brief(audit) → str`; `write_brief(audit) → Path` |
| **Outputs** | Writes `<run_dir>/audit/team_brief.md`; returns `Path`. |

### `evidence_audit/rollup.py`

| Field | Detail |
|---|---|
| **What it does** | Renders and writes the cross-program audit rollup summary (`cross_program_rollup.{md,json}`). |
| **Outputs** | Writes `~/qpr-collections/audit/cross_program_rollup.{md,json}`. |
| **How invoked** | Called by `evidence-audit --program all`. |

### `evidence_audit/cross_program.py`

| Field | Detail |
|---|---|
| **What it does** | Aggregates per-program audit dicts into a cross-program investment coordination table. |
| **Outputs** | Writes `~/qpr-collections/audit/cross_program_investments.{md,json}`. |
| **How invoked** | Called by `evidence-audit --program all`. |

### `evidence_audit/diagnosis.py`

| Field | Detail |
|---|---|
| **What it does** | Generates a narrative diagnosis report for high-value but unused files, enriched with historical run context from sibling experiment directories. |
| **Inputs** | `write_diagnosis(art, audit) → Path` |
| **Outputs** | Writes `<run_dir>/audit/unused_high_value_diagnosis.md`. |
| **How invoked** | `evidence-audit --diagnosis`. |

### `evidence_audit/workbook.py`

| Field | Detail |
|---|---|
| **What it does** | Writes the 5-sheet audit workbook as `.xlsx` (document usage, coverage gaps, citation matrix, top files, investment summary). |
| **Inputs** | `write_workbook(art, audit) → Path` |
| **Outputs** | Writes `<run_dir>/audit/{Program}_audit_workbook.xlsx`. |
| **External deps** | `openpyxl`. |
| **How invoked** | `evidence-audit --xlsx`. |

### `evidence_audit/render_utils.py`

| Field | Detail |
|---|---|
| **What it does** | Shared Markdown table formatter and dollar-amount parser used across audit sub-modules. |

---

## 9. Gold-Standard Verifier (gs_verifier)

### `gs_verifier/cli.py`

| Field | Detail |
|---|---|
| **What it does** | Top-level CLI for the gold-standard re-verifier — loads gold/doc_list/scoring, fans out to `_process_one_finding` via `ThreadPoolExecutor`, incremental-saves every 25 completions, optionally runs portfolio rollup. |
| **Inputs** | CLI: `--program --scope|--all-scopes --limit --workers --as-of-date --gold-path --out --trace --no-causal --rollup` |
| **Outputs** | Writes `{out}.json` (verdicts), `{out}.trace.jsonl`. |
| **External deps** | `dotenv`; all `gs_verifier/` sub-modules. |
| **How invoked** | `python -m src.qpr.gs_verifier --program PPP --all-scopes`. |
| **State between calls** | `causal_cache` dict (in-memory, shared across threads via lock) per run. |

### `gs_verifier/dual_verifier.py`

| Field | Detail |
|---|---|
| **What it does** | Runs two independent LLM verifiers (Opus A + GPT B) in parallel per finding; validates completeness; retries once on structural gaps; delegates to reconciler. |
| **Inputs** | `verify_finding(*, finding, scope_label, finding_type, evidence_bundle_rendered, causal_model_rendered, as_of, trace_path, verifier_a, verifier_b) → dict` |
| **Outputs** | Verdict dict with `verifier_a`, `verifier_b`, `reconciliation`; appends JSONL to `trace_path`. |
| **External deps** | `llm_utils.call_llm`; `model_defaults.ANALYSIS_MODEL`, `SYNTHESIS_MODEL`. |
| **State between calls** | `_SESSION_UUID` (stable per program run), `_trace_lock` (threading.Lock; module-level). |

### `gs_verifier/context_builder.py`

| Field | Detail |
|---|---|
| **What it does** | Deterministic Phase 1 context-bundle assembly — loads investment docs filtered to as-of date, packs full text within 420K-char budget, loads pipeline-provenance artifacts; caches per `(program, inv, finding_id, as_of)`. |
| **Inputs** | `build_bundle(finding, program, doc_list, scoring, as_of, cache=True) → ContextBundle`; `render_bundle_for_prompt(bundle) → str` |
| **Outputs** | `ContextBundle`; optionally writes cache JSON to `~/.gs_verifier_cache/`. |
| **State between calls** | `FULL_TEXT_BUDGET_CHARS=420_000`, `CACHE_ROOT` (module-level). |

### `gs_verifier/pipeline_artifacts.py`

| Field | Detail |
|---|---|
| **What it does** | Loads and caches source-run artifacts (EX-* excerpts, §-refs, causal models, link assessments, retrieval chunks, phase34 trace events) for verifier prompts. |
| **Inputs** | `get_run_artifacts(program, run_name) → RunArtifacts`; `get_artifacts_for_finding(program, finding) → RunArtifacts` |
| **State between calls** | `_ARTIFACT_CACHE: dict[(program, run_name), RunArtifacts]` (module-level, thread-safe). |

### `gs_verifier/reconciler.py`

| Field | Detail |
|---|---|
| **What it does** | Deterministic reconciliation of two verifier verdicts — exact fine-label agreement → locked; coarse-category agreement → locked on Opus fine label; genuine disagreement → flagged. |
| **Inputs** | `reconcile(a: dict, b: dict) → dict` |
| **External deps** | None. |

### `gs_verifier/verifier_prompts.py`

| Field | Detail |
|---|---|
| **What it does** | Houses all prompt text for schema v2 dual verification; `build_verifier_prompt(...)` assembles the final per-finding prompt; `compute_prompt_hash()` produces SHA256 fingerprint. |
| **State between calls** | `SCHEMA_VERSION="2.0"`, `PROMPT_VERSION="v2.0"` (module-level). |

### `gs_verifier/audit_extractor.py`

| Field | Detail |
|---|---|
| **What it does** | Post-processing extractor — flattens schema-v2 verdicts into JSONL audit records, builds overrule-basis frequency tables, extracts pipeline-wisdom-retention ledger statements. |
| **Inputs** | `run(input_path, output_dir) → dict`; CLI: `--input --output-dir` |
| **Outputs** | Writes `per_finding_records.jsonl`, `overrule_basis_frequencies.json`, `pipeline_wisdom_ledger.jsonl`, `audit_summary.md`. |

### `gs_verifier/build_tiered_gold.py`

| Field | Detail |
|---|---|
| **What it does** | Produces three tiered gold standard files: Tier 1 (strict agreed non-reject), Tier 2 (review queue), Tier 3 (dropped). |
| **Outputs** | Writes `gold_tier1_strict.json`, `gold_tier2_review.json`, `gold_tier3_dropped.jsonl`, `tier_summary.md`. |

### `gs_verifier/apply_verdicts.py`

| Field | Detail |
|---|---|
| **What it does** | Applies locked verifier verdicts to produce `gold_v4.json`; outputs rejected and flagged findings to separate files. |
| **Outputs** | Writes `gold_v4.json`, `rejected.jsonl`, `flagged_for_review.jsonl`. |

### `gs_verifier/portfolio_rollup.py`

| Field | Detail |
|---|---|
| **What it does** | Phase 4 rollup — clusters retained findings by shared investment ID across scopes to surface cross-BOW duplicate candidates. |

### `gs_verifier/summary_report.py`

| Field | Detail |
|---|---|
| **What it does** | Generates a human-readable Markdown summary of a re-verification run (locked-status distribution, agreement rates, taxonomy shift cross-tab). |
| **How invoked** | `python -m src.qpr.gs_verifier.summary_report --input ... --output ...` |

### `gs_verifier/causal_refresh.py`

| Field | Detail |
|---|---|
| **What it does** | Phase 1b — extracts a fresh per-investment causal model (forcing `NQPR_FREE_FORM_REASONING=true` to avoid the 77% finding-loss json_mode bug); caches per investment. |
| **Outputs** | Caches to `~/.gs_verifier_cache/{program}/{inv}/causal_model.json`. |

### `gs_verifier/discovery.py`

| Field | Detail |
|---|---|
| **What it does** | Stub for an inverse finding-discovery pass (raises `NotImplementedError`). |

---

## 10. Diagnostics

### `diagnostics/carve_out.py`

| Field | Detail |
|---|---|
| **What it does** | Tags ingested document chunks as canonical or non-canonical carve-outs using Jaccard word-bag similarity against the version-family representative; mutates chunk dicts in-place. |
| **Inputs** | `compute_carve_outs(chunks, *, threshold=0.3) → list` |
| **External deps** | `diagnostics.family_audit._tokenize_words`, `jaccard_words`. |

### `diagnostics/family_audit.py`

| Field | Detail |
|---|---|
| **What it does** | Builds version-family audit records for ingested documents, classifying non-canonical excerpts via Jaccard similarity and heuristic regex patterns (stale language, staff notes, future tense, past year). |
| **Inputs** | `build_family_audit(doc_list, evidence_files, excerpts_by_id) → list[dict]`; `jaccard_words(a, b) → float` |
| **External deps** | None. |

---

## 11. LLM Infrastructure & Shared Utilities

### `llm_utils.py`

| Field | Detail |
|---|---|
| **What it does** | Unified LLM gateway routing `claude-*` to Anthropic and `gpt-`/`o1-`/`o3-`/`o4-` to OpenAI; includes `safe_parse_json` (5-strategy extraction), JSONL trace logging, checkpoint save/load, and `trace_llm_context()` ContextVar context manager. |
| **Inputs** | `call_llm(prompt, system_msg, *, model, json_mode, max_tokens, images, thinking, ...) → str | dict`; `checkpoint_save(path, data)`, `checkpoint_load(path)` |
| **Outputs** | Returns LLM response; appends JSONL to `LLM_TRACE_FILE`. |
| **External deps** | `src.synthesis.llm_helpers._call_anthropic`, `_call_openai_fallback`; env vars `LLM_TRACE_FILE`, `NQPR_FREE_FORM_REASONING`. |
| **How invoked** | `from src.qpr.llm_utils import call_llm`; used everywhere. |
| **State between calls** | `_TRACE_WRITE_LOCK`, `_TRACE_EMIT_FAILURES`, `_TRACE_CTX` (module-level). |

### `model_defaults.py`

| Field | Detail |
|---|---|
| **What it does** | Declares all LLM model name constants, each overridable via an env var; single source of truth for model routing across the pipeline. |
| **Inputs** | Environment variables: `NQPR_ANALYSIS_MODEL`, `NQPR_SYNTHESIS_MODEL`, `NQPR_AGGREGATE_MODEL`, `NQPR_FAST_MODEL`, `NQPR_VISION_MODEL`, `NQPR_DEEP_WEB_MODEL`, `NQPR_VERIFICATION_MODEL`. |
| **Outputs** | Module-level string constants (e.g. `ANALYSIS_MODEL=gpt-5.5`, `SYNTHESIS_MODEL=claude-opus-4-7`). |
| **How invoked** | `from src.qpr.model_defaults import ANALYSIS_MODEL`. |

### `evidence_model.py`

| Field | Detail |
|---|---|
| **What it does** | Canonical dataclass definitions for the entire pipeline — from raw evidence through causal models, investment reports, adjudicated findings, and portfolio facts; includes `InvestmentFacts.from_scoring_and_timeline` for deterministic financial metric computation. |
| **Exports** | `Evidence`, `Finding`, `ResearchThread`, `ProgramContext`, `AnalystReport`, `ScopeOutput`, `LinkAssessment`, `InvestmentEvidencePack`, `InvestmentFacts`, `ScenarioResult`, `ClaimResult`, `AdjudicatedFinding`, `NecessityAssessment`, `BOWContext`, `Decision`, `InvestmentReport`. |
| **External deps** | `investment_timeline.InvestmentTimeline`. |
| **How invoked** | Imported everywhere; no CLI. |
| **State between calls** | `DECISION_TYPE_VOCABULARY` (frozenset 14), `_THIN_EVIDENCE_DECISION_TYPES` (frozenset 2; module-level). |

### `embedding_index.py`

| Field | Detail |
|---|---|
| **What it does** | Pre-computes and queries float32 numpy embedding vectors with cosine similarity; supports TF-IDF hybrid search via RRF, recency boost for time-sensitive doc types, incremental merge. |
| **Inputs** | `EmbeddingIndex(chunks, vectors, model)`; `build_embedding_index(chunks, output_path, model, batch_size) → EmbeddingIndex`; `search`, `hybrid_search`, `keyword_search` |
| **Outputs** | Writes `.npy` and `.json` files; returns `list[SearchResult]`. |
| **External deps** | `numpy`, `openai`, `sklearn` (TfidfVectorizer), `joblib`, `scipy.sparse`; env vars `QPR_VECTORS_FULLY_LOAD`, `NQPR_EMBED_CONCURRENCY`, `NQPR_RL_MAX_WALL_S`. |
| **State between calls** | Instance: `chunks`, `vectors`, `_norms`, `_collections`, `_bow_ids`, `_inv_ids`, `_doc_types`, `_tfidf_vectorizer`, `_tfidf_matrix`. |

### `freshness.py`

| Field | Detail |
|---|---|
| **What it does** | Annotates `AdjudicatedFinding.source_data_age_months`; computes `decision_priority` combining severity, freshness step function, and log-scaled allocation impact; renders staleness tags. |
| **Inputs** | `annotate_findings(findings, investments_by_id)`, `sort_findings_by_priority(findings, investments_by_id)`, `build_freshness_summary(...)` |
| **External deps** | env var `NQPR_STALE_FINDING_THRESHOLD_MO` (default 12). |
| **State between calls** | `STALE_THRESHOLD_MO`, `SUMMARY_THRESHOLD_MO=6`, `_SEVERITY_WEIGHT` dict (module-level). |

### `exceptions.py`

| Field | Detail |
|---|---|
| **What it does** | NQPR exception hierarchy rooted at `NQPRError`. |
| **Exports** | `NQPRError`, `LLMRefusalError`, `LLMParseError`, `CollectionError`, `CheckpointCorruptError`, `EmbeddingIndexError`, `ResearchDispatchError`, `IntelligenceCanonicalEmptyError`. |

### `file_ids.py`

| Field | Detail |
|---|---|
| **What it does** | Three file-ID schemes: legacy `{inv_id}__{filename}`, hashed `{inv_id}__{filename}__{SHA1[:10]}` for collision disambiguation, and a `preferred_investment_file_id` switcher. |
| **External deps** | `hashlib` (stdlib). |

### `file_registry.py`

| Field | Detail |
|---|---|
| **What it does** | Delta-aware file registry using SHA-256 content hashes; tracks per-step processing versions; provides atomic JSON writes via tempfile + fsync + `os.replace`. |
| **State between calls** | `PIPELINE_VERSIONS` dict (module-level). |

### `run_name.py`

| Field | Detail |
|---|---|
| **What it does** | Generates human-readable run names by combining words from three lists (adjectives, colors, nouns). |
| **Inputs** | `generate_run_name() → str` |

### `glossary.py`

| Field | Detail |
|---|---|
| **What it does** | Loads a program-specific glossary from the collection's `fulltext/C019.txt`; appends 15 BMGF-specific supplemental terms; returns truncated text for LLM prompts. |

### `safety_filter_log.py`

| Field | Detail |
|---|---|
| **What it does** | Thread-safe JSONL appender for LLM safety filter rejections; also calls `src.tracing.spans.add_span_event`. |
| **State between calls** | `_lock` (threading.Lock), `_runtime_dir` (module-level). |

### `web_credibility.py`

| Field | Detail |
|---|---|
| **What it does** | Classifies web search results into credibility tiers: Tier 1 via 55+-entry domain rule table; Tier 2 via LLM (`claude-haiku-4-5-20251001`) with per-domain caching. |
| **State between calls** | `_tier2_cache: dict` keyed by domain (module-level, persists per process). |

### `retrieval_benchmark.py`

| Field | Detail |
|---|---|
| **What it does** | Evaluates search index quality with recall@k, MRR, fuzzy passage match, bootstrap confidence intervals, Cohen's d, and go/no-go comparison reports. |
| **External deps** | `numpy`. |

### `shell_utils.py`

| Field | Detail |
|---|---|
| **What it does** | Single thin wrapper around `shlex.quote` for safe shell argument quoting. |

### `temporal_utils.py`

| Field | Detail |
|---|---|
| **What it does** | Extracts the maximum plausible year in [2000, today+1] from a filename string. |

### `table_cleanup.py`

| Field | Detail |
|---|---|
| **What it does** | Cleans LLM-extracted fulltext via 6 sequential transformations (removes `<br>` tags, repeated path headers, pipe table artifacts, empty pipes, code fences, vision preambles). |

### `migrate_investment_file_ids.py`

| Field | Detail |
|---|---|
| **What it does** | Plans and validates targeted migration for investment file-ID collisions in a collection; produces a JSON+CSV migration plan or a validation report. |
| **How invoked** | `python -m src.qpr.migrate_investment_file_ids --collection-dir ... --invest-dir ... --mode plan|validate` |

---

## 12. Search Backends

### `search/base.py`

| Field | Detail |
|---|---|
| **What it does** | Defines the `SearchIndex` protocol (runtime-checkable), `SearchResult` dataclass with legacy dict-mapping shim, `FilterSpec` frozen dataclass, and `RECENCY_BOOST_DOC_TYPES` constant. |

### `search/local.py`

| Field | Detail |
|---|---|
| **What it does** | `LocalSearchIndex` — SQLite B-tree metadata + memmap numpy vectors + L2 norms; supports per-thread SQLite connections; auto-migrates/repairs on load. Default backend. |
| **External deps** | `numpy`, `sqlite3`, `sklearn`. |
| **State between calls** | Instance: `_conns: dict[int, Connection]`, `_conn_lock`, `sqlite_path`, `vectors`, `norms`. |

### `search/qdrant.py`

| Field | Detail |
|---|---|
| **What it does** | `QdrantCollectionSearchIndex` wrapping qdrant-client with native hybrid (dense + sparse BM25) RRF via `query_points`. |
| **External deps** | `qdrant_client`, `fastembed` (optional for BM25), `openai`. |
| **State between calls** | `_sparse_model` (lazy), `_payload_cache` (lazy dict; instance-level). |

### `search/azure.py`

| Field | Detail |
|---|---|
| **What it does** | `AzureSearchIndex` against foundation-wide `edp-idm-index` (WUS2 private endpoint); always uses `QueryType.SEMANTIC` + `VectorizableTextQuery` hybrid; monkey-patches socket DNS. |
| **External deps** | `azure.search.documents`, `azure.identity`. |
| **State between calls** | `_CREDENTIAL` (lazy), `_SEARCH_CLIENTS` dict (module-level). |

### `search/migrate.py`

| Field | Detail |
|---|---|
| **What it does** | One-shot idempotent migration helper converting `chunks.json` to `chunks.sqlite` B-tree in batches of 5000. |
| **External deps** | `sqlite3`, `json` (stdlib). |

---

## 13. Dependency Graph

```
cli.py
├── Step 1: collect
│   collection_prep.py
│   ├── collection_setup.py
│   │   ├── vision_extraction.py → openai, fitz
│   │   └── document_index.py → call_llm
│   ├── investment_scoring.py → call_llm
│   ├── investment_intelligence.py
│   │   ├── investment_docs.py → fitz, docx, pptx, openpyxl
│   │   ├── doc_expectations.py → call_llm
│   │   ├── semantic_chunker.py
│   │   └── file_ids.py → hashlib
│   ├── multimodal_ingest.py
│   │   ├── batch_vision.py → openai, fitz
│   │   ├── vision_extraction.py
│   │   ├── semantic_chunker.py
│   │   └── table_cleanup.py
│   ├── embedding_index.py → numpy, openai, sklearn
│   │   └── search/local.py → sqlite3
│   ├── asset_registry.py
│   └── collection_health.py → numpy, fitz
│
├── Step 2: analyze
│   collection_loader.py
│   ├── collection_api.py → openai, sklearn, numpy
│   └── search/ (local | qdrant | azure)
│
│   research_analyst_agent.py
│   ├── causal_pipeline.py
│   │   ├── causal_model.py → anthropic
│   │   ├── rubric_evaluator.py → call_llm
│   │   ├── investigation_loop.py → openai, anthropic
│   │   ├── claim_investigator.py → call_llm (parallel ThreadPoolExecutor)
│   │   ├── science_investigator.py → AstaClient
│   │   └── decision_projection.py → call_llm
│   ├── investment_timeline.py → call_llm
│   ├── thread_sub_agent.py → openai
│   ├── numerical_analyst.py → openai (code_interpreter)
│   ├── numerical_verifier.py → call_llm
│   ├── allocation_verifier.py
│   ├── evaluation_comparison.py
│   └── report_assembler.py
│       ├── report_writer.py → call_llm
│       ├── exec_summary_passes.py
│       ├── exec_summary_critic.py → call_llm
│       ├── exec_summary/premise_investigator.py → pydantic, opentelemetry
│       ├── exec_summary/bet_premise_inventory.py
│       ├── exec_summary/bet_assessment_cache.py
│       ├── key_insights_passes.py
│       │   └── narration_tools.py → anthropic
│       ├── score_comparison_figures.py → matplotlib, numpy
│       ├── cluster_identification.py → call_llm
│       └── md_to_pdf.py → weasyprint, markdown
│
├── Step 3: prepare-research
│   research_planner.py → openai (text-embedding-3-small), numpy
│
├── Step 4: research
│   research_dispatch.py
│   ├── deep_web_research.py → openai
│   └── edison_query_rewriter.py → call_llm
│
├── Step 5: finalize
│   report_enrichment.py → call_llm
│
├── evidence-audit
│   evidence_audit/audit.py
│   ├── evidence_audit/loaders.py
│   ├── evidence_audit/citation_views.py
│   ├── evidence_audit/expected_docs.py → yaml, call_llm
│   ├── evidence_audit/nqpr_usage_diff.py
│   ├── evidence_audit/coverage.py
│   ├── evidence_audit/citations.py
│   ├── evidence_audit/weak_evidence.py
│   ├── evidence_audit/brief.py
│   ├── evidence_audit/diagnosis.py
│   ├── evidence_audit/workbook.py → openpyxl
│   ├── evidence_audit/rollup.py
│   └── evidence_audit/cross_program.py
│
└── gs_verifier
    gs_verifier/cli.py
    ├── gs_verifier/context_builder.py
    │   └── gs_verifier/pipeline_artifacts.py
    ├── gs_verifier/dual_verifier.py
    │   ├── gs_verifier/verifier_prompts.py
    │   └── gs_verifier/reconciler.py
    ├── gs_verifier/causal_refresh.py → causal_model.py
    └── gs_verifier/portfolio_rollup.py

Shared infrastructure (used by many):
llm_utils.py → synthesis.llm_helpers → anthropic, openai
model_defaults.py (env-var-overridable model names)
evidence_model.py (all pipeline dataclasses)
embedding_index.py → numpy, openai, sklearn
search/local.py, search/qdrant.py, search/azure.py
safety_filter_log.py → tracing.spans
freshness.py
exceptions.py
file_ids.py, file_registry.py
doc_types.py, doc_expectations.py
```

---

## 14. CLI Commands → Pipeline Stages

| CLI command | Handler | Pipeline stage | What it triggers |
|---|---|---|---|
| `python -m src.qpr collect --program P --data-root D` | `_cmd_collect` | Step 1 | `collection_prep.collect_program` — full 8-step ETL |
| `python -m src.qpr collect --check` | `_cmd_collect` | Step 1 (read-only) | `collection_health.check_collection` |
| `python -m src.qpr collect --repair` | `_cmd_collect` | Step 1 (repair) | `collection_health.repair_collection` |
| `python -m src.qpr precheck --program P` | `_cmd_precheck` | Pre-Step 2 | `precheck.run_checks` — fast integrity validation |
| `python -m src.qpr analyze --collection C` | `_cmd_analyze` | Step 2 | `research_analyst_agent.run_research_analyst` — full 6-phase analysis |
| `python -m src.qpr prepare-research --run-dir D` | `_cmd_prepare_research` | Step 3 | `research_planner.build_research_plan` |
| `python -m src.qpr research --run-dir D` | `_cmd_research` | Step 4 | `research_dispatch.dispatch_research` or `dispatch_edison_research` |
| `python -m src.qpr finalize --run-dir D` | `_cmd_finalize` | Step 5 | `report_enrichment.finalize_report` |
| `python -m src.qpr investigate --collection C --inv-id X --question Q` | `_cmd_investigate` | Standalone | `claim_investigator.investigate_claim` (single investment) |
| `python -m src.qpr evidence-audit --program P` | `_cmd_report_evidence_audit` | Post-hoc audit | `evidence_audit.audit.run_audit` |
| `python -m src.qpr.gs_verifier --program P --all-scopes` | `gs_verifier/cli.py` | Post-hoc verification | Dual LLM re-verification of gold standard findings |
| `python -m src.qpr.batch_vision submit|poll|collect|status` | `batch_vision.main` | Within Step 1 | OpenAI Batch API page-vision extraction |
| `python -m src.qpr.numerical_verifier --report R --collection C` | `numerical_verifier.__main__` | Within Step 2 | Numerical claim verification audit |
| `python -m src.qpr.allocation_verifier --report R --scoring S` | `allocation_verifier.main` | Within Step 2 | Dollar allocation verification + rewrite |
| `python -m src.qpr.migrate_investment_file_ids --collection-dir D --invest-dir E --mode plan|validate` | `migrate_investment_file_ids.main` | Maintenance | File ID collision migration planning |
| `python -m src.qpr.gs_verifier.summary_report --input I --output O` | `summary_report.main` | Post-verification | Human-readable verification summary report |
| `python -m src.qpr.gs_verifier.audit_extractor --input I --output-dir O` | `audit_extractor.main` | Post-verification | Flattened JSONL audit records from verdicts |
| `python -m src.qpr.gs_verifier.build_tiered_gold --program P --verified V --original O --out-dir D` | `build_tiered_gold.main` | Post-verification | Tier 1/2/3 gold standard files |
| `python -m src.qpr.gs_verifier.apply_verdicts --original O --verified V --out-gold G` | `apply_verdicts.main` | Post-verification | Applies locked verdicts to produce `gold_v4.json` |
| `python -m src.qpr.md_to_pdf input.md` | `md_to_pdf.main` | Utility | Markdown → PDF conversion |

---

## 15. Data Passed Between Stages

### Step 1 → Step 2 (Disk artifacts in `{PROGRAM}-ingested/`)

| Artifact | Format | Written by | Read by |
|---|---|---|---|
| `doc_list.json` | JSON array of doc metadata dicts | `collection_prep` | `collection_loader`, `evidence_audit/loaders` |
| `investment_scoring.json` | JSON dict `{inv_id: InvestmentDetail}` | `collection_prep` (from `investment_scoring`) | `collection_loader`, `allocation_verifier` |
| `bow_investment_map.json` | JSON dict `{bow_id: [inv_id, ...]}` | `collection_prep` | `collection_loader` |
| `investment_bow_rows.json` | JSON array of BOW-row objects | `collection_prep` | `collection_loader` |
| `investment_intelligence.json` | JSON dict `{inv_id: InvestmentIntelligence}` | `investment_intelligence` | `collection_loader`, `evidence_audit/expected_docs` |
| `embedding_index/chunks.json` | JSON array of `SemanticChunk` dicts | `collection_prep` (embedding step) | `collection_loader`, `LocalSearchIndex`, `numerical_verifier` |
| `embedding_index/chunks.sqlite` | SQLite B-tree | `search/migrate` (or `LocalSearchIndex.build`) | `LocalSearchIndex` at query time |
| `embedding_index/vectors.npy` | float32 numpy array | `embedding_index.build_embedding_index` | `LocalSearchIndex`, `EmbeddingIndex` |
| `embedding_index/config.json` | JSON (model name, n_chunks, dim) | `embedding_index` | `LocalSearchIndex`, `precheck` |
| `tfidf_index/tfidf_matrix.npz` | scipy sparse CSR | `embedding_index` | `LocalSearchIndex` (hybrid search) |
| `pages/{file_id}/index.json` | JSON `DocumentIndex` per doc | `document_index` | `collection_api.get_document_toc`, `read_section` |
| `pages/{file_id}/p{N}.txt` | Markdown page text | `multimodal_ingest` / `batch_vision` | `collection_api.read_pages`, `context_builder` |
| `intelligence/{inv_id}.json` | JSON `InvestmentIntelligence` cache | `investment_intelligence` | `investment_intelligence` resume path |
| `scoring_vision_cache.json` | JSON cached vision extraction | `investment_scoring` | `investment_scoring` resume path |

### Step 2 → Step 3 (in `{PROGRAM}-experiments/run-{name}/threads/`)

| Artifact | Format | Written by | Read by |
|---|---|---|---|
| `final_report.md` | Markdown | `report_assembler` | `research_planner`, `report_enrichment` |
| `analyst_report.json` | JSON `AnalystReport` | `research_analyst_agent` | `evidence_audit/loaders`, `gs_verifier/pipeline_artifacts` |
| `scope_outputs.json` | JSON `list[ScopeOutput]` | `research_analyst_agent` | `evidence_audit`, `gs_verifier/pipeline_artifacts` |
| `excerpts.csv` | CSV (file_id, section_id, text, §-ref) | `research_analyst_agent` (bibliography) | `evidence_audit`, `narration_tools`, `gs_verifier/pipeline_artifacts` |
| `numerical_provenance.json` | JSON provenance records | `numerical_analyst` | `numerical_verifier` |
| `verification_sources.json` | JSON source list | `research_analyst_agent` | `numerical_verifier` |
| `run_meta.json` | JSON run metadata (timings, coverage) | `_cmd_analyze` | Informational / monitoring |
| `trace.jsonl` | JSONL LLM call trace | `llm_utils` (`LLM_TRACE_FILE`) | `gs_verifier/pipeline_artifacts` |

### Step 3 → Step 4

| Artifact | Format | Written by | Read by |
|---|---|---|---|
| `research_plan.json` | JSON array of task dicts `{id, type, query, priority, linked_scope}` | `research_planner` | `_cmd_research`, `research_dispatch` |
| `research_plan.md` | Markdown (human review copy) | `research_planner` | Human reviewer |

### Step 4 → Step 5 (in `run_dir/research/`)

| Artifact | Format | Written by | Read by |
|---|---|---|---|
| `research/{scope}/{task}/result.json` | JSON per-task research result | `research_dispatch` | `report_enrichment` |
| `dispatch_results.json` | JSON summary of all task results | `research_dispatch` | `report_enrichment` |
| `edison_results.json` | JSON Edison batch results | `dispatch_edison_research` | `report_enrichment` |

### Step 5 → Final Deliverable

| Artifact | Format | Written by | Read by |
|---|---|---|---|
| `final_report_wresearch.md` | Markdown (enriched report) | `report_enrichment` | `md_to_pdf`, human delivery |
| `final_report.pdf` | PDF | `md_to_pdf` / `report_assembler.render_pdf` | Human delivery |

### Evidence Audit Inputs/Outputs

| Artifact | Format | Written by | Read by |
|---|---|---|---|
| `audit/evidence_audit.json` | JSON audit dict | `evidence_audit/audit` | `evidence_audit/brief`, `workbook`, `diagnosis`, `rollup` |
| `audit/team_brief.md` | Markdown | `evidence_audit/brief` | Human reviewer |
| `audit/expected_docs/{INV}.json` | JSON per-investment matcher output | `evidence_audit/expected_docs` | `evidence_audit/audit` |
| `inferred_labels/{INV}.json` | JSON `{satisfied_for[], rejected_for[]}` | `evidence_audit/expected_docs` | `evidence_audit/expected_docs` (caching/learning loop) |
| `audit/{PROGRAM}_audit_workbook.xlsx` | Excel (5 sheets) | `evidence_audit/workbook` | Human reviewer |
| `audit/unused_high_value_diagnosis.md` | Markdown | `evidence_audit/diagnosis` | Human reviewer |
| `audit/cross_program_rollup.{md,json}` | Markdown + JSON | `evidence_audit/rollup` | Human reviewer |
| `audit/cross_program_investments.{md,json}` | Markdown + JSON | `evidence_audit/cross_program` | Human reviewer |

### GS Verifier Inputs/Outputs

| Artifact | Format | Written by | Read by |
|---|---|---|---|
| Gold standard JSON (input) | JSON array of findings | Human / prior pipeline run | `gs_verifier/cli` |
| `{out}.json` (verdicts) | JSON `{scope_id: [verdict_dicts]}` | `gs_verifier/cli` | `gs_verifier/build_tiered_gold`, `apply_verdicts`, `audit_extractor`, `summary_report` |
| `gold_tier1_strict.json` | JSON Tier 1 gold | `build_tiered_gold` | Future pipeline training / evaluation |
| `gold_v4.json` | JSON updated gold | `apply_verdicts` | Next QPR cycle baseline |
| `~/.gs_verifier_cache/` | JSON per-bundle / per-causal-model cache | `context_builder`, `causal_refresh` | Same modules (resume) |
