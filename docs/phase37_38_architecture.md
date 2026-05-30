# Phase 3.7 → 3.8 Architecture: Necessity Check & Decision Projection

## Overview

These two phases form the final analytical layer before the causal subgraph ends.
Phase 3.7 (necessity_check) determines whether each investment is differentiated
from external alternatives. Phase 3.8 (decision_projection) converts that context
plus the accumulated link-assessment evidence into actionable portfolio decisions.

---

## Graph Position

```
                  [identify_gaps] ──────────────────────────────┐
                        │                                        │
          (synthesis chain end)                    (both join at necessity_check)
                                                                 │
[collect_science_results] ────────────────────────────────────► [necessity_check]
        │                                                              │
  (science chain end)                                    dispatch_decision_projections()
                                                               │          │
                                                     [project_scope_decisions × N]
                                                               │
                                                      [collect_decisions]
                                                               │
                                                  [clear_fanout_accumulators] → END
```

**Join semantics:** LangGraph waits for BOTH `identify_gaps` and
`collect_science_results` to complete before firing `necessity_check`.
This guarantees that `scope["synthesis"]`, `scope["gaps"]`, and
`scope["science_flags"]` are all populated when necessity_check reads them.

---

## Phase 3.7 — necessity_check

### What it does (old repo → new repo faithful port)

Implements the same 2-turn DISCOVER + VERIFY web-search loop as
`_phase37_necessity_check` in old repo `causal_pipeline.py:7814-8082`.

**Turn 1 — DISCOVER**
- Builds a discover query from `inv_id`, `org`, `title` (investment_facts), and
  the ToC link-name summary (first 5 link_assessments).
- Calls `search_web(discover_query, rationale)` — OpenAI Responses API with
  web_search tool (async; equivalent to old `_call_with_web_search` Turn 1).
- Calls `acall_llm(discover_prompt, system_msg=NECESSITY_DISCOVER_SYSTEM)` to
  parse raw search results into `{"candidates": [{name, funder, maturity_stage, source}]}`.
- Candidates that lack `name` or `source` URL are dropped (citation rule).

**Turn 2 — VERIFY** (only runs if Turn 1 returned ≥1 candidate)
- Calls `acall_llm(verify_prompt, system_msg=NECESSITY_VERIFY_SYSTEM)` with the
  candidates block embedded.
- The VERIFY system prompt applies the portfolio-logic rubric:
  1. Substitutability test (same outputs → substitutable)
  2. Failure-mode independence (different geo/product/platform → portfolio_bet)
  3. Coverage gap (different population/context → complementary)
- Returns structured JSON with: `differentiation`, `differentiation_rationale`,
  `redundancy_finding`, `counterfactual_loss`, `marginal_contribution`,
  `substitutes`, `portfolio_relationship`, `failure_mode_independence`,
  `confidence`, `sources`.

**Fallback** (Turn 1 returned 0 candidates OR Turn 2 failed)
- Calls `acall_llm(fallback_prompt, system_msg=NECESSITY_VERIFY_SYSTEM)` using
  only BoW context — no web search.
- Forces `confidence="low"` via `_coerce_necessity_payload(fallback_confidence_floor=True)`.

**Citation safeguard (`_coerce_necessity_payload`)**
- If LLM made substantive differentiation/redundancy claims without citing source
  URLs: clears `substitutes`, replaces `redundancy_finding` with an
  "(unverified)" marker, downgrades `confidence` to "low".
- Never backfills web-engine URLs into `sources` unless the LLM also cited at
  least one source itself.

### State consumed

| Field | Source phase | Notes |
|---|---|---|
| `scope["inv_id"]` | AnalyzeState.scopes | Primary investment ID |
| `scope["investment_facts"]` | Phase 1 (evaluate_investment_rubric) | org, title, approved_amount |
| `scope["link_assessments"]` | Phase 3.4 (collect_link_assessments) | Used for ToC link-name summary |
| `scope["causal_model"]` | Phase 3.2 (forecast_consequences) | Fallback for ToC link names |
| `scope["bow_context"]` | Phase 3.1.5 (enrich_bow_context_worker) | field_landscape, comparable_programs, benchmarks |
| `scope["synthesis"]` | Phase 3.5 (synthesize_findings) | Used as narrative excerpt |
| `state["synthesis_model"]` | AnalyzeState | LLM model for all calls |

### State produced

| Field | Type | Notes |
|---|---|---|
| `scope["necessity_assessment"]` | `str` (JSON) | Coerced NecessityAssessment dict; empty string on skip |

**JSON schema written to `scope["necessity_assessment"]`:**
```json
{
  "differentiation": "high|medium|low",
  "differentiation_rationale": "1-2 sentences",
  "redundancy_finding": "named programs or 'none identified'",
  "counterfactual_loss": "what field loses if investment disappears",
  "marginal_contribution": "high|medium|low",
  "substitutes": ["named alternatives"],
  "portfolio_relationship": "substitutable|complementary|portfolio_bet|unclear",
  "failure_mode_independence": "high|medium|low",
  "confidence": "high|medium|low",
  "sources": ["URL1", "URL2"],
  "scope_id": "scope_XXXX",
  "web_searches_performed": 1
}
```

### Adaptation notes (old → new)

| Old repo | New repo |
|---|---|
| Per-investment (one `InvestmentReport` per call) | Per-scope (one scope dict; primary `inv_id` used) |
| `_call_with_web_search()` — synchronous LLM+tools in one call | `search_web()` (raw search) + `acall_llm()` (parse) — two async calls |
| `NecessityAssessment` dataclass | JSON string on `scope["necessity_assessment"]` |
| `ThreadPoolExecutor(max_workers=6)` | Sequential per scope (graph parallelism handles scope-level concurrency) |
| Reads `ir.org`, `ir.title` from InvestmentReport | Reads `scope["investment_facts"]["org"]`, `["title"]` |
| Reads `ir.narrative[:800]` | Reads `scope["synthesis"][:800]` |
| Reads `BOWContext.field_landscape` etc. | Reads `scope["bow_context"]["field_landscape"]` etc. |

---

## Phase 3.8 — decision_projection

### What it does (old repo → new repo faithful port)

Implements the same goal-anchored prompt structure as `_build_decisions_prompt`
in old repo `decision_projection.py:192-295`.

**Prompt structure (§1-§4):**

**§1 — BoW strategic goal** (goal anchor for every decision)
- Sources from `scope["bow_context"]["field_landscape"][:1500]`.
- If absent: placeholder telling LLM to infer goal from scope label.
- Purpose: every decision the LLM emits must link back to this stated goal
  via the `goal_link` field.

**§2 — Per-INV portfolio role** (necessity assessment)
- Parses `scope["necessity_assessment"]` JSON string.
- Renders: `portfolio_relationship`, `differentiation`, `marginal_contribution`,
  `differentiation_rationale[:300]`, `counterfactual_loss[:200]`,
  `substitutes[:3]`.
- If no assessment: renders "(no necessity assessment available)".
- Purpose: tells the LLM WHY this investment exists in the portfolio so
  decisions are framed against its unique role, not just its evidence gaps.

**§3 — Evidence: link assessments**
- Calls `_render_link_for_prompt(la, link_financial)` per link assessment.
- `link_financial` is a lookup `{link_name → causal_model_link}` for financial
  data (`dollars_at_risk`, `months_at_risk`) not stored on the assessment dict.
- Renders: link name, status, confidence, prose analysis[:400], dollars_at_risk,
  months_at_risk, evidence_refs[:6].
- Token-budget: prose capped at 400 chars (up from 200 in the minimal version).

**§4 — Task instructions**
- Structured JSON schema for the LLM to emit decisions.
- Includes `substitution_path` field (from necessity.substitutes) — ported from
  old repo's `_TASK_INSTRUCTIONS`.
- Rules: decision_type from controlled vocab, triggering_link_ids must cite real
  link names, no $ fabrication.

**Post-LLM pipeline (unchanged from existing new-repo implementation):**
1. `_sanitize_candidate` — validate type, non-empty action, dedup triggering_link_ids
2. `_section1a_gate` — drop low-evidence decisions unless thin-type or corroboration ≥ 2
3. `_compute_rank_score` — corroboration × materiality × log10($) × urgency × evidence
4. `_apply_caps` — max DECISION_MAX_PER_INV per inv, DECISION_MAX_PER_SCOPE total

### State consumed

| Field | Source phase | Notes |
|---|---|---|
| `scope_output["bow_context"]` | Phase 3.1.5 | §1 goal anchor |
| `scope_output["necessity_assessment"]` | Phase 3.7 | §2 portfolio role (JSON string) |
| `scope_output["investment_facts"]` | Phase 1 | org, title for §2 header |
| `scope_output["link_assessments"]` | Phase 3.4 | §3 evidence |
| `scope_output["causal_model"]` | Phase 3.2 | §3 financial lookups |
| `scope_output["synthesis"]` | Phase 3.5 | (available; not directly in §3 anymore) |
| `scope_output["science_flags"]` | Phase 3.5d | (available if needed) |

### State produced

| Field | Type | Notes |
|---|---|---|
| `scope_decisions` (accumulator) | `list[dict]` | `{scope_id, decisions: [Decision.to_dict()]}` |
| `scope["decisions"]` | `list[dict]` | Written by `collect_decisions` from accumulator |

**Decision dict schema:**
```json
{
  "inv_id": "INV-XXXXXX",
  "bow_id": "BOW-XXX",
  "bow_ids": ["BOW-XXX"],
  "decision_type": "<vocab term>",
  "recommended_action": "specific action",
  "goal_link": "how this serves BoW goal",
  "substitution_path": "from necessity.substitutes",
  "triggering_link_ids": ["LinkName→Target"],
  "triggering_evidence": ["§0001", "§0002"],
  "aggregate_evidence_level": "",
  "corroboration_count": 2,
  "cost_impact_dollars": 5000000.0,
  "timeline_impact_months": 6.0,
  "confidence": "medium",
  "urgency": "quarterly",
  "materiality": "material",
  "rank_score": 3.14
}
```

### Adaptation notes (old → new)

| Old repo | New repo |
|---|---|
| Takes `ScopeOutput` dataclass | Takes `scope_output` dict |
| Iterates `scope_output.investment_reports` (per-inv necessity) | One necessity assessment per scope (primary inv_id) |
| `_render_link_for_prompt(la: LinkAssessment)` uses dataclass fields | `_render_link_for_prompt(la: dict, link_financial)` uses dict + causal_model lookup |
| `_filter_and_rank` resolves dollars/months directly from `LinkAssessment` | LLM provides corroboration_count/dollars/months (link dicts lack these in new schema) |
| Synchronous `call_llm()` | Async `acall_structured()` (Pydantic schema validation) |
| `ThreadPoolExecutor(max_workers=4)` | Graph-level fan-out via `Send()` in `dispatch_decision_projections` |

---

## State transition diagram

```
scope_outputs after collect_evidence_packs:
  {scope_id, inv_id, bow_ids, label,
   evidence_packs:[], link_assessments:[], science_flags:[],
   causal_model:None, bow_context:None,
   synthesis:"", critique:"", gaps:"",
   necessity_assessment:"", decisions:[]}

        ↓ [Phase 3.1] forecast_consequences
  + causal_model: {theory_of_change, links:[{name, mechanism, dollars_at_risk, months_at_risk}]}

        ↓ [Phase 3.1.5] enrich_bow_context_worker
  + bow_context: {field_landscape, comparable_programs, benchmarks, market_context}

        ↓ [Phase 3.4] collect_link_assessments
  + link_assessments: [{link_id, inv_id, scope_id, terminal_status, confidence,
                        prose, evidence_refs, findings:{}, ...}]

        ↓ [Phase 3.5] synthesize_findings → critique_synthesis → identify_gaps
  + synthesis: "prose"
  + critique: "prose"
  + gaps: "prose"

        ↓ [Phase 3.5d] collect_science_results
  + science_flags: [{question, terminal_status, answer, ...}]

        ↓ [Phase 3.7] necessity_check  ← JOIN of synthesis chain + science chain
  + necessity_assessment: '{"differentiation":"high","portfolio_relationship":"complementary",...}'

        ↓ [Phase 3.8] project_scope_decisions (fan-out via dispatch_decision_projections)
  + decisions: [{inv_id, decision_type, recommended_action, goal_link,
                 substitution_path, triggering_link_ids, rank_score, ...}]
```

---

## Key invariants

1. **necessity_assessment must be set before dispatch_decision_projections fires.**
   `necessity_check` is the only writer; `dispatch_decision_projections` is the
   only router that reads it. The graph edge `necessity_check → dispatch_decision_projections`
   guarantees ordering.

2. **Empty necessity_assessment is valid.** If `link_assessments` is empty (e.g.,
   extraction failed), `necessity_check` sets `scope["necessity_assessment"] = ""`
   and `_build_decisions_prompt` renders "(no necessity assessment available)" for §2.

3. **Citation rule.** `_coerce_necessity_payload` enforces that any named external
   program in the necessity assessment must have a source URL. Unsourced claims are
   cleared before writing to state. This prevents hallucinated program names from
   flowing into the decision prompt's §2.

4. **Financial data in §3 comes from causal_model.links, not link_assessment dicts.**
   The new repo's `InvestigationResult` schema does not carry `dollars_at_risk` /
   `months_at_risk` — those are set by `forecast_consequences` on `causal_model.links`
   and looked up by link name at prompt-construction time.

5. **Decisions are LLM-sourced for corroboration_count and dollars.**
   Unlike the old repo (which resolved dollars directly from `LinkAssessment`
   dataclass fields via `_resolve_cited_links`), the new repo relies on the LLM
   to estimate these from the evidence presented in §3. The financial lookup from
   causal_model.links provides the ground-truth numbers in the prompt so the LLM
   can cite them accurately.
