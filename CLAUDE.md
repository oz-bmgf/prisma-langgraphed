# Project rules for the LangGraph port verification

- SCOPE: only the analyze pipeline (OLD entry point `cli.py`) and its use of the
  collection API's search functionality. Ignore everything else in both repos.
- The OLD repo at `[prisma-langgraphed]` holds the source-of-truth CODE. The markdown
  docs `CODE_AUDIT.md`, `ARCHITECTURE.md`, and `AGENTS.md` live in THIS (NEW)
  repo `[prisma-ai-review]`. The LangGraph implementation is unverified until proven equivalent.
- All generated docs live in `/verification`. Always update the relevant doc
  (MAPPING.md, FINDINGS.md, etc.) as part of any task.

- **Use LangGraph-native constructs for EVERY change in this repo.** Before
  making any change, consult the project skills (the SKILL.md files added to
  this project) and implement using idiomatic LangGraph APIs — e.g. `StateGraph`,
  typed state with the correct reducers (`add_messages`, etc.), `@tool` /
  `ToolNode`, conditional edges and `Command` for routing, and prebuilt
  components where they fit — rather than hand-rolled Python control flow or
  ad-hoc API-call wrappers. If a change cannot be expressed natively, stop and
  flag it in FINDINGS.md instead of working around it.

- During VERIFICATION tasks: never modify application code. Only read and report.
- During FIX tasks: change only what's needed to satisfy the contract in
  CONTRACTS.md. No opportunistic refactors. Always show the diff first.
- Never change a prompt string, model name, or model parameter unless a finding
  explicitly calls for it.
- Work one node or one tool per task. Do not batch.
- Compare execution by trace (node/tool order, final state shape, key values),
  never by raw LLM text.