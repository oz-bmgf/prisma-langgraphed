"""NQPR pipeline configuration — single source of truth for all runtime constants.

Every constant reads from an environment variable with a sensible default.
Import from here; never hardcode model names, paths, limits, or thresholds
in node or core files (AGENTS.md §4).

Dotenv is loaded here at import time so any module that imports src.config
(directly or transitively) picks up .env values without a separate load_dotenv
call. override=False means existing shell env vars take precedence over .env.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the repository root (two levels up: src/config.py → src/ → repo root).
# override=False: shell env vars beat .env — safe for CI/CD and unit tests.
_ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(_ENV_PATH, override=False)

# ── Models ────────────────────────────────────────────────────────────────────

DEFAULT_RESEARCH_MODEL: str = os.getenv("NQPR_RESEARCH_MODEL", "claude-sonnet-4-6")
DEFAULT_SYNTHESIS_MODEL: str = os.getenv("NQPR_SYNTHESIS_MODEL", "claude-sonnet-4-6")
DEFAULT_ANALYSIS_MODEL: str = os.getenv("NQPR_ANALYSIS_MODEL", "claude-sonnet-4-6")
DEFAULT_FAST_MODEL: str = os.getenv("NQPR_FAST_MODEL", "claude-haiku-4-5-20251001")
DEFAULT_STRONG_MODEL: str = os.getenv("NQPR_STRONG_MODEL", "claude-opus-4-7")
DEFAULT_VISION_MODEL: str = os.getenv("NQPR_VISION_MODEL", "gpt-4o")
# code_interpreter runs on the OpenAI Responses API — must be an OpenAI model.
# Never set this to an Anthropic model; Claude does not support code_interpreter containers.
COMPUTE_MODEL: str = os.getenv("NQPR_COMPUTE_MODEL", "gpt-4o-mini")
DEFAULT_EMBEDDING_MODEL: str = os.getenv("NQPR_EMBEDDING_MODEL", "text-embedding-3-small")

# ── Model parameters ──────────────────────────────────────────────────────────

DEFAULT_MAX_TOKENS: int = int(os.getenv("NQPR_MAX_TOKENS", "4096"))

# Causal extraction needs a much larger context window than normal calls.
# Old repo used max_tokens=32000 for extract_causal_model to accommodate complex
# theory-of-change chains across large evidence packs.
CAUSAL_MAX_TOKENS: int = int(os.getenv("NQPR_CAUSAL_MAX_TOKENS", "32000"))
THINKING_BUDGET_TOKENS: int = int(os.getenv("NQPR_THINKING_BUDGET_TOKENS", "8000"))
# Minimum max_tokens when thinking mode is active (must exceed budget_tokens)
THINKING_MIN_MAX_TOKENS: int = 16000
VERIFIER_MAX_TOKENS: int = int(os.getenv("NQPR_VERIFIER_MAX_TOKENS", "2048"))
DEFAULT_TEMPERATURE: float = float(os.getenv("NQPR_DEFAULT_TEMPERATURE", "0.0"))
MAX_INVESTIGATION_ITERATIONS: int = int(os.getenv("NQPR_MAX_INVESTIGATION_ITERATIONS", "40"))
MAX_LINK_WORKERS: int = int(os.getenv("NQPR_MAX_LINK_WORKERS", "16"))

# ── Checkpointer ──────────────────────────────────────────────────────────────

# Backend: "sqlite" (default, local dev) or "postgres"
CHECKPOINTER_BACKEND: str = os.getenv("NQPR_CHECKPOINTER_BACKEND", "sqlite")

# SQLite path (used when backend = "sqlite")
CHECKPOINT_DB_PATH: Path = Path(
    os.getenv("NQPR_CHECKPOINT_DB", "~/.nqpr_checkpoints/checkpoints.db")
).expanduser()

# PostgreSQL DSN (used when backend = "postgres")
# Format: postgresql://user:password@host:port/dbname
CHECKPOINT_POSTGRES_DSN: str = os.getenv("NQPR_CHECKPOINT_POSTGRES_DSN", "")

# PostgreSQL connection pool size
CHECKPOINT_PG_MAX_CONNECTIONS: int = int(
    os.getenv("NQPR_CHECKPOINT_PG_MAX_CONNECTIONS", "10")
)

# Whether to pause at human interrupt nodes.
# Default False — pipeline runs unattended end to end.
# Set NQPR_HUMAN_INTERRUPTS=true only for interactive CLI sessions.
CHECKPOINT_HUMAN_INTERRUPTS: bool = (
    os.getenv("NQPR_HUMAN_INTERRUPTS", "false").lower() == "true"
)

# ── Paths ─────────────────────────────────────────────────────────────────────

COLLECTIONS_BASE_PATH: Path = Path(
    os.getenv("NQPR_COLLECTIONS_BASE", "~/qpr-collections")
).expanduser()

# ── Search and retrieval ──────────────────────────────────────────────────────

SEARCH_BACKEND: str = os.getenv("NQPR_SEARCH_BACKEND", "local")
EMBED_DIM: int = int(os.getenv("NQPR_EMBED_DIM", "1536"))
TOP_K_DEFAULT: int = int(os.getenv("NQPR_TOP_K_DEFAULT", "20"))
TOP_K_INVESTMENT_SEARCH: int = int(os.getenv("NQPR_TOP_K_INVESTMENT_SEARCH", "25"))
TOP_K_PORTFOLIO_SEARCH: int = int(os.getenv("NQPR_TOP_K_PORTFOLIO_SEARCH", "10"))

# Rubric evaluator evidence-pack budget (separate from narration/investigation top_k).
# OLD default: top_k=200 per S1 query, giving inv_budget=180 after strategy floor.
# Must be >> _MIN_STRATEGY_CHUNKS (20) so investment queries get meaningful top_k.
RUBRIC_TOP_K: int = int(os.getenv("NQPR_RUBRIC_TOP_K", "200"))

# ── Pipeline limits ───────────────────────────────────────────────────────────

MAX_CONCURRENCY: int = int(os.getenv("NQPR_MAX_CONCURRENCY", "16"))

# ── Thresholds ────────────────────────────────────────────────────────────────

SIMILARITY_THRESHOLD: float = float(os.getenv("NQPR_SIMILARITY_THRESHOLD", "0.85"))
CHUNK_TARGET_SIZE: int = int(os.getenv("NQPR_CHUNK_TARGET_SIZE", "1000"))
CHUNK_MAX_SIZE: int = int(os.getenv("NQPR_CHUNK_MAX_SIZE", "2000"))
CONTEXT_BUDGET_CHARS: int = int(os.getenv("NQPR_CONTEXT_BUDGET_CHARS", "420000"))
RECENCY_BOOST_PER_YEAR: float = float(os.getenv("NQPR_RECENCY_BOOST_PER_YEAR", "0.00005"))
RECENCY_BASELINE_YEAR: int = int(os.getenv("NQPR_RECENCY_BASELINE_YEAR", "2020"))

# ── Deep Web research ─────────────────────────────────────────────────────────

DEEP_WEB_PRIMARY_MODEL: str = os.getenv("NQPR_DEEP_WEB_MODEL", "o3-deep-research")
DEEP_WEB_FALLBACK_MODEL: str = os.getenv("NQPR_DEEP_WEB_FALLBACK_MODEL", "gpt-4o")
DEEP_WEB_TIMEOUT_SECONDS: int = int(os.getenv("NQPR_DEEP_WEB_TIMEOUT", "300"))
DEEP_WEB_MAX_ROUNDS: int = int(os.getenv("NQPR_DEEP_WEB_MAX_ROUNDS", "3"))

# ── Edison literature retrieval ────────────────────────────────────────────────

EDISON_API_KEY: str = os.getenv("EDISON_PLATFORM_API_KEY", "")
EDISON_TIMEOUT_SECONDS: int = int(os.getenv("NQPR_EDISON_TIMEOUT", "2400"))
EDISON_MAX_CONCURRENT: int = int(os.getenv("NQPR_EDISON_MAX_CONCURRENT", "4"))

# ── SLR / LBD timeouts ────────────────────────────────────────────────────────

SLR_TIMEOUT_SECONDS: int = int(os.getenv("NQPR_SLR_TIMEOUT", "600"))
LBD_TIMEOUT_SECONDS: int = int(os.getenv("NQPR_LBD_TIMEOUT", "600"))

# ── Asta (Semantic Scholar) ───────────────────────────────────────────────────

ASTA_API_KEY: str = os.getenv("ASTA_API_KEY", "")
ASTA_ENDPOINT: str = os.getenv("ASTA_ENDPOINT", "https://asta-tools.allen.ai/mcp/v1")
ASTA_TIMEOUT_SECONDS: int = int(os.getenv("ASTA_TIMEOUT", "90"))
ASTA_MAX_RETRIES: int = int(os.getenv("ASTA_MAX_RETRIES", "4"))
ASTA_RETRY_BASE_DELAY: float = float(os.getenv("ASTA_RETRY_BASE_DELAY", "1.0"))
SEMANTIC_SCHOLAR_API_KEY: str = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")

# ── OpenAlex ──────────────────────────────────────────────────────────────────

OPENALEX_EMAIL: str = os.getenv("OPENALEX_EMAIL", "")
OPENALEX_MAX_RESULTS: int = int(os.getenv("NQPR_OPENALEX_MAX_RESULTS", "50"))

# ── Investigation levers (env-gated) ─────────────────────────────────────────

# L4: coverage audit — forces the model to address 5 checklist items before
# submitting findings. Set NQPR_L4_COVERAGE_AUDIT=true to enable.
INVESTIGATION_L4_COVERAGE_AUDIT: bool = (
    os.getenv("NQPR_L4_COVERAGE_AUDIT", "false").lower() == "true"
)

# L1: elevated reasoning effort for investigation models
INVESTIGATION_L1_REASONING: bool = (
    os.getenv("NQPR_L1_REASONING", "false").lower() == "true"
)

# ── Science investigation ─────────────────────────────────────────────────────

# Max ASTA (Semantic Scholar) calls per science question before soft-cap kicks in.
ASTA_SOFT_CAP: int = int(os.getenv("NQPR_ASTA_SOFT_CAP", "5"))
SCIENCE_MAX_ITERATIONS: int = int(os.getenv("NQPR_SCIENCE_MAX_ITERATIONS", "8"))

# Consecutive rounds with zero new chunks before forcing insufficient_evidence.
CONSECUTIVE_EMPTY_THRESHOLD: int = int(os.getenv("NQPR_CONSECUTIVE_EMPTY_THRESHOLD", "3"))

# ── Decision projection ───────────────────────────────────────────────────────

# Max decisions per INV within a scope (inv_id="" decisions do not count).
DECISION_MAX_PER_INV: int = int(os.getenv("NQPR_DECISION_MAX_PER_INV", "3"))

# Max decisions per scope across all INVs.
DECISION_MAX_PER_SCOPE: int = int(os.getenv("NQPR_DECISION_MAX_PER_SCOPE", "8"))

# ── Orientation (Phase 1) ────────────────────────────────────────────────────

# Max tokens for the orientation LLM response (old repo enforced 8000).
ORIENTATION_MAX_TOKENS: int = int(os.getenv("NQPR_ORIENTATION_MAX_TOKENS", "8000"))

# ── Report assembly (Phase 6b) ────────────────────────────────────────────────

# Max retries on report structure validation failure (old repo: 5).
ASSEMBLY_MAX_RETRIES: int = int(os.getenv("NQPR_ASSEMBLY_MAX_RETRIES", "5"))

# ── NarrationToolbox call budget ─────────────────────────────────────────────

# Max tool calls per narrator session (old repo default: 15).
# When budget is exhausted, tools return "[budget exhausted — wrap up now]".
NARRATOR_CALL_BUDGET: int = int(os.getenv("NQPR_NARRATOR_CALL_BUDGET", "15"))

# ── Investigation — OpenAI Responses API stateful path (Orphan 10) ────────────

# When true AND the investigation model is an OpenAI model, use the Responses
# API with previous_response_id for stateful conversation continuity.
# Matches old investigation_loop._run_openai_loop() pattern.
INVESTIGATION_USE_RESPONSES_API: bool = (
    os.getenv("NQPR_USE_RESPONSES_API", "true").lower() == "true"
)
