"""NQPR pipeline CLI.

Subcommands:
  run     — start a new pipeline run
  resume  — resume from the last checkpoint
  status  — show checkpoint history for a run

Usage:
  python main.py run --program MOCK --run-name test-01 --base-dir ~/qpr-collections
  python main.py run --program MOCK --run-name test-01 --interactive
  python main.py resume --program MOCK --run-name test-01
  python main.py status --program MOCK --run-name test-01
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Env setup (load .env before any src imports)
# ---------------------------------------------------------------------------

# Safety fallback — src.config loads .env at import time; this handles direct
# script invocation before any src import has triggered that load.
from dotenv import load_dotenv

_env_path = Path(__file__).resolve().parent / ".env"
if _env_path.exists():
    load_dotenv(_env_path, override=False)

# Initialise OTEL before any src imports so LangChain/OpenAI/Anthropic
# instrumentors are registered before the first SDK call.
from observability.tracing import init_tracing as _init_tracing
_init_tracing()

from src.config import (
    CHECKPOINTER_BACKEND,
    CHECKPOINT_DB_PATH,
    CHECKPOINT_POSTGRES_DSN,
    COLLECTIONS_BASE_PATH,
    DEFAULT_RESEARCH_MODEL,
    DEFAULT_SYNTHESIS_MODEL,
)
from src.core.checkpointer import (
    build_checkpointer,
    get_checkpoint_state,
    list_checkpoints,
)
from src.graph.workflow import compile_graph, create_initial_state, make_thread_id


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _print_progress(event: dict) -> None:
    stage = event.get("current_stage", "?")
    errors = event.get("errors") or []
    print(f"  [{stage}] errors={len(errors)}")


def _build_backend(state: dict):
    """Rebuild the search backend from a checkpoint state dict.

    Called on resume when the original run config is no longer in scope.
    """
    from src.backends.factory import build_search_backend
    return build_search_backend(
        state.get("ingested_dir", ""),
        state.get("collection_name", ""),
    )


def _checkpoint_location() -> str:
    if CHECKPOINTER_BACKEND == "postgres":
        return CHECKPOINT_POSTGRES_DSN or "(DSN not set)"
    return str(CHECKPOINT_DB_PATH)


# ---------------------------------------------------------------------------
# run subcommand
# ---------------------------------------------------------------------------


async def cmd_run(args: argparse.Namespace) -> None:
    base_dir = str(Path(args.base_dir).expanduser())
    ingested_dir = args.ingested_dir or str(Path(base_dir) / f"{args.program}-ingested")
    collection_name = args.collection_name or args.program.lower()

    initial_state = create_initial_state(
        program=args.program,
        run_name=args.run_name,
        collection_name=collection_name,
        base_dir=base_dir,
        ingested_dir=ingested_dir,
        research_model=args.research_model,
        synthesis_model=args.synthesis_model,
        focus=args.focus or None,
    )
    thread_id = make_thread_id(args.program, args.run_name)

    print(f"Starting run: program={args.program} run_name={args.run_name}")
    print(f"  thread_id  : {thread_id}")
    print(f"  backend    : {CHECKPOINTER_BACKEND}")
    print(f"  checkpoint : {_checkpoint_location()}")
    print(f"  interactive: {args.interactive}")

    from src.backends.factory import build_search_backend
    search_backend = build_search_backend(ingested_dir, collection_name)

    config = {
        "configurable": {
            "thread_id": thread_id,
            "search_backend": search_backend,
            "pages_dir": str(Path(ingested_dir) / "pages"),
            # Run identity — mirrors state fields; nodes can read from either
            "program": args.program,
            "run_name": args.run_name,
            "collection_name": collection_name,
            "base_dir": base_dir,
            "ingested_dir": ingested_dir,
            "research_model": args.research_model,
            "synthesis_model": args.synthesis_model,
            "focus": args.focus or None,
        }
    }

    async with build_checkpointer() as checkpointer:
        graph = compile_graph(checkpointer, human_interrupts=args.interactive)
        last_event: dict = {}
        async for event in graph.astream(initial_state, config=config, stream_mode="values"):
            _print_progress(event)
            last_event = event

    errors = last_event.get("errors") or []
    print(f"\nRun finished. Errors: {len(errors)}")
    for e in errors[:5]:
        print(f"  ! {e}")


# ---------------------------------------------------------------------------
# resume subcommand
# ---------------------------------------------------------------------------


async def cmd_resume(args: argparse.Namespace) -> None:
    thread_id = make_thread_id(args.program, args.run_name)

    async with build_checkpointer() as checkpointer:
        state = await get_checkpoint_state(thread_id, checkpointer)
        if state is None:
            print(f"No checkpoint found for thread_id={thread_id}")
            print("Use 'main.py run' to start a new run.")
            return

        print(f"Resuming: program={args.program} run_name={args.run_name}")
        print(f"  thread_id  : {thread_id}")
        print(f"  backend    : {CHECKPOINTER_BACKEND}")
        print(f"  checkpoint : {_checkpoint_location()}")
        print(f"  last stage : {state.get('current_stage', 'unknown')}")
        print(f"  interactive: {args.interactive}")

        search_backend = _build_backend(state)
        ingested_dir = state.get("ingested_dir", "")
        config = {
            "configurable": {
                "thread_id": thread_id,
                "search_backend": search_backend,
                "pages_dir": str(Path(ingested_dir) / "pages") if ingested_dir else "",
                # Run identity restored from checkpoint state
                "program": state.get("program", args.program),
                "run_name": state.get("run_name", args.run_name),
                "collection_name": state.get("collection_name", ""),
                "base_dir": state.get("base_dir", ""),
                "ingested_dir": ingested_dir,
                "research_model": state.get("research_model", DEFAULT_RESEARCH_MODEL),
                "synthesis_model": state.get("synthesis_model", DEFAULT_SYNTHESIS_MODEL),
                "focus": state.get("focus"),
            }
        }

        graph = compile_graph(checkpointer, human_interrupts=args.interactive)
        last_event: dict = {}
        # Pass None as input — LangGraph restores state from the checkpoint automatically
        async for event in graph.astream(None, config=config, stream_mode="values"):
            _print_progress(event)
            last_event = event

    errors = last_event.get("errors") or []
    print(f"\nRun finished. Errors: {len(errors)}")
    for e in errors[:5]:
        print(f"  ! {e}")


# ---------------------------------------------------------------------------
# status subcommand
# ---------------------------------------------------------------------------


async def cmd_status(args: argparse.Namespace) -> None:
    thread_id = make_thread_id(args.program, args.run_name)

    async with build_checkpointer() as checkpointer:
        state = await get_checkpoint_state(thread_id, checkpointer)
        if state is None:
            print(f"No checkpoint found for thread_id={thread_id}")
            return

        checkpoints = await list_checkpoints(thread_id, checkpointer)

    print(f"\nRun: {thread_id}")
    print(f"Backend    : {CHECKPOINTER_BACKEND}")
    print(f"Checkpoint : {_checkpoint_location()}")

    print(f"\nCheckpoint history ({len(checkpoints)} snapshots):")
    for cp in checkpoints[:10]:
        print(
            f"  step {cp['step']:3d}  "
            f"{cp['node_name']:<30}  "
            f"{cp['created_at']}"
        )
    if len(checkpoints) > 10:
        print(f"  ... and {len(checkpoints) - 10} more")

    print(f"\nCurrent state summary:")
    print(f"  current_stage  : {state.get('current_stage', 'N/A')}")
    print(f"  precheck_passed: {state.get('precheck_passed')}")
    scopes = state.get("scopes") or []
    print(f"  scopes         : {len(scopes)}")
    print(f"  analyst_report : {'yes' if state.get('analyst_report') else 'no'}")
    research_plan = state.get("research_plan") or []
    print(f"  research_plan  : {len(research_plan)} tasks")
    print(f"  final_report   : {'yes' if state.get('final_report_md') else 'no'}")
    errors = state.get("errors") or []
    print(f"  errors         : {len(errors)}")

    # Tool trace summary
    try:
        from src.core.tool_tracing import summarise_traces
        trace_summary = summarise_traces(state)
        if trace_summary:
            print("\n  Tool call summary:")
            label_map = {
                "asta": "ASTA searches",
                "slr": "SLR tasks",
                "lbd": "LBD tasks",
                "deep_web": "Deep web tasks",
                "edison": "Edison tasks",
                "web_search": "Web searches",
                "compute": "Code interpreter",
                "collection_search": "Collection searches",
                "investigation": "Investigation loops",
            }
            for key, info in trace_summary.items():
                label = label_map.get(key, key)
                count = info["count"]
                avg_ms = info["avg_duration_ms"]
                errs = info["error_count"]
                if key == "investigation" and "avg_iterations" in info:
                    extra = f" (avg {info['avg_iterations']} iters each)"
                else:
                    extra = f" (avg {avg_ms}ms, {errs} errors)"
                print(f"    {label:<26}: {count:>4}{extra}")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="NQPR pipeline CLI",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # run
    p_run = sub.add_parser("run", help="Start a new pipeline run")
    p_run.add_argument("--program", required=True)
    p_run.add_argument("--run-name", required=True, dest="run_name")
    p_run.add_argument("--base-dir", default=str(COLLECTIONS_BASE_PATH), dest="base_dir")
    p_run.add_argument("--ingested-dir", default=None, dest="ingested_dir")
    p_run.add_argument("--collection-name", default=None, dest="collection_name")
    p_run.add_argument("--research-model", default=DEFAULT_RESEARCH_MODEL, dest="research_model")
    p_run.add_argument("--synthesis-model", default=DEFAULT_SYNTHESIS_MODEL, dest="synthesis_model")
    p_run.add_argument("--focus", default=None)
    p_run.add_argument(
        "--interactive",
        action="store_true",
        default=False,
        help=(
            "Pause at human review nodes (analyze, review_research_plan, "
            "research, approve_report). Default: run unattended."
        ),
    )

    # resume
    p_resume = sub.add_parser("resume", help="Resume from the last checkpoint")
    p_resume.add_argument("--program", required=True)
    p_resume.add_argument("--run-name", required=True, dest="run_name")
    p_resume.add_argument(
        "--interactive",
        action="store_true",
        default=False,
        help="Pause at human review nodes on resume.",
    )

    # status
    p_status = sub.add_parser("status", help="Show checkpoint history for a run")
    p_status.add_argument("--program", required=True)
    p_status.add_argument("--run-name", required=True, dest="run_name")

    return parser


def main() -> None:
    parser = _make_parser()
    args = parser.parse_args()

    dispatch = {
        "run": cmd_run,
        "resume": cmd_resume,
        "status": cmd_status,
    }
    handler = dispatch.get(args.command)
    if handler is None:
        parser.print_help()
        sys.exit(1)

    # asyncio-APPROVED-4: asyncio.run in CLI entry point — top-level event loop
    asyncio.run(handler(args))


if __name__ == "__main__":
    main()
