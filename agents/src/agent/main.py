#!/usr/bin/env python3
"""
main.py — CLI entry point for the Bitext Customer-Support ReAct Agent.

Usage:
    python main.py                          # uses bundled sample dataset
    python main.py --dataset path/to/data.csv
    python main.py --dataset path/to/data.csv --query "How many refund requests?"

The agent supports three query types:
  • Structured   — "What categories exist?" / "How many refund requests?"
  • Unstructured — "Summarise the FEEDBACK category."
  • Out-of-scope — politely declined (e.g. "Who won the 2024 Champions League?")
"""
from __future__ import annotations

import argparse
import os
import sys
import textwrap
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ── Resolve bundled dataset path ────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
DEFAULT_DATASET = SCRIPT_DIR / "bitext_full.csv"

# ── Colour helpers (graceful fallback if terminal has no colour) ─────────────
try:
    import shutil
    _COLOUR = shutil.get_terminal_size().columns > 0 and sys.stdout.isatty()
except Exception:
    _COLOUR = False

def _c(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _COLOUR else text

def bold(t: str)   -> str: return _c(t, "1")
def cyan(t: str)   -> str: return _c(t, "36")
def green(t: str)  -> str: return _c(t, "32")
def yellow(t: str) -> str: return _c(t, "33")
def red(t: str)    -> str: return _c(t, "31")
def dim(t: str)    -> str: return _c(t, "2")


BANNER = r"""
  ╔══════════════════════════════════════════════════════════╗
  ║   🤖  Bitext Customer-Support ReAct Agent               ║
  ║   Powered by LangGraph-style reasoning + Anthropic API  ║
  ╚══════════════════════════════════════════════════════════╝
"""

HELP_TEXT = """
Ask questions about the Bitext customer-support dataset. Examples:

  Structured queries:
    • What categories exist in the dataset?
    • How many refund requests did we get?
    • Show me 3 examples from the SHIPPING intent.
    • What is the distribution of intents in the ACCOUNT category?

  Unstructured queries:
    • Summarise the FEEDBACK category.
    • How do agents typically respond to cancellation requests?

  Out-of-scope (will be politely declined):
    • Who won the 2024 Champions League?
    • Write me a poem about customer service.

Commands:  help | quit | exit
"""


def print_trace(trace_list):
    # Ensure trace_list exists and is iterable
    if not trace_list:
        return

    for step in trace_list:
        # If the step is a LangChain message object, extract its content text
        if hasattr(step, "content"):
            step_text = f"[{step.type.upper()}]: {step.content}"
        elif isinstance(step, dict):
            step_text = step.get("content", str(step))
        else:
            step_text = str(step)

        # Line 82: Now we pass a guaranteed string to your dim formatter
        print(dim("  " + step_text))


def print_answer(answer: str) -> None:
    print()
    print(bold(green("  Answer:")))
    wrapped = textwrap.fill(answer, width=72, initial_indent="  ", subsequent_indent="  ")
    print(green(wrapped))
    print()


def run_interactive(dataset_path: str, session:str) -> None:
    # ── Import here so errors surface clearly ───────────────────────────────
    from tools import load_dataset
    from agent import run_graph

    # Load dataset once
    print(dim(f"  Loading dataset from {dataset_path}..."))
    try:
        df = load_dataset(dataset_path)
        print(dim(f"  ✓ Loaded {len(df):,} rows | "
                  f"{df['category'].nunique()} categories | "
                  f"{df['intent'].nunique()} intents"))
    except Exception as e:
        print(red(f"  ✗ Failed to load dataset: {e}"))
        sys.exit(1)

    print(BANNER)
    print(HELP_TEXT)

    while True:
        try:
            query = input(bold(cyan("You: "))).strip()
        except (EOFError, KeyboardInterrupt):
            print("\n" + dim("  Goodbye!"))
            break

        if not query:
            continue

        if query.lower() in {"quit", "exit", "q"}:
            print(dim("  Goodbye!"))
            break

        if query.lower() in {"help", "?"}:
            print(HELP_TEXT)
            continue

        print(dim(f"\n  [Processing …]"))

        try:
            state = run_graph(query, session_id=session)
        except Exception as e:
            print(red(f"\n  ✗ Agent error: {e}"))
            import traceback
            traceback.print_exc()
            continue

        # ── Print reasoning trace ────────────────────────────────────────────
        print_trace(state["trace"])

        # ── Print final answer ───────────────────────────────────────────────
        print_answer(state.get("final_answer", ""))


def run_single_query(dataset_path: str, query: str, session_id: str) -> None:
    from tools import load_dataset
    from agent import run_graph

    load_dataset(dataset_path)
    state = run_graph(query, session_id=session_id)
    print_trace(state["trace"])
    print_answer(state["final_answer"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bitext Customer-Support ReAct Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dataset",
        default=str(DEFAULT_DATASET),
        help="Path to the Bitext CSV dataset (default: bundled sample)",
    )
    parser.add_argument(
        "--query", "-q",
        default=None,
        help="Run a single query and exit (non-interactive mode)",
    )
    parser.add_argument(
        "--no-colour",
        action="store_true",
        help="Disable ANSI colour output",
    )

    parser.add_argument(
        "--session",
        default="default_session",
        help="Unique session ID thread token to persist/restore conversation state across turns",
    )

    args = parser.parse_args()

    if args.no_colour:
        global _COLOUR
        _COLOUR = False

    # ── API key check ────────────────────────────────────────────────────────
    if not os.environ.get("NEBIUS_API_KEY"):
        print(red("  ERROR: NEBIUS_API_KEY is not set."))
        print(dim("  Export it first:  export NEBIUS_API_KEY=v1. ..."))
        sys.exit(1)

    if args.query:
        run_single_query(args.dataset, args.query, args.session)
    else:
        run_interactive(args.dataset, args.session)


if __name__ == "__main__":
    main()
