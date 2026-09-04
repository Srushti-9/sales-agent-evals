"""CLI entry point: ``sales-agent ask | eval | experiment``.

Each subcommand sets up Phoenix tracing first (launching a local app if no
collector is reachable), so every run is observable in the Phoenix UI.
"""

from __future__ import annotations

import argparse
import sys


def _cmd_ask(args: argparse.Namespace) -> int:
    from .agent import ask
    from .tracing import setup_tracing

    _, session_url = setup_tracing()
    if session_url:
        print(f"Phoenix UI: {session_url}", file=sys.stderr)

    print(ask(args.question))
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    from .tracing import setup_tracing
    from . import evals

    setup_tracing()
    runners = {
        "tool-calling": evals.evaluate_tool_calling,
        "sql": evals.evaluate_sql_generation,
        "clarity": evals.evaluate_response_clarity,
        "code": evals.evaluate_generated_code,
    }
    selected = runners if args.which == "all" else {args.which: runners[args.which]}
    for name, fn in selected.items():
        result = fn()
        n = 0 if result is None else len(result)
        print(f"{name}: evaluated {n} span(s)")
    return 0


def _cmd_experiment(args: argparse.Namespace) -> int:
    from .tracing import setup_tracing
    from .experiments import run_convergence_experiment

    _, session_url = setup_tracing()
    if session_url:
        print(f"Phoenix UI: {session_url}", file=sys.stderr)

    experiment = run_convergence_experiment()
    print(experiment.as_dataframe().to_string())
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sales-agent")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ask = sub.add_parser("ask", help="Ask the agent a question")
    p_ask.add_argument("question", help="The question to ask")
    p_ask.set_defaults(func=_cmd_ask)

    p_eval = sub.add_parser("eval", help="Run evaluators over traced spans")
    p_eval.add_argument(
        "which",
        nargs="?",
        default="all",
        choices=["all", "tool-calling", "sql", "clarity", "code"],
    )
    p_eval.set_defaults(func=_cmd_eval)

    p_exp = sub.add_parser("experiment", help="Run the convergence experiment")
    p_exp.set_defaults(func=_cmd_experiment)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    raise SystemExit(args.func(args))
