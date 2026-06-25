#!/usr/bin/env python3
"""CLI entry point for llm-cost-calc."""

import argparse
import sys
from typing import List, Optional

from llm_cost_calc.calculator import cost, compare, UnknownModelError
from llm_cost_calc.prices import known_models


def _format_result(result) -> str:
    """Pretty-print a CostResult (or an error line)."""
    lines = [
        f"  Model       : {result.model}",
        f"  Input tokens: {result.input_tokens:,}  @ ${result.input_price:.2f}/1M",
        f"  Output tokens: {result.output_tokens:,}  @ ${result.output_price:.2f}/1M",
        f"  Input cost  : ${result.input_cost:.6f}",
        f"  Output cost : ${result.output_cost:.6f}",
        f"  ─────────────────────────────",
        f"  TOTAL COST  : ${result.total_cost:.6f}",
    ]
    return "\n".join(lines)


def cmd_cost(args: argparse.Namespace) -> int:
    """Handle `cost` subcommand."""
    try:
        result = cost(args.model, args.input, args.output, known_only=True)
        print(f"\n{_format_result(result)}\n")
    except UnknownModelError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    """Handle `compare` subcommand."""
    try:
        r_a, r_b, ratio = compare(args.model_a, args.model_b,
                                  args.input, args.output, known_only=True)
    except UnknownModelError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"\n{'─'*40}")
    print(f"  MODEL A  ─  {r_a.model}")
    print(f"{'─'*40}")
    print(f"  Input tokens : {r_a.input_tokens:,}  @ ${r_a.input_price:.2f}/1M")
    print(f"  Output tokens: {r_a.output_tokens:,}  @ ${r_a.output_price:.2f}/1M")
    print(f"  TOTAL        : ${r_a.total_cost:.6f}")
    print(f"\n{'─'*40}")
    print(f"  MODEL B  ─  {r_b.model}")
    print(f"{'─'*40}")
    print(f"  Input tokens : {r_a.input_tokens:,}  @ ${r_b.input_price:.2f}/1M")
    print(f"  Output tokens: {r_a.output_tokens:,}  @ ${r_b.output_price:.2f}/1M")
    print(f"  TOTAL        : ${r_b.total_cost:.6f}")
    print(f"\n  Model A costs {ratio:.2f}× Model B")
    print(f"  Difference   : ${abs(r_a.total_cost - r_b.total_cost):.6f}\n")
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    """Handle `list` subcommand."""
    models = known_models()
    col = args.columns or 3
    # Pad to fill last row
    rows_needed = -(-len(models) // col) * col
    padded = models + [""] * (rows_needed - len(models))
    for row in range(0, rows_needed, col):
        print("  " + "  ".join(f"{m:<38}" for m in padded[row:row + col]))
    print(f"\n({len(models)} models)\n")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llm-cost-calc",
        description="Compute and compare LLM API costs from token counts.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # cost
    p_cost = sub.add_parser("cost", help="Calculate cost for one model")
    p_cost.add_argument("model", help="Model name (e.g. gpt-4o)")
    p_cost.add_argument("input", type=int, help="Number of input tokens")
    p_cost.add_argument("output", type=int, help="Number of output tokens")

    # compare
    p_cmp = sub.add_parser("compare", help="Compare costs of two models")
    p_cmp.add_argument("model_a", help="First model name")
    p_cmp.add_argument("model_b", help="Second model name")
    p_cmp.add_argument("input", type=int, help="Number of input tokens")
    p_cmp.add_argument("output", type=int, help="Number of output tokens")

    # list
    p_lst = sub.add_parser("list", help="List all known models")
    p_lst.add_argument(
        "--columns", "-n", type=int, default=3,
        help="Number of columns (default: 3)",
    )

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "cost":
        return cmd_cost(args)
    elif args.command == "compare":
        return cmd_compare(args)
    elif args.command == "list":
        return cmd_list(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
