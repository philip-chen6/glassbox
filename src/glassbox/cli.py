from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .model_runner import build_report, run_forward


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="glassbox",
        description="Run a prompt through a transformer and return internal activation summaries.",
    )
    parser.add_argument(
        "prompt",
        nargs="*",
        help="Prompt text. If omitted, --prompt must be provided.",
    )
    parser.add_argument(
        "--prompt",
        dest="prompt_flag",
        help="Prompt text alternative to positional args.",
    )
    parser.add_argument(
        "--model",
        default="distilgpt2",
        help="Hugging Face model name to inspect (default: distilgpt2).",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help='Device to run on: "auto", "cpu", "cuda", or "mps".',
    )
    parser.add_argument(
        "--use-toy",
        action="store_true",
        help="Force the built-in toy transformer instead of a Hugging Face model.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to write JSON output.",
    )
    parser.add_argument(
        "--include-hidden",
        action="store_true",
        help="Include raw hidden state tensors in output (large).",
    )
    parser.add_argument(
        "--include-attention",
        action="store_true",
        help="Include raw attention tensors in output (large).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=24,
        help="Number of tokens to generate for answer text (default: 24).",
    )
    return parser.parse_args(argv)


def resolve_prompt(args: argparse.Namespace) -> str:
    if args.prompt_flag:
        return args.prompt_flag
    if args.prompt:
        return " ".join(args.prompt).strip()
    raise ValueError("No prompt provided. Pass positional text or --prompt.")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.max_new_tokens < 0:
        print("--max-new-tokens must be >= 0", file=sys.stderr)
        return 2
    try:
        prompt = resolve_prompt(args)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    artifacts, warning = run_forward(
        prompt=prompt,
        model_name=args.model,
        device=args.device,
        use_toy=args.use_toy,
        max_new_tokens=args.max_new_tokens,
    )
    report = build_report(
        prompt=prompt,
        artifacts=artifacts,
        warning=warning,
        include_hidden=args.include_hidden,
        include_attention=args.include_attention,
    )
    rendered = json.dumps(report, indent=2)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print(f"Wrote report to {args.output}")
    else:
        try:
            print(rendered)
        except BrokenPipeError:
            return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
