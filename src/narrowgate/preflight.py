"""Operator-facing preflight: does this endpoint emit well-formed tool_calls?

Run before anything else::

    python -m narrowgate.preflight [--config config.yaml] [--base-url URL] [--model NAME]

Exit codes: ``0`` usable · ``1`` endpoint reachable but not usable (or unreachable) · ``2`` the
config itself was rejected. Never exits 0 on a guess.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from typing import TextIO

from narrowgate.config import ConfigError, load_config, resolve_api_key
from narrowgate.llm import ToolCallSupport, probe_tool_calls

__all__ = ["main", "render_verdict"]


def render_verdict(model: str, base_url: str, support: ToolCallSupport) -> str:
    """Format the probe result for a human. Refuses to soften a failure into a warning."""
    lines = [
        f"narrowgate preflight — {model} @ {base_url}",
        "",
    ]
    if support.native:
        lines += [
            "VERDICT: USABLE — server returned a structured tool_calls array.",
            f"  {support.detail}",
        ]
    else:
        lines += [
            "VERDICT: NOT USABLE — no well-formed tool_calls from this endpoint.",
            f"  {support.detail}",
        ]
        if support.parser_hint:
            lines += [
                "",
                "  Try restarting vLLM with:",
                f"    --enable-auto-tool-choice --tool-call-parser {support.parser_hint}",
            ]
        else:
            lines += [
                "",
                "  No parser could be inferred from the reply. Check the model card for its",
                "  tool-call format and the matching vLLM --tool-call-parser value.",
            ]
        lines.append("  The agent will refuse to start until this passes.")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None, *, out: TextIO = sys.stdout) -> int:
    """CLI entry point. Returns the exit code; refuses to return 0 unless ``native`` is True.

    Reads the endpoint from ``--config`` (default ``config.yaml``); ``--base-url`` and
    ``--model`` override it. A config that fails validation is exit 2 with the reason.
    """
    parser = argparse.ArgumentParser(prog="narrowgate-preflight", description=__doc__)
    parser.add_argument("--config", default="config.yaml", help="path to config.yaml")
    parser.add_argument("--base-url", default=None, help="override llm.base_url")
    parser.add_argument("--model", default=None, help="override llm.model")
    parser.add_argument("--timeout", type=float, default=None, help="override llm.timeout_s")
    args = parser.parse_args(argv)

    if args.base_url and args.model:
        base_url, model, api_key, timeout_s = args.base_url, args.model, None, args.timeout or 60.0
    else:
        try:
            cfg = load_config(args.config)
            api_key = resolve_api_key(cfg)
        except ConfigError as exc:
            print(f"narrowgate preflight: {exc}", file=sys.stderr)
            return 2
        base_url = args.base_url or cfg.llm.base_url
        model = args.model or cfg.llm.model
        timeout_s = args.timeout or cfg.llm.timeout_s

    support = probe_tool_calls(base_url, model, api_key=api_key, timeout_s=timeout_s)
    print(render_verdict(model, base_url, support), file=out)
    return 0 if support.native else 1


if __name__ == "__main__":
    sys.exit(main())
