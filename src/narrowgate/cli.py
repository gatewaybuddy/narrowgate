"""Operator entry points.

Three commands, deliberately separate:

* ``narrowgate preflight``          -- probe the endpoint before trusting it
* ``narrowgate``                    -- run the agent loop
* ``narrowgate activate <name>``    -- promote a staged tool the agent proposed

``activate`` is its own command, run by a human, in its own process. That separation is the whole
self-extension design and not an interface convenience: see ``docs/SELF-EXTENSION.md``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .agent import Agent, StartupRefused, ensure_tool_calls_supported
from .audit import AuditLog
from .config import Config, ConfigError, load_config, resolve_api_key
from .llm import LLMClient
from .tools.builtin import ListDirTool, ReadFileTool, WriteFileTool, Workspace
from .tools.propose import ProposeTool
from .tools.registry import Registry

STAGED_DIR = Path("tools_staged")
ENABLED_DIR = Path("src/narrowgate/tools/enabled")


def build_registry(cfg: Config, audit: AuditLog) -> Registry:
    """Wire the built-in tools.

    Refuses to auto-discover anything. Every tool in the running harness appears as a literal
    line here or was promoted by a human through ``activate``; there is no third way in.
    """
    ws = Workspace(cfg.workspace.root, cfg.workspace.max_file_bytes)
    reg = Registry(network_allow_tools=cfg.network.allow_tools)
    for tool in (
        ReadFileTool(ws),
        ListDirTool(ws),
        WriteFileTool(ws),
        ProposeTool(staged_dir=STAGED_DIR, enabled_dir=ENABLED_DIR, audit=audit),
    ):
        reg.register(tool)
    return reg


def _load(path: str) -> Config:
    try:
        return load_config(path)
    except ConfigError as exc:
        print(f"narrowgate: config rejected\n{exc}", file=sys.stderr)
        raise SystemExit(2) from None


def cmd_preflight(args: argparse.Namespace) -> int:
    from .preflight import main as preflight_main

    return preflight_main(["--config", args.config])


def cmd_activate(args: argparse.Namespace) -> int:
    """Promote a staged tool. Delegates to the gate; never loads the tool here."""
    try:
        from .activate import main as activate_main
    except ImportError as exc:  # pragma: no cover - only while the gate is unbuilt
        print(f"narrowgate: activate is unavailable ({exc})", file=sys.stderr)
        return 2
    return activate_main([args.name, "--config", args.config])


def cmd_run(args: argparse.Namespace) -> int:
    cfg = _load(args.config)
    api_key = resolve_api_key(cfg)
    try:
        ensure_tool_calls_supported(cfg, api_key=api_key)
    except StartupRefused as exc:
        print(f"narrowgate: refusing to start\n{exc}", file=sys.stderr)
        return 1

    audit = AuditLog(cfg.audit.path)
    registry = build_registry(cfg, audit)
    client = LLMClient(cfg.llm.base_url, cfg.llm.model, api_key=api_key, timeout_s=cfg.llm.timeout_s)
    agent = Agent(cfg, client, registry, audit)

    print(f"narrowgate {cfg.llm.model} | tools: {', '.join(registry.names())}")
    print("ctrl-d to exit\n")
    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not line:
            continue
        print(agent.run(line, on_turn=_show_turn))


def _show_turn(turn) -> None:
    if turn.tool_calls:
        print(f"  [tools: {', '.join(turn.tool_calls)}]", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="narrowgate", description=__doc__.split("\n")[0])
    p.add_argument("--config", default="config.yaml")
    sub = p.add_subparsers(dest="cmd")

    sub.add_parser("preflight", help="probe the endpoint for tool_call support")
    act = sub.add_parser("activate", help="promote a staged tool (human gate)")
    act.add_argument("name")

    args = p.parse_args(argv)
    if args.cmd == "preflight":
        return cmd_preflight(args)
    if args.cmd == "activate":
        return cmd_activate(args)
    return cmd_run(args)


if __name__ == "__main__":
    raise SystemExit(main())
