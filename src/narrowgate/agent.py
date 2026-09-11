"""The conversation and tool-dispatch loop.

The loop owns three refusals, and they are the reason this file is small:

* it refuses to start against an endpoint that does not emit real ``tool_calls``
* it refuses to act on a tool call whose arguments did not parse
* it refuses to run forever

Everything else -- validating arguments, enforcing the network allow-list, auditing -- belongs to
:class:`~narrowgate.tools.registry.Registry` and happens before this file sees a result.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

from .audit import AuditLog, new_session_id
from .config import Config
from .llm import ChatResponse, LLMClient, LLMError, probe_tool_calls
from .tools.registry import Registry

SYSTEM_PROMPT = """You are an agent operating under a narrow tool interface.

You have no shell. You cannot execute code. The tools listed are the complete set of actions
available to you -- there is no way to obtain another one during this conversation.

If a task needs a capability you do not have, you may propose a new tool with `propose_tool`.
A proposal is written to disk for a human to read and approve. It does NOT become available to
you, now or later in this conversation. Do not plan around a proposed tool as though it exists.

Prefer saying you cannot do something over approximating it with a tool meant for another job."""


class StartupRefused(RuntimeError):
    """Raised when the harness will not start. Always carries the operator's next step."""


@dataclass
class Turn:
    """One model turn: what it said, and what it tried to do."""

    content: str | None
    tool_calls: list[str] = field(default_factory=list)


def ensure_tool_calls_supported(cfg: Config, *, api_key: str | None = None) -> None:
    """Refuse to start if the endpoint cannot emit structured tool calls.

    Refuses to guess. A model that *describes* calling a tool in prose looks like success to a
    naive loop and fails silently for the whole session, so this is checked once, up front, and
    the operator is told which ``--tool-call-parser`` to try.
    """
    if not cfg.agent.require_native_tool_calls:
        return
    support = probe_tool_calls(
        cfg.llm.base_url, cfg.llm.model, api_key=api_key, timeout_s=cfg.llm.timeout_s
    )
    if support.native:
        return
    hint = (
        f"\n  Try starting vLLM with: --enable-auto-tool-choice --tool-call-parser {support.parser_hint}"
        if support.parser_hint
        else "\n  No parser could be inferred from the response; check the model's tool-call format."
    )
    raise StartupRefused(
        f"Endpoint does not emit structured tool_calls for model {cfg.llm.model!r}.\n"
        f"  Server said: {support.detail}{hint}\n"
        "  Set agent.require_native_tool_calls: false to run anyway (the agent will be unable "
        "to use tools)."
    )


class Agent:
    """Drives one conversation. Owns no capabilities of its own."""

    def __init__(
        self,
        cfg: Config,
        client: LLMClient,
        registry: Registry,
        audit: AuditLog,
        *,
        session_id: str | None = None,
    ) -> None:
        self.cfg = cfg
        self.client = client
        self.registry = registry
        self.audit = audit
        self.session_id = session_id or new_session_id()
        self.messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]

    def run(self, user_input: str, *, on_turn: Callable[[Turn], None] | None = None) -> str:
        """Run until the model stops calling tools, or ``agent.max_turns`` is reached.

        Refuses to exceed the turn budget: a loop that cannot terminate is a loop that burns a
        local GPU all night. The budget is reported to the caller rather than raised, because a
        truncated answer is more useful than an exception.
        """
        self.messages.append({"role": "user", "content": user_input})
        specs = self.registry.specs()

        for turn_no in range(self.cfg.agent.max_turns):
            try:
                resp = self.client.chat(self.messages, tools=specs or None)
            except LLMError as exc:
                return f"[llm error on turn {turn_no + 1}: {exc}]"

            self._record_assistant(resp)
            if on_turn:
                on_turn(Turn(resp.content, [c.name for c in resp.tool_calls]))

            if not resp.tool_calls:
                return resp.content or ""

            for call in resp.tool_calls:
                self.messages.append(self._run_one(call))

        return (
            f"[stopped after {self.cfg.agent.max_turns} turns without a final answer; "
            "raise agent.max_turns if this was a legitimately long task]"
        )

    def _record_assistant(self, resp: ChatResponse) -> None:
        msg: dict[str, Any] = {"role": "assistant", "content": resp.content or ""}
        if resp.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": c.id,
                    "type": "function",
                    "function": {"name": c.name, "arguments": c.arguments_raw},
                }
                for c in resp.tool_calls
            ]
        self.messages.append(msg)

    def _run_one(self, call: Any) -> dict[str, Any]:
        """Dispatch one tool call and shape the reply message.

        Refuses to dispatch a call whose arguments did not parse as a JSON object. Guessing at
        malformed arguments is how a tool gets run with something its author never saw; the model
        is told plainly and gets to try again.
        """
        if call.malformed:
            return self._tool_msg(
                call,
                "arguments were not a JSON object and were not guessed at; "
                f"received: {call.arguments_raw[:200]!r}",
            )
        result = self.registry.dispatch(
            call.name, call.arguments or {}, audit=self.audit, session=self.session_id
        )
        body = result.content if result.ok else f"error: {result.error}"
        return self._tool_msg(call, body)

    @staticmethod
    def _tool_msg(call: Any, content: str) -> dict[str, Any]:
        return {
            "role": "tool",
            "tool_call_id": call.id,
            "name": call.name,
            "content": content if isinstance(content, str) else json.dumps(content),
        }
