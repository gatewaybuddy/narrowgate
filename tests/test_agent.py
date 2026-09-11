"""Tests for the agent loop -- refusal branches first.

The loop's value is in what it declines to do, so that is what is tested: a non-native endpoint,
malformed tool arguments, a runaway turn count, and an LLM error that must not escape.
"""

from __future__ import annotations

import pytest

from narrowgate.agent import Agent, StartupRefused, ensure_tool_calls_supported
from narrowgate.llm import ChatResponse, LLMError, ToolCall
from narrowgate.tools.base import ToolResult


class _Cfg:
    class agent:  # noqa: N801
        max_turns = 3
        require_native_tool_calls = True

    class llm:  # noqa: N801
        base_url = "http://x/v1"
        model = "m"
        timeout_s = 5


class _Audit:
    def before_call(self, *a, **k):
        return "cid"

    def after_call(self, *a, **k):
        return None


class _Registry:
    def __init__(self, result=None):
        self.calls: list[tuple[str, dict]] = []
        self._result = result or ToolResult(True, "ok")

    def specs(self):
        return [{"type": "function", "function": {"name": "t", "parameters": {}}}]

    def dispatch(self, name, args, *, audit, session=None):
        self.calls.append((name, args))
        return self._result


class _Client:
    def __init__(self, responses):
        self._responses = list(responses)

    def chat(self, messages, *, tools=None, tool_choice=None, temperature=None):
        r = self._responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def _resp(content=None, calls=()):
    return ChatResponse(content=content, tool_calls=list(calls), finish_reason="stop", raw={})


def _call(name="t", args=None, raw="{}", cid="c1"):
    return ToolCall(id=cid, name=name, arguments=args, arguments_raw=raw)


def test_refuses_to_start_when_endpoint_is_not_native(monkeypatch):
    """A model that only *describes* calling a tool must stop startup, not be discovered later."""
    from narrowgate import agent as mod

    monkeypatch.setattr(
        mod,
        "probe_tool_calls",
        lambda *a, **k: mod.probe_tool_calls.__self__ if False else _NotNative(),
    )
    with pytest.raises(StartupRefused) as exc:
        ensure_tool_calls_supported(_Cfg)
    assert "does not emit structured tool_calls" in str(exc.value)
    assert "--tool-call-parser hermes" in str(exc.value)


class _NotNative:
    native = False
    parser_hint = "hermes"
    detail = "model replied in prose: 'I will call t()'"


def test_allows_start_when_native(monkeypatch):
    from narrowgate import agent as mod

    monkeypatch.setattr(mod, "probe_tool_calls", lambda *a, **k: _Native())
    ensure_tool_calls_supported(_Cfg)  # must not raise


class _Native:
    native = True
    parser_hint = None
    detail = "ok"


def test_malformed_arguments_are_refused_not_guessed():
    """Arguments that did not parse must never reach a tool."""
    reg = _Registry()
    client = _Client([_resp(calls=[_call(args=None, raw="{oops")]), _resp(content="done")])
    a = Agent(_Cfg, client, reg, _Audit())
    a.run("go")
    assert reg.calls == [], "a tool was dispatched with unparsed arguments"
    tool_msg = [m for m in a.messages if m["role"] == "tool"][0]
    assert "not a JSON object" in tool_msg["content"]
    assert "{oops" in tool_msg["content"]


def test_turn_budget_is_enforced():
    """A model that calls tools forever must be stopped, not allowed to run the GPU all night."""
    reg = _Registry()
    client = _Client([_resp(calls=[_call(args={})]) for _ in range(3)])
    a = Agent(_Cfg, client, reg, _Audit())
    out = a.run("go")
    assert "stopped after 3 turns" in out
    assert len(reg.calls) == 3


def test_llm_error_is_returned_not_raised():
    a = Agent(_Cfg, _Client([LLMError("connection refused")]), _Registry(), _Audit())
    out = a.run("go")
    assert "llm error" in out and "connection refused" in out


def test_failed_tool_result_is_reported_to_the_model():
    reg = _Registry(ToolResult(False, "", "path escapes workspace root"))
    client = _Client([_resp(calls=[_call(args={})]), _resp(content="understood")])
    a = Agent(_Cfg, client, reg, _Audit())
    a.run("go")
    tool_msg = [m for m in a.messages if m["role"] == "tool"][0]
    assert "error: path escapes workspace root" == tool_msg["content"]


def test_system_prompt_tells_the_model_proposals_do_not_become_available():
    """The model must not plan around a proposed tool as though it exists."""
    a = Agent(_Cfg, _Client([]), _Registry(), _Audit())
    # Collapse whitespace: this asserts a PROPERTY of the prompt, not its line wrapping.
    sys_msg = " ".join(a.messages[0]["content"].split())
    assert "does NOT become available to you" in sys_msg
    assert "no shell" in sys_msg.lower()
    assert "cannot execute code" in sys_msg.lower()
