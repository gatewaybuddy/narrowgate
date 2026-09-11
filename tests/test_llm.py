"""llm.py — the probe must detect, never assume; the client must never retry.

No network: every test routes through ``httpx.MockTransport``.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from narrowgate.llm import LLMClient, LLMError, ToolCallSupport, probe_tool_calls

BASE = "http://vllm.test/v1"
MODEL = "nemotron-lightning"


def completion(
    *,
    content: str | None = None,
    tool_calls: list[dict[str, Any]] | None = None,
    finish_reason: str = "stop",
) -> dict[str, Any]:
    msg: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls is not None:
        msg["tool_calls"] = tool_calls
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "model": MODEL,
        "choices": [{"index": 0, "message": msg, "finish_reason": finish_reason}],
    }


def native_call(name: str = "get_weather", arguments: str = '{"city": "Paris"}') -> dict[str, Any]:
    return {
        "id": "call_1",
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


class Recorder:
    """MockTransport that records every request and serves a fixed handler."""

    def __init__(self, handler: Callable[[httpx.Request], httpx.Response]) -> None:
        self.requests: list[httpx.Request] = []
        self._handler = handler

    def transport(self) -> httpx.MockTransport:
        def _h(req: httpx.Request) -> httpx.Response:
            self.requests.append(req)
            return self._handler(req)

        return httpx.MockTransport(_h)


def serve_json(body: dict[str, Any], status: int = 200) -> Recorder:
    return Recorder(lambda _req: httpx.Response(status, json=body))


def probe(rec: Recorder, **kw: Any) -> ToolCallSupport:
    return probe_tool_calls(BASE, MODEL, transport=rec.transport(), **kw)


# -- probe: the refusal branches ----------------------------------------------------------------


def test_text_that_describes_a_tool_call_is_not_native() -> None:
    """The common failure: prose that looks like success and is not."""
    rec = serve_json(
        completion(
            content=(
                "I'll call the get_weather tool with city='Paris' to check the current "
                "conditions for you."
            )
        )
    )
    r = probe(rec)
    assert r.native is False
    assert r.parser_hint is None
    assert "no tool_calls array" in r.detail
    assert "get_weather tool with city='Paris'" in r.detail  # quotes what the server said


@pytest.mark.parametrize(
    ("content", "expected_hint"),
    [
        (
            '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call>',
            "hermes",
        ),
        (
            "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n"
            "</function>\n</tool_call>",
            "qwen3_coder",
        ),
        ('<|python_tag|>{"name": "get_weather", "parameters": {"city": "Paris"}}', "llama3_json"),
        ('[TOOL_CALLS] [{"name": "get_weather", "arguments": {"city": "Paris"}}]', "mistral"),
        ('<|tool_call|>[{"name": "get_weather", "arguments": {"city": "Paris"}}]', "granite"),
        ('[get_weather(city="Paris")]', "pythonic"),
        ('{"name": "get_weather", "arguments": {"city": "Paris"}}', "llama3_json"),
    ],
)
def test_textual_tool_call_formats_yield_parser_hint(content: str, expected_hint: str) -> None:
    r = probe(serve_json(completion(content=content)))
    assert r.native is False
    assert r.parser_hint == expected_hint
    assert f"--tool-call-parser {expected_hint}" in r.detail
    assert "content=" in r.detail


def test_empty_tool_calls_array_is_not_native() -> None:
    r = probe(serve_json(completion(content="Sure.", tool_calls=[])))
    assert r.native is False


def test_null_content_and_no_tool_calls_is_not_native() -> None:
    r = probe(serve_json(completion(content=None, finish_reason="length")))
    assert r.native is False
    assert "finish_reason='length'" in r.detail
    assert "<null>" in r.detail


def test_tool_calls_with_malformed_arguments_is_not_native() -> None:
    body = completion(
        tool_calls=[native_call(arguments='{"city": "Paris"')], finish_reason="tool_calls"
    )
    r = probe(serve_json(body))
    assert r.native is False
    assert "not a JSON object" in r.detail
    assert '{"city": "Paris"' in r.detail


def test_tool_calls_with_non_object_arguments_is_not_native() -> None:
    body = completion(tool_calls=[native_call(arguments='["Paris"]')], finish_reason="tool_calls")
    r = probe(serve_json(body))
    assert r.native is False


def test_tool_calls_naming_wrong_tool_is_not_native() -> None:
    body = completion(tool_calls=[native_call(name="search_web")], finish_reason="tool_calls")
    r = probe(serve_json(body))
    assert r.native is False
    assert "'search_web'" in r.detail


def test_tool_calls_missing_function_name_is_not_native() -> None:
    body = completion(tool_calls=[{"id": "x", "type": "function", "function": {"arguments": "{}"}}])
    r = probe(serve_json(body))
    assert r.native is False
    assert "function.name" in r.detail


def test_vllm_auto_tool_choice_400_gives_flag_advice() -> None:
    err = {
        "object": "error",
        "message": (
            '"auto" tool choice requires --enable-auto-tool-choice and --tool-call-parser to be set'
        ),
        "type": "BadRequestError",
        "code": 400,
    }
    r = probe(serve_json(err, status=400))
    assert r.native is False
    assert r.parser_hint == "hermes"
    assert "--enable-auto-tool-choice" in r.detail
    assert "HTTP 400" in r.detail


def test_other_http_error_is_reported_not_raised() -> None:
    r = probe(serve_json({"error": "no such model"}, status=404))
    assert r.native is False
    assert r.parser_hint is None
    assert "HTTP 404" in r.detail and "no such model" in r.detail


def test_connection_failure_is_reported_not_raised() -> None:
    def boom(_req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    r = probe(Recorder(boom))
    assert r.native is False
    assert r.parser_hint is None
    assert "ConnectError" in r.detail and "connection refused" in r.detail


def test_timeout_is_reported_not_raised() -> None:
    def slow(_req: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out")

    r = probe(Recorder(slow))
    assert r.native is False
    assert "ReadTimeout" in r.detail


def test_non_json_body_is_reported_not_raised() -> None:
    r = probe(Recorder(lambda _r: httpx.Response(200, text="<html>gateway</html>")))
    assert r.native is False
    assert "not JSON" in r.detail


def test_json_body_without_choices_is_reported_not_raised() -> None:
    r = probe(serve_json({"object": "list", "data": []}))
    assert r.native is False
    assert "no choices" in r.detail


# -- probe: the one path that may say native=True ----------------------------------------------


def test_structured_tool_calls_is_native() -> None:
    rec = serve_json(completion(tool_calls=[native_call()], finish_reason="tool_calls"))
    r = probe(rec)
    assert r.native is True
    assert r.parser_hint is None
    assert 'get_weather({"city": "Paris"})' in r.detail
    assert "note:" not in r.detail


def test_native_with_unexpected_finish_reason_is_noted() -> None:
    r = probe(serve_json(completion(tool_calls=[native_call()], finish_reason="stop")))
    assert r.native is True
    assert "finish_reason='stop'" in r.detail


def test_probe_sends_one_request_with_tools_and_auto_choice() -> None:
    rec = serve_json(completion(tool_calls=[native_call()], finish_reason="tool_calls"))
    probe(rec, api_key="k")
    assert len(rec.requests) == 1
    req = rec.requests[0]
    assert req.url == httpx.URL(f"{BASE}/chat/completions")
    assert req.headers["authorization"] == "Bearer k"
    body = json.loads(req.content)
    assert body["model"] == MODEL
    assert body["tool_choice"] == "auto"  # never "required" — that would mask a broken auto mode
    assert body["stream"] is False
    assert [t["function"]["name"] for t in body["tools"]] == ["get_weather"]


def test_probe_sends_no_auth_header_without_key() -> None:
    rec = serve_json(completion(content="hi"))
    probe(rec)
    assert "authorization" not in rec.requests[0].headers


def test_probe_never_retries_on_failure() -> None:
    def boom(_req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    rec = Recorder(boom)
    probe(rec)
    assert len(rec.requests) == 1


# -- client -------------------------------------------------------------------------------------


def test_client_never_retries_on_5xx() -> None:
    rec = serve_json({"error": "overloaded"}, status=503)
    with LLMClient(BASE, MODEL, transport=rec.transport()) as c:
        with pytest.raises(LLMError, match="HTTP 503"):
            c.chat([{"role": "user", "content": "hi"}])
    assert len(rec.requests) == 1


def test_client_never_retries_on_transport_error() -> None:
    def boom(_req: httpx.Request) -> httpx.Response:
        raise httpx.ReadError("reset")

    rec = Recorder(boom)
    with LLMClient(BASE, MODEL, transport=rec.transport()) as c:
        with pytest.raises(LLMError, match="ReadError"):
            c.chat([{"role": "user", "content": "hi"}])
    assert len(rec.requests) == 1


def test_client_omits_tools_and_tool_choice_when_no_tools() -> None:
    rec = serve_json(completion(content="hello"))
    with LLMClient(BASE, MODEL, transport=rec.transport(), max_tokens=77) as c:
        r = c.chat([{"role": "user", "content": "hi"}])
    body = json.loads(rec.requests[0].content)
    assert "tools" not in body and "tool_choice" not in body
    assert body["max_tokens"] == 77
    assert r.content == "hello" and r.tool_calls == [] and r.finish_reason == "stop"


def test_client_surfaces_malformed_arguments_without_raising() -> None:
    body = completion(tool_calls=[native_call(arguments="not json")], finish_reason="tool_calls")
    with LLMClient(BASE, MODEL, transport=serve_json(body).transport()) as c:
        r = c.chat([{"role": "user", "content": "hi"}], tools=[{"type": "function"}])
    (call,) = r.tool_calls
    assert call.malformed is True
    assert call.arguments is None
    assert call.arguments_raw == "not json"
    assert call.name == "get_weather"


def test_client_rejects_non_string_content() -> None:
    body = completion()
    body["choices"][0]["message"]["content"] = {"unexpected": "object"}
    with LLMClient(BASE, MODEL, transport=serve_json(body).transport()) as c:
        with pytest.raises(LLMError, match="content is not a string"):
            c.chat([{"role": "user", "content": "hi"}])


def test_client_rejects_tool_calls_that_is_not_a_list() -> None:
    body = completion(content=None)
    body["choices"][0]["message"]["tool_calls"] = {"function": {"name": "x"}}
    with LLMClient(BASE, MODEL, transport=serve_json(body).transport()) as c:
        with pytest.raises(LLMError, match="not a list"):
            c.chat([{"role": "user", "content": "hi"}])


def test_client_applies_configured_timeout() -> None:
    def never(_req: httpx.Request) -> httpx.Response:
        raise AssertionError("no request expected")

    with LLMClient(BASE, MODEL, timeout_s=7.5, transport=httpx.MockTransport(never)) as c:
        t = c._client.timeout
    assert t.read == 7.5 and t.write == 7.5 and t.pool == 7.5
    assert t.connect == 7.5  # min(10, timeout_s)
