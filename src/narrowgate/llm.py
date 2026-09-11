"""vLLM OpenAI-compatible chat client, plus tool-call capability DETECTION.

Two things live here and both are deliberately small:

* :class:`LLMClient` — one ``POST /chat/completions`` per :meth:`LLMClient.chat` call. **No
  retries.** A retried request can re-issue a side-effecting tool call; the agent loop decides
  what to do with a failure, this module never decides for it.
* :func:`probe_tool_calls` — sends one real request with a trivial ``tools`` array and reports
  what the server *actually* returned. Whether a given model emits structured ``tool_calls``
  under vLLM is model- and flag-specific and is unverified for our target as of 2026-09-11. The
  probe never assumes: ``native=True`` only when a well-formed ``tool_calls`` array came back.

The common failure this guards against: the model answers in *text* that describes calling the
tool (``<tool_call>{"name": ...}</tool_call>`` or plain prose). That looks like success to a
human skimming logs and is nothing of the kind — no tool ran, and nothing will.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

import httpx

__all__ = [
    "ChatResponse",
    "LLMClient",
    "LLMError",
    "ToolCall",
    "ToolCallSupport",
    "probe_tool_calls",
]


class LLMError(RuntimeError):
    """Transport failure, non-2xx status, or a response body that is not a chat completion."""


@dataclass(frozen=True)
class ToolCall:
    """One entry from ``choices[0].message.tool_calls``.

    ``arguments`` is ``None`` when the server's ``function.arguments`` string was not a JSON
    object; ``arguments_raw`` always holds the original string so the failure can be audited.
    """

    id: str
    name: str
    arguments: dict[str, Any] | None
    arguments_raw: str

    @property
    def malformed(self) -> bool:
        return self.arguments is None


@dataclass(frozen=True)
class ChatResponse:
    """The parts of a chat completion the agent loop needs. ``raw`` is the full decoded body."""

    content: str | None
    tool_calls: list[ToolCall]
    finish_reason: str | None
    raw: dict[str, Any] = field(repr=False)


@dataclass
class ToolCallSupport:
    """Result of :func:`probe_tool_calls`. Build to this exactly (see ARCHITECTURE.md)."""

    native: bool  # server returned a structured tool_calls array
    parser_hint: str | None
    detail: str


# --------------------------------------------------------------------------------------------
# client
# --------------------------------------------------------------------------------------------


def _headers(api_key: str | None) -> dict[str, str]:
    h = {"Content-Type": "application/json"}
    if api_key:
        h["Authorization"] = f"Bearer {api_key}"
    return h


def _parse_tool_calls(raw_calls: Any) -> list[ToolCall]:
    """Turn the server's ``tool_calls`` value into :class:`ToolCall` objects.

    Refuses to invent structure: an entry missing ``function.name`` is dropped and reported via
    :class:`LLMError` by the caller; bad ``arguments`` JSON yields ``arguments=None``.
    """
    if not isinstance(raw_calls, list):
        raise LLMError(f"message.tool_calls is not a list: {type(raw_calls).__name__}")
    out: list[ToolCall] = []
    for i, entry in enumerate(raw_calls):
        if not isinstance(entry, dict):
            raise LLMError(f"tool_calls[{i}] is not an object")
        fn = entry.get("function")
        if not isinstance(fn, dict) or not isinstance(fn.get("name"), str) or not fn["name"]:
            raise LLMError(f"tool_calls[{i}] has no function.name")
        args_raw = fn.get("arguments", "")
        if not isinstance(args_raw, str):
            # Some servers already decode arguments; normalise but keep a faithful raw string.
            args_raw = json.dumps(args_raw)
        parsed: dict[str, Any] | None
        try:
            candidate = json.loads(args_raw) if args_raw.strip() else {}
            parsed = candidate if isinstance(candidate, dict) else None
        except json.JSONDecodeError:
            parsed = None
        out.append(
            ToolCall(
                id=str(entry.get("id") or f"call_{i}"),
                name=fn["name"],
                arguments=parsed,
                arguments_raw=args_raw,
            )
        )
    return out


def _parse_completion(body: Any) -> ChatResponse:
    if not isinstance(body, dict):
        raise LLMError("response body is not a JSON object")
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise LLMError(f"response has no choices: {json.dumps(body)[:300]}")
    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise LLMError("choices[0].message missing or not an object")
    content = message.get("content")
    if content is not None and not isinstance(content, str):
        raise LLMError(f"message.content is not a string: {type(content).__name__}")
    calls = _parse_tool_calls(message.get("tool_calls") or [])
    finish = choices[0].get("finish_reason")
    return ChatResponse(
        content=content,
        tool_calls=calls,
        finish_reason=finish if isinstance(finish, str) else None,
        raw=body,
    )


class LLMClient:
    """Minimal chat-completions client for a vLLM OpenAI-compatible endpoint.

    Refuses to retry. Refuses to stream (one request, one parsed body — nothing the agent loop
    has to reassemble). Refuses to read secrets from anywhere but the ``api_key`` argument the
    caller resolved from the environment.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        api_key: str | None = None,
        timeout_s: float = 120.0,
        max_tokens: int = 4096,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers=_headers(api_key),
            timeout=httpx.Timeout(timeout_s, connect=min(10.0, timeout_s)),
            transport=transport,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> LLMClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        temperature: float | None = None,
    ) -> ChatResponse:
        """POST one chat completion and return the parsed :class:`ChatResponse`.

        Raises :class:`LLMError` on transport error, non-2xx status, or a body that is not a chat
        completion. Refuses to retry on any of those — the caller owns that decision because a
        retried request may duplicate a side-effecting tool call.
        """
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "stream": False,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto" if tool_choice is None else tool_choice
        if temperature is not None:
            payload["temperature"] = temperature
        try:
            resp = self._client.post("/chat/completions", json=payload)
        except httpx.HTTPError as exc:
            raise LLMError(f"request failed: {type(exc).__name__}: {exc}") from exc
        if resp.status_code // 100 != 2:
            raise LLMError(f"HTTP {resp.status_code}: {resp.text[:500]}")
        try:
            body = resp.json()
        except ValueError as exc:
            raise LLMError(f"response is not JSON: {resp.text[:300]}") from exc
        return _parse_completion(body)


# --------------------------------------------------------------------------------------------
# probe
# --------------------------------------------------------------------------------------------

_PROBE_TOOL_NAME = "get_weather"
_PROBE_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": _PROBE_TOOL_NAME,
            "description": "Get the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string", "description": "City name"}},
                "required": ["city"],
            },
        },
    }
]
_PROBE_MESSAGES: list[dict[str, Any]] = [
    {
        "role": "system",
        "content": (
            "You are a function-calling assistant. When a tool is relevant you MUST call it "
            "rather than answer in prose."
        ),
    },
    {
        "role": "user",
        "content": "What is the weather in Paris right now? Use the get_weather tool.",
    },
]

# Text markers that identify a model emitting tool calls in a *text* format vLLM can parse if the
# matching --tool-call-parser is enabled. Order matters: first match wins.
_TEXT_FORMAT_HINTS: list[tuple[re.Pattern[str], str, str]] = [
    (re.compile(r"<tool_call>\s*<function="), "qwen3_coder", "Qwen3-Coder XML-in-tags style"),
    (re.compile(r"<tool_call>"), "hermes", "Hermes/Qwen <tool_call>{json}</tool_call> style"),
    (re.compile(r"<function="), "qwen3_coder", "bare <function=...> style"),
    (re.compile(r"<\|python_tag\|>"), "llama3_json", "Llama 3.x <|python_tag|> style"),
    (re.compile(r"\[TOOL_CALLS\]"), "mistral", "Mistral [TOOL_CALLS] style"),
    (re.compile(r"<\|tool_call\|>|<function_call>"), "granite", "Granite style"),
    (re.compile(r"<\|tool▁calls▁begin\|>"), "deepseek_v3", "DeepSeek V3 style"),
    (re.compile(r"^\s*\[\s*[a-z_][a-z0-9_]*\("), "pythonic", "pythonic [fn(arg=...)] style"),
    (
        re.compile(r"^\s*\{\s*\"name\"\s*:\s*\"[^\"]+\"\s*,\s*\"(arguments|parameters)\""),
        "llama3_json",
        'bare JSON {"name":..., "arguments":...} style',
    ),
]

_AUTO_CHOICE_ERR = re.compile(r"enable-auto-tool-choice|tool-call-parser", re.IGNORECASE)


def _quote(text: str | None, limit: int = 400) -> str:
    if text is None:
        return "<null>"
    text = text.strip()
    if len(text) > limit:
        return repr(text[:limit] + "…")
    return repr(text)


def _hint_from_text(content: str) -> tuple[str | None, str]:
    for pattern, hint, label in _TEXT_FORMAT_HINTS:
        if pattern.search(content):
            return hint, label
    return None, "no recognisable tool-call markup"


def probe_tool_calls(
    base_url: str,
    model: str,
    *,
    api_key: str | None = None,
    timeout_s: float = 60.0,
    transport: httpx.BaseTransport | None = None,
) -> ToolCallSupport:
    """Send one real chat request with a trivial ``tools`` array and report what came back.

    ``native=True`` **only** when ``choices[0].message.tool_calls`` is a non-empty list whose
    first entry names the probe tool and carries ``arguments`` that decode to a JSON object.
    Everything else is ``native=False`` with ``detail`` quoting the server's actual response and
    ``parser_hint`` naming a ``--tool-call-parser`` worth trying (or ``None`` if the text format
    was not recognised — the probe refuses to guess a parser it has no evidence for).

    Refuses to raise: a transport error, timeout, or non-2xx status is reported as
    ``native=False`` so the operator sees the reason instead of a traceback. Refuses to retry.
    Refuses to use ``tool_choice="required"`` — that forces structured output via guided decoding
    and would report success for a server whose ``auto`` mode (the one the agent uses) is broken.

    ``transport`` is a test seam for :class:`httpx.MockTransport`; production callers omit it.
    """
    try:
        with LLMClient(
            base_url,
            model,
            api_key=api_key,
            timeout_s=timeout_s,
            max_tokens=256,
            transport=transport,
        ) as client:
            resp = client.chat(
                _PROBE_MESSAGES, tools=_PROBE_TOOLS, tool_choice="auto", temperature=0
            )
    except LLMError as exc:
        msg = str(exc)
        if _AUTO_CHOICE_ERR.search(msg):
            return ToolCallSupport(
                native=False,
                parser_hint="hermes",
                detail=(
                    "server rejected tool_choice=auto — vLLM needs --enable-auto-tool-choice "
                    "and a --tool-call-parser (hermes is the usual first try; check the model "
                    f"card). Server said: {msg}"
                ),
            )
        return ToolCallSupport(
            native=False,
            parser_hint=None,
            detail=f"probe request did not complete: {msg}",
        )

    if resp.tool_calls:
        call = resp.tool_calls[0]
        if call.name != _PROBE_TOOL_NAME:
            return ToolCallSupport(
                native=False,
                parser_hint=None,
                detail=(
                    f"server returned a tool_calls array but the first call names "
                    f"{call.name!r}, not {_PROBE_TOOL_NAME!r}; "
                    f"arguments={_quote(call.arguments_raw)}"
                ),
            )
        if call.malformed:
            return ToolCallSupport(
                native=False,
                parser_hint=None,
                detail=(
                    "server returned a tool_calls array but function.arguments is not a JSON "
                    f"object: {_quote(call.arguments_raw)} — a parser is enabled but is emitting "
                    "malformed arguments; try a different --tool-call-parser for this model"
                ),
            )
        note = (
            ""
            if resp.finish_reason == "tool_calls"
            else (f" (note: finish_reason={resp.finish_reason!r}, expected 'tool_calls')")
        )
        return ToolCallSupport(
            native=True,
            parser_hint=None,
            detail=(
                f"structured tool_calls returned: {call.name}({json.dumps(call.arguments)}){note}"
            ),
        )

    content = resp.content
    hint, label = _hint_from_text(content or "")
    if hint is not None:
        return ToolCallSupport(
            native=False,
            parser_hint=hint,
            detail=(
                f"no tool_calls array; the model wrote the call as TEXT ({label}). "
                f"Restart vLLM with --enable-auto-tool-choice --tool-call-parser {hint}. "
                f"Server returned content={_quote(content)}"
            ),
        )
    return ToolCallSupport(
        native=False,
        parser_hint=None,
        detail=(
            f"no tool_calls array and {label} in the reply — the model answered in prose instead "
            f"of calling the tool. finish_reason={resp.finish_reason!r}, "
            f"content={_quote(content)}"
        ),
    )
