"""Registry and base-layer tests. Every refusal branch in the contract has a test here."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from narrowgate.tools.base import (
    ArgumentError,
    SchemaError,
    ToolResult,
    validate_args,
    validate_schema,
    validate_tool_name,
)
from narrowgate.tools.registry import RegistrationError, Registry

# --- stubs -----------------------------------------------------------------------------------

ECHO_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "text": {"type": "string", "minLength": 1},
        "times": {"type": "integer", "minimum": 1, "maximum": 5},
    },
    "required": ["text"],
    "additionalProperties": False,
}


class EchoTool:
    name = "echo"
    description = "Repeat text."
    schema = ECHO_SCHEMA
    requires_network = False

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def run(self, **kwargs: Any) -> ToolResult:
        self.calls.append(kwargs)
        return ToolResult(True, kwargs["text"] * kwargs.get("times", 1))


class NetTool(EchoTool):
    name = "fetch"
    description = "Pretend to fetch."
    requires_network = True


class RaisingTool(EchoTool):
    name = "boom"
    description = "Always raises."

    def run(self, **kwargs: Any) -> ToolResult:
        self.calls.append(kwargs)
        raise RuntimeError("kaboom")


class WrongReturnTool(EchoTool):
    name = "wrong"
    description = "Returns the wrong type."

    def run(self, **kwargs: Any) -> ToolResult:
        return "not a ToolResult"  # type: ignore[return-value]


class RecordingAudit:
    def __init__(self) -> None:
        self.events: list[tuple[str, str, Mapping[str, Any], str | None, ToolResult | None]] = []

    def before_call(self, *, tool: str, args: Mapping[str, Any], session: str | None) -> None:
        self.events.append(("before", tool, dict(args), session, None))

    def after_call(
        self, *, tool: str, args: Mapping[str, Any], session: str | None, result: ToolResult
    ) -> None:
        self.events.append(("after", tool, dict(args), session, result))


class BrokenBeforeAudit(RecordingAudit):
    def before_call(self, *, tool: str, args: Mapping[str, Any], session: str | None) -> None:
        raise OSError("disk full")


class BrokenAfterAudit(RecordingAudit):
    def after_call(
        self, *, tool: str, args: Mapping[str, Any], session: str | None, result: ToolResult
    ) -> None:
        raise OSError("disk full")


def make_tool(**overrides: Any) -> EchoTool:
    tool = EchoTool()
    for key, value in overrides.items():
        setattr(tool, key, value)
    return tool


# --- ToolResult -------------------------------------------------------------------------------


def test_toolresult_is_frozen_and_failure_needs_error() -> None:
    ok = ToolResult(True, "x")
    with pytest.raises(AttributeError):
        ok.content = "y"  # type: ignore[misc]
    with pytest.raises(ValueError):
        ToolResult(False, "")


# --- name / schema validation -----------------------------------------------------------------


@pytest.mark.parametrize("name", ["abc", "read_file", "a1_", "a" + "b" * 63])
def test_valid_tool_names(name: str) -> None:
    assert validate_tool_name(name) == name


@pytest.mark.parametrize(
    "name", ["ab", "Abc", "1abc", "_abc", "a-b-c", "a b", "a" + "b" * 64, "", None, 42, "run.sh"]
)
def test_invalid_tool_names(name: object) -> None:
    with pytest.raises(ValueError):
        validate_tool_name(name)


@pytest.mark.parametrize(
    "schema",
    [
        "not a dict",
        {"type": "string", "additionalProperties": False},
        {"type": "object", "properties": {}},  # additionalProperties unspecified
        {"type": "object", "properties": {}, "additionalProperties": True},
        {
            "type": "object",
            "properties": {"x": {"type": "string", "format": "uri"}},
            "additionalProperties": False,
        },  # unsupported keyword => silently unenforced
        {
            "type": "object",
            "properties": {"x": {"type": "string"}},
            "required": ["y"],
            "additionalProperties": False,
        },
        {"type": "object", "properties": {"x": {"type": "strng"}}, "additionalProperties": False},
        {
            "type": "object",
            "properties": {"x": {"type": "string", "pattern": "("}},
            "additionalProperties": False,
        },
        {
            "type": "object",
            "properties": {"x": {"type": "string", "maxLength": -1}},
            "additionalProperties": False,
        },
        {
            "type": "object",
            "properties": {"x": {"type": "integer", "minimum": True}},
            "additionalProperties": False,
        },
        {"type": "object", "properties": {"x": {"enum": []}}, "additionalProperties": False},
        {
            "type": "object",
            "properties": {"x": {"$ref": "#/defs/y"}},
            "additionalProperties": False,
        },
    ],
)
def test_bad_schemas_are_refused(schema: object) -> None:
    with pytest.raises(SchemaError):
        validate_schema(schema)


def test_good_schema_accepted() -> None:
    validate_schema(ECHO_SCHEMA)


# --- validate_args ----------------------------------------------------------------------------


def test_validate_args_happy_path_returns_plain_dict() -> None:
    out = validate_args(ECHO_SCHEMA, {"text": "hi", "times": 2})
    assert out == {"text": "hi", "times": 2}
    assert type(out) is dict


@pytest.mark.parametrize(
    "args",
    [
        None,
        [],
        "text=hi",
        {},  # missing required
        {"text": ""},  # minLength
        {"text": 5},
        {"text": "hi", "times": 0},
        {"text": "hi", "times": 6},
        {"text": "hi", "times": True},  # bool is not an integer
        {"text": "hi", "times": 2.0},
        {"text": "hi", "extra": 1},  # undeclared key
        {1: "x", "text": "hi"},  # non-str key
    ],
)
def test_validate_args_refusals(args: object) -> None:
    with pytest.raises(ArgumentError):
        validate_args(ECHO_SCHEMA, args)


def test_validate_args_nested_and_arrays() -> None:
    schema = {
        "type": "object",
        "properties": {
            "mode": {"enum": ["a", "b"]},
            "tags": {"type": "array", "items": {"type": "string", "pattern": "^t"}, "maxItems": 2},
            "opts": {
                "type": "object",
                "properties": {"n": {"type": "number", "exclusiveMinimum": 0}},
                "additionalProperties": False,
            },
        },
        "additionalProperties": False,
    }
    validate_schema(schema)
    validate_args(schema, {"mode": "a", "tags": ["t1", "t2"], "opts": {"n": 0.5}})
    for bad in (
        {"mode": "c"},
        {"tags": ["x"]},
        {"tags": ["t1", "t2", "t3"]},
        {"opts": {"n": 0}},
        {"opts": {"z": 1}},
    ):
        with pytest.raises(ArgumentError):
            validate_args(schema, bad)


# --- register ---------------------------------------------------------------------------------


def test_register_and_get() -> None:
    reg = Registry()
    tool = EchoTool()
    reg.register(tool)
    assert reg.get("echo") is tool
    assert reg.get("nope") is None
    assert reg.get(None) is None  # type: ignore[arg-type]
    assert reg.names() == ["echo"]


def test_register_duplicate_refused() -> None:
    reg = Registry()
    reg.register(EchoTool())
    with pytest.raises(RegistrationError, match="already registered"):
        reg.register(EchoTool())


@pytest.mark.parametrize(
    "overrides",
    [
        {"name": "Echo"},
        {"name": "ab"},
        {"name": None},
        {"schema": {"type": "object"}},
        {"schema": None},
        {"requires_network": "no"},
        {"requires_network": 0},
        {"requires_network": None},
        {"description": ""},
        {"description": None},
        {"run": "not callable"},
    ],
)
def test_register_refuses_bad_declarations(overrides: dict[str, Any]) -> None:
    reg = Registry()
    with pytest.raises(RegistrationError):
        reg.register(make_tool(**overrides))
    assert reg.names() == []


def test_register_refuses_tool_missing_requires_network() -> None:
    class Undeclared:
        name = "undeclared"
        description = "no network flag"
        schema = ECHO_SCHEMA

        def run(self, **kwargs: Any) -> ToolResult:
            return ToolResult(True, "")

    with pytest.raises(RegistrationError, match="requires_network"):
        Registry().register(Undeclared())


def test_registry_has_no_discovery_surface() -> None:
    """The registry exposes nothing that takes a module, path, or package to scan."""
    public = {n for n in dir(Registry) if not n.startswith("_")}
    assert public == {"register", "get", "names", "is_dispatchable", "specs", "dispatch"}


# --- specs ------------------------------------------------------------------------------------


def test_specs_openai_format_and_network_filtering() -> None:
    reg = Registry()
    reg.register(EchoTool())
    reg.register(NetTool())
    specs = reg.specs()
    assert [s["function"]["name"] for s in specs] == ["echo"]
    assert specs[0] == {
        "type": "function",
        "function": {"name": "echo", "description": "Repeat text.", "parameters": ECHO_SCHEMA},
    }

    allowed = Registry(network_allow_tools=["fetch"])
    allowed.register(EchoTool())
    allowed.register(NetTool())
    assert [s["function"]["name"] for s in allowed.specs()] == ["echo", "fetch"]


# --- dispatch ---------------------------------------------------------------------------------


def test_dispatch_happy_path_audits_before_and_after() -> None:
    reg = Registry()
    tool = EchoTool()
    reg.register(tool)
    audit = RecordingAudit()
    res = reg.dispatch("echo", {"text": "ab", "times": 2}, audit=audit, session="s1")
    assert res == ToolResult(True, "abab")
    assert tool.calls == [{"text": "ab", "times": 2}]
    assert [e[0] for e in audit.events] == ["before", "after"]
    assert audit.events[0][1:4] == ("echo", {"text": "ab", "times": 2}, "s1")
    assert audit.events[1][4] == res


def test_dispatch_unknown_tool_is_refused_and_audited() -> None:
    reg = Registry()
    audit = RecordingAudit()
    res = reg.dispatch("ghost", {}, audit=audit)
    assert not res.ok and "unknown tool" in (res.error or "")
    assert [e[0] for e in audit.events] == ["before", "after"]
    assert audit.events[1][4] == res


@pytest.mark.parametrize(
    "args",
    [{}, {"text": ""}, {"text": "x", "extra": 1}, {"text": "x", "times": True}, None, "x"],
)
def test_dispatch_validates_args_before_calling_tool(args: object) -> None:
    reg = Registry()
    tool = EchoTool()
    reg.register(tool)
    res = reg.dispatch("echo", args, audit=RecordingAudit())  # type: ignore[arg-type]
    assert not res.ok
    assert "invalid arguments" in (res.error or "")
    assert tool.calls == []  # the tool never saw the bad arguments


def test_dispatch_refuses_network_tool_unless_allowed() -> None:
    reg = Registry()
    tool = NetTool()
    reg.register(tool)
    res = reg.dispatch("fetch", {"text": "x"}, audit=RecordingAudit())
    assert not res.ok and "network" in (res.error or "")
    assert tool.calls == []

    allowed = Registry(network_allow_tools={"fetch"})
    tool2 = NetTool()
    allowed.register(tool2)
    assert allowed.dispatch("fetch", {"text": "x"}, audit=RecordingAudit()).ok
    assert tool2.calls == [{"text": "x"}]


def test_dispatch_tool_exception_does_not_propagate() -> None:
    reg = Registry()
    reg.register(RaisingTool())
    audit = RecordingAudit()
    res = reg.dispatch("boom", {"text": "x"}, audit=audit)
    assert not res.ok and "RuntimeError" in (res.error or "") and "kaboom" in (res.error or "")
    assert audit.events[1][4] == res  # the failure itself was audited


def test_dispatch_wrong_return_type_is_failure() -> None:
    reg = Registry()
    reg.register(WrongReturnTool())
    res = reg.dispatch("wrong", {"text": "x"}, audit=RecordingAudit())
    assert not res.ok and "not ToolResult" in (res.error or "")


def test_dispatch_fails_closed_when_before_audit_raises() -> None:
    reg = Registry()
    tool = EchoTool()
    reg.register(tool)
    res = reg.dispatch("echo", {"text": "x"}, audit=BrokenBeforeAudit())
    assert not res.ok and "audit unavailable" in (res.error or "")
    assert tool.calls == []  # unaudited call never ran


def test_dispatch_withholds_result_when_after_audit_raises() -> None:
    reg = Registry()
    tool = EchoTool()
    reg.register(tool)
    res = reg.dispatch("echo", {"text": "secret"}, audit=BrokenAfterAudit())
    assert not res.ok
    assert res.content == ""
    assert "could not be audited" in (res.error or "")
    assert tool.calls == [{"text": "secret"}]


def test_dispatch_never_raises_on_garbage_inputs() -> None:
    reg = Registry()
    reg.register(EchoTool())
    audit = RecordingAudit()
    for name, args in [(None, {}), (42, {"text": "x"}), ("echo", object())]:
        res = reg.dispatch(name, args, audit=audit)  # type: ignore[arg-type]
        assert isinstance(res, ToolResult) and not res.ok


def test_dispatch_survives_audit_object_without_methods() -> None:
    reg = Registry()
    reg.register(EchoTool())
    res = reg.dispatch("echo", {"text": "x"}, audit=object())  # type: ignore[arg-type]
    assert not res.ok and "audit unavailable" in (res.error or "")
