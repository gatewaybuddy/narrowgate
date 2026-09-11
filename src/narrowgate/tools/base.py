"""Tool protocol, result type, and argument-validation helpers.

Everything in this module refuses to import from the rest of narrowgate: it is the bottom of
the dependency graph so that ``audit.py`` and ``registry.py`` can both depend on it.

The JSON Schema validator here is deliberately a *subset*. ``validate_schema`` refuses any
keyword it does not implement, so a constraint can never be silently unenforced: a schema that
passes registration is a schema whose every keyword is actually checked at dispatch time.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

TOOL_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]{2,63}$")

# Keywords the validator implements. Anything else in a schema is rejected at registration.
_ANNOTATION_KEYWORDS = frozenset({"title", "description", "examples"})
_TYPE_KEYWORDS = frozenset({"type", "enum", "const"})
_OBJECT_KEYWORDS = frozenset({"properties", "required", "additionalProperties"})
_STRING_KEYWORDS = frozenset({"minLength", "maxLength", "pattern"})
_NUMBER_KEYWORDS = frozenset({"minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"})
_ARRAY_KEYWORDS = frozenset({"items", "minItems", "maxItems"})
_SUPPORTED_KEYWORDS = (
    _ANNOTATION_KEYWORDS
    | _TYPE_KEYWORDS
    | _OBJECT_KEYWORDS
    | _STRING_KEYWORDS
    | _NUMBER_KEYWORDS
    | _ARRAY_KEYWORDS
)
_JSON_TYPES = frozenset({"string", "integer", "number", "boolean", "object", "array", "null"})


@dataclass(frozen=True)
class ToolResult:
    """Outcome of one tool call. ``ok=False`` always carries a non-empty ``error``."""

    ok: bool
    content: str
    error: str | None = None

    def __post_init__(self) -> None:
        if not self.ok and not self.error:
            raise ValueError("a failed ToolResult must carry an error message")


@runtime_checkable
class Tool(Protocol):
    """A named capability. Declares its argument schema and whether it needs the network.

    Implementations are ordinary classes; nothing here is auto-discovered.
    """

    name: str  # ^[a-z][a-z0-9_]{2,63}$
    description: str
    schema: dict  # JSON Schema (subset, see validate_schema) for arguments
    requires_network: bool  # declared here, enforced by the registry

    def run(self, **kwargs: Any) -> ToolResult: ...


class Audit(Protocol):
    """What the registry needs from an audit log. The real ``audit.py`` object satisfies this.

    Both hooks are called for every dispatch, including refused ones. If either raises, the
    registry fails closed: the tool does not run (``before_call``) or its result is withheld
    (``after_call``).
    """

    def before_call(self, *, tool: str, args: Mapping[str, Any], session: str | None) -> None: ...

    def after_call(
        self, *, tool: str, args: Mapping[str, Any], session: str | None, result: ToolResult
    ) -> None: ...


class SchemaError(ValueError):
    """A tool schema is malformed or uses a keyword the validator does not implement."""


class ArgumentError(ValueError):
    """Arguments do not satisfy the tool's schema."""


def validate_tool_name(name: object) -> str:
    """Return ``name`` if it matches ``^[a-z][a-z0-9_]{2,63}$``; raise ``ValueError`` otherwise.

    Refuses non-strings, uppercase, leading digits/underscores, punctuation, and anything
    shorter than 3 or longer than 64 characters.
    """
    if not isinstance(name, str):
        raise ValueError(f"tool name must be a str, got {type(name).__name__}")
    if not TOOL_NAME_PATTERN.fullmatch(name):
        raise ValueError(f"tool name {name!r} does not match {TOOL_NAME_PATTERN.pattern}")
    return name


def validate_schema(schema: object) -> None:
    """Raise ``SchemaError`` unless ``schema`` is an object schema this module fully enforces.

    Refuses: non-dict schemas; a top level whose ``type`` is not exactly ``"object"``; a top
    level that does not set ``additionalProperties: false`` (a tool that accepts undeclared
    arguments has not declared its interface); any keyword outside the supported subset;
    ``required`` names that are not in ``properties``; malformed values for any keyword.
    """
    if not isinstance(schema, dict):
        raise SchemaError(f"schema must be a dict, got {type(schema).__name__}")
    if schema.get("type") != "object":
        raise SchemaError("top-level schema must have type 'object'")
    if schema.get("additionalProperties", True) is not False:
        raise SchemaError("top-level schema must set additionalProperties: false")
    _check_subschema(schema, path="$")


def _check_subschema(schema: object, *, path: str) -> None:
    if not isinstance(schema, dict):
        raise SchemaError(f"{path}: subschema must be a dict")
    for key in schema:
        if not isinstance(key, str):
            raise SchemaError(f"{path}: non-string keyword {key!r}")
        if key not in _SUPPORTED_KEYWORDS:
            raise SchemaError(f"{path}: unsupported keyword {key!r}")

    if "type" in schema:
        types = schema["type"]
        as_list = types if isinstance(types, list) else [types]
        if not as_list or any(t not in _JSON_TYPES for t in as_list):
            raise SchemaError(f"{path}: invalid type {types!r}")

    if "enum" in schema:
        if not isinstance(schema["enum"], list) or not schema["enum"]:
            raise SchemaError(f"{path}: enum must be a non-empty list")

    props = schema.get("properties")
    if props is not None:
        if not isinstance(props, dict):
            raise SchemaError(f"{path}: properties must be a dict")
        for pname, sub in props.items():
            if not isinstance(pname, str):
                raise SchemaError(f"{path}: property name {pname!r} is not a str")
            _check_subschema(sub, path=f"{path}.{pname}")

    required = schema.get("required")
    if required is not None:
        if not isinstance(required, list) or not all(isinstance(r, str) for r in required):
            raise SchemaError(f"{path}: required must be a list of str")
        missing = [r for r in required if r not in (props or {})]
        if missing:
            raise SchemaError(f"{path}: required names not in properties: {missing}")

    if "additionalProperties" in schema and not isinstance(schema["additionalProperties"], bool):
        raise SchemaError(f"{path}: additionalProperties must be a bool")

    for key in ("minLength", "maxLength", "minItems", "maxItems"):
        if key in schema and (_is_bool(schema[key]) or not isinstance(schema[key], int)):
            raise SchemaError(f"{path}: {key} must be an int")
        if key in schema and schema[key] < 0:
            raise SchemaError(f"{path}: {key} must be >= 0")

    for key in _NUMBER_KEYWORDS:
        if key in schema and (_is_bool(schema[key]) or not isinstance(schema[key], int | float)):
            raise SchemaError(f"{path}: {key} must be a number")

    if "pattern" in schema:
        if not isinstance(schema["pattern"], str):
            raise SchemaError(f"{path}: pattern must be a str")
        try:
            re.compile(schema["pattern"])
        except re.error as exc:
            raise SchemaError(f"{path}: invalid pattern: {exc}") from exc

    if "items" in schema:
        _check_subschema(schema["items"], path=f"{path}[]")


def validate_args(schema: Mapping[str, Any], args: object) -> dict[str, Any]:
    """Return ``args`` as a plain dict if it satisfies ``schema``; raise ``ArgumentError`` if not.

    Refuses non-mapping arguments, non-string keys, undeclared keys (unless the schema opted
    in with ``additionalProperties: true``), missing required keys, and any value that violates
    its property subschema. ``bool`` is never accepted where ``integer`` or ``number`` is
    declared, even though Python treats it as an int.
    """
    if not isinstance(args, Mapping):
        raise ArgumentError(f"arguments must be an object, got {type(args).__name__}")
    for key in args:
        if not isinstance(key, str):
            raise ArgumentError(f"argument name {key!r} is not a str")
    errors: list[str] = []
    _check_value(schema, dict(args), path="$", errors=errors)
    if errors:
        raise ArgumentError("; ".join(errors))
    return dict(args)


def _check_value(schema: Mapping[str, Any], value: Any, *, path: str, errors: list[str]) -> None:
    if "type" in schema:
        types = schema["type"]
        as_list = types if isinstance(types, list) else [types]
        if not any(_matches_type(value, t) for t in as_list):
            errors.append(f"{path}: expected {types}, got {_json_type_name(value)}")
            return
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: not one of {schema['enum']!r}")
    if "const" in schema and value != schema["const"]:
        errors.append(f"{path}: must equal {schema['const']!r}")

    if isinstance(value, str):
        n = len(value)
        if "minLength" in schema and n < schema["minLength"]:
            errors.append(f"{path}: shorter than {schema['minLength']}")
        if "maxLength" in schema and n > schema["maxLength"]:
            errors.append(f"{path}: longer than {schema['maxLength']}")
        if "pattern" in schema and not re.search(schema["pattern"], value):
            errors.append(f"{path}: does not match {schema['pattern']!r}")

    if isinstance(value, int | float) and not _is_bool(value):
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(f"{path}: below minimum {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            errors.append(f"{path}: above maximum {schema['maximum']}")
        if "exclusiveMinimum" in schema and value <= schema["exclusiveMinimum"]:
            errors.append(f"{path}: not above {schema['exclusiveMinimum']}")
        if "exclusiveMaximum" in schema and value >= schema["exclusiveMaximum"]:
            errors.append(f"{path}: not below {schema['exclusiveMaximum']}")

    if isinstance(value, list):
        if "minItems" in schema and len(value) < schema["minItems"]:
            errors.append(f"{path}: fewer than {schema['minItems']} items")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            errors.append(f"{path}: more than {schema['maxItems']} items")
        if "items" in schema:
            for i, item in enumerate(value):
                _check_value(schema["items"], item, path=f"{path}[{i}]", errors=errors)

    if isinstance(value, dict):
        props: Mapping[str, Any] = schema.get("properties", {})
        for req in schema.get("required", []):
            if req not in value:
                errors.append(f"{path}: missing required {req!r}")
        if schema.get("additionalProperties", False) is not True:
            extra = sorted(k for k in value if k not in props)
            if extra:
                errors.append(f"{path}: undeclared arguments {extra}")
        for pname, sub in props.items():
            if pname in value:
                _check_value(sub, value[pname], path=f"{path}.{pname}", errors=errors)


def _is_bool(value: object) -> bool:
    return isinstance(value, bool)


def _matches_type(value: Any, json_type: str) -> bool:
    match json_type:
        case "string":
            return isinstance(value, str)
        case "boolean":
            return _is_bool(value)
        case "integer":
            return isinstance(value, int) and not _is_bool(value)
        case "number":
            return isinstance(value, int | float) and not _is_bool(value)
        case "object":
            return isinstance(value, dict)
        case "array":
            return isinstance(value, list)
        case "null":
            return value is None
    return False


def _json_type_name(value: Any) -> str:
    for name in ("boolean", "integer", "number", "string", "array", "object", "null"):
        if _matches_type(value, name):
            return name
    return type(value).__name__


def require_str(
    kwargs: Mapping[str, Any], key: str, *, max_len: int, allow_empty: bool = False
) -> str:
    """Fetch a string argument a tool is about to act on, re-checking it inside the tool.

    Tools call this even though the registry already validated the schema, because invariant 3
    says a tool validates its own inputs. Refuses a missing key, a non-``str``, a string longer
    than ``max_len``, an empty string unless ``allow_empty``, and any string containing a NUL
    byte. Raises ``ArgumentError``.
    """
    if key not in kwargs:
        raise ArgumentError(f"missing argument {key!r}")
    value = kwargs[key]
    if not isinstance(value, str):
        raise ArgumentError(f"{key!r} must be a str, got {type(value).__name__}")
    if not value and not allow_empty:
        raise ArgumentError(f"{key!r} must not be empty")
    if len(value) > max_len:
        raise ArgumentError(f"{key!r} longer than {max_len} characters")
    if "\x00" in value:
        raise ArgumentError(f"{key!r} contains a NUL byte")
    return value
