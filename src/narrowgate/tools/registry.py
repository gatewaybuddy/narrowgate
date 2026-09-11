"""Explicit tool registry and the single dispatch path every tool call goes through.

There is no auto-discovery here, no import scanning, no entry points, no directory walk. A tool
exists in a registry only because code called ``register`` with a specific object. That is the
invariant: a registry that discovers things can be made to discover something an operator never
approved.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from narrowgate.tools.base import (
    ArgumentError,
    Audit,
    SchemaError,
    Tool,
    ToolResult,
    validate_args,
    validate_schema,
    validate_tool_name,
)


class RegistrationError(ValueError):
    """``register`` refused a tool: duplicate name, bad name, bad schema, or bad declaration."""


class Registry:
    """Holds explicitly registered tools and dispatches calls to them under the contract.

    ``network_allow_tools`` is the operator's opt-in list (``network.allow_tools`` in config).
    A tool that declares ``requires_network=True`` and is not named there is never run.
    """

    def __init__(self, network_allow_tools: Iterable[str] = ()) -> None:
        self._tools: dict[str, Tool] = {}
        self._network_allowed: frozenset[str] = frozenset(network_allow_tools)

    def register(self, tool: Tool) -> None:
        """Add ``tool`` under its declared name.

        Raises ``RegistrationError`` and refuses to register: a name already taken; a name
        that does not match ``^[a-z][a-z0-9_]{2,63}$``; a schema the validator cannot fully
        enforce (see ``validate_schema``); a tool whose ``requires_network`` is missing or not
        exactly a ``bool``; a tool without a callable ``run`` or a non-empty str ``description``.
        Registration never imports, instantiates, or calls anything.
        """
        try:
            name = validate_tool_name(getattr(tool, "name", None))
        except ValueError as exc:
            raise RegistrationError(str(exc)) from exc
        if name in self._tools:
            raise RegistrationError(f"tool {name!r} is already registered")
        description = getattr(tool, "description", None)
        if not isinstance(description, str) or not description.strip():
            raise RegistrationError(f"tool {name!r}: description must be a non-empty str")
        requires_network = getattr(tool, "requires_network", None)
        if not isinstance(requires_network, bool):
            raise RegistrationError(f"tool {name!r}: requires_network must be declared as a bool")
        if not callable(getattr(tool, "run", None)):
            raise RegistrationError(f"tool {name!r}: run must be callable")
        try:
            validate_schema(getattr(tool, "schema", None))
        except SchemaError as exc:
            raise RegistrationError(f"tool {name!r}: {exc}") from exc
        self._tools[name] = tool

    def get(self, name: str) -> Tool | None:
        """Return the registered tool called ``name``, or ``None``. Never creates or loads one."""
        if not isinstance(name, str):
            return None
        return self._tools.get(name)

    def names(self) -> list[str]:
        """Sorted names of every registered tool, whether or not it is currently dispatchable."""
        return sorted(self._tools)

    def is_dispatchable(self, name: str) -> bool:
        """True iff ``name`` is registered and, if it needs the network, the operator allowed it."""
        tool = self.get(name)
        if tool is None:
            return False
        return not tool.requires_network or name in self._network_allowed

    def specs(self) -> list[dict]:
        """OpenAI ``tools=[...]`` entries for every dispatchable tool, sorted by name.

        Refuses to advertise a network tool the operator has not allowed: the model is never
        shown a capability that ``dispatch`` would refuse for policy reasons.
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": self._tools[name].description,
                    "parameters": self._tools[name].schema,
                },
            }
            for name in self.names()
            if self.is_dispatchable(name)
        ]

    def dispatch(
        self,
        name: str,
        args: dict,
        *,
        audit: Audit,
        session: str | None = None,
    ) -> ToolResult:
        """Run tool ``name`` with ``args`` and return its result. Never raises.

        In order: audit ``before_call``; refuse an unknown name; refuse a network tool the
        operator has not allowed; validate ``args`` against the schema and refuse on any
        violation, before the tool sees them; run the tool; audit ``after_call``.

        Refuses to run the tool if ``before_call`` raises (fail closed: an unaudited call does
        not happen). If the tool raises, or returns something that is not a ``ToolResult``, the
        outcome is ``ok=False``. If ``after_call`` raises, the tool's output is withheld and
        ``ok=False`` is returned, because a result whose outcome was not recorded is not one the
        agent gets to act on. Every one of these paths returns ``ToolResult(ok=False, ...)``
        rather than propagating into the agent loop.
        """
        try:
            return self._dispatch(name, args, audit=audit, session=session)
        except Exception as exc:  # last line of defence: nothing escapes into the loop
            return ToolResult(False, "", f"dispatch failed: {type(exc).__name__}: {exc}")

    def _dispatch(
        self, name: object, args: object, *, audit: Audit, session: str | None
    ) -> ToolResult:
        tool_name = name if isinstance(name, str) else repr(name)
        audit_args: Mapping[str, Any] = args if isinstance(args, Mapping) else {"_raw": repr(args)}

        try:
            audit.before_call(tool=tool_name, args=audit_args, session=session)
        except Exception as exc:
            return ToolResult(
                False, "", f"audit unavailable ({type(exc).__name__}: {exc}); refusing to run"
            )

        result = self._decide_and_run(tool_name, args)

        try:
            audit.after_call(tool=tool_name, args=audit_args, session=session, result=result)
        except Exception as exc:
            return ToolResult(
                False,
                "",
                f"tool {tool_name!r} ran but its outcome could not be audited "
                f"({type(exc).__name__}: {exc}); result withheld",
            )
        return result

    def _decide_and_run(self, name: str, args: object) -> ToolResult:
        tool = self.get(name)
        if tool is None:
            return ToolResult(False, "", f"unknown tool {name!r}")
        if tool.requires_network and name not in self._network_allowed:
            return ToolResult(
                False,
                "",
                f"tool {name!r} requires network access and is not in network.allow_tools",
            )
        try:
            checked = validate_args(tool.schema, args)
        except ArgumentError as exc:
            return ToolResult(False, "", f"invalid arguments for {name!r}: {exc}")
        try:
            result = tool.run(**checked)
        except Exception as exc:
            return ToolResult(False, "", f"tool {name!r} raised {type(exc).__name__}: {exc}")
        if not isinstance(result, ToolResult):
            return ToolResult(
                False, "", f"tool {name!r} returned {type(result).__name__}, not ToolResult"
            )
        return result
