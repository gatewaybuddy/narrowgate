"""narrowgate tool surface.

Importing this package registers nothing. Tools become available only when code constructs a
``Registry`` and calls ``register`` on specific objects; there is no discovery of any kind.
"""

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
from narrowgate.tools.builtin import builtin_tools
from narrowgate.tools.registry import RegistrationError, Registry

__all__ = [
    "ArgumentError",
    "Audit",
    "RegistrationError",
    "Registry",
    "SchemaError",
    "Tool",
    "ToolResult",
    "builtin_tools",
    "validate_args",
    "validate_schema",
    "validate_tool_name",
]
