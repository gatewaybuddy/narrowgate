"""Load and validate ``config.yaml``. Fail closed on anything ambiguous.

The contract (``docs/ARCHITECTURE.md``) says a config that cannot be fully validated is a startup
error, never a default. Concretely this module refuses:

* unknown keys at any level (a typo'd key is not silently ignored — it is a rejected config);
* a ``workspace.root`` that resolves (symlinks followed) outside the current working directory;
* a ``network.allow_tools`` entry that is not a syntactically valid tool name, or is duplicated;
* a missing required key, an empty file, or a top-level document that is not a mapping;
* an ``llm.api_key_env`` that names an environment variable which is not set;
* any attempt to switch the audit log off — there is no key for it, so ``extra="forbid"`` rejects
  ``audit.enabled: false`` and every spelling of the same idea.

Secrets never live here. Config carries the *name* of an environment variable; the value is read
from the process environment at load time and is never stored on the config object.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from narrowgate.tools.base import TOOL_NAME_PATTERN

__all__ = [
    "AgentConfig",
    "AuditConfig",
    "Config",
    "ConfigError",
    "LlmConfig",
    "NetworkConfig",
    "TOOL_NAME_RE",
    "WorkspaceConfig",
    "load_config",
    "resolve_api_key",
]

# The registry's definition of a tool name is the only definition: import it, never copy it.
TOOL_NAME_RE = TOOL_NAME_PATTERN

_ENV_NAME_RE = re.compile(r"^[A-Z_][A-Z0-9_]*$")


class ConfigError(ValueError):
    """Raised for any config that cannot be fully validated. Message is operator-readable."""


class _Strict(BaseModel):
    """Base for every config section: unknown keys are an error, not a warning."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


class LlmConfig(_Strict):
    """The ``llm:`` section. ``api_key_env`` is a variable *name*; the value is never stored."""

    base_url: str = Field(min_length=1)
    model: str = Field(min_length=1)
    api_key_env: str | None = None
    timeout_s: float = Field(gt=0)
    max_tokens: int = Field(gt=0)

    @field_validator("base_url")
    @classmethod
    def _http_scheme(cls, v: str) -> str:
        if not (v.startswith("http://") or v.startswith("https://")):
            raise ValueError("llm.base_url must start with http:// or https://")
        return v.rstrip("/")

    @field_validator("api_key_env")
    @classmethod
    def _env_var_name(cls, v: str | None) -> str | None:
        if v is None:
            return None
        if not _ENV_NAME_RE.match(v):
            raise ValueError(
                f"llm.api_key_env must be an environment variable NAME (e.g. NARROWGATE_API_KEY), "
                f"got {v!r}. Never put the key value in config."
            )
        return v


class WorkspaceConfig(_Strict):
    """The ``workspace:`` section. ``root`` is validated against the cwd in :func:`load_config`."""

    root: Path
    max_file_bytes: int = Field(gt=0)


class NetworkConfig(_Strict):
    """The ``network:`` section. Deny-by-default; ``allow_tools`` is the only opt-in."""

    allow_tools: list[str] = Field(default_factory=list)

    @field_validator("allow_tools")
    @classmethod
    def _valid_tool_names(cls, v: list[str]) -> list[str]:
        seen: set[str] = set()
        for name in v:
            if not isinstance(name, str) or not TOOL_NAME_RE.match(name):
                raise ValueError(
                    f"network.allow_tools entry {name!r} is not a valid tool name "
                    f"(must match {TOOL_NAME_RE.pattern})"
                )
            if name in seen:
                raise ValueError(f"network.allow_tools lists {name!r} more than once")
            seen.add(name)
        return v


class AuditConfig(_Strict):
    """The ``audit:`` section. Only a path. There is deliberately no enable/disable key."""

    path: Path

    @field_validator("path")
    @classmethod
    def _non_empty(cls, v: Path) -> Path:
        if str(v) in ("", "."):
            raise ValueError("audit.path must be a file path")
        return v


class AgentConfig(_Strict):
    """The ``agent:`` section."""

    max_turns: int = Field(gt=0)
    require_native_tool_calls: bool


class Config(_Strict):
    """The whole validated config. Every section is required."""

    llm: LlmConfig
    workspace: WorkspaceConfig
    network: NetworkConfig
    audit: AuditConfig
    agent: AgentConfig


def _format_validation_error(exc: ValidationError) -> str:
    lines = []
    for err in exc.errors():
        loc = ".".join(str(p) for p in err["loc"]) or "<root>"
        lines.append(f"  {loc}: {err['msg']}")
    return "config rejected:\n" + "\n".join(lines)


def _validate_workspace_root(root: Path, cwd: Path) -> Path:
    """Resolve ``root`` against ``cwd`` and refuse it unless it stays inside ``cwd``."""
    cwd_resolved = cwd.resolve()
    candidate = root if root.is_absolute() else cwd_resolved / root
    resolved = candidate.resolve()  # follows symlinks; a symlink escape is still an escape
    if not resolved.is_relative_to(cwd_resolved):
        raise ConfigError(
            f"config rejected:\n  workspace.root: {str(root)!r} resolves to {str(resolved)!r}, "
            f"which is outside the current working directory {str(cwd_resolved)!r}"
        )
    if resolved.exists() and not resolved.is_dir():
        raise ConfigError(
            f"config rejected:\n  workspace.root: {str(resolved)!r} exists and is not a directory"
        )
    return resolved


def resolve_api_key(cfg: Config, env: Mapping[str, str] | None = None) -> str | None:
    """Return the API key named by ``llm.api_key_env``, or ``None`` if no variable is configured.

    Refuses to return an empty string: an ``api_key_env`` that is set to a variable which is unset
    or blank raises :class:`ConfigError` naming the variable. The value is returned to the caller
    and is never written onto the config object.
    """
    name = cfg.llm.api_key_env
    if name is None:
        return None
    environ = os.environ if env is None else env
    value = environ.get(name)
    if value is None or value.strip() == "":
        raise ConfigError(
            f"config rejected:\n  llm.api_key_env names {name!r} but that environment variable "
            f"is not set (or is blank). Export it, or set llm.api_key_env to null for an "
            f"unauthenticated endpoint."
        )
    return value


def load_config(
    path: str | Path,
    *,
    cwd: str | Path | None = None,
    env: Mapping[str, str] | None = None,
) -> Config:
    """Load ``path`` as YAML and return a fully validated :class:`Config`.

    Refuses (raises :class:`ConfigError`) rather than defaulting when: the file is missing or not
    a YAML mapping; any required key is absent; any unknown key is present; ``workspace.root``
    resolves outside ``cwd`` (default: the process cwd); an ``allow_tools`` entry is not a valid
    tool name; or ``llm.api_key_env`` names a variable that is unset. Uses ``yaml.safe_load``
    only — no tags, no object construction.
    """
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"config rejected:\n  cannot read {str(path)!r}: {exc}") from exc

    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigError(f"config rejected:\n  {str(path)!r} is not valid YAML: {exc}") from exc

    if raw is None:
        raise ConfigError(f"config rejected:\n  {str(path)!r} is empty")
    if not isinstance(raw, dict):
        raise ConfigError(
            f"config rejected:\n  {str(path)!r} top level must be a mapping, "
            f"got {type(raw).__name__}"
        )

    try:
        cfg = Config.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(_format_validation_error(exc)) from exc

    base = Path.cwd() if cwd is None else Path(cwd)
    root = _validate_workspace_root(cfg.workspace.root, base)
    cfg = cfg.model_copy(update={"workspace": cfg.workspace.model_copy(update={"root": root})})

    # Presence check only. The value is not retained; callers use resolve_api_key() when needed.
    resolve_api_key(cfg, env)
    return cfg
