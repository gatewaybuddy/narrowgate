"""propose_tool — the agent-side half of self-extension.

The agent may WRITE a new tool. It may never LOAD one. This module writes a proposal to
``tools_staged/<name>.py`` and stops. Nothing in this file imports, execs, compiles or otherwise
runs the proposed source; the only thing done to it is ``ast.parse`` (a static parse) so that a
file that can never be activated is refused up front.

Deliberately absent from this module:

* any import of ``importlib`` / ``runpy`` — there is no load path to protect by construction
* the activation-time static checks (see ``narrowgate.activate``). Running them here would give
  the agent a signal to iterate against until its proposal passes. The agent gets one bit back:
  "recorded, NOT active".

See ``docs/SELF-EXTENSION.md`` for the reasoning.
"""

from __future__ import annotations

import ast
import hashlib
import sys
from pathlib import Path
from typing import Any, Protocol

from narrowgate.tools.base import TOOL_NAME_PATTERN, ToolResult, validate_tool_name

MAX_SOURCE_BYTES = 64 * 1024
MAX_DESCRIPTION_CHARS = 512
MAX_RATIONALE_CHARS = 4096

# The default staging directory is cwd-relative, matching every other path in config.example.yaml.
DEFAULT_STAGED_DIR = Path("tools_staged")
# tools/enabled/ lives next to this file. Used only to refuse a proposal that would shadow an
# already-activated tool — nothing in here reads from it.
DEFAULT_ENABLED_DIR = Path(__file__).resolve().parent / "enabled"


class AuditSink(Protocol):
    """The slice of ``narrowgate.audit.AuditLog`` this module needs: ``record(event, **fields)``.

    Injected rather than imported so the module has no opinion about where the log lives.
    """

    def record(self, event: str, /, **fields: Any) -> object: ...


NOT_ACTIVE_NOTICE = (
    "PROPOSAL RECORDED. THE TOOL IS NOT ACTIVE.\n"
    "\n"
    "'{name}' does NOT exist as a tool. It is not registered, it cannot be called, and it will\n"
    "not become callable in this session under any circumstances. Calling '{name}' will fail.\n"
    "\n"
    "The proposed source was written to a staging directory that the harness never imports\n"
    "from. Whether it ever becomes a tool is a separate decision made by a human operator, in a\n"
    "separate process, at a time of their choosing, by running: narrowgate activate {name}\n"
    "You cannot trigger, accelerate, poll, or influence that decision. It may never happen.\n"
    "\n"
    "Do not retry this proposal under the same name; the name is now taken. Do not attempt to\n"
    "work around this by proposing a tool that loads other files — that is the boundary this\n"
    "mechanism exists to hold.\n"
    "\n"
    "staged file: {path}\n"
    "sha256: {digest}"
)


def _refuse(error: str) -> ToolResult:
    return ToolResult(ok=False, content="", error=error)


def check_tool_name(name: object) -> str | None:
    """Return an error string if ``name`` is not an acceptable tool name, else ``None``.

    Delegates to ``narrowgate.tools.base.validate_tool_name`` so the filename rule and the
    registry rule cannot drift. Refuses anything that is not a ``str`` matching
    ``^[a-z][a-z0-9_]{2,63}$`` exactly: path separators, ``..``, absolute paths, null bytes,
    whitespace, uppercase, and non-ASCII lookalikes (the classes are literal ASCII ranges, not
    ``\\w``/``\\d``) all fail. The name becomes a filename stem, so nothing a filesystem could
    interpret may pass.
    """
    try:
        validate_tool_name(name)
    except ValueError as exc:
        return str(exc)
    return None


def _parse_only(source: str) -> str | None:
    """Static parse. Returns an error string on failure, ``None`` on success.

    Uses ``ast.parse`` and nothing else: no ``compile`` to bytecode, no import, no execution.
    A parse failure means the file could never pass activation, so it is refused here rather
    than left to rot in staging.
    """
    try:
        ast.parse(source, filename="<proposal>")
    except SyntaxError as exc:
        return f"source does not parse: line {exc.lineno}: {exc.msg}"
    except (ValueError, RecursionError, MemoryError) as exc:
        # ValueError: null bytes (3.11). RecursionError/MemoryError: pathological nesting.
        return f"source does not parse: {type(exc).__name__}: {exc}"
    return None


def _staged_path(staged_dir: Path, name: str) -> Path | None:
    """Resolve ``<staged_dir>/<name>.py`` and prove it lands directly inside ``staged_dir``.

    Returns ``None`` if the resolved path's parent is not exactly the resolved staging
    directory. With the name regex already enforced this cannot fail, but the check is kept
    so the invariant does not depend on the regex alone.
    """
    root = staged_dir.resolve()
    candidate = (root / f"{name}.py").resolve()
    if candidate.parent != root or candidate.name != f"{name}.py":
        return None
    return candidate


def propose_tool(
    name: str,
    description: str,
    python_source: str,
    rationale: str,
    *,
    staged_dir: Path = DEFAULT_STAGED_DIR,
    enabled_dir: Path = DEFAULT_ENABLED_DIR,
    audit: AuditSink | None = None,
) -> ToolResult:
    """Write a proposed tool to ``tools_staged/<name>.py`` and report that it is NOT active.

    Refuses to:

    * write anywhere other than directly inside ``staged_dir`` — the name is validated against
      the tool-name regex and the resolved path is re-checked, so traversal, absolute paths and
      null bytes cannot escape;
    * overwrite anything: an already-activated tool (``tools/enabled/<name>.py``) or an existing
      staged proposal. A proposal is a decision; changing one is a new proposal under a new name.
      This also closes the window in which an agent could rewrite a file between an operator
      reading it and confirming it;
    * import, exec, compile or run the proposed source in any way. The source is parsed with
      ``ast.parse`` for syntax and then treated as opaque bytes;
    * report anything the activation checks would say. The agent gets no signal to iterate
      against; the only feedback is "recorded" or a reason the proposal was not recorded.

    On success returns ``ok=True`` with content stating unambiguously that the tool is not
    active and that only a human, in another process, can change that.
    """
    err = check_tool_name(name)
    if err is not None:
        return _refuse(err)

    if not isinstance(description, str) or not description.strip():
        return _refuse("description must be a non-empty string")
    if len(description) > MAX_DESCRIPTION_CHARS:
        return _refuse(f"description exceeds {MAX_DESCRIPTION_CHARS} characters")
    if not isinstance(rationale, str) or not rationale.strip():
        return _refuse("rationale must be a non-empty string")
    if len(rationale) > MAX_RATIONALE_CHARS:
        return _refuse(f"rationale exceeds {MAX_RATIONALE_CHARS} characters")
    if not isinstance(python_source, str) or not python_source.strip():
        return _refuse("python_source must be a non-empty string")

    source_bytes = python_source.encode("utf-8")
    if len(source_bytes) > MAX_SOURCE_BYTES:
        return _refuse(f"python_source exceeds {MAX_SOURCE_BYTES} bytes")

    err = _parse_only(python_source)
    if err is not None:
        return _refuse(err)

    target = _staged_path(staged_dir, name)
    if target is None:
        return _refuse("refusing: resolved path is not inside the staging directory")

    if (enabled_dir / f"{name}.py").exists():
        return _refuse(f"refusing: '{name}' is already an activated tool and cannot be replaced")
    if target.exists() or target.is_symlink():
        return _refuse(
            f"refusing: a proposal named '{name}' already exists in staging; proposals are not "
            "overwritten. Use a different name."
        )

    digest = hashlib.sha256(source_bytes).hexdigest()

    try:
        staged_dir.mkdir(exist_ok=True)
        # "x" = O_CREAT|O_EXCL: fails if anything (including a symlink) already exists at the
        # path, so the existence check above is not a TOCTOU window.
        with open(target, "xb") as fh:
            fh.write(source_bytes)
    except FileExistsError:
        return _refuse(f"refusing: '{name}' appeared in staging concurrently; not overwritten")
    except OSError as exc:
        return _refuse(f"could not write staged proposal: {exc.__class__.__name__}: {exc}")

    if audit is not None:
        audit.record(
            "tool_proposed",
            name=name,
            description=description,
            rationale=rationale,
            sha256=digest,
            source_bytes=len(source_bytes),
            source=python_source,
            staged_path=str(target),
            active=False,
        )

    # Belt and braces: the invariant this module exists for, asserted rather than assumed.
    assert name not in sys.modules, "proposal must never be loaded"

    return ToolResult(
        ok=True,
        content=NOT_ACTIVE_NOTICE.format(name=name, path=target, digest=digest),
    )


class ProposeTool:
    """The registrable ``Tool`` wrapping :func:`propose_tool`.

    ``requires_network`` is False. ``run`` refuses everything :func:`propose_tool` refuses and
    additionally refuses unexpected keyword arguments rather than ignoring them.
    """

    name = "propose_tool"
    description = (
        "Propose a new tool for a human operator to review. Writes Python source to a staging "
        "area that is NEVER loaded by the harness. The proposal is NOT active after this call "
        "and may never become active; only a human can activate it, in a separate process."
    )
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "pattern": TOOL_NAME_PATTERN.pattern,
                "description": "Tool name; becomes tools_staged/<name>.py. Must be unused.",
            },
            "description": {"type": "string", "maxLength": MAX_DESCRIPTION_CHARS},
            "python_source": {
                "type": "string",
                "description": (
                    "Full module source containing exactly one class implementing Tool with "
                    "name, description, schema, requires_network and run()."
                ),
            },
            "rationale": {
                "type": "string",
                "maxLength": MAX_RATIONALE_CHARS,
                "description": "Why the operator should consider approving this.",
            },
        },
        "required": ["name", "description", "python_source", "rationale"],
        "additionalProperties": False,
    }
    requires_network = False

    def __init__(
        self,
        *,
        staged_dir: Path = DEFAULT_STAGED_DIR,
        enabled_dir: Path = DEFAULT_ENABLED_DIR,
        audit: AuditSink | None = None,
    ) -> None:
        self._staged_dir = staged_dir
        self._enabled_dir = enabled_dir
        self._audit = audit

    def run(self, **kwargs: Any) -> ToolResult:
        """Dispatch entry point. Refuses unknown or missing arguments; never loads anything."""
        expected = {"name", "description", "python_source", "rationale"}
        extra = set(kwargs) - expected
        if extra:
            return _refuse(f"unexpected arguments: {sorted(extra)}")
        missing = expected - set(kwargs)
        if missing:
            return _refuse(f"missing arguments: {sorted(missing)}")
        return propose_tool(
            kwargs["name"],
            kwargs["description"],
            kwargs["python_source"],
            kwargs["rationale"],
            staged_dir=self._staged_dir,
            enabled_dir=self._enabled_dir,
            audit=self._audit,
        )
