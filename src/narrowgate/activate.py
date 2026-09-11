"""activate — the human-side gate that promotes a staged tool to ``tools/enabled/``.

This runs in the OPERATOR's process, never the agent's. It

1. prints the full staged source, numbered, so the operator reads what they approve;
2. runs static checks over the ``ast`` (never by importing) and prints every violation with
   its line number and source line;
3. requires the operator to type the tool's name — not ``y`` — to confirm;
4. copies the *reviewed bytes* into ``tools/enabled/<name>.py`` and appends an audit entry with
   the source hash and the approving user;
5. does NOT load the tool. Registration happens on the next harness start.

Rejection is terminal for the proposal: nothing here offers a fix-and-retry loop, and the staged
file is left in place (with its name burned in ``propose_tool``) as evidence for the operator.

WHAT THE STATIC CHECKER IS — AND IS NOT
---------------------------------------
It is a review aid built on a DENYLIST of modules, names and attribute shapes. A denylist over
Python is inherently incomplete: the language exposes reflection through strings, and a syntax
walk cannot follow strings. ``docs/SELF-EXTENSION.md`` states the control is human review, not
containment; every rule here exists to make the review *easier*, not to replace it.

Known gaps, stated so nobody mistakes silence for coverage:

* **No dataflow.** ``p = "out.txt"; open(p, "w")`` is refused (path not literal) even though it
  is harmless; conversely, the checker only sees names and attributes *as written* — it is the
  absence of any string-driven reflection API from the allowed surface that carries the weight.
  Every such API I know of is denied (``getattr`` with non-literal names, ``operator.attrgetter``,
  ``pydoc.locate``, ``unittest.mock``, ``typing.get_type_hints`` on string annotations, ...).
  One I do not know of gets through.
* **Third-party modules** with eval-like features (``jinja2``, ``pandas.eval``, ``numexpr``,
  ``sympy``) are not denied because narrowgate does not depend on them; if the deployment
  environment has them installed, an approved tool may import them.
* **Reads are unrestricted.** The spec only constrains write-mode ``open``. A tool may read any
  file the harness can read. This is by spec and is the operator's to judge.
* **Library APIs that take a path and write to it** are enumerated, not derived: ``open`` (with
  its mode) and the pathlib/gzip/logging/argparse names in ``_WRITE_ATTRS`` are covered;
  ``xml.etree.ElementTree.ElementTree.write(path)`` and its cousins are not. The operator reads
  the source; this list makes the common ones loud.
* **Resource exhaustion** (``while True``, memory bombs, ReDoS via ``re``) is not checked.
* **Module-level assignments** may call functions (``X = build()``) and so run at harness start;
  only allowed names can be reached, but they *do* run before any tool is dispatched.
* **Module aliasing through allowed modules** is closed only for the attribute names in
  ``DENIED_ATTRS`` (``pathlib.os``, ``random._os``, ``dataclasses.sys``...). A stdlib module that
  re-exports a dangerous module under a novel name is not caught.
"""

from __future__ import annotations

import ast
import getpass
import hashlib
import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

from narrowgate.tools.propose import (
    DEFAULT_ENABLED_DIR,
    DEFAULT_STAGED_DIR,
    MAX_SOURCE_BYTES,
    AuditSink,
    check_tool_name,
)

# --- denylists ------------------------------------------------------------------------------

# From docs/SELF-EXTENSION.md, verbatim.
SPEC_DENIED_MODULES: frozenset[str] = frozenset(
    {"os", "subprocess", "socket", "ctypes", "importlib", "builtins"}
)
# Extensions, each of which is a documented route to the same capabilities:
#   process/exec   : sys (sys.modules, _getframe), runpy, code, codeop, pty, signal,
#                    multiprocessing, asyncio (create_subprocess_*), webbrowser, timeit (exec),
#                    doctest, unittest (mock.patch imports by string), pdb/bdb/trace/profile
#   object graphs  : inspect (frames), gc (get_referents), types (CodeType), pydoc (locate)
#   serialisation  : pickle, marshal, shelve, dbm (arbitrary object construction)
#   filesystem     : shutil, tempfile, fileinput (inplace), zipfile, tarfile, sqlite3, mailbox,
#                    compileall, py_compile (write anywhere); glob is fine (read-only)
#   string-resolved: logging.config (dictConfig resolves "class": "os.system" and CALLS it),
#                    site (addsitedir executes .pth files), ensurepip/venv/pip (spawn installers)
#   C-level twins  : posix, nt, _thread, _posixsubprocess, _socket, _ctypes, _imp, _io, _pickle,
#                    zipimport, pkgutil
# Matching is by dotted prefix: "logging.config" denies "logging.config.x" but not "logging".
EXTENDED_DENIED_MODULES: frozenset[str] = frozenset(
    {
        "sys", "runpy", "code", "codeop", "pty", "signal", "multiprocessing", "asyncio",
        "webbrowser", "antigravity", "timeit", "doctest", "unittest", "pdb", "bdb", "trace",
        "profile", "cProfile", "inspect", "gc", "types", "pydoc", "pickle", "marshal", "shelve",
        "dbm", "shutil", "tempfile", "fileinput", "zipfile", "tarfile", "sqlite3", "mailbox",
        "compileall", "py_compile", "logging.config", "site", "ensurepip", "venv", "pip",
        "idlelib", "posix", "nt", "_thread", "_posixsubprocess", "_socket", "_ctypes", "_imp",
        "_io", "_pickle", "zipimport", "pkgutil",
    }
)  # fmt: skip
DENIED_MODULES: frozenset[str] = SPEC_DENIED_MODULES | EXTENDED_DENIED_MODULES

# Importing one of these while declaring requires_network = False is a contradiction; the
# registry gates on the declaration, so the declaration must be honest.
NETWORK_MODULES: frozenset[str] = frozenset(
    {
        "urllib", "http", "httpx", "requests", "ftplib", "smtplib", "poplib", "imaplib",
        "telnetlib", "xmlrpc", "ssl", "socketserver", "aiohttp", "websocket", "websockets",
        "nntplib", "logging.handlers",
    }
)  # fmt: skip

# Names that may never appear, in any context. Spec: eval, exec, compile, __import__.
FORBIDDEN_NAMES: frozenset[str] = frozenset(
    {
        "eval", "exec", "compile", "__import__", "globals", "locals", "vars", "breakpoint",
        "help", "input", "exit", "quit", "__builtins__", "__loader__", "__spec__",
        "__build_class__", "memoryview",
    }
)  # fmt: skip

# Builtins allowed ONLY as a direct call with a literal, non-dunder, non-denied attribute name.
# Any other appearance (aliasing, passing as a value) is refused.
GUARDED_BUILTINS: frozenset[str] = frozenset({"getattr", "setattr", "delattr", "hasattr"})

# Attribute names refused wherever they appear.
_FRAME_ATTRS = {
    "f_globals", "f_locals", "f_builtins", "f_back", "f_code", "f_trace", "gi_frame", "gi_code",
    "cr_frame", "cr_code", "ag_frame", "ag_code", "tb_frame", "tb_next", "co_code", "co_consts",
}  # fmt: skip
# Path-taking constructors/functions that create or write files without a call named "open".
_WRITE_ATTRS = {
    "write_text", "write_bytes", "unlink", "rmdir", "mkdir", "rename", "replace", "touch",
    "chmod", "symlink_to", "hardlink_to", "link_to", "rmtree", "remove", "removedirs",
    "makedirs", "system", "popen", "FileType", "FileIO", "GzipFile", "BZ2File", "LZMAFile",
    "basicConfig", "FileHandler", "RotatingFileHandler", "TimedRotatingFileHandler",
    "WatchedFileHandler", "addsitedir",
}  # fmt: skip
# Handlers that open sockets from inside logging, invisible to the requires_network declaration.
_NETWORK_ATTRS = {
    "HTTPHandler", "SocketHandler", "DatagramHandler", "SMTPHandler", "SysLogHandler",
}  # fmt: skip
# String-driven reflection: attribute names passed as strings, type expressions evaluated from
# strings, or module paths resolved from strings. The AST cannot see inside a string.
_REFLECTION_ATTRS = {
    "attrgetter", "methodcaller", "get_type_hints", "get_annotations", "_eval_type",
    "evaluate_forward_ref", "_evaluate", "unsafe_load", "unsafe_load_all", "full_load",
    "full_load_all", "UnsafeLoader", "FullLoader", "Loader", "CLoader", "CFullLoader",
    "CUnsafeLoader", "locate", "_importer", "_resolve", "dictConfig", "fileConfig", "listen",
    "resolve_name", "import_module", "create_model",
}  # fmt: skip
# Callables that accept a TYPE EXPRESSION and evaluate it if given as a string (pydantic
# resolves forward references with eval in the caller's namespace). Allowed only with a
# Name/Attribute/Subscript argument, i.e. a type written as code the checker can see.
TYPE_EVALUATING_CALLS: frozenset[str] = frozenset({"TypeAdapter", "ForwardRef"})
DENIED_ATTRS: frozenset[str] = frozenset(
    DENIED_MODULES
    | {f"_{m}" for m in DENIED_MODULES if not m.startswith("_")}
    | _FRAME_ATTRS
    | _WRITE_ATTRS
    | _NETWORK_ATTRS
    | _REFLECTION_ATTRS
)

# The only dunder attributes a tool has a legitimate reason to touch.
ALLOWED_DUNDER_ATTRS: frozenset[str] = frozenset({"__init__", "__post_init__"})

# Modes that can create or modify a file. Spec says w/a; x (create) and + (update) also write.
WRITE_MODE_CHARS: frozenset[str] = frozenset("wax+")

REQUIRED_TOOL_ATTRS: tuple[str, ...] = ("name", "description", "schema", "requires_network")

# Top-level statements that do not execute arbitrary code at import time.
_ALLOWED_TOPLEVEL = (ast.Import, ast.ImportFrom, ast.ClassDef, ast.FunctionDef, ast.Assign,
                     ast.AnnAssign)  # fmt: skip


# --- results ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class Violation:
    """One static-check failure, with enough context to be printed and understood."""

    line: int
    reason: str
    source: str

    def render(self) -> str:
        return f"  line {self.line}: {self.reason}\n    | {self.source}"


@dataclass(frozen=True)
class ActivationRecord:
    """What was activated, by whom, and the hash of exactly the bytes that were installed."""

    name: str
    sha256: str
    approved_by: str
    staged_path: Path
    enabled_path: Path


class ActivationError(Exception):
    """Base for every way activation can end without installing the tool."""


class ActivationRejected(ActivationError):
    """Static checks failed. Terminal for this proposal; carries the violations."""

    def __init__(self, name: str, violations: list[Violation]) -> None:
        super().__init__(f"'{name}' rejected: {len(violations)} violation(s)")
        self.name = name
        self.violations = violations


class ActivationDeclined(ActivationError):
    """The operator did not type the tool's name."""


class ActivationAborted(ActivationError):
    """The staged file changed between review and confirmation, or could not be installed."""


# --- the checker -----------------------------------------------------------------------------


def _is_dunder(ident: str) -> bool:
    return len(ident) > 4 and ident.startswith("__") and ident.endswith("__")


def _prefixes(dotted: str) -> list[str]:
    """``"a.b.c"`` -> ``["a", "a.b", "a.b.c"]``."""
    parts = dotted.split(".")
    return [".".join(parts[: i + 1]) for i in range(len(parts))]


def _matches(dotted: str, modules: frozenset[str]) -> bool:
    return any(prefix in modules for prefix in _prefixes(dotted))


def _literal_str(node: ast.expr | None) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


class _Checker(ast.NodeVisitor):
    """Walks the tree once, collecting every violation. Never evaluates anything."""

    def __init__(self, lines: list[str], workspace_root: Path) -> None:
        self.lines = lines
        self.root = workspace_root.resolve()
        self.violations: list[Violation] = []
        self.imported_network_modules: list[tuple[int, str]] = []

    # helpers

    def fail(self, node: ast.AST, reason: str) -> None:
        line = getattr(node, "lineno", 0)
        src = self.lines[line - 1].rstrip() if 0 < line <= len(self.lines) else "<no source>"
        self.violations.append(Violation(line=line, reason=reason, source=src))

    def _check_module_name(self, node: ast.stmt, dotted: str) -> None:
        if _matches(dotted, DENIED_MODULES):
            self.fail(node, f"import of denied module '{dotted}'")
        elif _matches(dotted, NETWORK_MODULES):
            self.imported_network_modules.append((node.lineno, dotted))

    def _check_literal_attr_name(self, node: ast.Call, arg: ast.expr | None, what: str) -> bool:
        name = _literal_str(arg)
        if name is None:
            self.fail(node, f"{what} with a non-literal attribute name (cannot be reviewed)")
            return False
        if _is_dunder(name):
            self.fail(node, f"{what} on dunder name '{name}'")
            return False
        if name in DENIED_ATTRS:
            self.fail(node, f"{what} on denied attribute '{name}'")
            return False
        return True

    # imports

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self._check_module_name(node, alias.name)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.level or node.module is None:
            self.fail(node, "relative import (cannot be reviewed statically)")
            return
        self._check_module_name(node, node.module)
        for alias in node.names:
            if alias.name == "*":
                self.fail(node, "star import (imported names cannot be reviewed)")
            elif f"{node.module}.{alias.name}" in DENIED_MODULES:
                self.fail(node, f"import of denied module '{node.module}.{alias.name}'")
            elif alias.name in DENIED_ATTRS or _is_dunder(alias.name):
                # from logging import config / from typing import get_type_hints
                self.fail(node, f"import of denied name '{alias.name}'")

    # names and attributes

    def visit_Name(self, node: ast.Name) -> None:
        if node.id in FORBIDDEN_NAMES:
            self.fail(node, f"use of forbidden name '{node.id}'")
        elif node.id in GUARDED_BUILTINS:
            self.fail(node, f"'{node.id}' may only be called directly with a literal name")
        elif node.id == "open":
            self.fail(node, "'open' may only be called directly")
        elif _is_dunder(node.id) and isinstance(node.ctx, ast.Load):
            self.fail(node, f"read of dunder name '{node.id}'")

    def visit_Attribute(self, node: ast.Attribute) -> None:
        attr = node.attr
        if _is_dunder(attr) and attr not in ALLOWED_DUNDER_ATTRS:
            self.fail(node, f"access to dunder attribute '.{attr}'")
        elif attr in DENIED_ATTRS:
            self.fail(node, f"access to denied attribute '.{attr}'")
        elif attr == "open":
            self.fail(node, "'.open' may only be called directly")
        self.visit(node.value)

    # calls

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        handled_func = False

        if isinstance(func, ast.Name):
            if func.id in GUARDED_BUILTINS:
                self._check_literal_attr_name(
                    node, node.args[1] if len(node.args) > 1 else None, f"{func.id}()"
                )
                handled_func = True
            elif func.id == "open":
                self._check_open(node, is_method=False)
                handled_func = True
            elif func.id in TYPE_EVALUATING_CALLS:
                self._check_type_arg(node, func.id)
        elif isinstance(func, ast.Attribute):
            if func.attr == "open":
                self._check_open(node, is_method=True)
                # still check the receiver chain (e.g. pathlib.os.open)
                self.visit(func.value)
                handled_func = True
            elif (
                func.attr == "load"
                and isinstance(func.value, ast.Name)
                and (func.value.id == "yaml")
            ):
                self.fail(node, "yaml.load() can construct arbitrary objects; use yaml.safe_load")
            elif func.attr in TYPE_EVALUATING_CALLS:
                self._check_type_arg(node, func.attr)

        if not handled_func:
            self.visit(func)
        for arg in node.args:
            self.visit(arg)
        for kw in node.keywords:
            self.visit(kw.value)

    def _check_type_arg(self, node: ast.Call, what: str) -> None:
        """``TypeAdapter("...")`` evaluates its string; only a type written as code is allowed."""
        first = node.args[0] if node.args else None
        if first is None or not isinstance(first, ast.Name | ast.Attribute | ast.Subscript):
            self.fail(node, f"{what}() with a non-type argument (strings are evaluated)")

    def _check_open(self, node: ast.Call, *, is_method: bool) -> None:
        """Refuse write-mode opens unless the target is a literal path under the workspace root.

        Cannot see through variables: a non-literal mode or path is refused, not guessed.
        A write-mode ``.open()`` method call is always refused because the receiver (the path)
        is an object, not a literal.
        """
        if any(isinstance(a, ast.Starred) for a in node.args) or any(
            kw.arg is None for kw in node.keywords
        ):
            self.fail(node, "open() with *args/**kwargs (mode cannot be reviewed)")
            return
        kwargs = {kw.arg: kw.value for kw in node.keywords}
        # builtin: open(file, mode, ...) -> mode is positional arg 1.
        # method:  path.open(mode, ...)  -> mode is positional arg 0 (path is the receiver).
        mode_pos = 0 if is_method else 1
        mode_node = node.args[mode_pos] if len(node.args) > mode_pos else kwargs.get("mode")
        if mode_node is None:
            return  # default mode is 'r'
        mode = _literal_str(mode_node)
        if mode is None:
            self.fail(node, "open() with a non-literal mode (cannot be reviewed)")
            return
        if not (set(mode) & WRITE_MODE_CHARS):
            return
        if is_method:
            self.fail(node, f".open(mode={mode!r}) writes via an object path (cannot be reviewed)")
            return
        path_node = node.args[0] if node.args else kwargs.get("file")
        path = _literal_str(path_node)
        if path is None:
            self.fail(node, f"open(mode={mode!r}) with a non-literal path (cannot be reviewed)")
            return
        target = Path(path)
        if not target.is_absolute():
            target = self.root / target
        if not target.resolve().is_relative_to(self.root):
            self.fail(node, f"open(mode={mode!r}) targets {path!r}, outside the workspace root")

    # annotations: a string annotation is an expression the checker cannot see, and several
    # libraries (typing.get_type_hints, pydantic) evaluate them.

    def _check_annotation(self, ann: ast.expr | None) -> None:
        if ann is not None and _literal_str(ann) is not None:
            self.fail(ann, "string annotation (evaluated by typing/pydantic; cannot be reviewed)")

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self._check_annotation(node.annotation)
        self.generic_visit(node)

    def visit_arg(self, node: ast.arg) -> None:
        self._check_annotation(node.annotation)
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._check_annotation(node.returns)
        self.generic_visit(node)

    visit_AsyncFunctionDef = visit_FunctionDef  # type: ignore[assignment]


def _check_toplevel(tree: ast.Module, checker: _Checker) -> None:
    for stmt in tree.body:
        if isinstance(stmt, _ALLOWED_TOPLEVEL):
            continue
        if isinstance(stmt, ast.Expr) and _literal_str(stmt.value) is not None:
            continue  # docstring
        checker.fail(
            stmt,
            "module-level statement runs at harness start; only imports, class/def and "
            "assignments are allowed",
        )


def _class_attr(cls: ast.ClassDef, attr: str) -> ast.expr | None:
    """Return the value node of a class-body assignment ``attr = ...`` / ``attr: T = ...``."""
    for stmt in cls.body:
        if isinstance(stmt, ast.Assign):
            if any(isinstance(t, ast.Name) and t.id == attr for t in stmt.targets):
                return stmt.value
        elif isinstance(stmt, ast.AnnAssign):
            if isinstance(stmt.target, ast.Name) and stmt.target.id == attr:
                return stmt.value
    return None


def _check_tool_class(tree: ast.Module, tool_name: str, checker: _Checker) -> None:
    """Require exactly one top-level class with a ``run`` method and honest declarations.

    Refuses: zero or several candidate classes; a class whose ``name`` literal differs from the
    staged name (the registry would install it under a different name than was approved);
    ``requires_network`` that is missing, non-literal, or not a bool; and a
    ``requires_network = False`` alongside an import of a network-capable module.
    """
    candidates = [
        stmt
        for stmt in tree.body
        if isinstance(stmt, ast.ClassDef)
        and any(isinstance(b, ast.FunctionDef) and b.name == "run" for b in stmt.body)
    ]
    if len(candidates) != 1:
        checker.fail(
            tree if not candidates else candidates[1],
            f"expected exactly one top-level class with a run() method, found {len(candidates)}",
        )
        return
    cls = candidates[0]

    for attr in REQUIRED_TOOL_ATTRS:
        if _class_attr(cls, attr) is None:
            checker.fail(cls, f"Tool class must declare '{attr}' in its class body")

    name_value = _literal_str(_class_attr(cls, "name"))
    if name_value is not None and name_value != tool_name:
        checker.fail(cls, f"Tool class declares name {name_value!r}; staged as {tool_name!r}")
    elif _class_attr(cls, "name") is not None and name_value is None:
        checker.fail(cls, "'name' must be a string literal")

    rn = _class_attr(cls, "requires_network")
    if rn is not None:
        if not (isinstance(rn, ast.Constant) and isinstance(rn.value, bool)):
            checker.fail(rn, "'requires_network' must be the literal True or False")
        elif rn.value is False and checker.imported_network_modules:
            for line, mod in checker.imported_network_modules:
                checker.violations.append(
                    Violation(
                        line=line,
                        reason=f"imports network module '{mod}' but declares "
                        "requires_network = False",
                        source=checker.lines[line - 1].rstrip(),
                    )
                )


def static_check(source: str, *, tool_name: str, workspace_root: Path) -> list[Violation]:
    """Statically check proposed tool source. Returns every violation found (empty = pass).

    Parses with ``ast.parse`` only. Refuses to import, exec, compile to bytecode, or evaluate
    any part of the source, including annotations. A source that does not parse is a single
    violation at the failing line.
    """
    lines = source.splitlines()
    try:
        tree = ast.parse(source, filename=f"<staged:{tool_name}>")
    except SyntaxError as exc:
        line = exc.lineno or 0
        src = lines[line - 1].rstrip() if 0 < line <= len(lines) else "<no source>"
        return [Violation(line=line, reason=f"does not parse: {exc.msg}", source=src)]
    except (ValueError, RecursionError, MemoryError) as exc:
        return [Violation(line=0, reason=f"does not parse: {exc}", source="<no source>")]

    checker = _Checker(lines, workspace_root)
    checker.visit(tree)
    _check_toplevel(tree, checker)
    _check_tool_class(tree, tool_name, checker)
    return sorted(checker.violations, key=lambda v: (v.line, v.reason))


# --- the gate --------------------------------------------------------------------------------


def load_staged(name: str, staged_dir: Path = DEFAULT_STAGED_DIR) -> tuple[Path, bytes]:
    """Read ``<staged_dir>/<name>.py`` once and return (path, bytes).

    Refuses names that fail the tool-name regex, paths that resolve outside ``staged_dir``,
    symlinks, non-regular files, and files over ``MAX_SOURCE_BYTES``. The bytes returned are
    the only bytes ever considered: what is displayed, checked, hashed and installed.
    """
    err = check_tool_name(name)
    if err is not None:
        raise ActivationAborted(err)
    root = staged_dir.resolve()
    path = root / f"{name}.py"
    if path.resolve().parent != root:
        raise ActivationAborted("staged path resolves outside the staging directory")
    if path.is_symlink():
        raise ActivationAborted(f"{path} is a symlink; refusing")
    if not path.is_file():
        raise ActivationAborted(f"no staged proposal named '{name}' at {path}")
    if path.stat().st_size > MAX_SOURCE_BYTES:
        raise ActivationAborted(f"{path} exceeds {MAX_SOURCE_BYTES} bytes; refusing")
    return path, path.read_bytes()


def render_source(source: str) -> str:
    """Number every line. Full source, no truncation — the operator reads all of it."""
    lines = source.splitlines() or [""]
    width = len(str(len(lines)))
    return "\n".join(f"{i:>{width}} | {line}" for i, line in enumerate(lines, 1))


def _emit(out: TextIO, *parts: str) -> None:
    out.write("\n".join(parts) + "\n")
    out.flush()


def activate(
    name: str,
    *,
    staged_dir: Path = DEFAULT_STAGED_DIR,
    enabled_dir: Path = DEFAULT_ENABLED_DIR,
    workspace_root: Path,
    confirm: Callable[[str], str],
    out: TextIO = sys.stdout,
    audit: AuditSink | None = None,
    approver: str | None = None,
) -> ActivationRecord:
    """Review, check, confirm and install one staged tool. Never loads it.

    Refuses to: skip printing the full source; proceed past any static-check violation
    (raises ``ActivationRejected`` — terminal, no retry path); accept anything other than the
    exact tool name as confirmation (``ActivationDeclined``); install bytes other than the ones
    that were displayed (``ActivationAborted`` if the staged file changed underneath the
    review); overwrite an existing enabled tool; import the tool after installing it.
    """
    staged_path, source_bytes = load_staged(name, staged_dir)
    digest = hashlib.sha256(source_bytes).hexdigest()
    approver = approver or getpass.getuser()
    try:
        source = source_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ActivationAborted(f"staged file is not UTF-8: {exc}") from exc

    _emit(
        out,
        f"=== staged tool: {name}",
        f"    path:   {staged_path}",
        f"    sha256: {digest}",
        f"    bytes:  {len(source_bytes)}",
        "",
        render_source(source),
        "",
    )

    violations = static_check(source, tool_name=name, workspace_root=workspace_root)
    if violations:
        _emit(out, f"REJECTED: {len(violations)} violation(s). This proposal is terminal.")
        for v in violations:
            _emit(out, v.render())
        _audit(
            audit,
            "tool_rejected",
            name=name,
            sha256=digest,
            reviewer=approver,
            violations=[
                {"line": v.line, "reason": v.reason, "source": v.source} for v in violations
            ],  # fmt: skip
        )
        raise ActivationRejected(name, violations)

    _emit(out, "Static checks passed. This is NOT a safety guarantee; you read the source above.")
    typed = confirm(f"Type the tool name '{name}' to activate it (anything else declines): ")
    if typed.rstrip("\r\n") != name:
        _emit(out, "Declined. Nothing was installed.")
        _audit(audit, "tool_declined", name=name, sha256=digest, reviewer=approver)
        raise ActivationDeclined(f"'{name}' not confirmed")

    # Install exactly the reviewed bytes. If the staged file changed while the operator was
    # reading, something else is writing to staging — stop, do not install either version.
    if staged_path.read_bytes() != source_bytes:
        _audit(audit, "tool_activation_aborted", name=name, sha256=digest,
               reviewer=approver, reason="staged file changed during review")  # fmt: skip
        raise ActivationAborted(f"{staged_path} changed during review; nothing installed")

    enabled_path = (enabled_dir / f"{name}.py").resolve()
    if enabled_path.parent != enabled_dir.resolve():
        raise ActivationAborted("enabled path resolves outside tools/enabled")
    try:
        enabled_dir.mkdir(parents=True, exist_ok=True)
        with open(enabled_path, "xb") as fh:
            fh.write(source_bytes)
    except FileExistsError as exc:
        raise ActivationAborted(f"{enabled_path} already exists; refusing to overwrite") from exc
    staged_path.unlink()

    _audit(
        audit,
        "tool_activated",
        name=name,
        sha256=digest,
        approved_by=approver,
        staged_path=str(staged_path),
        enabled_path=str(enabled_path),
        source_bytes=len(source_bytes),
    )
    _emit(
        out,
        f"Activated: {enabled_path}",
        "NOT loaded. The tool is registered on the next harness start.",
    )
    assert name not in sys.modules, "activation must never load the tool"
    return ActivationRecord(
        name=name,
        sha256=digest,
        approved_by=approver,
        staged_path=staged_path,
        enabled_path=enabled_path,
    )


def _audit(sink: AuditSink | None, event: str, /, **fields: Any) -> None:
    if sink is not None:
        sink.record(event, **fields)


def main(argv: Iterable[str] | None = None) -> int:
    """CLI entry: ``narrowgate activate <name>``.

    Refuses to run without an interactive stdin: the typed-name confirmation is a human
    gesture, and piping ``yes`` into it would turn it back into the ``y/N`` it replaces.
    Exit codes: 0 activated · 2 rejected by static checks · 3 declined · 4 aborted.
    """
    import argparse

    from narrowgate.audit import AuditLog
    from narrowgate.config import ConfigError, load_config

    parser = argparse.ArgumentParser(prog="narrowgate activate")
    parser.add_argument("name", help="staged tool name (tools_staged/<name>.py)")
    parser.add_argument("--config", default="config.yaml", help="source of workspace/audit paths")
    parser.add_argument("--staged-dir", type=Path, default=DEFAULT_STAGED_DIR)
    parser.add_argument("--enabled-dir", type=Path, default=DEFAULT_ENABLED_DIR)
    parser.add_argument("--workspace-root", type=Path, help="overrides config workspace.root")
    parser.add_argument("--audit", type=Path, help="overrides config audit.path")
    args = parser.parse_args(list(argv) if argv is not None else None)

    if not sys.stdin.isatty():
        print("activate requires an interactive terminal; refusing to read confirmation "
              "from a pipe.", file=sys.stderr)  # fmt: skip
        return 4

    workspace_root, audit_path = args.workspace_root, args.audit
    if workspace_root is None or audit_path is None:
        try:
            cfg = load_config(args.config)
        except ConfigError as exc:
            print(f"aborted: {exc}", file=sys.stderr)
            return 4
        workspace_root = workspace_root or cfg.workspace.root
        audit_path = audit_path or cfg.audit.path

    try:
        activate(
            args.name,
            staged_dir=args.staged_dir,
            enabled_dir=args.enabled_dir,
            workspace_root=workspace_root,
            confirm=input,
            audit=AuditLog(audit_path),
        )
    except ActivationRejected:
        return 2
    except ActivationDeclined:
        return 3
    except ActivationAborted as exc:
        print(f"aborted: {exc}", file=sys.stderr)
        return 4
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
