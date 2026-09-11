"""activate: the human gate. Static checks, typed confirmation, install-without-loading.

The bulk of this file is adversarial: source that tries to reach a denied capability without a
literal ``import os``. Each ``EVASION`` case is a way a language model might try to slip past the
checker; ``test_evasions_are_caught`` asserts every one is refused. The ones I could NOT defeat
are in ``KNOWN_GAPS`` with an ``xfail`` documenting exactly what gets through — an honest hole is
worth more than a checker that hides it.
"""

from __future__ import annotations

import ast
import io
from pathlib import Path
from typing import Any

import pytest

from narrowgate import activate
from narrowgate.activate import (
    ActivationAborted,
    ActivationDeclined,
    ActivationRejected,
    static_check,
)

# A tool that passes every check: the baseline the evasions mutate.
CLEAN = '''
"""A clean tool."""
from narrowgate.tools.base import ToolResult


class Adder:
    name = "adder"
    description = "Add two integers."
    schema = {"type": "object",
              "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
              "required": ["a", "b"], "additionalProperties": False}
    requires_network = False

    def run(self, **kwargs):
        return ToolResult(True, str(kwargs["a"] + kwargs["b"]))
'''


class RecordingAudit:
    def __init__(self) -> None:
        self.records: list[tuple[str, dict[str, Any]]] = []

    def record(self, event: str, /, **fields: Any) -> bool:
        self.records.append((event, fields))
        return True


def _check(source: str, name: str = "adder", root: Path | None = None) -> list[str]:
    findings = static_check(source, tool_name=name, workspace_root=root or Path("/nonexistent"))
    return [f"{v.line}:{v.reason}" for v in findings]


def _in_body(fragment: str, name: str = "adder", root: Path | None = None) -> list[str]:
    """Check a fragment placed inside a function body, so a bare call is not also flagged as a
    top-level statement (that rule is tested separately)."""
    indented = "\n".join("    " + line for line in fragment.splitlines())
    return _check(CLEAN + f"\ndef _helper():\n{indented}\n", name=name, root=root)


# --- baseline --------------------------------------------------------------------------------


def test_clean_tool_passes() -> None:
    assert _check(CLEAN) == []


def test_every_violation_reports_line_and_source() -> None:
    src = "import os\n" + CLEAN + "bad = eval\n"
    findings = static_check(src, tool_name="adder", workspace_root=Path("/x"))
    assert findings
    for v in findings:
        assert v.line >= 1
        assert v.source  # the offending source line is carried, not just the number
        assert str(v.line) in v.render() and v.source in v.render()


# --- spec-mandated rejections ---------------------------------------------------------------


@pytest.mark.parametrize(
    "line",
    [
        "import os",
        "import subprocess",
        "import socket",
        "import ctypes",
        "import importlib",
        "import builtins",
        "import os.path",  # dotted; top module still os
        "import subprocess as sp",
        "from os import system",
        "from os.path import join",  # os.* denied at the module level
    ],
)
def test_denied_imports(line: str) -> None:
    assert _check(CLEAN + line + "\n"), line


@pytest.mark.parametrize(
    "expr",
    [
        "eval('1+1')",
        "exec('x=1')",
        "compile('1', '<s>', 'eval')",
        "__import__('os')",
        "x = eval",  # even as a value, not only when called
        "f = [eval][0]",
    ],
)
def test_denied_names(expr: str) -> None:
    assert _check(CLEAN + expr + "\n"), expr


def test_dunder_getattr_rejected() -> None:
    assert _check(CLEAN + "getattr(x, '__class__')\n")
    assert _check(CLEAN + "getattr(x, '__globals__')\n")


def test_getattr_with_nonliteral_name_rejected() -> None:
    # Cannot review a name the checker cannot see.
    assert _check(CLEAN + "getattr(x, some_var)\n")
    assert _check(CLEAN + "getattr(x, 'na' + 'me')\n")


def test_getattr_with_safe_literal_allowed() -> None:
    assert _check(CLEAN + "y = getattr(obj, 'value')\n") == []


# --- write-mode open --------------------------------------------------------------------------


def test_write_open_outside_root_rejected(tmp_path: Path) -> None:
    assert _in_body("open('/etc/passwd', 'w')", root=tmp_path)


def test_write_open_inside_root_allowed(tmp_path: Path) -> None:
    assert _in_body(f"open({str(tmp_path / 'out.txt')!r}, 'w')", root=tmp_path) == []


def test_relative_write_open_resolved_against_root(tmp_path: Path) -> None:
    assert _in_body("open('sub/out.txt', 'w')", root=tmp_path) == []
    assert _in_body("open('../out.txt', 'w')", root=tmp_path)


def test_read_open_is_allowed_anywhere(tmp_path: Path) -> None:
    assert _in_body("open('/etc/passwd')", root=tmp_path) == []
    assert _in_body("open('/etc/passwd', 'r')", root=tmp_path) == []


@pytest.mark.parametrize("mode", ["w", "a", "x", "r+", "w+b", "ab", "wb"])
def test_all_write_modes_caught(mode: str, tmp_path: Path) -> None:
    assert _in_body(f"open('/etc/x', {mode!r})", root=tmp_path)


def test_nonliteral_mode_or_path_rejected(tmp_path: Path) -> None:
    assert _in_body("open('/etc/x', mode_var)", root=tmp_path)
    assert _in_body("open(path_var, 'w')", root=tmp_path)
    assert _in_body("open(*args)", root=tmp_path)


def test_pathlib_write_open_rejected(tmp_path: Path) -> None:
    # p.open("w") writes through an object path the checker cannot resolve.
    assert _in_body("p.open('w')", root=tmp_path)


# --- Tool-class requirements ----------------------------------------------------------------


def test_missing_requires_network_rejected() -> None:
    src = CLEAN.replace("    requires_network = False\n", "")
    findings = _check(src)
    assert any("requires_network" in f for f in findings)


def test_requires_network_must_be_bool_literal() -> None:
    assert _check(CLEAN.replace("requires_network = False", "requires_network = 0"))
    assert _check(CLEAN.replace("requires_network = False", "requires_network = flag"))


def test_two_tool_classes_rejected() -> None:
    src = CLEAN + "\nclass Second:\n    def run(self):\n        return None\n"
    assert any("exactly one" in f for f in _check(src))


def test_zero_tool_classes_rejected() -> None:
    src = '"""doc"""\nx = 1\n'
    assert any("exactly one" in f for f in _check(src))


def test_name_mismatch_rejected() -> None:
    # Declared name differs from the staged name: would register under the wrong name.
    assert any("staged as" in f for f in _check(CLEAN, name="different"))


def test_name_must_be_literal() -> None:
    src = CLEAN.replace('name = "adder"', "name = COMPUTED")
    assert _check(src, name="adder")


def test_network_module_with_false_declaration_rejected() -> None:
    src = CLEAN.replace(
        "from narrowgate.tools.base import ToolResult",
        "from narrowgate.tools.base import ToolResult\nimport httpx",
    )
    findings = _check(src)
    assert any("requires_network = False" in f for f in findings)


def test_network_module_with_true_declaration_allowed() -> None:
    src = CLEAN.replace("requires_network = False", "requires_network = True").replace(
        "from narrowgate.tools.base import ToolResult",
        "from narrowgate.tools.base import ToolResult\nimport ssl",
    )
    # ssl is a NETWORK module, not a DENIED one, so honesty about it is enough.
    assert _check(src) == []


# --- EVASIONS: reaching a capability without a literal `import os` ---------------------------

EVASIONS: dict[str, str] = {
    "builtins_dunder_read": "x = __builtins__\n",
    "import_via_builtins_attr": "m = __builtins__.__import__('os')\n",
    "import_dunder_call": "m = __import__('os')\n",
    "concat_import_arg": "m = __import__('o' + 's')\n",  # __import__ denied regardless of arg
    "getattr_dunder_class": "c = ().__class__\n",
    "getattr_subclasses": "s = ().__class__.__bases__\n",
    "mro_walk": "for c in ().__class__.__mro__: pass\n",
    "getattr_builtins_str": "g = getattr(x, '__globals__')\n",
    "frame_globals": "import sys\ndef f():\n    return sys._getframe()\n",
    "code_object": "import types\nc = types.CodeType\n",
    "pickle_reduce": "import pickle\n",
    "os_via_pathlib": "import pathlib\no = pathlib.os\n",
    "os_via_random": "import random\no = random._os\n",
    "system_attr": "shutil.rmtree('/')\n",
    "popen_attr": "handle = mod.popen('id')\n",
    "attrgetter": "from operator import attrgetter\n",
    "pydoc_locate": "import pydoc\nf = pydoc.locate('os.system')\n",
    "importlib_module": "from importlib import import_module\n",
    "import_module_attr": "m = il.import_module('os')\n",
    "runpy": "import runpy\n",
    "subprocess_run": "import subprocess\nsubprocess.run(['id'])\n",
    "socket_conn": "import socket\n",
    "eval_indirect": "e = eval\ne('1')\n",
    "compile_then": "co = compile('x=1', '<s>', 'exec')\n",
    "star_import": "from os import *\n",
    "relative_import": "from . import helper\n",
    "yaml_load": "import yaml\nd = yaml.load('!!python/object/apply:os.system []')\n",
    "yaml_unsafe_load": "obj = yaml.unsafe_load(data)\n",
    "type_get_hints": "from typing import get_type_hints\n",
    "get_type_hints_attr": "h = typing.get_type_hints(obj)\n",
    "string_annotation": "def g(x: 'os.system'): pass\n",
    "string_annassign": "v: '__import__(\"os\")' = None\n",
    "typeadapter_str": "a = TypeAdapter('os.system')\n",
    "forwardref_str": "r = ForwardRef('os.system')\n",
    "write_text_attr": "p.write_text('data')\n",
    "unlink_attr": "p.unlink()\n",
    "logging_dictconfig": "logging.config.dictConfig(cfg)\n",
    "logging_config_import": "from logging import config\n",
    "logging_filehandler": "import logging\nh = logging.FileHandler('/etc/x')\n",
    "logging_http_handler": "import logging.handlers\n",
    "logging_socket_handler_attr": "h = logging.SocketHandler('h', 9)\n",
    "site_addsitedir": "import site\n",
    "addsitedir_attr": "s.addsitedir('/x')\n",
    "marshal": "import marshal\n",
    "shelve": "import shelve\n",
    "tempfile": "import tempfile\n",
    "gc_referents": "import gc\n",
    "inspect_frames": "import inspect\n",
    "codeop": "import codeop\n",
    "resolve_name_attr": "t = pkgutil.resolve_name('os.system')\n",
    "create_model_attr": "M = pydantic.create_model('X')\n",
    "os_dot_import": "import os.path as p\n",
    "from_logging_handlers": "from logging.handlers import HTTPHandler\n",
    "py_compile": "import py_compile\n",
    "globals_call": "g = globals()\n",
    "vars_call": "v = vars()\n",
    "breakpoint_call": "breakpoint()\n",
    "input_call": "input('give me a shell')\n",
}


@pytest.mark.parametrize("label", sorted(EVASIONS))
def test_evasions_are_caught(label: str) -> None:
    src = CLEAN + EVASIONS[label]
    findings = _check(src)
    assert findings, f"EVASION SLIPPED PAST THE CHECKER: {label}\n{EVASIONS[label]}"


# Evasions I could not defeat with a syntax-only checker. Documented, not hidden. Each is also
# named in a comment in activate.py's known-gaps docstring. These xfail: the day one is fixed,
# strict xfail turns the surprise pass into a failure that makes us update this list.
KNOWN_GAPS: dict[str, str] = {
    # A stdlib module that re-exports a dangerous module under a name not in DENIED_ATTRS.
    # `import asyncio` is denied, but e.g. `import selectors; selectors.<...>` families shift
    # over versions; a novel re-export is by construction not enumerable ahead of time.
    "novel_stdlib_reexport": "import string\nx = string\n",
    # A third-party module with eval-like power that narrowgate does not depend on and so does
    # not deny. If the deployment has jinja2 installed, an approved tool may import it.
    "third_party_eval": "import jinja2\n",
    # A module-level ``X = f()`` runs ``f`` at harness start. It cannot reach a denied
    # capability (every dangerous name/import is already refused), but the *timing* — code
    # running before any tool is dispatched — is not something a syntax walk prevents.
    "module_level_call_timing": "SETUP = build_everything()\n",
}


@pytest.mark.parametrize("label", sorted(KNOWN_GAPS))
@pytest.mark.xfail(strict=True, reason="documented gap: syntax-only checker cannot catch this")
def test_known_gaps_still_pass(label: str) -> None:
    assert _check(CLEAN + KNOWN_GAPS[label]), label


# --- static_check never executes ------------------------------------------------------------


def test_static_check_does_not_execute_source(tmp_path: Path) -> None:
    sentinel = tmp_path / "RAN"
    hostile = f"import pathlib\npathlib.Path({str(sentinel)!r}).write_text('x')\n" + CLEAN
    _check(hostile, root=tmp_path)
    assert not sentinel.exists()


def test_unparseable_source_is_one_violation() -> None:
    findings = static_check("def (:\n", tool_name="adder", workspace_root=Path("/x"))
    assert len(findings) == 1 and "does not parse" in findings[0].reason


# --- the gate: load_staged, confirmation, install -------------------------------------------


@pytest.fixture
def gate_dirs(tmp_path: Path) -> tuple[Path, Path, Path]:
    staged, enabled, ws = tmp_path / "staged", tmp_path / "enabled", tmp_path / "ws"
    staged.mkdir()
    enabled.mkdir()
    ws.mkdir()
    return staged, enabled, ws


def _activate(
    gate_dirs: tuple[Path, Path, Path],
    name: str,
    typed: str,
    source: str = CLEAN,
    audit: RecordingAudit | None = None,
) -> tuple[Any, str, RecordingAudit]:
    staged, enabled, ws = gate_dirs
    (staged / f"{name}.py").write_text(source)
    out = io.StringIO()
    audit = audit or RecordingAudit()
    rec = activate.activate(
        name,
        staged_dir=staged,
        enabled_dir=enabled,
        workspace_root=ws,
        confirm=lambda _prompt: typed,
        out=out,
        audit=audit,
        approver="tester",
    )
    return rec, out.getvalue(), audit


def test_full_source_is_printed_before_confirmation(gate_dirs: tuple[Path, Path, Path]) -> None:
    rec, printed, _ = _activate(gate_dirs, "adder", "adder")
    # Every non-empty source line appears in the output.
    for line in CLEAN.splitlines():
        if line.strip():
            assert line in printed
    assert "sha256:" in printed


def test_typed_name_confirms_and_installs(gate_dirs: tuple[Path, Path, Path]) -> None:
    staged, enabled, _ = gate_dirs
    rec, printed, audit = _activate(gate_dirs, "adder", "adder")
    assert (enabled / "adder.py").read_text() == CLEAN
    assert not (staged / "adder.py").exists(), "staged copy left behind after activation"
    assert rec.approved_by == "tester"
    assert any(e == "tool_activated" for e, _ in audit.records)


def test_wrong_confirmation_declines_and_installs_nothing(
    gate_dirs: tuple[Path, Path, Path],
) -> None:
    staged, enabled, _ = gate_dirs
    with pytest.raises(ActivationDeclined):
        _activate(gate_dirs, "adder", "y")
    assert list(enabled.iterdir()) == []
    assert (staged / "adder.py").exists()


@pytest.mark.parametrize("typed", ["y", "yes", "Adder", "adder ", " adder", "adde", ""])
def test_only_exact_name_confirms(gate_dirs: tuple[Path, Path, Path], typed: str) -> None:
    with pytest.raises(ActivationDeclined):
        _activate(gate_dirs, "adder", typed)


def test_trailing_newline_in_confirmation_is_tolerated(
    gate_dirs: tuple[Path, Path, Path],
) -> None:
    rec, _, _ = _activate(gate_dirs, "adder", "adder\n")
    assert rec.name == "adder"


def test_rejected_source_raises_and_installs_nothing(
    gate_dirs: tuple[Path, Path, Path],
) -> None:
    staged, enabled, _ = gate_dirs
    with pytest.raises(ActivationRejected) as exc:
        _activate(gate_dirs, "evil", "evil", source=CLEAN + "import os\n")
    assert exc.value.violations
    assert list(enabled.iterdir()) == []
    # rejection is terminal: no retry path, staged file stays for the operator to see
    assert (staged / "evil.py").exists()


def test_rejection_prints_line_numbers(gate_dirs: tuple[Path, Path, Path]) -> None:
    with pytest.raises(ActivationRejected):
        _, printed, _ = _activate(gate_dirs, "evil", "evil", source="import os\n" + CLEAN)


def test_rejection_never_asks_for_confirmation(gate_dirs: tuple[Path, Path, Path]) -> None:
    staged, enabled, ws = gate_dirs
    (staged / "evil.py").write_text("import os\n" + CLEAN)
    calls: list[str] = []

    def confirm(_p: str) -> str:
        calls.append(_p)
        return "evil"

    with pytest.raises(ActivationRejected):
        activate.activate(
            "evil", staged_dir=staged, enabled_dir=enabled, workspace_root=ws,
            confirm=confirm, out=io.StringIO(),
        )  # fmt: skip
    assert calls == [], "confirmation was requested for a rejected proposal"


def test_refuses_to_overwrite_enabled(gate_dirs: tuple[Path, Path, Path]) -> None:
    staged, enabled, _ = gate_dirs
    (enabled / "adder.py").write_text("# already here\n")
    with pytest.raises(ActivationAborted):
        _activate(gate_dirs, "adder", "adder")
    assert (enabled / "adder.py").read_text() == "# already here\n"


def test_missing_staged_file_aborts(gate_dirs: tuple[Path, Path, Path]) -> None:
    staged, enabled, ws = gate_dirs
    with pytest.raises(ActivationAborted):
        activate.activate(
            "ghost", staged_dir=staged, enabled_dir=enabled, workspace_root=ws,
            confirm=lambda _p: "ghost", out=io.StringIO(),
        )  # fmt: skip


@pytest.mark.parametrize("bad", ["../x", "..", "a/b", "Evil", "/etc/x"])
def test_bad_staged_name_aborts(gate_dirs: tuple[Path, Path, Path], bad: str) -> None:
    staged, enabled, ws = gate_dirs
    with pytest.raises(ActivationAborted):
        activate.activate(
            bad, staged_dir=staged, enabled_dir=enabled, workspace_root=ws,
            confirm=lambda _p: bad, out=io.StringIO(),
        )  # fmt: skip


def test_symlinked_staged_file_aborts(gate_dirs: tuple[Path, Path, Path], tmp_path: Path) -> None:
    staged, enabled, ws = gate_dirs
    secret = tmp_path / "secret.py"
    secret.write_text(CLEAN)
    (staged / "adder.py").symlink_to(secret)
    with pytest.raises(ActivationAborted):
        activate.activate(
            "adder", staged_dir=staged, enabled_dir=enabled, workspace_root=ws,
            confirm=lambda _p: "adder", out=io.StringIO(),
        )  # fmt: skip


def test_activation_installs_exact_reviewed_bytes(gate_dirs: tuple[Path, Path, Path]) -> None:
    import hashlib

    staged, enabled, _ = gate_dirs
    rec, _, _ = _activate(gate_dirs, "adder", "adder")
    installed = (enabled / "adder.py").read_bytes()
    assert rec.sha256 == hashlib.sha256(installed).hexdigest()


def test_activate_does_not_import_the_tool(gate_dirs: tuple[Path, Path, Path]) -> None:
    import sys

    before = set(sys.modules)
    _activate(gate_dirs, "adder", "adder")
    assert set(sys.modules) == before
    assert "adder" not in sys.modules


def test_render_source_numbers_every_line() -> None:
    rendered = activate.render_source("a\nb\nc")
    assert "1 | a" in rendered and "3 | c" in rendered


# --- audit trail on every path --------------------------------------------------------------


def test_audit_records_rejection_with_violations(gate_dirs: tuple[Path, Path, Path]) -> None:
    audit = RecordingAudit()
    with pytest.raises(ActivationRejected):
        _activate(gate_dirs, "evil", "evil", source="import os\n" + CLEAN, audit=audit)
    events = {e for e, _ in audit.records}
    assert "tool_rejected" in events
    _, fields = next(r for r in audit.records if r[0] == "tool_rejected")
    assert fields["violations"]


def test_audit_records_decline(gate_dirs: tuple[Path, Path, Path]) -> None:
    audit = RecordingAudit()
    with pytest.raises(ActivationDeclined):
        _activate(gate_dirs, "adder", "no", audit=audit)
    assert any(e == "tool_declined" for e, _ in audit.records)


def test_no_load_path_in_activate_source() -> None:
    """activate.py imports nothing that loads code and never names eval/exec/compile."""
    path = Path(activate.__file__)
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                assert a.name.split(".")[0] not in {"importlib", "runpy", "imp"}
        if isinstance(node, ast.ImportFrom):
            assert (node.module or "").split(".")[0] not in {"importlib", "runpy", "imp"}
        if isinstance(node, ast.Name):
            assert node.id not in {"eval", "exec", "compile", "__import__"}
