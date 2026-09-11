"""propose_tool: writes to staging, refuses everything else, loads nothing.

Every test that hands the tool hostile input also asserts that nothing landed outside the
staging directory and that ``sys.modules`` did not change. The second assertion is the point of
the project.
"""

from __future__ import annotations

import ast
import hashlib
import sys
from pathlib import Path
from typing import Any

import pytest

from narrowgate.tools.base import Tool, ToolResult
from narrowgate.tools.propose import (
    MAX_DESCRIPTION_CHARS,
    MAX_RATIONALE_CHARS,
    MAX_SOURCE_BYTES,
    ProposeTool,
    check_tool_name,
    propose_tool,
)
from narrowgate.tools.registry import Registry

GOOD_SOURCE = '''
"""A harmless tool."""
from narrowgate.tools.base import ToolResult


class Shout:
    name = "shout"
    description = "Uppercase the text."
    schema = {"type": "object", "properties": {"text": {"type": "string"}},
              "required": ["text"], "additionalProperties": False}
    requires_network = False

    def run(self, **kwargs):
        return ToolResult(True, kwargs["text"].upper())
'''

# Would prove execution if it ever ran: creates a sentinel file at import time.
SIDE_EFFECT_SOURCE = """
import pathlib
pathlib.Path({sentinel!r}).write_text("executed")
class Evil:
    name = "evil"
    description = "d"
    schema = {{"type": "object"}}
    requires_network = False
    def run(self, **kw):
        return None
"""


class RecordingAudit:
    def __init__(self) -> None:
        self.records: list[tuple[str, dict[str, Any]]] = []

    def record(self, event: str, /, **fields: Any) -> bool:
        self.records.append((event, fields))
        return True


@pytest.fixture
def dirs(tmp_path: Path) -> tuple[Path, Path]:
    staged, enabled = tmp_path / "tools_staged", tmp_path / "enabled"
    staged.mkdir()
    enabled.mkdir()
    return staged, enabled


def _tree(root: Path) -> set[str]:
    return {str(p.relative_to(root)) for p in root.rglob("*")}


def _propose(dirs: tuple[Path, Path], name: str = "shout", **over: Any) -> ToolResult:
    staged, enabled = dirs
    kwargs: dict[str, Any] = dict(
        description="Uppercase the text.", python_source=GOOD_SOURCE, rationale="testing"
    )
    kwargs.update(over)
    return propose_tool(name, staged_dir=staged, enabled_dir=enabled, **kwargs)


# --- happy path -----------------------------------------------------------------------------


def test_writes_exact_bytes_to_staging_only(dirs: tuple[Path, Path], tmp_path: Path) -> None:
    staged, enabled = dirs
    before = _tree(tmp_path)
    audit = RecordingAudit()
    res = propose_tool(
        "shout", "d", GOOD_SOURCE, "r", staged_dir=staged, enabled_dir=enabled, audit=audit
    )
    assert res.ok, res.error
    assert (staged / "shout.py").read_bytes() == GOOD_SOURCE.encode()
    assert _tree(tmp_path) - before == {"tools_staged/shout.py"}
    assert list(enabled.iterdir()) == []


def test_result_says_not_active_unambiguously(dirs: tuple[Path, Path]) -> None:
    res = _propose(dirs)
    text = res.content
    assert "NOT ACTIVE" in text
    assert "cannot be called" in text
    assert "narrowgate activate shout" in text
    assert "human operator" in text
    # No word that could be read as "you now have this capability".
    lowered = text.lower()
    for word in ("activated", "available", "ready to use", "registered.", "success"):
        assert word not in lowered, word
    # And nothing about the activation checks: no signal to iterate against.
    for word in ("violation", "reject", "passed", "static"):
        assert word not in lowered, word


def test_audit_receives_full_source_and_hash(dirs: tuple[Path, Path]) -> None:
    staged, enabled = dirs
    audit = RecordingAudit()
    propose_tool(
        "shout", "desc", GOOD_SOURCE, "why", staged_dir=staged, enabled_dir=enabled, audit=audit
    )
    assert len(audit.records) == 1
    event, fields = audit.records[0]
    assert event == "tool_proposed"
    assert fields["source"] == GOOD_SOURCE
    assert fields["sha256"] == hashlib.sha256(GOOD_SOURCE.encode()).hexdigest()
    assert fields["active"] is False
    assert fields["rationale"] == "why" and fields["description"] == "desc"
    assert "ts" not in fields and "event" not in fields  # AuditLog.record owns those


def test_works_without_audit_sink(dirs: tuple[Path, Path]) -> None:
    assert _propose(dirs).ok


# --- the invariant: nothing is loaded ---------------------------------------------------------


def test_proposal_is_never_imported(dirs: tuple[Path, Path], tmp_path: Path) -> None:
    staged, _ = dirs
    sentinel = tmp_path / "EXECUTED"
    before = set(sys.modules)
    res = _propose(dirs, name="evil", python_source=SIDE_EFFECT_SOURCE.format(sentinel=sentinel))
    assert res.ok, res.error
    assert not sentinel.exists(), "proposed source ran"
    assert set(sys.modules) == before
    assert "evil" not in sys.modules
    assert not any(
        getattr(m, "__file__", None) and str(staged) in str(m.__file__)
        for m in list(sys.modules.values())
    ), "a module was loaded from the staging directory"
    assert not (staged / "__pycache__").exists()


def test_hostile_source_is_accepted_as_a_proposal_and_not_run(
    dirs: tuple[Path, Path], tmp_path: Path
) -> None:
    """propose does not run the activation checks: that is activate's job and giving the
    agent the verdict here would be a feedback loop to optimise against."""
    hostile = "import os\nos.system('touch %s')\nclass T:\n    def run(self): pass\n" % (
        tmp_path / "PWNED"
    )
    res = _propose(dirs, name="hostile", python_source=hostile)
    assert res.ok, res.error
    assert not (tmp_path / "PWNED").exists()


@pytest.mark.parametrize(
    "module", ["src/narrowgate/tools/propose.py", "src/narrowgate/activate.py"]
)
def test_my_modules_contain_no_load_path(module: str) -> None:
    """Static self-audit: no importlib/runpy import and no eval/exec/compile/__import__ name."""
    path = Path(__file__).resolve().parents[1] / module
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                assert a.name.split(".")[0] not in {"importlib", "runpy", "imp"}, a.name
        if isinstance(node, ast.ImportFrom):
            assert (node.module or "").split(".")[0] not in {"importlib", "runpy", "imp"}
        if isinstance(node, ast.Name):
            assert node.id not in {"eval", "exec", "compile", "__import__"}, node.lineno
        if isinstance(node, ast.Attribute):
            assert node.attr not in {"import_module", "spec_from_file_location", "exec_module"}


# --- name validation: traversal and lookalikes ----------------------------------------------


@pytest.mark.parametrize(
    "bad",
    [
        "../evil", "..", ".", "/etc/cron.d/x", "sub/evil", "evil/", "a/../../b", "..\\evil",
        "evil\x00", "evil\x00.py", "evil.py", "evil\n", "evil ", " evil", "", "ev", "a" * 65,
        "Evil", "1evil", "_evil", "e-vil", "e.vil", "e;vil", "evil$",
        "\uff45\uff56\uff49\uff4c",  # fullwidth "evil" (NFKC lookalike)
        "\u0435vil",  # cyrillic е
        "evi\u200bl",  # zero-width space
        "\u00e9vil",  # é
        "evil\u2044..\u2044x",  # fraction slash
        "\u0661evil",  # arabic-indic digit
    ],
)  # fmt: skip
def test_bad_names_write_nothing(dirs: tuple[Path, Path], tmp_path: Path, bad: str) -> None:
    before = _tree(tmp_path)
    res = _propose(dirs, name=bad)
    assert not res.ok
    assert res.error
    assert _tree(tmp_path) == before


@pytest.mark.parametrize("bad", [None, 123, b"evil", ["evil"], {"name": "evil"}])
def test_non_string_names_refused(dirs: tuple[Path, Path], tmp_path: Path, bad: Any) -> None:
    before = _tree(tmp_path)
    assert not _propose(dirs, name=bad).ok
    assert _tree(tmp_path) == before


def test_check_tool_name_matches_registry_rule() -> None:
    assert check_tool_name("abc") is None
    assert check_tool_name("a1_") is None
    assert check_tool_name("ab") is not None
    assert check_tool_name("Abc") is not None
    assert check_tool_name(None) is not None


# --- overwrite refusals ---------------------------------------------------------------------


def test_refuses_to_shadow_an_activated_tool(dirs: tuple[Path, Path]) -> None:
    staged, enabled = dirs
    (enabled / "shout.py").write_text("# active\n")
    res = _propose(dirs)
    assert not res.ok and "already an activated tool" in (res.error or "")
    assert not (staged / "shout.py").exists()
    assert (enabled / "shout.py").read_text() == "# active\n"


def test_refuses_to_overwrite_a_staged_proposal(dirs: tuple[Path, Path]) -> None:
    """Closes the window where an agent rewrites a file between review and confirmation."""
    staged, _ = dirs
    assert _propose(dirs).ok
    second = _propose(dirs, python_source=GOOD_SOURCE.replace("upper", "lower"))
    assert not second.ok and "already exists" in (second.error or "")
    assert (staged / "shout.py").read_bytes() == GOOD_SOURCE.encode()


def test_refuses_when_staged_path_is_a_symlink(dirs: tuple[Path, Path], tmp_path: Path) -> None:
    staged, _ = dirs
    outside = tmp_path / "outside.py"
    (staged / "shout.py").symlink_to(outside)
    res = _propose(dirs)
    assert not res.ok
    assert not outside.exists(), "wrote through a symlink"


def test_refuses_dangling_then_created_symlink_race(dirs: tuple[Path, Path]) -> None:
    """Exclusive create: even if the existence check were bypassed, O_EXCL refuses a symlink."""
    staged, _ = dirs
    (staged / "shout.py").symlink_to(staged / "elsewhere.py")
    res = propose_tool("shout", "d", GOOD_SOURCE, "r", staged_dir=staged, enabled_dir=dirs[1])
    assert not res.ok
    assert not (staged / "elsewhere.py").exists()


# --- source validation: parse only ----------------------------------------------------------


@pytest.mark.parametrize(
    "src",
    [
        "def (:\n",
        "class T:\n  name = \x00\n",
        "x = 1\n  y = 2\n",
        "(" * 5000 + ")" * 5000,
    ],
)
def test_unparseable_source_refused_without_crash(
    dirs: tuple[Path, Path], tmp_path: Path, src: str
) -> None:
    before = _tree(tmp_path)
    res = _propose(dirs, python_source=src)
    assert not res.ok and "does not parse" in (res.error or "")
    assert _tree(tmp_path) == before


def test_size_and_type_limits(dirs: tuple[Path, Path], tmp_path: Path) -> None:
    before = _tree(tmp_path)
    assert not _propose(dirs, python_source="x = 1\n" * (MAX_SOURCE_BYTES // 6 + 1)).ok
    assert not _propose(dirs, python_source="").ok
    assert not _propose(dirs, python_source="   \n").ok
    assert not _propose(dirs, python_source=b"x = 1").ok
    assert not _propose(dirs, python_source=None).ok
    assert not _propose(dirs, description="").ok
    assert not _propose(dirs, description="d" * (MAX_DESCRIPTION_CHARS + 1)).ok
    assert not _propose(dirs, description=42).ok
    assert not _propose(dirs, rationale="").ok
    assert not _propose(dirs, rationale="r" * (MAX_RATIONALE_CHARS + 1)).ok
    assert not _propose(dirs, rationale=None).ok
    assert _tree(tmp_path) == before


def test_source_with_null_byte_refused(dirs: tuple[Path, Path]) -> None:
    res = _propose(dirs, python_source="x = 1\n\x00")
    assert not res.ok


# --- ProposeTool through the real registry --------------------------------------------------


class _Audit:
    def before_call(self, **kw: Any) -> None:
        pass

    def after_call(self, **kw: Any) -> None:
        pass


def test_propose_tool_registers_and_dispatches(dirs: tuple[Path, Path]) -> None:
    staged, enabled = dirs
    tool = ProposeTool(staged_dir=staged, enabled_dir=enabled, audit=RecordingAudit())
    assert isinstance(tool, Tool)
    assert tool.requires_network is False
    reg = Registry()
    reg.register(tool)
    res = reg.dispatch(
        "propose_tool",
        {"name": "shout", "description": "d", "python_source": GOOD_SOURCE, "rationale": "r"},
        audit=_Audit(),
    )
    assert res.ok, res.error
    assert (staged / "shout.py").exists()
    assert "NOT ACTIVE" in res.content


def test_registry_rejects_undeclared_and_missing_arguments(dirs: tuple[Path, Path]) -> None:
    staged, enabled = dirs
    reg = Registry()
    reg.register(ProposeTool(staged_dir=staged, enabled_dir=enabled))
    extra = reg.dispatch(
        "propose_tool",
        {
            "name": "shout",
            "description": "d",
            "python_source": "x=1",
            "rationale": "r",
            "path": "/etc/x",
        },  # fmt: skip
        audit=_Audit(),
    )
    assert not extra.ok
    missing = reg.dispatch("propose_tool", {"name": "shout"}, audit=_Audit())
    assert not missing.ok
    traversal = reg.dispatch(
        "propose_tool",
        {"name": "../x", "description": "d", "python_source": "x=1", "rationale": "r"},
        audit=_Audit(),
    )
    assert not traversal.ok
    assert list(staged.iterdir()) == []


def test_run_refuses_unexpected_kwargs_even_without_registry(dirs: tuple[Path, Path]) -> None:
    staged, enabled = dirs
    tool = ProposeTool(staged_dir=staged, enabled_dir=enabled)
    assert not tool.run(name="shout", description="d", python_source="x=1", rationale="r",
                        load=True).ok  # fmt: skip
    assert not tool.run(name="shout").ok
    assert list(staged.iterdir()) == []
