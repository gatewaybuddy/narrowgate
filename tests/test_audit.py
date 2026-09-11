"""audit.py — append-only, never raises, loud on failure, no off switch."""

from __future__ import annotations

import inspect
import json
import stat
from pathlib import Path

import pytest

from narrowgate import audit as audit_mod
from narrowgate.audit import AuditLog, new_session_id


def read_records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


# -- happy path as control ----------------------------------------------------------------------


def test_before_and_after_records(tmp_path: Path) -> None:
    log = AuditLog(tmp_path / "audit" / "ng.jsonl")
    sid = new_session_id()
    cid = log.before_call(sid, "read_file", {"path": "notes.md"})
    log.after_call(sid, "read_file", cid, ok=True, content_len=12)

    recs = read_records(log.path)
    assert [r["event"] for r in recs] == ["tool_call.before", "tool_call.after"]
    before, after = recs
    assert before["session_id"] == after["session_id"] == sid
    assert before["call_id"] == after["call_id"] == cid
    assert before["tool"] == after["tool"] == "read_file"
    assert before["arguments"] == {"path": "notes.md"}
    assert after["outcome"] == {"ok": True, "error": None, "content_len": 12}
    assert all("ts" in r for r in recs)
    assert log.failures == 0


def test_failed_outcome_is_recorded(tmp_path: Path) -> None:
    log = AuditLog(tmp_path / "a.jsonl")
    log.after_call("s", "write_file", "c1", ok=False, error="path escapes root")
    (rec,) = read_records(log.path)
    assert rec["outcome"] == {"ok": False, "error": "path escapes root", "content_len": None}


# -- append-only --------------------------------------------------------------------------------


def test_never_truncates_existing_content(tmp_path: Path) -> None:
    path = tmp_path / "a.jsonl"
    path.write_text('{"event":"pre-existing"}\n', encoding="utf-8")
    AuditLog(path).write({"event": "new"})
    AuditLog(path).write({"event": "newer"})  # a second instance must also only append
    assert [r["event"] for r in read_records(path)] == ["pre-existing", "new", "newer"]


def test_file_is_created_private(tmp_path: Path) -> None:
    log = AuditLog(tmp_path / "a.jsonl")
    log.write({"event": "x"})
    mode = stat.S_IMODE(log.path.stat().st_mode)
    assert mode & 0o077 == 0, f"audit file is group/world accessible: {oct(mode)}"


def test_path_is_read_only() -> None:
    log = AuditLog("/nonexistent/a.jsonl")
    with pytest.raises(AttributeError):
        log.path = Path("/elsewhere")  # type: ignore[misc]


# -- never raises, always loud ------------------------------------------------------------------


def test_unwritable_path_does_not_raise_and_is_loud(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    target = tmp_path / "audit.jsonl"
    target.mkdir()  # a directory where the file should be: every open() will fail
    log = AuditLog(target)

    cid = log.before_call("s", "read_file", {"path": "x"})  # must still return a call id
    log.after_call("s", "read_file", cid, ok=True)

    assert isinstance(cid, str) and cid
    assert log.failures == 2
    err = capsys.readouterr().err
    assert err.count("narrowgate audit: FAILED") == 2
    assert str(target) in err
    assert "tool_call.before" in err and "tool_call.after" in err


def test_parent_dir_creation_failure_is_loud_not_fatal(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    blocker = tmp_path / "file"
    blocker.write_text("i am a file, not a directory")
    log = AuditLog(blocker / "sub" / "a.jsonl")
    assert log.write({"event": "x"}) is False
    assert log.failures == 1
    assert "FAILED" in capsys.readouterr().err


def test_unserialisable_arguments_are_still_recorded(tmp_path: Path) -> None:
    log = AuditLog(tmp_path / "a.jsonl")

    class Opaque:
        def __repr__(self) -> str:
            return "<Opaque handle>"

    log.before_call("s", "tool", {"obj": Opaque(), "b": b"\xff"})
    (rec,) = read_records(log.path)
    assert rec["arguments"]["obj"] == "<Opaque handle>"
    assert "xff" in rec["arguments"]["b"]
    assert log.failures == 0


def test_write_survives_closed_stderr(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Even the failure-reporting path must not raise."""
    import io
    import sys

    target = tmp_path / "audit.jsonl"
    target.mkdir()
    closed = io.StringIO()
    closed.close()
    monkeypatch.setattr(sys, "stderr", closed)
    log = AuditLog(target)
    assert log.write({"event": "x"}) is False
    assert log.failures == 1


# -- no off switch ------------------------------------------------------------------------------


def test_no_disable_surface_exists() -> None:
    """Invariant 5. If someone adds an enable/disable knob this test is meant to fail."""
    forbidden = {"enabled", "disable", "disabled", "off", "enable", "set_enabled", "mute"}
    public = {n for n in dir(AuditLog) if not n.startswith("_")}
    assert not (public & forbidden), public & forbidden
    init_params = set(inspect.signature(AuditLog.__init__).parameters) - {"self"}
    assert init_params == {"path"}
    src = inspect.getsource(audit_mod)
    assert "O_APPEND" in src
    assert "O_TRUNC" not in src
    assert '"w"' not in src and "'w'" not in src


def test_session_ids_are_unique() -> None:
    ids = {new_session_id() for _ in range(64)}
    assert len(ids) == 64
