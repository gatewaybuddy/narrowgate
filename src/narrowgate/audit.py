"""Append-only structured audit log (invariant 5).

Every tool call produces two records: one *before* it runs (name, arguments, session) and one
*after* it returns (outcome). Records are JSON Lines appended to the configured path.

What this module refuses to do:

* It has no off switch. There is no ``enabled`` flag, no null sink, no sampling. If you want fewer
  audit records you want a different project.
* It never raises into its caller. A full disk, a bad path, an unserialisable argument — the
  failure is printed loudly to stderr and counted in :attr:`AuditLog.failures`, and the harness
  keeps running. Silently losing an audit record is the one failure mode we cannot tolerate, so
  the failure is never silent; but a broken audit sink must not become a denial of service.
* It never truncates or rewrites the file. The file is opened with ``O_APPEND`` for every write.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any

__all__ = ["AuditLog", "new_session_id"]


def new_session_id() -> str:
    """Return a fresh random session id. Refuses to be predictable: uuid4, not a counter."""
    return uuid.uuid4().hex


def _utc_now() -> str:
    return _dt.datetime.now(tz=_dt.UTC).isoformat(timespec="milliseconds")


class AuditLog:
    """Append-only JSONL audit sink.

    Construct once per process with the path from ``config.audit.path``. The parent directory is
    created on first write if missing. The file is created ``0600``.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._failures = 0

    @property
    def path(self) -> Path:
        """Where records go. Read-only: the sink cannot be redirected after construction."""
        return self._path

    @property
    def failures(self) -> int:
        """Number of records that could not be written. Non-zero means stderr has details."""
        return self._failures

    # -- public API ---------------------------------------------------------------------------

    def before_call(self, session_id: str, tool: str, arguments: dict[str, Any]) -> str:
        """Record that ``tool`` is about to run with ``arguments``. Returns a ``call_id``.

        Refuses to raise. Refuses to redact: the arguments are logged as given, because a record
        that hides what the tool was asked to do is not an audit record.
        """
        call_id = uuid.uuid4().hex
        self.write(
            {
                "event": "tool_call.before",
                "session_id": session_id,
                "call_id": call_id,
                "tool": tool,
                "arguments": arguments,
            }
        )
        return call_id

    def after_call(
        self,
        session_id: str,
        tool: str,
        call_id: str,
        *,
        ok: bool,
        error: str | None = None,
        content_len: int | None = None,
    ) -> None:
        """Record the outcome of the call started by :meth:`before_call`. Refuses to raise."""
        self.write(
            {
                "event": "tool_call.after",
                "session_id": session_id,
                "call_id": call_id,
                "tool": tool,
                "outcome": {"ok": ok, "error": error, "content_len": content_len},
            }
        )

    def write(self, record: dict[str, Any]) -> bool:
        """Append one record (a ``ts`` field is added). Returns ``True`` on success.

        Refuses to raise for any reason. On failure prints ``narrowgate audit: FAILED ...`` to
        stderr, increments :attr:`failures`, and returns ``False``. Refuses to open the file in
        any mode other than append.
        """
        try:
            payload = {"ts": _utc_now(), **record}
            line = json.dumps(payload, default=repr, ensure_ascii=False, separators=(",", ":"))
            self._append(line + "\n")
            return True
        except Exception as exc:  # noqa: BLE001 — by contract, nothing escapes this method
            self._failures += 1
            try:
                print(
                    f"narrowgate audit: FAILED to write record to {self._path} "
                    f"({type(exc).__name__}: {exc}); event={record.get('event')!r} "
                    f"failures_so_far={self._failures}",
                    file=sys.stderr,
                    flush=True,
                )
            except Exception:  # noqa: BLE001 — even stderr can be closed; still do not raise
                pass
            return False

    # -- internals ----------------------------------------------------------------------------

    def _append(self, line: str) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self._path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
        try:
            data = line.encode("utf-8")
            view = memoryview(data)
            while view:
                n = os.write(fd, view)
                view = view[n:]
            os.fsync(fd)
        finally:
            os.close(fd)
