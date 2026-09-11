"""Built-in file tools: ``read_file``, ``list_dir``, ``write_file``. All confined to one root.

Every tool here treats its ``path`` argument the way a public HTTP endpoint treats a URL: it is
attacker-controlled until proven otherwise. The single choke point is ``Workspace.confine``,
which turns an untrusted relative path into a real path under the root or refuses.

Symlinks are refused outright, not merely checked. A symlink that happens to point inside the
root today can point outside tomorrow, and a path whose components are all real directories
under the root is the only kind whose meaning cannot change between the check and the open.

None of these tools touches the network; each declares ``requires_network = False``.
"""

from __future__ import annotations

import os
import re
import stat
from pathlib import Path
from typing import Any

from narrowgate.tools.base import ArgumentError, Tool, ToolResult, require_str

MAX_PATH_CHARS = 1024
_DRIVE_LETTER = re.compile(r"^[A-Za-z]:")
_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]")


class PathRefused(ValueError):
    """The path was refused by ``Workspace.confine``. The message names the reason, not the root."""


class Workspace:
    """A resolved root directory plus the size limit every file tool shares."""

    def __init__(self, root: Path | str, max_file_bytes: int) -> None:
        if isinstance(max_file_bytes, bool) or not isinstance(max_file_bytes, int):
            raise ValueError("max_file_bytes must be an int")
        if max_file_bytes <= 0:
            raise ValueError("max_file_bytes must be > 0")
        real = Path(root).resolve()
        if not real.is_dir():
            raise ValueError("workspace root must be an existing directory")
        self.root = real
        self.max_file_bytes = max_file_bytes

    def confine(self, rel: str, *, allow_missing_leaf: bool = False) -> Path:
        """Map a relative ``rel`` to a real path under the root, or raise ``PathRefused``.

        Refuses: non-``str``; NUL or any other control character; more than
        ``MAX_PATH_CHARS`` characters; backslashes; a drive-letter prefix; an absolute path; a
        leading ``~``; an empty segment (``a//b``); any ``..`` segment anywhere; any component
        under the root that is a symlink, whether it points inside or outside the root; a
        missing component, unless it is the last one and ``allow_missing_leaf`` is set; and,
        as a final independent check, any result whose ``os.path.realpath`` is not under the
        root. Never follows a symlink and never creates anything.
        """
        if not isinstance(rel, str):
            raise PathRefused(f"path must be a str, got {type(rel).__name__}")
        if len(rel) > MAX_PATH_CHARS:
            raise PathRefused(f"path longer than {MAX_PATH_CHARS} characters")
        if _CONTROL_CHARS.search(rel):
            raise PathRefused("path contains a control character")
        if "\\" in rel:
            raise PathRefused("path contains a backslash")
        if _DRIVE_LETTER.match(rel):
            raise PathRefused("path has a drive-letter prefix")
        if rel.startswith("/"):
            raise PathRefused("absolute paths are not allowed")
        if rel.startswith("~"):
            raise PathRefused("home-directory paths are not allowed")

        segments = rel.split("/")
        if len(segments) > 1 and segments[-1] == "":
            segments.pop()  # a single trailing slash is tolerated
        parts: list[str] = []
        for seg in segments:
            if seg in ("", ".") and rel not in ("", "."):
                if seg == "":
                    raise PathRefused("path contains an empty segment")
                continue
            if seg == "..":
                raise PathRefused("path contains a '..' segment")
            if seg not in ("", "."):
                parts.append(seg)

        target = self.root
        for i, seg in enumerate(parts):
            target = target / seg
            try:
                st = os.lstat(target)
            except FileNotFoundError:
                if i < len(parts) - 1:
                    raise PathRefused(f"{rel!r}: parent directory does not exist") from None
                if allow_missing_leaf:
                    break
                raise PathRefused(f"{rel!r}: not found") from None
            except OSError as exc:
                raise PathRefused(f"{rel!r}: {exc.strerror}") from None
            if stat.S_ISLNK(st.st_mode):
                raise PathRefused(f"{rel!r}: path contains a symlink")

        real = Path(os.path.realpath(target))
        if real != self.root and not real.is_relative_to(self.root):
            raise PathRefused(f"{rel!r}: resolves outside the workspace")
        return target


def _path_schema(*, required: bool, description: str) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "minLength": 0 if not required else 1,
                "maxLength": MAX_PATH_CHARS,
                "description": description,
            }
        },
        "required": ["path"] if required else [],
        "additionalProperties": False,
    }


def _refuse(message: str) -> ToolResult:
    return ToolResult(False, "", message)


class ReadFileTool:
    """Read one UTF-8 text file under the workspace root."""

    name = "read_file"
    description = (
        "Read a UTF-8 text file inside the workspace. Path is relative to the workspace root."
    )
    requires_network = False

    def __init__(self, workspace: Workspace) -> None:
        self.ws = workspace
        self.schema = _path_schema(required=True, description="Relative path of the file to read")

    def run(self, **kwargs: Any) -> ToolResult:
        """Return the file's text, or ``ok=False`` with the reason.

        Refuses everything ``Workspace.confine`` refuses; a directory, FIFO, socket, or device;
        a file larger than ``max_file_bytes`` (checked before and again after opening); and
        bytes that are not valid UTF-8. Opens with ``O_NOFOLLOW`` so a symlink swapped in
        between the check and the open is refused by the kernel.
        """
        try:
            rel = require_str(kwargs, "path", max_len=MAX_PATH_CHARS)
            target = self.ws.confine(rel)
        except (ArgumentError, PathRefused) as exc:
            return _refuse(str(exc))
        try:
            st = os.lstat(target)
            if stat.S_ISDIR(st.st_mode):
                return _refuse(f"{rel!r} is a directory, not a file")
            if not stat.S_ISREG(st.st_mode):
                return _refuse(f"{rel!r} is not a regular file")
            if st.st_size > self.ws.max_file_bytes:
                return _refuse(f"{rel!r} is larger than {self.ws.max_file_bytes} bytes")
            data = _read_regular(target, self.ws.max_file_bytes)
        except OSError as exc:
            return _refuse(f"{rel!r}: {exc.strerror or exc}")
        except _NotRegular:
            return _refuse(f"{rel!r} is not a regular file")
        except _TooLarge:
            return _refuse(f"{rel!r} is larger than {self.ws.max_file_bytes} bytes")
        try:
            return ToolResult(True, data.decode("utf-8"))
        except UnicodeDecodeError:
            return _refuse(f"{rel!r} is not valid UTF-8 text")


class ListDirTool:
    """List one directory under the workspace root, without following anything."""

    name = "list_dir"
    description = (
        "List entries of a directory inside the workspace. Empty path means the workspace root. "
        "Directories end with '/', symlinks with '@'."
    )
    requires_network = False
    max_entries = 2000

    def __init__(self, workspace: Workspace) -> None:
        self.ws = workspace
        self.schema = _path_schema(
            required=False, description="Relative path of the directory; empty for the root"
        )

    def run(self, **kwargs: Any) -> ToolResult:
        """Return one entry per line, sorted, or ``ok=False`` with the reason.

        Refuses everything ``Workspace.confine`` refuses and a path that is not a directory.
        Never descends, never follows symlinks (they are listed with an ``@`` marker and
        nothing else), and truncates after ``max_entries`` with a note saying so.
        """
        try:
            rel = require_str(
                {"path": "", **kwargs}, "path", max_len=MAX_PATH_CHARS, allow_empty=True
            )
            target = self.ws.confine(rel)
        except (ArgumentError, PathRefused) as exc:
            return _refuse(str(exc))
        try:
            st = os.lstat(target)
            if not stat.S_ISDIR(st.st_mode):
                return _refuse(f"{rel or '.'!r} is not a directory")
            lines: list[str] = []
            with os.scandir(target) as it:
                for entry in it:
                    if entry.is_symlink():
                        lines.append(entry.name + "@")
                    elif entry.is_dir(follow_symlinks=False):
                        lines.append(entry.name + "/")
                    else:
                        lines.append(entry.name)
        except OSError as exc:
            return _refuse(f"{rel or '.'!r}: {exc.strerror or exc}")
        lines.sort()
        if len(lines) > self.max_entries:
            hidden = len(lines) - self.max_entries
            lines = lines[: self.max_entries] + [f"... ({hidden} more entries not shown)"]
        return ToolResult(True, "\n".join(lines) if lines else "(empty)")


class WriteFileTool:
    """Create or overwrite one UTF-8 text file under the workspace root."""

    name = "write_file"
    description = (
        "Write UTF-8 text to a file inside the workspace, creating or replacing it. "
        "The parent directory must already exist."
    )
    requires_network = False

    def __init__(self, workspace: Workspace) -> None:
        self.ws = workspace
        self.schema = {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": MAX_PATH_CHARS,
                    "description": "Relative path of the file to write",
                },
                "content": {
                    "type": "string",
                    "maxLength": workspace.max_file_bytes,
                    "description": "Full text content of the file",
                },
            },
            "required": ["path", "content"],
            "additionalProperties": False,
        }

    def run(self, **kwargs: Any) -> ToolResult:
        """Write ``content`` to ``path`` and report the byte count, or ``ok=False`` with the reason.

        Refuses everything ``Workspace.confine`` refuses; a path whose parent does not exist
        (it does not create directories); an existing target that is not a regular file; content
        that is not a ``str``, cannot be encoded as UTF-8, or exceeds ``max_file_bytes`` once
        encoded. Opens with ``O_NOFOLLOW`` and checks the opened descriptor is a regular file
        before writing, so nothing outside the root can be written through a swapped-in link.
        """
        try:
            rel = require_str(kwargs, "path", max_len=MAX_PATH_CHARS)
            target = self.ws.confine(rel, allow_missing_leaf=True)
        except (ArgumentError, PathRefused) as exc:
            return _refuse(str(exc))
        content = kwargs.get("content")
        if not isinstance(content, str):
            return _refuse(f"'content' must be a str, got {type(content).__name__}")
        try:
            data = content.encode("utf-8")
        except UnicodeEncodeError:
            return _refuse("'content' is not encodable as UTF-8")
        if len(data) > self.ws.max_file_bytes:
            return _refuse(f"content is {len(data)} bytes; limit is {self.ws.max_file_bytes}")
        if target == self.ws.root:
            return _refuse(f"{rel!r} is a directory, not a file")
        try:
            if not stat.S_ISDIR(os.lstat(target.parent).st_mode):
                return _refuse(f"parent of {rel!r} is not a directory")
        except FileNotFoundError:
            return _refuse(f"parent of {rel!r} does not exist")
        try:
            st = os.lstat(target)
        except FileNotFoundError:
            st = None
        except OSError as exc:
            return _refuse(f"{rel!r}: {exc.strerror or exc}")
        if st is not None:
            if stat.S_ISDIR(st.st_mode):
                return _refuse(f"{rel!r} is a directory, not a file")
            if not stat.S_ISREG(st.st_mode):
                return _refuse(f"{rel!r} exists and is not a regular file")
        try:
            _write_regular(target, data)
        except OSError as exc:
            return _refuse(f"{rel!r}: {exc.strerror or exc}")
        except _NotRegular:
            return _refuse(f"{rel!r} is not a regular file")
        return ToolResult(True, f"wrote {len(data)} bytes to {rel!r}")


class _NotRegular(Exception):
    pass


class _TooLarge(Exception):
    pass


def _read_regular(target: Path, limit: int) -> bytes:
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK
    fd = os.open(target, flags)
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            raise _NotRegular
        if st.st_size > limit:
            raise _TooLarge
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(fd, 65536)
            if not chunk:
                break
            total += len(chunk)
            if total > limit:
                raise _TooLarge
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(fd)


def _write_regular(target: Path, data: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK
    fd = os.open(target, flags, 0o644)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise _NotRegular
        os.ftruncate(fd, 0)
        view = memoryview(data)
        while view:
            written = os.write(fd, view)
            view = view[written:]
    finally:
        os.close(fd)


def builtin_tools(root: Path | str, max_file_bytes: int) -> list[Tool]:
    """Instantiate the three built-in tools over one ``Workspace``.

    Returns them; it does not register them anywhere and does not discover anything else.
    Refuses (``ValueError``) a root that is not an existing directory or a non-positive limit.
    """
    ws = Workspace(root, max_file_bytes)
    return [ReadFileTool(ws), ListDirTool(ws), WriteFileTool(ws)]
