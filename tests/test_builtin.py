"""Built-in file tools. The escape cases use real filesystem objects, not string matching."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from narrowgate.tools.base import ToolResult
from narrowgate.tools.builtin import (
    ListDirTool,
    ReadFileTool,
    Workspace,
    WriteFileTool,
    builtin_tools,
)
from narrowgate.tools.registry import Registry

LIMIT = 64


class Env:
    def __init__(self, tmp_path: Path) -> None:
        self.root = tmp_path / "root"
        self.outside = tmp_path / "outside"
        (self.root / "sub").mkdir(parents=True)
        self.outside.mkdir()
        (self.root / "hello.txt").write_text("hello\n")
        (self.root / "sub" / "inner.txt").write_text("inner\n")
        (self.outside / "secret.txt").write_text("SECRET\n")
        self.ws = Workspace(self.root, LIMIT)
        self.read = ReadFileTool(self.ws)
        self.ls = ListDirTool(self.ws)
        self.write = WriteFileTool(self.ws)


class NullAudit:
    def before_call(self, **kwargs: object) -> None:
        pass

    def after_call(self, **kwargs: object) -> None:
        pass


@pytest.fixture
def env(tmp_path: Path) -> Env:
    return Env(tmp_path)


def refused(res: ToolResult) -> str:
    assert not res.ok, f"expected refusal, got ok with content {res.content!r}"
    assert res.error
    return res.error


# --- declarations -----------------------------------------------------------------------------


def test_builtin_tools_declare_no_network_and_register(tmp_path: Path) -> None:
    (tmp_path / "w").mkdir()
    tools = builtin_tools(tmp_path / "w", 1024)
    assert [t.name for t in tools] == ["read_file", "list_dir", "write_file"]
    assert all(t.requires_network is False for t in tools)
    reg = Registry()
    for t in tools:
        reg.register(t)
    assert reg.names() == ["list_dir", "read_file", "write_file"]


def test_workspace_refuses_bad_root_and_limit(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        Workspace(tmp_path / "missing", 10)
    (tmp_path / "f").write_text("x")
    with pytest.raises(ValueError):
        Workspace(tmp_path / "f", 10)
    (tmp_path / "d").mkdir()
    for bad in (0, -1, True, "10"):
        with pytest.raises(ValueError):
            Workspace(tmp_path / "d", bad)  # type: ignore[arg-type]


# --- happy paths ------------------------------------------------------------------------------


def test_read_list_write_roundtrip(env: Env) -> None:
    assert env.read.run(path="hello.txt") == ToolResult(True, "hello\n")
    assert env.read.run(path="sub/inner.txt").content == "inner\n"
    assert env.read.run(path="./sub/inner.txt").content == "inner\n"

    assert env.ls.run(path="").content == "hello.txt\nsub/"
    assert env.ls.run(path=".").content == "hello.txt\nsub/"
    assert env.ls.run(path="sub/").content == "inner.txt"
    assert env.ls.run().content == "hello.txt\nsub/"

    res = env.write.run(path="sub/new.txt", content="fresh")
    assert res.ok and "5 bytes" in res.content
    assert (env.root / "sub" / "new.txt").read_text() == "fresh"
    assert env.write.run(path="sub/new.txt", content="over").ok
    assert (env.root / "sub" / "new.txt").read_text() == "over"
    assert env.write.run(path="empty.txt", content="").ok
    assert (env.root / "empty.txt").read_bytes() == b""


def test_list_dir_empty_and_truncation(env: Env) -> None:
    (env.root / "e").mkdir()
    assert env.ls.run(path="e").content == "(empty)"
    for i in range(5):
        (env.root / "e" / f"f{i}").write_text("")
    env.ls.max_entries = 3
    out = env.ls.run(path="e").content.splitlines()
    assert out[:3] == ["f0", "f1", "f2"] and "2 more" in out[3]


# --- traversal --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path",
    [
        "..",
        "../outside/secret.txt",
        "sub/../../outside/secret.txt",
        "sub/..",
        "../root/hello.txt",  # would resolve back inside; still refused
        "/etc/passwd",
        "//etc/passwd",
        str(Path("/tmp").resolve()),
        "~/x",
        "~",
        "sub\\..\\hello.txt",
        "C:hello.txt",
        "c:/hello.txt",
        "sub//inner.txt",
        "hello.txt\x00",
        "hel\x00lo.txt",
        "hello\n.txt",
        "hello\x1b.txt",
        "hello\x7f.txt",
        "x" * 1025,
        "",
    ],
)
def test_read_refuses_traversal_and_nonsense(env: Env, path: str) -> None:
    err = refused(env.read.run(path=path))
    assert "SECRET" not in err
    assert str(env.root) not in err  # never leaks the absolute root


@pytest.mark.parametrize("path", ["../outside/x.txt", "/tmp/x.txt", "sub/../../x.txt", "x\x00"])
def test_write_refuses_traversal_and_creates_nothing(env: Env, path: str) -> None:
    before = set(os.listdir(env.outside)) | set(os.listdir("/tmp"))
    refused(env.write.run(path=path, content="pwned"))
    after = set(os.listdir(env.outside)) | set(os.listdir("/tmp"))
    assert after == before


@pytest.mark.parametrize("path", ["..", "../outside", "/", "sub/../.."])
def test_list_refuses_traversal(env: Env, path: str) -> None:
    refused(env.ls.run(path=path))


def test_non_str_path_is_refused_not_raised(env: Env) -> None:
    for bad in (5, None, ["hello.txt"], b"hello.txt"):
        refused(env.read.run(path=bad))
        refused(env.ls.run(path=bad))
        refused(env.write.run(path=bad, content="x"))
    refused(env.read.run())  # missing path
    refused(env.write.run(path="a.txt"))  # missing content
    refused(env.write.run(path="a.txt", content=b"bytes"))
    assert not (env.root / "a.txt").exists()


# --- symlink escape ---------------------------------------------------------------------------


def test_symlink_file_pointing_outside_is_refused(env: Env) -> None:
    link = env.root / "leak.txt"
    link.symlink_to(env.outside / "secret.txt")
    assert link.read_text() == "SECRET\n"  # the OS would happily follow it

    err = refused(env.read.run(path="leak.txt"))
    assert "symlink" in err
    refused(env.write.run(path="leak.txt", content="pwned"))
    assert (env.outside / "secret.txt").read_text() == "SECRET\n"


def test_symlink_dir_pointing_outside_is_refused(env: Env) -> None:
    (env.root / "esc").symlink_to(env.outside, target_is_directory=True)
    assert (env.root / "esc" / "secret.txt").read_text() == "SECRET\n"

    refused(env.read.run(path="esc/secret.txt"))
    refused(env.ls.run(path="esc"))
    refused(env.ls.run(path="esc/"))
    refused(env.write.run(path="esc/new.txt", content="pwned"))
    assert not (env.outside / "new.txt").exists()


def test_relative_symlink_climbing_out_is_refused(env: Env) -> None:
    # A relative link whose text contains '..' — the path string given to the tool does not.
    (env.root / "sub" / "up").symlink_to(Path("..") / ".." / "outside")
    assert (env.root / "sub" / "up" / "secret.txt").read_text() == "SECRET\n"
    refused(env.read.run(path="sub/up/secret.txt"))
    refused(env.write.run(path="sub/up/x.txt", content="pwned"))
    assert not (env.outside / "x.txt").exists()


def test_symlink_pointing_inside_root_is_still_refused(env: Env) -> None:
    """Refusing all symlinks is the policy: a link inside today can point outside tomorrow."""
    (env.root / "alias.txt").symlink_to(env.root / "hello.txt")
    (env.root / "subalias").symlink_to(env.root / "sub", target_is_directory=True)
    refused(env.read.run(path="alias.txt"))
    refused(env.read.run(path="subalias/inner.txt"))
    refused(env.ls.run(path="subalias"))
    refused(env.write.run(path="alias.txt", content="x"))
    assert (env.root / "hello.txt").read_text() == "hello\n"


def test_dangling_symlink_is_refused_for_write(env: Env) -> None:
    (env.root / "dangling.txt").symlink_to(env.outside / "created-by-write.txt")
    refused(env.write.run(path="dangling.txt", content="pwned"))
    assert not (env.outside / "created-by-write.txt").exists()
    refused(env.read.run(path="dangling.txt"))


def test_symlinks_are_listed_with_marker_but_not_followed(env: Env) -> None:
    (env.root / "esc").symlink_to(env.outside, target_is_directory=True)
    (env.root / "leak.txt").symlink_to(env.outside / "secret.txt")
    out = env.ls.run(path="").content.splitlines()
    assert "esc@" in out and "leak.txt@" in out
    assert "secret.txt" not in env.ls.run(path="").content


def test_symlinked_root_itself_is_fine(tmp_path: Path) -> None:
    """Only the root may be a symlink (e.g. ./workspace -> elsewhere); it is resolved once."""
    real = tmp_path / "real"
    real.mkdir()
    (real / "a.txt").write_text("a")
    (tmp_path / "rootlink").symlink_to(real, target_is_directory=True)
    ws = Workspace(tmp_path / "rootlink", LIMIT)
    assert ws.root == real.resolve()
    assert ReadFileTool(ws).run(path="a.txt").content == "a"
    assert WriteFileTool(ws).run(path="b.txt", content="b").ok
    assert (real / "b.txt").read_text() == "b"


# --- size limits ------------------------------------------------------------------------------


def test_read_refuses_oversized_file(env: Env) -> None:
    (env.root / "big.txt").write_bytes(b"x" * (LIMIT + 1))
    assert "larger than" in refused(env.read.run(path="big.txt"))
    (env.root / "exact.txt").write_bytes(b"x" * LIMIT)
    assert env.read.run(path="exact.txt").ok


def test_write_refuses_oversized_content_in_bytes_not_chars(env: Env) -> None:
    refused(env.write.run(path="big.txt", content="x" * (LIMIT + 1)))
    assert not (env.root / "big.txt").exists()
    assert env.write.run(path="ok.txt", content="x" * LIMIT).ok
    # 'é' is 2 bytes in UTF-8: LIMIT/2 + 1 of them exceed the byte limit despite fewer chars
    refused(env.write.run(path="wide.txt", content="é" * (LIMIT // 2 + 1)))
    assert not (env.root / "wide.txt").exists()
    assert env.write.run(path="wide-ok.txt", content="é" * (LIMIT // 2)).ok


def test_write_schema_caps_content_length(env: Env) -> None:
    assert env.write.schema["properties"]["content"]["maxLength"] == LIMIT


# --- wrong kind of filesystem object ----------------------------------------------------------


def test_read_refuses_directory_and_special_files(env: Env) -> None:
    assert "directory" in refused(env.read.run(path="sub"))
    refused(env.read.run(path=""))
    os.mkfifo(env.root / "pipe")
    assert "regular" in refused(env.read.run(path="pipe"))


def test_list_refuses_file(env: Env) -> None:
    assert "not a directory" in refused(env.ls.run(path="hello.txt"))
    assert "not a directory" in refused(env.ls.run(path="sub/inner.txt"))


def test_write_refuses_directory_targets_and_missing_parent(env: Env) -> None:
    assert "directory" in refused(env.write.run(path="sub", content="x"))
    assert (env.root / "sub").is_dir()
    assert "directory" in refused(env.write.run(path="sub/", content="x"))
    assert "parent" in refused(env.write.run(path="nope/new.txt", content="x"))
    assert not (env.root / "nope").exists()
    assert "directory" in refused(env.write.run(path="hello.txt/child", content="x"))
    os.mkfifo(env.root / "pipe")
    assert "regular" in refused(env.write.run(path="pipe", content="x"))


def test_read_refuses_non_utf8(env: Env) -> None:
    (env.root / "bin.dat").write_bytes(b"\xff\xfe\x00\x01")
    assert "UTF-8" in refused(env.read.run(path="bin.dat"))


def test_write_refuses_lone_surrogates(env: Env) -> None:
    assert "UTF-8" in refused(env.write.run(path="s.txt", content="\ud800"))
    assert not (env.root / "s.txt").exists()


# --- through the registry ---------------------------------------------------------------------


def test_end_to_end_via_registry(env: Env) -> None:
    reg = Registry()
    for t in builtin_tools(env.root, LIMIT):
        reg.register(t)
    (env.root / "leak.txt").symlink_to(env.outside / "secret.txt")
    audit = NullAudit()

    assert reg.dispatch("read_file", {"path": "hello.txt"}, audit=audit).content == "hello\n"
    assert not reg.dispatch("read_file", {"path": "../outside/secret.txt"}, audit=audit).ok
    assert not reg.dispatch("read_file", {"path": "leak.txt"}, audit=audit).ok
    assert not reg.dispatch("read_file", {}, audit=audit).ok
    assert not reg.dispatch("read_file", {"path": "hello.txt", "mode": "rb"}, audit=audit).ok
    assert not reg.dispatch("write_file", {"path": "x", "content": "x" * 65}, audit=audit).ok
    assert not reg.dispatch("list_dir", {"path": 3}, audit=audit).ok
    assert reg.dispatch("list_dir", {}, audit=audit).ok
    for spec in reg.specs():
        assert spec["function"]["parameters"]["additionalProperties"] is False
