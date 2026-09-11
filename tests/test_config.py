"""config.py — the refusal branches. A config that should be rejected, and is."""

from __future__ import annotations

import copy
import os
from pathlib import Path

import pytest
import yaml

from narrowgate.config import ConfigError, load_config, resolve_api_key

GOOD: dict = {
    "llm": {
        "base_url": "http://localhost:8001/v1",
        "model": "nemotron-lightning",
        "api_key_env": "NARROWGATE_API_KEY",
        "timeout_s": 120,
        "max_tokens": 4096,
    },
    "workspace": {"root": "./workspace", "max_file_bytes": 1048576},
    "network": {"allow_tools": []},
    "audit": {"path": "./audit/narrowgate.jsonl"},
    "agent": {"max_turns": 24, "require_native_tool_calls": True},
}
ENV = {"NARROWGATE_API_KEY": "test-key-not-a-secret"}


def write(tmp_path: Path, data: object, name: str = "config.yaml") -> Path:
    p = tmp_path / name
    p.write_text(yaml.safe_dump(data) if not isinstance(data, str) else data, encoding="utf-8")
    return p


def mutate(**sections: dict) -> dict:
    cfg = copy.deepcopy(GOOD)
    for section, changes in sections.items():
        cfg[section].update(changes)
    return cfg


# -- control ------------------------------------------------------------------------------------


def test_example_config_loads(tmp_path: Path) -> None:
    cfg = load_config(write(tmp_path, GOOD), cwd=tmp_path, env=ENV)
    assert cfg.workspace.root == (tmp_path / "workspace").resolve()
    assert cfg.llm.base_url == "http://localhost:8001/v1"
    assert cfg.network.allow_tools == []
    # The secret value is never stored on the config object.
    assert "test-key-not-a-secret" not in repr(cfg)
    assert "test-key-not-a-secret" not in cfg.model_dump_json()


def test_repo_example_yaml_is_itself_valid(tmp_path: Path) -> None:
    example = Path(__file__).resolve().parents[1] / "config.example.yaml"
    cfg = load_config(example, cwd=tmp_path, env=ENV)
    assert cfg.agent.require_native_tool_calls is True


# -- workspace.root -----------------------------------------------------------------------------


@pytest.mark.parametrize("root", ["../elsewhere", "/etc", "./workspace/../../up"])
def test_rejects_workspace_root_outside_cwd(tmp_path: Path, root: str) -> None:
    p = write(tmp_path, mutate(workspace={"root": root}))
    with pytest.raises(ConfigError, match="workspace.root"):
        load_config(p, cwd=tmp_path, env=ENV)


def test_rejects_workspace_root_symlink_escape(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    (cwd / "workspace").symlink_to(outside, target_is_directory=True)
    p = write(cwd, mutate(workspace={"root": "./workspace"}))
    with pytest.raises(ConfigError, match="outside the current working directory"):
        load_config(p, cwd=cwd, env=ENV)


def test_rejects_workspace_root_that_is_a_file(tmp_path: Path) -> None:
    (tmp_path / "workspace").write_text("not a dir")
    p = write(tmp_path, GOOD)
    with pytest.raises(ConfigError, match="not a directory"):
        load_config(p, cwd=tmp_path, env=ENV)


def test_absolute_root_inside_cwd_is_accepted(tmp_path: Path) -> None:
    p = write(tmp_path, mutate(workspace={"root": str(tmp_path / "ws")}))
    cfg = load_config(p, cwd=tmp_path, env=ENV)
    assert cfg.workspace.root == (tmp_path / "ws").resolve()


# -- network.allow_tools ------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad",
    ["Read_File", "1tool", "ab", "has-dash", "has space", "x" * 65, "rm -rf /", "../escape"],
)
def test_rejects_invalid_tool_names_in_allow_tools(tmp_path: Path, bad: str) -> None:
    p = write(tmp_path, mutate(network={"allow_tools": [bad]}))
    with pytest.raises(ConfigError, match="allow_tools"):
        load_config(p, cwd=tmp_path, env=ENV)


def test_rejects_non_string_allow_tools_entry(tmp_path: Path) -> None:
    p = write(tmp_path, mutate(network={"allow_tools": [42]}))
    with pytest.raises(ConfigError, match="allow_tools"):
        load_config(p, cwd=tmp_path, env=ENV)


def test_rejects_duplicate_allow_tools(tmp_path: Path) -> None:
    p = write(tmp_path, mutate(network={"allow_tools": ["fetch_url", "fetch_url"]}))
    with pytest.raises(ConfigError, match="more than once"):
        load_config(p, cwd=tmp_path, env=ENV)


def test_rejects_allow_tools_that_is_not_a_list(tmp_path: Path) -> None:
    p = write(tmp_path, mutate(network={"allow_tools": "fetch_url"}))
    with pytest.raises(ConfigError, match="allow_tools"):
        load_config(p, cwd=tmp_path, env=ENV)


# -- missing / unknown keys ---------------------------------------------------------------------


@pytest.mark.parametrize("section", ["llm", "workspace", "network", "audit", "agent"])
def test_rejects_missing_section(tmp_path: Path, section: str) -> None:
    data = copy.deepcopy(GOOD)
    del data[section]
    with pytest.raises(ConfigError, match=section):
        load_config(write(tmp_path, data), cwd=tmp_path, env=ENV)


@pytest.mark.parametrize(
    ("section", "key"),
    [
        ("llm", "base_url"),
        ("llm", "model"),
        ("llm", "timeout_s"),
        ("llm", "max_tokens"),
        ("workspace", "root"),
        ("workspace", "max_file_bytes"),
        ("audit", "path"),
        ("agent", "max_turns"),
        ("agent", "require_native_tool_calls"),
    ],
)
def test_rejects_missing_required_key(tmp_path: Path, section: str, key: str) -> None:
    data = copy.deepcopy(GOOD)
    del data[section][key]
    with pytest.raises(ConfigError, match=f"{section}.{key}"):
        load_config(write(tmp_path, data), cwd=tmp_path, env=ENV)


def test_rejects_unknown_top_level_key(tmp_path: Path) -> None:
    data = copy.deepcopy(GOOD)
    data["shell"] = {"enabled": True}
    with pytest.raises(ConfigError, match="shell"):
        load_config(write(tmp_path, data), cwd=tmp_path, env=ENV)


def test_rejects_unknown_nested_key(tmp_path: Path) -> None:
    p = write(tmp_path, mutate(llm={"api_key": "sk-live-oops"}))
    with pytest.raises(ConfigError, match="api_key"):
        load_config(p, cwd=tmp_path, env=ENV)


@pytest.mark.parametrize("key", ["enabled", "disabled", "off", "level"])
def test_audit_cannot_be_disabled_from_config(tmp_path: Path, key: str) -> None:
    """Invariant 5: there is no off switch, so every spelling of one is an unknown key."""
    p = write(tmp_path, mutate(audit={key: False}))
    with pytest.raises(ConfigError, match=f"audit.{key}"):
        load_config(p, cwd=tmp_path, env=ENV)


def test_rejects_empty_audit_path(tmp_path: Path) -> None:
    p = write(tmp_path, mutate(audit={"path": ""}))
    with pytest.raises(ConfigError, match="audit.path"):
        load_config(p, cwd=tmp_path, env=ENV)


# -- value ranges -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("llm", "timeout_s", 0),
        ("llm", "timeout_s", -1),
        ("llm", "max_tokens", 0),
        ("llm", "base_url", "localhost:8001/v1"),
        ("llm", "base_url", ""),
        ("llm", "model", ""),
        ("workspace", "max_file_bytes", 0),
        ("agent", "max_turns", 0),
        ("agent", "require_native_tool_calls", "yes please"),
    ],
)
def test_rejects_out_of_range_values(tmp_path: Path, section: str, key: str, value: object) -> None:
    p = write(tmp_path, mutate(**{section: {key: value}}))
    with pytest.raises(ConfigError, match=f"{section}.{key}"):
        load_config(p, cwd=tmp_path, env=ENV)


# -- file shape ---------------------------------------------------------------------------------


def test_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="cannot read"):
        load_config(tmp_path / "nope.yaml", cwd=tmp_path, env=ENV)


def test_rejects_empty_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="empty"):
        load_config(write(tmp_path, ""), cwd=tmp_path, env=ENV)


def test_rejects_non_mapping_document(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="must be a mapping"):
        load_config(write(tmp_path, "- just\n- a list\n"), cwd=tmp_path, env=ENV)


def test_rejects_invalid_yaml(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not valid YAML"):
        load_config(write(tmp_path, "llm: [unclosed\n"), cwd=tmp_path, env=ENV)


def test_rejects_yaml_python_tags(tmp_path: Path) -> None:
    """safe_load only: a config file must not be able to construct arbitrary objects."""
    doc = "llm: !!python/object/apply:os.system ['echo pwned']\n"
    with pytest.raises(ConfigError, match="not valid YAML"):
        load_config(write(tmp_path, doc), cwd=tmp_path, env=ENV)


# -- secrets ------------------------------------------------------------------------------------


def test_rejects_unset_api_key_env(tmp_path: Path) -> None:
    p = write(tmp_path, GOOD)
    with pytest.raises(ConfigError, match="NARROWGATE_API_KEY.*not set"):
        load_config(p, cwd=tmp_path, env={})


def test_rejects_blank_api_key_env(tmp_path: Path) -> None:
    p = write(tmp_path, GOOD)
    with pytest.raises(ConfigError, match="not set"):
        load_config(p, cwd=tmp_path, env={"NARROWGATE_API_KEY": "   "})


@pytest.mark.parametrize("bad", ["sk-abc123", "lower_case", "has-dash", ""])
def test_rejects_api_key_env_that_is_not_a_var_name(tmp_path: Path, bad: str) -> None:
    p = write(tmp_path, mutate(llm={"api_key_env": bad}))
    with pytest.raises(ConfigError, match="api_key_env"):
        load_config(p, cwd=tmp_path, env={bad: "x"})


def test_null_api_key_env_means_unauthenticated(tmp_path: Path) -> None:
    p = write(tmp_path, mutate(llm={"api_key_env": None}))
    cfg = load_config(p, cwd=tmp_path, env={})
    assert cfg.llm.api_key_env is None
    assert resolve_api_key(cfg, {}) is None


def test_resolve_api_key_reads_process_env_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NARROWGATE_API_KEY", "from-process-env")
    cfg = load_config(write(tmp_path, GOOD), cwd=tmp_path)
    assert resolve_api_key(cfg) == "from-process-env"
    monkeypatch.delenv("NARROWGATE_API_KEY")
    assert "NARROWGATE_API_KEY" not in os.environ
    with pytest.raises(ConfigError, match="not set"):
        resolve_api_key(cfg)


def test_default_cwd_is_process_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    cfg = load_config(write(tmp_path, GOOD), env=ENV)
    assert cfg.workspace.root == (tmp_path / "workspace").resolve()
