# narrowgate — architecture contract

**This file is the contract.** Every component is built against it. If an implementation disagrees
with this document, the document wins or the document gets changed first — not silently diverged from.

## Thesis

An agent harness whose safety argument does not depend on the model's cooperation.

> **A tool surface is not a sandbox. It is a narrow interface.**
> The value is not isolation — it is that every capability is something a human deliberately wrote
> and named, instead of the open set of "whatever a shell can do."

Target: a single operator running open-weight models on local hardware (NVIDIA DGX Spark / GB10)
behind a vLLM OpenAI-compatible endpoint, in an environment where data must not leave.

## Non-negotiable invariants

These are the reason the project exists. A change that breaks one is a rewrite, not a patch.

1. **No arbitrary execution tool.** There is no `run_shell`, no `eval`, no `exec`, no templating
   engine with expression support. Ever. If a capability is needed, it becomes a named tool.
2. **The agent may WRITE new tools. It may never LOAD them.** Self-extension is propose-only;
   activation is a separate human action in a separate process. (See `docs/SELF-EXTENSION.md`.)
3. **Every tool validates its own inputs** as if it were a public HTTP endpoint, because it is one.
   A narrow tool with a wide implementation is still wide.
4. **Network egress is deny-by-default**, per-tool, declared in config. A tool that can reach the
   network says so in its declaration and the operator opts in.
5. **Every tool call is audited** before it runs and after it returns — name, arguments, caller
   session, outcome. Audit writes are append-only and must not be disableable from config.
6. **No secrets in the repo.** Config carries references (env var names, file paths), never values.

## Layout

```
src/narrowgate/
  config.py      load + validate config; fail closed on anything ambiguous
  llm.py         vLLM OpenAI-compatible client; tool_call capability DETECTION
  audit.py       append-only structured audit log
  tools/
    base.py      Tool protocol, arg schema, validation helpers
    registry.py  EXPLICIT registration only — no auto-discovery, no import scanning
    builtin.py   read_file, list_dir, write_file (sandboxed to a configured root)
    propose.py   the self-extension tool (writes to tools_staged/, never loads)
  agent.py       the conversation + tool-dispatch loop
scripts/
  preflight.py   probe the vLLM endpoint: does this model emit well-formed tool_calls?
  activate.py    the human-side gate that promotes a staged tool
```

## Interfaces (build to these exactly)

```python
# tools/base.py
@dataclass(frozen=True)
class ToolResult:
    ok: bool
    content: str
    error: str | None = None

class Tool(Protocol):
    name: str                      # ^[a-z][a-z0-9_]{2,63}$
    description: str
    schema: dict                   # JSON Schema for arguments
    requires_network: bool         # declared, enforced by registry
    def run(self, **kwargs) -> ToolResult: ...
```

```python
# tools/registry.py
class Registry:
    def register(self, tool: Tool) -> None       # raises on dup name / bad name / bad schema
    def get(self, name: str) -> Tool | None
    def specs(self) -> list[dict]                # OpenAI tools=[] format
    def dispatch(self, name: str, args: dict, *, audit) -> ToolResult
```

`dispatch` MUST: reject unknown tool · validate `args` against `schema` before calling ·
refuse a `requires_network` tool unless config allows it · audit before and after · never raise
into the agent loop (return `ToolResult(ok=False, ...)`).

```python
# llm.py
@dataclass
class ToolCallSupport:
    native: bool          # server returned a structured tool_calls array
    parser_hint: str | None
    detail: str

def probe_tool_calls(
    base_url: str, model: str, *,
    api_key: str | None = None,
    timeout_s: float = 60,
    transport=None,          # test seam; never set in production
) -> ToolCallSupport: ...
```

⭐ **`probe_tool_calls` is load-bearing.** Whether Nemotron emits well-formed OpenAI `tool_calls`
under vLLM is UNVERIFIED as of 2026-09-11. The harness must **detect and report**, never assume. If
`native` is False the agent refuses to start and tells the operator which `--tool-call-parser` to try.

## Python / deps

Python ≥3.11. Standard library plus `httpx`, `pydantic`, `pyyaml`. **No agent framework dependency.**
Pin exact versions. Every dependency added must be justified in the PR.

## Definition of done, per component

- Type hints throughout; `ruff` clean
- Unit tests covering the **failure** branches, not just the happy path
- No network calls in tests
- Docstring on every public function stating what it refuses to do
