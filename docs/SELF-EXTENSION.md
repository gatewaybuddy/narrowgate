# Self-extension — how the agent grows its own capabilities, safely

This is the feature that makes narrowgate worth publishing, and the one most likely to be
implemented wrongly. Read all of it.

## The requirement

The operator wants the agent to help build out its own tooling once deployed. That is a real need:
the alternative is a human hand-writing every capability forever.

## Why the obvious version is unsafe

The obvious version gives the agent a way to write a Python file and a way to import it. Those are
two individually reasonable capabilities that **compose into arbitrary code execution** — the agent
writes `evil.py` and then loads it, and every boundary in the system is now decorative. Nothing in
the model has to be malicious for this to happen; a goal-pursuing system routes around friction, and
"I cannot do X" reads as friction.

**This is the capability-composition problem, and it does not have a clever solution. It has a
structural one.**

## The design: propose, never load

```
agent ──propose_tool()──▶ tools_staged/<name>.py   [INERT — never imported]
                                   │
                                   ▼
                      human runs: narrowgate activate <name>
                                   │
                        ┌──────────┴──────────┐
                        │  shows full source  │
                        │  static checks      │
                        │  typed confirmation │
                        └──────────┬──────────┘
                                   ▼
                        src/narrowgate/tools/enabled/
                        (registered on next start)
```

**The two halves run in different processes, at different times, under different authority.** The
agent's process can write to `tools_staged/` and nothing else. Only the operator's CLI can promote.

## `propose_tool` — the agent-side tool

```
propose_tool(name: str, description: str, python_source: str, rationale: str) -> ToolResult
```

MUST:
- write **only** to `tools_staged/<name>.py`, path-validated, no traversal, no overwrite of an
  already-activated tool
- return a message stating plainly that the tool is **not active** and requires operator approval —
  the agent must not be able to believe it succeeded in gaining a capability
- audit the full proposed source
- **never** import, exec, compile, or syntax-check by execution. Static parse with `ast` only.

MUST NOT:
- accept a path, a module name, or anything that resolves outside `tools_staged/`
- have any code path that ends in the new file being loaded in the same process

## `activate.py` — the human-side gate

1. Print the **full source**, not a summary. The operator reads what they are approving.
2. **Static checks** (via `ast`, never by importing):
   - reject `import os`, `subprocess`, `socket`, `ctypes`, `importlib`, `builtins`
   - reject `eval`, `exec`, `compile`, `__import__`, `getattr` on dunder names
   - reject any name matching `open` used with a mode containing `w`/`a` outside the configured root
   - require exactly one class implementing `Tool`, with `requires_network` explicitly declared
   - **every rejection prints the offending line number and source line**
3. Require the operator to type the tool's name to confirm. Not `y/N` — a habit-proof gesture.
4. Move to `tools/enabled/`, record an audit entry with the source hash and the approving user.
5. **Do not hot-load.** Registration happens on next harness start, so activation and use are
   separated in time as well as authority.

## What this deliberately does NOT do

- **It does not sandbox the proposed code.** Once a human activates a tool it runs with the harness's
  authority. The control is human review, not containment — and the design says so honestly rather
  than implying a guarantee it cannot make.
- **It does not let the agent iterate on a rejected tool automatically.** Rejection is terminal for
  that proposal; a new proposal is a new decision. This is deliberate: an agent that can retry until
  the checks pass is an agent optimising against the checks.

## Residual risk, stated plainly

A human who approves without reading has defeated this entirely. The gate converts an invisible
technical risk into a visible human decision. **That is the whole claim — it is not a guarantee of
safety, it is a guarantee that someone decided.**
