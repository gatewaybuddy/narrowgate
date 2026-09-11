# narrowgate

A minimal agent harness for local open-weight models, built so that **its safety argument does not
depend on the model's cooperation.**

Targets a vLLM OpenAI-compatible endpoint on local hardware (built against NVIDIA DGX Spark / GB10)
in environments where data must not leave the machine.

## Why another harness

Most agent harnesses grow a `run_shell` tool early, because it is the fastest way to make the agent
useful. That single tool subsumes every other boundary in the system — and it does so silently, since
nothing about it looks like a security decision at the time.

narrowgate takes the opposite trade deliberately:

> **A tool surface is not a sandbox. It is a narrow interface.**
> The value is not isolation — it is that every capability is something a human deliberately wrote
> and named, instead of the open set of "whatever a shell can do."

Slower to make useful. Much harder to make dangerous by accident.

## Invariants

1. **No arbitrary execution tool.** No `run_shell`, no `eval`, no templating with expression support.
2. **The agent may WRITE new tools. It may never LOAD them.** See [self-extension](docs/SELF-EXTENSION.md).
3. **Every tool validates its inputs like a public HTTP endpoint** — because that is what it is.
4. **Egress is deny-by-default**, declared per tool, opted into by the operator.
5. **Every tool call is audited** before and after. Audit cannot be disabled from config.
6. **No secrets in the repo.** Config holds references, never values.

## Self-extension, in one picture

```
agent ──propose_tool()──▶ tools_staged/   [INERT — never imported]
                               │
                    human runs: narrowgate activate <name>
                               │  full source shown · static checks · typed confirmation
                               ▼
                    tools/enabled/  (registered on next start)
```

The two halves run in **different processes, at different times, under different authority.** The
agent can write a proposal and nothing else. Only the operator can promote it.

This does not sandbox approved code — once activated, a tool runs with the harness's authority. The
control is human review, not containment, and the docs say so rather than implying a guarantee they
cannot make.

## Status

Early. Built 2026-09-11. Not yet run against a production workload.

⚠️ **Whether a given model emits well-formed OpenAI `tool_calls` under vLLM is model- and
version-specific.** `scripts/preflight.py` probes for it and the harness **refuses to start** rather
than guessing — if your model needs a `--tool-call-parser`, you will be told which to try.

## Quick start

```bash
pip install -e .
cp config.example.yaml config.yaml     # edit: base_url, model, workspace_root
python -m narrowgate.preflight         # probe the endpoint before anything else
python -m narrowgate                   # start the agent loop
```

## Docs

- [Architecture contract](docs/ARCHITECTURE.md) — the interfaces, and what each component must refuse
- [Self-extension](docs/SELF-EXTENSION.md) — the propose/activate split, and its residual risk

## Licence

MIT. See [LICENSE](LICENSE).
