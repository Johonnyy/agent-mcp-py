# agent-mcp-py

The convention layer every app in the ecosystem uses to expose itself as an MCP
server. Wraps the official `mcp` Python SDK with bearer auth, a conversation-depth
guard, usage logging, and sync-store registration already wired in.

See [CLAUDE.md](CLAUDE.md) for the design spec and the ecosystem context.

**This repo implements v0.1.0 — all seven modules, 175 tests, verified end to end
against a live server over streamable HTTP.**

```
app author writes                agent_mcp adds                     the SDK does
─────────────────                ──────────────                     ────────────
@mcp.tool(read_only=True)   →    bearer auth (401 at the edge)  →   tools/list
def get_balance(...)             depth guard (X-Agent-Depth)        tools/call
                                 confirmation gate (X-Confirmed)    resources/read
                                 usage row per call
                                 $ref-free JSON schema
```

## Install

Consumed as a pinned git dependency — never from PyPI:

```toml
# pyproject.toml
agent-mcp-py = { git = "https://github.com/Johonnyy/agent-mcp-py", tag = "v0.1.0" }
```

```bash
# pip equivalent
pip install "agent-mcp-py @ git+https://github.com/Johonnyy/agent-mcp-py@v0.1.0"
```

The distribution is `agent-mcp-py`; the import is **`agent_mcp`**. Mismatched on
purpose so the package name matches the repo.

## Usage

```python
from fastapi import FastAPI
from contextlib import asynccontextmanager
from agent_mcp import AgentMCPServer

mcp = AgentMCPServer(
    app_name="finance",
    version="0.1.0",
    sync_store_url=settings.sync_store_url,
    auth_keys_env="FINANCE_MCP_KEYS",   # comma-separated "name:token" pairs
)

@mcp.resource("finance://transactions/recent{?limit}")
def recent_transactions(limit: int = 20) -> list[dict]:
    """Every resource URI mirrors a real dashboard view — this is what the
    transactions table on the frontend renders from too."""
    return db.get_recent_transactions(limit)

@mcp.tool(read_only=True)
def get_balance(account: str) -> float:
    return db.get_balance(account)

@mcp.tool(read_only=False, requires_confirmation=True)
def create_invoice(customer_id: str, amount: float) -> dict:
    return db.create_invoice(customer_id, amount)

@asynccontextmanager
async def lifespan(app: FastAPI):
    async with mcp.lifespan():        # required — see "Mounting" below
        yield

app = FastAPI(lifespan=lifespan)
app.router.routes.extend(mcp.routes())        # MCP at /mcp
app.router.routes.extend(mcp.usage_routes())  # summary at /agent/usage
```

Both decorators return the function **unwrapped**, so the app's own HTTP handler
calls the very same object the model calls — a tool mirrors a real user action
rather than a parallel code path that drifts from it.

## Mounting

Use `mcp.routes()`. `app.mount("/mcp", mcp.asgi_app())` also works but has a sharp
edge: `Mount("/mcp")` only matches `/mcp/...`, so a bare `POST /mcp` falls through
to the host router's `redirect_slashes` and gets a 307 — and **the real MCP client
does not follow that redirect**. `routes()` claims the exact path too, so both forms
work. If you mount by hand, make sure every caller uses the trailing slash;
`agent_mcp.mcp_url()` always produces it.

The `lifespan` is not optional. A mounted sub-app's own lifespan never runs, so the
host must enter `session_manager.run()` or the very first request fails.

## Layout

| Path | Role |
|---|---|
| [depth.py](src/agent_mcp/depth.py) | **stdlib-only leaf.** Header constants, `MAX_AGENT_DEPTH`, `DepthExceeded`, next-hop threading |
| [auth.py](src/agent_mcp/auth.py) | **stdlib-only leaf.** Bearer verification as pure functions |
| [schema.py](src/agent_mcp/schema.py) | Tool-name rules, JSON-Schema `$ref` inlining |
| [registry.py](src/agent_mcp/registry.py) | Peer discovery — `PeerRecord`, `resolve()`, `refresh()` |
| [config.py](src/agent_mcp/config.py) | `AgentMCPSettings` (env prefix `AGENT_MCP_`) |
| [usage_log.py](src/agent_mcp/usage_log.py) | `UsageSink` protocol, SQLite default, `/agent/usage` data |
| [errors.py](src/agent_mcp/errors.py) | The error taxonomy: what becomes a protocol error vs. `isError` text |
| [decorators.py](src/agent_mcp/decorators.py) | The signature surgery and the guard path |
| [middleware.py](src/agent_mcp/middleware.py) | Pure-ASGI auth gate, mount-path normalisation |
| [sync_client.py](src/agent_mcp/sync_client.py) | `register()` / `heartbeat_loop()` |
| [server.py](src/agent_mcp/server.py) | `AgentMCPServer` — composes all of the above |

## Conventions this enforces

- **Query tools are `read_only=True`**, risky ones `requires_confirmation=True`. Both
  are visible on the wire: `readOnlyHint` in standard annotations, and both flags in
  `_meta` under `dev.johnny.agent-mcp/` (the SDK's `ToolAnnotations` silently drops
  unknown fields, so a custom one there would vanish). Enforcement always reads the
  in-process policy, never the wire.
- **Depth is capped at 5 hops** via `X-Conversation-Id` / `X-Agent-Depth`.
  `mcp.call_peer()` threads them automatically and refuses an over-cap hop before it
  leaves the process.
- **Tool names must be snake_case, ≤40 chars, no `__`** — `agent_runtime` namespaces
  them `<server>__<tool>`, and the double underscore is the separator. Violations
  fail at import, not at call time.
- **Input schemas ship `$ref`-free.** Pydantic emits `$defs`/`$ref` for nested
  models and several model providers reject them, so they are inlined at
  registration. A recursive model is refused with a message saying what to do
  instead.
- **Usage stays local.** Rows go to this app's own database, never a shared one. The
  spawner aggregates by querying each app's `/agent/usage`.
- **No model names anywhere, and no LLM SDK dependency.** If a resource needs a
  model, the host app passes in a callable. A test enforces this.

## Error taxonomy

| Failure | What the caller sees |
|---|---|
| Missing/bad bearer token | HTTP **401** at the ASGI edge |
| Malformed `X-Agent-Depth` | `MCPError(INVALID_PARAMS)` |
| Inbound call over the depth cap | `MCPError(INVALID_REQUEST)` |
| Outbound `call_peer` over the cap | ordinary exception → `isError: true` |
| Missing `X-Confirmed` | ordinary exception → `isError: true` |
| Any exception in a tool body | `isError: true` |
| Sync store unreachable | a logged warning, nothing else |

The inbound/outbound asymmetry is deliberate: an outbound refusal happens inside a
tool body, where a bad peer call must not take down an otherwise fine turn.

## Interop with `agent-runtime`

`agent_runtime` imports `agent_mcp.depth` for the header constants and the cap, so
it can pre-check client-side. That module is **stdlib-only**, and `agent_mcp`'s
`__init__` is lazy (PEP 562) so importing it does not drag in mcp, pydantic, httpx2,
starlette and uvicorn. There is a test that fails if this regresses.

Both libraries can write one SQLite file (`agent_mcp_usage` / `agent_runtime_usage`),
sharing the join columns `conversation_id`, `app_name`, `depth`, and `created_at`
(ISO-8601 UTC seconds, never epoch floats). WAL plus a busy timeout makes the
co-tenancy safe.

Pin `mcp>=2.0,<3`. v2 renamed `FastMCP` to `MCPServer` and this package targets that
API exclusively — it will not import under 1.x.

## Local development

```bash
git clone https://github.com/Johonnyy/agent-mcp-py
cd agent-mcp-py

python -m venv .venv
.venv/Scripts/activate            # Windows;  .venv/bin/activate on Linux/macOS
pip install -e ".[dev]"
cp .env.example .env

pytest                            # whole suite, no network required
pytest tests/test_decorators.py   # the signature surgery
pytest tests/test_end_to_end.py   # a real uvicorn server over HTTP
```

The suite needs no API key and makes no outbound calls: the sync store is a fake,
auth uses known-good/known-bad tokens, and the end-to-end file binds an ephemeral
local port.
