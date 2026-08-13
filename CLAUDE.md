# CLAUDE.md — agent-mcp-py

The ecosystem-wide context lives in the `amber` repo's CLAUDE.md; this file covers
**this** repo. Read the "Conventions every app must follow" section there before
changing anything here, because this package is what enforces those conventions.

## What this is

`agent-mcp-py` is the convention layer every Python app uses to expose itself as an
MCP server. It wraps the official `mcp` SDK's server so that an app author writing
`@mcp.tool(read_only=True)` gets bearer auth, the conversation-depth guard, usage
logging, and sync-store registration without naming any of them.

It is **item #1 in the ecosystem build order** — Amber's own MCP server,
`finance-agent`, the FreeCallMe sidecar, and every future app are built on it. That
makes backwards compatibility matter more here than anywhere else in the ecosystem.

Consumed as a pinned git dependency, never from PyPI:
`agent-mcp-py = { git = "...", tag = "v0.1.0" }`.

## Core principle

The app is the product; this library is invisible plumbing. It must never force a
choice on the host app — no FastAPI dependency (Starlette routes only), no LLM SDK,
no model names, no required database. Every feature degrades to a warning rather
than a failure when its backing service is absent, because **an app must run
standalone with zero knowledge the ecosystem exists.**

## Architecture

Two layers, and the split is load-bearing:

**Leaf modules** (`depth.py`, `auth.py`, `schema.py`) are pure functions with no
third-party imports. `agent_runtime` imports `agent_mcp.depth` to pre-check the
depth cap client-side, and must not have to install a server stack to read an
integer. This is why `__init__.py` is lazy (PEP 562) — importing a submodule runs
its parent package first, so an eager `__init__` would drag mcp, pydantic, httpx2,
starlette and uvicorn into that import. `tests/test_depth.py` fails if it regresses.

**Composed modules** (`decorators.py`, `middleware.py`, `server.py`) import the SDK
and wire the leaves together. Import direction is strictly one-way: `errors → depth`,
`sync_client → registry`, never back.

### The guard path

`decorators.py` is the crux; everything else is plumbing. The SDK only hands request
headers to a handler that declares a `Context` parameter, so we wrap the author's
function in one that declares it, strip it off before calling them, and keep the
advertised JSON schema as *their* signature. Three pitfalls, all silent if they
regress — the tool keeps working while the guards quietly stop running:

1. `functools.wraps` sets `__wrapped__` and `inspect.signature` follows it, so
   `__signature__` must be assigned **after** `wraps`.
2. `__globals__` cannot be copied, so annotations are pre-resolved to **real type
   objects** rather than left as strings for the SDK to `eval` in the wrong
   namespace.
3. `find_context_parameter` swallows every exception and returns `None` — it fails
   *open*. The registration-time self-check turns that into a loud `RuntimeError`.
   **Never remove it.**

### Enforcement placement

Auth lives in *both* the ASGI gate and the tool wrapper, and that is not redundancy.
The gate is the only thing protecting static resources, `resources/read` and
`tools/list`; the wrapper is the only thing covering stdio, a raw
`streamable_http_app()` mount, and in-process `Client(server.mcp)`. Depth is checked
in the wrapper (where it can produce a well-formed MCP error) with the gate as a
backstop. The confirmation gate is wrapper-only — the middleware would have to parse
the JSON-RPC body to learn the tool name.

ContextVars set in ASGI middleware do **not** reach a tool handler: streamable HTTP
routes messages through anyio memory object streams into a separate session task.
Set them inside the wrapper instead, which is what `current_call()` does.

## Commands

Stack: Python 3.11+, `src/` layout, setuptools, pytest with `asyncio_mode = "auto"`.

```bash
python -m venv .venv
.venv/Scripts/activate            # Windows;  .venv/bin/activate on Linux/macOS
pip install -e ".[dev]"

pytest                            # 175 tests, no network
pytest tests/test_decorators.py   # the signature surgery — highest value
pytest tests/test_end_to_end.py   # a real uvicorn server over streamable HTTP
```

## Conventions

Matches the `amber` repo, which is the reference for style:

- `from __future__ import annotations` in every module; long module docstrings that
  explain the design rule and *why*, including the tradeoff.
- pydantic-settings with `env_prefix="AGENT_MCP_"`, `# --- Section ---` banners,
  `@lru_cache get_settings()`, and runtime code calling `get_settings()` **inside**
  functions so tests can clear the cache.
- Raw `sqlite3`, no ORM: a module-level `_SCHEMA` of `CREATE TABLE IF NOT EXISTS`
  applied with `executescript`, one shared connection with `check_same_thread=False`
  under a `threading.Lock`, rows out as `[dict(r) for r in rows]`, timestamps as
  ISO-8601 UTC seconds TEXT.
- Sync I/O wrapped at the call site in `await asyncio.to_thread(...)`, never inside
  the sync layer.
- `asyncio.CancelledError` re-raised **before** any broad `except Exception`, which
  carries `# noqa: BLE001` and a reason.
- stdlib `logging`, `logger = logging.getLogger(__name__)`, %-style lazy formatting,
  bracketed `[%s]` context prefix. **Never log a token** — log the caller name.
- Tests: no `conftest.py` (fixtures are local and duplicated on purpose),
  `monkeypatch` only, hand-written `_FakeAsyncClient`/`_FakeResponse` with a
  `capture` dict, settings built via a local `_settings(**over)` helper using
  `_env_file=None`, SQLite isolation via `":memory:"`, full-sentence test names.

## Interop contract with `agent-runtime`

Agreed and encoded in tests — changing any of it is a breaking change:

- `agent_mcp.depth` exports `HEADER_CONVERSATION_ID`, `HEADER_AGENT_DEPTH`,
  `HEADER_CONFIRMED`, `MAX_AGENT_DEPTH`, `DepthExceeded`, and stays stdlib-only.
- `agent_mcp.registry.resolve(name)` is **synchronous and does no I/O**; records
  carry at minimum `{name, base_url, token}`.
- Tool names match `^[a-z][a-z0-9_]{0,39}$` with no `__` (the `<server>__<tool>`
  namespace separator).
- Input schemas are plain JSON Schema, `type: "object"` at the root, no `$ref`,
  `$defs`, or root-level composition.
- `read_only` / `requires_confirmation` are on the wire (`readOnlyHint` plus both in
  `_meta`), so `list_tools()` alone tells a caller what is safe.
- Tool errors surface as `isError: true` with text, never HTTP 5xx.
- Streamable HTTP at a fixed `/mcp`, so the registry stores base URLs only.
- Both libraries may share one SQLite file: tables `agent_mcp_usage` /
  `agent_runtime_usage`, join columns `conversation_id` / `app_name` / `depth` /
  `created_at`, WAL plus a busy timeout.

## SDK notes (verified against mcp 2.0.0)

Things that are true now and were not obvious:

- `mcp.server.fastmcp` **no longer exists**; it is `mcp.server.mcpserver.MCPServer`.
  Pin `mcp>=2.0,<3`; the package cannot import under 1.x.
- `ToolAnnotations` ignores unknown fields **silently** — a custom
  `requiresConfirmation` there vanishes with no error. It has to ride in `_meta`.
- `tool.parameters` is a plain mutable dict and mutating it *is* reflected in
  `list_tools()`. That is the hook for schema flattening, and the only place the
  package touches `_tool_manager`.
- A **static** resource URI may take no handler parameters and cannot receive a
  `Context` at all. Only `{...}` templates can. Static resources therefore cannot be
  identity-logged — steer authors to `{?param}` templates.
- `session_manager` raises `RuntimeError` until `streamable_http_app()` has run, so
  `lifespan()` builds the app first, and `asgi_app()` must stay idempotent.
- `Mount("/mcp")` does not match a bare `/mcp`; the host router 307s, and the real
  MCP client does **not** follow it. Hence `routes()`.
- Raising `MCPError` from a tool is a top-level JSON-RPC error in v2 (it was
  `isError` in v1). Ordinary exceptions become `isError: true`.

## Current state

- **v0.1.0**: all seven specced modules plus `schema.py`, `errors.py` and
  `middleware.py`. 175 tests green, verified end to end against a live server.
- **Known gaps**, each recorded in code where it bites:
  - Output schemas are a `cached_property` and not reliably mutable, so a `$ref` in
    one is a warning rather than a rejection. `structured_output=False` is the
    escape hatch.
  - Static resources cannot be identity-logged (structural, not fixable here).
  - `X-Confirmed` has no source in Amber yet: `app/protocol.py` has 7 frame types
    and no tool-event frame, so there is no path for a human to approve anything.
    The flag is enforced and advertised regardless — a tool that is unreachable
    without approval is the right default. Adding that frame is Amber's work.
  - The sync store does not exist yet, so `AGENT_MCP_PEERS` is the real discovery
    mechanism. `register()`/`refresh()` are written against the intended shape and
    tested with a fake.

## Next

1. Build the hosted sync store, then check `sync_client`'s assumed HTTP shape
   (`POST /servers`, `GET /servers`) against it.
2. `agent-runtime`, then `agent-spawner`.
3. Amber's refactor: remove OpenClaw, swap the brain onto `agent-runtime`, and add
   her own MCP server on this package — the first real consumer.
