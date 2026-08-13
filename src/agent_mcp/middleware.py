"""The ASGI gate — authentication at the transport edge, plus a mount-path fix.

Two jobs, both of which have to happen before a request becomes an MCP message.

**Authentication.** The per-tool wrapper in `decorators.py` also checks the bearer
token, so why here as well? Because the wrapper only sees *tool calls*. Static
resource reads, ``resources/read``, ``tools/list`` and ``initialize`` all reach the
server without touching a wrapped function, and a static resource cannot receive a
``Context`` at all (the SDK refuses). Without this gate an unauthenticated caller
could enumerate every tool and read every static resource. Rejecting at the HTTP
layer also means such a caller never allocates a session.

The two checks are not redundant in the other direction either: this middleware only
exists if the app mounts :meth:`AgentMCPServer.asgi_app`. An app running stdio,
mounting the raw ``streamable_http_app()``, or driving an in-process
``Client(server.mcp)`` bypasses it entirely — and the wrapper still holds.

**The mount-path fix.** ``Mount("/mcp")`` does not match a bare ``/mcp`` — only
``/mcp/...`` — so the *host's* router answers ``/mcp`` with a **307 to /mcp/**.
Verified against a live server, and verified to matter: the real MCP client does
**not** follow that redirect (it fails with "Unexpected content type"), so an app
whose peers address it without the trailing slash is simply broken.

That redirect happens in the host router before this middleware is reached, so it
cannot be fixed from in here alone — :meth:`AgentMCPServer.routes` registers an
extra exact-path route for the bare form, and this gate normalises the path the
child sees. ``strip_prefix`` is how the two halves agree: the child app is a
``streamable_http_app`` with a single endpoint at ``/``, so a request that arrived
at the mount path itself must be rewritten to ``/`` before routing sees it.

Written as pure ASGI rather than Starlette's ``BaseHTTPMiddleware`` on purpose:
``BaseHTTPMiddleware`` proxies the response body through a task group, which is
known to break long-lived SSE streams — and the streamable-HTTP GET channel is
exactly that.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from agent_mcp import auth as auth_mod
from agent_mcp import depth as depth_mod

logger = logging.getLogger(__name__)

Scope = dict[str, Any]
Receive = Callable[[], Awaitable[Mapping[str, Any]]]
Send = Callable[[Mapping[str, Any]], Awaitable[None]]


def headers_from_scope(scope: Scope) -> dict[str, str]:
    """ASGI's list of lowercased byte pairs as a plain string dict."""
    out: dict[str, str] = {}
    for raw_key, raw_value in scope.get("headers") or []:
        try:
            out[raw_key.decode("latin-1").lower()] = raw_value.decode("latin-1")
        except Exception:  # noqa: BLE001 — a malformed header is not fatal
            continue
    return out


async def _reject(send: Send, status: int, code: str, message: str) -> None:
    body = json.dumps({"error": code, "message": message}).encode("utf-8")
    headers = [
        (b"content-type", b"application/json"),
        (b"content-length", str(len(body)).encode("ascii")),
    ]
    if status == 401:
        headers.append((b"www-authenticate", b'Bearer realm="agent-mcp"'))
    await send(
        {"type": "http.response.start", "status": status, "headers": headers}
    )
    await send({"type": "http.response.body", "body": body})


class BearerGate:
    """Authenticates every HTTP request and backstops the depth cap."""

    def __init__(
        self,
        app: Any,
        *,
        keys: Mapping[str, str],
        allow_anonymous: bool = False,
        max_depth: int = depth_mod.MAX_AGENT_DEPTH,
        fix_mount_path: bool = True,
        strip_prefix: str = "",
    ) -> None:
        self.app = app
        self.keys = dict(keys)
        self.allow_anonymous = allow_anonymous
        self.max_depth = max_depth
        self.fix_mount_path = fix_mount_path
        self.strip_prefix = strip_prefix.rstrip("/")

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope.get("type") != "http":
            # Lifespan and websocket traffic pass straight through.
            await self.app(scope, receive, send)
            return

        if self.fix_mount_path:
            # Normalise to the child's single endpoint. "" is what Mount leaves
            # behind; strip_prefix is the exact-path route registered for the bare
            # form. See the module docstring on the 307.
            path = scope.get("path", "")
            if path in ("", "/") or (self.strip_prefix and path == self.strip_prefix):
                scope = {**scope, "path": "/", "raw_path": b"/"}

        headers = headers_from_scope(scope)

        result = auth_mod.verify_bearer(
            headers.get("authorization"),
            self.keys,
            allow_anonymous=self.allow_anonymous,
        )
        if not result.ok:
            logger.warning(
                "Rejected MCP request to %s: %s", scope.get("path", "?"), result.reason
            )
            await _reject(send, 401, "unauthorized", result.reason)
            return

        try:
            parsed = depth_mod.parse_depth(headers, limit=self.max_depth)
            depth_mod.check_depth(
                parsed.depth,
                limit=self.max_depth,
                conversation_id=parsed.conversation_id,
            )
        except ValueError as exc:
            await _reject(send, 400, "bad_request", str(exc))
            return
        except depth_mod.DepthExceeded as exc:
            logger.warning("[%s] %s", exc.conversation_id, exc)
            await _reject(send, 403, "depth_exceeded", str(exc))
            return

        scope = {**scope, "agent_mcp.caller": result.caller}
        await self.app(scope, receive, send)
