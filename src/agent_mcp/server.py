"""``AgentMCPServer`` — the one class an app imports.

Wraps the SDK's ``MCPServer`` so that registering a tool also registers its policy,
wraps it in the guard path, flattens its JSON schema, and advertises its flags. The
app author sees only::

    mcp = AgentMCPServer(app_name="finance", version="0.1.0")

    @mcp.tool(read_only=True)
    def get_balance(account: str) -> float:
        return db.get_balance(account)

    app.mount("/mcp", mcp.asgi_app())

Two ordering constraints are baked in because both fail in ways that are invisible
until the first request:

* ``session_manager`` raises ``RuntimeError`` until ``streamable_http_app()`` has
  been called (verified). :meth:`lifespan` therefore builds the app first.
* :meth:`asgi_app` **must be idempotent**. Calling ``streamable_http_app()`` twice
  creates two session managers, and the one the lifespan runs would not be the one
  serving traffic.

Deliberately not used: the SDK's ``TokenVerifier``/``AuthSettings`` (OAuth-shaped,
and this library's entire auth need is a string comparison) and v2's
``middleware=[ServerMiddleware]`` (the SDK docs call that surface provisional —
founding a security model on it means rewriting at the next minor release).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from contextlib import asynccontextmanager
from typing import Any, TypeVar

import httpx2  # not httpx: mcp v2 ships httpx2, a separate distribution
from mcp import Client
from mcp.client.streamable_http import streamable_http_client
from mcp.server.mcpserver import Context, MCPServer
from mcp.types import ToolAnnotations
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Mount, Route

from agent_mcp import auth as auth_mod
from agent_mcp import depth as depth_mod
from agent_mcp import registry as registry_mod
from agent_mcp import schema as schema_mod
from agent_mcp import sync_client
from agent_mcp.config import AgentMCPSettings, get_settings
from agent_mcp.decorators import (
    CallScope,
    Guard,
    ResourcePolicy,
    ToolPolicy,
    build_resource_wrapper,
    build_static_resource_wrapper,
    build_tool_wrapper,
    confirmation_notice,
    current_call,
    tool_meta,
)
from agent_mcp.errors import PeerCallRefused
from agent_mcp.middleware import BearerGate
from agent_mcp.registry import PeerRecord, PeerRegistry
from agent_mcp.usage_log import NullUsageSink, SQLiteUsageSink, UsageSink

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])

#: URI template variables look like {var} or {?query,vars}.
_TEMPLATE_MARKER = "{"


class AgentMCPServer:
    """An MCP server with the ecosystem's conventions already wired in."""

    def __init__(
        self,
        *,
        app_name: str,
        version: str = "0.1.0",
        sync_store_url: str | None = None,
        auth_keys_env: str | None = None,
        auth_keys: Sequence[str] | Mapping[str, str] | str | None = None,
        public_url: str | None = None,
        usage_sink: UsageSink | None = None,
        peers: Mapping[str, PeerRecord] | None = None,
        instructions: str | None = None,
        settings: AgentMCPSettings | None = None,
    ) -> None:
        cfg = settings or get_settings()
        self.settings = cfg

        self.app_name = app_name or cfg.app_name
        schema_mod.validate_tool_name(self.app_name)
        self.version = version or cfg.version
        self.public_url = (
            public_url if public_url is not None else cfg.public_url
        ).rstrip("/")
        self.sync_store_url = (
            sync_store_url if sync_store_url is not None else cfg.sync_store_url
        )
        self.max_agent_depth = cfg.max_agent_depth

        self._keys = self._resolve_keys(auth_keys, auth_keys_env, cfg)
        self.allow_anonymous = cfg.allow_anonymous

        if usage_sink is not None:
            self._sink: UsageSink = usage_sink
        elif cfg.usage_enabled:
            self._sink = SQLiteUsageSink(cfg.usage_db_path)
        else:
            self._sink = NullUsageSink()

        self._registry = PeerRegistry(
            static={
                **registry_mod.load_static_peers(cfg.peers, cfg.peer_token),
                **(peers or {}),
            }
        )
        registry_mod.default_registry().set_static(dict(self._registry._static))

        self._guard = Guard(
            app_name=self.app_name,
            keys=self._keys,
            allow_anonymous=self.allow_anonymous,
            max_depth=self.max_agent_depth,
            sink=self._sink,
            usage_enabled=cfg.usage_enabled,
        )

        self._mcp = MCPServer(
            name=self.app_name, version=self.version, instructions=instructions
        )
        self._tools: dict[str, ToolPolicy] = {}
        self._resources: dict[str, ResourcePolicy] = {}
        self._descriptions: dict[str, str] = {}
        self._asgi_app: Any = None

    # --- construction helpers ---

    @staticmethod
    def _resolve_keys(
        auth_keys: Sequence[str] | Mapping[str, str] | str | None,
        auth_keys_env: str | None,
        cfg: AgentMCPSettings,
    ) -> dict[str, str]:
        """Bearer keys from (in order) the constructor, a named env var, settings."""
        if isinstance(auth_keys, Mapping):
            return dict(auth_keys)
        if isinstance(auth_keys, str):
            return auth_mod.parse_keys(auth_keys)
        if auth_keys is not None:
            return auth_mod.parse_keys(",".join(auth_keys))
        if auth_keys_env:
            import os

            raw = os.environ.get(auth_keys_env, "")
            if raw.strip():
                return auth_mod.parse_keys(raw)
        return auth_mod.parse_keys(cfg.resolved_keys())

    # --- registration ---

    def tool(
        self,
        *,
        read_only: bool = False,
        requires_confirmation: bool = False,
        name: str | None = None,
        title: str | None = None,
        description: str | None = None,
        destructive: bool | None = None,
        idempotent: bool | None = None,
        open_world: bool | None = None,
        structured_output: bool | None = None,
        meta: dict[str, Any] | None = None,
    ) -> Callable[[F], F]:
        """Register a tool. Returns the function **unwrapped**.

        Returning the original means the app's own HTTP handler can call the very
        same object the model calls — the ecosystem rule that a tool mirrors a real
        user action rather than a parallel code path that drifts from it.
        """

        def decorator(fn: F) -> F:
            tool_name = name or fn.__name__
            schema_mod.validate_tool_name(tool_name)
            if tool_name in self._tools:
                raise ValueError(f"Tool already registered: {tool_name!r}")

            policy = ToolPolicy(
                name=tool_name,
                read_only=read_only,
                requires_confirmation=requires_confirmation,
            )
            text = confirmation_notice(
                description or (fn.__doc__ or "").strip(), policy
            )
            wrapper = build_tool_wrapper(fn, policy=policy, guard=self._guard)

            annotations = ToolAnnotations(
                title=title,
                read_only_hint=read_only,
                # The spec treats these as meaningless when readOnlyHint is set, so
                # leave them unset rather than asserting something uninformative.
                destructive_hint=None if read_only else destructive,
                idempotent_hint=None if read_only else idempotent,
                open_world_hint=open_world,
            )

            registered = self._mcp._tool_manager.add_tool(
                wrapper,
                name=tool_name,
                title=title,
                description=text,
                annotations=annotations,
                meta={**tool_meta(policy, self.app_name), **(meta or {})},
                structured_output=structured_output,
            )
            self._finalise_schema(registered, tool_name)

            self._tools[tool_name] = policy
            self._descriptions[tool_name] = text
            return fn

        return decorator

    def _finalise_schema(self, registered: Any, tool_name: str) -> None:
        """Inline ``$ref``s and assert the result is wire-safe.

        The only place in the package that touches ``_tool_manager``'s output. It is
        private API, but the alternative is reimplementing ``func_metadata``, and
        confining it to one call site makes an SDK rename a one-line fix. Verified:
        ``parameters`` is a plain mutable dict and mutating it is reflected in
        ``list_tools()``.
        """
        params = getattr(registered, "parameters", None)
        if not isinstance(params, dict):  # pragma: no cover — SDK shape changed
            logger.warning(
                "Tool %s: cannot reach the input schema to flatten it; a nested "
                "model would ship a $ref that some providers reject.",
                tool_name,
            )
            return
        try:
            flattened = schema_mod.flatten_schema(params)
            schema_mod.assert_wire_safe(flattened, what=f"input schema for {tool_name!r}")
        except ValueError:
            # Undo the registration so a bad tool cannot be half-present.
            with contextlib.suppress(Exception):
                self._mcp._tool_manager._tools.pop(tool_name, None)
            raise
        registered.parameters = flattened

        output = getattr(registered, "output_schema", None)
        if isinstance(output, dict) and schema_mod.has_refs(output):
            # output_schema is a cached_property and not reliably mutable, so this
            # is a warning rather than a rejection. Pass structured_output=False to
            # drop it entirely if a provider objects.
            logger.warning(
                "Tool %s: output schema contains $ref/$defs, which some providers "
                "reject. Pass structured_output=False if that becomes a problem.",
                tool_name,
            )

    def resource(
        self,
        uri: str,
        *,
        name: str | None = None,
        title: str | None = None,
        description: str | None = None,
        mime_type: str | None = None,
        meta: dict[str, Any] | None = None,
    ) -> Callable[[F], F]:
        """Register a resource. Returns the function unwrapped.

        A resource URI should mirror a real dashboard view: if a screen renders it,
        there is a resource returning the same data, not an agent-only variant.
        """

        def decorator(fn: F) -> F:
            res_name = name or fn.__name__
            if uri in self._resources:
                raise ValueError(f"Resource already registered: {uri!r}")

            templated = _TEMPLATE_MARKER in uri
            import inspect as _inspect

            has_params = bool(_inspect.signature(fn).parameters)
            if not templated and has_params:
                raise ValueError(
                    f"Resource {uri!r} has a static URI but {res_name!r} declares "
                    "parameters, which the MCP SDK rejects. Make the URI a template "
                    f"instead — for a query parameter use "
                    f"{uri}{{?{','.join(_inspect.signature(fn).parameters)}}} — or "
                    "remove the parameters."
                )

            policy = ResourcePolicy(uri=uri, name=res_name, templated=templated)
            if templated:
                wrapper = build_resource_wrapper(fn, policy=policy, guard=self._guard)
            else:
                # Static URIs cannot receive a Context, so the caller and
                # conversation id are unknowable here; the ASGI gate is what
                # authenticates these reads.
                wrapper = build_static_resource_wrapper(
                    fn, policy=policy, guard=self._guard
                )

            self._mcp.resource(
                uri,
                name=res_name,
                title=title,
                description=description or (fn.__doc__ or "").strip(),
                mime_type=mime_type,
                meta=meta,
            )(wrapper)

            self._resources[uri] = policy
            return fn

        return decorator

    # --- serving ---

    @property
    def mcp(self) -> MCPServer:
        """The underlying SDK server — escape hatch for anything not wrapped here."""
        return self._mcp

    def tool_policies(self) -> list[ToolPolicy]:
        return list(self._tools.values())

    def resource_policies(self) -> list[ResourcePolicy]:
        return list(self._resources.values())

    def asgi_app(self, *, path: str = "/") -> Any:
        """The mountable ASGI app, gated. Idempotent — the same object every time.

        ``streamable_http_path="/"`` so the mount prefix *is* the public path, plus
        the gate's empty-path rewrite, so both ``/mcp`` and ``/mcp/`` work without a
        307 that a non-redirect-following client would silently fail on.
        """
        if self._asgi_app is not None:
            return self._asgi_app

        if not self._keys and not self.allow_anonymous:
            raise RuntimeError(
                f"[{self.app_name}] Refusing to serve: no bearer keys are "
                "configured. Set AGENT_MCP_KEYS (or the env var named by "
                "auth_keys_env), pass auth_keys=..., or set "
                "AGENT_MCP_ALLOW_ANONYMOUS=true for local development."
            )
        if self.allow_anonymous:
            logger.warning(
                "[%s] Anonymous access is ENABLED — every MCP call is admitted "
                "without a bearer token. Never do this in production.",
                self.app_name,
            )

        inner = self._mcp.streamable_http_app(streamable_http_path=path)
        self._asgi_app = BearerGate(
            inner,
            keys=self._keys,
            allow_anonymous=self.allow_anonymous,
            max_depth=self.max_agent_depth,
            strip_prefix=registry_mod.MCP_MOUNT_PATH,
        )
        return self._asgi_app

    def routes(self, path: str = registry_mod.MCP_MOUNT_PATH) -> list[Any]:
        """Routes serving MCP at ``path``, with **and** without a trailing slash.

        Prefer this over ``app.mount(...)``. ``Mount("/mcp")`` only matches
        ``/mcp/...``, so a bare ``POST /mcp`` falls through to the host router's
        ``redirect_slashes`` and gets a 307 — which the real MCP client does not
        follow (verified: it fails with "Unexpected content type"). Since the
        redirect is emitted by the host's router before any middleware of ours runs,
        the only fix is to claim the exact path too::

            app.router.routes.extend(mcp.routes())
            app.router.routes.extend(mcp.usage_routes())

        An app that would rather mount by hand can still do
        ``app.mount("/mcp", mcp.asgi_app())`` — it just has to make sure every
        caller uses the trailing slash. :func:`agent_mcp.registry.mcp_url` always
        produces it.
        """
        gate = self.asgi_app()
        clean = "/" + path.strip("/")
        return [
            Mount(clean, app=gate),
            # Exact-path route for the bare form; the gate rewrites it to "/".
            Route(
                clean,
                endpoint=gate,
                methods=["GET", "POST", "DELETE", "OPTIONS"],
            ),
        ]

    def usage_routes(self, path: str = "/agent/usage") -> list[Route]:
        """Starlette routes for the usage summary the spawner reads.

        Starlette rather than a FastAPI ``APIRouter``: starlette is already a
        transitive dependency of ``mcp`` and FastAPI is not, and this library must
        not force a web framework on a consumer. FastAPI accepts these directly via
        ``app.router.routes.extend(...)``.
        """

        async def handler(request: Request) -> JSONResponse:
            # Authenticated by calling the same plain function the gate uses — the
            # payoff of keeping auth a function rather than middleware.
            result = auth_mod.verify_bearer(
                request.headers.get("authorization"),
                self._keys,
                allow_anonymous=self.allow_anonymous,
            )
            if not result.ok:
                return JSONResponse(
                    {"error": "unauthorized", "message": result.reason},
                    status_code=401,
                    headers={"WWW-Authenticate": 'Bearer realm="agent-mcp"'},
                )
            since = request.query_params.get("since")
            summary = await asyncio.to_thread(self._sink.summary, since=since)
            return JSONResponse({"app": self.app_name, **summary})

        return [Route(path, handler, methods=["GET"])]

    @asynccontextmanager
    async def lifespan(self) -> AsyncIterator[None]:
        """Run the session manager and the background sync task.

        A mounted sub-app's own lifespan never runs, so the host app must enter
        ``session_manager.run()`` or the very first request fails. Wrap the host's
        lifespan around this one::

            @asynccontextmanager
            async def lifespan(app):
                async with mcp.lifespan():
                    yield
        """
        self.asgi_app()  # idempotent; also guarantees session_manager exists
        async with self._mcp.session_manager.run():
            task = asyncio.create_task(self._startup_sync())
            try:
                yield
            finally:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
                with contextlib.suppress(Exception):
                    self._sink.close()

    async def _startup_sync(self) -> None:
        """Register once, then heartbeat. Never awaited on the startup path."""
        cfg = self.settings

        async def do_register() -> bool:
            return await sync_client.register(
                sync_store_url=self.sync_store_url,
                descriptor=sync_client.build_descriptor(
                    app_name=self.app_name,
                    version=self.version,
                    base_url=self.public_url,
                    tools=self.tool_policies(),
                    resources=self.resource_policies(),
                    descriptions=self._descriptions,
                ),
                token=cfg.sync_store_token,
                timeout_s=cfg.sync_timeout_s,
            )

        await do_register()
        if self.sync_store_url:
            await self._registry.refresh(
                self.sync_store_url,
                token=cfg.sync_store_token,
                timeout_s=cfg.sync_timeout_s,
            )
            registry_mod.default_registry()._discovered = dict(
                self._registry._discovered
            )

        def prune() -> int:
            return self._sink.prune(cfg.usage_retention_days)

        await sync_client.heartbeat_loop(
            interval_s=cfg.heartbeat_interval_s,
            register_fn=do_register,
            on_tick=lambda: asyncio.to_thread(prune),
        )

    # --- outbound ---

    async def call_peer(
        self,
        server_name: str,
        tool_name: str,
        *,
        ctx: Context | None = None,
        confirmed: bool = False,
        timeout_s: float | None = None,
        **kwargs: Any,
    ) -> str:
        """Call a tool on another MCP server, threading the depth headers.

        The depth cap is checked **locally** before the request is built, so an
        over-limit hop never leaves this process. Failures here are ordinary
        exceptions, not ``MCPError``: this runs inside a tool body, where a bad peer
        call should be reported back to the model rather than taking down the turn.
        """
        scope = self._scope_for_outbound(ctx)
        record = registry_mod.resolve(server_name) or self._registry.resolve(
            server_name
        )
        if record is None:
            known = ", ".join(self._registry.known()) or "none configured"
            raise PeerCallRefused(
                f"Unknown peer {server_name!r}. Known peers: {known}."
            )

        try:
            headers = depth_mod.next_hop_headers(
                scope, limit=self.max_agent_depth, confirmed=confirmed
            )
        except depth_mod.DepthExceeded as exc:
            raise PeerCallRefused(
                f"Cannot call {server_name}.{tool_name}: {exc}. Answer with what you "
                "already have."
            ) from None

        token = record.token or self.settings.peer_token
        if token:
            headers["Authorization"] = f"Bearer {token}"

        async with httpx2.AsyncClient(
            headers=headers, timeout=timeout_s or self.settings.peer_timeout_s
        ) as http_client:
            transport = streamable_http_client(
                registry_mod.mcp_url(record), http_client=http_client
            )
            async with Client(transport) as client:
                result = await client.call_tool(tool_name, kwargs)
        return _result_text(result)

    def _scope_for_outbound(self, ctx: Context | None) -> CallScope:
        """Depth state for an outgoing hop: explicit ctx, then ambient, then fresh."""
        if ctx is not None:
            parsed = depth_mod.parse_depth(
                getattr(ctx, "headers", None), limit=self.max_agent_depth
            )
            return CallScope(
                conversation_id=parsed.conversation_id,
                depth=parsed.depth,
                caller=auth_mod.CALLER_LOCAL,
                confirmed=parsed.confirmed,
            )
        ambient = current_call()
        if ambient is not None:
            return ambient
        fresh = depth_mod.parse_depth(None)
        return CallScope(
            conversation_id=fresh.conversation_id,
            depth=0,
            caller=auth_mod.CALLER_LOCAL,
        )


def _result_text(result: Any) -> str:
    """Flatten a ``CallToolResult`` into the text an LLM reads back."""
    parts: list[str] = []
    for block in getattr(result, "content", None) or []:
        text = getattr(block, "text", None)
        if text:
            parts.append(text)
    joined = "\n".join(parts)
    if getattr(result, "is_error", False):
        return f"Error from peer: {joined}" if joined else "Error from peer."
    return joined
