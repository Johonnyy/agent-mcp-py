"""The wrapper that turns a plain function into a guarded MCP tool.

This is the crux of the library; everything else is plumbing around it. An app
author writes::

    @mcp.tool(read_only=True)
    def get_balance(account: str) -> float: ...

and gets auth, the depth cap, the confirmation gate, and usage logging without
naming any of them. To do that we must see the request's headers, which the SDK
delivers only to a handler that declares a ``Context`` parameter — so we wrap the
author's function in one that declares it, strip it back off before calling them,
and make certain the JSON schema the model sees is still *their* signature.

The mechanics are fiddly and three of the pitfalls are silent, so they are spelled
out here rather than discovered later.

**1. ``functools.wraps`` sets ``__wrapped__``, and ``inspect.signature`` follows it.**
Left alone, the SDK would introspect the *original* function, see no ``Context``,
inject nothing, and our ``kwargs.pop`` would yield ``None`` — every guard silently
disabled while the tool still works. Assigning ``__signature__`` stops the unwrap
(``inspect.unwrap`` halts at an object that has one), and it must be assigned
*after* ``wraps``, which copies ``fn.__dict__`` over the top.

**2. ``__globals__`` cannot be copied.** The SDK resolves annotations with
``inspect.signature(fn, eval_str=True)`` and ``typing.get_type_hints(fn)``, both of
which evaluate *string* annotations against the function's own globals. Our wrapper's
globals are this module's, so a tool annotated ``-> Invoice`` where ``Invoice`` lives
in the author's module would fail to resolve. We therefore resolve the hints once
against the author's namespace and store **real type objects**, which skips
evaluation entirely.

**3. ``find_context_parameter`` swallows every exception and returns ``None``.**
Confirmed in the installed SDK source. It fails *open*: if annotation resolution
breaks for any reason, injection silently stops and the guards stop with it. The
registration-time self-check at the end of :func:`build_tool_wrapper` converts that
into a loud ``RuntimeError`` at import. **Do not remove it.**

The wrapper is always ``async``, even for a sync tool. The guard path and the usage
write are async, one code path is better than two, and it means a blocking tool runs
off the event loop for free. The tradeoff: a sync tool now runs in a worker thread
and must be thread-safe.

Bound methods and ``functools.partial`` objects are not supported — ``get_type_hints``
does not handle them predictably. Decorate module-level functions.
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import logging
import time
import typing
from collections.abc import Callable, Mapping
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

from mcp.server.mcpserver import Context
from mcp.server.mcpserver.utilities.context_injection import find_context_parameter

from agent_mcp import auth as auth_mod
from agent_mcp import depth as depth_mod
from agent_mcp.errors import ConfirmationRequired, bad_headers_error, depth_error
from agent_mcp.usage_log import UsageEvent

logger = logging.getLogger(__name__)

#: The parameter we inject. Never name it with a leading underscore: the SDK's
#: ``func_metadata`` rejects parameters starting with ``_``.
CTX_PARAM = "agent_mcp_ctx"

#: Metadata namespace for our flags on the wire. The label rules for ``_meta`` keys
#: require prefix labels that start and end alphanumeric, and reserve ``mcp`` and
#: ``modelcontextprotocol`` — ``dev.johnny.agent-mcp`` satisfies both.
META_PREFIX = "dev.johnny.agent-mcp"
META_REQUIRES_CONFIRMATION = f"{META_PREFIX}/requiresConfirmation"
META_READ_ONLY = f"{META_PREFIX}/readOnly"
META_APP = f"{META_PREFIX}/app"


@dataclass(frozen=True)
class ToolPolicy:
    """What the server enforces for one tool. The authoritative copy.

    Enforcement reads *this*, never the wire metadata. ``_meta`` is advertisement for
    callers; a policy check that round-tripped through client-controllable data
    would be a way to turn the confirmation gate off by editing a request.
    """

    name: str
    read_only: bool = False
    requires_confirmation: bool = False


@dataclass(frozen=True)
class ResourcePolicy:
    uri: str
    name: str
    templated: bool


@dataclass(frozen=True)
class CallScope:
    """Ambient state for the call currently executing on this task."""

    conversation_id: str
    depth: int
    caller: str
    confirmed: bool = False


#: Set *inside* the wrapper, immediately before the author's function runs.
#:
#: This direction is safe, and the opposite one is not. A ContextVar set in an outer
#: ASGI middleware does **not** reach the handler, because streamable HTTP routes
#: JSON-RPC messages through anyio memory object streams into a separate session
#: task. Setting it here means the same task for an async tool, and
#: ``asyncio.to_thread`` copies the context for a sync one — so ``call_peer`` can
#: pick up the conversation id without every tool having to thread it by hand.
_current_call: ContextVar[CallScope | None] = ContextVar(
    "agent_mcp_current_call", default=None
)


def current_call() -> CallScope | None:
    """The call scope for the tool running on this task, if any."""
    return _current_call.get()


class Guard:
    """The per-call checks, shared by tool and resource wrappers.

    Constructed once by the server and handed to every wrapper, so the whole policy
    surface is one object to fake in a test.
    """

    def __init__(
        self,
        *,
        app_name: str,
        keys: Mapping[str, str],
        allow_anonymous: bool,
        max_depth: int,
        sink: Any,
        usage_enabled: bool = True,
    ) -> None:
        self.app_name = app_name
        self.keys = dict(keys)
        self.allow_anonymous = allow_anonymous
        self.max_depth = max_depth
        self.sink = sink
        self.usage_enabled = usage_enabled

    def authenticate(self, headers: Mapping[str, str] | None) -> str:
        """Return the caller name, or raise.

        When ``headers`` is ``None`` there is no transport to authenticate — stdio,
        or an in-process ``Client(server.mcp)`` — so the call is local by
        definition. The ASGI gate is what protects real network traffic; this is the
        backstop for the paths that bypass it.
        """
        if headers is None:
            return auth_mod.CALLER_LOCAL
        result = auth_mod.verify_bearer(
            auth_mod.header_value(headers, "authorization"),
            self.keys,
            allow_anonymous=self.allow_anonymous,
        )
        if not result.ok:
            raise PermissionError(result.reason)
        return result.caller

    def scope(self, headers: Mapping[str, str] | None, caller: str) -> CallScope:
        """Parse and check depth, returning the scope for this call."""
        try:
            parsed = depth_mod.parse_depth(headers, limit=self.max_depth)
        except ValueError as exc:
            raise bad_headers_error(str(exc)) from None
        try:
            depth_mod.check_depth(
                parsed.depth,
                limit=self.max_depth,
                conversation_id=parsed.conversation_id,
            )
        except depth_mod.DepthExceeded as exc:
            raise depth_error(exc) from None
        return CallScope(
            conversation_id=parsed.conversation_id,
            depth=parsed.depth,
            caller=caller,
            confirmed=parsed.confirmed,
        )

    async def log(self, event: UsageEvent) -> None:
        """Write one usage row. Never raises — logging must not fail a call."""
        if not self.usage_enabled:
            return
        try:
            await asyncio.to_thread(self.sink.record, event)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 — a log write must never break a tool call
            logger.exception("Failed to write usage row for %s", event.name)


def _extract_ctx(kwargs: dict[str, Any], ctx_name: str, injected: bool) -> Any:
    """Take the Context out of kwargs (ours) or read it (the author's)."""
    if injected:
        return kwargs.pop(ctx_name, None)
    return kwargs.get(ctx_name)


def _build_wrapper(
    fn: Callable[..., Any],
    *,
    kind: str,
    name: str,
    guard: Guard,
    policy: ToolPolicy | None,
) -> Callable[..., Any]:
    """Shared machinery for tool and resource wrappers. See the module docstring."""
    sig = inspect.signature(fn)

    # Resolve annotations ONCE, in the author's own module namespace (pitfall 2).
    try:
        hints = typing.get_type_hints(fn)
    except Exception as exc:  # noqa: BLE001 — surface as a clear registration error
        raise ValueError(
            f"Cannot resolve type annotations for {kind} {name!r}: {exc}. "
            "Every parameter and the return value need annotations the library can "
            "import at registration time."
        ) from exc

    existing_ctx = find_context_parameter(fn)
    injected = existing_ctx is None
    ctx_name = existing_ctx or CTX_PARAM

    if injected and CTX_PARAM in sig.parameters:
        raise ValueError(
            f"{kind} {name!r} declares a parameter named {CTX_PARAM!r}, which is "
            "reserved by agent_mcp. Rename it, or annotate it as "
            "`ctx: Context` to receive the request context yourself."
        )

    if injected:
        params = list(sig.parameters.values())
        var_kw = [p for p in params if p.kind is inspect.Parameter.VAR_KEYWORD]
        head = [p for p in params if p.kind is not inspect.Parameter.VAR_KEYWORD]
        new_sig = sig.replace(
            parameters=[
                *head,
                inspect.Parameter(
                    CTX_PARAM, inspect.Parameter.KEYWORD_ONLY, annotation=Context
                ),
                *var_kw,
            ]
        )
    else:
        new_sig = sig

    is_async = inspect.iscoroutinefunction(fn)

    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        ctx = _extract_ctx(kwargs, ctx_name, injected)
        headers = getattr(ctx, "headers", None) if ctx is not None else None

        # The timer and the log live outside the guard checks on purpose: a call
        # refused for a missing confirmation, a bad token, or an over-cap depth is
        # still a call, and a refused privileged action is precisely the event worth
        # having in an audit trail. Everything below therefore reports through the
        # same `finally`.
        started = time.perf_counter()
        caller = auth_mod.CALLER_UNAUTHENTICATED
        scope: CallScope | None = None
        token = None
        ok, error = True, None
        try:
            caller = guard.authenticate(headers)
            scope = guard.scope(headers, caller)

            if (
                policy is not None
                and policy.requires_confirmation
                and not scope.confirmed
            ):
                raise ConfirmationRequired(policy.name)

            token = _current_call.set(scope)
            if is_async:
                return await fn(*args, **kwargs)
            return await asyncio.to_thread(fn, *args, **kwargs)
        except asyncio.CancelledError:
            raise  # let an interrupt unwind cleanly
        except Exception as exc:
            ok, error = False, f"{type(exc).__name__}: {exc}"
            raise
        finally:
            if token is not None:
                _current_call.reset(token)
            await guard.log(
                UsageEvent(
                    kind=kind,
                    name=name,
                    app_name=guard.app_name,
                    caller=caller,
                    ok=ok,
                    error=error,
                    latency_ms=int((time.perf_counter() - started) * 1000),
                    conversation_id=scope.conversation_id if scope else None,
                    depth=scope.depth if scope else 0,
                )
            )

    functools.wraps(fn)(wrapper)
    # After wraps (pitfall 1), and with real type objects rather than strings
    # (pitfall 2).
    wrapper.__annotations__ = {**hints, CTX_PARAM: Context} if injected else dict(hints)
    wrapper.__signature__ = new_sig.replace(
        return_annotation=hints.get("return", sig.return_annotation)
    )

    # Pitfall 3: prove the SDK agrees, or fail loudly at import.
    seen = find_context_parameter(wrapper)
    if seen != ctx_name:
        raise RuntimeError(
            f"agent_mcp: context injection self-check failed for {kind} {name!r} "
            f"(expected {ctx_name!r}, the SDK sees {seen!r}). The guards would be "
            "silently disabled, so registration is refused. This usually means the "
            "MCP SDK's internals changed."
        )
    return wrapper


def build_tool_wrapper(
    fn: Callable[..., Any], *, policy: ToolPolicy, guard: Guard
) -> Callable[..., Any]:
    """Wrap an author's tool function with the guard path."""
    return _build_wrapper(
        fn, kind="tool", name=policy.name, guard=guard, policy=policy
    )


def build_resource_wrapper(
    fn: Callable[..., Any], *, policy: ResourcePolicy, guard: Guard
) -> Callable[..., Any]:
    """Wrap an author's resource function.

    Only *templated* resources reach the full guard path: the SDK refuses to inject
    a ``Context`` into a static-URI handler, so for those there are no headers, the
    caller is unknown, and the row is logged with a null conversation id. The ASGI
    gate is the only thing authenticating a static resource read — which is the
    strongest argument for it existing.
    """
    return _build_wrapper(
        fn, kind="resource", name=policy.name, guard=guard, policy=None
    )


def build_static_resource_wrapper(
    fn: Callable[..., Any], *, policy: ResourcePolicy, guard: Guard
) -> Callable[..., Any]:
    """Wrap a static-URI resource: timing and outcome only, no header access."""
    is_async = inspect.iscoroutinefunction(fn)

    @functools.wraps(fn)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        started = time.perf_counter()
        ok, error = True, None
        try:
            if is_async:
                return await fn(*args, **kwargs)
            return await asyncio.to_thread(fn, *args, **kwargs)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            ok, error = False, f"{type(exc).__name__}: {exc}"
            raise
        finally:
            await guard.log(
                UsageEvent(
                    kind="resource",
                    name=policy.name,
                    app_name=guard.app_name,
                    caller=auth_mod.CALLER_LOCAL,
                    ok=ok,
                    error=error,
                    latency_ms=int((time.perf_counter() - started) * 1000),
                )
            )

    return wrapper


def tool_meta(policy: ToolPolicy, app_name: str) -> dict[str, Any]:
    """The ``_meta`` block advertising our flags to callers.

    ``read_only`` is duplicated here even though it also rides in the standard
    ``readOnlyHint`` annotation, so a consumer reads **one** dict for both flags
    instead of one from ``annotations`` and one from ``_meta``. Verified: extra
    fields on ``ToolAnnotations`` are silently dropped (``extra="ignore"``), so
    ``requiresConfirmation`` has nowhere else to go.
    """
    return {
        META_REQUIRES_CONFIRMATION: policy.requires_confirmation,
        META_READ_ONLY: policy.read_only,
        META_APP: app_name,
    }


def confirmation_notice(description: str | None, policy: ToolPolicy) -> str:
    """Append the approval requirement to a tool's description.

    A plain LLM client never looks at ``_meta``; the description is the only channel
    the model itself reliably reads.
    """
    text = (description or "").strip()
    if not policy.requires_confirmation:
        return text
    notice = (
        "Requires confirmation: this action is only carried out after explicit "
        "human approval."
    )
    return f"{text}\n\n{notice}".strip()
