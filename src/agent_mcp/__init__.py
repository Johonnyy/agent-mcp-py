"""agent_mcp — the convention layer for exposing an app as an MCP server.

Import :class:`AgentMCPServer` and decorate functions; auth, the conversation-depth
guard, usage logging, and sync-store registration come with them::

    from agent_mcp import AgentMCPServer

    mcp = AgentMCPServer(app_name="finance", version="0.1.0")

    @mcp.tool(read_only=True)
    def get_balance(account: str) -> float:
        return db.get_balance(account)

    app.mount("/mcp", mcp.asgi_app())

``Context`` and ``MCPError`` are re-exported so an app never imports from ``mcp.*``
directly, and therefore never has to care that v2 moved everything.

**Every re-export here is lazy** (PEP 562). That is not a micro-optimisation — it is
the interop contract with ``agent_runtime``, which imports ``agent_mcp.depth`` for
the header constants and the depth cap. Importing a submodule runs its parent
package first, so an eager ``__init__`` would mean ``from agent_mcp.depth import
MAX_AGENT_DEPTH`` pulls in mcp, pydantic, httpx2, starlette and uvicorn to read an
integer. With ``__getattr__`` the leaf modules stay leaves. There is a test that
fails if this regresses.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

__version__ = "0.1.0"

#: Public name -> the module that defines it. Resolved on first access.
_EXPORTS: dict[str, str] = {
    # Server
    "AgentMCPServer": "agent_mcp.server",
    # Config
    "AgentMCPSettings": "agent_mcp.config",
    "get_settings": "agent_mcp.config",
    # Auth
    "AuthResult": "agent_mcp.auth",
    "verify_bearer": "agent_mcp.auth",
    # Depth (stdlib-only leaf)
    "CallDepth": "agent_mcp.depth",
    "DepthExceeded": "agent_mcp.depth",
    "HEADER_AGENT_DEPTH": "agent_mcp.depth",
    "HEADER_CONFIRMED": "agent_mcp.depth",
    "HEADER_CONVERSATION_ID": "agent_mcp.depth",
    "MAX_AGENT_DEPTH": "agent_mcp.depth",
    "check_depth": "agent_mcp.depth",
    "next_hop_headers": "agent_mcp.depth",
    "parse_depth": "agent_mcp.depth",
    # Policies
    "CallScope": "agent_mcp.decorators",
    "ResourcePolicy": "agent_mcp.decorators",
    "ToolPolicy": "agent_mcp.decorators",
    "current_call": "agent_mcp.decorators",
    # Errors
    "ConfirmationRequired": "agent_mcp.errors",
    "PeerCallRefused": "agent_mcp.errors",
    "MCPError": "agent_mcp.errors",
    # Registry
    "MCP_MOUNT_PATH": "agent_mcp.registry",
    "PeerRecord": "agent_mcp.registry",
    "known_peers": "agent_mcp.registry",
    "mcp_url": "agent_mcp.registry",
    "resolve": "agent_mcp.registry",
    # Usage log
    "NullUsageSink": "agent_mcp.usage_log",
    "SQLiteUsageSink": "agent_mcp.usage_log",
    "UsageEvent": "agent_mcp.usage_log",
    "UsageSink": "agent_mcp.usage_log",
}

#: Not in _EXPORTS because it lives in the SDK, not in one of our modules.
_SDK_EXPORTS: dict[str, str] = {"Context": "mcp.server.mcpserver"}

__all__ = [*sorted(_EXPORTS), *sorted(_SDK_EXPORTS), "__version__"]


def __getattr__(name: str) -> Any:
    module_name = _EXPORTS.get(name) or _SDK_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module_name), name)
    globals()[name] = value  # cache, so this runs once per name
    return value


def __dir__() -> list[str]:
    return sorted(__all__)


if TYPE_CHECKING:  # pragma: no cover — for type checkers and IDEs only
    from mcp.server.mcpserver import Context

    from agent_mcp.auth import AuthResult, verify_bearer
    from agent_mcp.config import AgentMCPSettings, get_settings
    from agent_mcp.decorators import (
        CallScope,
        ResourcePolicy,
        ToolPolicy,
        current_call,
    )
    from agent_mcp.depth import (
        HEADER_AGENT_DEPTH,
        HEADER_CONFIRMED,
        HEADER_CONVERSATION_ID,
        MAX_AGENT_DEPTH,
        CallDepth,
        DepthExceeded,
        check_depth,
        next_hop_headers,
        parse_depth,
    )
    from agent_mcp.errors import ConfirmationRequired, MCPError, PeerCallRefused
    from agent_mcp.registry import (
        MCP_MOUNT_PATH,
        PeerRecord,
        known_peers,
        mcp_url,
        resolve,
    )
    from agent_mcp.server import AgentMCPServer
    from agent_mcp.usage_log import (
        NullUsageSink,
        SQLiteUsageSink,
        UsageEvent,
        UsageSink,
    )
