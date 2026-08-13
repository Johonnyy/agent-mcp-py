"""The error taxonomy — deciding what a failure looks like on the wire.

This is the one module in the guard path that imports ``mcp``, which is why it is
separate from `depth.py`: ``agent_runtime`` needs the depth constants without the
server stack, and only the translation into a protocol error needs the SDK.

The SDK gives us exactly two channels, verified against the installed package in
``mcp/server/mcpserver/tools/base.py``:

* ``raise MCPError(...)`` propagates uncaught and fails the whole ``tools/call`` as
  a **JSON-RPC error**. The calling client raises; the model sees nothing.
* Any other exception is caught and returned as ``CallToolResult(is_error=True)``
  with the message as text. The model reads it and can try something else.

The SDK docs frame the choice as "could a smarter model have avoided this?", and
that is the right rule for tool bodies. Our guards need one more distinction, so the
placement is:

============================  ==========================================
Missing/bad bearer token      HTTP **401** in the ASGI gate — never
                              reaches MCP at all
Malformed ``X-Agent-Depth``   ``MCPError(INVALID_PARAMS)`` — a broken
                              caller, not a model mistake; unretryable
**Inbound** depth over cap    ``MCPError(INVALID_REQUEST)`` — the
                              loop-detection backstop. Callers pre-check,
                              so arriving here means the pre-check was
                              bypassed; it must be a hard protocol error
                              rather than text a model might paraphrase
                              and retry
**Outbound** hop over cap     ordinary exception → ``isError: true``
Missing ``X-Confirmed``       ordinary exception → ``isError: true``
Exception in a tool body      ``isError: true`` (the SDK does this)
============================  ==========================================

The inbound/outbound asymmetry is deliberate and is the only place the taxonomy
departs from "who could have avoided this". An outbound over-cap hop is raised
*inside* a tool body, where Amber's rule wins: a bad tool never crashes a turn. The
model is told the peer call would exceed the depth limit and answers with what it
has. Making it a protocol error would take down a turn that was otherwise fine.
"""

from __future__ import annotations

from mcp import MCPError
from mcp.types import INVALID_PARAMS, INVALID_REQUEST

from agent_mcp.depth import DepthExceeded

__all__ = [
    "ConfirmationRequired",
    "PeerCallRefused",
    "MCPError",
    "depth_error",
    "bad_headers_error",
]


class ConfirmationRequired(Exception):
    """A ``requires_confirmation`` tool was called without ``X-Confirmed``.

    An ordinary exception on purpose: this is an *expected, recoverable* state in an
    agent loop, not a protocol violation. The harness reads the message, gets human
    approval, and retries with the header. Forcing every caller into JSON-RPC
    exception handling for the normal path would be hostile.
    """

    def __init__(self, tool_name: str) -> None:
        self.tool_name = tool_name
        super().__init__(
            f"Tool {tool_name!r} requires explicit human approval. Retry with the "
            "header 'X-Confirmed: true' once the action has been approved."
        )


class PeerCallRefused(Exception):
    """An outbound ``call_peer`` was refused before it left the process."""


def depth_error(exc: DepthExceeded) -> MCPError:
    """Translate an inbound over-cap call into a hard protocol error."""
    return MCPError(
        INVALID_REQUEST,
        f"Agent depth limit reached: this call arrived at depth {exc.depth} and the "
        f"limit is {exc.limit}. A chain of agents is calling each other too deeply; "
        "the request was refused to prevent a loop.",
    )


def bad_headers_error(message: str) -> MCPError:
    """Translate a malformed depth header into an unretryable protocol error."""
    return MCPError(INVALID_PARAMS, message)
