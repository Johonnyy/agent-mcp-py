"""Conversation depth — the rule that stops agents calling each other forever.

MCP has no native notion of "how many agents deep is this call?", so the ecosystem
carries it in three headers: a stable ``X-Conversation-Id`` threaded through every
hop, an ``X-Agent-Depth`` counter incremented on each one, and ``X-Confirmed`` for
the human-approval gate. A call arriving above :data:`MAX_AGENT_DEPTH` is refused.
Without this, one server whose tool calls a peer whose tool calls back is an
infinite, *billable* loop.

**This module is a stdlib-only leaf and must stay that way.** ``agent_runtime``
imports these constants so its loop can pre-check the cap client-side before
spending a request — it must not have to install ``mcp`` (and starlette, uvicorn,
opentelemetry, pyjwt…) to learn that the limit is 5. There is exactly one
definition of that number and it lives here. The translation of :class:`DepthExceeded`
into a wire-level MCP error needs ``mcp`` and therefore lives in `errors.py`; the
import direction is strictly ``errors -> depth`` and never back.

Design rules worth knowing before you change anything here:

* **Header lookup is case-insensitive.** ASGI hands over lowercased bytes and the
  SDK's ``ctx.headers`` is lowercased too (verified against a live server), but
  a caller constructing headers by hand will write ``X-Agent-Depth``. Both sides
  are lowercased rather than trusting either.
* **A missing depth header means depth 0, not an error.** A human clicking a
  button in a dashboard is depth 0 by definition; demanding the header would break
  every non-agent caller.
* **A malformed depth header is a hard error, not a coerce-to-zero.** Silently
  reading ``"abc"`` as 0 turns a broken caller into an invisible loop-guard bypass,
  which is the one failure this module exists to prevent.
* **``confirmed`` does not propagate to the next hop by default.** A human approved
  *this* action, not whatever the callee decides to do downstream.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from uuid import uuid4

HEADER_CONVERSATION_ID = "X-Conversation-Id"
HEADER_AGENT_DEPTH = "X-Agent-Depth"
HEADER_CONFIRMED = "X-Confirmed"

#: Hops allowed before a call is refused. The single source of truth — config reads
#: it from here rather than repeating the literal, and so does ``agent_runtime``.
MAX_AGENT_DEPTH = 5

#: Header values accepted as "yes" for ``X-Confirmed``. Deliberately short: a typo
#: like ``yes`` must not read as approval for a destructive tool.
_TRUTHY = frozenset({"true", "1"})


class DepthExceeded(Exception):
    """Raised when a call arrives at, or would be sent past, the depth cap."""

    def __init__(
        self, depth: int, limit: int, conversation_id: str | None = None
    ) -> None:
        self.depth = depth
        self.limit = limit
        self.conversation_id = conversation_id
        super().__init__(
            f"agent depth {depth} exceeds the limit of {limit}"
            + (f" (conversation {conversation_id})" if conversation_id else "")
        )


@dataclass(frozen=True)
class CallDepth:
    """The depth state of one inbound call, parsed from its headers."""

    conversation_id: str
    depth: int
    confirmed: bool = False


def _lower(headers: Mapping[str, str] | None) -> dict[str, str]:
    """Headers with lowercased keys. ``None`` (stdio, in-process) becomes empty."""
    if not headers:
        return {}
    return {str(k).lower(): v for k, v in headers.items()}


def parse_depth(
    headers: Mapping[str, str] | None, *, limit: int = MAX_AGENT_DEPTH
) -> CallDepth:
    """Read the depth headers off an inbound call.

    A missing conversation id is minted here so every call has one to log and to
    forward, even when the caller is a human client that knows nothing about the
    convention. ``limit`` is accepted only to be echoed into the error raised for a
    negative depth; the cap itself is checked by :func:`check_depth`.

    Raises ``ValueError`` if ``X-Agent-Depth`` is present but not a non-negative
    integer.
    """
    raw = _lower(headers)
    conversation_id = (raw.get(HEADER_CONVERSATION_ID.lower()) or "").strip()
    if not conversation_id:
        conversation_id = uuid4().hex

    depth_raw = (raw.get(HEADER_AGENT_DEPTH.lower()) or "").strip()
    if not depth_raw:
        depth = 0
    else:
        try:
            depth = int(depth_raw)
        except ValueError:
            raise ValueError(
                f"{HEADER_AGENT_DEPTH} must be a non-negative integer, got "
                f"{depth_raw[:32]!r}"
            ) from None
        if depth < 0:
            raise ValueError(
                f"{HEADER_AGENT_DEPTH} must be a non-negative integer, got {depth}"
            )

    confirmed = (raw.get(HEADER_CONFIRMED.lower()) or "").strip().lower() in _TRUTHY
    return CallDepth(
        conversation_id=conversation_id, depth=depth, confirmed=confirmed
    )


def check_depth(
    depth: int, *, limit: int = MAX_AGENT_DEPTH, conversation_id: str | None = None
) -> None:
    """Raise :class:`DepthExceeded` if ``depth`` is at or past the cap.

    At-or-past, not merely past: a call arriving *at* the limit has no hops left, so
    admitting it would let it make one more.
    """
    if depth >= limit:
        raise DepthExceeded(depth, limit, conversation_id)


def next_hop_headers(
    current: CallDepth,
    *,
    limit: int = MAX_AGENT_DEPTH,
    confirmed: bool = False,
) -> dict[str, str]:
    """Headers for a call this server makes *out* to a peer.

    This is the function ``agent_runtime`` calls, and it **re-checks the cap before
    building anything** — a caller that forgets to pre-check still cannot emit an
    over-limit request, because there are no headers to send it with. Belt and
    braces on purpose: the failure being prevented is expensive and silent.
    """
    check_depth(
        current.depth, limit=limit, conversation_id=current.conversation_id
    )
    headers = {
        HEADER_CONVERSATION_ID: current.conversation_id,
        HEADER_AGENT_DEPTH: str(current.depth + 1),
    }
    if confirmed:
        headers[HEADER_CONFIRMED] = "true"
    return headers


def is_confirmed(headers: Mapping[str, str] | None) -> bool:
    """True if ``X-Confirmed`` marks this call as human-approved."""
    return (
        (_lower(headers).get(HEADER_CONFIRMED.lower()) or "").strip().lower()
        in _TRUTHY
    )
