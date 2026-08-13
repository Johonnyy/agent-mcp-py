"""Peer discovery — turning a server *name* into somewhere to send a request.

``mcp.call_peer("finance", "get_balance", account="main")`` has to resolve
``"finance"`` to a URL and a token from somewhere. That somewhere is this module: a
process-local cache of :class:`PeerRecord`, filled from a static environment map and
(once it exists) refreshed from the hosted sync store.

The read half lives here and the write half in `sync_client.py`, but the *record
shape* is defined once, here, and imported there — one client, not two. That matters
because ``agent_runtime`` also resolves peers, and a second implementation would
drift from this one the first time a field was added.

**:func:`resolve` is synchronous and performs no I/O, ever.** It reads a dict. That
is what lets an agent loop call it on the hot path without an await and without
touching ``httpx2``; only the background heartbeat calls :func:`refresh`. Getting
this backwards — a resolve that quietly makes a network call — would put a remote
dependency inside every tool dispatch.

Resolution order is explicit argument, then static env map, then sync-store cache.
**Static configuration beats discovery on purpose**, so pointing an app at a local
peer during an incident is not silently undone by the next heartbeat.

This module must not import ``mcp.server.*``: ``agent_runtime`` depends on it and
should not have to install a server stack to look up a URL. ``httpx2`` is fair game
— it arrives with ``mcp`` regardless, so it costs nothing, and :func:`refresh` is
called from async code where ``urllib`` would force a thread per network call.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field

import httpx2  # not httpx: mcp v2 ships httpx2, a separate distribution

logger = logging.getLogger(__name__)

#: Where every server in the ecosystem mounts its MCP endpoint. Fixed by convention
#: so the registry can store bare base URLs, and exported so no app hardcodes it.
MCP_MOUNT_PATH = "/mcp"


@dataclass(frozen=True)
class PeerRecord:
    """One discoverable MCP server."""

    name: str
    base_url: str
    token: str = ""
    version: str = ""
    tools: tuple[str, ...] = field(default=())


def mcp_url(record: PeerRecord) -> str:
    """The streamable-HTTP endpoint for a peer.

    Ends with a slash deliberately: a mounted MCP app answers ``/mcp`` with a 307
    redirect to ``/mcp/`` (verified against a live server), and a client that does
    not re-POST on a redirect fails silently. Always address the canonical form.
    """
    return record.base_url.rstrip("/") + MCP_MOUNT_PATH + "/"


def load_static_peers(spec: str | None, token: str = "") -> dict[str, PeerRecord]:
    """Parse ``"finance=https://a,school=https://b"`` into records.

    Tolerates whitespace and trailing commas so a hand-edited ``.env`` line does not
    become a startup failure.
    """
    peers: dict[str, PeerRecord] = {}
    for entry in (spec or "").split(","):
        entry = entry.strip()
        if not entry:
            continue
        name, sep, url = entry.partition("=")
        name, url = name.strip(), url.strip()
        if not sep or not name or not url:
            logger.warning("Ignoring malformed peer entry: %r", entry[:80])
            continue
        peers[name] = PeerRecord(name=name, base_url=url, token=token)
    return peers


class PeerRegistry:
    """The two-layer peer cache: static overrides in front of discovered records."""

    def __init__(
        self,
        static: Mapping[str, PeerRecord] | None = None,
        discovered: Mapping[str, PeerRecord] | None = None,
    ) -> None:
        self._static: dict[str, PeerRecord] = dict(static or {})
        self._discovered: dict[str, PeerRecord] = dict(discovered or {})

    def resolve(self, name: str) -> PeerRecord | None:
        """Look up a peer. Synchronous, no I/O."""
        return self._static.get(name) or self._discovered.get(name)

    def known(self) -> list[str]:
        return sorted({*self._static, *self._discovered})

    def set_static(self, records: Mapping[str, PeerRecord]) -> None:
        self._static = dict(records)

    async def refresh(
        self,
        sync_store_url: str,
        *,
        token: str = "",
        timeout_s: float = 5.0,
    ) -> int:
        """Pull the peer list from the sync store into the discovered layer.

        Returns the number of records loaded, and **never raises**: discovery is a
        convenience, and an app whose registry is unreachable must keep serving with
        whatever it already knows.
        """
        if not sync_store_url:
            return 0
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        try:
            async with httpx2.AsyncClient(timeout=timeout_s) as client:
                response = await client.get(
                    sync_store_url.rstrip("/") + "/servers", headers=headers
                )
                response.raise_for_status()
                payload = response.json()
        except Exception as exc:  # noqa: BLE001 — discovery must never break serving
            logger.warning("Peer refresh failed (%s): %s", sync_store_url, exc)
            return 0

        records = payload.get("servers", payload) if isinstance(payload, dict) else payload
        if not isinstance(records, list):
            logger.warning("Peer refresh: unexpected payload shape from sync store")
            return 0

        loaded: dict[str, PeerRecord] = {}
        for raw in records:
            if not isinstance(raw, dict):
                continue
            name = str(raw.get("name") or "").strip()
            base_url = str(raw.get("base_url") or "").strip()
            if not name or not base_url:
                continue
            loaded[name] = PeerRecord(
                name=name,
                base_url=base_url,
                token=str(raw.get("token") or ""),
                version=str(raw.get("version") or ""),
                tools=tuple(raw.get("tools") or ()),
            )
        self._discovered = loaded
        logger.info("Peer refresh: %d server(s) from the sync store", len(loaded))
        return len(loaded)


#: Process-wide registry, so `resolve` works as a bare function for callers (notably
#: agent_runtime) that have no handle on a server object.
_registry = PeerRegistry()


def default_registry() -> PeerRegistry:
    return _registry


def resolve(
    name: str, *, records: Mapping[str, PeerRecord] | None = None
) -> PeerRecord | None:
    """Resolve a peer name. Explicit ``records`` win over everything cached."""
    if records and name in records:
        return records[name]
    return _registry.resolve(name)


def known_peers() -> list[str]:
    return _registry.known()


async def refresh(
    sync_store_url: str, *, token: str = "", timeout_s: float = 5.0
) -> int:
    return await _registry.refresh(
        sync_store_url, token=token, timeout_s=timeout_s
    )
