"""Registration with the hosted sync store — the discovery half of the ecosystem.

On startup a server pushes a descriptor of itself — name, URL, tools, resources — so
Lucidity and every other agent can find it without anyone hand-editing a config
file. A heartbeat re-registers periodically so the store can age out servers that
have gone away.

**Nothing here may block or break startup.** The sync store is a convenience; an app
must run standalone with zero knowledge the ecosystem exists, which is the founding
principle of the whole project. So :func:`register` catches everything, logs a
warning, and returns ``False``. It is launched as a task from the lifespan and never
awaited on the startup path. An unreachable store costs discoverability and nothing
else.

The record *shape* is imported from `registry.py` rather than redefined here: that
module owns reads and this one owns writes, but there is one definition of what a
peer looks like. Two would drift the first time a field was added.

The tool and resource lists come from the server's in-process policy registry, not
from ``await mcp.list_tools()`` — it is synchronous, needs no event loop, and is the
same authoritative source that enforcement reads.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Sequence
from typing import Any

import httpx2  # not httpx: mcp v2 ships httpx2, a separate distribution

from agent_mcp.decorators import ResourcePolicy, ToolPolicy
from agent_mcp.registry import MCP_MOUNT_PATH

logger = logging.getLogger(__name__)


def build_descriptor(
    *,
    app_name: str,
    version: str,
    base_url: str,
    tools: Sequence[ToolPolicy],
    resources: Sequence[ResourcePolicy],
    descriptions: dict[str, str] | None = None,
) -> dict[str, Any]:
    """The JSON body pushed to the sync store."""
    desc = descriptions or {}
    return {
        "name": app_name,
        "version": version,
        "base_url": base_url.rstrip("/"),
        "mcp_path": MCP_MOUNT_PATH,
        "tools": [
            {
                "name": t.name,
                "description": desc.get(t.name, ""),
                "read_only": t.read_only,
                "requires_confirmation": t.requires_confirmation,
            }
            for t in tools
        ],
        "resources": [
            {"uri": r.uri, "name": r.name, "templated": r.templated}
            for r in resources
        ],
    }


async def register(
    *,
    sync_store_url: str,
    descriptor: dict[str, Any],
    token: str = "",
    timeout_s: float = 5.0,
) -> bool:
    """Push this server's descriptor. Returns success; never raises.

    Refuses to register without a ``base_url``: a service cannot work out its own
    externally reachable address, and publishing an unreachable one to the registry
    is worse than being absent from it — every peer that resolves the name would
    then fail to connect.
    """
    app_name = descriptor.get("name", "?")
    if not sync_store_url:
        logger.debug("[%s] No sync store configured; skipping registration", app_name)
        return False
    if not descriptor.get("base_url"):
        logger.warning(
            "[%s] No public URL configured (AGENT_MCP_PUBLIC_URL); skipping sync "
            "store registration. The server works normally but is not discoverable.",
            app_name,
        )
        return False

    headers = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        async with httpx2.AsyncClient(timeout=timeout_s) as client:
            response = await client.post(
                sync_store_url.rstrip("/") + "/servers",
                json=descriptor,
                headers=headers,
            )
            response.raise_for_status()
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001 — registration must never break startup
        logger.warning("[%s] Sync store unreachable (%s): %s", app_name, sync_store_url, exc)
        return False

    logger.info("[%s] Registered with the sync store at %s", app_name, sync_store_url)
    return True


async def heartbeat_loop(
    *,
    interval_s: float,
    register_fn: Callable[[], Any],
    on_tick: Callable[[], Any] | None = None,
) -> None:
    """Re-register every ``interval_s`` seconds until cancelled.

    ``on_tick`` is the hook the server uses to prune old usage rows — retention work
    belongs on this timer rather than on the write path of a tool call.
    """
    if interval_s <= 0:
        return
    while True:
        try:
            await asyncio.sleep(interval_s)
            await register_fn()
            if on_tick is not None:
                result = on_tick()
                if asyncio.iscoroutine(result):
                    await result
        except asyncio.CancelledError:
            raise  # shutdown — let it unwind
        except Exception:  # noqa: BLE001 — a bad tick must not kill the heartbeat
            logger.exception("Heartbeat tick failed")
