"""Usage logging — who called what, how long it took, and whether it worked.

Every tool and resource call produces one row: caller identity, name, latency,
outcome, and the conversation id that ties it to the rest of a multi-agent
exchange. The spawner's cost view later reads these through each app's own
``/agent/usage`` endpoint rather than a shared table, which is what keeps every app
independently deployable — there is no central database to be down.

**The sink is pluggable and the default is SQLite.** A standalone app that has never
heard of the ecosystem gets a working log with zero configuration. An app that
already owns a database — Amber, whose memory store is a SQLite file — passes its
own sink and gets one connection instead of two. That second case is the reason the
protocol exists: the alternative is two libraries opening the same file with two
locks and two schemas, which works, but only because of WAL and a busy timeout.

Co-tenancy rules, agreed with ``agent_runtime`` so both can write ``amber.db``:

* Our table is ``agent_mcp_usage``; theirs is ``agent_runtime_usage``. Each is
  created ``IF NOT EXISTS`` and neither touches the other's.
* ``PRAGMA journal_mode=WAL`` before the schema. It is a persistent property of the
  file, so both libraries setting it is idempotent.
* ``PRAGMA busy_timeout``. WAL removes reader/writer blocking but **not**
  writer/writer; without a timeout, two libraries writing at once surfaces as
  ``database is locked`` in the middle of a tool call.
* Shared join columns, exact names and types: ``conversation_id TEXT``,
  ``app_name TEXT``, ``depth INTEGER``, and ``created_at TEXT`` as ISO-8601 UTC
  seconds. Amber uses TEXT timestamps throughout — never epoch floats.

This layer is synchronous sqlite3, following Amber's store: one shared connection
with ``check_same_thread=False``, every access serialized under a lock. Callers wrap
it in ``asyncio.to_thread`` so a write never blocks the event loop.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS agent_mcp_usage (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    kind            TEXT    NOT NULL,              -- 'tool' | 'resource'
    name            TEXT    NOT NULL,
    app_name        TEXT    NOT NULL,
    caller          TEXT    NOT NULL,
    conversation_id TEXT,
    depth           INTEGER NOT NULL DEFAULT 0,
    ok              INTEGER NOT NULL,              -- 1 | 0
    error           TEXT,
    latency_ms      INTEGER NOT NULL,
    created_at      TEXT    NOT NULL               -- ISO-8601 UTC, seconds
);
CREATE INDEX IF NOT EXISTS idx_agent_mcp_usage_created
    ON agent_mcp_usage (created_at);
CREATE INDEX IF NOT EXISTS idx_agent_mcp_usage_name
    ON agent_mcp_usage (name, created_at);
CREATE INDEX IF NOT EXISTS idx_agent_mcp_usage_conv
    ON agent_mcp_usage (conversation_id);
"""


def now_iso() -> str:
    """Current UTC time as ISO-8601 seconds — lexically sortable, and the exact
    format ``agent_runtime`` writes so the two tables join cleanly."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass(frozen=True)
class UsageEvent:
    kind: str
    name: str
    app_name: str
    caller: str
    ok: bool
    latency_ms: int
    conversation_id: str | None = None
    depth: int = 0
    error: str | None = None
    created_at: str = ""

    def stamped(self) -> UsageEvent:
        """This event with ``created_at`` filled in if it wasn't already."""
        if self.created_at:
            return self
        return UsageEvent(**{**self.__dict__, "created_at": now_iso()})


@runtime_checkable
class UsageSink(Protocol):
    """Where usage rows go. Implement this to write into your own database."""

    def record(self, event: UsageEvent) -> None: ...

    def summary(self, *, since: str | None = None, limit: int = 100) -> dict: ...

    def prune(self, older_than_days: int) -> int: ...

    def close(self) -> None: ...


class NullUsageSink:
    """Drops everything. Used when logging is disabled and in tests."""

    def record(self, event: UsageEvent) -> None:
        return None

    def summary(self, *, since: str | None = None, limit: int = 100) -> dict:
        return {"totals": {"calls": 0, "errors": 0}, "by_tool": [], "by_caller": []}

    def prune(self, older_than_days: int) -> int:
        return 0

    def close(self) -> None:
        return None


def _percentile(values: list[int], pct: float) -> int:
    """Nearest-rank percentile. Small row counts make interpolation pointless."""
    if not values:
        return 0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, round(pct / 100 * len(ordered) + 0.5) - 1))
    return ordered[index]


class SQLiteUsageSink:
    """The default sink: one table in a SQLite file, safe to share with a co-tenant."""

    def __init__(self, path: str = "agent_mcp.db", *, busy_timeout_ms: int = 5000) -> None:
        self.path = path
        # check_same_thread=False: the connection is reused from asyncio.to_thread
        # worker threads. Safe because every access is serialized by _lock.
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            # WAL before the schema, and busy_timeout before any write: see the
            # module docstring on sharing this file with agent_runtime.
            if path != ":memory:":
                self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute(f"PRAGMA busy_timeout={int(busy_timeout_ms)}")
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    def record(self, event: UsageEvent) -> None:
        event = event.stamped()
        with self._lock:
            self._conn.execute(
                "INSERT INTO agent_mcp_usage (kind, name, app_name, caller, "
                "conversation_id, depth, ok, error, latency_ms, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    event.kind,
                    event.name,
                    event.app_name,
                    event.caller,
                    event.conversation_id,
                    int(event.depth),
                    1 if event.ok else 0,
                    event.error,
                    int(event.latency_ms),
                    event.created_at,
                ),
            )
            self._conn.commit()

    def rows(self, *, since: str | None = None, limit: int = 1000) -> list[dict]:
        sql = (
            "SELECT id, kind, name, app_name, caller, conversation_id, depth, ok, "
            "error, latency_ms, created_at FROM agent_mcp_usage"
        )
        params: list[object] = []
        if since:
            sql += " WHERE created_at >= ?"
            params.append(since)
        sql += " ORDER BY id DESC LIMIT ?"
        params.append(int(limit))
        with self._lock:
            found = self._conn.execute(sql, params).fetchall()
        return [dict(r) for r in found]

    def summary(self, *, since: str | None = None, limit: int = 100) -> dict:
        """Aggregate view for the ``/agent/usage`` endpoint."""
        where = " WHERE created_at >= ?" if since else ""
        params: tuple = (since,) if since else ()
        with self._lock:
            total = self._conn.execute(
                "SELECT COUNT(*) AS calls, "
                "SUM(CASE WHEN ok = 0 THEN 1 ELSE 0 END) AS errors "
                f"FROM agent_mcp_usage{where}",
                params,
            ).fetchone()
            per_tool = self._conn.execute(
                "SELECT name, COUNT(*) AS calls, "
                "SUM(CASE WHEN ok = 0 THEN 1 ELSE 0 END) AS errors "
                f"FROM agent_mcp_usage{where} GROUP BY name "
                "ORDER BY calls DESC LIMIT ?",
                (*params, int(limit)),
            ).fetchall()
            per_caller = self._conn.execute(
                "SELECT caller, COUNT(*) AS calls, "
                "SUM(CASE WHEN ok = 0 THEN 1 ELSE 0 END) AS errors "
                f"FROM agent_mcp_usage{where} GROUP BY caller "
                "ORDER BY calls DESC LIMIT ?",
                (*params, int(limit)),
            ).fetchall()
            # One pass for latencies: the overall percentiles and the per-tool ones
            # come out of the same rows, so there is no reason to query twice.
            all_latencies: list[int] = []
            by_tool_latency: dict[str, list[int]] = {}
            for row in self._conn.execute(
                f"SELECT name, latency_ms FROM agent_mcp_usage{where}", params
            ).fetchall():
                latency = int(row["latency_ms"])
                all_latencies.append(latency)
                by_tool_latency.setdefault(row["name"], []).append(latency)

        return {
            "since": since,
            "totals": {
                "calls": int(total["calls"] or 0),
                "errors": int(total["errors"] or 0),
                "p50_ms": _percentile(all_latencies, 50),
                "p95_ms": _percentile(all_latencies, 95),
            },
            "by_tool": [
                {
                    "name": r["name"],
                    "calls": int(r["calls"]),
                    "errors": int(r["errors"] or 0),
                    "p95_ms": _percentile(by_tool_latency.get(r["name"], []), 95),
                }
                for r in per_tool
            ],
            "by_caller": [
                {
                    "caller": r["caller"],
                    "calls": int(r["calls"]),
                    "errors": int(r["errors"] or 0),
                }
                for r in per_caller
            ],
        }

    def prune(self, older_than_days: int) -> int:
        """Delete rows older than ``older_than_days``. Returns the count removed."""
        if older_than_days <= 0:
            return 0
        cutoff_epoch = datetime.now(timezone.utc).timestamp() - older_than_days * 86400
        cutoff = datetime.fromtimestamp(cutoff_epoch, tz=timezone.utc).isoformat(
            timespec="seconds"
        )
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM agent_mcp_usage WHERE created_at < ?", (cutoff,)
            )
            self._conn.commit()
            return cur.rowcount

    def close(self) -> None:
        with self._lock:
            self._conn.close()
