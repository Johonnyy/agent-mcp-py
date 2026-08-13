"""Tests for the usage log, including co-tenancy with agent_runtime's table."""

import re
import sqlite3

import pytest

from agent_mcp.usage_log import (
    NullUsageSink,
    SQLiteUsageSink,
    UsageEvent,
    now_iso,
)


@pytest.fixture
def sink():
    s = SQLiteUsageSink(":memory:")
    yield s
    s.close()


def _event(**over):
    base = dict(
        kind="tool",
        name="get_balance",
        app_name="finance",
        caller="amber",
        ok=True,
        latency_ms=12,
        conversation_id="conv-1",
        depth=1,
    )
    base.update(over)
    return UsageEvent(**base)


def test_a_recorded_event_reads_back(sink):
    sink.record(_event())
    rows = sink.rows()
    assert len(rows) == 1
    row = rows[0]
    assert row["name"] == "get_balance"
    assert row["app_name"] == "finance"
    assert row["caller"] == "amber"
    assert row["conversation_id"] == "conv-1"
    assert row["depth"] == 1
    assert row["ok"] == 1


def test_a_failure_row_carries_the_error_text(sink):
    sink.record(_event(ok=False, error="ValueError: nope"))
    row = sink.rows()[0]
    assert row["ok"] == 0
    assert row["error"] == "ValueError: nope"


def test_created_at_is_iso_8601_utc_seconds(sink):
    """The exact format agent_runtime writes, so the two tables join cleanly.
    Amber uses TEXT timestamps everywhere — never epoch floats."""
    sink.record(_event())
    created = sink.rows()[0]["created_at"]
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\+00:00", created), created
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\+00:00", now_iso())


def test_an_explicit_created_at_is_preserved(sink):
    sink.record(_event(created_at="2020-01-01T00:00:00+00:00"))
    assert sink.rows()[0]["created_at"] == "2020-01-01T00:00:00+00:00"


def test_summary_groups_by_tool_and_caller_with_counts(sink):
    sink.record(_event(name="a", caller="amber", latency_ms=10))
    sink.record(_event(name="a", caller="amber", latency_ms=30, ok=False, error="x"))
    sink.record(_event(name="b", caller="spawner", latency_ms=20))

    summary = sink.summary()
    assert summary["totals"]["calls"] == 3
    assert summary["totals"]["errors"] == 1

    by_tool = {row["name"]: row for row in summary["by_tool"]}
    assert by_tool["a"]["calls"] == 2
    assert by_tool["a"]["errors"] == 1
    assert by_tool["b"]["calls"] == 1

    by_caller = {row["caller"]: row for row in summary["by_caller"]}
    assert by_caller["amber"]["calls"] == 2
    assert by_caller["spawner"]["calls"] == 1


def test_summary_reports_latency_percentiles(sink):
    for latency in (10, 20, 30, 40, 1000):
        sink.record(_event(latency_ms=latency))
    totals = sink.summary()["totals"]
    assert totals["p95_ms"] == 1000
    assert 10 <= totals["p50_ms"] <= 40


def test_summary_of_an_empty_log_does_not_divide_by_zero(sink):
    summary = sink.summary()
    assert summary["totals"] == {"calls": 0, "errors": 0, "p50_ms": 0, "p95_ms": 0}


def test_the_since_filter_excludes_older_rows(sink):
    sink.record(_event(created_at="2020-01-01T00:00:00+00:00"))
    sink.record(_event(created_at="2030-01-01T00:00:00+00:00"))
    assert sink.summary()["totals"]["calls"] == 2
    assert sink.summary(since="2025-01-01T00:00:00+00:00")["totals"]["calls"] == 1
    assert len(sink.rows(since="2025-01-01T00:00:00+00:00")) == 1


def test_prune_deletes_only_rows_older_than_the_cutoff(sink):
    sink.record(_event(created_at="2000-01-01T00:00:00+00:00"))
    sink.record(_event())  # now
    assert sink.prune(30) == 1
    assert len(sink.rows()) == 1


def test_prune_with_a_non_positive_retention_is_a_no_op(sink):
    sink.record(_event(created_at="2000-01-01T00:00:00+00:00"))
    assert sink.prune(0) == 0
    assert len(sink.rows()) == 1


def test_the_table_co_exists_with_agent_runtimes_table():
    """Both libraries land in Amber's amber.db. Each creates its own table with
    CREATE TABLE IF NOT EXISTS and must tolerate the other's schema."""
    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE agent_runtime_usage ("
        " id INTEGER PRIMARY KEY AUTOINCREMENT,"
        " conversation_id TEXT, app_name TEXT, depth INTEGER, created_at TEXT)"
    )
    conn.commit()

    # Same connection, our schema applied on top.
    sink = SQLiteUsageSink(":memory:")
    sink._conn.close()
    sink._conn = conn
    conn.row_factory = sqlite3.Row
    with sink._lock:
        conn.executescript(
            "CREATE TABLE IF NOT EXISTS agent_mcp_usage ("
            " id INTEGER PRIMARY KEY AUTOINCREMENT, kind TEXT NOT NULL,"
            " name TEXT NOT NULL, app_name TEXT NOT NULL, caller TEXT NOT NULL,"
            " conversation_id TEXT, depth INTEGER NOT NULL DEFAULT 0,"
            " ok INTEGER NOT NULL, error TEXT, latency_ms INTEGER NOT NULL,"
            " created_at TEXT NOT NULL)"
        )
        conn.commit()

    sink.record(_event())
    tables = {
        r[0]
        for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    assert {"agent_mcp_usage", "agent_runtime_usage"} <= tables
    assert len(sink.rows()) == 1

    # The agreed join columns exist in both, with matching names.
    ours = {r[1] for r in conn.execute("PRAGMA table_info(agent_mcp_usage)")}
    theirs = {r[1] for r in conn.execute("PRAGMA table_info(agent_runtime_usage)")}
    assert {"conversation_id", "app_name", "depth", "created_at"} <= ours & theirs
    sink.close()


def test_a_real_file_gets_wal_mode(tmp_path):
    """WAL is what makes two libraries writing one file safe. It is persistent per
    file, so both setting it is idempotent."""
    path = str(tmp_path / "usage.db")
    sink = SQLiteUsageSink(path)
    mode = sink._conn.execute("PRAGMA journal_mode").fetchone()[0]
    assert mode.lower() == "wal"
    sink.close()


def test_the_null_sink_swallows_everything():
    sink = NullUsageSink()
    sink.record(_event())
    assert sink.summary()["totals"]["calls"] == 0
    assert sink.prune(1) == 0
    sink.close()
