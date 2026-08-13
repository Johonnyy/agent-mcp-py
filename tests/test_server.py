"""End-to-end tests through an in-process client, plus the guard paths.

Header-bearing cases use a small fake Context passed straight to the wrapper: an
in-process ``Client`` carries no HTTP headers (verified — ``ctx.headers`` is
``None``), so this is the only way to exercise depth and confirmation without
standing up a socket. The real HTTP path is covered by ``test_end_to_end.py``.
"""

import pytest
from mcp import Client, MCPError

from agent_mcp import AgentMCPServer, AgentMCPSettings
from agent_mcp.auth import CALLER_LOCAL
from agent_mcp.decorators import Guard, ToolPolicy, build_tool_wrapper
from agent_mcp.depth import HEADER_AGENT_DEPTH, HEADER_CONFIRMED, HEADER_CONVERSATION_ID
from agent_mcp.errors import ConfirmationRequired
from agent_mcp.usage_log import NullUsageSink, SQLiteUsageSink


class _FakeContext:
    """Stands in for mcp Context — the wrapper only ever reads `.headers`."""

    def __init__(self, headers=None):
        self.headers = headers


@pytest.fixture
def sink():
    s = SQLiteUsageSink(":memory:")
    yield s
    s.close()


def _settings(**over):
    base = dict(
        app_name="finance",
        allow_anonymous=True,
        usage_enabled=True,
        usage_db_path=":memory:",
        sync_store_url="",
    )
    base.update(over)
    return AgentMCPSettings(_env_file=None, **base)


def _server(sink, **over):
    return AgentMCPServer(
        app_name="finance", version="0.1.0", settings=_settings(**over), usage_sink=sink
    )


def _guard(sink, **over):
    base = dict(
        app_name="finance",
        keys={},
        allow_anonymous=True,
        max_depth=5,
        sink=sink,
        usage_enabled=True,
    )
    base.update(over)
    return Guard(**base)


async def test_a_successful_call_returns_a_result_and_writes_one_usage_row(sink):
    server = _server(sink)

    @server.tool(read_only=True)
    def get_balance(account: str) -> float:
        """Balance."""
        return 12.5

    async with Client(server.mcp) as client:
        result = await client.call_tool("get_balance", {"account": "main"})

    assert result.is_error is False
    assert "12.5" in result.content[0].text

    rows = sink.rows()
    assert len(rows) == 1
    assert rows[0]["name"] == "get_balance"
    assert rows[0]["ok"] == 1
    assert rows[0]["kind"] == "tool"
    assert rows[0]["caller"] == CALLER_LOCAL
    assert rows[0]["latency_ms"] >= 0


async def test_a_raising_tool_becomes_is_error_and_still_writes_a_row(sink):
    """Amber's rule: a bad tool never crashes a turn. The model reads the text and
    can try something else."""
    server = _server(sink)

    @server.tool()
    def boom() -> str:
        """Always fails."""
        raise ValueError("kaboom")

    async with Client(server.mcp) as client:
        result = await client.call_tool("boom", {})

    assert result.is_error is True
    assert "kaboom" in result.content[0].text

    rows = sink.rows()
    assert len(rows) == 1
    assert rows[0]["ok"] == 0
    assert "kaboom" in rows[0]["error"]


async def test_a_resource_read_is_logged(sink):
    server = _server(sink)

    @server.resource("finance://summary")
    def summary() -> dict:
        """Doc."""
        return {"ok": True}

    async with Client(server.mcp) as client:
        await client.read_resource("finance://summary")

    assert sink.rows()[0]["kind"] == "resource"


async def test_an_inbound_call_over_the_depth_cap_raises_a_protocol_error(sink):
    """A hard JSON-RPC error, not isError text: callers pre-check the cap, so
    arriving here means the pre-check was bypassed."""
    guard = _guard(sink, max_depth=5)

    def tool() -> str:
        """Doc."""
        return "ok"

    wrapper = build_tool_wrapper(tool, policy=ToolPolicy("tool"), guard=guard)
    ctx = _FakeContext({HEADER_AGENT_DEPTH: "5", HEADER_CONVERSATION_ID: "c1"})

    with pytest.raises(MCPError, match="depth limit"):
        await wrapper(agent_mcp_ctx=ctx)

    assert sink.rows()[0]["ok"] == 0


async def test_a_call_below_the_cap_is_admitted_and_records_its_depth(sink):
    guard = _guard(sink)

    def tool() -> str:
        """Doc."""
        return "ok"

    wrapper = build_tool_wrapper(tool, policy=ToolPolicy("tool"), guard=guard)
    ctx = _FakeContext({HEADER_AGENT_DEPTH: "3", HEADER_CONVERSATION_ID: "conv-x"})
    assert await wrapper(agent_mcp_ctx=ctx) == "ok"

    row = sink.rows()[0]
    assert row["depth"] == 3
    assert row["conversation_id"] == "conv-x"


async def test_a_malformed_depth_header_is_an_unretryable_protocol_error(sink):
    guard = _guard(sink)

    def tool() -> str:
        """Doc."""
        return "ok"

    wrapper = build_tool_wrapper(tool, policy=ToolPolicy("tool"), guard=guard)
    with pytest.raises(MCPError, match=HEADER_AGENT_DEPTH):
        await wrapper(agent_mcp_ctx=_FakeContext({HEADER_AGENT_DEPTH: "abc"}))


async def test_a_confirmation_tool_without_the_header_is_refused_recoverably(sink):
    """An ordinary exception, so it surfaces as isError text the harness can act on
    rather than a protocol error every caller must catch."""
    guard = _guard(sink)

    def create_invoice(amount: float) -> str:
        """Doc."""
        return "made"

    policy = ToolPolicy("create_invoice", requires_confirmation=True)
    wrapper = build_tool_wrapper(create_invoice, policy=policy, guard=guard)

    with pytest.raises(ConfirmationRequired, match="X-Confirmed"):
        await wrapper(amount=1.0, agent_mcp_ctx=_FakeContext({}))


async def test_a_refused_confirmation_is_still_audited(sink):
    """A blocked privileged action is exactly the event worth having in the log."""
    guard = _guard(sink)

    def create_invoice(amount: float) -> str:
        """Doc."""
        return "made"

    policy = ToolPolicy("create_invoice", requires_confirmation=True)
    wrapper = build_tool_wrapper(create_invoice, policy=policy, guard=guard)

    with pytest.raises(ConfirmationRequired):
        await wrapper(amount=1.0, agent_mcp_ctx=_FakeContext({}))

    row = sink.rows()[0]
    assert row["name"] == "create_invoice"
    assert row["ok"] == 0
    assert "ConfirmationRequired" in row["error"]


async def test_a_confirmation_tool_with_the_header_runs(sink):
    guard = _guard(sink)

    def create_invoice(amount: float) -> str:
        """Doc."""
        return f"invoice:{amount}"

    policy = ToolPolicy("create_invoice", requires_confirmation=True)
    wrapper = build_tool_wrapper(create_invoice, policy=policy, guard=guard)

    ctx = _FakeContext({HEADER_CONFIRMED: "true"})
    assert await wrapper(amount=2.0, agent_mcp_ctx=ctx) == "invoice:2.0"
    assert sink.rows()[0]["ok"] == 1


async def test_an_unrecognised_token_is_refused_by_the_wrapper_backstop(sink):
    """The ASGI gate normally catches this, but stdio and in-process clients bypass
    it — the wrapper is what covers those."""
    guard = _guard(sink, keys={"good": "amber"}, allow_anonymous=False)

    def tool() -> str:
        """Doc."""
        return "ok"

    wrapper = build_tool_wrapper(tool, policy=ToolPolicy("tool"), guard=guard)

    with pytest.raises(PermissionError, match="unrecognised"):
        await wrapper(agent_mcp_ctx=_FakeContext({"authorization": "Bearer bad"}))

    assert sink.rows()[0]["caller"] == "unauthenticated"


async def test_a_recognised_token_names_the_caller_in_the_log(sink):
    guard = _guard(sink, keys={"good": "amber"}, allow_anonymous=False)

    def tool() -> str:
        """Doc."""
        return "ok"

    wrapper = build_tool_wrapper(tool, policy=ToolPolicy("tool"), guard=guard)
    await wrapper(agent_mcp_ctx=_FakeContext({"authorization": "Bearer good"}))
    assert sink.rows()[0]["caller"] == "amber"


async def test_no_headers_at_all_counts_as_a_local_call(sink):
    """stdio and in-process clients have no transport to authenticate."""
    guard = _guard(sink, keys={"good": "amber"}, allow_anonymous=False)

    def tool() -> str:
        """Doc."""
        return "ok"

    wrapper = build_tool_wrapper(tool, policy=ToolPolicy("tool"), guard=guard)
    assert await wrapper(agent_mcp_ctx=_FakeContext(None)) == "ok"
    assert sink.rows()[0]["caller"] == CALLER_LOCAL


def test_asgi_app_is_idempotent(sink):
    """Two session managers would mean the lifespan runs one and traffic hits the
    other — invisible until the first request fails."""
    server = _server(sink)
    assert server.asgi_app() is server.asgi_app()


def test_serving_with_no_keys_and_no_anonymous_flag_is_refused(sink):
    """Fail closed: a forgotten env var is a loud refusal, not an open server."""
    server = _server(sink, allow_anonymous=False, keys="")
    with pytest.raises(RuntimeError, match="no bearer keys"):
        server.asgi_app()


def test_explicit_auth_keys_satisfy_the_fail_closed_check(sink):
    server = AgentMCPServer(
        app_name="finance",
        settings=_settings(allow_anonymous=False, keys=""),
        auth_keys=["amber:secret"],
        usage_sink=sink,
    )
    assert server.asgi_app() is not None


def test_auth_keys_can_be_read_from_a_named_env_var(monkeypatch, sink):
    monkeypatch.setenv("FINANCE_MCP_KEYS", "amber:fromenv")
    server = AgentMCPServer(
        app_name="finance",
        settings=_settings(allow_anonymous=False, keys=""),
        auth_keys_env="FINANCE_MCP_KEYS",
        usage_sink=sink,
    )
    assert server._keys == {"fromenv": "amber"}


def test_the_app_name_must_itself_be_a_valid_namespace(sink):
    with pytest.raises(ValueError, match="snake_case"):
        AgentMCPServer(app_name="Finance App", settings=_settings(), usage_sink=sink)


async def test_usage_routes_require_authentication(sink):
    from starlette.applications import Starlette
    from starlette.testclient import TestClient

    server = AgentMCPServer(
        app_name="finance",
        settings=_settings(allow_anonymous=False, keys="amber:secret"),
        usage_sink=sink,
    )

    @server.tool(read_only=True)
    def ping() -> str:
        """Doc."""
        return "pong"

    async with Client(server.mcp) as client:
        await client.call_tool("ping", {})

    app = Starlette(routes=server.usage_routes())
    with TestClient(app) as http:
        assert http.get("/agent/usage").status_code == 401
        response = http.get(
            "/agent/usage", headers={"Authorization": "Bearer secret"}
        )
        assert response.status_code == 200
        body = response.json()
        assert body["app"] == "finance"
        assert body["totals"]["calls"] == 1
        assert body["by_tool"][0]["name"] == "ping"


async def test_call_peer_refuses_an_unknown_peer_by_name(sink):
    server = _server(sink)
    from agent_mcp.errors import PeerCallRefused

    with pytest.raises(PeerCallRefused, match="Unknown peer"):
        await server.call_peer("nope", "some_tool")


async def test_call_peer_refuses_a_hop_that_would_exceed_the_cap(sink):
    """An ordinary exception, not a protocol error: this fires inside a tool body,
    where a bad peer call must not take down an otherwise fine turn."""
    from agent_mcp.errors import PeerCallRefused
    from agent_mcp.registry import PeerRecord

    server = AgentMCPServer(
        app_name="finance",
        settings=_settings(peers="school=https://school.example"),
        peers={"school": PeerRecord("school", "https://school.example")},
        usage_sink=sink,
    )
    ctx = _FakeContext({HEADER_AGENT_DEPTH: "5", HEADER_CONVERSATION_ID: "c1"})

    with pytest.raises(PeerCallRefused, match="depth"):
        await server.call_peer("school", "lookup", ctx=ctx)
