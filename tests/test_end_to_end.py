"""The real thing: a Starlette app, a live uvicorn server, HTTP over a socket.

Everything else in the suite runs in-process, where ``ctx.headers`` is ``None`` and
the ASGI gate never executes. This file is what proves the parts that only exist on
the wire — the 401, the mount-path shim, header propagation into a tool, and the
usage endpoint — actually work when mounted the way an app will mount them.

Starlette rather than FastAPI on purpose: FastAPI is not a dependency of this
library and must not become one. If it works here it works under FastAPI, which is
a Starlette app.
"""

import asyncio
import socket
import threading
import time
from contextlib import asynccontextmanager

import httpx2
import pytest
import uvicorn
from mcp import Client
from mcp.client.streamable_http import streamable_http_client
from mcp.server.mcpserver import Context
from starlette.applications import Starlette

from agent_mcp import AgentMCPServer, AgentMCPSettings
from agent_mcp.depth import (
    HEADER_AGENT_DEPTH,
    HEADER_CONFIRMED,
    HEADER_CONVERSATION_ID,
)
from agent_mcp.usage_log import SQLiteUsageSink

TOKEN = "s3cret"
INIT = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": "2026-07-28",
        "capabilities": {},
        "clientInfo": {"name": "test", "version": "1"},
    },
}
ACCEPT = {
    "Accept": "application/json, text/event-stream",
    "Content-Type": "application/json",
}


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def live():
    """A real server on a real port, torn down at the end of the module."""
    sink = SQLiteUsageSink(":memory:")
    settings = AgentMCPSettings(
        _env_file=None,
        app_name="finance",
        keys=f"amber:{TOKEN}",
        allow_anonymous=False,
        usage_enabled=True,
        sync_store_url="",
    )
    mcp = AgentMCPServer(
        app_name="finance", version="0.1.0", settings=settings, usage_sink=sink
    )

    @mcp.tool(read_only=True)
    def get_balance(account: str) -> float:
        """Balance for an account."""
        return 12.5

    @mcp.tool(read_only=False, requires_confirmation=True)
    def create_invoice(customer_id: str, amount: float) -> str:
        """Create an invoice."""
        return f"invoice for {customer_id}: {amount}"

    @mcp.tool(read_only=True)
    def echo_depth(ctx: Context) -> str:
        """Report the depth headers this call arrived with."""
        headers = ctx.headers or {}
        return f"{headers.get('x-agent-depth')}|{headers.get('x-conversation-id')}"

    @asynccontextmanager
    async def lifespan(app):
        async with mcp.lifespan():
            yield

    app = Starlette(routes=mcp.routes(), lifespan=lifespan)
    app.router.routes.extend(mcp.usage_routes())

    port = _free_port()
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(200):
        if server.started:
            break
        time.sleep(0.05)
    assert server.started, "uvicorn did not start"

    yield f"http://127.0.0.1:{port}", sink

    server.should_exit = True
    thread.join(timeout=10)
    sink.close()


# --- the mount-path shim ---


async def test_the_mcp_endpoint_serves_both_with_and_without_a_trailing_slash(live):
    """Mount("/mcp") alone only matches /mcp/..., so a bare POST /mcp falls through
    to the host router's redirect_slashes and 307s — and the real MCP client does
    not follow that (it fails with "Unexpected content type"). routes() claims the
    exact path as well, so neither form redirects."""
    base, _ = live
    async with httpx2.AsyncClient(follow_redirects=False, timeout=10) as http:
        for path in ("/mcp", "/mcp/"):
            response = await http.post(
                f"{base}{path}",
                json=INIT,
                headers={**ACCEPT, "Authorization": f"Bearer {TOKEN}"},
            )
            assert response.status_code != 307, f"{path} still redirects"
            assert response.status_code == 200, f"{path} -> {response.status_code}"


async def test_a_real_client_connects_without_the_trailing_slash(live):
    """The failure this all exists to prevent, exercised through the actual SDK
    client rather than a raw POST."""
    base, _ = live
    async with httpx2.AsyncClient(
        headers={"Authorization": f"Bearer {TOKEN}"}, timeout=15
    ) as http:
        async with Client(
            streamable_http_client(f"{base}/mcp", http_client=http)
        ) as client:
            result = await client.call_tool("get_balance", {"account": "main"})
    assert result.is_error is False


# --- auth at the transport edge ---


async def test_a_request_without_a_token_is_rejected_with_401(live):
    base, _ = live
    async with httpx2.AsyncClient(timeout=10) as http:
        response = await http.post(f"{base}/mcp/", json=INIT, headers=ACCEPT)
    assert response.status_code == 401
    assert response.headers.get("www-authenticate", "").startswith("Bearer")
    assert response.json()["error"] == "unauthorized"


async def test_a_request_with_a_wrong_token_is_rejected_with_401(live):
    base, _ = live
    async with httpx2.AsyncClient(timeout=10) as http:
        response = await http.post(
            f"{base}/mcp/",
            json=INIT,
            headers={**ACCEPT, "Authorization": "Bearer wrong"},
        )
    assert response.status_code == 401


async def test_an_over_cap_request_is_refused_at_the_edge(live):
    base, _ = live
    async with httpx2.AsyncClient(timeout=10) as http:
        response = await http.post(
            f"{base}/mcp/",
            json=INIT,
            headers={
                **ACCEPT,
                "Authorization": f"Bearer {TOKEN}",
                HEADER_AGENT_DEPTH: "9",
            },
        )
    assert response.status_code == 403
    assert response.json()["error"] == "depth_exceeded"


async def test_a_malformed_depth_header_is_a_400(live):
    base, _ = live
    async with httpx2.AsyncClient(timeout=10) as http:
        response = await http.post(
            f"{base}/mcp/",
            json=INIT,
            headers={
                **ACCEPT,
                "Authorization": f"Bearer {TOKEN}",
                HEADER_AGENT_DEPTH: "abc",
            },
        )
    assert response.status_code == 400


# --- real MCP traffic ---


@asynccontextmanager
async def _client(base: str, **headers):
    async with httpx2.AsyncClient(
        headers={"Authorization": f"Bearer {TOKEN}", **headers}, timeout=15
    ) as http:
        async with Client(
            streamable_http_client(f"{base}/mcp/", http_client=http)
        ) as client:
            yield client


async def test_a_tool_call_succeeds_over_http_and_is_attributed_to_its_caller(live):
    base, sink = live
    async with _client(base) as client:
        result = await client.call_tool("get_balance", {"account": "main"})
    assert result.is_error is False
    assert "12.5" in result.content[0].text

    row = next(r for r in sink.rows() if r["name"] == "get_balance")
    assert row["ok"] == 1
    assert row["caller"] == "amber"  # from the name:token key, never the token


async def test_the_confirmation_gate_blocks_without_the_header(live):
    base, sink = live
    async with _client(base) as client:
        result = await client.call_tool(
            "create_invoice", {"customer_id": "c1", "amount": 5.0}
        )
    assert result.is_error is True
    assert "X-Confirmed" in result.content[0].text

    row = next(r for r in sink.rows() if r["name"] == "create_invoice")
    assert row["ok"] == 0  # the refusal is audited


async def test_the_confirmation_gate_admits_the_call_with_the_header(live):
    base, _ = live
    async with _client(base, **{HEADER_CONFIRMED: "true"}) as client:
        result = await client.call_tool(
            "create_invoice", {"customer_id": "c2", "amount": 7.0}
        )
    assert result.is_error is False
    assert "invoice for c2" in result.content[0].text


async def test_depth_headers_reach_the_tool_and_are_recorded(live):
    base, sink = live
    async with _client(
        base, **{HEADER_AGENT_DEPTH: "2", HEADER_CONVERSATION_ID: "conv-e2e"}
    ) as client:
        result = await client.call_tool("echo_depth", {})
    assert "2|conv-e2e" in result.content[0].text

    row = next(r for r in sink.rows() if r["name"] == "echo_depth")
    assert row["depth"] == 2
    assert row["conversation_id"] == "conv-e2e"


async def test_the_advertised_tool_list_carries_the_flags_a_caller_needs(live):
    base, _ = live
    async with _client(base) as client:
        listed = await client.list_tools()
    by_name = {t.name: t for t in listed.tools}
    dumped = by_name["create_invoice"].model_dump(
        by_alias=True, mode="json", exclude_none=True
    )
    assert dumped["_meta"]["dev.johnny.agent-mcp/requiresConfirmation"] is True
    assert (
        by_name["get_balance"]
        .model_dump(by_alias=True, mode="json", exclude_none=True)["annotations"][
            "readOnlyHint"
        ]
        is True
    )


# --- the usage endpoint the spawner reads ---


async def test_the_usage_endpoint_requires_auth_and_summarises_real_calls(live):
    base, _ = live
    async with _client(base) as client:
        await client.call_tool("get_balance", {"account": "x"})

    async with httpx2.AsyncClient(timeout=10) as http:
        assert (await http.get(f"{base}/agent/usage")).status_code == 401
        response = await http.get(
            f"{base}/agent/usage", headers={"Authorization": f"Bearer {TOKEN}"}
        )
    assert response.status_code == 200
    body = response.json()
    assert body["app"] == "finance"
    assert body["totals"]["calls"] >= 1
    assert any(row["name"] == "get_balance" for row in body["by_tool"])
    assert any(row["caller"] == "amber" for row in body["by_caller"])
