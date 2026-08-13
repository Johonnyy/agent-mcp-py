"""Tests for sync-store registration. The store is a fake; no network."""

import asyncio

import pytest

from agent_mcp import sync_client
from agent_mcp.decorators import ResourcePolicy, ToolPolicy


class _FakeResponse:
    def __init__(self, status: int = 200):
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class _FakeAsyncClient:
    """Stands in for httpx2.AsyncClient and records the outgoing request."""

    def __init__(self, capture: dict, *, boom: Exception | None = None, status: int = 200, **kwargs):
        self._capture = capture
        self._boom = boom
        self._status = status
        capture.setdefault("posts", [])
        capture["init"] = kwargs

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, url, json=None, headers=None):
        if self._boom:
            raise self._boom
        self._capture["posts"].append(
            {"url": url, "json": json, "headers": headers or {}}
        )
        return _FakeResponse(self._status)


def _descriptor(**over):
    base = dict(
        app_name="finance",
        version="0.1.0",
        base_url="https://finance.example",
        tools=[
            ToolPolicy("get_balance", read_only=True),
            ToolPolicy("create_invoice", requires_confirmation=True),
        ],
        resources=[ResourcePolicy("finance://summary", "summary", False)],
        descriptions={"get_balance": "Balance for an account."},
    )
    base.update(over)
    return sync_client.build_descriptor(**base)


def test_the_descriptor_carries_the_flags_a_caller_needs():
    d = _descriptor()
    assert d["name"] == "finance"
    assert d["mcp_path"] == "/mcp"
    by_name = {t["name"]: t for t in d["tools"]}
    assert by_name["get_balance"]["read_only"] is True
    assert by_name["get_balance"]["description"] == "Balance for an account."
    assert by_name["create_invoice"]["requires_confirmation"] is True
    assert d["resources"] == [
        {"uri": "finance://summary", "name": "summary", "templated": False}
    ]


def test_the_descriptor_strips_a_trailing_slash_from_the_base_url():
    assert _descriptor(base_url="https://x/")["base_url"] == "https://x"


async def test_register_posts_the_descriptor_with_the_bearer_token(monkeypatch):
    capture: dict = {}
    monkeypatch.setattr(
        sync_client.httpx2, "AsyncClient", lambda **kw: _FakeAsyncClient(capture, **kw)
    )
    ok = await sync_client.register(
        sync_store_url="https://store/", descriptor=_descriptor(), token="secret"
    )
    assert ok is True
    post = capture["posts"][0]
    assert post["url"] == "https://store/servers"
    assert post["headers"]["Authorization"] == "Bearer secret"
    assert post["json"]["name"] == "finance"


async def test_an_unreachable_store_returns_false_and_does_not_raise(monkeypatch):
    """The founding principle: an app runs standalone. A dead registry costs
    discoverability and nothing else."""
    monkeypatch.setattr(
        sync_client.httpx2,
        "AsyncClient",
        lambda **kw: _FakeAsyncClient({}, boom=OSError("connection refused"), **kw),
    )
    assert await sync_client.register(
        sync_store_url="https://store", descriptor=_descriptor()
    ) is False


async def test_an_error_status_returns_false_and_does_not_raise(monkeypatch):
    monkeypatch.setattr(
        sync_client.httpx2,
        "AsyncClient",
        lambda **kw: _FakeAsyncClient({}, status=500, **kw),
    )
    assert await sync_client.register(
        sync_store_url="https://store", descriptor=_descriptor()
    ) is False


async def test_no_store_url_makes_no_request(monkeypatch):
    def explode(**kwargs):
        raise AssertionError("must not build a client without a store URL")

    monkeypatch.setattr(sync_client.httpx2, "AsyncClient", explode)
    assert await sync_client.register(sync_store_url="", descriptor=_descriptor()) is False


async def test_no_public_url_makes_no_request(monkeypatch):
    """Registering an unreachable address is worse than being absent: every peer
    that resolved the name would then fail to connect."""

    def explode(**kwargs):
        raise AssertionError("must not register without a public URL")

    monkeypatch.setattr(sync_client.httpx2, "AsyncClient", explode)
    assert await sync_client.register(
        sync_store_url="https://store", descriptor=_descriptor(base_url="")
    ) is False


async def test_the_heartbeat_registers_repeatedly_then_cancels_cleanly(monkeypatch):
    calls = {"register": 0, "tick": 0, "sleeps": 0}

    async def fake_sleep(seconds):
        calls["sleeps"] += 1
        if calls["sleeps"] > 3:
            raise asyncio.CancelledError
        return None

    monkeypatch.setattr(sync_client.asyncio, "sleep", fake_sleep)

    async def register_fn():
        calls["register"] += 1
        return True

    def on_tick():
        calls["tick"] += 1

    with pytest.raises(asyncio.CancelledError):
        await sync_client.heartbeat_loop(
            interval_s=1.0, register_fn=register_fn, on_tick=on_tick
        )
    assert calls["register"] == 3
    assert calls["tick"] == 3


async def test_a_failing_tick_does_not_kill_the_heartbeat(monkeypatch):
    calls = {"sleeps": 0, "register": 0}

    async def fake_sleep(seconds):
        calls["sleeps"] += 1
        if calls["sleeps"] > 3:
            raise asyncio.CancelledError

    monkeypatch.setattr(sync_client.asyncio, "sleep", fake_sleep)

    async def register_fn():
        calls["register"] += 1
        raise RuntimeError("store exploded")

    with pytest.raises(asyncio.CancelledError):
        await sync_client.heartbeat_loop(interval_s=1.0, register_fn=register_fn)
    assert calls["register"] == 3  # kept going despite every attempt failing


async def test_a_non_positive_interval_disables_the_heartbeat():
    async def register_fn():
        raise AssertionError("should never be called")

    await sync_client.heartbeat_loop(interval_s=0, register_fn=register_fn)
