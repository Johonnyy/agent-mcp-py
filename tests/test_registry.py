"""Tests for peer resolution. The sync store is faked; no network."""

import pytest

from agent_mcp import registry as registry_mod
from agent_mcp.registry import (
    MCP_MOUNT_PATH,
    PeerRecord,
    PeerRegistry,
    load_static_peers,
    mcp_url,
)


class _FakeResponse:
    def __init__(self, payload, status: int = 200):
        self._payload = payload
        self.status_code = status

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class _FakeAsyncClient:
    """Stands in for httpx2.AsyncClient; records the request for assertions."""

    def __init__(self, payload, capture: dict, *, boom: Exception | None = None, **kwargs):
        self._payload = payload
        self._capture = capture
        self._boom = boom
        capture["init"] = kwargs

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def get(self, url, headers=None):
        if self._boom:
            raise self._boom
        self._capture["url"] = url
        self._capture["headers"] = headers or {}
        return _FakeResponse(self._payload)


def test_static_peers_parse_with_whitespace_and_trailing_commas():
    peers = load_static_peers(" finance=https://a , school=https://b ,", token="t")
    assert set(peers) == {"finance", "school"}
    assert peers["finance"].base_url == "https://a"
    assert peers["finance"].token == "t"


def test_malformed_static_entries_are_skipped_not_fatal():
    peers = load_static_peers("finance=https://a,broken,=https://c,name=")
    assert set(peers) == {"finance"}


def test_empty_static_spec_yields_nothing():
    assert load_static_peers("") == {}
    assert load_static_peers(None) == {}


def test_resolve_returns_none_for_an_unknown_peer():
    assert PeerRegistry().resolve("nope") is None


def test_static_config_beats_discovery():
    """A local override during an incident must not be stomped by the registry."""
    reg = PeerRegistry(
        static={"finance": PeerRecord("finance", "https://local")},
        discovered={"finance": PeerRecord("finance", "https://stale")},
    )
    assert reg.resolve("finance").base_url == "https://local"


def test_discovered_peers_are_used_when_there_is_no_static_override():
    reg = PeerRegistry(discovered={"school": PeerRecord("school", "https://s")})
    assert reg.resolve("school").base_url == "https://s"
    assert reg.known() == ["school"]


@pytest.mark.parametrize("base", ["https://x", "https://x/", "https://x///"])
def test_mcp_url_appends_the_mount_path_exactly_once(base):
    assert mcp_url(PeerRecord("n", base)) == f"https://x{MCP_MOUNT_PATH}/"


def test_mcp_url_ends_with_a_slash_to_dodge_the_307_redirect():
    # A mounted MCP app answers /mcp with a 307 to /mcp/; a client that does not
    # re-POST on redirect breaks silently. Always address the canonical form.
    assert mcp_url(PeerRecord("n", "https://x")).endswith("/")


async def test_refresh_loads_records_from_the_sync_store(monkeypatch):
    capture: dict = {}
    payload = {
        "servers": [
            {"name": "finance", "base_url": "https://f", "token": "tk",
             "version": "1.0", "tools": ["get_balance"]},
            {"name": "bad"},  # no base_url — skipped
        ]
    }
    monkeypatch.setattr(
        registry_mod.httpx2,
        "AsyncClient",
        lambda **kw: _FakeAsyncClient(payload, capture, **kw),
    )
    reg = PeerRegistry()
    assert await reg.refresh("https://store/", token="secret") == 1
    assert capture["url"] == "https://store/servers"
    assert capture["headers"]["Authorization"] == "Bearer secret"
    record = reg.resolve("finance")
    assert record.base_url == "https://f"
    assert record.tools == ("get_balance",)


async def test_refresh_against_an_unreachable_store_returns_zero_and_does_not_raise(
    monkeypatch,
):
    monkeypatch.setattr(
        registry_mod.httpx2,
        "AsyncClient",
        lambda **kw: _FakeAsyncClient(None, {}, boom=OSError("connection refused"), **kw),
    )
    reg = PeerRegistry(static={"a": PeerRecord("a", "https://a")})
    assert await reg.refresh("https://store") == 0
    # Whatever it already knew survives.
    assert reg.resolve("a") is not None


async def test_refresh_with_no_store_url_makes_no_request(monkeypatch):
    def explode(**kwargs):
        raise AssertionError("refresh must not build a client without a store URL")

    monkeypatch.setattr(registry_mod.httpx2, "AsyncClient", explode)
    assert await PeerRegistry().refresh("") == 0


async def test_refresh_tolerates_an_unexpected_payload_shape(monkeypatch):
    monkeypatch.setattr(
        registry_mod.httpx2,
        "AsyncClient",
        lambda **kw: _FakeAsyncClient({"servers": "not a list"}, {}, **kw),
    )
    assert await PeerRegistry().refresh("https://store") == 0


def test_resolve_performs_no_io_at_all(monkeypatch):
    """The hot path: agent_runtime calls resolve() per tool dispatch and must never
    pay for a network round trip."""

    def explode(*args, **kwargs):
        raise AssertionError("resolve() must not touch the network")

    monkeypatch.setattr(registry_mod.httpx2, "AsyncClient", explode)
    reg = PeerRegistry(static={"finance": PeerRecord("finance", "https://f")})
    assert reg.resolve("finance").base_url == "https://f"
    assert reg.resolve("missing") is None
