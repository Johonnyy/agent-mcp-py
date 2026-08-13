"""Tests for the signature surgery — the highest-value file in the suite.

Every pitfall in `decorators.py` fails *silently* if it regresses: the tool keeps
working while the guards quietly stop running. These tests are what turn that into
a red suite.
"""

import json
from typing import Optional

import pytest
from mcp import Client
from mcp.server.mcpserver import Context

from agent_mcp import AgentMCPServer, AgentMCPSettings
from agent_mcp.decorators import (
    CTX_PARAM,
    META_READ_ONLY,
    META_REQUIRES_CONFIRMATION,
    Guard,
    ToolPolicy,
    build_tool_wrapper,
)
from agent_mcp.usage_log import NullUsageSink


class Invoice:
    """A type defined in *this* module — the __globals__ pitfall needs one."""

    def __init__(self, amount: float) -> None:
        self.amount = amount


def _settings(**over):
    base = dict(
        app_name="finance",
        allow_anonymous=True,
        usage_enabled=False,
        usage_db_path=":memory:",
    )
    base.update(over)
    return AgentMCPSettings(_env_file=None, **base)


def _server(**over):
    return AgentMCPServer(
        app_name="finance",
        version="0.1.0",
        settings=_settings(**over),
        usage_sink=NullUsageSink(),
    )


def _guard():
    return Guard(
        app_name="finance",
        keys={},
        allow_anonymous=True,
        max_depth=5,
        sink=NullUsageSink(),
        usage_enabled=False,
    )


async def _schema_for(server, name):
    async with Client(server.mcp) as client:
        listed = await client.list_tools()
    return next(t for t in listed.tools if t.name == name)


async def test_the_injected_context_never_appears_in_the_advertised_schema():
    server = _server()

    @server.tool(read_only=True)
    def get_balance(account: str) -> float:
        """Balance."""
        return 1.0

    tool = await _schema_for(server, "get_balance")
    assert CTX_PARAM not in json.dumps(tool.input_schema)
    assert set(tool.input_schema["properties"]) == {"account"}


async def test_user_defaults_and_annotations_survive_the_wrapper():
    server = _server()

    @server.tool()
    def search(query: str, limit: int = 20, verbose: bool = False) -> str:
        """Search."""
        return query

    schema = (await _schema_for(server, "search")).input_schema
    assert schema["properties"]["limit"]["default"] == 20
    assert schema["properties"]["limit"]["type"] == "integer"
    assert schema["properties"]["verbose"]["default"] is False
    assert schema["required"] == ["query"]


async def test_an_annotation_naming_a_type_from_the_authors_module_resolves():
    """The __globals__ pitfall, tested head on.

    The wrapper's globals are agent_mcp.decorators', not the author's. If we let the
    SDK evaluate the string annotation ``-> Invoice`` it would NameError — or worse,
    find_context_parameter would swallow it and silently disable every guard.
    """
    server = _server()

    @server.tool()
    def make_invoice(customer: str, amount: float = 1.0) -> Invoice:
        """Create an invoice."""
        return Invoice(amount)

    schema = (await _schema_for(server, "make_invoice")).input_schema
    assert set(schema["properties"]) == {"customer", "amount"}
    assert CTX_PARAM not in json.dumps(schema)


async def test_an_optional_annotation_survives():
    server = _server()

    @server.tool()
    def maybe(limit: Optional[int] = None) -> str:
        """Optional arg."""
        return str(limit)

    schema = (await _schema_for(server, "maybe")).input_schema
    assert "anyOf" in schema["properties"]["limit"]


async def test_a_sync_tool_runs():
    server = _server()

    @server.tool()
    def sync_tool(x: str) -> str:
        """Sync."""
        return f"sync:{x}"

    async with Client(server.mcp) as client:
        result = await client.call_tool("sync_tool", {"x": "a"})
    assert result.is_error is False
    assert "sync:a" in result.content[0].text


async def test_an_async_tool_runs():
    server = _server()

    @server.tool()
    async def async_tool(x: str) -> str:
        """Async."""
        return f"async:{x}"

    async with Client(server.mcp) as client:
        result = await client.call_tool("async_tool", {"x": "a"})
    assert result.is_error is False
    assert "async:a" in result.content[0].text


async def test_a_user_declared_context_is_honoured_without_a_second_injection():
    server = _server()

    @server.tool()
    def with_ctx(x: str, ctx: Context) -> str:
        """Takes its own context."""
        return f"{x}:{ctx is not None}"

    schema = (await _schema_for(server, "with_ctx")).input_schema
    assert set(schema["properties"]) == {"x"}
    assert CTX_PARAM not in json.dumps(schema)

    async with Client(server.mcp) as client:
        result = await client.call_tool("with_ctx", {"x": "a"})
    assert "a:True" in result.content[0].text


def test_declaring_the_reserved_parameter_name_is_refused():
    guard = _guard()

    def bad(agent_mcp_ctx: str) -> str:
        """Collides with the injected parameter."""
        return agent_mcp_ctx

    with pytest.raises(ValueError, match="reserved"):
        build_tool_wrapper(bad, policy=ToolPolicy("bad"), guard=guard)


def test_an_unresolvable_annotation_is_a_clear_registration_error():
    guard = _guard()
    ns: dict = {}
    exec(
        "def broken(x: 'NoSuchTypeAnywhere') -> str:\n"
        "    '''doc'''\n"
        "    return x\n",
        ns,
    )
    with pytest.raises(ValueError, match="Cannot resolve type annotations"):
        build_tool_wrapper(ns["broken"], policy=ToolPolicy("broken"), guard=guard)


def test_a_duplicate_tool_name_is_refused():
    server = _server()

    @server.tool()
    def dup() -> str:
        """First."""
        return "a"

    with pytest.raises(ValueError, match="already registered"):

        @server.tool(name="dup")
        def dup2() -> str:
            """Second."""
            return "b"


def test_an_invalid_tool_name_is_refused_at_registration():
    server = _server()

    with pytest.raises(ValueError, match="snake_case"):

        @server.tool(name="getBalance")
        def bad_name() -> str:
            """Doc."""
            return "x"


def test_the_decorator_returns_the_function_unwrapped():
    """So the app's own HTTP handler calls the same object the model calls, rather
    than a parallel code path that drifts from it."""
    server = _server()

    def original(x: str) -> str:
        """Doc."""
        return f"plain:{x}"

    returned = server.tool()(original)
    assert returned is original
    assert returned("a") == "plain:a"


async def test_read_only_becomes_a_standard_annotation_on_the_wire():
    server = _server()

    @server.tool(read_only=True)
    def query() -> str:
        """Doc."""
        return "x"

    tool = await _schema_for(server, "query")
    dumped = tool.model_dump(by_alias=True, mode="json", exclude_none=True)
    assert dumped["annotations"]["readOnlyHint"] is True


async def test_requires_confirmation_rides_in_meta_because_annotations_drop_extras():
    """Verified against the SDK: ToolAnnotations ignores unknown fields silently, so
    a custom requiresConfirmation there would vanish without an error."""
    server = _server()

    @server.tool(read_only=False, requires_confirmation=True)
    def risky(x: str) -> str:
        """Doc."""
        return x

    tool = await _schema_for(server, "risky")
    dumped = tool.model_dump(by_alias=True, mode="json", exclude_none=True)
    assert dumped["_meta"][META_REQUIRES_CONFIRMATION] is True
    assert dumped["_meta"][META_READ_ONLY] is False


async def test_read_only_is_duplicated_into_meta_so_callers_read_one_dict():
    server = _server()

    @server.tool(read_only=True)
    def query() -> str:
        """Doc."""
        return "x"

    tool = await _schema_for(server, "query")
    dumped = tool.model_dump(by_alias=True, mode="json", exclude_none=True)
    assert dumped["annotations"]["readOnlyHint"] is True
    assert dumped["_meta"][META_READ_ONLY] is True


async def test_a_confirmation_tool_says_so_in_its_description():
    """A plain LLM client never reads _meta; the description is the only channel the
    model itself reliably sees."""
    server = _server()

    @server.tool(requires_confirmation=True)
    def risky() -> str:
        """Does something dangerous."""
        return "x"

    tool = await _schema_for(server, "risky")
    assert "Requires confirmation" in tool.description
    assert "Does something dangerous" in tool.description


async def test_a_nested_model_argument_is_flattened_to_a_ref_free_schema():
    from pydantic import BaseModel

    class Address(BaseModel):
        street: str
        city: str = "nowhere"

    server = _server()

    @server.tool()
    def ship(addr: Address) -> str:
        """Ship to an address."""
        return addr.city

    schema = (await _schema_for(server, "ship")).input_schema
    text = json.dumps(schema)
    assert "$ref" not in text
    assert "$defs" not in text
    # The definition was inlined, not dropped.
    assert schema["properties"]["addr"]["properties"]["city"]["default"] == "nowhere"


def test_a_recursive_model_argument_is_refused_and_leaves_no_half_tool():
    from pydantic import BaseModel

    class Node(BaseModel):
        name: str
        child: "Node | None" = None

    Node.model_rebuild()
    server = _server()

    with pytest.raises(ValueError, match="recursive"):

        @server.tool()
        def walk(node: Node) -> str:
            """Doc."""
            return node.name

    assert "walk" not in server.mcp._tool_manager._tools
    assert not any(t.name == "walk" for t in server.tool_policies())


def test_a_static_resource_with_parameters_raises_our_teaching_error():
    server = _server()

    with pytest.raises(ValueError) as excinfo:

        @server.resource("finance://transactions/recent")
        def recent(limit: int = 20) -> list[dict]:
            """Doc."""
            return []

    message = str(excinfo.value)
    assert "{?limit}" in message  # spells out the fix
    assert "static URI" in message


def test_a_templated_resource_registers():
    server = _server()

    @server.resource("finance://transactions/recent{?limit}")
    def recent(limit: int = 20) -> list[dict]:
        """Doc."""
        return []

    policies = {r.uri: r for r in server.resource_policies()}
    assert policies["finance://transactions/recent{?limit}"].templated is True


def test_a_static_resource_without_parameters_registers():
    server = _server()

    @server.resource("finance://summary")
    def summary() -> dict:
        """Doc."""
        return {}

    assert server.resource_policies()[0].templated is False


def test_a_duplicate_resource_uri_is_refused():
    server = _server()

    @server.resource("finance://summary")
    def summary() -> dict:
        """Doc."""
        return {}

    with pytest.raises(ValueError, match="already registered"):

        @server.resource("finance://summary")
        def summary2() -> dict:
            """Doc."""
            return {}
