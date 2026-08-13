"""Tests for the depth leaf — header parsing, the cap, and next-hop threading.

No fakes: everything here is a pure function over a dict.
"""

import subprocess
import sys

import pytest

from agent_mcp.depth import (
    HEADER_AGENT_DEPTH,
    HEADER_CONFIRMED,
    HEADER_CONVERSATION_ID,
    MAX_AGENT_DEPTH,
    CallDepth,
    DepthExceeded,
    check_depth,
    is_confirmed,
    next_hop_headers,
    parse_depth,
)


def test_missing_headers_mean_depth_zero_and_a_fresh_conversation_id():
    parsed = parse_depth(None)
    assert parsed.depth == 0
    assert parsed.confirmed is False
    assert len(parsed.conversation_id) == 32  # uuid4().hex


def test_each_call_without_a_conversation_id_gets_a_distinct_one():
    assert parse_depth({}).conversation_id != parse_depth({}).conversation_id


def test_headers_are_read_case_insensitively():
    parsed = parse_depth(
        {"x-agent-depth": "2", "x-conversation-id": "abc", "x-confirmed": "true"}
    )
    assert (parsed.depth, parsed.conversation_id, parsed.confirmed) == (2, "abc", True)

    parsed = parse_depth(
        {HEADER_AGENT_DEPTH: "3", HEADER_CONVERSATION_ID: "def"}
    )
    assert (parsed.depth, parsed.conversation_id) == (3, "def")


@pytest.mark.parametrize("bad", ["abc", "-1", "1e3", "2.5", "  x  "])
def test_a_malformed_depth_header_is_a_hard_error_not_a_coerce_to_zero(bad):
    with pytest.raises(ValueError, match=HEADER_AGENT_DEPTH):
        parse_depth({HEADER_AGENT_DEPTH: bad})


def test_an_empty_depth_header_falls_back_to_zero():
    assert parse_depth({HEADER_AGENT_DEPTH: "   "}).depth == 0


def test_a_call_below_the_cap_is_allowed():
    check_depth(MAX_AGENT_DEPTH - 1)


def test_a_call_at_the_cap_is_refused_because_it_has_no_hops_left():
    with pytest.raises(DepthExceeded) as excinfo:
        check_depth(MAX_AGENT_DEPTH, conversation_id="conv-1")
    assert excinfo.value.limit == MAX_AGENT_DEPTH
    assert excinfo.value.depth == MAX_AGENT_DEPTH
    assert "conv-1" in str(excinfo.value)


def test_next_hop_increments_the_depth_and_keeps_the_conversation_id():
    headers = next_hop_headers(CallDepth(conversation_id="conv-9", depth=2))
    assert headers[HEADER_AGENT_DEPTH] == "3"
    assert headers[HEADER_CONVERSATION_ID] == "conv-9"
    assert HEADER_CONFIRMED not in headers


def test_next_hop_refuses_to_build_headers_for_an_over_cap_call():
    # A caller that forgets to pre-check still cannot emit the request.
    with pytest.raises(DepthExceeded):
        next_hop_headers(CallDepth(conversation_id="c", depth=MAX_AGENT_DEPTH))


def test_confirmation_does_not_propagate_downstream_unless_asked_for():
    scope = CallDepth(conversation_id="c", depth=0, confirmed=True)
    assert HEADER_CONFIRMED not in next_hop_headers(scope)
    assert next_hop_headers(scope, confirmed=True)[HEADER_CONFIRMED] == "true"


@pytest.mark.parametrize("value,expected", [
    ("true", True), ("TRUE", True), ("True", True), ("1", True),
    ("yes", False), ("y", False), ("0", False), ("", False), ("false", False),
])
def test_only_true_and_one_count_as_confirmation(value, expected):
    assert is_confirmed({HEADER_CONFIRMED: value}) is expected
    assert parse_depth({HEADER_CONFIRMED: value}).confirmed is expected


def test_is_confirmed_handles_absent_headers():
    assert is_confirmed(None) is False
    assert is_confirmed({}) is False


def test_depth_module_imports_no_third_party_packages():
    """The interop contract as an executable assertion.

    ``agent_runtime`` imports these constants to pre-check the cap client-side. If
    this module ever grows an import of mcp/pydantic/httpx2/starlette, that becomes
    a ~15-package install for the sake of an integer, and the contract is broken.
    """
    code = (
        "import sys; import agent_mcp.depth; "
        "heavy = sorted({m.split('.')[0] for m in sys.modules} & "
        "{'mcp','pydantic','httpx2','starlette','anyio','uvicorn'}); "
        "print(','.join(heavy))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "", (
        f"agent_mcp.depth pulled in third-party packages: {result.stdout.strip()}"
    )
