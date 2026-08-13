"""Tests for bearer verification — known-good/known-bad tokens, no real auth infra."""

import pytest

from agent_mcp.auth import (
    CALLER_ANONYMOUS,
    extract_bearer,
    fingerprint,
    header_value,
    parse_keys,
    verify_bearer,
    verify_token,
)


def test_plain_comma_separated_tokens_get_a_fingerprint_identity():
    keys = parse_keys("abc123,def456")
    assert set(keys) == {"abc123", "def456"}
    assert keys["abc123"].startswith("sha256:")
    assert len(keys["abc123"]) == len("sha256:") + 8


def test_named_tokens_keep_their_name_for_attribution():
    keys = parse_keys("amber:abc123,spawner:def456")
    assert keys == {"abc123": "amber", "def456": "spawner"}


def test_blank_entries_and_whitespace_are_tolerated():
    assert parse_keys("  amber:abc  , , def  ,") == {
        "abc": "amber",
        "def": fingerprint("def"),
    }
    assert parse_keys("") == {}
    assert parse_keys(None) == {}


@pytest.mark.parametrize("header,expected", [
    ("Bearer abc123", "abc123"),
    ("bearer abc123", "abc123"),
    ("BEARER  abc123  ", "abc123"),
    ("abc123", ""),          # a bare token is not a credential
    ("Basic abc123", ""),
    ("", ""),
    (None, ""),
])
def test_bearer_extraction_is_tolerant_but_requires_the_scheme(header, expected):
    assert extract_bearer(header) == expected


def test_a_listed_token_is_accepted_and_names_its_caller():
    result = verify_bearer("Bearer abc123", parse_keys("amber:abc123"))
    assert result.ok is True
    assert result.caller == "amber"


def test_an_unlisted_token_is_rejected():
    result = verify_bearer("Bearer wrong", parse_keys("amber:abc123"))
    assert result.ok is False
    assert "unrecognised" in result.reason


def test_a_missing_authorization_header_is_rejected():
    result = verify_bearer(None, parse_keys("amber:abc123"))
    assert result.ok is False
    assert "missing or malformed" in result.reason


def test_no_configured_keys_fails_closed():
    result = verify_bearer("Bearer anything", {})
    assert result.ok is False
    assert "no bearer keys configured" in result.reason


def test_allow_anonymous_is_the_explicit_opt_out():
    result = verify_bearer(None, {}, allow_anonymous=True)
    assert result.ok is True
    assert result.caller == CALLER_ANONYMOUS


def test_allow_anonymous_admits_an_unrecognised_token_when_no_keys_are_set():
    result = verify_bearer("Bearer whatever", {}, allow_anonymous=True)
    assert result.ok is True
    assert result.caller == CALLER_ANONYMOUS


def test_verify_token_checks_every_key_without_short_circuiting():
    """The loop must not break on the first match — a break would let the number of
    comparisons leak which prefix was right."""
    checked: list[str] = []

    class CountingKeys(dict):
        def items(self):
            for key, value in super().items():
                checked.append(key)
                yield key, value

    keys = CountingKeys({"aaa": "first", "bbb": "second", "ccc": "third"})
    assert verify_token("aaa", keys) == "first"
    assert checked == ["aaa", "bbb", "ccc"]


def test_header_lookup_is_case_insensitive():
    assert header_value({"Authorization": "Bearer x"}, "authorization") == "Bearer x"
    assert header_value({"authorization": "Bearer x"}, "Authorization") == "Bearer x"
    assert header_value(None, "authorization") is None
    assert header_value({}, "authorization") is None


def test_a_fingerprint_is_stable_and_does_not_leak_the_token():
    assert fingerprint("secret") == fingerprint("secret")
    assert "secret" not in fingerprint("secret")
