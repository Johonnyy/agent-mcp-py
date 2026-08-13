"""Tests for tool-name rules and JSON-Schema flattening. No fakes needed."""

import pytest

from agent_mcp.schema import (
    SchemaTooComplex,
    assert_wire_safe,
    flatten_schema,
    has_refs,
    validate_tool_name,
)


@pytest.mark.parametrize("name", [
    "get_balance", "list_tasks", "a", "x2", "add_task_to_the_list_now",
])
def test_good_tool_names_are_accepted(name):
    validate_tool_name(name)


@pytest.mark.parametrize("name,reason", [
    ("getBalance", "snake_case"),
    ("get__balance", "__"),
    ("_private", "snake_case"),
    ("2fast", "snake_case"),
    ("trailing_", "underscore"),
    ("", "empty"),
    ("x" * 41, "40"),
    ("has-dash", "snake_case"),
    ("Has_Upper", "snake_case"),
])
def test_bad_tool_names_are_rejected_with_a_useful_message(name, reason):
    with pytest.raises(ValueError, match=reason):
        validate_tool_name(name)


def test_a_single_ref_is_inlined():
    schema = {
        "type": "object",
        "$defs": {"Address": {"type": "object", "properties": {"city": {"type": "string"}}}},
        "properties": {"addr": {"$ref": "#/$defs/Address"}},
    }
    out = flatten_schema(schema)
    assert out["properties"]["addr"] == {
        "type": "object",
        "properties": {"city": {"type": "string"}},
    }
    assert "$defs" not in out


def test_nested_defs_are_inlined_recursively():
    schema = {
        "type": "object",
        "$defs": {
            "Inner": {"type": "object", "properties": {"x": {"type": "integer"}}},
            "Outer": {
                "type": "object",
                "properties": {"inner": {"$ref": "#/$defs/Inner"}},
            },
        },
        "properties": {"outer": {"$ref": "#/$defs/Outer"}},
    }
    out = flatten_schema(schema)
    assert out["properties"]["outer"]["properties"]["inner"]["properties"] == {
        "x": {"type": "integer"}
    }
    assert not has_refs(out)


def test_sibling_keys_next_to_a_ref_are_merged_and_win():
    schema = {
        "type": "object",
        "$defs": {"A": {"type": "object", "description": "generic"}},
        "properties": {
            "a": {"$ref": "#/$defs/A", "description": "the specific one"}
        },
    }
    out = flatten_schema(schema)
    assert out["properties"]["a"]["description"] == "the specific one"
    assert out["properties"]["a"]["type"] == "object"


def test_a_property_level_anyof_survives_untouched():
    # This is what `int | None` compiles to; rewriting it would break every
    # optional argument.
    schema = {
        "type": "object",
        "properties": {
            "limit": {"anyOf": [{"type": "integer"}, {"type": "null"}], "default": None}
        },
    }
    assert flatten_schema(schema)["properties"]["limit"]["anyOf"] == [
        {"type": "integer"},
        {"type": "null"},
    ]
    assert_wire_safe(flatten_schema(schema))


def test_refs_inside_a_list_of_schemas_are_inlined():
    schema = {
        "type": "object",
        "$defs": {"A": {"type": "string"}},
        "properties": {"x": {"anyOf": [{"$ref": "#/$defs/A"}, {"type": "null"}]}},
    }
    out = flatten_schema(schema)
    assert out["properties"]["x"]["anyOf"] == [{"type": "string"}, {"type": "null"}]


def test_refs_inside_array_items_are_inlined():
    schema = {
        "type": "object",
        "$defs": {"A": {"type": "string"}},
        "properties": {"xs": {"type": "array", "items": {"$ref": "#/$defs/A"}}},
    }
    assert flatten_schema(schema)["properties"]["xs"]["items"] == {"type": "string"}


def test_schema_noise_is_dropped():
    out = flatten_schema({"type": "object", "$schema": "https://json-schema.org/draft"})
    assert "$schema" not in out


def test_a_recursive_model_is_rejected_with_an_actionable_message():
    schema = {
        "type": "object",
        "$defs": {
            "Node": {
                "type": "object",
                "properties": {"child": {"$ref": "#/$defs/Node"}},
            }
        },
        "properties": {"root": {"$ref": "#/$defs/Node"}},
    }
    with pytest.raises(SchemaTooComplex) as excinfo:
        flatten_schema(schema)
    message = str(excinfo.value)
    assert "Node" in message
    assert "recursive" in message
    assert "JSON string" in message  # tells the author what to do instead


def test_an_unresolvable_reference_is_rejected():
    with pytest.raises(SchemaTooComplex, match="cannot resolve"):
        flatten_schema({"type": "object", "properties": {"a": {"$ref": "#/$defs/Nope"}}})


def test_a_remote_reference_is_rejected():
    with pytest.raises(SchemaTooComplex, match="local"):
        flatten_schema(
            {"type": "object", "properties": {"a": {"$ref": "https://x/schema.json"}}}
        )


@pytest.mark.parametrize("key", ["oneOf", "anyOf", "allOf", "not"])
def test_root_level_composition_is_rejected(key):
    with pytest.raises(SchemaTooComplex, match=key):
        assert_wire_safe({"type": "object", key: [{"type": "object"}]})


def test_a_non_object_root_is_rejected():
    with pytest.raises(SchemaTooComplex, match="object"):
        assert_wire_safe({"type": "array"})


def test_a_surviving_ref_is_caught_by_the_wire_safety_check():
    with pytest.raises(SchemaTooComplex, match=r"\$ref"):
        assert_wire_safe({"type": "object", "properties": {"a": {"$ref": "#/$defs/A"}}})


def test_has_refs_finds_refs_at_any_depth():
    assert has_refs({"a": [{"b": {"$ref": "#/x"}}]}) is True
    assert has_refs({"a": {"$defs": {}}}) is True
    assert has_refs({"a": [{"b": {"type": "string"}}]}) is False
