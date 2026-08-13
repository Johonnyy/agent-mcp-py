"""Tool-name rules and JSON-Schema flattening — making our tools wire-safe.

Two constraints from downstream, both enforced at *registration* time so a bad tool
fails loudly at import rather than mysteriously at call time in production.

**Names.** ``agent_runtime`` aggregates tools from many servers into one
OpenAI-compatible ``tools=[...]`` list and namespaces them ``<server>__<tool>`` to
break collisions. That means our names must be snake_case, must not themselves
contain a double underscore (it is the separator), and must leave room inside the
provider limit of 64 characters — hence a cap of 40 here.

**Schemas.** The MCP SDK builds a tool's input schema by handing the function
signature to pydantic, and pydantic emits ``$defs``/``$ref`` for any nested model or
enum. Verified against the installed SDK: a tool taking a nested model produces
``{"addr": {"$ref": "#/$defs/Address"}}``. OpenRouter's passthrough to several
providers rejects ``$ref``, ``$defs``, and root-level composition outright, and
there is no SDK setting that turns this off. So we inline the references ourselves.

The split between rewriting and rejecting is deliberate:

* **Rewrite silently** when the result is semantically identical — inlining a
  ``$ref`` changes nothing about what the schema accepts, so there is nothing for an
  author to decide.
* **Reject loudly** when it is not — a recursive model cannot be inlined at all
  (the expansion does not terminate), and quietly truncating it would ship a schema
  that lies about the tool. The error message names the offending definition and
  says what to do instead.

Note that ``anyOf`` *inside* a property is fine and must survive untouched — it is
what ``int | None`` compiles to, and it appears in almost every optional argument.
Only root-level composition is banned.
"""

from __future__ import annotations

import re
from copy import deepcopy
from typing import Any

#: Names must start with a letter, be snake_case, and fit in 40 characters. The
#: provider limit is 64; the remaining 24 are the ``<server>__`` prefix that
#: ``agent_runtime`` adds when it merges tools from several servers.
TOOL_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,39}$")

#: The provider ceiling for a function name in an OpenAI-compatible ``tools=[...]``
#: entry. Everything below is budgeted against it.
MAX_WIRE_NAME_LEN = 64
MAX_TOOL_NAME_LEN = 40
#: 64 - len("__") - 40. An app name is itself half of every namespaced tool name
#: ``<app>__<tool>``, so it needs a tighter cap than a tool name — otherwise two
#: individually legal names combine into an illegal one, and the failure lands in
#: whichever agent aggregated them rather than in the app that chose the name.
MAX_APP_NAME_LEN = MAX_WIRE_NAME_LEN - 2 - MAX_TOOL_NAME_LEN

_ROOT_COMPOSITION = ("oneOf", "anyOf", "allOf", "not")

#: Keys whose values are a schema, a list of schemas, or a mapping of schemas.
_SCHEMA_VALUE_KEYS = ("additionalProperties", "items", "not", "propertyNames")
_SCHEMA_LIST_KEYS = ("anyOf", "oneOf", "allOf", "prefixItems")
_SCHEMA_MAP_KEYS = ("properties", "patternProperties", "$defs", "definitions")


class SchemaTooComplex(ValueError):
    """A schema cannot be made wire-safe without changing its meaning."""


def validate_tool_name(name: str) -> None:
    """Raise ``ValueError`` unless ``name`` is safe to namespace and ship."""
    if not name:
        raise ValueError("tool name must not be empty")
    if "__" in name:
        raise ValueError(
            f"tool name {name!r} contains '__', which agent_runtime uses as the "
            "<server>__<tool> namespace separator; use single underscores"
        )
    if name.endswith("_"):
        raise ValueError(f"tool name {name!r} must not end with an underscore")
    if len(name) > MAX_TOOL_NAME_LEN:
        raise ValueError(
            f"tool name {name!r} is {len(name)} characters; the limit is "
            f"{MAX_TOOL_NAME_LEN} so that '<server>__' namespacing still fits "
            "inside the provider limit of 64"
        )
    if not TOOL_NAME_RE.match(name):
        raise ValueError(
            f"tool name {name!r} must be snake_case: lowercase letters, digits and "
            "underscores, starting with a letter"
        )


def validate_app_name(name: str) -> None:
    """Raise ``ValueError`` unless ``name`` is safe as a tool-name namespace.

    Same character rules as a tool, but shorter: the app name is prefixed onto every
    one of its tools as ``<app>__<tool>`` when an agent aggregates several servers,
    so it has to fit in the budget left by the longest legal tool name.
    """
    if not name:
        raise ValueError("app name must not be empty")
    if len(name) > MAX_APP_NAME_LEN:
        raise ValueError(
            f"app name {name!r} is {len(name)} characters; the limit is "
            f"{MAX_APP_NAME_LEN} so that '<app>__<tool>' stays inside the "
            f"{MAX_WIRE_NAME_LEN}-character provider limit even for a "
            f"{MAX_TOOL_NAME_LEN}-character tool name"
        )
    validate_tool_name(name)


def namespaced_name(app_name: str, tool_name: str) -> str:
    """How an aggregating agent will address one of our tools."""
    return f"{app_name}__{tool_name}"


def _resolve_pointer(root: dict[str, Any], ref: str) -> dict[str, Any]:
    """Resolve a local JSON pointer like ``#/$defs/Address``."""
    if not ref.startswith("#/"):
        raise SchemaTooComplex(
            f"only local references are supported, got {ref!r}; remote $ref cannot "
            "be inlined and is rejected by several model providers"
        )
    node: Any = root
    for part in ref[2:].split("/"):
        part = part.replace("~1", "/").replace("~0", "~")
        if not isinstance(node, dict) or part not in node:
            raise SchemaTooComplex(f"cannot resolve reference {ref!r}")
        node = node[part]
    if not isinstance(node, dict):
        raise SchemaTooComplex(f"reference {ref!r} does not point at an object")
    return node


def _inline(node: Any, root: dict[str, Any], seen: tuple[str, ...]) -> Any:
    """Recursively replace ``$ref`` with the definition it points at."""
    if isinstance(node, list):
        return [_inline(item, root, seen) for item in node]
    if not isinstance(node, dict):
        return node

    if "$ref" in node:
        ref = node["$ref"]
        if not isinstance(ref, str):
            raise SchemaTooComplex(f"$ref must be a string, got {type(ref).__name__}")
        if ref in seen:
            name = ref.rsplit("/", 1)[-1]
            raise SchemaTooComplex(
                f"model {name!r} is recursive (it refers to itself through {ref!r}), "
                "so its schema cannot be inlined. Model providers reject $ref, so "
                "this tool cannot be exposed as-is — flatten the model by hand, or "
                "accept a JSON string and parse it inside the tool."
            )
        target = _inline(_resolve_pointer(root, ref), root, (*seen, ref))
        # pydantic emits siblings next to a $ref for a documented nested model
        # ({"$ref": ..., "description": ...}). The sibling is the more specific
        # statement, so it wins over the same key in the target.
        siblings = {k: _inline(v, root, seen) for k, v in node.items() if k != "$ref"}
        if not isinstance(target, dict):
            return target
        return {**target, **siblings}

    out: dict[str, Any] = {}
    for key, value in node.items():
        if key in _SCHEMA_MAP_KEYS and isinstance(value, dict):
            out[key] = {k: _inline(v, root, seen) for k, v in value.items()}
        elif key in _SCHEMA_LIST_KEYS and isinstance(value, list):
            out[key] = [_inline(v, root, seen) for v in value]
        elif key in _SCHEMA_VALUE_KEYS:
            out[key] = _inline(value, root, seen)
        else:
            out[key] = value
    return out


def flatten_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of ``schema`` with every local ``$ref`` inlined.

    Also drops the now-unreferenced ``$defs``/``definitions`` blocks and ``$schema``,
    which are noise at best and a rejection at worst. Raises
    :class:`SchemaTooComplex` for recursive or unresolvable references.
    """
    if not isinstance(schema, dict):
        raise SchemaTooComplex(
            f"input schema must be an object, got {type(schema).__name__}"
        )

    root = deepcopy(schema)
    flattened = _inline(root, root, ())
    if not isinstance(flattened, dict):  # pragma: no cover — root is always a dict
        raise SchemaTooComplex("input schema did not flatten to an object")

    flattened.pop("$defs", None)
    flattened.pop("definitions", None)
    flattened.pop("$schema", None)
    return flattened


def assert_wire_safe(schema: dict[str, Any], *, what: str = "input schema") -> None:
    """Raise ``ValueError`` if ``schema`` would be rejected downstream."""
    if not isinstance(schema, dict):
        raise SchemaTooComplex(f"{what} must be an object")

    if schema.get("type") != "object":
        raise SchemaTooComplex(
            f"{what} must have type 'object' at the root, got "
            f"{schema.get('type')!r}"
        )

    for key in _ROOT_COMPOSITION:
        if key in schema:
            raise SchemaTooComplex(
                f"{what} uses root-level {key!r}, which several model providers "
                "reject. Express the alternatives as optional properties instead."
            )

    def walk(node: Any, path: str) -> None:
        if isinstance(node, list):
            for i, item in enumerate(node):
                walk(item, f"{path}[{i}]")
            return
        if not isinstance(node, dict):
            return
        if "$ref" in node:
            raise SchemaTooComplex(
                f"{what} still contains a $ref at {path or 'the root'} after "
                "flattening"
            )
        for key, value in node.items():
            walk(value, f"{path}.{key}" if path else key)

    walk(schema, "")


def has_refs(schema: Any) -> bool:
    """True if ``schema`` contains a ``$ref`` or ``$defs`` anywhere."""
    if isinstance(schema, dict):
        if "$ref" in schema or "$defs" in schema:
            return True
        return any(has_refs(v) for v in schema.values())
    if isinstance(schema, list):
        return any(has_refs(v) for v in schema)
    return False
