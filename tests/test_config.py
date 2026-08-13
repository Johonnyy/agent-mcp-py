"""Tests for settings — env prefix, the auth_keys_env trap, and the no-models rule."""

import pathlib
import re

import pytest

import agent_mcp.config as config_module
from agent_mcp.config import AgentMCPSettings
from agent_mcp.depth import MAX_AGENT_DEPTH


@pytest.fixture
def fresh_settings():
    """Reset the settings singleton on both sides so env changes take effect and
    don't leak into other tests."""
    config_module.get_settings.cache_clear()
    yield
    config_module.get_settings.cache_clear()


def _settings(**over):
    return AgentMCPSettings(_env_file=None, **over)


def test_values_come_from_the_agent_mcp_env_prefix(monkeypatch, fresh_settings):
    monkeypatch.setenv("AGENT_MCP_APP_NAME", "school")
    monkeypatch.setenv("AGENT_MCP_ALLOW_ANONYMOUS", "true")
    settings = config_module.get_settings()
    assert settings.app_name == "school"
    assert settings.allow_anonymous is True


def test_the_depth_cap_is_not_a_duplicated_literal():
    """One definition of the limit, in depth.py, so agent_runtime's client-side
    pre-check can never disagree with our server-side enforcement."""
    assert _settings().max_agent_depth == MAX_AGENT_DEPTH


def test_keys_are_read_from_the_plain_field_by_default():
    assert _settings(keys="amber:abc").resolved_keys() == "amber:abc"


def test_auth_keys_env_overrides_the_plain_field(monkeypatch):
    monkeypatch.setenv("FINANCE_MCP_KEYS", "spawner:xyz")
    settings = _settings(keys="amber:abc", auth_keys_env="FINANCE_MCP_KEYS")
    assert settings.resolved_keys() == "spawner:xyz"


def test_auth_keys_env_falls_back_when_the_named_variable_is_empty(monkeypatch):
    """The documented trap: pydantic-settings loads .env into the model but does not
    export it to os.environ, so a key placed only in .env would otherwise vanish and
    — fail-closed — reject everything for no visible reason."""
    monkeypatch.delenv("FINANCE_MCP_KEYS", raising=False)
    settings = _settings(keys="amber:abc", auth_keys_env="FINANCE_MCP_KEYS")
    assert settings.resolved_keys() == "amber:abc"


def test_defaults_are_safe(fresh_settings):
    settings = _settings()
    assert settings.allow_anonymous is False  # fails closed
    assert settings.sync_store_url == ""      # no registration by default
    assert settings.usage_enabled is True


def test_the_package_hardcodes_no_model_names():
    """If a resource ever needs an LLM it takes a callable from the host app. This
    library must stay usable by an app that has no model at all."""
    package = pathlib.Path(config_module.__file__).parent
    # Actual model identifiers and SDK imports — not the word "OpenAI" appearing in
    # prose, which legitimately does (schema.py explains the OpenAI-compatible tool
    # format it has to satisfy).
    model_id = re.compile(
        r"claude-[a-z0-9]|gpt-[0-9]|gemini-[0-9]|llama-[0-9]|o[0-9]-mini",
        re.IGNORECASE,
    )
    sdk_import = re.compile(
        r"^\s*(?:import|from)\s+(?:openai|anthropic|litellm|google\.generativeai)\b"
    )
    offenders = []
    for path in package.rglob("*.py"):
        for number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if model_id.search(line) or sdk_import.match(line):
                offenders.append(f"{path.name}:{number}: {line.strip()}")
    assert not offenders, "model/provider references found:\n" + "\n".join(offenders)


def test_no_llm_sdk_is_a_dependency():
    root = pathlib.Path(config_module.__file__).parents[2]
    text = (root / "pyproject.toml").read_text(encoding="utf-8")
    deps = text.split("dependencies = [", 1)[1].split("]", 1)[0]
    for banned in ("openai", "anthropic", "litellm"):
        assert banned not in deps, f"{banned} must not be a dependency"
