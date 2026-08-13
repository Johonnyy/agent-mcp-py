"""Central configuration for the MCP layer.

Every knob an app can turn lives here, read from environment variables (prefix
``AGENT_MCP_``) and an optional ``.env``. Constructor arguments to
:class:`~agent_mcp.server.AgentMCPServer` override these, so an app that already has
its own settings object can pass values straight in and never touch this.

**This module hardcodes no model names and depends on no LLM SDK.** If a resource
ever needs to call a model, the host app passes in a callable. That rule keeps the
library usable by an app that has no LLM at all, and is enforced by a test.

Note that ``max_agent_depth`` defaults to :data:`agent_mcp.depth.MAX_AGENT_DEPTH`
rather than repeating the literal ``5`` — there is one definition of the cap in this
codebase and `depth.py` owns it.
"""

from __future__ import annotations

import os
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from agent_mcp.depth import MAX_AGENT_DEPTH

#: Env var consulted for bearer keys when the app names no other.
DEFAULT_AUTH_KEYS_ENV = "AGENT_MCP_KEYS"


class AgentMCPSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="AGENT_MCP_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Identity ---
    # Short, snake_case service name. Also the namespace agent_runtime prefixes onto
    # this server's tools, so it shows up in every aggregated tool list.
    app_name: str = "app"
    version: str = "0.1.0"
    # This server's externally reachable base URL, with no /mcp suffix. Required to
    # register with the sync store — a service genuinely cannot work its own public
    # URL out, and publishing an unreachable one is worse than not registering.
    public_url: str = ""

    # --- Auth ---
    # Comma-separated bearer tokens, each optionally "name:token" so usage rows are
    # attributable. Read from this field first; see auth_keys_env for the override.
    keys: str = Field(
        default="",
        description="Comma-separated bearer tokens, optionally 'name:token'.",
    )
    # Name of a *different* env var to read keys from (e.g. "FINANCE_MCP_KEYS"), for
    # apps that would rather not namespace their secret under AGENT_MCP_. Read via
    # os.environ — see the trap documented on `resolved_keys`.
    auth_keys_env: str = ""
    # Escape hatch for local development. When true, a request with no (or an
    # unrecognised) token is admitted as caller "anonymous". Off by default: this
    # library fails closed, so a forgotten key is a loud refusal rather than a
    # silently open server.
    allow_anonymous: bool = False

    # --- Depth ---
    max_agent_depth: int = MAX_AGENT_DEPTH

    # --- Usage log ---
    usage_enabled: bool = True
    # SQLite file for the usage log. Defaults to its own file; an app that wants the
    # rows in its existing DB points this at that file (WAL makes the co-tenancy
    # safe) or passes a UsageSink of its own.
    usage_db_path: str = "agent_mcp.db"
    usage_retention_days: int = 30

    # --- Sync store ---
    # The hosted config/discovery store. Empty disables registration entirely; the
    # app still serves normally, it is just undiscoverable.
    sync_store_url: str = ""
    sync_store_token: str = ""
    heartbeat_interval_s: float = 300.0
    sync_timeout_s: float = 5.0

    # --- Peers (outbound call_peer) ---
    # Static "name=https://host,other=https://host" map. Until the hosted sync store
    # exists this is the primary discovery mechanism, not a stopgap; it also always
    # wins over the store so a local override during an incident sticks.
    peers: str = ""
    peer_token: str = ""
    peer_timeout_s: float = 30.0

    @property
    def auth_enabled(self) -> bool:
        return bool(self.resolved_keys()) or not self.allow_anonymous

    def resolved_keys(self) -> str:
        """The raw key string, honouring ``auth_keys_env`` when it is set.

        **The trap this method exists to soften:** ``auth_keys_env`` names an
        environment variable, which can only be read through ``os.environ`` — but
        pydantic-settings loads ``.env`` into the model *without* exporting anything
        into the process environment. So a developer who sets
        ``FINANCE_MCP_KEYS=...`` in ``.env`` only, and points ``auth_keys_env`` at
        it, gets an empty list and (fail-closed) a server that rejects everything
        for no visible reason.

        Mitigations, all three in play: the default source is the plain ``keys``
        field, which *does* read ``.env``; ``auth_keys_env`` is consulted only when
        explicitly set, and falls back to ``keys`` when the named variable is empty;
        and ``AgentMCPServer(auth_keys=...)`` bypasses both.
        """
        if self.auth_keys_env:
            from_env = os.environ.get(self.auth_keys_env, "")
            if from_env.strip():
                return from_env
        return self.keys


@lru_cache
def get_settings() -> AgentMCPSettings:
    """Process-wide settings singleton.

    Cached so ``.env`` is parsed once. Tests clear it with
    ``get_settings.cache_clear()`` after mutating the environment.
    """
    return AgentMCPSettings()


# Convenience handle for call sites that don't need lazy loading. Runtime code
# should still call get_settings() inside functions so tests can clear the cache.
settings = get_settings()
