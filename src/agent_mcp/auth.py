"""Bearer-token verification — the whole auth story, deliberately.

Every app in the ecosystem is a small personal service behind Caddy, called by a
handful of known agents. The right amount of auth for that is "compare the
presented token against a short list", and the wrong amount is OAuth. The SDK ships
a ``TokenVerifier``/``AuthSettings`` pair, but it is issuer/resource-server shaped,
the two must be passed together or it raises, and it drags a discovery-metadata
surface into a library whose entire requirement is a string comparison. So this is
thirty lines of pure functions instead.

Pure functions, following Amber's convention: settings are passed **in** as
arguments rather than read from a global, so every branch is testable without
touching the environment or standing up a server.

Two design rules:

* **Fail closed.** No configured keys means every call is rejected, not that auth is
  off. The opposite default — Amber's ``auth_secret=""`` meaning "open socket" — is
  right for a personal voice loop on one box and wrong for a library that will front
  finance data over the network, where a forgotten env var should be a loud failure
  rather than a silent open door. ``allow_anonymous`` is the explicit opt-out.
* **Never log a token, not even truncated.** Usage rows need an attributable caller,
  so keys may be written ``name:token`` and the name is what gets logged. A bare
  token falls back to a ``sha256:`` fingerprint, which identifies a caller across
  rows without being reversible into the secret.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
from collections.abc import Mapping
from dataclasses import dataclass

logger = logging.getLogger(__name__)

#: Identity stamped on calls arriving with no transport to authenticate — stdio, or
#: an in-process ``Client(server.mcp)``. Distinct from "anonymous", which means a
#: real network call that was let through by the escape hatch.
CALLER_LOCAL = "local"
CALLER_ANONYMOUS = "anonymous"
#: Stamped on the usage row for a call that was refused before it authenticated, so
#: a rejected attempt still leaves a trace rather than vanishing.
CALLER_UNAUTHENTICATED = "unauthenticated"


@dataclass(frozen=True)
class AuthResult:
    ok: bool
    caller: str
    reason: str = ""


def fingerprint(token: str) -> str:
    """A stable, non-reversible label for a caller presenting a bare token."""
    return "sha256:" + hashlib.sha256(token.encode("utf-8")).hexdigest()[:8]


def parse_keys(raw: str | None) -> dict[str, str]:
    """Parse a comma-separated key list into ``{token: caller_name}``.

    Accepts both the plain form from the spec (``"abc123,def456"``) and a
    ``name:token`` form (``"amber:abc123,spawner:def456"``) that makes usage rows
    attributable. Blank entries and surrounding whitespace are ignored so a trailing
    comma in a ``.env`` file is not a footgun.
    """
    keys: dict[str, str] = {}
    for entry in (raw or "").split(","):
        entry = entry.strip()
        if not entry:
            continue
        name, sep, token = entry.partition(":")
        if sep and token.strip():
            keys[token.strip()] = name.strip() or fingerprint(token.strip())
        else:
            keys[entry] = fingerprint(entry)
    return keys


def extract_bearer(header: str | None) -> str:
    """Pull the token out of an ``Authorization`` value.

    Tolerates extra internal whitespace and odd casing of the scheme. A bare token
    with no ``Bearer`` scheme is rejected — accepting it would mean a malformed
    header could be mistaken for a valid credential.
    """
    if not header:
        return ""
    scheme, _, rest = header.strip().partition(" ")
    if scheme.lower() != "bearer":
        return ""
    return rest.strip()


def verify_token(token: str, keys: Mapping[str, str]) -> str | None:
    """Return the caller name for ``token``, or ``None``.

    Compares against **every** key with :func:`hmac.compare_digest` and without
    short-circuiting, so neither the wall-clock time nor the number of comparisons
    leaks which prefix was correct.
    """
    match: str | None = None
    for known, caller in keys.items():
        if hmac.compare_digest(token, known):
            match = caller
    return match


def verify_bearer(
    header: str | None,
    keys: Mapping[str, str],
    *,
    allow_anonymous: bool = False,
) -> AuthResult:
    """Authenticate one inbound request from its ``Authorization`` header."""
    if not keys:
        if allow_anonymous:
            return AuthResult(True, CALLER_ANONYMOUS)
        return AuthResult(
            False,
            "",
            "no bearer keys configured (set AGENT_MCP_KEYS, or "
            "AGENT_MCP_ALLOW_ANONYMOUS=true for local development)",
        )

    token = extract_bearer(header)
    if not token:
        if allow_anonymous:
            return AuthResult(True, CALLER_ANONYMOUS)
        return AuthResult(False, "", "missing or malformed Authorization header")

    caller = verify_token(token, keys)
    if caller is None:
        return AuthResult(False, "", "unrecognised bearer token")
    return AuthResult(True, caller)


def header_value(headers: Mapping[str, str] | None, name: str) -> str | None:
    """Case-insensitive header lookup.

    ``ctx.headers`` arrives lowercased over HTTP (verified against a live server) but
    a hand-built mapping in a test will not be, so neither side is trusted.
    """
    if not headers:
        return None
    wanted = name.lower()
    for key, value in headers.items():
        if str(key).lower() == wanted:
            return value
    return None
