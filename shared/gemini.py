"""Gemini Live (§8 coding round): mint short-lived ephemeral tokens so the browser
can open a realtime Live session (screen-watch + voice) without ever seeing the
real API key. If no key is configured the coding round falls back to snapshot
vision, so this module is entirely optional.
"""
import logging

import requests

from shared.config import get_settings

logger = logging.getLogger(__name__)

_TOKEN_URL = "https://generativelanguage.googleapis.com/v1alpha/auth_tokens"


def enabled() -> bool:
    return bool(get_settings().gemini_api_key)


def mint_ephemeral_token(uses: int = 2) -> str:
    """Create an ephemeral auth token for client-side Live use; returns the token
    string the browser passes as its apiKey (v1alpha). Raises on failure."""
    s = get_settings()
    if not s.gemini_api_key:
        raise RuntimeError("GEMINI_API_KEY not set")
    resp = requests.post(f"{_TOKEN_URL}?key={s.gemini_api_key}",
                         json={"uses": uses}, timeout=15)
    resp.raise_for_status()
    name = resp.json().get("name", "")
    if not name:
        raise RuntimeError("no token returned")
    return name
