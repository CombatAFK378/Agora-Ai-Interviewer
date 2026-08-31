"""Builds Agora RTC tokens.

An Agora token is a short-lived signed pass that proves a participant is allowed
into a specific channel with a specific numeric user id (uid). Both the browser
(candidate) and the bot (media worker) need one to join the same channel.

Uses the pure-Python `agora-token-builder`, so it runs anywhere (no native libs).
"""
import time
from agora_token_builder import RtcTokenBuilder

# Agora role: 1 = PUBLISHER (can send audio). Both candidate and bot publish.
ROLE_PUBLISHER = 1


def build_rtc_token(
    app_id: str,
    app_certificate: str,
    channel_name: str,
    uid: int,
    ttl_seconds: int = 3600,
) -> str:
    """Return a token valid for `ttl_seconds` for (channel_name, uid)."""
    now = int(time.time())
    privilege_expire = now + ttl_seconds
    return RtcTokenBuilder.buildTokenWithUid(
        app_id,
        app_certificate,
        channel_name,
        uid,
        ROLE_PUBLISHER,
        privilege_expire,
    )
