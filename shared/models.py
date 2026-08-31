"""Pydantic models shared across services.

Phase 1 only needs the session-start contract between the web client and the
media worker. More models (transcript turns, claims, scores) arrive in later
phases.
"""
from pydantic import BaseModel


class SessionStartResponse(BaseModel):
    """What the media worker hands back so the browser can join Agora."""
    app_id: str          # Agora App ID the web SDK needs
    channel: str         # channel both candidate and bot join
    uid: int             # numeric uid assigned to the candidate (browser)
    token: str           # candidate's join token, scoped to (channel, uid)
