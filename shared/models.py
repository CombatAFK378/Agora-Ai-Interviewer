"""Pydantic models shared across services.

Phase 1 added the session-start contract. Phase 2 adds the transcript turn and a
minimal audit event. Both are held in memory by the media worker for now; the
Postgres/Redis storage layer arrives in a later phase (see ARCHITECTURE §3, §8).
"""
import time

from pydantic import BaseModel, Field


class SessionStartResponse(BaseModel):
    """What the media worker hands back so the browser can join Agora."""
    app_id: str          # Agora App ID the web SDK needs
    channel: str         # channel both candidate and bot join
    uid: int             # numeric uid assigned to the candidate (browser)
    token: str           # candidate's join token, scoped to (channel, uid)


class TranscriptTurn(BaseModel):
    """One spoken turn — candidate answer or an agent question.

    `truncated` / `truncation_char` capture interruption: if an agent turn was
    cut off, the ledger must cite what the candidate *actually heard*
    (text[:truncation_char]), not what was generated (ARCHITECTURE §4).
    """
    seq: int
    speaker: str                       # "candidate" or an agent id
    text: str
    truncated: bool = False
    truncation_char: int | None = None


class AuditEvent(BaseModel):
    """Append-only record of a notable event (interruption, floor grant, …).

    Phase 2 only emits interruptions; the full audit surface fills in as later
    phases add floor control, scoring and overrides (ARCHITECTURE §3).
    """
    interview_id: str
    type: str
    data: dict = Field(default_factory=dict)
    ts: float = Field(default_factory=time.time)
