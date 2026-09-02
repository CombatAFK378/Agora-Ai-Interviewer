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


# Claim status (ARCHITECTURE §3). Resume-derived claims start UNVERIFIED (Phase 7);
# interview claims are SOLID (concrete, specific) or VAGUE (hedgy, hand-wavy).
CLAIM_SOLID = "SOLID"
CLAIM_VAGUE = "VAGUE"
CLAIM_UNVERIFIED = "UNVERIFIED"


class Claim(BaseModel):
    """One entry in the evidence ledger — a factual claim the candidate made,
    extracted inside a bid call and tied to a competency (ARCHITECTURE §3, §4).

    `strength` (0–1) and `status` drive competency coverage; `contradicts_claim_id`
    links a claim that conflicts with an earlier one (resume-vs-interview
    contradiction comes free in Phase 7 once resume claims seed the ledger).
    """
    id: str
    interview_id: str
    text: str
    competency: str                       # competency key
    source_turn: int                      # transcript_turn seq it came from
    strength: float                       # 0–1
    status: str = CLAIM_SOLID             # SOLID | VAGUE | UNVERIFIED
    noticed_by: list[str] = Field(default_factory=list)  # agent ids
    contradicts_claim_id: str | None = None
    ts: float = Field(default_factory=time.time)


# Conviction (ARCHITECTURE §6) — computed deterministically at lock, never asked.
CONVICTION_STRONG = "STRONG"
CONVICTION_NEUTRAL = "NEUTRAL"


class AgentScore(BaseModel):
    """One interviewer's locked score, written once at the final bell (§3, §6).

    Immutable after lock. `evidence` references the claim ids / turns behind the
    score so every number links back to the transcript (§13.8).
    """
    interview_id: str
    agent_id: str
    competency_scores: dict[str, float] = Field(default_factory=dict)  # key -> 0..1
    overall: float = 0.5                       # 0..1
    conviction: str = CONVICTION_NEUTRAL       # STRONG | NEUTRAL (deterministic)
    evidence: list[str] = Field(default_factory=list)   # claim ids / "turn N"
    rationale: str = ""


class DebateStatement(BaseModel):
    """One statement in the sequential debate (§6). A STRONG agent's MOVE is
    rejected in code (`rejected=True`), keeping its locked score."""
    interview_id: str
    round: int
    agent_id: str
    statement: str
    action: str                                # HOLD | MOVE
    score_before: float
    score_after: float
    rejected: bool = False


# Panel recommendation (§6). The split is the output — never an average (§13.6).
REC_PROCEED = "PROCEED"
REC_PROCEED_FLAGGED = "PROCEED_FLAGGED"
REC_INSUFFICIENT = "INSUFFICIENT_SIGNAL"
REC_DECLINE = "DECLINE"


class PanelConclusion(BaseModel):
    """The Orchestrator's final read of the panel (§6). Reports the split; it
    does not out-vote anyone."""
    interview_id: str
    recommendation: str
    headline: str                              # the split, in words
    unresolved: list[dict] = Field(default_factory=list)  # [{item, evidence}]
    reasoning: str = ""


class InterviewReport(BaseModel):
    """The locked, hashable record produced at the final bell (§6)."""
    interview_id: str
    scores: list[AgentScore]
    debate: list[DebateStatement]
    conclusion: PanelConclusion
    coverage: dict[str, float] = Field(default_factory=dict)
    locked_hash: str = ""                      # SHA-256 of the canonical record
