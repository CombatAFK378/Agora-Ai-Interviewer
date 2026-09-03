"""Phase 6: Ask the Panel (ARCHITECTURE §7).

After the interview is locked, the recruiter interrogates the record. Nothing is
recomputed for ordinary questions — the agents are spokespeople reading back a
decision, grounded strictly in the stored record (grounding guardrail: if it's
not written down, the answer is "I don't have that"). Counterfactuals do re-run
one agent's scoring on a hypothetical, but they are stored separately and NEVER
mutate the locked scores, so the hash still verifies (§7, §13.7).
"""
import logging
from dataclasses import dataclass, field

from shared import llm_router, prompts, scoring
from shared.config import get_settings
from shared.models import AskAnswer, Override, WhatIfQuery

logger = logging.getLogger(__name__)


@dataclass
class PanelRecord:
    """Everything Ask the Panel reads back — held in memory for the session."""
    interview_id: str
    report: object                 # InterviewReport
    transcript: list               # list[TranscriptTurn]
    claims: list                   # list[Claim]
    audit: list = field(default_factory=list)      # list[dict]
    what_ifs: list = field(default_factory=list)    # list[WhatIfQuery]
    overrides: list = field(default_factory=list)   # list[Override]
    contexts: dict = field(default_factory=dict)    # agent id -> role/rubric grounding (§9)


def _cap(s: str, n: int) -> str:
    """Trim to a word boundary with an ellipsis — never mid-word."""
    s = (s or "").strip()
    if len(s) <= n:
        return s
    return s[:n].rsplit(" ", 1)[0].rstrip() + "…"


def render_record(r: PanelRecord) -> str:
    """Flatten the locked record into the grounding context for an answer.

    Kept LEAN on purpose: this is sent on EVERY Ask-the-Panel question, so a
    bloated context both slows generation and burns the fast model's per-minute
    token budget (tipping answers onto slow fallback models). We keep everything
    that grounds an answer — every score, every claim, the candidate's own words
    in full, the conclusion — and only trim the verbose parts (long interviewer
    questions especially; the candidate's words + the evidence ledger carry the
    substance). Phase 8 replaces this with per-question retrieval (RAG)."""
    rep = r.report
    c = rep.conclusion
    out = [f"CONCLUSION: {c.recommendation} — {c.headline}"]
    if c.reasoning:
        out.append(f"Reasoning: {_cap(c.reasoning, 400)}")
    if c.unresolved:
        out.append("Unresolved: " + "; ".join(
            f"{u.get('item')} [{u.get('evidence')}]" for u in c.unresolved))

    out.append("\nLOCKED SCORES:")
    for s in rep.scores:
        a = prompts.AGENTS[s.agent_id]
        comps = ", ".join(f"{k} {v:.2f}" for k, v in s.competency_scores.items()) or "not assessed"
        out.append(f"- {a.name} ({a.title}): overall {s.overall:.2f} [{s.conviction}] — "
                   f"{comps}. {_cap(s.rationale, 180)}")

    out.append("\nDEBATE:")
    for d in rep.debate:
        a = prompts.AGENTS[d.agent_id]
        out.append(f"- {a.name}: {d.action}"
                   f"{' (STRONG hold enforced)' if d.rejected else ''} — {_cap(d.statement, 160)}")

    out.append("\nEVIDENCE LEDGER:")
    for cl in r.claims:
        out.append(f"- turn {cl.source_turn} · {cl.competency} · {cl.status} — {_cap(cl.text, 140)}")

    # Candidate turns carry the substance, so keep them fuller; interviewer
    # questions are often long — trim them hard.
    out.append("\nTRANSCRIPT:")
    for t in r.transcript:
        is_cand = t.speaker not in prompts.AGENTS
        who = "Candidate" if is_cand else prompts.AGENTS[t.speaker].title
        txt = t.text if not t.truncated else (t.text[: t.truncation_char or 0] + " …[interrupted]")
        out.append(f"- turn {t.seq} {who}: {_cap(txt, 320 if is_cand else 120)}")
    return "\n".join(out)


def answer(record: PanelRecord, question: str, mode: str = "open",
           target: str | None = None) -> AskAnswer:
    """Answer a recruiter question, grounded in the record (fast model —
    spokesperson, not judge).

    Addressed → that interviewer answers, no override. Open → the Orchestrator
    answers AND carries the override tool: if the recruiter's message is a clear
    override instruction, it overrules the recommendation on the spot (§7)."""
    model = get_settings().llm_fast_model

    if mode == "addressed" and target in prompts.AGENTS:
        # The recruiter addressed someone who wasn't on this role's panel (§9 dossier
        # dropped them) — say so plainly instead of improvising a non-existent view.
        on_panel = {s.agent_id for s in record.report.scores}
        if target not in on_panel:
            who = prompts.AGENTS[target]
            msg = (f"I wasn't on the panel for this role, so I didn't assess this "
                   f"candidate — {who.title.lower()} wasn't a fit for what this job needs. "
                   "Happy to point you to whoever did.")
            return AskAnswer(question=question, mode="addressed", target=target,
                             answered_by=target, answer=msg)
        messages = prompts.build_ask_prompt(target, render_record(record), question)
        text = llm_router.chat(messages, model=model, max_tokens=180,
                               temperature=0.4, reasoning_effort="low")
        return AskAnswer(question=question, mode="addressed", target=target,
                         answered_by=target, answer=text)

    # Open → Orchestrator answers, grounded in the record. Overrides are handled
    # deterministically by the caller (voice: AskPipeline; text: the override
    # button → /panel/override), not by a fragile LLM tool-call.
    messages = prompts.build_ask_prompt("orchestrator", render_record(record), question)
    text = llm_router.chat(messages, model=model, max_tokens=180,
                           temperature=0.4, reasoning_effort="low")
    return AskAnswer(question=question, mode="open", answered_by="orchestrator", answer=text)


def counterfactual(record: PanelRecord, turn: int, hypothetical: str, agent_id: str) -> WhatIfQuery:
    """Re-score ONLY `agent_id` on a hypothetical answer at `turn` (§7). Stored
    as a what-if; the locked score is never touched."""
    score = next((s for s in record.report.scores if s.agent_id == agent_id), None)
    if score is None:
        raise ValueError(f"unknown agent {agent_id!r}")
    messages = prompts.build_counterfactual_prompt(
        agent_id, render_record(record), turn, hypothetical, score.overall,
        record.contexts.get(agent_id, ""))
    data = scoring._parse_json(scoring.reason(messages, max_tokens=400))
    wq = WhatIfQuery(
        interview_id=record.interview_id, agent_id=agent_id, source_turn=turn,
        hypothetical=hypothetical[:400], original_overall=score.overall,
        new_overall=scoring._clamp(data.get("new_overall", score.overall)),
        changes=str(data.get("changes", ""))[:400],
    )
    record.what_ifs.append(wq)
    return wq


def override(record: PanelRecord, decision: str, reason: str) -> Override:
    """Log a recruiter override (§7, §13.10). Original recommendation is kept."""
    ov = Override(
        interview_id=record.interview_id,
        original_recommendation=record.report.conclusion.recommendation,
        decision=decision, reason=reason[:500], locked_hash=record.report.locked_hash,
    )
    record.overrides.append(ov)
    return ov
