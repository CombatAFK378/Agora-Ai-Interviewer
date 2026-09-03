"""Phase 6: Ask the Panel (ARCHITECTURE §7).

After the interview is locked, the recruiter interrogates the record. Nothing is
recomputed for ordinary questions — the agents are spokespeople reading back a
decision, grounded strictly in the stored record (grounding guardrail: if it's
not written down, the answer is "I don't have that"). Counterfactuals do re-run
one agent's scoring on a hypothetical, but they are stored separately and NEVER
mutate the locked scores, so the hash still verifies (§7, §13.7).
"""
import json
import logging
from dataclasses import dataclass, field

from shared import llm_router, prompts, scoring
from shared.config import get_settings
from shared.models import AskAnswer, Override, WhatIfQuery

logger = logging.getLogger(__name__)

# The Orchestrator's override tool (§7). It fires ONLY on a clear, explicit
# recruiter instruction to overrule — so a recruiter can override by voice/typing
# ("override this to decline, …") with no button. Questions never trigger it.
_OVERRIDE_TOOL = [{
    "type": "function",
    "function": {
        "name": "override_recommendation",
        "description": (
            "Record a recruiter override of the panel's recommendation. ONLY call "
            "this when the recruiter gives a CLEAR, explicit instruction to override "
            "or overrule the recommendation (e.g. 'override this to decline', "
            "'I'm overruling this — mark it proceed'). NEVER call it for a question, "
            "a hypothetical, or general discussion."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "decision": {"type": "string",
                             "enum": ["PROCEED", "PROCEED_FLAGGED", "INSUFFICIENT_SIGNAL", "DECLINE"]},
                "reason": {"type": "string", "description": "the recruiter's stated reason"},
            },
            "required": ["decision", "reason"],
        },
    },
}]


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


def render_record(r: PanelRecord) -> str:
    """Flatten the locked record into the grounding context for an answer."""
    rep = r.report
    c = rep.conclusion
    out = [f"CONCLUSION: {c.recommendation} — {c.headline}"]
    if c.reasoning:
        out.append(f"Reasoning: {c.reasoning}")
    if c.unresolved:
        out.append("Unresolved: " + "; ".join(
            f"{u.get('item')} [{u.get('evidence')}]" for u in c.unresolved))

    out.append("\nLOCKED SCORES:")
    for s in rep.scores:
        a = prompts.AGENTS[s.agent_id]
        comps = ", ".join(f"{k} {v:.2f}" for k, v in s.competency_scores.items()) or "not assessed"
        out.append(f"- {a.name} ({a.title}): overall {s.overall:.2f} [{s.conviction}] — "
                   f"{comps}. {s.rationale}")

    out.append("\nDEBATE:")
    for d in rep.debate:
        a = prompts.AGENTS[d.agent_id]
        out.append(f"- {a.name}: {d.action}"
                   f"{' (STRONG hold enforced)' if d.rejected else ''} — {d.statement}")

    out.append("\nEVIDENCE LEDGER:")
    for cl in r.claims:
        out.append(f"- [{cl.id}] turn {cl.source_turn} · {cl.competency} · {cl.status} · "
                   f"strength {cl.strength:.1f} — {cl.text}")

    out.append("\nTRANSCRIPT:")
    for t in r.transcript:
        who = prompts.AGENTS[t.speaker].title if t.speaker in prompts.AGENTS else "Candidate"
        txt = t.text if not t.truncated else (t.text[: t.truncation_char or 0] + " …[interrupted]")
        out.append(f"- turn {t.seq} {who}: {txt}")

    grants = [a for a in r.audit if a.get("type") == "floor_grant"]
    if grants:
        out.append("\nFLOOR GRANTS (who was chosen to ask, and why):")
        for a in grants:
            d = a.get("data", {})
            out.append(f"- {d.get('winner')} (λ={d.get('lambda')}, all_low={d.get('all_low')})")
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
        messages = prompts.build_ask_prompt(target, render_record(record), question)
        text = llm_router.chat(messages, model=model, max_tokens=350,
                               temperature=0.4, reasoning_effort="low")
        return AskAnswer(question=question, mode="addressed", target=target,
                         answered_by=target, answer=text)

    # Open → Orchestrator, with the override tool available.
    messages = prompts.build_ask_prompt("orchestrator", render_record(record), question)
    messages[0]["content"] += (
        "\n\nYou also have an `override_recommendation` tool. Call it ONLY when the "
        "recruiter clearly and explicitly instructs you to override the recommendation; "
        "for anything else, just answer.")
    try:
        msg = llm_router.complete_with_tools(messages, _OVERRIDE_TOOL, model=model, max_tokens=400)
    except Exception as e:
        logger.warning("tool completion failed (%s); plain answer", e)
        text = llm_router.chat(messages, model=model, max_tokens=350,
                               temperature=0.4, reasoning_effort="low")
        return AskAnswer(question=question, mode="open", answered_by="orchestrator", answer=text)

    for call in (msg.get("tool_calls") or []):
        fn = call.get("function", {})
        if fn.get("name") == "override_recommendation":
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except Exception:
                args = {}
            ov = override(record, str(args.get("decision", "")).upper(), str(args.get("reason", "")))
            ans = (f"Done — I've overridden the recommendation from "
                   f"{ov.original_recommendation.replace('_', ' ')} to "
                   f"{ov.decision.replace('_', ' ')} and logged your reason. The panel's "
                   "original recommendation stays on record.")
            return AskAnswer(question=question, mode="open", answered_by="orchestrator",
                             answer=ans, override=ov)

    text = (msg.get("content") or "").strip() or "(no answer)"
    return AskAnswer(question=question, mode="open", answered_by="orchestrator", answer=text)


def counterfactual(record: PanelRecord, turn: int, hypothetical: str, agent_id: str) -> WhatIfQuery:
    """Re-score ONLY `agent_id` on a hypothetical answer at `turn` (§7). Stored
    as a what-if; the locked score is never touched."""
    score = next((s for s in record.report.scores if s.agent_id == agent_id), None)
    if score is None:
        raise ValueError(f"unknown agent {agent_id!r}")
    messages = prompts.build_counterfactual_prompt(
        agent_id, render_record(record), turn, hypothetical, score.overall)
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
