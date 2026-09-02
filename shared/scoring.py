"""Phase 5: lock, debate, conclusion (ARCHITECTURE §6).

Runs once at the final bell, on the reasoning model (latency-insensitive — the
candidate has left). Blind, independent scoring per interviewer off the evidence
ledger; deterministic conviction; a SHA-256 hash of the canonical record;
sequential debate where a STRONG agent physically cannot move its score; and one
Orchestrator conclusion that reports the split (never an average).

Nothing here is a web/Agora concern, so it lifts cleanly into the standalone
Orchestrator service later.
"""
import hashlib
import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor

from shared import llm_router, prompts
from shared.config import get_settings
from shared.models import (
    CONVICTION_NEUTRAL, CONVICTION_STRONG,
    REC_DECLINE, REC_INSUFFICIENT, REC_PROCEED, REC_PROCEED_FLAGGED,
    AgentScore, DebateStatement, InterviewReport, PanelConclusion,
)

logger = logging.getLogger(__name__)

_JSON = re.compile(r"\{.*\}", re.DOTALL)
_VALID_RECS = {REC_PROCEED, REC_PROCEED_FLAGGED, REC_INSUFFICIENT, REC_DECLINE}


def reason(messages: list[dict], max_tokens: int = 600) -> str:
    """Call the reasoning model, retrying through transient overloads; fall back
    to gpt-oss (reliable via the Groq key pool) only if it stays down."""
    settings = get_settings()
    try:
        return llm_router.chat(messages, model=settings.llm_reasoning_model,
                               temperature=0.4, max_tokens=max_tokens, timeout=90,
                               retries=3, use_fallback_chain=False)
    except Exception as e:
        logger.warning("reasoning model failed (%s); falling back to gpt-oss", e)
        return llm_router.chat(messages, model="groq:openai/gpt-oss-120b",
                               temperature=0.4, max_tokens=max_tokens,
                               reasoning_effort="high", use_fallback_chain=False)


def _clamp(v, lo=0.0, hi=1.0) -> float:
    try:
        return max(lo, min(hi, float(v)))
    except (TypeError, ValueError):
        return 0.5


def _parse_json(raw: str) -> dict:
    m = _JSON.search(raw)
    if not m:
        raise ValueError(f"no JSON object in: {raw[:120]!r}")
    return json.loads(m.group(0))


# ---- lock (scoring) ----------------------------------------------------

def compute_conviction(overall: float, evidence_count: int) -> str:
    """Deterministic (§6, §13.5): STRONG only if the score is far from the middle
    AND backed by enough evidence. Never asked of the model."""
    s = get_settings()
    if abs(overall - 0.5) > s.conviction_margin and evidence_count >= s.conviction_min_evidence:
        return CONVICTION_STRONG
    return CONVICTION_NEUTRAL


def score_agent(agent_id: str, transcript: list, claims: list, interview_id: str) -> AgentScore:
    my_comps = set(prompts._agent_competencies(agent_id))
    raw = reason(prompts.build_scoring_prompt(agent_id, transcript, claims), max_tokens=700)
    data = _parse_json(raw)
    comp = {str(k): _clamp(v) for k, v in (data.get("competency_scores") or {}).items()}
    overall = _clamp(data.get("overall", (sum(comp.values()) / len(comp)) if comp else 0.5))
    evidence = [str(e) for e in (data.get("evidence") or [])][:20]
    # evidence_count for conviction is the objective count of ledger claims in
    # this agent's area, not just what the model chose to cite.
    ev_count = sum(1 for c in claims if c.competency in my_comps)
    return AgentScore(
        interview_id=interview_id, agent_id=agent_id,
        competency_scores=comp, overall=overall,
        conviction=compute_conviction(overall, ev_count),
        evidence=evidence, rationale=str(data.get("rationale", ""))[:400],
    )


def lock(panel: list[str], transcript: list, claims: list, interview_id: str) -> list[AgentScore]:
    """Five independent scoring passes (isolated contexts), run in parallel."""
    with ThreadPoolExecutor(max_workers=len(panel)) as ex:
        return list(ex.map(lambda a: score_agent(a, transcript, claims, interview_id), panel))


def canonical_hash(transcript: list, claims: list, scores: list[AgentScore]) -> str:
    """SHA-256 of the canonical {transcript, claims, scores, convictions} (§6).
    Computed on the LOCKED scores, before debate — the tamper-evident record."""
    canon = {
        "transcript": [
            {"seq": t.seq, "speaker": t.speaker, "text": t.text,
             "truncated": t.truncated, "truncation_char": t.truncation_char}
            for t in transcript
        ],
        "claims": [
            {"id": c.id, "text": c.text, "competency": c.competency,
             "strength": c.strength, "status": c.status, "source_turn": c.source_turn}
            for c in claims
        ],
        "scores": [
            {"agent": s.agent_id, "competency_scores": s.competency_scores,
             "overall": s.overall, "conviction": s.conviction}
            for s in scores
        ],
    }
    blob = json.dumps(canon, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# ---- debate ------------------------------------------------------------

def _debate_order(scores: list[AgentScore]) -> list[str]:
    """Widest-diverging pair first, so the disagreement surfaces early (§6)."""
    agents = [s.agent_id for s in scores]
    overall = {s.agent_id: s.overall for s in scores}
    best = None
    for i in range(len(agents)):
        for j in range(i + 1, len(agents)):
            d = abs(overall[agents[i]] - overall[agents[j]])
            if best is None or d > best[0]:
                best = (d, agents[i], agents[j])
    if best is None:
        return agents
    first, second = best[1], best[2]
    return [first, second] + [a for a in agents if a not in (first, second)]


def run_debate(scores: list[AgentScore], interview_id: str) -> list[DebateStatement]:
    """Sequential debate. Locked scores are shown constant to every speaker; a
    STRONG agent's MOVE is rejected in code, keeping its locked score (§13.5)."""
    order = _debate_order(scores)
    score_map = {s.agent_id: s for s in scores}
    scores_summary = "\n".join(
        f"- {prompts.AGENTS[s.agent_id].name} ({s.agent_id}): overall {s.overall:.2f}, {s.conviction}"
        for s in scores
    )
    statements: list[DebateStatement] = []
    ctx: list[dict] = []
    for agent_id in order:
        s = score_map[agent_id]
        try:
            data = _parse_json(reason(prompts.build_debate_prompt(agent_id, scores_summary, ctx, s.conviction), max_tokens=400))
        except Exception as e:
            logger.warning("debate %s failed: %s", agent_id, e)
            data = {"action": "HOLD", "statement": "(no comment)", "new_overall": s.overall}
        action = str(data.get("action", "HOLD")).upper()
        statement = str(data.get("statement", ""))[:400]
        new_overall = _clamp(data.get("new_overall", s.overall))
        rejected = False
        score_after = s.overall
        if action == "MOVE" and s.conviction == CONVICTION_STRONG:
            rejected = True            # code-enforced HOLD — a STRONG agent can't move
            action = "HOLD"
        elif action == "MOVE":
            score_after = new_overall  # a NEUTRAL agent may adjust its position
        statements.append(DebateStatement(
            interview_id=interview_id, round=1, agent_id=agent_id, statement=statement,
            action=action, score_before=s.overall, score_after=score_after, rejected=rejected,
        ))
        ctx.append({"agent": agent_id, "statement": statement})
    return statements


# ---- conclusion --------------------------------------------------------

def conclude(interview_id: str, scores: list[AgentScore],
             statements: list[DebateStatement], coverage: dict,
             claims: list) -> PanelConclusion:
    scores_summary = "\n".join(
        f"- {prompts.AGENTS[s.agent_id].name} ({s.agent_id}): {s.overall:.2f} ({s.conviction}); {s.rationale}"
        for s in scores
    )
    debate_summary = "\n".join(
        f"- {prompts.AGENTS[st.agent_id].name}: {st.action}"
        f"{' [STRONG hold enforced]' if st.rejected else ''} — {st.statement}"
        for st in statements
    ) or "(no debate)"
    solid = sum(1 for c in claims if c.status == "SOLID")
    mean_cov = (sum(coverage.values()) / len(coverage)) if coverage else 0.0
    cov_summary = (", ".join(f"{k} {int(v * 100)}%" for k, v in coverage.items()) or "(none)"
                   + f"\nEvidence volume: {len(claims)} claims ({solid} solid); "
                   f"mean coverage {int(mean_cov * 100)}%. Thin evidence → INSUFFICIENT_SIGNAL, not DECLINE.")
    try:
        data = _parse_json(reason(prompts.build_conclusion_prompt(scores_summary, debate_summary, cov_summary), max_tokens=700))
    except Exception as e:
        logger.warning("conclusion failed: %s", e)
        data = {"recommendation": REC_INSUFFICIENT, "headline": "Conclusion unavailable",
                "unresolved": [], "reasoning": str(e)[:200]}
    rec = str(data.get("recommendation", REC_INSUFFICIENT)).upper()
    if rec not in _VALID_RECS:
        rec = REC_INSUFFICIENT
    # Deterministic guard: with essentially no evidence there is nothing to judge —
    # never DECLINE, always flag for a human (§5). Nuanced calls stay with the model.
    if len(claims) < 2 and rec == REC_DECLINE:
        rec = REC_INSUFFICIENT
    unresolved = [
        {"item": str(u.get("item", "")), "evidence": str(u.get("evidence", ""))}
        for u in (data.get("unresolved") or []) if isinstance(u, dict)
    ][:10]
    return PanelConclusion(
        interview_id=interview_id, recommendation=rec,
        headline=str(data.get("headline", ""))[:300],
        unresolved=unresolved, reasoning=str(data.get("reasoning", ""))[:1500],
    )


def build_report(interview_id: str, panel: list[str], transcript: list,
                 claims: list, coverage: dict) -> InterviewReport:
    """The full final-bell pipeline: lock → hash → debate → conclusion (§6)."""
    logger.info("scoring: locking %d interviewers over %d turns / %d claims",
                len(panel), len(transcript), len(claims))
    scores = lock(panel, transcript, claims, interview_id)
    locked_hash = canonical_hash(transcript, claims, scores)
    statements = run_debate(scores, interview_id)
    conclusion = conclude(interview_id, scores, statements, coverage, claims)
    logger.info("scoring done: recommendation=%s hash=%s", conclusion.recommendation, locked_hash[:12])
    return InterviewReport(
        interview_id=interview_id, scores=scores, debate=statements,
        conclusion=conclusion, coverage=coverage, locked_hash=locked_hash,
    )
