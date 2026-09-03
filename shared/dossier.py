"""JD + résumé ingestion → the interview dossier (ARCHITECTURE §9).

Parsed ONCE at interview creation (never per turn), on the reasoning model. The
five interviewers and their competency domains are fixed; the dossier decides
which of them sit on the panel, how competencies are weighted for this role,
what "strong" means per interviewer (rubrics), and pre-registers the résumé's
claims so the ledger can verify or contradict them during the interview.
"""
import logging

from shared import prompts, scoring
from shared.competencies import DEFAULT_COMPETENCIES
from shared.models import Dossier

logger = logging.getLogger(__name__)

_COMP_KEYS = [c.key for c in DEFAULT_COMPETENCIES]


def _short_phrase(f, limit: int = 48) -> str:
    """A focus phrase, trimmed to a WORD boundary so the spoken intro never cuts
    off mid-word (e.g. 'research summariza')."""
    s = " ".join(str(f).split())          # collapse whitespace
    if len(s) <= limit:
        return s
    cut = s[:limit].rsplit(" ", 1)[0]
    return cut or s[:limit]


def _agent_catalog() -> str:
    lines = []
    for aid in prompts.PANEL_IDS:
        comps = [c.key for c in DEFAULT_COMPETENCIES if aid in c.owners]
        lines.append(f"- {aid} ({prompts.AGENTS[aid].title}) — covers: {', '.join(comps)}")
    return "\n".join(lines)


_SYSTEM = (
    "You configure an AI interview panel from a job description and a candidate "
    "résumé. The available interviewers (fixed) and the competencies each covers:\n"
    "{catalog}\n"
    "Valid competency keys: {keys}\n\n"
    "From the JD and résumé, decide:\n"
    "1. role and seniority (junior | mid | senior | staff).\n"
    "2. candidate_name: the candidate's FIRST name from the résumé (for greeting). "
    "Empty string if you truly can't find it.\n"
    "3. focus: 4-6 SHORT phrases (2-4 words each, no parentheticals) naming the "
    "specific skills, tools, or domains THIS JD calls for that the interview should "
    "probe (e.g. 'RAG systems', 'fine-tuning LLMs', 'vector databases', 'fintech "
    "domain'). Concrete, from the JD.\n"
    "4. panel: the interviewer ids that actually matter for THIS role — a subset; "
    "drop ones that don't apply (e.g. no 'customer' interviewer for a pure backend "
    "role, no 'coding' for a non-coding role). Always keep 'hiring_manager'.\n"
    "5. competency_weights: importance 0.0-1.0 for the competencies that matter for "
    "this role (use only valid keys).\n"
    "6. rubrics: for EACH interviewer on the panel, one sentence on what a STRONG "
    "candidate looks like for THIS specific role in their area.\n"
    "7. resume_claims: concrete, checkable achievements from the résumé — name the "
    "actual projects/roles (e.g. 'Built LogInsight, a real-time log analyzer with "
    "Kafka + RAG'), each mapped to a valid competency key. These are what the panel "
    "will dig into, so keep the specifics.\n\n"
    "Output ONLY compact JSON:\n"
    '{{"role":"...","seniority":"...","candidate_name":"...",'
    '"summary":"<one line on role + candidate>","focus":["..."],'
    '"panel":["..."],"competency_weights":{{"<key>":<0-1>}},'
    '"rubrics":{{"<agent_id>":"..."}},'
    '"resume_claims":[{{"text":"...","competency":"<key>"}}]}}'
)


def build_dossier(jd: str, resume: str) -> Dossier:
    """Parse JD + résumé into a Dossier. Falls back to a generic all-panel dossier
    if parsing fails, so interview creation never blocks on it."""
    system = _SYSTEM.format(catalog=_agent_catalog(), keys=", ".join(_COMP_KEYS))
    user = f"JOB DESCRIPTION:\n{(jd or '(none)').strip()[:6000]}\n\n" \
           f"RÉSUMÉ:\n{(resume or '(none)').strip()[:6000]}\n\nOutput ONLY the JSON."
    try:
        data = scoring._parse_json(scoring.reason(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            max_tokens=1200))
    except Exception:
        logger.exception("dossier parse failed; using generic dossier")
        return Dossier()

    panel = [a for a in (data.get("panel") or []) if a in prompts.PANEL_IDS]
    if "hiring_manager" not in panel:
        panel.append("hiring_manager")
    panel = panel or list(prompts.PANEL_IDS)

    weights = {k: max(0.0, min(1.0, float(v)))
               for k, v in (data.get("competency_weights") or {}).items() if k in _COMP_KEYS}
    rubrics = {k: str(v)[:400] for k, v in (data.get("rubrics") or {}).items() if k in panel}
    claims = [
        {"text": str(c.get("text", ""))[:200], "competency": str(c.get("competency", ""))}
        for c in (data.get("resume_claims") or []) if c.get("text")
    ][:20]
    focus = [_short_phrase(f) for f in (data.get("focus") or []) if str(f).strip()][:8]
    name = str(data.get("candidate_name", "")).strip()[:40]

    return Dossier(
        role=str(data.get("role", "Software Engineer"))[:120],
        seniority=str(data.get("seniority", "mid"))[:40],
        candidate_name=name,
        summary=str(data.get("summary", ""))[:300],
        focus=focus, panel=panel, competency_weights=weights,
        rubrics=rubrics, resume_claims=claims,
    )


def role_context(dossier: Dossier, agent_id: str) -> str:
    """LEAN role/rubric context for the token-hot paths (bids, scoring). Kept short
    on purpose — bids fire five times per turn."""
    parts = [f"This interview is for a {dossier.seniority} {dossier.role}."]
    if dossier.focus:
        parts.append(f"The role calls for: {', '.join(dossier.focus)}.")
    rubric = dossier.rubrics.get(agent_id)
    if rubric:
        parts.append(f"For your area, a strong candidate: {rubric}")
    parts.append("Judge at the level expected for this seniority.")
    return " ".join(parts)


def question_context(dossier: Dossier, agent_id: str) -> str:
    """RICH context for QUESTION generation (once per turn, fast model): the lean
    role context PLUS the candidate's actual résumé highlights, so the interviewer
    can ground questions in specific projects/experience by name (§9)."""
    parts = [role_context(dossier, agent_id)]
    if dossier.candidate_name:
        parts.append(f"The candidate is {dossier.candidate_name}.")
    if dossier.resume_claims:
        highlights = "\n".join(f"- {c.get('text', '')}" for c in dossier.resume_claims[:12])
        parts.append(
            "The candidate's résumé highlights (use these — reference specific "
            "projects/experience by name, and probe whether the depth is real):\n"
            f"{highlights}")
    parts.append("Mix questions grounded in their résumé with questions about what "
                 "this role needs. Ask about real things on their résumé, not generic "
                 "textbook questions.")
    return "\n".join(parts)
