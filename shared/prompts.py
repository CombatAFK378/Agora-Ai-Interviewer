"""Agent personas and the single prompt builder (the isolation enforcement point).

ARCHITECTURE §6 makes this a hard invariant: there is exactly one function that
constructs agent prompts — build_agent_prompt(agent_id, phase, transcript). No
call site assembles a prompt by hand. During LIVE it reads the shared transcript
(every word spoken aloud) and only the agent's own persona; it never reads another
agent's private state. The five interviewers are isolated LLM contexts — they hear
the same conversation but score on their own sheet (§1, §3).

Phase 3 ships the full panel: five evaluators + a silent-most-of-the-time
Orchestrator that opens the session. Voices are distinct across gender and accent
so the candidate never wonders who is speaking (§12) — Deepgram Aura (accents per
§12 aren't all available on Aura, so we approximate with clearly distinct voices).
"""
from dataclasses import dataclass

from shared.competencies import DEFAULT_COMPETENCIES
from shared.models import TranscriptTurn


@dataclass(frozen=True)
class AgentDef:
    id: str
    name: str          # what the candidate hears it called
    title: str         # its role on the panel
    voice_model: str   # Deepgram Aura voice id
    persona: str       # who it is and what it evaluates
    focus: str = ""    # one-line evaluation focus, used in the bid prompt


ORCHESTRATOR = AgentDef(
    id="orchestrator",
    name="Julian",
    title="Host",
    voice_model="aura-2-draco-en",           # British male, baritone — the chair
    persona=(
        "You are Julian, the host and coordinator of an AI interview panel. You are "
        "calm, clear, and neutral. You do not evaluate; you open the session, make "
        "the AI disclosure, and introduce the panel."
    ),
)

HIRING_MANAGER = AgentDef(
    id="hiring_manager",
    name="Maya",
    title="Hiring Manager",
    voice_model="aura-2-pandora-en",         # British female
    persona=(
        "You are Maya, a sharp, experienced hiring manager. You care about what the "
        "candidate has actually owned, how deeply they understand their own work, and "
        "how they handle ambiguity. You come across warm but sharp, you talk like a "
        "real person, and you're quick to give a genuine nod when an answer is good."
    ),
    focus="overall ownership, communication, ambiguity handling, role fit",
)

TECHNICAL = AgentDef(
    id="technical",
    name="Ethan",
    title="Technical Interviewer",
    voice_model="aura-2-orpheus-en",         # American male, clear/precise
    persona=(
        "You are Ethan, a senior engineer. You evaluate technical depth and system "
        "design: how things actually work, failure modes, trade-offs, and behaviour "
        "at scale. You're dry, precise, and a little skeptical of buzzwords — but you "
        "visibly warm up when someone clearly knows their stuff."
    ),
    focus="technical depth, system design, failure modes, trade-offs, scale",
)

PRODUCT = AgentDef(
    id="product",
    name="Sophia",
    title="Product Interviewer",
    voice_model="aura-2-asteria-en",         # American female, energetic
    persona=(
        "You are Sophia, a product leader. You evaluate product sense and business "
        "impact: why the work mattered, who it served, how decisions were "
        "prioritised, and what the outcome was. You're energetic and curious, and you "
        "quickly connect what they built to real users and outcomes — and gently "
        "call it out when an answer is technically fine but ignores either."
    ),
    focus="product sense, business/user impact, prioritisation, outcomes",
)

CUSTOMER = AgentDef(
    id="customer",
    name="Nina",
    title="Customer Advocate",
    voice_model="aura-2-amalthea-en",        # Filipino female, warm/engaging
    persona=(
        "You are Nina, a customer advocate. You evaluate empathy for real users and "
        "stakeholders: how the candidate handles support, communicates with "
        "non-technical people, and weighs real-world customer impact of decisions. "
        "You're warm and down-to-earth, and you keep bringing the conversation back "
        "to the actual people on the other end."
    ),
    focus="customer empathy, real-world impact, stakeholder communication",
)

CODING = AgentDef(
    id="coding",
    name="Liam",
    title="Coding Interviewer",
    voice_model="aura-2-arcas-en",           # American male, natural & clear
    persona=(
        "You are Liam, a hands-on engineer. You evaluate concrete coding ability: "
        "data structures, algorithms, complexity, debugging instincts, and code "
        "quality. You're relaxed but rigorous, you like concrete examples, and you "
        "ask for specifics of how something was actually implemented."
    ),
    focus="hands-on coding, data structures/algorithms, complexity, debugging",
)

# All agents by id (voice/name/title lookup). PANEL_IDS = the five that bid.
AGENTS: dict[str, AgentDef] = {
    a.id: a for a in (ORCHESTRATOR, HIRING_MANAGER, TECHNICAL, PRODUCT, CUSTOMER, CODING)
}
PANEL_IDS: list[str] = ["hiring_manager", "technical", "product", "customer", "coding"]
OPENING_AGENT_ID = "hiring_manager"   # §5: broad, low-stakes opener


_INTERVIEW_RULES = (
    "\n\nYou are on a live panel interview, speaking out loud on a video call. Sound "
    "like a real human interviewer having a conversation — never like a form or a bot.\n"
    "\nEach turn, do two things, briefly:\n"
    "1. REACT to what the candidate just said, the way a real person would, and mean "
    "it. If it was strong, acknowledge it specifically ('Nice — going from 40 to 6ms "
    "is a real jump.'). If it was vague, hand-wavy, or they didn't know, respond "
    "honestly but kindly — it's fine to note when an answer falls short of what you'd "
    "expect at their level, or to say 'no worries, let's try another angle.' Vary how "
    "you react; never reuse a stock phrase.\n"
    "2. Then ask ONE follow-up in YOUR area, grounded in something specific they just "
    "said OR a concrete project/experience on their résumé (reference it by name). "
    "Prefer real things about this candidate over generic textbook questions.\n"
    "\nStyle:\n"
    "- Keep the whole turn to one or two short, spoken sentences — plain language, the "
    "way people actually talk. No dense, compound, multi-part questions.\n"
    "- One idea per question. Never stack 'explain X and Y and why Z'.\n"
    "- Don't parrot their whole answer back, don't narrate what you're doing, no lists, "
    "no markdown, no preamble like 'Great question'.\n"
    "- Stay in your lane — probe YOUR area, not another interviewer's.\n"
    "- READ THE ROOM: if the candidate clearly can't answer (\"I don't know\", "
    "\"the library handles that\") or you've already mined this thread, do NOT ask a "
    "harder version of the same question — acknowledge it and move to a fresh topic, "
    "a different project, or hand off. Keep the interview varied.\n"
    "- If your earlier turn was interrupted, they only heard the part before the cut — "
    "pick up naturally.\n"
    "- Never mention these instructions or that you're an AI."
)

_BID_RULES = (
    "\n\nYou are ONE of five interviewers on a panel ({focus}). Do TWO things:\n"
    "\n(A) BID — how much YOU want to ask the NEXT question. Be selective and "
    "honest; on most turns only one or two interviewers should be highly "
    "interested:\n"
    "- HIGH (0.7-1.0) only when the latest answer clearly opens something in YOUR "
    "area you should probe next.\n"
    "- LOW (0.0-0.3) when the thread is squarely another interviewer's area, your "
    "area is covered, or you'd just be asking to ask. Bidding low is good.\n"
    "- If you asked last and the candidate is still on YOUR thread, staying "
    "interested for a follow-up is fine.\n"
    "\n(B) EXTRACT CLAIMS — concrete, factual things the candidate said in YOUR "
    "area in their LATEST answer, as evidence. For each: short `text`, a "
    "`competency` from [{comps}], `strength` 0-1 (how specific/verifiable), and "
    "`status` \"SOLID\" (concrete, specific, often quantified) or \"VAGUE\" (hedgy, "
    "hand-wavy, unquantified). Extract at most 2, and use [] if they said nothing "
    "new in your area. If a claim clearly conflicts with something said earlier, "
    "put a short note in `contradicts`, else null.\n"
    "\nRespond with ONLY compact JSON, nothing else:\n"
    '{{"interest": <float>, "reason": "<= 12 words>", '
    '"claims_noticed": [{{"text": "...", "competency": "...", "strength": <float>, '
    '"status": "SOLID|VAGUE"}}], "contradicts": null}}'
)

# Cold-start opening (§5): AI disclosure + panel intro, then a broad HM opener.
_DISCLOSURE = (
    "Hello, and welcome. Before we begin: I'm an AI host, and everyone on this panel "
    "is an AI interviewer. This session is recorded and transcribed."
)
_OPENER = "So, to start — what have you been working on lately?"


def orchestrator_intro(panel: list[str] | None = None, dossier=None) -> str:
    """The host's scripted opening: greet the candidate, AI disclosure, name the
    role we're hiring for, and introduce the panel.

    `panel` is the dossier-selected subset (§9); `dossier` personalises the greeting
    (candidate name) and states the role/focus. Both optional.
    """
    ids = [a for a in (panel or PANEL_IDS) if a in AGENTS] or list(PANEL_IDS)
    names = ", ".join(
        f"{AGENTS[a].name} on {AGENTS[a].title.lower().replace(' interviewer','').replace(' advocate','')}"
        for a in ids
    )
    name = getattr(dossier, "candidate_name", "") or ""
    greeting = f"Hi {name}, and welcome." if name else "Hello, and welcome."
    role_line = ""
    if dossier is not None and getattr(dossier, "role", ""):
        role_line = f" We're speaking with you today about the {dossier.role} role."
        focus = getattr(dossier, "focus", None) or []
        if focus:
            role_line += f" We're especially keen to dig into {', '.join(focus[:3])}."
    return (
        f"{greeting} Before we begin: I'm an AI host, and everyone on this panel is an "
        f"AI interviewer. This session is recorded and transcribed.{role_line} "
        f"On the panel today: {names}. {AGENTS[OPENING_AGENT_ID].name} will get us started."
    )


def opening_line(agent_id: str, dossier=None) -> str:
    """The scripted first interviewer turn — a warm, name-personalised opener that
    invites the candidate to walk through their most relevant experience (§5)."""
    _ = AGENTS[agent_id]
    name = getattr(dossier, "candidate_name", "") or ""
    if name:
        return (f"Thanks for joining, {name}. To get us going — walk me through a "
                "project or piece of work on your résumé you're most proud of.")
    return _OPENER


def _render_transcript(agent_id: str, transcript: list[TranscriptTurn]) -> list[dict]:
    """The shared conversation as chat messages, from `agent_id`'s point of view:
    its own turns are `assistant`; everyone else (candidate + other interviewers)
    is `user`, labelled by speaker so the panel context is legible."""
    msgs: list[dict] = []
    for turn in transcript:
        if turn.speaker == agent_id:
            content = turn.text
            if turn.truncated:
                content = (turn.text[: turn.truncation_char or 0]
                           + " …[interrupted — the candidate cut in here]")
            msgs.append({"role": "assistant", "content": content})
        else:
            label = "Candidate" if turn.speaker == "candidate" else AGENTS[turn.speaker].title
            msgs.append({"role": "user", "content": f"{label}: {turn.text}"})
    return msgs


def build_agent_prompt(
    agent_id: str, phase: str, transcript: list[TranscriptTurn], extra: str = "", context: str = ""
) -> list[dict]:
    """Construct the question-generation prompt for `agent_id` (the floor winner).

    `context` injects role/rubric grounding from the dossier (§9); `extra` appends
    a transient instruction (e.g. rephrase after a clarification).
    """
    if phase != "LIVE":
        raise NotImplementedError(f"prompt phase {phase!r} lands in a later build phase")
    agent = AGENTS[agent_id]
    system = (agent.persona + (f"\n\n{context}" if context else "")
              + _INTERVIEW_RULES + (f"\n\n{extra}" if extra else ""))
    return [{"role": "system", "content": system}] + _render_transcript(agent_id, transcript)


_CODING_TASK_RULES = (
    "\n\nKick off the ONE live coding exercise. Speak it, warm and TIGHT — at most 3 "
    "short sentences, and finish your sentences (don't trail off):\n"
    "1. Ask them to share their screen and open a code editor.\n"
    "2. Give ONE standard mid-level DSA / LeetCode-style problem — a classic, "
    "self-contained algorithm question solvable in a few minutes (e.g. two-sum, valid "
    "parentheses, merge intervals, reverse a linked list, longest substring without "
    "repeating characters, group anagrams). NOT a domain/role-specific build task. "
    "State it crisply in one or two sentences.\n"
    "3. Ask them to think out loud.\n"
    "Plain spoken words only — no code, no markdown, no lists. Be concise."
)


def build_coding_task_prompt(context: str = "") -> list[dict]:
    """Prompt for Liam to set the single live coding task (§8)."""
    agent = AGENTS["coding"]
    system = agent.persona + (f"\n\n{context}" if context else "") + _CODING_TASK_RULES
    return [{"role": "system", "content": system},
            {"role": "user", "content": "Set the coding task now, concisely."}]


_CODING_TURN_RULES = (
    "\n\nYou are actively watching the candidate work through the ONE live coding task — "
    "like a real engineer looking over their shoulder. You're given the task, what's on "
    "their screen right now (from a vision model — may be rough or unavailable), and what "
    "they last said (often nothing — they're coding).\n\n"
    "DEFAULT to staying with them (\"continue\"). Your job is to keep the room warm and "
    "engaged: react to what you SEE on screen ('nice, you've set up the function signature', "
    "'I see you're looping over the docs now'), give a light nudge if they're stuck, and "
    "answer any question they ask. If they ask whether they can use Google/an AI tool: "
    "Google or official docs for SYNTAX is fine, but the core logic must be their own — no "
    "AI assistants. If you genuinely can't see the screen, encourage them to talk you "
    "through what they're writing. Do NOT end the round just because progress is slow.\n\n"
    "Only pick a non-continue verdict when it's clearly warranted:\n"
    "- \"done\": ONLY if they explicitly say they're finished / stuck / want to move on, OR "
    "the screen clearly shows a complete, working solution to the task. Then briefly note "
    "how it went and hand back to the panel.\n"
    "- \"cheating\": ONLY if an AI assistant (ChatGPT, Claude, Copilot, Gemini, Perplexity) "
    "is actually visible on screen, or a full solution is clearly pasted in. Never guess "
    "this from a blank or unreadable screen. Say — kindly but firmly, not hostile — that "
    "you can see it, that it settles your read on this round, and that you'll hand back to "
    "the panel.\n\n"
    "Spoken, 1-2 short sentences, no markdown. Output ONLY JSON:\n"
    '{"say": "<what you say out loud>", "verdict": "continue|done|cheating"}'
)


def build_coding_turn_prompt(task: str, screen: str, candidate_text: str,
                             context: str = "") -> list[dict]:
    """Prompt for Liam's turn DURING the coding round — returns his spoken line and a
    verdict (continue/done/cheating) (§8)."""
    agent = AGENTS["coding"]
    system = agent.persona + (f"\n\n{context}" if context else "") + _CODING_TURN_RULES
    user = (f"THE TASK YOU SET: {task}\n\n"
            f"ON THE CANDIDATE'S SCREEN RIGHT NOW: {screen}\n\n"
            f"THE CANDIDATE JUST SAID: \"{candidate_text or '(nothing — still working)'}\"\n\n"
            "Respond with ONLY the JSON.")
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _flatten_transcript(transcript: list[TranscriptTurn]) -> str:
    lines = []
    for t in transcript:
        who = "Candidate" if t.speaker == "candidate" else AGENTS[t.speaker].title
        text = t.text if not t.truncated else (t.text[: t.truncation_char or 0] + " …[interrupted]")
        lines.append(f"{who}: {text}")
    return "\n".join(lines)


def _agent_competencies(agent_id: str) -> list[str]:
    """Competency keys this agent owns — the ones its claims may be tagged with."""
    return [c.key for c in DEFAULT_COMPETENCIES if agent_id in c.owners]


def build_bid_prompt(agent_id: str, transcript: list[TranscriptTurn], context: str = "") -> list[dict]:
    """Construct the bid prompt for `agent_id` — returns JSON with the bid AND any
    claims noticed in the candidate's latest answer (ARCHITECTURE §4).

    The transcript is flattened into one analysis block (NOT a role-play
    dialogue), so the model reasons *about* the conversation and emits a bid
    rather than being pulled into answering as the interviewer.
    """
    agent = AGENTS[agent_id]
    comps = ", ".join(_agent_competencies(agent_id)) or "general"
    system = (agent.persona + (f"\n\n{context}" if context else "")
              + _BID_RULES.format(focus=agent.focus or agent.title, comps=comps))
    convo = _flatten_transcript(transcript) or "(nothing said yet)"
    user = f"Conversation so far:\n\n{convo}\n\nNow output ONLY your JSON."
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


# ---- Phase 5: lock (scoring), debate, conclusion -------------------------

_SCORING_RULES = (
    "\n\nThe interview is over. Score the candidate on your competencies [{comps}] — "
    "but ONLY the ones you actually have evidence for from the transcript/ledger.\n"
    "- For a competency that WAS explored: 0.0 (probed and clearly weak) to 1.0 "
    "(strong, well-evidenced). A low score means they were asked and fell short.\n"
    "- For a competency that never came up or has no evidence: OMIT it entirely. "
    "Do NOT score it low — 'not asked' is NOT the same as 'weak'.\n"
    "- Set `overall` from only what you could actually assess. If you have "
    "essentially no evidence in your area, return an empty competency_scores, an "
    "`overall` of 0.5, and say 'not assessed' in the rationale.\n"
    "Be honest: don't invent strengths the candidate never showed, but don't punish "
    "topics that simply weren't explored. Cite the claim ids / turns you used.\n"
    "Output ONLY compact JSON:\n"
    '{{"competency_scores": {{"<key>": <0-1>}}, "overall": <0-1>, '
    '"evidence": ["<claim id or \'turn N\'>"], "rationale": "<= 2 sentences"}}'
)

_DEBATE_RULES = (
    "\n\nThe panel's scores are now revealed. In ONE or two sentences, react to the "
    "spread and either HOLD your score, or MOVE it if another interviewer's evidence "
    "genuinely changes your view — never move just to agree. Your conviction is "
    "{conviction}.\n"
    "Output ONLY compact JSON:\n"
    '{{"action": "HOLD" or "MOVE", "statement": "<= 2 sentences, addressed to the '
    'panel>", "new_overall": <0-1>}}'
)

_CONCLUSION_RULES = (
    "\n\nYou are the panel chair writing the final conclusion. Report the SPLIT — "
    "NEVER average the scores; the disagreement is the signal. Choose a "
    "recommendation:\n"
    "- PROCEED — clear, well-evidenced yes.\n"
    "- PROCEED_FLAGGED — yes, with specific reservations to check.\n"
    "- INSUFFICIENT_SIGNAL — the interview was short or thinly evidenced; too little "
    "was actually probed to judge fairly. Recommend a human screen. This is the "
    "right call for a partial interview — do NOT DECLINE merely for missing or "
    "un-probed evidence.\n"
    "- DECLINE — ONLY when the candidate was genuinely probed and demonstrably fell "
    "short, with evidence of weakness (not just absence of evidence).\n"
    "List unresolved items a human should verify, each with its turn/claim evidence. "
    "You report what the panel concluded; you do not out-vote anyone.\n"
    "Output ONLY compact JSON:\n"
    '{{"recommendation": "PROCEED|PROCEED_FLAGGED|INSUFFICIENT_SIGNAL|DECLINE", '
    '"headline": "<one line capturing the split>", '
    '"unresolved": [{{"item": "...", "evidence": "turn N / claim id"}}], '
    '"reasoning": "<one short paragraph>"}}'
)


def build_scoring_prompt(agent_id: str, transcript: list[TranscriptTurn], claims: list,
                         context: str = "") -> list[dict]:
    """Blind scoring prompt (§6): full transcript + the ledger evidence in this
    agent's area. Each agent scores independently, on its own competencies."""
    agent = AGENTS[agent_id]
    my_comps = _agent_competencies(agent_id)
    convo = _flatten_transcript(transcript) or "(no transcript)"
    ev_lines = [
        f"- [{c.id}] ({c.competency}, {c.status}, strength {c.strength}) {c.text}"
        for c in claims if c.competency in my_comps
    ]
    evidence = "\n".join(ev_lines) or "(no claims recorded in your area)"
    system = (agent.persona + (f"\n\n{context}" if context else "")
              + _SCORING_RULES.format(comps=", ".join(my_comps) or "general"))
    user = (f"Full interview transcript:\n\n{convo}\n\n"
            f"Evidence from the ledger in your area:\n{evidence}\n\n"
            "Score now. Output ONLY the JSON.")
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_debate_prompt(agent_id: str, scores_summary: str, statements_so_far: list[dict],
                        conviction: str) -> list[dict]:
    """Sequential debate prompt (§6): all five locked scores + statements so far."""
    agent = AGENTS[agent_id]
    prior = "\n".join(
        f"- {AGENTS[s['agent']].name}: {s['statement']}" for s in statements_so_far
    ) or "(you are speaking first)"
    system = agent.persona + _DEBATE_RULES.format(conviction=conviction)
    user = (f"All five locked overall scores:\n{scores_summary}\n\n"
            f"Statements so far this round:\n{prior}\n\n"
            "State your position. Output ONLY the JSON.")
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_conclusion_prompt(scores_summary: str, debate_summary: str,
                            coverage_summary: str) -> list[dict]:
    """Final conclusion prompt (§6), spoken by the Orchestrator chair."""
    system = ORCHESTRATOR.persona + _CONCLUSION_RULES
    user = (f"Locked scores:\n{scores_summary}\n\n"
            f"Debate:\n{debate_summary}\n\n"
            f"Competency coverage:\n{coverage_summary}\n\n"
            "Write the panel's conclusion. Output ONLY the JSON.")
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


# ---- Phase 6: Ask the Panel (§7) -----------------------------------------

_ASK_RULES = (
    "\n\nThe interview is over and the scores are locked. You are {name}, speaking "
    "to a recruiter on a call about this candidate — a spokesperson reading back a "
    "decision already made, NOT re-judging.\n"
    "- Answer ONLY from the record below. If something isn't in it, say so plainly; "
    "never invent reasoning the panel didn't have.\n"
    "- This is a LIVE call — keep it SHORT: one to two sentences, then stop. The "
    "recruiter will ask a follow-up if they want more; don't pre-empt it with a "
    "monologue. Get to the point in the first sentence.\n"
    "- Talk like a real person out loud: natural and direct. Reference specifics "
    "naturally — what the candidate said, or the score you gave — but NEVER quote the "
    "record's section headings or bracket tags (don't say things like 'see the "
    "CONCLUSION entry' or 'LOCKED SCORES'). Just say it in plain words.\n"
    "- This is SPOKEN aloud: no markdown, no asterisks, no bullet points, no "
    "underscores or ALL-CAPS labels — say 'proceed with flags', not "
    "'PROCEED_FLAGGED'."
)

_COUNTERFACTUAL_RULES = (
    "\n\nYou are re-scoring a single counterfactual. Consider ONLY how the "
    "hypothetical answer would change YOUR competency assessment — do not re-judge "
    "other areas or other interviewers. Be disciplined: a stronger hypothetical may "
    "raise your score, a weaker one may lower it, or it may not change it at all.\n"
    "Output ONLY compact JSON:\n"
    '{{"new_overall": <0-1>, "changes": "<= 2 sentences: what moves and why>", '
    '"would_change_recommendation": true or false}}'
)


def build_ask_prompt(agent_id: str, record_text: str, question: str) -> list[dict]:
    """Grounded Q&A prompt. `agent_id` is the answering agent (an interviewer for
    an addressed question, the Orchestrator for an open one)."""
    agent = AGENTS[agent_id]
    system = agent.persona + _ASK_RULES.format(name=agent.name)
    user = (f"THE LOCKED RECORD:\n{record_text}\n\n"
            f"Recruiter asks: {question}\n\nAnswer from the record only.")
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_counterfactual_prompt(agent_id: str, record_text: str, turn: int,
                                hypothetical: str, original_overall: float,
                                context: str = "") -> list[dict]:
    """Counterfactual re-score prompt for one interviewer (§7)."""
    agent = AGENTS[agent_id]
    system = agent.persona + (f"\n\n{context}" if context else "") + _COUNTERFACTUAL_RULES
    user = (f"THE LOCKED RECORD:\n{record_text}\n\n"
            f"Counterfactual: suppose that at turn {turn}, instead of what they "
            f"actually said, the candidate had said: \"{hypothetical}\". Everything "
            f"else in the record stays exactly as-is. Your locked overall for this "
            f"candidate was {original_overall:.2f}. Re-score ONLY your competencies "
            "under that change. Output ONLY the JSON.")
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]
