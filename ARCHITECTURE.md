# 968ms — Architecture

An adaptive AI interview panel. Five AI interviewers with different evaluation
objectives conduct a live voice interview, coordinated by a silent Orchestrator
that decides who speaks. Scores are private during the interview, locked at the
end, then debated. A recruiter can rejoin the session later and cross-examine
any interviewer about its reasoning.

This document is the implementation contract. It describes structure, data flow
and invariants — not code.

## 0. Stack and layout

**Backend: Python + FastAPI.** Both server components are FastAPI apps. Python is
the right call here because Silero VAD and Smart Turn v3.1 both ship Python ONNX
inference, and Agora has a Python server SDK — the media worker needs all three
in one process.

**Frontend: React.** Candidate room and recruiter dashboard, using the Agora Web
SDK for audio and RTM for speaker signals.

```
media-worker/     FastAPI. One process per interview. Agora, VAD,
                  Smart Turn, Sarvam STT, Deepgram Flux TTS.
orchestrator/     FastAPI. OpenAI-compatible endpoint, floor control,
                  ledger, lock/debate/conclusion, Ask the Panel.
api/              FastAPI. Interviews CRUD, reports, overrides, auth.
web/              React. Candidate room + recruiter dashboard.
shared/           Pydantic models, prompt builder, LLM router, DB layer.
models/           Downloaded ONNX weights (Silero, Smart Turn).
```

`shared/` matters more than it looks. The prompt builder and the LLM router both
live there, and both are enforcement points for invariants in §13 — keep them in
one place that every service imports.

---

## 1. The one idea that shapes everything

**Agora carries audio. The agents live on our server.**

There is exactly **one** bot participant in the Agora channel. It has no LLM, no
memory and no opinions. It is an ear and a mouth: it receives candidate audio and
publishes synthesised speech.

The five interviewers are **not** RTC participants. They are five isolated LLM
contexts on our server. The candidate sees five tiles because our own frontend
draws five tiles and highlights whichever one is currently speaking, driven by an
RTM signal.

Consequences:

- 1× RTC minutes instead of 5×
- Agents cannot hear each other, because they were never in the room
- Turn-taking is trivial — we own the single outbound audio stream
- Voice identity is a TTS parameter, not an infrastructure concern

### Portability requirement

The Orchestrator **must** expose an OpenAI-compatible `POST /v1/chat/completions`
endpoint. This is non-negotiable and costs almost nothing to honour.

It means we can run on plain Agora RTC (free, unlimited testing) during
development, and optionally switch to Agora Conversational AI Engine for demo day
— which would hand us their VAD, turn detection and interruption handling — by
pointing ConvAI's custom-LLM setting at the same endpoint. Same orchestrator code
either way. Decide later, never rewrite.

---

## 2. Components

```
┌──────────────────────────────────────────────────────────────┐
│ WEB CLIENT                                                    │
│  · candidate room: 5 interviewer tiles, captions, screen share│
│  · recruiter dashboard: results, "Join session" button        │
└───────────────┬───────────────────────────┬──────────────────┘
                │ Agora RTC (audio)         │ HTTPS
                │ Agora RTM (signals)       │
┌───────────────▼───────────────┐   ┌───────▼──────────────────┐
│ MEDIA WORKER (1 per interview)│   │ API / DASHBOARD SERVICE   │
│  · joins Agora channel        │   │  · interviews CRUD        │
│  · Silero VAD                 │   │  · reports               │
│  · Smart Turn v3.1            │   │  · overrides, audit       │
│  · Sarvam STT (streaming)     │   └───────┬──────────────────┘
│  · Deepgram Flux TTS          │           │
│  · publishes one audio track  │           │
└───────────────┬───────────────┘           │
                │ HTTP                       │
┌───────────────▼───────────────────────────▼──────────────────┐
│ ORCHESTRATOR SERVICE                                          │
│  · OpenAI-compatible endpoint                                 │
│  · bid collection + floor control                             │
│  · coverage map + time budget                                 │
│  · evidence ledger writes                                     │
│  · lock / debate / conclusion                                 │
│  · Ask the Panel session revival                              │
└───────────────┬───────────────────────────┬──────────────────┘
                │                            │
┌───────────────▼──────────────┐   ┌────────▼──────────────────┐
│ AGENT LAYER (stateless)      │   │ STORAGE                    │
│  · 5 evaluators + chair      │   │  · Postgres (durable)      │
│  · prompt builder (isolation)│   │  · Redis (live turn state) │
│  · LLM router + fallbacks    │   │  · object store (audio)    │
└──────────────────────────────┘   └───────────────────────────┘
```

### Model routing

| Job | Frequency | Latency | Model |
|---|---|---|---|
| Bids + claim extraction | every turn, ×5 parallel | critical | `openai/gpt-oss-120b` |
| Question generation | every turn | critical | `openai/gpt-oss-120b` |
| Final scoring | once, ×5 | none | `nvidia/nemotron-3-ultra-550b-a55b` |
| Debate statements | once, ×5 | none | Nemotron 3 Ultra |
| Orchestrator conclusion | once | none | Nemotron 3 Ultra |
| Ask the Panel replies | on demand | ~1s | `gpt-oss-120b` |
| Counterfactual re-score | on demand | ~2s | Nemotron 3 Ultra |

Rule: **fast model where the candidate is waiting, deep model where the decision
is made.**

Every LLM call goes through one router module with a fallback chain
(`primary → secondary → openrouter/free`). Free model IDs rotate without notice;
never hardcode a single ID at a call site.

### The Orchestrator is a service, not a single model

Most of what the Orchestrator does is deterministic code. Only two jobs use an
LLM. Keep this boundary sharp — it is an engineering decision *and* a defensible
claim.

| Job | Implementation |
|---|---|
| Collecting bids | code — parallel HTTP fan-out |
| Priority formula, coverage map, time budget | code — arithmetic |
| Tie-breaking, recency penalties | code |
| Choosing who speaks | code |
| Writing audit events | code |
| Reading the debate, writing the conclusion | Nemotron |
| Answering in Ask the Panel | gpt-oss; Nemotron for counterfactuals |

Turn-taking being deterministic is the point: we can show the exact five numbers
behind every floor grant and the formula that picked the winner. An LLM router
would be slower, non-reproducible, and unable to explain itself four days later.

---

## 3. Data model

All tables carry `interview_id`. There is no global mutable state anywhere.

### Shared vs private

Two classes of data, and the distinction is the whole product:

- **Shared** — the transcript (every word spoken aloud, candidate answers *and*
  agent questions) and the evidence ledger. All agents read these.
- **Private** — each agent's own scores, notes and reasoning. Only its owner
  reads it, and only until lock.

The Orchestrator reads the shared data **plus all five private states**. It is the
only component that can. Isolation means agents cannot see each other's
judgments — it does not mean they are deaf to the conversation. Same as a real
panel: everyone hears every question and answer, everyone scores on their own
sheet, the chair collects all five sheets.

**`interview`** — id, status (`CREATED|LIVE|LOCKED|CONCLUDED`), channel name,
time budget, started/ended timestamps, panel composition, locked hash.

**`candidate_dossier`** — one per interview. Parsed JD and resume: role,
seniority, required competencies with weights, and resume claims extracted as
pre-registered ledger entries.

**`transcript_turn`** — sequence number, speaker (`candidate` or an agent id),
text, audio offset, start/end ms, `truncated` flag and truncation point if the
turn was interrupted.

**`competency`** — per interview. Name, weight, target evidence count, owning
agent ids. Seeded from the dossier.

**`claim`** — the evidence ledger. Extracted from candidate answers. Fields:
text, competency, source turn + character span, strength (0–1), status
(`SOLID | VAGUE | UNVERIFIED`), `contradicts_claim_id`, and which agent noticed
it.

Resume-derived claims start as `UNVERIFIED`. When the candidate says something in
the interview that supports or contradicts a resume claim, the ledger links them.
That is where resume-vs-interview contradiction detection comes from, free.

**`agent_state`** — one row per (interview, agent). The agent's private working
memory: running estimate per competency, private notes, evidence references.
**Never read across agents during the LIVE phase.**

**`agent_score`** — written once at lock. Final score per competency and overall,
`conviction` (`STRONG | NEUTRAL`), evidence reference list. Immutable.

**`debate_statement`** — round, agent, statement text, action (`HOLD | MOVE`),
score before, score after, rejection flag if a MOVE was refused.

**`panel_conclusion`** — recommendation, headline split, unresolved items with
their evidence timestamps, full reasoning text.

**`audit_event`** — append-only. Every floor grant, score write, lock, debate
statement, recruiter question, counterfactual and override. This is what makes
Ask the Panel answerable and the record defensible.

**`what_if_query`** — counterfactuals from Ask the Panel. Stored separately and
**never** allowed to mutate `agent_score`.

**`override`** — recruiter decision, reason, timestamp, resulting status.

---

## 4. The live turn loop

```
1  Candidate speaks
2  Silero VAD detects speech onset, then a pause (stop_secs = 0.2)
3  Smart Turn v3.1 classifies the segment: complete or incomplete
      incomplete → keep listening, no LLM call
4  Sarvam STT finalises the utterance → write transcript_turn
5  Fan out 5 bid calls IN PARALLEL, one per interviewer
6  Orchestrator computes floor priority, picks a winner
7  Winning agent generates its question
8  Deepgram Flux streams TTS in that agent's voice
9  Media worker publishes audio; RTM signal lights the tile
10 Ledger updated with claims noticed by all five agents
```

Steps 5 and 10 use the same call. Each bid response returns:

```
{ interest: 0.0-1.0,
  reason: "one line, for the audit log",
  claims_noticed: [ {text, competency, strength, status} ],
  contradicts: [claim_id] }
```

Different agents notice different things — Product spots business claims,
Technical spots failure-mode claims. Five perspectives on one answer is a feature,
not duplication.

### Latency budget

| Stage | Target |
|---|---|
| Smart Turn decision | 15 ms |
| STT finalisation | 150 ms |
| 5 parallel bid calls | 400 ms |
| Question generation | 500 ms |
| TTS time-to-first-byte | 150 ms |
| **Total** | **~1.2 s** |

Acceptable — in an interview a short pause reads as the interviewer considering
the answer. Two optimisations if it feels slow:

- **Speculative bidding.** Fire bid calls on the interim transcript as soon as
  Smart Turn leans "complete". Discard if the candidate resumes.
- **Stream TTS.** Start publishing audio on the first sentence rather than
  waiting for the full question.

### Interruption

If VAD detects candidate speech while TTS is playing:

1. Stop pushing TTS audio immediately
2. Mark the agent's `transcript_turn` as `truncated` with the character offset
   actually delivered
3. Log the interruption as an audit event

Step 2 is not optional. If Product was cut off halfway through a question, the
ledger must cite what the candidate **actually heard**, not what was generated.
Otherwise Ask the Panel will quote a question that was never asked.

---

## 5. Floor control

```
priority(agent) = interest × (1 + λ · gap) × recency_penalty
```

- `interest` — the agent's own bid, 0–1
- `gap` — `1 − coverage` of that agent's weakest owned competency
- `λ` — coverage weight. Starts ~0.5, rises toward ~1.5 in the final third of the
  time budget, so late questions chase what we still don't know
- `recency_penalty` — 0.5 if the agent spoke last turn, 0.8 two turns ago, else
  1.0. Stops one interviewer monopolising the room

`coverage(competency) = min(1, Σ claim.strength / target_evidence)`

The Orchestrator writes an audit event for every floor grant containing all five
priorities and the winner. This is what lets it answer *"why did Product get that
turn?"* four days later.

### Cold start — the first question

Turn one has no candidate answer, so bidding cannot run. The opening is scripted.

1. **Orchestrator speaks.** AI disclosure — the candidate is told plainly that all
   interviewers are AI and the session is recorded and transcribed — then
   introduces the five by name. This satisfies the PS11 disclosure requirement and
   lets the candidate hear each voice before it carries meaning.
2. **Hiring Manager asks the opener.** Broad and low-stakes: what they have been
   working on. Deliberately *not* Technical — opening hard depresses the signal
   quality of everything after, because a nervous candidate underperforms for the
   first two minutes regardless of ability. Real panels know this.
3. **Turn two onward: normal bidding.** A broad opening answer touches several
   competencies at once, so the ledger and coverage map hold real values
   immediately.

The opener is a dossier field (`opening_agent`), defaulting to Hiring Manager. A
support-engineering role might open with Customer and a scenario instead.

### Tie-breaking

The formula usually separates near-equal bids by itself: five agents all bidding
0.8 diverge once coverage gap and recency penalty are applied. Genuine ties are
rare.

When priorities land within 0.05, resolve in this fixed order:

1. Highest competency weight from the dossier
2. Fewest turns taken so far
3. Stable priority order — a fixed list

**Never randomise.** The same inputs must always produce the same winner, because
Ask the Panel has to answer *"why did Product get that turn?"* days later.

### When every bid is low

If all five interests fall below ~0.25, the panel has nothing it still needs to
ask — either the candidate is comprehensively understood, or the conversation has
stalled. Do not let the panel ask filler.

Trigger one of:

- switch to a scenario or role-play question on the weakest-covered competency
- compress — cut the remaining question budget and move toward wrap-up
- flag `insufficient signal` if coverage is thin despite low interest

This is common with strong candidates and looks bad if unhandled.

### Low confidence never ends an interview early

If a competency's confidence plateaus, the Orchestrator **reallocates**:

- shift remaining time to a different competency, or
- shorten the remaining question count, or
- flag `insufficient signal — recommend human screen`

It never terminates the session. This is a product decision, not a technical one,
and it is deliberate.

---

## 6. Scoring, locking, debate

Everything in this section runs on the reasoning model —
`nvidia/nemotron-3-ultra-550b-a55b`, referenced in code as `LLM_REASONING_MODEL`.
Nothing here is latency-sensitive; the candidate has left. Use the deepest free
model available and let it take its time.

### Isolation is a code invariant, not a convention

There must be exactly one function that constructs agent prompts:

```
build_agent_prompt(agent_id, interview_id, phase)
```

During `phase = LIVE` it reads the shared transcript, the shared ledger, and
**only that agent's own** `agent_state`. Any attempt to read another agent's
state in LIVE phase raises. Make the blindness impossible to break by accident —
it is the core claim of the product.

### Lock

At the final bell:

1. Run 5 independent scoring passes over the full transcript, one per agent, each
   using only its own state. These are separate calls with separate contexts.
2. Compute `conviction` **deterministically**, not by asking the model:
   `STRONG` if `|score − 0.5| > 0.25` and `evidence_count ≥ 3`, else `NEUTRAL`.
3. Write `agent_score` rows. Mark them immutable.
4. SHA-256 the canonical JSON of `{transcript, claims, scores, convictions}`.
   Store the hash on the interview.

### Debate

Sequential, not parallel — agents must be able to respond to each other.

```
statements = []
for agent in debate_order:
    prompt = base
           + all five locked scores (NOW revealed)
           + statements so far this round
           + conviction rule
    response = call(reasoning_model)
    if agent.conviction == STRONG and response.action == MOVE:
        reject the move, keep locked score, log rejection
    statements.append(response)
```

The conviction rule is enforced **in code**. A STRONG agent physically cannot
move its score, whatever the model outputs. Prompt instructions are guidance;
this is a guarantee.

`debate_order`: put the widest-diverging pair first so the disagreement surfaces
early.

### Conclusion

One final call, on Nemotron. Input: all five locked scores, all statements, who
held and who moved, and the coverage map. Output:

- recommendation (`PROCEED | PROCEED_FLAGGED | INSUFFICIENT_SIGNAL | DECLINE`)
- headline split — never an average
- unresolved items, each with the evidence timestamps a human must check
- reasoning text

The Orchestrator reports what the panel concluded. It does not out-vote anyone.

---

## 7. Ask the Panel

The differentiating feature. It works because nothing is thrown away.

### Revival

When the recruiter clicks Join:

1. Open a new Agora channel for the recruiter
2. Rehydrate each agent: system prompt + full transcript + its own final notes +
   its locked score + its debate statement + the panel conclusion
3. The Orchestrator persona joins too, with the conclusion and every floor-grant
   audit event

Nothing is recomputed. The record is read back exactly as it was locked.

### Question routing

Three modes:

- **Addressed** — the UI lets the recruiter pick a target ("ask Product"). Route
  directly.
- **Open** — no target. The Orchestrator answers, and may cite a specific agent.
- **Counterfactual** — recruiter supplies a hypothetical answer at a specific
  turn. Re-run only that agent's scoring for the affected claims, return the
  diff. Write a `what_if_query` row.

**Counterfactuals never mutate `agent_score`.** The hash must still verify after
any number of what-if questions.

### Which model answers

Most recruiter questions need no new judgment. Nemotron already reasoned at lock
time, and its conclusion, reasoning and debate statements are stored as text.
Retrieving and phrasing that is reading comprehension, not judgment — so the fast
model handles it.

| Question | Model | Latency |
|---|---|---|
| "Why did you recommend X?" | gpt-oss | ~1 s |
| "What did he say at 11:20?" | gpt-oss | ~1 s |
| "What was Product's objection?" | gpt-oss | ~1 s |
| "If he had said X, would that change?" | Nemotron | ~2–3 s |

gpt-oss is a **spokesperson, not a judge**. It reads out a decision that was made
and recorded hours earlier.

**Grounding guardrail — mandatory.** The Ask the Panel prompt must instruct:
answer only from the stored conclusion, debate statements, ledger and audit log;
if it is not in the record, say so. Without this the model will invent reasoning
the panel never had, at exactly the moment a recruiter is testing whether the
scores can be trusted.

Counterfactual latency is fine unhedged — a recruiter asking "would a better
answer have changed your mind?" expects a pause. Scope the re-score to the
affected claims only, not the whole competency.

### Override

Recruiter confirms or overrules. Write an `override` row with the reason, linked
to the interview hash. The original recommendation stays visible forever
alongside it.

---

## 8. Concurrency and isolation

One interview = one Agora channel + one media worker process + one row set.

- **Partition key is `interview_id`, everywhere.** Every Redis key is
  `iv:{interview_id}:*`. Every Postgres row has the FK. No exceptions.
- **Media workers are ephemeral and single-tenant.** Spawned on interview start,
  killed on end. A crashed worker takes down one interview, never two.
- **The Orchestrator service is stateless and horizontally scalable.** All live
  turn state lives in Redis; all durable state in Postgres. Any instance can
  serve any request.
- **Agora channel name derives from `interview_id`.** Tokens are scoped per
  channel per participant with short TTL, so a leaked token grants nothing
  outside one interview.
- **LLM rate limits are global**, not per interview. The router needs a shared
  token bucket in Redis plus exponential backoff, or twenty concurrent interviews
  will collectively trip the provider's per-minute cap. This is the real scaling
  constraint — not CPU.

---

## 9. JD and resume ingestion

Parsed **once**, at interview creation. Never per turn.

```
JD + resume → dossier
  → required competencies with weights   (seeds `competency` table)
  → panel composition                    (which interviewers to spawn)
  → per-agent rubrics                    (what "good" means for this role)
  → resume claims as UNVERIFIED ledger entries
```

Panel composition is dynamic: a backend role might spawn Technical, Coding and
Hiring Manager but not Customer. A support-engineering role would do the
opposite.

The pre-registered resume claims are the quiet win. When the candidate says
something that supports or contradicts a resume claim, the ledger links them
automatically, and contradiction detection covers "what he wrote" versus "what he
said" without any extra machinery.

---

## 10. Failure modes

| Failure | Response |
|---|---|
| LLM 429 | Fallback chain; shared Redis token bucket; exponential backoff |
| LLM slow (>2s) | Emit a short filler line in the agent's voice, keep waiting |
| Malformed JSON from bid call | Retry once, then default `interest = 0.3` and continue |
| Free model delisted | Router falls through to next ID; alert on fallback rate |
| STT drops | Buffer audio, retry; never silently lose a turn |
| TTS fails | Fall back to on-screen captions; the interview continues |
| Agora disconnect | Reconnect, resume from Redis turn state |
| Worker crash | Interview marked `INTERRUPTED`; transcript up to that point is preserved and hashable |

**Demo-day rule:** do not run on a free-tier model. Put a small balance on a paid
model and flip a config flag before presenting. A `:free` endpoint being
rate-limited mid-demo is a bad way to lose.

---

## 11. Build phases

Each phase ends in something demoable. Do not start a phase before the previous
one runs end to end.

**Phase 1 — Audio spine.**
Web client joins an Agora channel. Media worker joins, receives audio, runs Silero
VAD, transcribes with Sarvam, synthesises a hardcoded reply with Flux, publishes
it back. No LLM. Success: you speak, it echoes back in a synthetic voice.

**Phase 2 — One interviewer.**
Add Smart Turn v3.1 and one LLM-backed interviewer with a system prompt. Add
interruption handling with truncation logging. Success: a real, interruptible
one-on-one voice interview.

**Phase 3 — The panel.**
Five agents with distinct prompts and voices. Parallel bid calls. Floor control
with the priority formula. Prompt-builder isolation invariant. RTM speaker
signals and five tiles in the UI. Success: the PS11 example scenario works — a
technically correct answer with no customer impact routes to Product.

**Phase 4 — Evidence ledger.**
Claim extraction inside the bid call. Competency coverage map. Contradiction and
vagueness linking. Live coverage bars on an internal debug view. Success: every
claim has a timestamp and a competency.

**Phase 5 — Lock, debate, conclusion.**
Blind scoring passes. Deterministic conviction. Hashing. Sequential debate with
code-enforced HOLD. Orchestrator conclusion. Report page with the split and
evidence links. Success: a full interview produces a defensible report.

**Phase 6 — Ask the Panel.**
Session revival, question routing, counterfactual re-scoring, override logging.
Success: a recruiter rejoins a completed interview and gets timestamped answers.

**Phase 7 — Dossier.**
JD and resume upload, competency seeding, panel auto-composition, rubric
generation, resume claims as UNVERIFIED entries.

**Phase 8 — Polish.**
Coding interviewer with screen-share frames into a vision model. Confidence
trajectory chart. Spoken AI disclosure and persistent badge. Recruiter dashboard.

Phases 1–6 are the product. Phases 7–8 are what make it a demo people remember.

---

## 12. Configuration

```
AGORA_APP_ID, AGORA_APP_CERTIFICATE
SARVAM_API_KEY
DEEPGRAM_API_KEY
OPENROUTER_API_KEY
LLM_FAST_MODEL          openai/gpt-oss-120b
LLM_REASONING_MODEL     nvidia/nemotron-3-ultra-550b-a55b
LLM_FALLBACK_CHAIN      comma-separated ids
VAD_STOP_SECS           0.2
SMART_TURN_MODEL_PATH   ./models/smart-turn-v3.1.onnx
INTERVIEW_TIME_BUDGET_S 1200
COVERAGE_LAMBDA_START   0.5
COVERAGE_LAMBDA_END     1.5
CONVICTION_MARGIN       0.25
CONVICTION_MIN_EVIDENCE 3
DEMO_MODE               false   # true = paid model IDs, no free tier
```

Voice assignment (Deepgram Flux, `flux-{voice}-en`, plus `expressivity` −2…2):

| Agent | Accent | Expressivity |
|---|---|---|
| Technical | Indian | −1 |
| Product | American | +1 |
| Hiring Manager | British | 0 |
| Customer | Indian | +2 |
| Coding | Singaporean | −1 |
| Orchestrator | British | −2 |

Pick the specific voice ids from Deepgram's Flux catalog. Vary pace and pitch,
not just gender — the candidate must never wonder who is speaking.

---

## 13. Non-negotiables

1. The Orchestrator exposes an OpenAI-compatible endpoint.
2. Agent isolation is enforced by the prompt builder, not by convention.
3. Floor control is deterministic code, never a model. Same inputs, same winner,
   always — no randomised tie-breaks.
4. Ask the Panel answers only from the stored record. If it is not written down,
   the answer is "I don't have that."
5. Conviction is computed deterministically; a STRONG hold cannot be overridden
   by model output.
6. Scores are never averaged. The split is the output.
7. Counterfactuals never mutate the locked record.
8. Every score links to a transcript span and timestamp.
9. Low confidence reallocates time; it never ends an interview.
10. The system recommends. A human decides. Every override is logged.
