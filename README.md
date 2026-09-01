# 968ms — Coordinated AI Interview Panel

An adaptive AI interview panel. See [ARCHITECTURE.md](ARCHITECTURE.md) for the full design.

This repo is being built **phase by phase**. Current status: **Phase 2 — One interviewer.**

---

## Phase 2: what works

You open a web page and join. A real interviewer (the Hiring Manager) greets you,
discloses that it's AI, and asks an opening question. You answer out loud; it
listens, decides you're actually done (not just pausing mid-thought), and asks a
natural follow-up. You can **talk over it to interrupt** — it stops immediately
and the record notes exactly how much of its question you actually heard.

```
your mic → Agora → media worker → Silero VAD → Smart Turn v3.1 → Sarvam STT
                                                                      ↓
                                                              transcript turn
                                                                      ↓
        speaker ← Agora ← media worker ← Deepgram Aura TTS ← LLM (gpt-oss)
```

- **Smart Turn v3.1** classifies each pause as complete/incomplete, so a short
  VAD stop doesn't cut people off mid-sentence.
- **One LLM interviewer** with a system persona, driven through the shared LLM
  router (`shared/llm_router.py`) with a fallback chain.
- **Interruption + truncation:** barge-in stops playback and marks the agent's
  transcript turn `truncated` at the character actually delivered, plus an audit
  event — so a later "what did it ask?" cites what was heard, not what was
  generated.

### Phase 1: what worked

The audio spine (mic → VAD → STT → TTS → speaker) — an echo bot, no LLM. Phase 2
keeps that spine and puts a brain on it.

### Why Docker?

Agora's server-side audio SDK only ships Linux/macOS binaries — it can't run on
native Windows. So the **media worker** runs in a Linux container. Your browser
still runs natively on Windows. Everything else (config, tokens) is plain Python.

---

## Layout

```
media-worker/   FastAPI app: Agora join, VAD, STT, TTS, echo pipeline (runs in Docker)
web/            Candidate web page (Agora Web SDK)  — served by the media worker
shared/         Config loader, Agora token builder, Pydantic models
venv/           Local Python venv (Windows) for tooling/tests
```

---

## Running Phase 1

### 1. Add your API keys

Copy the template and fill it in:

```bash
cp .env.example .env
```

You need these keys in `.env`:

| Key | Where to get it |
|---|---|
| `AGORA_APP_ID` + `AGORA_APP_CERTIFICATE` | Agora Console → your project (enable the App Certificate under Security) |
| `SARVAM_API_KEY` | Sarvam dashboard |
| `DEEPGRAM_API_KEY` | Deepgram console |
| `OPENROUTER_API_KEY` | OpenRouter dashboard (new in Phase 2 — the interviewer's brain) |
| `GROQ_API_KEY` | Groq console (optional — makes the fast path sub-second via gpt-oss-120b; without it the router uses free OpenRouter models) |

### 2. Get the Smart Turn weights (optional but recommended)

```bash
python models/download_smart_turn.py
```

Without it the interview still runs; it just treats every pause as end-of-turn.
See [models/README.md](models/README.md).

### 3. Build and run the media worker

```bash
docker compose up --build
```

First build downloads torch + the Agora SDK, so it takes a few minutes. When it's
up you'll see `Uvicorn running on http://0.0.0.0:8080`.

### 4. Open the page

Go to **http://localhost:8080** in your browser. Click **Join Interview** and
allow the microphone. The interviewer greets you and asks a question — answer,
then pause. Try talking over it to interrupt.

Watch the container logs (`docker compose` output) to see each stage:
`candidate turn → Smart Turn p(complete)=… → heard: '…' → Hiring Manager speaks`
(and `INTERRUPTED: …` when you barge in).

---

## Notes / decisions

- **"Flux" vs "Aura":** the architecture doc calls the TTS "Deepgram Flux", but
  Deepgram's TTS product is actually **Aura** (`aura-2-thalia-en`). "Flux" is
  their STT turn model. We use Aura; the voice name lives in [media-worker/tts.py](media-worker/tts.py).
- **One session per process** still. Multi-tenant workers come later.
- **LLM routing:** every model call goes through `shared/llm_router.py` with a
  cross-provider fallback chain — never hardcode a model id at a call site. Model
  ids carry a provider prefix (`groq:` / `openrouter:`; unprefixed = OpenRouter).
  The fast path is `groq:openai/gpt-oss-120b` (sub-second); until a `GROQ_API_KEY`
  is set the router falls through to free OpenRouter models automatically. Prompts
  are built in one place, `shared/prompts.build_agent_prompt`.
- **The panel** (five interviewers, voices, floor control) arrives in **Phase 3**.
- **Frontend is still vanilla HTML on purpose.** The architecture calls for React,
  and it arrives in **Phase 3** — that's where the UI first has real state (five
  tiles, active-speaker highlight via RTM, captions). Through Phase 2 the page is
  just join + mic, so React would be ceremony with nothing to manage. Don't
  convert it early.
