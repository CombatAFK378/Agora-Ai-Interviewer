# 968ms — Coordinated AI Interview Panel

An adaptive AI interview panel. See [ARCHITECTURE.md](ARCHITECTURE.md) for the full design.

This repo is being built **phase by phase**. Current status: **Phase 1 — Audio spine.**

---

## Phase 1: what works

You open a web page, join, and talk. A bot (running in a Linux Docker container)
hears you, transcribes your speech, and reads it back in a synthetic voice. No AI
brain yet — this proves the whole audio path end to end:

```
your mic → Agora → media worker → Silero VAD → Sarvam STT
                                                    ↓
        speaker ← Agora ← media worker ← Deepgram Aura TTS ← "You said: …"
```

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

You need four keys in `.env`:

| Key | Where to get it |
|---|---|
| `AGORA_APP_ID` + `AGORA_APP_CERTIFICATE` | Agora Console → your project (enable the App Certificate under Security) |
| `SARVAM_API_KEY` | Sarvam dashboard |
| `DEEPGRAM_API_KEY` | Deepgram console |

### 2. Build and run the media worker

```bash
docker compose up --build
```

First build downloads torch + the Agora SDK, so it takes a few minutes. When it's
up you'll see `Uvicorn running on http://0.0.0.0:8080`.

### 3. Open the page

Go to **http://localhost:8080** in your browser. Click **Join Interview**, allow
the microphone, say a sentence, then pause. Within ~1–2 seconds the bot repeats
what you said.

Watch the container logs (`docker compose` output) to see each stage:
`utterance ended → heard: '…' → speaking reply`.

---

## Notes / decisions

- **"Flux" vs "Aura":** the architecture doc calls the TTS "Deepgram Flux", but
  Deepgram's TTS product is actually **Aura** (`aura-2-thalia-en`). "Flux" is
  their STT turn model. We use Aura; the voice name lives in [media-worker/tts.py](media-worker/tts.py).
- **One session per process** in Phase 1. Multi-tenant workers come later.
- **Smart Turn v3.1** (better end-of-turn detection) and a real LLM interviewer
  arrive in **Phase 2**.
