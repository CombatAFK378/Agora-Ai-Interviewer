"""Media worker HTTP surface (Phase 3).

Endpoints:
  GET  /                 -> serves the candidate web page (React panel room)
  GET  /health           -> liveness check
  GET  /panel            -> the five interviewers' roster (for the tiles)
  POST /session/start    -> mints Agora tokens, joins the bot, starts the pipeline
  POST /session/interrupt-> manual barge-in
  POST /session/stop     -> tear down
  WS   /session/events   -> live speaker signals (who's talking) + captions

One interview per process still. Phase 3 makes it a five-agent panel and streams
speaker signals to the UI over a WebSocket (our stand-in for Agora RTM).
"""
import asyncio
import logging
import os
import uuid

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

from shared.config import get_settings
from shared.agora_token import build_rtc_token
from shared.models import InterviewReport, SessionStartResponse
from shared import prompts, scoring

from agora_session import AgoraSession
from pipeline import InterviewPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("media-worker")

BOT_UID = 1
CANDIDATE_UID = 100
TOKEN_TTL_SECONDS = 3600

app = FastAPI(title="968ms media worker")

# Held for the lifetime of the single active session.
_session: AgoraSession | None = None
_pipeline: InterviewPipeline | None = None
_report: InterviewReport | None = None   # last locked report (Phase 5)

# The built Vite app (dist/). In Docker this is set to /app/web/dist. For local
# frontend dev, run `npm run dev` instead (Vite serves the app and proxies here).
WEB_DIR = os.environ.get(
    "WEB_DIR", os.path.join(os.path.dirname(__file__), "..", "web", "dist")
)


class EventBroker:
    """Bridges pipeline events (emitted from worker threads) to WebSocket clients
    (async). Each connection gets a queue; publish() hops onto the event loop
    thread-safely and drops events for a slow/full consumer rather than blocking."""

    def __init__(self):
        self._subs: set[asyncio.Queue] = set()
        self._loop: asyncio.AbstractEventLoop | None = None

    def bind_loop(self, loop: asyncio.AbstractEventLoop):
        self._loop = loop

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=128)
        self._subs.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue):
        self._subs.discard(q)

    def publish(self, event: dict):
        loop = self._loop
        if loop is None:
            return
        for q in list(self._subs):
            loop.call_soon_threadsafe(self._offer, q, event)

    @staticmethod
    def _offer(q: asyncio.Queue, event: dict):
        try:
            q.put_nowait(event)
        except asyncio.QueueFull:
            pass


_broker = EventBroker()


def _panel_roster() -> list[dict]:
    return [
        {"id": a, "name": prompts.AGENTS[a].name, "title": prompts.AGENTS[a].title}
        for a in prompts.PANEL_IDS
    ]


@app.get("/panel")
def panel():
    return {"agents": _panel_roster()}


@app.websocket("/session/events")
async def session_events(ws: WebSocket):
    await ws.accept()
    _broker.bind_loop(asyncio.get_running_loop())
    q = _broker.subscribe()
    try:
        await ws.send_json({"type": "panel", "agents": _panel_roster()})
        while True:
            event = await q.get()
            await ws.send_json(event)
    except WebSocketDisconnect:
        pass
    except Exception:
        logger.exception("events websocket error")
    finally:
        _broker.unsubscribe(q)


@app.get("/health")
def health():
    return {"status": "ok", "session_active": _session is not None}


@app.post("/session/start", response_model=SessionStartResponse)
def session_start():
    global _session, _pipeline
    settings = get_settings()

    missing = [
        name
        for name, val in {
            "AGORA_APP_ID": settings.agora_app_id,
            "AGORA_APP_CERTIFICATE": settings.agora_app_certificate,
            "SARVAM_API_KEY": settings.sarvam_api_key,
            "DEEPGRAM_API_KEY": settings.deepgram_api_key,
            "OPENROUTER_API_KEY": settings.openrouter_api_key,
        }.items()
        if not val
    ]
    if missing:
        raise HTTPException(500, f"Missing config: {', '.join(missing)}")

    # Tear down any previous session (one per process).
    _teardown()

    interview_id = uuid.uuid4().hex[:12]
    channel = f"iv-{interview_id[:8]}"
    sample_rate = settings.audio_sample_rate

    bot_token = build_rtc_token(
        settings.agora_app_id, settings.agora_app_certificate, channel, BOT_UID, TOKEN_TTL_SECONDS
    )
    candidate_token = build_rtc_token(
        settings.agora_app_id, settings.agora_app_certificate, channel, CANDIDATE_UID, TOKEN_TTL_SECONDS
    )

    session = AgoraSession(
        app_id=settings.agora_app_id,
        channel=channel,
        bot_uid=BOT_UID,
        token=bot_token,
        sample_rate=sample_rate,
    )
    pipeline = InterviewPipeline(
        session=session,
        settings=settings,
        interview_id=interview_id,
    )
    # The opener fires when the candidate joins the channel.
    session.set_on_user_joined(pipeline.on_candidate_joined)
    # Stream speaker signals / captions to the UI over the events WebSocket.
    pipeline.on_event = _broker.publish

    session.start()
    pipeline.start()

    _session, _pipeline = session, pipeline
    logger.info(f"session started: interview={interview_id} channel={channel}")

    return SessionStartResponse(
        app_id=settings.agora_app_id,
        channel=channel,
        uid=CANDIDATE_UID,
        token=candidate_token,
    )


@app.post("/session/interrupt")
def session_interrupt():
    """Candidate pressed the Interrupt button — cut the interviewer off now."""
    if _pipeline is None:
        raise HTTPException(409, "no active session")
    interrupted = _pipeline.request_interrupt()
    return {"status": "ok", "interrupted": interrupted}


@app.post("/session/conclude", response_model=InterviewReport)
def session_conclude():
    """Final bell (§6): freeze the interview, then lock → debate → conclusion.
    Blocking and slow (reasoning model, once) — the UI shows a 'scoring…' state."""
    global _report
    if _pipeline is None:
        raise HTTPException(409, "no active session")
    _pipeline.freeze()
    transcript, claims, coverage, panel = _pipeline.report_inputs()
    if not any(t.speaker == "candidate" for t in transcript):
        raise HTTPException(400, "no candidate answers to score yet")
    logger.info("concluding interview %s", _pipeline.interview_id)
    _report = scoring.build_report(_pipeline.interview_id, panel, transcript, claims, coverage)
    return _report


@app.get("/report", response_model=InterviewReport)
def get_report():
    if _report is None:
        raise HTTPException(404, "no report yet")
    return _report


@app.post("/session/stop")
def session_stop():
    _teardown()
    return {"status": "stopped"}


def _teardown():
    global _session, _pipeline
    if _pipeline is not None:
        _pipeline.stop()
        _pipeline = None
    if _session is not None:
        _session.stop()
        _session = None


# Serve the built React app (dist/) at "/". Mounted LAST so the API routes and
# the events WebSocket above take precedence; the SPA catches everything else.
# Absent in local frontend-dev mode (use `npm run dev`), so guard on existence.
if os.path.isdir(WEB_DIR):
    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
else:
    logger.warning("WEB_DIR %s not found; not serving the frontend "
                   "(run `cd web && npm run dev` for the React app)", WEB_DIR)
