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
import io
import logging
import os
import uuid

from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from shared.config import get_settings
from shared.agora_token import build_rtc_token
from shared.models import AskAnswer, Dossier, InterviewReport, Override, SessionStartResponse, WhatIfQuery
from shared import ask_panel, dossier as dossier_mod, prompts, scoring, store

from agora_session import AgoraSession
from pipeline import InterviewPipeline
from ask_pipeline import AskPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("media-worker")

BOT_UID = 1
CANDIDATE_UID = 100
RECRUITER_UID = 200
TOKEN_TTL_SECONDS = 3600

app = FastAPI(title="968ms media worker")

# Held for the lifetime of the single active session.
_session: AgoraSession | None = None
_pipeline: InterviewPipeline | None = None
_ask_pipeline: AskPipeline | None = None   # voice Ask the Panel (Phase 6)
_report: InterviewReport | None = None   # last locked report (Phase 5)
_panel_record: ask_panel.PanelRecord | None = None   # for Ask the Panel (Phase 6)
_dossier: Dossier | None = None          # active interview's dossier (Phase 7)


class StartRequest(BaseModel):
    jd: str = ""                  # job description (§9); empty → generic all-panel
    resume: str = ""             # candidate résumé (§9)


class DossierRequest(BaseModel):
    jd: str = ""
    resume: str = ""


class JoinRequest(BaseModel):
    interview_id: str | None = None   # revive a stored interview; else the last one


class AskRequest(BaseModel):
    question: str
    mode: str = "open"           # open | addressed
    target: str | None = None    # agent id, if addressed


class CounterfactualRequest(BaseModel):
    turn: int
    hypothetical: str
    agent_id: str


class OverrideRequest(BaseModel):
    decision: str
    reason: str = ""

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


MAX_PDF_BYTES = 10 * 1024 * 1024   # 10 MB — a JD/résumé is never bigger


@app.post("/parse-pdf")
async def parse_pdf(file: UploadFile = File(...)):
    """Extract plain text from an uploaded JD or résumé PDF (§9). Text is returned
    to the client so it can be reviewed/edited, then sent back with the dossier."""
    from pypdf import PdfReader
    name = (file.filename or "").lower()
    if not name.endswith(".pdf") and (file.content_type or "") != "application/pdf":
        raise HTTPException(400, "please upload a PDF file")
    data = await file.read()
    if len(data) > MAX_PDF_BYTES:
        raise HTTPException(413, "PDF too large (max 10 MB)")
    try:
        reader = PdfReader(io.BytesIO(data))
        text = "\n".join((page.extract_text() or "") for page in reader.pages).strip()
    except Exception as e:
        logger.warning("PDF parse failed: %s", e)
        raise HTTPException(400, "couldn't read that PDF — is it a scanned image?")
    if not text:
        raise HTTPException(422, "no selectable text found (scanned/image-only PDF)")
    return {"filename": file.filename, "pages": len(reader.pages), "text": text}


@app.post("/session/dossier", response_model=Dossier)
def session_dossier(req: DossierRequest):
    """Preview the parsed dossier (§9) before starting — panel, weights, rubrics,
    résumé claims — so the recruiter can review the panel the JD/résumé produced."""
    if not (req.jd.strip() or req.resume.strip()):
        raise HTTPException(400, "provide a JD and/or résumé to parse")
    return dossier_mod.build_dossier(req.jd, req.resume)


@app.post("/session/start", response_model=SessionStartResponse)
def session_start(req: StartRequest | None = None):
    global _session, _pipeline, _report, _panel_record, _dossier
    settings = get_settings()
    req = req or StartRequest()

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

    # Tear down any previous session (one per process) and clear the last report.
    _teardown()
    _report = None
    _panel_record = None

    # Parse JD + résumé into the dossier once (§9): panel, weights, rubrics,
    # résumé claims. Empty inputs → generic all-panel interview.
    _dossier = None
    if req.jd.strip() or req.resume.strip():
        _dossier = dossier_mod.build_dossier(req.jd, req.resume)
        logger.info("dossier: role=%r panel=%s", _dossier.role, _dossier.panel)

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
        dossier=_dossier,
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
        panel=list(pipeline.panel),
    )


class FrameRequest(BaseModel):
    frame: str   # data: URI of a screen-share JPEG/PNG


@app.post("/coding/frame")
def coding_frame(req: FrameRequest):
    """Latest screen-share frame from the browser during the coding round (§8)."""
    if _pipeline is None:
        raise HTTPException(409, "no active session")
    _pipeline.set_frame(req.frame)
    return {"status": "ok"}


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
    _pipeline.assess_coding()   # §8: thorough vision pass on the final screen → coding evidence
    transcript, claims, coverage, panel, contexts = _pipeline.report_inputs()
    if not any(t.speaker == "candidate" for t in transcript):
        raise HTTPException(400, "no candidate answers to score yet")
    global _panel_record
    logger.info("concluding interview %s", _pipeline.interview_id)
    _report = scoring.build_report(_pipeline.interview_id, panel, transcript, claims,
                                   coverage, contexts)
    _report.trajectory = _pipeline.trajectory()   # per-turn confidence chart (§11)
    # Retain the full record so the recruiter can interrogate it (Phase 6).
    _panel_record = ask_panel.PanelRecord(
        interview_id=_pipeline.interview_id, report=_report,
        transcript=transcript, claims=claims, contexts=contexts,
        audit=[a.model_dump() for a in _pipeline.audit],
    )
    # Persist for the recruiter dashboard (§11) — survives restarts.
    try:
        store.save_record(
            _panel_record,
            candidate_name=(_dossier.candidate_name if _dossier else ""),
            role=(_dossier.role if _dossier else ""),
        )
    except Exception:
        logger.exception("failed to persist interview %s", _pipeline.interview_id)
    return _report


@app.post("/panel/join", response_model=SessionStartResponse)
def panel_join(req: JoinRequest | None = None):
    """Voice Ask the Panel (§7): the recruiter joins a channel and talks to the
    panel about the locked record. Tears down the interview session and starts a
    recruiter Q&A session on a fresh channel.

    With an `interview_id`, revives that stored interview (dashboard rejoin);
    otherwise uses the interview just concluded in this process."""
    global _session, _pipeline, _ask_pipeline, _panel_record
    req = req or JoinRequest()
    if req.interview_id and (_panel_record is None
                             or _panel_record.interview_id != req.interview_id):
        loaded = store.load_record(req.interview_id)
        if loaded is None:
            raise HTTPException(404, "interview not found")
        _panel_record = loaded
    if _panel_record is None:
        raise HTTPException(409, "no locked report yet — finish & score first")
    settings = get_settings()

    _teardown()  # the interview is over; reclaim the single AgoraService

    channel = f"ask-{_panel_record.interview_id[:8]}"
    bot_token = build_rtc_token(settings.agora_app_id, settings.agora_app_certificate,
                                channel, BOT_UID, TOKEN_TTL_SECONDS)
    recruiter_token = build_rtc_token(settings.agora_app_id, settings.agora_app_certificate,
                                      channel, RECRUITER_UID, TOKEN_TTL_SECONDS)

    session = AgoraSession(app_id=settings.agora_app_id, channel=channel,
                           bot_uid=BOT_UID, token=bot_token, sample_rate=settings.audio_sample_rate)
    ask = AskPipeline(session=session, settings=settings, record=_panel_record,
                      on_event=_broker.publish)
    session.set_on_user_joined(ask.on_recruiter_joined)
    session.start()
    ask.start()
    _session, _ask_pipeline = session, ask
    logger.info("panel voice session started on channel %s", channel)
    return SessionStartResponse(app_id=settings.agora_app_id, channel=channel,
                                uid=RECRUITER_UID, token=recruiter_token)


@app.post("/panel/ask", response_model=AskAnswer)
def panel_ask(req: AskRequest):
    """Ask the Panel (§7): grounded Q&A over the locked record."""
    if _panel_record is None:
        raise HTTPException(409, "no locked report yet — finish & score first")
    return ask_panel.answer(_panel_record, req.question, req.mode, req.target)


@app.post("/panel/counterfactual", response_model=WhatIfQuery)
def panel_counterfactual(req: CounterfactualRequest):
    """Counterfactual re-score (§7) — never mutates the locked scores."""
    if _panel_record is None:
        raise HTTPException(409, "no locked report yet")
    try:
        wq = ask_panel.counterfactual(_panel_record, req.turn, req.hypothetical, req.agent_id)
        _persist_panel_record()
        return wq
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/panel/override", response_model=Override)
def panel_override(req: OverrideRequest):
    """Log a recruiter override (§7). The original recommendation is kept."""
    if _panel_record is None:
        raise HTTPException(409, "no locked report yet")
    ov = ask_panel.override(_panel_record, req.decision, req.reason)
    _persist_panel_record()
    return ov


def _persist_panel_record():
    """Re-save the current record (metadata preserved) after a mutation."""
    if _panel_record is not None:
        try:
            store.save_record(_panel_record)
        except Exception:
            logger.exception("failed to persist override/what-if")


@app.get("/report", response_model=InterviewReport)
def get_report():
    if _report is None:
        raise HTTPException(404, "no report yet")
    return _report


# ---- Recruiter dashboard (Phase 8, §11) --------------------------------

@app.get("/interviews")
def list_interviews():
    """Rows for the recruiter dashboard: past interviews, newest first."""
    return store.list_summaries()


@app.post("/interviews/{interview_id}/open", response_model=InterviewReport)
def open_interview(interview_id: str):
    """Open a stored interview: make it the active record (so Ask-the-Panel,
    override, counterfactual and voice-join all operate on it) and return its
    locked report."""
    global _panel_record, _report
    rec = store.load_record(interview_id)
    if rec is None:
        raise HTTPException(404, "interview not found")
    _panel_record = rec
    _report = rec.report
    return rec.report


@app.post("/session/stop")
def session_stop():
    _teardown()
    return {"status": "stopped"}


def _teardown():
    global _session, _pipeline, _ask_pipeline
    if _pipeline is not None:
        _pipeline.stop()
        _pipeline = None
    if _ask_pipeline is not None:
        _ask_pipeline.stop()
        _ask_pipeline = None
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
