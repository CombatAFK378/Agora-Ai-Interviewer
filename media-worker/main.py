"""Media worker HTTP surface (Phase 2).

Endpoints:
  GET  /               -> serves the candidate web page
  GET  /health         -> liveness check
  POST /session/start  -> mints Agora tokens, joins the bot, starts the pipeline,
                          and returns what the browser needs to join the channel

We still run exactly one interview session per process (multi-tenant workers
come later). Phase 2 swaps the echo pipeline for one LLM-backed interviewer.
"""
import logging
import os
import uuid

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from shared.config import get_settings
from shared.agora_token import build_rtc_token
from shared.models import SessionStartResponse

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

WEB_DIR = os.environ.get("WEB_DIR", os.path.join(os.path.dirname(__file__), "..", "web"))


@app.get("/")
def index():
    return FileResponse(os.path.join(WEB_DIR, "index.html"))


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
