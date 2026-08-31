"""Media worker HTTP surface (Phase 1).

Endpoints:
  GET  /               -> serves the candidate web page
  GET  /health         -> liveness check
  POST /session/start  -> mints Agora tokens, joins the bot, starts the pipeline,
                          and returns what the browser needs to join the channel

For Phase 1 we run exactly one interview session per process.
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
from pipeline import EchoPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("media-worker")

BOT_UID = 1
CANDIDATE_UID = 100
TOKEN_TTL_SECONDS = 3600

app = FastAPI(title="968ms media worker")

# Held for the lifetime of the single active session.
_session: AgoraSession | None = None
_pipeline: EchoPipeline | None = None

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
        }.items()
        if not val
    ]
    if missing:
        raise HTTPException(500, f"Missing config: {', '.join(missing)}")

    # Tear down any previous session (one per process in Phase 1).
    _teardown()

    channel = f"iv-{uuid.uuid4().hex[:8]}"
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
    session.start()

    pipeline = EchoPipeline(
        session=session,
        sarvam_key=settings.sarvam_api_key,
        deepgram_key=settings.deepgram_api_key,
        sample_rate=sample_rate,
        stop_secs=settings.vad_stop_secs,
    )
    pipeline.start()

    _session, _pipeline = session, pipeline
    logger.info(f"session started on channel {channel}")

    return SessionStartResponse(
        app_id=settings.agora_app_id,
        channel=channel,
        uid=CANDIDATE_UID,
        token=candidate_token,
    )


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
