"""Text-to-speech via Deepgram Aura.

We ask Deepgram for raw linear16 PCM at our working sample rate so the bytes can
be pushed straight into Agora with no resampling or container parsing.

Two paths:
  - `speak_stream()` — the WebSocket API, which streams audio *as it is
    generated* (first bytes in ~300 ms). This is what the live turn loop uses:
    time-to-first-audio stops depending on the length of the whole sentence.
  - `synthesize()` — the plain REST call, which returns the full clip at once
    (~2.5 s for a sentence). Kept as the fallback if the socket can't be used.

(Note: the architecture doc calls this "Flux". Deepgram's TTS product is actually
"Aura"; "Flux" is their STT turn model. We use Aura here and keep the voice name
in config so it's a one-line change if that ever moves.)
"""
import json
import logging
from typing import Callable, Optional
from urllib.parse import urlencode

import requests

logger = logging.getLogger(__name__)

DEEPGRAM_SPEAK_URL = "https://api.deepgram.com/v1/speak"
DEEPGRAM_SPEAK_WS = "wss://api.deepgram.com/v1/speak"


def synthesize(
    text: str,
    api_key: str,
    model: str = "aura-2-thalia-en",
    sample_rate: int = 16000,
) -> bytes:
    """Return raw linear16 PCM (mono, `sample_rate` Hz) for the given text."""
    resp = requests.post(
        DEEPGRAM_SPEAK_URL,
        params={
            "model": model,
            "encoding": "linear16",
            "sample_rate": str(sample_rate),
            "container": "none",   # raw PCM, no WAV header
        },
        headers={
            "Authorization": f"Token {api_key}",
            "Content-Type": "application/json",
        },
        json={"text": text},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.content


def speak_stream(
    text: str,
    api_key: str,
    on_chunk: Callable[[bytes], None],
    *,
    model: str = "aura-2-thalia-en",
    sample_rate: int = 16000,
    is_active: Optional[Callable[[], bool]] = None,
) -> int:
    """Stream PCM for `text`, calling `on_chunk(pcm)` as audio arrives.

    Returns total bytes emitted. If `is_active` is given and returns False, we
    stop early (the turn was interrupted). Falls back to the REST call if the
    WebSocket path is unavailable, so a socket problem never drops a turn.
    """
    try:
        return _stream_ws(text, api_key, on_chunk, model, sample_rate, is_active)
    except Exception:
        logger.warning("TTS stream failed; falling back to REST", exc_info=True)
        audio = synthesize(text, api_key, model=model, sample_rate=sample_rate)
        if audio:
            on_chunk(audio)
        return len(audio)


def _stream_ws(text, api_key, on_chunk, model, sample_rate, is_active) -> int:
    import websocket  # websocket-client (sync); imported lazily so REST still works

    qs = urlencode({
        "model": model,
        "encoding": "linear16",
        "sample_rate": str(sample_rate),
        "container": "none",
    })
    ws = websocket.create_connection(
        f"{DEEPGRAM_SPEAK_WS}?{qs}",
        header=[f"Authorization: Token {api_key}"],
        timeout=30,
    )
    total = 0
    try:
        ws.send(json.dumps({"type": "Speak", "text": text}))
        ws.send(json.dumps({"type": "Flush"}))
        while True:
            if is_active is not None and not is_active():
                break  # interrupted — stop pulling audio we won't play
            frame = ws.recv()
            if isinstance(frame, (bytes, bytearray)):
                if frame:
                    on_chunk(bytes(frame))
                    total += len(frame)
            else:  # text frame = JSON control message
                msg = json.loads(frame)
                if msg.get("type") == "Flushed":
                    break
    finally:
        try:
            ws.send(json.dumps({"type": "Close"}))
        except Exception:
            pass
        ws.close()
    return total
