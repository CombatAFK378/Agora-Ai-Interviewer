"""Speech-to-text via Sarvam.

We collect an utterance as raw 16 kHz mono PCM, wrap it in a WAV container
(Sarvam wants a real audio file), and POST it. Returns the transcript string.

The free tier rate-limits (429) under a fast back-and-forth, so we retry a couple
of times with backoff before giving up (ARCHITECTURE §10: "STT drops — buffer
audio, retry; never silently lose a turn").
"""
import io
import logging
import time
import wave

import requests

logger = logging.getLogger(__name__)

SARVAM_STT_URL = "https://api.sarvam.ai/speech-to-text"
_RETRYABLE_STATUS = {408, 425, 429, 500, 502, 503, 504}
# Sarvam's sync STT rejects audio longer than ~30 s (400). Split long turns into
# pieces just under that and stitch the transcripts back together.
STT_MAX_SECS = 29


def _pcm_to_wav(pcm_bytes: bytes, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)          # 16-bit
        wav.setframerate(sample_rate)
        wav.writeframes(pcm_bytes)
    return buf.getvalue()


def transcribe(
    pcm_bytes: bytes,
    api_key: str,
    sample_rate: int = 16000,
    model: str = "saaras:v3",
    language_code: str = "en-IN",
    retries: int = 2,
) -> str:
    """Transcribe an utterance. Turns longer than Sarvam's ~30 s limit are split
    into ≤29 s chunks and the transcripts joined, so a long monologue isn't lost."""
    max_bytes = STT_MAX_SECS * sample_rate * 2
    if len(pcm_bytes) > max_bytes:
        parts = []
        for i in range(0, len(pcm_bytes), max_bytes):
            piece = _transcribe_one(pcm_bytes[i:i + max_bytes], api_key,
                                    sample_rate, model, language_code, retries)
            if piece:
                parts.append(piece)
        return " ".join(parts).strip()
    return _transcribe_one(pcm_bytes, api_key, sample_rate, model, language_code, retries)


def _transcribe_one(
    pcm_bytes: bytes,
    api_key: str,
    sample_rate: int,
    model: str,
    language_code: str,
    retries: int,
) -> str:
    wav_bytes = _pcm_to_wav(pcm_bytes, sample_rate)
    for attempt in range(retries + 1):
        resp = requests.post(
            SARVAM_STT_URL,
            headers={"api-subscription-key": api_key},
            files={"file": ("utterance.wav", wav_bytes, "audio/wav")},
            data={"model": model, "language_code": language_code},
            timeout=30,
        )
        if resp.status_code in _RETRYABLE_STATUS and attempt < retries:
            backoff = min(0.5 * (2 ** attempt), 4.0)
            logger.warning("Sarvam STT %s; retry %d/%d in %.1fs",
                           resp.status_code, attempt + 1, retries, backoff)
            time.sleep(backoff)
            continue
        resp.raise_for_status()
        return resp.json().get("transcript", "").strip()
    return ""
