"""Speech-to-text via Sarvam.

We collect an utterance as raw 16 kHz mono PCM, wrap it in a WAV container
(Sarvam wants a real audio file), and POST it. Returns the transcript string.
"""
import io
import wave

import requests

SARVAM_STT_URL = "https://api.sarvam.ai/speech-to-text"


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
) -> str:
    wav_bytes = _pcm_to_wav(pcm_bytes, sample_rate)
    resp = requests.post(
        SARVAM_STT_URL,
        headers={"api-subscription-key": api_key},
        files={"file": ("utterance.wav", wav_bytes, "audio/wav")},
        data={"model": model, "language_code": language_code},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json().get("transcript", "").strip()
