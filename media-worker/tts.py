"""Text-to-speech via Deepgram Aura.

We ask Deepgram for raw linear16 PCM at our working sample rate so the bytes can
be pushed straight into Agora with no resampling or container parsing.

(Note: the architecture doc calls this "Flux". Deepgram's TTS product is actually
"Aura"; "Flux" is their STT turn model. We use Aura here and keep the voice name
in config so it's a one-line change if that ever moves.)
"""
import requests

DEEPGRAM_SPEAK_URL = "https://api.deepgram.com/v1/speak"


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
