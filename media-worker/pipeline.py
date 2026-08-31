"""The Phase 1 "brain" - except there's no brain yet, just an echo.

Runs in its own thread so the Agora audio callback is never blocked. Pulls raw
PCM off the session's inbound queue, feeds it to the VAD, and when the candidate
finishes an utterance:

    utterance PCM --> Sarvam STT --> canned reply text --> Deepgram TTS --> speak

No LLM. The reply just repeats what it heard, which proves the whole audio
spine (mic -> VAD -> STT -> TTS -> speaker) works end to end.
"""
import logging
import threading

from agora_session import AgoraSession
from vad import SileroVAD
import stt
import tts

logger = logging.getLogger(__name__)


class EchoPipeline:
    def __init__(
        self,
        session: AgoraSession,
        sarvam_key: str,
        deepgram_key: str,
        sample_rate: int = 16000,
        stop_secs: float = 0.6,
    ):
        self.session = session
        self.sarvam_key = sarvam_key
        self.deepgram_key = deepgram_key
        self.sample_rate = sample_rate
        self.vad = SileroVAD(sample_rate=sample_rate, stop_secs=stop_secs)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="pipeline")

    def start(self):
        self._stop.clear()
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=2)

    def _run(self):
        logger.info("EchoPipeline running")
        while not self._stop.is_set():
            try:
                pcm = self.session.inbound.get(timeout=0.2)
            except Exception:
                continue
            utterance = self.vad.process(pcm)
            if utterance:
                self._handle_utterance(utterance)

    def _handle_utterance(self, pcm: bytes):
        secs = len(pcm) / (self.sample_rate * 2)
        logger.info(f"utterance ended: {secs:.1f}s of speech")
        try:
            transcript = stt.transcribe(pcm, self.sarvam_key, self.sample_rate)
        except Exception:
            logger.exception("STT failed")
            return
        if not transcript:
            logger.info("empty transcript, skipping reply")
            return
        logger.info(f"heard: {transcript!r}")

        reply = f"You said: {transcript}"
        try:
            audio = tts.synthesize(reply, self.deepgram_key, sample_rate=self.sample_rate)
        except Exception:
            logger.exception("TTS failed")
            return
        logger.info(f"speaking reply ({len(audio)} bytes PCM)")
        self.session.enqueue_playback(audio)
