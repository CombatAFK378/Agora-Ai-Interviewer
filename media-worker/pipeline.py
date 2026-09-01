"""Phase 2: one real interviewer.

The live turn loop for a single LLM-backed interviewer (ARCHITECTURE §4), minus
the panel/bidding/floor-control that arrive in Phase 3:

    candidate speech
      -> Silero VAD (pause)
      -> Smart Turn v3.1 (complete? else keep listening)
      -> Sarvam STT -> transcript_turn (candidate)
      -> LLM question generation (Hiring Manager persona)
      -> Deepgram TTS -> speak
      -> transcript_turn (agent)

Plus barge-in: if the candidate starts talking while the agent is speaking, we
stop playback immediately, mark the agent's turn `truncated` at the character the
candidate actually heard, and log an interruption audit event.

Everything runs on background threads so the VAD-feeding loop is never blocked —
that's what makes interruption responsive: the moment audio starts playing, the
loop is already watching for the candidate to cut in.

Transcript and audit live in memory for Phase 2; the storage layer is later.
"""
import logging
import queue
import re
import threading
import time

import numpy as np

from agora_session import AgoraSession
from smart_turn import SmartTurn
from vad import SileroVAD
import stt
import tts

from shared import llm_router, prompts
from shared.config import Settings
from shared.models import AuditEvent, TranscriptTurn

logger = logging.getLogger(__name__)

# A "complete" utterance never accumulates forever: if Smart Turn keeps saying
# "incomplete", force a turn once we've buffered this much speech.
PENDING_MAX_SECS = 15.0
# If Smart Turn said "incomplete" and the candidate then goes quiet for this long
# (no resumed speech), finalise the turn anyway — they really did stop.
PENDING_TIMEOUT_SECS = 2.0
# Let the browser finish subscribing to the bot's audio track before the opener,
# so the candidate actually hears it (ARCHITECTURE §5 cold start).
OPENER_DELAY_SECS = 1.5
# Below this RMS a VAD segment is background noise / speaker echo, not an answer —
# skip it rather than spend a rate-limited STT call on silence. int16 full scale
# is 32767; normal speech sits well above this floor.
MIN_SPEECH_RMS = 350.0
# The agent gets to deliver at least this much of its turn before a barge-in can
# cut it. Stops a stray onset (or the very first frame) from truncating it to
# zero, and lets short questions land.
BARGEIN_GRACE_SECS = 1.0


class InterviewPipeline:
    def __init__(
        self,
        session: AgoraSession,
        settings: Settings,
        interview_id: str,
        agent_id: str = prompts.HIRING_MANAGER.id,
    ):
        self.session = session
        self.settings = settings
        self.interview_id = interview_id
        self.agent_id = agent_id
        self.sample_rate = settings.audio_sample_rate
        self.allow_bargein = settings.allow_bargein

        self.vad = SileroVAD(
            sample_rate=self.sample_rate,
            stop_secs=settings.vad_stop_secs,
            threshold=0.7,      # stricter, so breaths/mouth noise don't trigger
            bargein_ms=750.0,   # need a real sustained utterance to interrupt,
                                # not a "hmm"/"okay"/cough backchannel
            # No barge-in callback in half-duplex mode (we also stop feeding the
            # VAD while the agent speaks, below) so echo can't derail the turn.
            on_speech_start=self._on_speech_start if self.allow_bargein else None,
        )
        self.smart_turn = SmartTurn(
            settings.smart_turn_model_path, settings.smart_turn_threshold
        )

        # Shared, in-memory for Phase 2.
        self.transcript: list[TranscriptTurn] = []
        self.audit: list[AuditEvent] = []
        self._tx_lock = threading.Lock()
        self._seq = 0

        # Accumulated speech Smart Turn judged "incomplete", awaiting more.
        self._pending = bytearray()
        self._pending_since: float | None = None

        # Reply bookkeeping.
        self._reply_lock = threading.Lock()
        self._replying = False
        self._opened = False
        # (turn, segments) for the turn currently being spoken, where segments is
        # a growing list of (cumulative_audio_bytes, cumulative_text_chars).
        self._current_reply: tuple[TranscriptTurn, list[tuple[int, int]]] | None = None

        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="pipeline")

    # ---- lifecycle ---------------------------------------------------------

    def start(self):
        self._stop.clear()
        self._thread.start()
        logger.info(
            "InterviewPipeline running (agent=%s, smart_turn=%s)",
            self.agent_id, "on" if self.smart_turn.enabled else "off",
        )

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=2)

    def request_interrupt(self) -> bool:
        """Manual barge-in from the candidate's Interrupt button.

        This is the *only* interrupt path in half-duplex mode (the normal mic
        never cuts the agent off). It works whether or not VAD barge-in is on.
        Returns True if the agent was actually speaking and got stopped.
        """
        if not self.session.is_speaking():
            return False
        logger.info("manual interrupt requested")
        self._handle_interruption()
        return True

    def on_candidate_joined(self, user_id=None):
        """Deliver the scripted opener, once, shortly after the candidate joins."""
        with self._reply_lock:
            if self._opened:
                return
            self._opened = True

        def _open():
            time.sleep(OPENER_DELAY_SECS)
            logger.info("delivering opener")
            self._start_reply(prompts.opening_line(self.agent_id))

        threading.Thread(target=_open, daemon=True, name="opener").start()

    # ---- main loop ---------------------------------------------------------

    def _run(self):
        while not self._stop.is_set():
            try:
                pcm = self.session.inbound.get(timeout=0.2)
            except queue.Empty:
                self._check_pending_timeout()
                continue
            # Half-duplex: while the agent speaks, drop incoming audio entirely so
            # the agent's own voice echoing into the mic can't become a phantom
            # turn or false barge-in. (Full-duplex keeps feeding the VAD.)
            if not self.allow_bargein and self.session.is_speaking():
                continue
            utterance = self.vad.process(pcm)
            if utterance is not None:
                self._on_vad_utterance(utterance)
            else:
                self._check_pending_timeout()

    def _on_vad_utterance(self, utterance: bytes):
        combined = bytes(self._pending) + utterance
        self._pending_since = None

        capped = len(combined) >= int(PENDING_MAX_SECS * self.sample_rate * 2)
        if capped or self.smart_turn.is_complete(combined):
            self._pending = bytearray()
            self._process_candidate_turn(combined)
        else:
            # Not done yet — keep listening, no LLM call (ARCHITECTURE §4).
            logger.info("Smart Turn: incomplete, keep listening")
            self._pending = bytearray(combined)
            self._pending_since = time.monotonic()

    def _check_pending_timeout(self):
        if not self._pending or self._pending_since is None:
            return
        if time.monotonic() - self._pending_since >= PENDING_TIMEOUT_SECS:
            logger.info("pending utterance timed out; finalising")
            pcm = bytes(self._pending)
            self._pending = bytearray()
            self._pending_since = None
            self._process_candidate_turn(pcm)

    def _process_candidate_turn(self, pcm: bytes):
        secs = len(pcm) / (self.sample_rate * 2)
        rms = _rms(pcm)
        if rms < MIN_SPEECH_RMS:
            logger.info("candidate turn: %.1fs but too quiet (rms=%.0f); skipping", secs, rms)
            return
        logger.info("candidate turn: %.1fs of speech (rms=%.0f)", secs, rms)
        try:
            text = stt.transcribe(pcm, self.settings.sarvam_api_key, self.sample_rate)
        except Exception:
            logger.exception("STT failed")
            return
        if not text:
            logger.info("empty transcript, skipping")
            return
        if _is_filler(text):
            logger.info("filler/backchannel %r; skipping (not a turn)", text)
            return
        logger.info("heard: %r", text)
        self._add_turn("candidate", text)
        self._start_reply(None)  # None -> generate the agent's next question

    # ---- reply (runs off the main loop so barge-in stays responsive) -------

    def _start_reply(self, scripted_text: str | None):
        with self._reply_lock:
            if self._replying:
                # Already speaking/generating; the current turn owns the floor.
                return
            self._replying = True
        threading.Thread(
            target=self._reply_worker, args=(scripted_text,), daemon=True, name="reply"
        ).start()

    def _reply_worker(self, scripted_text: str | None):
        try:
            text = scripted_text if scripted_text is not None else self._generate_reply()
            if text:
                self._speak_agent(text)
                # Hold the floor until playback finishes or is interrupted.
                while self.session.is_speaking() and not self._stop.is_set():
                    time.sleep(0.05)
        except Exception:
            logger.exception("reply failed")
        finally:
            with self._reply_lock:
                self._current_reply = None
                self._replying = False

    def _generate_reply(self) -> str:
        messages = prompts.build_agent_prompt(
            self.agent_id, "LIVE", self._snapshot_transcript()
        )
        try:
            return llm_router.chat(
                messages, model=self.settings.llm_fast_model,
                max_tokens=120, temperature=0.7, reasoning_effort="low",
            )
        except Exception:
            logger.exception("LLM generation failed; staying silent this turn")
            return ""

    def _speak_agent(self, text: str):
        """Synthesize and play the reply sentence by sentence (streamed TTS).

        Playing the first sentence while later ones are still synthesizing cuts
        time-to-first-audio from "whole reply" to "first sentence" (§4).
        `segments` records cumulative (audio_bytes, text_chars) so an interruption
        can map delivered audio back to the character the candidate actually heard.
        """
        agent = prompts.AGENTS[self.agent_id]
        turn = self._add_turn(self.agent_id, text)
        segments: list[tuple[int, int]] = []
        with self._reply_lock:
            self._current_reply = (turn, segments)

        self.session.begin_speech()
        cum_audio = 0
        cum_chars = 0
        for piece in _split_sentences(text):
            if self._stop.is_set() or not self.session.is_speaking():
                break  # interrupted — stop synthesizing the rest
            n = tts.speak_stream(
                piece, self.settings.deepgram_api_key,
                self.session.add_speech,
                model=agent.voice_model, sample_rate=self.sample_rate,
                is_active=self.session.is_speaking,
            )
            cum_chars += len(piece)
            if n <= 0:
                continue
            cum_audio += n
            segments.append((cum_audio, cum_chars))
        self.session.end_speech()

        if segments:
            logger.info(
                "%s speaks (%d chars, %d sentences, %d bytes PCM)",
                agent.title, len(text), len(segments), cum_audio,
            )
        else:
            logger.warning("no audio produced for reply")

    # ---- interruption ------------------------------------------------------

    def _on_speech_start(self):
        """Sustained speech from the candidate. If the agent is mid-turn (and past
        its grace window), the candidate is cutting in — an interruption (§4)."""
        if not self.session.is_speaking():
            return
        if self.session.speaking_elapsed() < BARGEIN_GRACE_SECS:
            return  # let the agent get its first words out; VAD will re-signal
        self._handle_interruption()

    def _handle_interruption(self):
        delivered = self.session.interrupt()
        with self._reply_lock:
            current = self._current_reply
            self._current_reply = None
        if current is None:
            return
        turn, segments = current
        char = _delivered_to_char(segments, delivered, len(turn.text))
        if char <= 0:
            # Nothing was actually heard — drop the phantom turn so it doesn't
            # pollute the transcript / LLM context.
            with self._tx_lock:
                if self.transcript and self.transcript[-1] is turn:
                    self.transcript.pop()
            logger.info("barge-in before any audio; dropped un-heard agent turn")
            return
        with self._tx_lock:
            turn.truncated = True
            turn.truncation_char = char
        total = segments[-1][0] if segments else 0
        logger.info(
            "INTERRUPTED: agent delivered %d/%d chars; heard=%r",
            char, len(turn.text), turn.text[:char],
        )
        self._audit("interruption", {
            "agent": self.agent_id,
            "turn_seq": turn.seq,
            "truncation_char": char,
            "delivered_bytes": delivered,
            "synthesized_bytes": total,
        })

    # ---- transcript / audit helpers ---------------------------------------

    def _add_turn(self, speaker: str, text: str) -> TranscriptTurn:
        with self._tx_lock:
            self._seq += 1
            turn = TranscriptTurn(seq=self._seq, speaker=speaker, text=text)
            self.transcript.append(turn)
        return turn

    def _snapshot_transcript(self) -> list[TranscriptTurn]:
        with self._tx_lock:
            return [t.model_copy() for t in self.transcript]

    def _audit(self, event_type: str, data: dict):
        self.audit.append(AuditEvent(interview_id=self.interview_id, type=event_type, data=data))


# Split on sentence-final punctuation, keeping the delimiter and trailing space
# so the pieces concatenate back to the exact original text (char offsets stay
# aligned for truncation). Short trailing fragments merge into the previous piece.
_SENTENCE_RE = re.compile(r".*?(?:[.!?]+(?:\s+|$)|$)", re.DOTALL)


def _split_sentences(text: str) -> list[str]:
    pieces = [m.group(0) for m in _SENTENCE_RE.finditer(text) if m.group(0)]
    merged: list[str] = []
    for p in pieces:
        # Don't ship a tiny fragment ("Dr.", "OK.") as its own TTS request.
        if merged and len(p.strip()) < 12:
            merged[-1] += p
        else:
            merged.append(p)
    return merged or [text]


# Standalone acknowledgements / noises that aren't real answers. If a whole turn
# is just one of these, it's a backchannel — don't make the interviewer respond
# to it (and don't let it count as the candidate's answer).
_FILLER = {
    "hmm", "hm", "mm", "mhm", "mmhm", "uh", "um", "uhh", "erm", "ah", "oh",
    "okay", "ok", "yeah", "yep", "yup", "cough", "hi", "hello",
}


def _is_filler(text: str) -> bool:
    words = re.findall(r"[a-z']+", text.lower())
    return len(words) <= 1 and (not words or words[0] in _FILLER)


def _rms(pcm_bytes: bytes) -> float:
    if not pcm_bytes:
        return 0.0
    samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
    return float(np.sqrt(np.mean(samples * samples)))


def _delivered_to_char(segments: list[tuple[int, int]], delivered: int, text_len: int) -> int:
    """Map delivered audio bytes to a character offset via the per-sentence
    (cumulative audio, cumulative chars) checkpoints, interpolating within the
    sentence that was playing when the interruption landed."""
    if not segments:
        return 0
    prev_audio, prev_char = 0, 0
    for cum_audio, cum_char in segments:
        if delivered <= cum_audio:
            span = cum_audio - prev_audio
            frac = (delivered - prev_audio) / span if span else 1.0
            char = round(prev_char + frac * (cum_char - prev_char))
            return max(0, min(text_len, char))
        prev_audio, prev_char = cum_audio, cum_char
    return min(text_len, prev_char)
