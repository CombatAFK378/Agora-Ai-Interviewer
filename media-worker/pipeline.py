"""Phase 3: the five-agent panel.

The live turn loop (ARCHITECTURE §4) for the full panel:

    candidate speech
      -> Silero VAD (pause) -> Smart Turn v3.1 (complete?)
      -> Sarvam STT -> transcript_turn (candidate)
      -> fan out 5 bid calls IN PARALLEL (one per interviewer)
      -> Orchestrator floor control picks a winner (deterministic)
      -> winning agent generates its question
      -> Deepgram streaming TTS in that agent's voice -> speak
      -> transcript_turn (agent), floor grant audited, speaker signal emitted

Cold start (§5): the Orchestrator opens with the AI disclosure + panel intro,
the Hiring Manager asks a broad opener, then turn two onward is normal bidding.

Barge-in stays as in Phase 2: the Interrupt button (and, if ALLOW_BARGEIN, the
mic) stops whoever is speaking and truncates their turn at the character heard.

Transcript, audit and floor state live in memory for now; the storage layer and
the extraction of the Orchestrator into its own service come later.
"""
import logging
import queue
import re
import threading
import time
from typing import Callable, Optional

import numpy as np

from agora_session import AgoraSession
from smart_turn import SmartTurn
from vad import SileroVAD
import stt
import tts

from shared import llm_router, orchestrator, prompts
from shared.competencies import DEFAULT_COMPETENCIES
from shared.config import Settings
from shared.ledger import Ledger
from shared.models import AuditEvent, TranscriptTurn

logger = logging.getLogger(__name__)

PENDING_MAX_SECS = 30.0
# After Smart Turn says "not done yet", how long the candidate may stay silent
# (thinking, gathering words) before we finalize the turn anyway. Short values
# cut people off mid-thought; this is generous on purpose.
PENDING_TIMEOUT_SECS = 4.0
OPENER_DELAY_SECS = 1.5
MIN_SPEECH_RMS = 350.0
BARGEIN_GRACE_SECS = 1.0
# Question generation reads the recent thread (the full transcript is still kept
# for scoring/dashboard). Larger than the bid window so questions stay coherent.
QUESTION_CONTEXT_TURNS = 16


class InterviewPipeline:
    def __init__(self, session: AgoraSession, settings: Settings, interview_id: str):
        self.session = session
        self.settings = settings
        self.interview_id = interview_id
        self.sample_rate = settings.audio_sample_rate
        self.allow_bargein = settings.allow_bargein

        self.panel = prompts.PANEL_IDS
        self.floor = orchestrator.FloorController(
            panel_ids=self.panel,
            competencies=DEFAULT_COMPETENCIES,
            time_budget_s=settings.interview_time_budget_s,
            lambda_start=settings.coverage_lambda_start,
            lambda_end=settings.coverage_lambda_end,
        )
        # The evidence ledger: claims extracted inside bids, competency coverage.
        self.ledger = Ledger(interview_id, DEFAULT_COMPETENCIES)
        self._competencies = DEFAULT_COMPETENCIES

        self.vad = SileroVAD(
            sample_rate=self.sample_rate,
            stop_secs=settings.vad_stop_secs,
            threshold=0.7,
            bargein_ms=750.0,
            on_speech_start=self._on_speech_start if self.allow_bargein else None,
        )
        self.smart_turn = SmartTurn(settings.smart_turn_model_path, settings.smart_turn_threshold)

        # Shared, in-memory for Phase 3.
        self.transcript: list[TranscriptTurn] = []
        self.audit: list[AuditEvent] = []
        self._tx_lock = threading.Lock()
        self._seq = 0

        # UI/speaker signals (wired to a WebSocket by main; None = just log).
        self.on_event: Optional[Callable[[dict], None]] = None

        self._pending = bytearray()
        self._pending_since: float | None = None

        self._reply_lock = threading.Lock()
        self._replying = False
        self._opened = False
        self._current_reply: TranscriptTurn | None = None   # turn currently being spoken
        self._last_speaker: str | None = None   # last panel agent to hold the floor

        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="pipeline")

    # ---- lifecycle ---------------------------------------------------------

    def start(self):
        self._stop.clear()
        self._thread.start()
        logger.info(
            "InterviewPipeline running (panel=%s, smart_turn=%s, bargein=%s)",
            ",".join(self.panel), "on" if self.smart_turn.enabled else "off",
            self.allow_bargein,
        )

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=2)

    def request_interrupt(self) -> bool:
        """Manual barge-in from the candidate's Interrupt button."""
        if not self.session.is_speaking():
            return False
        logger.info("manual interrupt requested")
        self._handle_interruption()
        return True

    def on_candidate_joined(self, user_id=None):
        """Cold start (§5): host disclosure + panel intro, then the HM opener."""
        with self._reply_lock:
            if self._opened:
                return
            self._opened = True
        self._launch(self._cold_open)

    # ---- main loop ---------------------------------------------------------

    def _run(self):
        while not self._stop.is_set():
            try:
                pcm = self.session.inbound.get(timeout=0.2)
            except queue.Empty:
                self._check_pending_timeout()
                continue
            # Half-duplex: ignore the mic entirely while an agent is speaking so
            # its own voice can't become a phantom turn / false barge-in.
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
        self._emit({"type": "heard", "text": text})
        self._launch(self._panel_reply)

    # ---- panel reply (off the main loop so barge-in stays responsive) ------

    def _launch(self, target: Callable[[], None]):
        with self._reply_lock:
            if self._replying:
                return
            self._replying = True
        threading.Thread(target=self._worker, args=(target,), daemon=True, name="reply").start()

    def _worker(self, target: Callable[[], None]):
        try:
            target()
        except Exception:
            logger.exception("reply worker failed")
        finally:
            with self._reply_lock:
                self._current_reply = None
                self._replying = False
            self._emit({"type": "idle"})

    def _cold_open(self):
        time.sleep(OPENER_DELAY_SECS)
        logger.info("cold start: host intro + opener")
        self._speak_and_wait("orchestrator", prompts.orchestrator_intro())
        if self._stop.is_set():
            return
        opener = prompts.opening_line(prompts.OPENING_AGENT_ID)
        self._speak_and_wait(prompts.OPENING_AGENT_ID, opener)
        self.floor.record(prompts.OPENING_AGENT_ID)

    def _panel_reply(self):
        snapshot = self._snapshot_transcript()

        # A clarification ("can you repeat?", "I didn't get it") is not a new
        # answer — the SAME interviewer should rephrase, not open the floor.
        last_cand = next((t.text for t in reversed(snapshot) if t.speaker == "candidate"), "")
        if _is_clarification(last_cand) and self._last_speaker in self.panel:
            who = self._last_speaker
            logger.info("clarification -> %s rephrases (no re-bid)", who)
            self._audit("clarification_reroute", {"agent": who, "candidate_said": last_cand})
            text = self._generate_question(
                who,
                extra=("The candidate didn't understand your previous question. Rephrase it "
                       "more simply and concretely in ONE short sentence — don't move on."),
            )
            if text:
                self._speak_and_wait(who, text)
                self.floor.record(who)
            return

        self._emit({"type": "thinking"})
        bids = orchestrator.collect_bids(self.panel, snapshot)

        # Steps 5 and 10 are the same call (§4): the bids also carry the claims
        # each interviewer noticed. Write them to the ledger, then refresh the
        # coverage map that feeds `gap` in floor control (§5).
        self._ingest_claims(bids, snapshot)
        self.floor.set_coverage(self.ledger.coverage())
        self._emit_ledger()

        decision = self.floor.decide(bids)

        summary = "  ".join(
            f"{a}:{decision.priorities[a]:.2f}(i={bids[a].interest:.2f})" for a in self.panel
        )
        logger.info("floor grant -> %s (λ=%.2f)%s | %s",
                    decision.winner, decision.lam,
                    " [all-low]" if decision.all_low else "", summary)
        self._audit("floor_grant", {
            "winner": decision.winner,
            "lambda": round(decision.lam, 3),
            "all_low": decision.all_low,
            "priorities": {a: round(p, 4) for a, p in decision.priorities.items()},
            "bids": {a: {"interest": bids[a].interest, "reason": bids[a].reason} for a in self.panel},
            "coverage": {k: round(v, 3) for k, v in self.ledger.coverage().items()},
        })

        pivot = (
            "The panel has little left to ask on the current thread. Change "
            "direction: open a NEW topic in your area — a different project or "
            "experience the candidate hasn't covered yet, or a short scenario — "
            "rather than drilling the same thing. Acknowledge briefly, then pivot."
        ) if decision.all_low else ""
        text = self._generate_question(decision.winner, extra=pivot)
        if not text:
            return
        self._speak_and_wait(decision.winner, text)
        self.floor.record(decision.winner)

    def _generate_question(self, agent_id: str, extra: str = "") -> str:
        transcript = self._snapshot_transcript()[-QUESTION_CONTEXT_TURNS:]
        messages = prompts.build_agent_prompt(agent_id, "LIVE", transcript, extra)
        try:
            return llm_router.chat(
                messages, model=self.settings.llm_fast_model,
                max_tokens=120, temperature=0.7, reasoning_effort="low",
            )
        except Exception:
            logger.exception("question generation failed for %s; staying silent", agent_id)
            return ""

    def _speak_and_wait(self, agent_id: str, text: str):
        self._speak(agent_id, text)
        while self.session.is_speaking() and not self._stop.is_set():
            time.sleep(0.05)

    def _speak(self, agent_id: str, text: str):
        """Stream the agent's whole turn in its own voice, in ONE TTS call.

        One streaming call (not per-sentence) means continuous audio with no gap
        between the reaction and the question — the reconnect gap was the audible
        break. Truncation on interrupt maps delivered vs synthesized bytes to a
        character offset over the full text.
        """
        agent = prompts.AGENTS[agent_id]
        if agent_id in self.panel:
            self._last_speaker = agent_id
        turn = self._add_turn(agent_id, text)
        with self._reply_lock:
            self._current_reply = turn
        self._emit({"type": "speaking", "agent": agent_id,
                    "name": agent.name, "title": agent.title, "text": text})

        self.session.begin_speech()
        n = tts.speak_stream(
            text, self.settings.deepgram_api_key, self.session.add_speech,
            model=agent.voice_model, sample_rate=self.sample_rate,
            is_active=self.session.is_speaking,
        )
        self.session.end_speech()
        if n > 0:
            logger.info("%s (%s) speaks: %r", agent.name, agent.title, text)
        else:
            logger.warning("no audio produced for %s", agent_id)

    # ---- interruption ------------------------------------------------------

    def _on_speech_start(self):
        if not self.session.is_speaking():
            return
        if self.session.speaking_elapsed() < BARGEIN_GRACE_SECS:
            return
        self._handle_interruption()

    def _handle_interruption(self):
        delivered, total = self.session.interrupt()
        with self._reply_lock:
            turn = self._current_reply
            self._current_reply = None
        if turn is None:
            return
        frac = (delivered / total) if total else 0.0
        char = max(0, min(len(turn.text), round(len(turn.text) * frac)))
        if char <= 0:
            with self._tx_lock:
                if self.transcript and self.transcript[-1] is turn:
                    self.transcript.pop()
            logger.info("barge-in before any audio; dropped un-heard agent turn")
            return
        with self._tx_lock:
            turn.truncated = True
            turn.truncation_char = char
        logger.info("INTERRUPTED: %s delivered %d/%d chars; heard=%r",
                    turn.speaker, char, len(turn.text), turn.text[:char])
        self._audit("interruption", {
            "agent": turn.speaker, "turn_seq": turn.seq,
            "truncation_char": char, "delivered_bytes": delivered,
        })

    # ---- transcript / audit / events --------------------------------------

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

    def _emit(self, event: dict):
        cb = self.on_event
        if cb is not None:
            try:
                cb(event)
            except Exception:
                logger.exception("on_event handler failed")

    def _ingest_claims(self, bids: dict, snapshot: list[TranscriptTurn]):
        """Write the claims each interviewer noticed into the evidence ledger."""
        source_turn = next((t.seq for t in reversed(snapshot) if t.speaker == "candidate"), 0)
        added = 0
        for aid in self.panel:
            for c in bids[aid].claims:
                self.ledger.add(
                    text=c["text"], competency=c["competency"], source_turn=source_turn,
                    strength=c["strength"], status=c["status"], noticed_by=aid,
                    contradicts_text=bids[aid].contradicts,
                )
                added += 1
        if added:
            logger.info("ledger: +%d claim(s) from turn %d (%d total)",
                        added, source_turn, len(self.ledger.claims()))

    def _emit_ledger(self):
        cov = self.ledger.coverage()
        self._emit({
            "type": "ledger",
            "coverage": [{"key": c.key, "name": c.name, "value": round(cov.get(c.key, 0.0), 3)}
                         for c in self._competencies],
            "claims": [{"text": cl.text, "competency": cl.competency,
                        "strength": round(cl.strength, 2), "status": cl.status,
                        "turn": cl.source_turn, "contradicts": cl.contradicts_claim_id}
                       for cl in self.ledger.claims()[-25:]],
            "contradictions": len(self.ledger.contradictions()),
        })


# ---- helpers ---------------------------------------------------------------

_FILLER = {
    "hmm", "hm", "mm", "mhm", "mmhm", "uh", "um", "uhh", "erm", "ah", "oh",
    "okay", "ok", "yeah", "yep", "yup", "cough", "hi", "hello",
}


def _is_filler(text: str) -> bool:
    words = re.findall(r"[a-z']+", text.lower())
    return len(words) <= 1 and (not words or words[0] in _FILLER)


# The candidate is asking the interviewer to repeat/clarify — route back to the
# same interviewer to rephrase, don't re-open the floor.
_CLARIFY = (
    "repeat", "didn't get", "did not get", "didn't understand", "did not understand",
    "don't understand", "do not understand", "not able to understand", "understand the question",
    "come again", "pardon", "say that again", "what do you mean", "what are you saying",
    "couldn't hear", "could not hear", "what was that", "can you rephrase", "rephrase",
)


def _is_clarification(text: str) -> bool:
    t = text.lower()
    return any(p in t for p in _CLARIFY)


def _rms(pcm_bytes: bytes) -> float:
    if not pcm_bytes:
        return 0.0
    samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
    return float(np.sqrt(np.mean(samples * samples)))
