"""Voice 'Ask the Panel' (ARCHITECTURE §7) — the joinable, spoken revival.

After the interview is locked, the recruiter joins a channel and simply talks to
the panel. It reuses the whole voice stack (Agora + VAD + Smart Turn + Sarvam
STT + Deepgram TTS) but points at the *locked record* instead of running an
interview: recruiter speaks -> STT -> ask_panel.answer() (Orchestrator, grounded,
with the override tool) -> the answer is spoken back in the Orchestrator's voice.

Simpler than the live interview: one voice answering at a time, no bidding or
floor control. Half-duplex — the mic is ignored while the panel is speaking.
Counterfactuals stay in the text UI (they need a turn + hypothetical); voice
handles open Q&A and spoken overrides.
"""
import logging
import queue
import threading
import time

from agora_session import AgoraSession
from smart_turn import SmartTurn
from vad import SileroVAD
import stt
import tts

from pipeline import MIN_SPEECH_RMS, _is_filler, _rms

# Ask-the-Panel (recruiter) keeps automatic turn-taking (Smart Turn + a short
# timeout) — the manual "Done" model is candidate-only. Own constants so the
# interview pipeline can change its turn logic without affecting this.
PENDING_MAX_SECS = 30.0
PENDING_TIMEOUT_SECS = 2.0   # commit the recruiter's question after this much silence
from shared import ask_panel, prompts, store
from shared.config import Settings
from shared.models import REC_DECLINE, REC_PROCEED, REC_PROCEED_FLAGGED

logger = logging.getLogger(__name__)

# Deterministic voice-override detection (§7). Overriding the panel's
# recommendation is too important to depend on an LLM tool-call (which can 400 or
# fall back to a plain answer that can't override) — so we catch clear spoken
# commands in code and apply them directly.
_OVERRIDE_TRIGGERS = ("override", "overrule", "overwrite")
_ACCEPT_PHRASES = (
    "accept the candidate", "accept this candidate", "accept him", "accept her",
    "select the candidate", "select this candidate", "select him", "select her",
    "hire him", "hire her", "hire the candidate", "hire this candidate",
    "proceed with the candidate", "proceed with this", "move forward with",
    "go ahead with", "approve the candidate", "approve this candidate",
)
_REJECT_PHRASES = (
    "reject the candidate", "reject this candidate", "reject him", "reject her",
    "decline the candidate", "decline this candidate", "do not hire", "don't hire",
    "pass on the candidate", "pass on this candidate", "drop the candidate",
)
_ACCEPT_WORDS = ("accept", "select", "hire", "proceed", "approve", "go ahead", "move forward")
_REJECT_WORDS = ("reject", "decline", "pass", "drop")

# A question mentioning "reject the candidate" ("why did you reject the candidate?")
# is NOT an override command. If any of these interrogative signals appear, we
# treat the utterance as a question and never override on it. This was the bug:
# "why did he reject the candidate" matched _REJECT_PHRASES and flipped the record.
_QUESTION_SIGNALS = (
    "?", "why did", "why do", "why is", "why was", "why are", "why does", "why would",
    "what are", "what is", "what was", "what were", "what do", "what did", "what about",
    "how did", "how do", "how does", "how was", "how come",
    "when did", "which ", "who ", "where ",
    "tell me", "explain", "walk me through", "talk about", "his weakness", "her weakness",
    "weaknesses", "his strength", "her strength", "score so low", "so low",
)


def _looks_like_question(t: str) -> bool:
    """A question / request for information — never an override command."""
    return any(sig in t for sig in _QUESTION_SIGNALS)


def _detect_override(text: str):
    """Return (decision, reason) ONLY if the recruiter clearly *commanded* an
    override (decline or proceed — no other states), else None.

    Guards: a bare 'override' with no direction returns None (the panel asks which
    way); a *question* that merely mentions accept/reject ('why did you reject the
    candidate?') returns None so the panel answers instead of flipping the record."""
    t = text.lower()
    if _looks_like_question(t):
        return None
    trig = any(w in t for w in _OVERRIDE_TRIGGERS)
    accept = any(p in t for p in _ACCEPT_PHRASES) or (trig and any(w in t for w in _ACCEPT_WORDS))
    reject = any(p in t for p in _REJECT_PHRASES) or (trig and any(w in t for w in _REJECT_WORDS))
    # Override is binary: DECLINE or PROCEED. (No PROCEED_FLAGGED from voice —
    # the recruiter is making a clean call.)
    if reject and not accept:
        return REC_DECLINE, text[:500]
    if accept and not reject:
        return REC_PROCEED, text[:500]
    return None

GREETING_DELAY_SECS = 1.2
# The panel must deliver at least this much before the recruiter can barge in
# (stops the answer's first words echoing in and cutting itself off).
BARGEIN_GRACE_SECS = 0.35
# Barge-in only on LOUD, sustained speech. On speakers, Agora AEC removes most of
# the panel's voice from the mic; the quiet residual that leaks through stays
# below this, while the recruiter's direct voice is well above it — so you can
# interrupt on speakers without the panel echo-interrupting itself. int16 scale.
BARGEIN_MIN_RMS = 900.0   # easier to cut the panel off (was 1300 — needed shouting)


class AskPipeline:
    def __init__(self, session: AgoraSession, settings: Settings,
                 record: ask_panel.PanelRecord, on_event=None):
        self.session = session
        self.settings = settings
        self.record = record
        self.on_event = on_event
        self.sample_rate = settings.audio_sample_rate

        # Barge-in enabled: the recruiter can talk over the panel to cut in.
        self.vad = SileroVAD(sample_rate=self.sample_rate, stop_secs=settings.vad_stop_secs,
                             threshold=0.7, bargein_ms=250.0, on_speech_start=self._on_bargein)
        self.smart_turn = SmartTurn(settings.smart_turn_model_path, settings.smart_turn_threshold)

        self._pending = bytearray()
        self._pending_since: float | None = None
        self._recent_rms = 0.0        # decaying peak loudness, for barge-in gating
        self._lock = threading.Lock()
        self._busy = False
        self._greeted = False
        self._last_target: str | None = None   # sticky addressee for follow-ups

        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="ask-pipeline")

    def start(self):
        self._thread.start()
        logger.info("AskPipeline running for interview %s", self.record.interview_id)

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=2)

    def on_recruiter_joined(self, user_id=None):
        with self._lock:
            if self._greeted:
                return
            self._greeted = True

        def _greet():
            time.sleep(GREETING_DELAY_SECS)
            rec = self.record.report.conclusion.recommendation.replace("_", " ").lower()
            self._speak("orchestrator",
                        f"Welcome back. The panel concluded: {rec}. Ask me anything about "
                        "the candidate, or tell me to override the recommendation.")

        threading.Thread(target=_greet, daemon=True, name="ask-greet").start()

    # ---- loop --------------------------------------------------------------

    def _run(self):
        while not self._stop.is_set():
            try:
                pcm = self.session.inbound.get(timeout=0.2)
            except queue.Empty:
                self._check_pending()
                continue
            # Track recent loudness (decaying peak) so barge-in can require a loud
            # voice, not quiet post-AEC echo residue.
            self._recent_rms = max(_rms(pcm), self._recent_rms * 0.85)
            # Feed the VAD even while the panel is speaking, so the recruiter can
            # barge in (_on_bargein stops the answer); their words then become the
            # next question.
            utterance = self.vad.process(pcm)
            if utterance is not None:
                self._on_utterance(utterance)
            else:
                self._check_pending()

    def _on_utterance(self, utterance: bytes):
        # Don't commit on Smart Turn's "complete" — it fires mid-sentence and cut the
        # recruiter's question off, garbling the transcript. Accumulate and commit on a
        # generous silence timeout instead, so the whole question is captured.
        combined = bytes(self._pending) + utterance
        capped = len(combined) >= int(PENDING_MAX_SECS * self.sample_rate * 2)
        if capped:
            self._pending = bytearray()
            self._pending_since = None
            self._handle(combined)
        else:
            self._pending = bytearray(combined)
            self._pending_since = time.monotonic()

    def _check_pending(self):
        if self._pending and self._pending_since is not None:
            if time.monotonic() - self._pending_since >= PENDING_TIMEOUT_SECS:
                pcm = bytes(self._pending)
                self._pending = bytearray()
                self._pending_since = None
                self._handle(pcm)

    def _handle(self, pcm: bytes):
        # If the panel is still speaking, this utterance finalized without a loud
        # barge-in stopping the answer → it's echo, not a question. Discard it.
        # (A real barge-in calls interrupt() first, so is_speaking() is False here.)
        if self.session.is_speaking():
            return
        if _rms(pcm) < MIN_SPEECH_RMS:
            return
        try:
            text = stt.transcribe(pcm, self.settings.sarvam_api_key, self.sample_rate)
        except Exception:
            logger.exception("STT failed")
            return
        if not text or _is_filler(text):
            return
        logger.info("recruiter asks: %r", text)
        self._emit({"type": "heard", "text": text})
        with self._lock:
            if self._busy:
                return
            self._busy = True
        threading.Thread(target=self._answer, args=(text,), daemon=True, name="ask-answer").start()

    def _on_bargein(self):
        """Recruiter started talking over the panel — stop the answer and listen.
        Requires LOUD sustained speech so speaker echo (quiet after AEC) can't
        trigger it."""
        if (self.session.is_speaking()
                and self.session.speaking_elapsed() >= BARGEIN_GRACE_SECS
                and self._recent_rms >= BARGEIN_MIN_RMS):
            self.session.interrupt()
            with self._lock:
                self._busy = False   # let their new question be answered
            self._emit({"type": "idle"})
            logger.info("recruiter barge-in — answer interrupted")

    # Aliases so "ask the DM" / "what does HR think" route to the Hiring Manager,
    # etc. Names and full titles are matched too.
    _ALIASES = {
        "hiring_manager": ("dm", "hr", "hiring manager", "hiring"),
        "technical": ("technical interviewer", "technical", "tech interviewer"),
        "product": ("product interviewer", "product manager"),
        "customer": ("customer advocate", "customer"),
        "coding": ("coding interviewer", "coding"),
    }

    @classmethod
    def _detect_target(cls, text: str) -> str | None:
        """Route to a specific interviewer if the recruiter named one; else open."""
        t = text.lower()
        for aid in prompts.PANEL_IDS:
            if prompts.AGENTS[aid].name.lower() in t:
                return aid
        for aid, aliases in cls._ALIASES.items():
            if aid in prompts.PANEL_IDS and any(a in t for a in aliases):
                return aid
        return None

    # Words that pull the conversation back to the whole panel / host (or are a
    # decision the host handles), overriding the sticky addressee.
    _OPEN_WORDS = (
        "overall", "recommendation", "the panel", "whole panel", "everyone", "host",
        "final decision", "conclusion", "override", "overrule", "accept", "reject",
        "decline", "proceed", "hire", "move forward",
    )

    def _resolve_target(self, text: str) -> str | None:
        """Sticky addressing: a named interviewer wins and is remembered; a plain
        follow-up stays with the last one; panel/host/decision words go open."""
        named = self._detect_target(text)
        if named:
            self._last_target = named
            return named
        t = text.lower()
        if any(k in t for k in self._OPEN_WORDS):
            self._last_target = None
            return None
        return self._last_target   # follow-up with no name → stay with the last interviewer

    def _answer(self, text: str):
        try:
            self._emit({"type": "thinking"})
            # Deterministic override (§7): a clear spoken command applies immediately,
            # never dropped to a flaky LLM tool-call.
            ov_intent = _detect_override(text)
            if ov_intent is not None:
                decision, _reason = ov_intent
                current = self.record.report.conclusion.recommendation
                proceed_now = current in (REC_PROCEED, REC_PROCEED_FLAGGED)
                # Override only flips the record to the OTHER state. If it's already
                # there ("decline the candidate" when already declined), it's a no-op
                # — don't announce a bogus override; answer the question instead.
                already = (decision == REC_DECLINE and not proceed_now) or \
                          (decision == REC_PROCEED and proceed_now)
                if already:
                    state = "declined" if not proceed_now else "cleared to proceed"
                    logger.info("ask: override to %s ignored — already %s", decision, current)
                    self._speak("orchestrator",
                                f"The panel has already {state} this candidate, so there's "
                                "nothing to flip. If you want to reverse that, tell me to "
                                f"{'proceed with' if not proceed_now else 'decline'} the candidate.")
                    return
                ov = ask_panel.override(self.record, decision, _reason)
                try:                       # persist so the dashboard reflects it (§11)
                    store.save_record(self.record)
                except Exception:
                    logger.exception("failed to persist voice override")
                self._emit({"type": "override", "override": ov.model_dump()})
                orig = ov.original_recommendation.replace("_", " ").lower()
                new = ov.decision.replace("_", " ").lower()
                self._speak("orchestrator",
                            f"Done. I've overridden the panel's recommendation from "
                            f"{orig} to {new}, and logged that you asked for it. The "
                            "panel's original call stays on record.")
                return
            target = self._resolve_target(text)
            logger.info("ask: routing to %s", target or "open/orchestrator")
            if target:
                ans = ask_panel.answer(self.record, text, mode="addressed", target=target)
            else:
                ans = ask_panel.answer(self.record, text)   # open → Orchestrator + override tool
            logger.info("ask: %s answers: %r", ans.answered_by, (ans.answer or "")[:160])
            if ans.override is not None:
                self._emit({"type": "override", "override": ans.override.model_dump()})
            self._speak(ans.answered_by, ans.answer)
        except Exception:
            logger.exception("ask answer failed")
        finally:
            with self._lock:
                self._busy = False
            self._emit({"type": "idle"})

    def _speak(self, agent_id: str, text: str):
        agent = prompts.AGENTS.get(agent_id) or prompts.ORCHESTRATOR
        self._emit({"type": "speaking", "agent": agent_id,
                    "name": agent.name, "title": agent.title, "text": text})
        self.session.begin_speech()
        try:
            tts.speak_stream(text, self.settings.deepgram_api_key, self.session.add_speech,
                             model=agent.voice_model, sample_rate=self.sample_rate,
                             is_active=self.session.is_speaking)
        finally:
            self.session.end_speech()
        while self.session.is_speaking() and not self._stop.is_set():
            time.sleep(0.05)

    def _emit(self, event: dict):
        if self.on_event is not None:
            try:
                self.on_event(event)
            except Exception:
                logger.exception("on_event failed")
