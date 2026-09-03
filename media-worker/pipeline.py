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
import json
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

from shared import dossier as dossier_mod
from shared import gemini
from shared import llm_router, orchestrator, prompts
from shared.competencies import DEFAULT_COMPETENCIES, Competency
from shared.config import Settings
from shared.ledger import Ledger
from shared.models import AuditEvent, Dossier, TranscriptTurn

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
# Coding round (§8): how often Liam proactively looks at the screen and comments
# while the candidate codes, between their spoken turns.
CODING_WATCH_SECS = 22.0
# Safety net for the Gemini-driven round: end it if the browser never reports back.
CODING_MAX_SECS = 600.0
# The coding round is a REQUIRED phase, not a bid: trigger it after this many
# candidate answers if Liam is on the panel (so it happens even if the candidate
# never steers toward coding).
CODING_TRIGGER_TURNS = 5


class InterviewPipeline:
    def __init__(self, session: AgoraSession, settings: Settings, interview_id: str,
                 dossier: Dossier | None = None):
        self.session = session
        self.settings = settings
        self.interview_id = interview_id
        self.sample_rate = settings.audio_sample_rate
        self.allow_bargein = settings.allow_bargein

        # Dossier (§9): panel composition, competency weights, per-agent rubrics,
        # and résumé claims. Falls back to the full panel / defaults if absent.
        self.dossier = dossier
        self.panel = list(dossier.panel) if (dossier and dossier.panel) else list(prompts.PANEL_IDS)
        # `panel` is the LIVE panel (Liam is removed from it once the coding round
        # wraps); `_full_panel` is everyone who participated, used for scoring so
        # Liam is still scored on the coding evidence at the end (§8).
        self._full_panel = list(self.panel)
        self._competencies = _derive_competencies(dossier, self.panel)
        # Lean context (role + focus + rubric) for the token-hot paths (bids, scoring);
        # rich context (adds the candidate's résumé highlights) for question generation.
        self.contexts = {a: (dossier_mod.role_context(dossier, a) if dossier else "")
                         for a in self.panel}
        self.q_contexts = {a: (dossier_mod.question_context(dossier, a) if dossier else "")
                           for a in self.panel}

        self.floor = orchestrator.FloorController(
            panel_ids=self.panel,
            competencies=self._competencies,
            time_budget_s=settings.interview_time_budget_s,
            lambda_start=settings.coverage_lambda_start,
            lambda_end=settings.coverage_lambda_end,
        )
        # The evidence ledger: claims extracted inside bids, competency coverage.
        self.ledger = Ledger(interview_id, self._competencies)
        # Pre-register résumé claims as UNVERIFIED evidence (§9) — verified or
        # contradicted as the candidate speaks.
        if dossier:
            for rc in dossier.resume_claims:
                self.ledger.add(text=rc.get("text", ""), competency=rc.get("competency", ""),
                                source_turn=0, strength=0.4, status="UNVERIFIED",
                                noticed_by="resume")

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
        # Confidence trajectory (§11 Phase 8): a coverage snapshot per turn, so the
        # report can chart how the panel's evidence built up over the interview.
        self._trajectory: list[dict] = []

        # Coding round (§8): Liam issues ONE live task, the candidate shares their
        # screen, frames stream in here. A FAST vision model reads the latest frame
        # to ground Liam's live follow-ups; a thorough pass at conclude scores it.
        self._frame: str | None = None            # latest screen frame (data: URI)
        self._frame_lock = threading.Lock()
        self._coding_task: str | None = None       # the one task Liam set
        self._coding_in_panel = "coding" in self.panel
        self._coding_mode = False                  # round active → floor locked to Liam
        self._coding_done = False                  # round wrapped → Liam off the panel
        self._coding_gemini = False                # round driven by browser Gemini Live
        self._coding_lock = threading.Lock()       # one Liam coding turn at a time
        self._coding_thread: threading.Thread | None = None
        self._reply_n = 0                          # candidate answers handled (coding trigger)

        # UI/speaker signals (wired to a WebSocket by main; None = just log).
        self.on_event: Optional[Callable[[dict], None]] = None

        self._pending = bytearray()
        self._pending_since: float | None = None

        self._reply_lock = threading.Lock()
        self._replying = False
        self._opened = False
        self._current_reply: TranscriptTurn | None = None   # turn currently being spoken
        self._last_speaker: str | None = None   # last panel agent to hold the floor
        self._concluded = False   # set at the final bell; stops new turns

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

    def freeze(self):
        """Final bell: stop taking new turns AND cut off any in-flight speech so
        no interviewer keeps talking while/after the record is scored."""
        self._concluded = True
        try:
            self.session.interrupt()
        except Exception:
            pass

    def report_inputs(self):
        """Snapshot the full record for scoring: (transcript, claims, coverage, panel, contexts)."""
        with self._tx_lock:
            transcript = [t.model_copy() for t in self.transcript]
        # Score everyone who participated (incl. Liam even after he left the live
        # panel), not just whoever is still live.
        return (transcript, self.ledger.claims(), self.ledger.coverage(),
                list(self._full_panel), dict(self.contexts))

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
            if self._replying or self._concluded:
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
        self._speak_and_wait("orchestrator", prompts.orchestrator_intro(self.panel, self.dossier))
        if self._stop.is_set():
            return
        opener = prompts.opening_line(prompts.OPENING_AGENT_ID, self.dossier)
        self._speak_and_wait(prompts.OPENING_AGENT_ID, opener)
        self.floor.record(prompts.OPENING_AGENT_ID)

    def _panel_reply(self):
        snapshot = self._snapshot_transcript()

        # Coding round is a locked phase (§8): while it's on, Liam owns the floor —
        # no bidding, no switching to other interviewers — until he wraps it. When
        # Gemini Live drives it (browser), the server stays idle and just waits for
        # the result; otherwise the snapshot-vision engine runs the turn here.
        if self._coding_mode:
            if not self._coding_gemini:
                self._coding_turn(snapshot)
            return

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

        # The coding round is a REQUIRED phase (§8), not something Liam has to win a
        # bid for — trigger it deterministically part-way through so it happens even
        # if the candidate never steers toward code.
        self._reply_n += 1
        if ("coding" in self.panel and self._coding_task is None
                and not self._coding_done and self._reply_n >= CODING_TRIGGER_TURNS):
            self._enter_coding_round()
            return

        self._emit({"type": "thinking"})
        bids = orchestrator.collect_bids(self.panel, snapshot, self.contexts)

        # Steps 5 and 10 are the same call (§4): the bids also carry the claims
        # each interviewer noticed. Write them to the ledger, then refresh the
        # coverage map that feeds `gap` in floor control (§5).
        self._ingest_claims(bids, snapshot)
        self.floor.set_coverage(self.ledger.coverage())
        self._emit_ledger()
        self._record_trajectory()

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

        # First time Liam takes the floor, kick off the locked coding round (§8)
        # instead of a normal question.
        if decision.winner == "coding" and self._coding_task is None:
            self._enter_coding_round()
            return

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

    def _resume_after_coding(self):
        """Proactively continue the interview the moment the coding round ends —
        instead of waiting for the candidate to speak — so there's no dead air after
        Liam hands back (§8). Runs via _launch, so it's serialized with normal replies."""
        time.sleep(0.8)
        if self._stop.is_set() or self._concluded or self._coding_mode or not self.panel:
            return
        snapshot = self._snapshot_transcript()
        self._emit({"type": "thinking"})
        bids = orchestrator.collect_bids(self.panel, snapshot, self.contexts)
        self._ingest_claims(bids, snapshot)
        self.floor.set_coverage(self.ledger.coverage())
        self._emit_ledger()
        self._record_trajectory()
        decision = self.floor.decide(bids)
        extra = ("The live coding round just finished and Liam has handed back to the "
                 "panel. In ONE short sentence acknowledge moving on from the coding "
                 "exercise, then ask a fresh question in YOUR area. Do not wait for the "
                 "candidate to speak first.")
        text = self._generate_question(decision.winner, extra=extra)
        if text:
            self._speak_and_wait(decision.winner, text)
            self.floor.record(decision.winner)

    def _generate_question(self, agent_id: str, extra: str = "") -> str:
        transcript = self._snapshot_transcript()[-QUESTION_CONTEXT_TURNS:]
        messages = prompts.build_agent_prompt(agent_id, "LIVE", transcript, extra,
                                              self.q_contexts.get(agent_id, ""))
        try:
            return llm_router.chat(
                messages, model=self.settings.llm_fast_model,
                max_tokens=120, temperature=0.7, reasoning_effort="low",
            )
        except Exception:
            logger.exception("question generation failed for %s; staying silent", agent_id)
            return ""

    # ---- coding round (§8) -------------------------------------------------

    def set_frame(self, data_url: str) -> None:
        """Store the latest screen-share frame (a data: URI) from the browser."""
        with self._frame_lock:
            self._frame = data_url

    def _get_frame(self) -> str | None:
        with self._frame_lock:
            return self._frame

    def _make_coding_task(self) -> str:
        try:
            return llm_router.chat(
                prompts.build_coding_task_prompt(self.q_contexts.get("coding", "")),
                model=self.settings.llm_fast_model, max_tokens=320,
                temperature=0.6, reasoning_effort="low",
            ).strip()
        except Exception:
            logger.exception("coding task generation failed; using a generic one")
            return ("Let's do a quick coding exercise. Please share your screen and open "
                    "an editor, then write a function that returns the indices of the two "
                    "numbers in a list that add up to a target. Think out loud as you go.")

    def _enter_coding_round(self):
        """Liam sets the ONE task and locks the floor to himself (§8). Gemini Live runs
        the round in the browser when configured; otherwise the snapshot engine runs
        it here."""
        self._coding_mode = True
        task = self._make_coding_task()
        self._coding_task = task
        if gemini.enabled():
            # Browser-driven: Gemini greets + runs the round; the server stays idle
            # and waits for finish_coding_external. A watchdog prevents a stall.
            self._coding_gemini = True
            # Speak a bridge line over the normal panel voice first, so Liam isn't
            # silent while the candidate decides to share (Gemini can't start until
            # the screen is shared).
            self._speak_and_wait("coding",
                                 "Alright, let's do a short live coding exercise. Please "
                                 "share your entire screen, and I'll give you the problem "
                                 "and walk through it with you.")
            self._emit({"type": "coding_gemini", "task": task})
            logger.info("coding round: Gemini Live (browser) — floor locked, server idle")
            threading.Thread(target=self._coding_watchdog, daemon=True,
                             name="coding-watchdog").start()
            return
        # Fallback: snapshot-vision engine, Liam runs it from here.
        self._emit({"type": "coding_task", "text": task})
        logger.info("coding round: snapshot vision — floor locked to Liam")
        self._speak_and_wait("coding", task)
        self.floor.record("coding")
        self._coding_thread = threading.Thread(target=self._coding_loop, daemon=True,
                                               name="coding-watch")
        self._coding_thread.start()

    def _coding_watchdog(self):
        """Safety net for the Gemini round: if the browser session dies without
        reporting a result, end the round so the interview doesn't stall (§8)."""
        waited = 0.0
        while (waited < CODING_MAX_SECS and self._coding_mode
               and not self._stop.is_set() and not self._concluded):
            time.sleep(1.0)
            waited += 1.0
        if self._coding_mode and self._coding_gemini:
            logger.warning("coding watchdog fired (%.0fs); ending round", waited)
            self._end_coding_round("done", "(coding round ended by timeout)")

    def finish_coding_external(self, verdict: str, summary: str = ""):
        """Called from /coding/result when the browser Gemini session ends the round."""
        if not self._coding_mode:
            return
        verdict = (verdict or "done").lower()
        if verdict not in ("done", "cheating", "error"):
            verdict = "done"
        self._end_coding_round(verdict, (summary or "").strip() or "(no summary)")

    def _coding_loop(self):
        """While the round is on, comment on the screen every CODING_WATCH_SECS —
        so Liam stays engaged instead of going silent between the candidate's turns."""
        while self._coding_mode and not self._stop.is_set() and not self._concluded:
            waited = 0.0
            while (waited < CODING_WATCH_SECS and self._coding_mode
                   and not self._stop.is_set() and not self._concluded):
                time.sleep(0.5)
                waited += 0.5
            if not self._coding_mode:
                break
            # Don't cut in while anyone is mid-turn.
            if self.session.is_speaking() or self._pending:
                continue
            self._do_coding_turn("")   # proactive: no new candidate utterance

    def _screen_read(self) -> str:
        """Fast vision read of the current screen (with fallback chain). Best-effort —
        a failed read returns a marker, and Liam adapts verbally."""
        frame = self._get_frame()
        if not frame:
            return "(No screen shared yet.)"
        try:
            read = llm_router.see(
                "You are watching a candidate's screen during a live coding interview. In "
                f"1-2 sentences describe ONLY what is actually visible: the code / language / "
                "approach, and any obvious bug. IMPORTANT: explicitly note if any AI chat "
                "assistant (ChatGPT, Claude, Copilot, Gemini, Perplexity) is open or visible.",
                frame, model=self.settings.llm_vision_fast, max_tokens=180, timeout=8.0,
                retries=0,   # live: fail fast through the chain, don't stack backoffs
            ).strip()
            self._emit({"type": "screen_read", "text": read})
            return read or "(Screen shared but nothing readable.)"
        except Exception as e:
            logger.warning("live screen read unavailable (%s)", type(e).__name__)
            return "(Screen not readable right now.)"

    def _coding_turn(self, snapshot: list[TranscriptTurn]):
        """Candidate-triggered Liam turn inside the coding round."""
        cand = next((t.text for t in reversed(snapshot) if t.speaker == "candidate"), "")
        self._do_coding_turn(cand)

    def _do_coding_turn(self, candidate_text: str):
        """One Liam turn in the locked round — read the screen, react, decide. Serialized
        so the watch loop and candidate turns never talk over each other."""
        if not self._coding_lock.acquire(blocking=False):
            return          # another coding turn is in flight; skip this one
        try:
            if not self._coding_mode:
                return
            self._emit({"type": "thinking"})
            screen = self._screen_read()
            say, verdict = self._coding_decide(candidate_text, screen)
            # Snapshot fallback does NOT flag cheating — single-frame vision is too
            # unreliable and false-flags a candidate's own AI-library code. Cheating
            # detection is Gemini Live's job (continuous video). Here: continue / done.
            if verdict == "cheating":
                verdict = "continue"
            if not say:
                say = "Keep going — talk me through what you're writing."
            if not self._coding_mode:      # round may have ended while we thought
                return
            self._speak_and_wait("coding", say)
            self.floor.record("coding")
            if verdict == "done":
                self._end_coding_round("done", screen)
        finally:
            self._coding_lock.release()

    def _coding_decide(self, candidate_text: str, screen: str) -> tuple[str, str]:
        """Ask the fast model for Liam's spoken line + a verdict. Text model (Groq),
        so it's reliable even when vision is flaky."""
        try:
            raw = llm_router.chat(
                prompts.build_coding_turn_prompt(self._coding_task, screen, candidate_text,
                                                 self.q_contexts.get("coding", "")),
                model=self.settings.llm_fast_model, max_tokens=240,
                temperature=0.5, reasoning_effort="low",
            )
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            data = json.loads(m.group(0)) if m else {}
            say = str(data.get("say", "")).strip()
            verdict = str(data.get("verdict", "continue")).strip().lower()
            if verdict not in ("continue", "done", "cheating"):
                verdict = "continue"
            return say, verdict
        except Exception:
            logger.exception("coding decide failed; continuing")
            return "", "continue"

    def _end_coding_round(self, verdict: str, detail: str):
        """Wrap the coding round: record evidence and drop Liam from the panel so the
        rest of the interview continues without him (§8). `detail` is the Gemini
        summary or the final screen read. Idempotent."""
        if not self._coding_mode:
            return
        self._coding_mode = False
        self._coding_done = True
        self.panel = [a for a in self.panel if a != "coding"]
        self.floor.panel_ids = [a for a in self.floor.panel_ids if a != "coding"]
        detail = (detail or "").strip()
        if verdict == "cheating":
            self.ledger.add(
                text=f"Coding round: outside/AI help detected — integrity flag, coding "
                     f"not demonstrated. {detail[:180]}",
                competency="coding", source_turn=self._seq, strength=0.9,
                status="SOLID", noticed_by="coding")
        elif verdict == "error":
            self.ledger.add(
                text="Coding round could not run (technical issue) — not assessed.",
                competency="coding", source_turn=self._seq, strength=0.2,
                status="VAGUE", noticed_by="coding")
        else:  # done
            self.ledger.add(
                text=f"Coding round: {detail[:200]}",
                competency="coding", source_turn=self._seq, strength=0.7,
                status="SOLID", noticed_by="coding")
        self._emit({"type": "coding_done", "verdict": verdict})
        logger.info("coding round ended: verdict=%s; Liam removed from panel", verdict)
        # Proactively resume the panel so there's no dead air waiting for the
        # candidate to speak (§8). Via _launch so it's serialized with normal replies.
        self._launch(self._resume_after_coding)

    def assess_coding(self) -> None:
        """Conclude-time safety net (§8): if the round never formally wrapped (e.g. the
        interview ended mid-task), do one thorough vision read so coding isn't blank.
        If the round already ended, evidence is recorded — nothing to do."""
        if self._coding_done or not (self._coding_in_panel and self._coding_task):
            return
        frame = self._get_frame()
        if not frame:
            self.ledger.add(text="Coding task set but the candidate never shared a screen / wrote code.",
                            competency="coding", source_turn=self._seq, strength=0.3,
                            status="VAGUE", noticed_by="vision")
            return
        try:
            read = llm_router.see(
                "Assess a candidate's screen at the END of a live coding task. "
                f"The task was: {self._coding_task}\n"
                "Judge ONLY what is visible. In 2-3 sentences: did they produce a working "
                "solution, what approach, and note correctness, edge cases, quality. Also "
                "note if an AI assistant is visible (that would be cheating). Be fair.",
                frame, model=self.settings.llm_vision_model, max_tokens=300, timeout=60.0,
            ).strip()
            if read:
                cheat = _looks_like_cheating(read)
                self.ledger.add(
                    text=(f"Coding round (final screen): {read}" if not cheat else
                          f"Coding round: AI assistant visible — integrity flag. {read}"),
                    competency="coding", source_turn=self._seq,
                    strength=0.9 if cheat else 0.7, status="SOLID", noticed_by="vision")
                logger.info("coding vision assessment recorded (conclude)")
        except Exception:
            logger.exception("coding vision assessment failed; no coding evidence added")

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
        if self._concluded:      # the bell rang mid-generation — don't start speaking
            return
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

    def _record_trajectory(self):
        """Snapshot per-competency coverage at the current turn, plus the panel-wide
        mean and how much of the evidence is SOLID. Charted in the report (§11)."""
        cov = self.ledger.coverage()
        claims = self.ledger.claims()
        solid = sum(1 for c in claims if c.status == "SOLID")
        per = {c.key: round(cov.get(c.key, 0.0), 3) for c in self._competencies}
        mean = round(sum(per.values()) / len(per), 3) if per else 0.0
        self._trajectory.append({
            "turn": self._seq,
            "mean": mean,
            "coverage": per,
            "claims": len(claims),
            "solid": solid,
        })

    def trajectory(self) -> list[dict]:
        return list(self._trajectory)


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


_CHEAT_TERMS = (
    "chatgpt", "chat gpt", "openai", "claude", "copilot", "gemini", "bard",
    "perplexity", "phind", "you.com", "ai assistant", "ai chat",
)


def _looks_like_cheating(screen_text: str) -> bool:
    """Deterministic backstop: an AI assistant named in the screen read = cheating,
    regardless of what the model concluded."""
    t = (screen_text or "").lower()
    return any(term in t for term in _CHEAT_TERMS)


def _derive_competencies(dossier: Optional[Dossier], panel: list[str]) -> list[Competency]:
    """Per-interview competency set (§9): keep the defaults owned by someone on the
    panel, and override their weight from the dossier where the JD specified one."""
    comps = [c for c in DEFAULT_COMPETENCIES if any(o in panel for o in c.owners)]
    if dossier and dossier.competency_weights:
        comps = [
            Competency(c.key, c.name, dossier.competency_weights.get(c.key, c.weight),
                       c.target_evidence, c.owners)
            for c in comps
        ]
    return comps or list(DEFAULT_COMPETENCIES)
