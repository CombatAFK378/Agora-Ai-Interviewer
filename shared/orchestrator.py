"""The Orchestrator: deterministic floor control + parallel bid collection.

ARCHITECTURE §2/§5: almost everything the Orchestrator does is plain code, not a
model. Turn-taking is deterministic so we can show the exact five numbers behind
every floor grant and reproduce the same winner days later (Ask the Panel). Only
the bid *content* comes from LLMs; combining bids into a decision is arithmetic.

    priority(agent) = interest × (1 + λ·gap) × recency_penalty

- interest       the agent's own bid, 0-1
- gap            1 − coverage of that agent's weakest owned competency
- λ              coverage weight, ramps start→end across the time budget
- recency_penalty 0.5 if it spoke last turn, 0.8 two turns ago, else 1.0

Coverage comes from the evidence ledger (Phase 4); until then it is 0, so gap is
1 and routing is interest+recency driven — enough for the PS11 routing demo.

This module is import-safe and stateless-per-call except FloorController, which
holds one interview's live turn state. It has no web framework or Agora
dependency, so it lifts cleanly into the standalone Orchestrator service later
(§13.1) without a rewrite.
"""
import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from shared import llm_router, prompts
from shared.competencies import Competency
from shared.models import TranscriptTurn

logger = logging.getLogger(__name__)

LOW_INTEREST = 0.25   # below this for everyone → panel has little left to ask (§5)
TIE_BAND = 0.05       # priorities within this are treated as a tie (§5)
# Bids fire 5× per turn and are the token hot path. A bid only needs the recent
# thread to decide "do I want to ask next?" — the full transcript is still stored
# and used by scoring / dashboard (which run once, on the big-context model).
# "What's been covered" will come from the ledger + coverage map (Phase 4), not
# from re-reading the whole history on every bid.
BID_CONTEXT_TURNS = 8


@dataclass
class Bid:
    agent_id: str
    interest: float
    reason: str


@dataclass
class FloorDecision:
    winner: str
    priorities: dict[str, float]
    lam: float
    bids: dict[str, Bid]
    all_low: bool


class FloorController:
    def __init__(
        self,
        panel_ids: list[str],
        competencies: list[Competency],
        time_budget_s: float,
        lambda_start: float,
        lambda_end: float,
    ):
        self.panel_ids = list(panel_ids)
        self.comps = competencies
        self.time_budget = time_budget_s
        self.l0 = lambda_start
        self.l1 = lambda_end
        self._start = time.monotonic()
        self._turn = 0                                   # floor grants so far
        self._last_spoke: dict[str, int] = {}            # agent -> turn it last spoke
        self._turns_taken = {a: 0 for a in panel_ids}
        self._coverage: dict[str, float] = {}            # competency key -> 0..1 (Phase 4)
        self._agent_weight = {
            a: max((c.weight for c in competencies if a in c.owners), default=0.0)
            for a in panel_ids
        }

    # --- formula pieces --------------------------------------------------
    def lambda_now(self) -> float:
        if self.time_budget <= 0:
            return self.l0
        frac = min(1.0, (time.monotonic() - self._start) / self.time_budget)
        return self.l0 + (self.l1 - self.l0) * frac

    def coverage(self, competency_key: str) -> float:
        return self._coverage.get(competency_key, 0.0)

    def gap(self, agent_id: str) -> float:
        owned = [c for c in self.comps if agent_id in c.owners]
        if not owned:
            return 1.0
        weakest = min(self.coverage(c.key) for c in owned)
        return 1.0 - weakest

    def recency_penalty(self, agent_id: str) -> float:
        last = self._last_spoke.get(agent_id)
        if last is None:
            return 1.0
        d = (self._turn + 1) - last     # distance to the turn we're deciding now
        if d <= 1:
            return 0.5
        if d == 2:
            return 0.8
        return 1.0

    # --- decision --------------------------------------------------------
    def decide(self, bids: dict[str, Bid]) -> FloorDecision:
        lam = self.lambda_now()
        pr = {
            a: (bids[a].interest if a in bids else 0.0)
               * (1.0 + lam * self.gap(a))
               * self.recency_penalty(a)
            for a in self.panel_ids
        }
        top = max(pr.values()) if pr else 0.0
        contenders = [a for a in self.panel_ids if top - pr[a] <= TIE_BAND]
        if len(contenders) > 1:
            # Fixed tie-break (§5): competency weight, then fewest turns, then
            # stable panel order. Never random.
            contenders.sort(key=lambda a: (-self._agent_weight[a],
                                           self._turns_taken[a],
                                           self.panel_ids.index(a)))
        winner = contenders[0]
        all_low = all((bids[a].interest if a in bids else 0.0) < LOW_INTEREST
                      for a in self.panel_ids)
        return FloorDecision(winner, pr, lam, bids, all_low)

    def record(self, agent_id: str) -> None:
        self._turn += 1
        self._last_spoke[agent_id] = self._turn
        if agent_id in self._turns_taken:
            self._turns_taken[agent_id] += 1


# --- parallel bid collection --------------------------------------------

def collect_bids(panel_ids: list[str], transcript: list[TranscriptTurn]) -> dict[str, Bid]:
    """Fan out one bid call per interviewer, in parallel (ARCHITECTURE §4).

    Only the most recent turns are sent to each bid (token hot path); the caller
    keeps the full transcript for scoring/dashboard.
    """
    recent = transcript[-BID_CONTEXT_TURNS:]
    with ThreadPoolExecutor(max_workers=len(panel_ids)) as ex:
        results = ex.map(lambda a: _one_bid(a, recent), panel_ids)
    return {b.agent_id: b for b in results}


def _one_bid(agent_id: str, transcript: list[TranscriptTurn]) -> Bid:
    messages = prompts.build_bid_prompt(agent_id, transcript)
    for attempt in range(2):   # malformed JSON → retry once, then default (§10)
        try:
            # gpt-oss spends completion tokens on reasoning before the JSON, so
            # give it enough headroom that the (tiny) JSON isn't truncated — but
            # keep it modest to stay under the tokens-per-minute cap.
            raw = llm_router.chat(
                messages, max_tokens=160, temperature=0.3, reasoning_effort="low"
            )
            interest, reason = _parse_bid(raw)
            return Bid(agent_id, interest, reason)
        except Exception as e:
            logger.warning("bid %s attempt %d failed: %s", agent_id, attempt + 1, e)
    logger.warning("bid %s defaulting to interest=0.3", agent_id)
    return Bid(agent_id, 0.3, "default (bid unparseable)")


_JSON_OBJ = re.compile(r"\{.*\}", re.DOTALL)


def _parse_bid(raw: str) -> tuple[float, str]:
    m = _JSON_OBJ.search(raw)
    if not m:
        raise ValueError(f"no JSON object in bid: {raw!r}")
    data = json.loads(m.group(0))
    interest = float(data["interest"])
    interest = max(0.0, min(1.0, interest))
    reason = str(data.get("reason", "")).strip()[:120]
    return interest, reason
