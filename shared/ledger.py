"""The evidence ledger — the shared record of what the candidate has actually
demonstrated (ARCHITECTURE §3, §5).

Claims are extracted inside the bid calls (all five agents, each noticing things
in their own area) and land here. The ledger dedups near-duplicate claims so five
agents noticing the same thing doesn't inflate coverage, links contradictions,
and computes the competency coverage map that feeds `gap` in floor control:

    coverage(competency) = min(1, Σ claim.strength / target_evidence)

In-memory per interview for now; it moves to Postgres with the storage layer.
"""
import re
import uuid

from shared.competencies import Competency
from shared.models import Claim


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _overlap(a: str, b: str) -> float:
    """Jaccard token overlap — a cheap 'are these the same claim?' signal."""
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


DEDUP_THRESHOLD = 0.6        # same competency + this much overlap → same claim
CONTRADICT_THRESHOLD = 0.5   # overlap needed to link a flagged contradiction


class Ledger:
    def __init__(self, interview_id: str, competencies: list[Competency]):
        self.interview_id = interview_id
        self._targets = {c.key: c.target_evidence for c in competencies}
        self._comp_keys = [c.key for c in competencies]
        self._claims: list[Claim] = []

    def add(
        self,
        text: str,
        competency: str,
        source_turn: int,
        strength: float,
        status: str,
        noticed_by: str,
        contradicts_text: str | None = None,
    ) -> Claim:
        """Add a claim, merging it into a near-duplicate if one exists."""
        competency = competency if competency in self._targets else self._guess_competency(text, competency)
        strength = max(0.0, min(1.0, strength))

        # Merge into an existing claim in the same competency if very similar.
        for c in self._claims:
            if c.competency == competency and _overlap(c.text, text) >= DEDUP_THRESHOLD:
                if noticed_by not in c.noticed_by:
                    c.noticed_by.append(noticed_by)
                c.strength = max(c.strength, strength)
                if status == "SOLID":     # a solid sighting upgrades a vague one
                    c.status = "SOLID"
                return c

        claim = Claim(
            id=uuid.uuid4().hex[:8],
            interview_id=self.interview_id,
            text=text,
            competency=competency,
            source_turn=source_turn,
            strength=strength,
            status=status,
            noticed_by=[noticed_by],
        )
        if contradicts_text:
            match = self._find_contradiction(contradicts_text, competency)
            if match is not None:
                claim.contradicts_claim_id = match.id
        self._claims.append(claim)
        return claim

    def _guess_competency(self, text: str, fallback: str) -> str:
        """Map a loose competency label ('ownership') to a known key
        ('ownership_comm') by substring/prefix; fall back to the first key."""
        if fallback in self._targets:
            return fallback
        f = (fallback or "").lower()
        if f:
            for key in self._comp_keys:
                if f in key or key in f or key.split("_")[0] in f:
                    return key
        return self._comp_keys[0] if self._comp_keys else fallback

    def _find_contradiction(self, contradicts_text: str, competency: str) -> Claim | None:
        best, best_score = None, CONTRADICT_THRESHOLD
        for c in self._claims:
            score = _overlap(c.text, contradicts_text)
            if score >= best_score:
                best, best_score = c, score
        return best

    def coverage(self) -> dict[str, float]:
        """{competency_key: min(1, Σ strength / target_evidence)}."""
        cov = {k: 0.0 for k in self._comp_keys}
        for c in self._claims:
            if c.competency in cov:
                cov[c.competency] += c.strength
        return {k: min(1.0, v / self._targets[k]) if self._targets[k] else 0.0
                for k, v in cov.items()}

    def claims(self) -> list[Claim]:
        return list(self._claims)

    def contradictions(self) -> list[Claim]:
        return [c for c in self._claims if c.contradicts_claim_id]
