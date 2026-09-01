"""Competencies the panel evaluates, and who owns each.

Seeded here as a default for a generic software role. In Phase 7 these come from
the parsed JD + résumé (the dossier, ARCHITECTURE §9); until then this default
set drives floor control's coverage/gap and tie-breaking (§5).

`coverage` per competency is Σ(claim.strength) / target_evidence, capped at 1 —
but claims are the evidence ledger, which lands in Phase 4. Until then coverage
is 0 for every competency, so `gap` is 1 everywhere and floor control is driven
by bid interest + recency (which is exactly what the PS11 routing demo needs).
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Competency:
    key: str
    name: str
    weight: float              # relative importance (tie-break + future weighting)
    target_evidence: float     # Σ strength needed for full coverage
    owners: tuple[str, ...]    # agent ids responsible for probing it


DEFAULT_COMPETENCIES: list[Competency] = [
    Competency("tech_depth", "Technical depth", 1.0, 3, ("technical",)),
    Competency("system_design", "System design", 1.0, 3, ("technical", "coding")),
    Competency("coding", "Coding & problem solving", 1.0, 3, ("coding",)),
    Competency("product_sense", "Product sense", 0.8, 3, ("product",)),
    Competency("customer_impact", "Customer impact", 0.8, 3, ("customer",)),
    Competency("ownership_comm", "Ownership & communication", 0.7, 3, ("hiring_manager",)),
]
