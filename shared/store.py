"""Phase 8: durable persistence for locked interviews (ARCHITECTURE §11).

One JSON file per interview under settings.data_dir. This is enough to power the
recruiter dashboard (list past candidates, open any report) and to revive
Ask-the-Panel on a completed interview even after a restart. The live path is the
in-memory record; this is the durable copy. A real deployment swaps this for a
database — the surface is deliberately tiny (save / list / load).
"""
import json
import logging
import os
import time

from shared.ask_panel import PanelRecord
from shared.config import get_settings
from shared.models import Claim, InterviewReport, Override, TranscriptTurn, WhatIfQuery

logger = logging.getLogger(__name__)


def _dir() -> str:
    d = get_settings().data_dir
    os.makedirs(d, exist_ok=True)
    return d


def _path(interview_id: str) -> str:
    return os.path.join(_dir(), f"{interview_id}.json")


def _read(interview_id: str) -> dict | None:
    p = _path(interview_id)
    if not os.path.exists(p):
        return None
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        logger.exception("failed to read interview %s", interview_id)
        return None


def save_record(record: PanelRecord, *, candidate_name: str = "", role: str = "") -> dict:
    """Write (or update) the interview's JSON. `created_at` and the candidate
    name/role are preserved from an existing file so re-saves (after an override
    or counterfactual) don't lose them."""
    prev = _read(record.interview_id) or {}
    created_at = prev.get("created_at", time.time())
    candidate_name = candidate_name or prev.get("candidate_name", "")
    role = role or prev.get("role", "")
    rep = record.report
    data = {
        "interview_id": record.interview_id,
        "created_at": created_at,
        "candidate_name": candidate_name,
        "role": role,
        "recommendation": rep.conclusion.recommendation,
        "headline": rep.conclusion.headline,
        "override": record.overrides[-1].decision if record.overrides else None,
        "report": rep.model_dump(),
        "transcript": [t.model_dump() for t in record.transcript],
        "claims": [c.model_dump() for c in record.claims],
        "contexts": record.contexts,
        "audit": record.audit,
        "what_ifs": [w.model_dump() for w in record.what_ifs],
        "overrides": [o.model_dump() for o in record.overrides],
    }
    tmp = _path(record.interview_id) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    os.replace(tmp, _path(record.interview_id))   # atomic
    logger.info("saved interview %s (%s)", record.interview_id, candidate_name or "unnamed")
    return data


_SUMMARY_KEYS = ("interview_id", "created_at", "candidate_name", "role",
                 "recommendation", "headline", "override")


def list_summaries() -> list[dict]:
    """Lightweight rows for the dashboard list, newest first."""
    d = _dir()
    out = []
    for fn in os.listdir(d):
        if not fn.endswith(".json"):
            continue
        data = _read(fn[:-5])
        if data:
            out.append({k: data.get(k) for k in _SUMMARY_KEYS})
    out.sort(key=lambda s: s.get("created_at") or 0, reverse=True)
    return out


def load_report(interview_id: str) -> InterviewReport | None:
    data = _read(interview_id)
    return InterviewReport.model_validate(data["report"]) if data else None


def load_record(interview_id: str) -> PanelRecord | None:
    """Rebuild the full PanelRecord so Ask-the-Panel can revive on it."""
    data = _read(interview_id)
    if not data:
        return None
    return PanelRecord(
        interview_id=data["interview_id"],
        report=InterviewReport.model_validate(data["report"]),
        transcript=[TranscriptTurn.model_validate(t) for t in data.get("transcript", [])],
        claims=[Claim.model_validate(c) for c in data.get("claims", [])],
        audit=data.get("audit", []),
        what_ifs=[WhatIfQuery.model_validate(w) for w in data.get("what_ifs", [])],
        overrides=[Override.model_validate(o) for o in data.get("overrides", [])],
        contexts=data.get("contexts", {}),
    )
