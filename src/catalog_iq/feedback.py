"""Thumbs up / down on answers, and the metrics they give.

Every answer the API returns is logged with an answer_id and what code knows about it:
agent, status, evidence level, outcome, retries, model, catalog build. A rating
references that id, so the client cannot claim a different agent or evidence level.
No question or answer text is stored: answers carry real names once the vault
restores them, and free text could hold anything. Down votes take a reason from a
fixed list for the same reason.

One rating per answer; rating again replaces it.

    catalog-iq feedback          # the metrics as JSON
    GET /feedback/summary        # the same
"""

import os
import sqlite3
import threading
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from catalog_iq.core.config import ROOT
from catalog_iq.evals.outcomes import run_outcome

DEFAULT_PATH = ROOT / "data" / "feedback" / "feedback.db"
Rating = Literal["up", "down"]
Reason = Literal["wrong", "incomplete", "should_have_answered", "not_helpful"]
ANSWERED = {"catalog_lookup": {"answered"}}  # other agents: every accepted status is an answer (as in the evals)

SCHEMA = """
CREATE TABLE IF NOT EXISTS answers (
    answer_id TEXT PRIMARY KEY, created_at TEXT NOT NULL, agent TEXT NOT NULL, status TEXT,
    evidence TEXT, outcome TEXT NOT NULL, attempts INTEGER, model TEXT, catalog_version TEXT);
CREATE TABLE IF NOT EXISTS ratings (
    answer_id TEXT PRIMARY KEY REFERENCES answers(answer_id), rating TEXT NOT NULL CHECK (rating IN ('up', 'down')),
    reason TEXT, rated_at TEXT NOT NULL);
"""


def feedback_path() -> Path:
    return Path(os.environ.get("FEEDBACK_DB") or DEFAULT_PATH)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _ratio(n: int, d: int) -> float | None:
    return round(n / d, 3) if d else None


class FeedbackStore:
    def __init__(self, path: Path | None = None):
        self.path = path or feedback_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.executescript(SCHEMA)

    def record_answer(self, answer_id: str, agent: str, response: dict, model: str | None,
                      catalog_version: str) -> None:
        g = response.get("guardrails") or {}
        checks = [f"{e['part']}:{e['check']}" for e in g.get("events", [])]
        outcome = run_outcome(g.get("outcome", "passed"), response.get("status", ""), checks, ANSWERED.get(agent))
        with self.lock, self.conn:
            self.conn.execute("INSERT INTO answers VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                              (answer_id, _now(), agent, response.get("status"), response.get("confidence"),
                               outcome, g.get("attempts"), model, catalog_version))

    def rate(self, answer_id: str, rating: Rating, reason: Reason | None = None) -> bool:
        """False when no such answer was logged."""
        with self.lock, self.conn:
            if not self.conn.execute("SELECT 1 FROM answers WHERE answer_id = ?", (answer_id,)).fetchone():
                return False
            self.conn.execute("INSERT OR REPLACE INTO ratings VALUES (?, ?, ?, ?)",
                              (answer_id, rating, reason if rating == "down" else None, _now()))
            return True

    def summary(self) -> dict:
        with self.lock:
            rows = self.conn.execute(
                "SELECT a.agent, a.evidence, a.outcome, r.rating, r.reason "
                "FROM answers a LEFT JOIN ratings r USING (answer_id)").fetchall()
        return summarize(rows)


def _group(rows) -> dict:
    rated = [r for r in rows if r[3]]
    up = sum(r[3] == "up" for r in rated)
    return {"answers": len(rows), "rated": len(rated), "feedback_rate": _ratio(len(rated), len(rows)),
            "thumbs_up_rate": _ratio(up, len(rated))}


def _by(rows, i: int) -> dict:
    return {k: _group([r for r in rows if r[i] == k]) for k in sorted({r[i] for r in rows if r[i]})}


def summarize(rows: list[tuple]) -> dict:
    """`rows` are (agent, evidence, outcome, rating or None, reason or None), one per answer.

    feedback_rate: share of answers rated; thumbs_up_rate: share of rated answers rated up.
    by_evidence shows whether "high" evidence is rated up more often, as the evals' calibration
    does for correctness; high_evidence_thumbs_down is the user-reported counterpart of high_but_wrong.
    """
    down = [r for r in rows if r[3] == "down"]
    return {**_group(rows),
            "by_agent": _by(rows, 0), "by_evidence": _by(rows, 1), "by_outcome": _by(rows, 2),
            "down_reasons": dict(Counter(r[4] or "none_given" for r in down)),
            "high_evidence_thumbs_down": sum(r[1] == "high" for r in down)}
