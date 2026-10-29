"""Thumbs up / down: the store, its metrics, and the API endpoints (no network)."""

import sqlite3

import pytest
from fastapi.testclient import TestClient

from catalog_iq.api import create_app
from catalog_iq.catalog.db import catalog_path
from catalog_iq.feedback import FeedbackStore, summarize
from test_api import lookup_agent, router_says

ANSWERED = {"status": "answered", "confidence": "high",
            "guardrails": {"attempts": 1, "events": [], "outcome": "passed"}}
REFUSED = {"status": "insufficient_context", "confidence": "low",
           "guardrails": {"attempts": 2, "outcome": "refused",
                          "events": [{"part": "output", "check": "unsupported_claim", "detail": "x"}]}}


def test_the_outcome_is_computed_from_the_logged_answer(feedback_db):
    store = FeedbackStore(feedback_db)
    store.record_answer("a", "catalog_lookup", ANSWERED, "fake/fake-1", "v1")
    store.record_answer("b", "catalog_lookup", REFUSED, "fake/fake-1", "v1")
    store.record_answer("c", "pi_classifier", {**ANSWERED, "status": "not_found"}, None, "v1")
    outcomes = dict(store.conn.execute("SELECT answer_id, outcome FROM answers"))
    assert outcomes == {"a": "answered", "b": "verification_refusal", "c": "answered"}


def test_rating_an_unknown_answer_fails_and_rating_again_replaces(feedback_db):
    store = FeedbackStore(feedback_db)
    assert not store.rate("nope", "up")
    store.record_answer("a", "catalog_lookup", ANSWERED, None, "v1")
    assert store.rate("a", "down", "wrong")
    assert store.rate("a", "up", "wrong")  # a reason only goes with a down vote
    assert store.conn.execute("SELECT rating, reason FROM ratings").fetchall() == [("up", None)]


def test_summary():
    rows = [("catalog_lookup", "high", "answered", "up", None),
            ("catalog_lookup", "high", "answered", "down", "wrong"),
            ("catalog_lookup", "low", "verification_refusal", "down", "should_have_answered"),
            ("pi_classifier", "medium", "answered", None, None)]
    s = summarize(rows)
    assert (s["answers"], s["rated"], s["feedback_rate"], s["thumbs_up_rate"]) == (4, 3, 0.75, 0.333)
    assert s["by_agent"]["pi_classifier"] == {"answers": 1, "rated": 0, "feedback_rate": 0.0, "thumbs_up_rate": None}
    assert s["by_evidence"]["high"]["thumbs_up_rate"] == 0.5
    assert s["by_outcome"]["verification_refusal"]["thumbs_up_rate"] == 0.0
    assert s["down_reasons"] == {"wrong": 1, "should_have_answered": 1}
    assert s["high_evidence_thumbs_down"] == 1
    assert summarize([])["thumbs_up_rate"] is None


@pytest.mark.skipif(not catalog_path().exists(), reason="catalog.db missing: run python3 scripts/ingest_catalog.py")
def test_rate_an_answer_through_the_api(feedback_db):
    c = TestClient(create_app(providers={
        "router": router_says({"agent": "catalog_lookup", "request": "Who owns dim_customers?", "reason": "owner"}),
        "catalog_lookup": lookup_agent()}))
    answer_id = c.post("/ask", json={"request": "Who owns dim_customers?"}).json()["answers"][0]["answer_id"]

    assert c.post("/feedback", json={"answer_id": answer_id, "rating": "down", "reason": "incomplete"}).status_code == 204
    assert c.post("/feedback", json={"answer_id": "unknown", "rating": "up"}).status_code == 404
    assert c.post("/feedback", json={"answer_id": answer_id, "rating": "meh"}).status_code == 422
    s = c.get("/feedback/summary").json()
    assert (s["answers"], s["rated"], s["thumbs_up_rate"], s["down_reasons"]) == (1, 1, 0.0, {"incomplete": 1})

    stored = sqlite3.connect(feedback_db).execute("SELECT * FROM answers").fetchone()
    assert not any("dim_customers" in str(v) or "Maya" in str(v) for v in stored)  # no question or answer text
