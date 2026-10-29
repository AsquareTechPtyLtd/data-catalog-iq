"""Catalog freshness: version, as_of, stale and failed ingestion, in the API (freshness design)."""

import json
import shutil
import sqlite3
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from catalog_iq.api import create_app
from catalog_iq.catalog.db import catalog_path
from catalog_iq.catalog.freshness import catalog_info, catalog_version
from catalog_iq.core.loop import AgentRun
from catalog_iq.router import RUNNERS
from test_api import FakeResponse, lookup_agent, router_says
from test_stream import OWNER, stream

NOW = datetime(2026, 10, 29, 12, tzinfo=timezone.utc)


def tiny_catalog(path, built="2026-10-29T10:00:00+00:00", collibra="2026-10-27T12:00:00Z",
                 dbt="2026-10-28T12:00:00Z"):
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE ingest_runs (run_id TEXT, ingested_at TEXT, schema_version INTEGER,
                                  source TEXT, path TEXT, sha256 TEXT);
        CREATE TABLE settings (key TEXT PRIMARY KEY, value TEXT);
    """)
    conn.executemany("INSERT INTO ingest_runs VALUES ('run1', ?, 4, ?, 'p', 'h')",
                     [(built, "collibra"), (built, "dbt")])
    conn.executemany("INSERT INTO settings VALUES (?, ?)",
                     [("collibra_exported_at", collibra), ("dbt_generated_at", dbt)])
    conn.commit()
    conn.row_factory = sqlite3.Row  # as catalog.db.connect does
    return conn


def log(db, *statuses):
    with (db.parent / "ingest_log.jsonl").open("a") as f:
        for s in statuses:
            f.write(json.dumps({"status": s, "finished_at": "2026-10-29T11:00:00+00:00",
                                **({"error": "people is empty"} if s == "failed" else {})}) + "\n")


def test_as_of_is_the_oldest_source(tmp_path):
    info = catalog_info(tiny_catalog(tmp_path / "catalog.db"), now=NOW)
    assert info.version == "run1" and info.built_at == "2026-10-29T10:00:00+00:00"
    assert info.as_of == "2026-10-27T12:00:00+00:00" and info.age_days == 2.0
    assert info.sources == {"collibra": "2026-10-27T12:00:00Z", "dbt": "2026-10-28T12:00:00Z"}
    assert not info.stale and info.max_age_hours == 168 and info.failed_ingest is None


def test_a_source_without_a_date_counts_as_built_at(tmp_path):
    conn = tiny_catalog(tmp_path / "catalog.db", built="2026-10-20T00:00:00+00:00")
    conn.execute("DELETE FROM settings")
    info = catalog_info(conn, now=NOW)
    assert info.as_of == "2026-10-20T00:00:00+00:00" and info.sources == {"collibra": None, "dbt": None}


def test_stale_past_the_max_age(tmp_path, monkeypatch):
    conn = tiny_catalog(tmp_path / "catalog.db", collibra="2026-10-01T02:00:00Z")
    info = catalog_info(conn, now=NOW)
    assert info.stale and info.age_days == 28.4
    monkeypatch.setenv("CATALOG_MAX_AGE_HOURS", "720")
    assert not catalog_info(conn, now=NOW).stale


def test_failed_ingest_only_while_it_is_the_latest_attempt(tmp_path):
    db = tmp_path / "catalog.db"
    conn = tiny_catalog(db)
    log(db, "published", "failed")
    failed = catalog_info(conn, now=NOW).failed_ingest
    assert failed.error == "people is empty" and failed.finished_at == "2026-10-29T11:00:00+00:00"
    log(db, "rolled_back")
    assert catalog_info(conn, now=NOW).failed_ingest is None


# --- API, on a copy of the real catalog (away from the real ingest log)

@pytest.fixture
def db(tmp_path):
    if not catalog_path().exists():
        pytest.skip("catalog.db missing: run python3 scripts/ingest_catalog.py")
    path = tmp_path / "catalog.db"
    shutil.copy(catalog_path(), path)
    return path


def app(db, **providers):
    return TestClient(create_app(catalog=db, providers=providers))


def test_catalog_endpoint_and_health(db, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    monkeypatch.setenv("GEMINI_API_KEY", "x")
    with sqlite3.connect(db) as conn:
        version = catalog_version(conn)
    monkeypatch.setenv("CATALOG_MAX_AGE_HOURS", "100000")
    c = app(db)
    info = c.get("/catalog").json()
    assert info["version"] == version and not info["stale"] and info["failed_ingest"] is None
    assert c.get("/health").json()["status"] == "ok"

    monkeypatch.setenv("CATALOG_MAX_AGE_HOURS", "1")  # the synthetic exports are weeks old
    h = c.get("/health").json()
    assert h["status"] == "degraded" and h["catalog"]["stale"]

    monkeypatch.setenv("CATALOG_MAX_AGE_HOURS", "100000")
    log(db, "failed")
    h = c.get("/health").json()
    assert h["status"] == "degraded" and h["catalog"]["failed_ingest"]["error"] == "people is empty"
    assert c.get("/catalog").json()["failed_ingest"]["error"] == "people is empty"


def test_every_answer_carries_the_catalog(db, monkeypatch):
    c = app(db, router=router_says(OWNER), catalog_lookup=lookup_agent())
    body = c.post("/ask", json={"request": "Who owns dim_customers?"}).json()
    assert body["catalog"]["version"] and body["answers"][0]["catalog_version"] == body["catalog"]["version"]

    monkeypatch.setitem(RUNNERS, "pi_classifier", lambda r, conn, p: AgentRun(FakeResponse(agent="pi", summary="s"), [], []))
    direct = c.post("/agents/pi_classifier", json={"table": "page_views"}).json()
    assert direct["catalog"]["version"] == body["catalog"]["version"]


def test_stream_decision_carries_the_catalog(db):
    c = app(db, router=router_says(OWNER), catalog_lookup=lookup_agent())
    events = stream(c)
    decision = next(e for e in events if e["event"] == "decision")
    answer = next(e for e in events if e["event"] == "answer")
    assert answer["catalog_version"] == decision["catalog"]["version"]
    assert set(decision["catalog"]) >= {"as_of", "stale", "age_days", "failed_ingest"}
