"""Tests for the column profiles computed at ingest (data-model §3.2, PI agent D5)."""

import importlib.util
import json
from pathlib import Path

import pytest

from catalog_iq.catalog.db import catalog_path, connect

spec = importlib.util.spec_from_file_location(
    "ingest_catalog", Path(__file__).resolve().parents[1] / "scripts" / "ingest_catalog.py")
ingest = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ingest)

needs_catalog = pytest.mark.skipif(not catalog_path().exists(),
                                   reason="catalog.db missing: run python3 scripts/ingest_catalog.py")


def test_profile_ratios():
    p = ingest.profile_column(["a@x.com", "b@y.org", "a@x.com", None])
    assert p == {"row_count": 4, "null_ratio": 0.25, "distinct_count": 2, "distinct_ratio": 0.667,
                 "avg_length": 7.0, "avg_words": 1.0, "pattern_matches": {"email": 1.0}}


def test_profile_numbers_and_empty():
    p = ingest.profile_column([1.5, 2.0, 2.0])
    assert p["avg_length"] is None and p["pattern_matches"] == {} and p["distinct_ratio"] == 0.667
    assert ingest.profile_column([None, None])["distinct_ratio"] is None
    assert ingest.profile_column([])["null_ratio"] is None


@pytest.mark.parametrize("value, pattern", [
    ("+1-775-555-0117", "phone"), ("(503) 555-0171", "phone"), ("203.0.113.12", "ipv4"),
    ("2003-06-23", "iso_date"), ("2026-09-22T10:00:00", "iso_timestamp"), ("Nora Abbott", "person_name"),
    ("0332", "four_digits"), ("83722", "five_digits"), ("0" * 64, "sha256_hex")])
def test_patterns_match_one_shape(value, pattern):
    assert {k for k, p in ingest.PROFILE_PATTERNS.items() if p.match(value)} == {pattern}


@pytest.fixture(scope="module")
def conn():
    c = connect()
    yield c
    c.close()


def profile(conn, fqn, column):
    r = conn.execute("SELECT * FROM column_profiles WHERE table_fqn = ? AND column_name = ?", (fqn, column)).fetchone()
    return {**dict(r), "pattern_matches": json.loads(r["pattern_matches"])}


@needs_catalog
def test_every_warehouse_column_is_profiled(conn):
    missing = conn.execute("""SELECT count(*) FROM columns c JOIN tables t ON t.fqn = c.table_fqn
                              WHERE c.in_warehouse = 1 AND NOT EXISTS (SELECT 1 FROM column_profiles p
                              WHERE p.table_fqn = c.table_fqn AND p.column_name = c.name)""").fetchone()[0]
    assert missing == 0


@needs_catalog
def test_profiles_hold_aggregates_only(conn):
    for r in conn.execute("SELECT * FROM column_profiles"):
        assert all(isinstance(r[k], (int, float, type(None))) for k in r.keys()
                   if k not in ("table_fqn", "column_name", "pattern_matches"))
        matches = json.loads(r["pattern_matches"])
        assert set(matches) <= set(ingest.PROFILE_PATTERNS) and all(0 < v <= 1 for v in matches.values())


@needs_catalog
def test_pi_agent_signals(conn):
    returns, tickets = "fernwood_dw.raw_returns.return_requests", "fernwood_dw.raw_helpdesk.tickets"
    assert profile(conn, returns, "contact_email")["pattern_matches"] == {"email": 1.0}
    subject, body = profile(conn, tickets, "subject"), profile(conn, tickets, "body")
    assert subject["distinct_count"] == 8 and subject["distinct_ratio"] < 0.01   # template-driven
    assert body["distinct_ratio"] > 0.5 and body["avg_words"] > 10              # free text
    assert profile(conn, "fernwood_dw.marts.dim_customers", "phone")["pattern_matches"] == {"phone": 1.0}
    assert "phone" not in profile(conn, "fernwood_dw.marts.mart_web_sessions", "ip_address")["pattern_matches"]
