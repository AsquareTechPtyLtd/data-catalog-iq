"""Tests for the Data Category Recommender eval: answer key vs catalog, scoring, metrics."""

import pytest

from catalog_iq.catalog.db import catalog_path, connect
from catalog_iq.evals.data_category import TableKey, load_key, score, summarize, truth_rank
from catalog_iq.tools.data_category_evidence import get_data_category_evidence
from catalog_iq.tools.data_categories import search_data_categories

pytestmark = pytest.mark.skipif(not catalog_path().exists(),
                                reason="catalog.db missing: run python3 scripts/ingest_catalog.py")


@pytest.fixture(scope="module")
def conn():
    c = connect()
    yield c
    c.close()


# --- answer key -----------------------------------------------------------------------


def test_key_sets():
    key = load_key()
    sets = {s: [k.table for k in key if k.set == s] for s in ("open", "dev", "heldout")}
    assert (len(sets["open"]), len(sets["dev"]), len(sets["heldout"])) == (3, 19, 8)
    assert len({k.table for k in key}) == 30


def test_open_tables_are_the_ones_without_a_data_category(conn):
    no_data_category = {r[0] for r in conn.execute("SELECT fqn FROM tables WHERE layer IN ('source', 'mart') "
                                         "AND data_category_id IS NULL AND status != 'in_development'")}
    assert {k.table for k in load_key(("open",))} == no_data_category


def test_blind_truth_matches_the_catalog(conn):
    """The key comes from the Collibra export; agreeing with catalog.db also checks ingest."""
    recorded = dict(conn.execute("SELECT fqn, data_category_id FROM tables WHERE layer IN ('source', 'mart') "
                                 "AND data_category_id IS NOT NULL"))
    blind = {k.table: k.data_category for k in load_key(("dev", "heldout"))}
    assert blind == recorded


def test_key_data_categories_exist(conn):
    data_categories = {r[0] for r in conn.execute("SELECT data_category_id FROM data_categories")}
    for k in load_key():
        assert k.data_category in data_categories and set(k.acceptable) <= data_categories
        assert k.data_category not in k.acceptable


def test_true_data_category_is_retrievable_for_every_open_table(conn):
    """Retrieval caps accuracy: each open table's own words must find its true data category."""
    for k in load_key(("open",)):
        r = get_data_category_evidence(conn, k.table)
        assert ("definition", (k.data_category,)) in {(s["signal"], tuple(s["suggests"])) for s in r["signals"]}


# --- scoring --------------------------------------------------------------------------


KEY = TableKey(table="fernwood_dw.raw_crm.contacts", set="dev", data_category="DC-001", acceptable=["DC-002"],
               trap="dob pulls toward DC-002")


def response(status="recommended", data_category_id="DC-001", confidence="high", outcome="answered", attempts=1):
    recommendation = {"data_category_id": data_category_id} if status == "recommended" else None
    return {"status": status, "recommendation": recommendation,
            "confidence": confidence,
            "guardrails": {"outcome": outcome, "attempts": attempts, "events": []}}


@pytest.mark.parametrize("resp, outcome", [
    (response(), "correct"),
    (response(data_category_id="DC-002"), "acceptable"),
    (response(data_category_id="DC-008"), "wrong"),
    (response(status="no_fit"), "no_fit"),
    (response(status="needs_manual_review", confidence="low"), "flagged"),
    (response(status="needs_manual_review", outcome="refused"), "refused"),
    (response(status="not_found"), "error"),
])
def test_score_outcomes(resp, outcome):
    r = score(resp, [], KEY)
    assert r.outcome == outcome and r.passed == (outcome == "correct")


def test_truth_rank_from_search_results(conn):
    results = [search_data_categories(conn, "customer shopper"), search_data_categories(conn, "loyalty"),
               {"status": "found", "signals": []}]  # not a search result
    assert truth_rank(results, "DC-001") == 1
    assert truth_rank(results, "DC-005") is None  # listed (complete) but never matched


def test_summarize():
    searched = [{"data_categories": [{"data_category_id": "DC-001", "rank": 2, "matched": True}], "complete": True}]
    results = [score(response(), searched, KEY, 1),
               score(response(data_category_id="DC-002", confidence="medium"), [], KEY, 2),
               score(response(data_category_id="DC-008", confidence="medium", attempts=2), [], KEY, 3),
               score(response(status="needs_manual_review", confidence="low"), [], KEY, 4)]
    s = summarize(results)
    assert (s["accuracy"], s["lenient_accuracy"], s["precision"], s["flagged_rate"]) == (0.25, 0.5, 0.333, 0.25)
    assert s["calibration"] == {"high": 1.0, "medium": 0.5}
    assert s["retrieval_recall_at_20"] == 0.25 and s["truth_rank_1"] == 0.0
    assert s["retrieval_mrr"] == 0.125  # one run ranked 2nd, three never retrieved
    assert s["traps"] == {KEY.table: "1/4"} and s["guardrails"]["retry_rate"] == 0.25
    assert KEY.table in s["errors_by_table"]
