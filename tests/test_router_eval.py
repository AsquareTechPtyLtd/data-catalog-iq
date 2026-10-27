"""Tests for the router eval: the key file and scoring (no network)."""

import pytest

from catalog_iq.catalog.db import catalog_path, connect
from catalog_iq.evals.router import load_key, score, summarize
from catalog_iq.router import _table_names

pytestmark = pytest.mark.skipif(not catalog_path().exists(),
                                reason="catalog.db missing: run python3 scripts/ingest_catalog.py")


@pytest.fixture(scope="module")
def names():
    c = connect()
    yield _table_names(c)
    c.close()


def key(i):
    return next(k for k in load_key(("dev", "heldout")) if k.id == i)


def decision(*routes, status="routed", confidence="high", alternative=None):
    return {"status": status, "routes": [dict(r) for r in routes], "confidence": confidence,
            "alternative": alternative}


def r(agent, table=None, request="q"):
    return {"agent": agent, "table": table, "request": request}


def test_key_shape():
    keys = load_key(("dev", "heldout"))
    assert len(keys) == 41 and sum(k.set == "dev" for k in keys) == 29
    assert len({k.id for k in keys}) == 41
    assert all(k.expect.clarify != bool(k.expect.routes) for k in keys)


def test_multi_intent_order_ignored(names):
    k = key("M1")
    got = decision(r("pi_classifier", "fernwood_dw.marts.fct_orders"), r("catalog_lookup"))
    res = score(got, k, 1, names)
    assert res.route_ok and res.table_ok and res.passed


def test_wrong_table_fails_table_only(names):
    res = score(decision(r("pi_classifier", "fernwood_dw.raw_storefront.customers")), key("P1"), 1, names)
    assert res.route_ok and not res.table_ok and not res.passed


def test_clarify_expected(names):
    assert score(decision(status="clarify", confidence="low"), key("C1"), 1, names).passed
    d = decision(r("data_category_recommender", "fernwood_dw.raw_web.page_views"))
    assert not score(d, key("C1"), 1, names).passed


def test_rewrite_must_name_the_table(names):
    k = key("F2")
    assert score(decision(r("catalog_lookup", request="Who owns mart_web_sessions?")), k, 1, names).passed
    res = score(decision(r("catalog_lookup", request="What about its owner?")), k, 1, names)
    assert res.route_ok and res.rewrite_ok is False and not res.passed


def test_summary_calibration(names):
    k1, k2 = key("L1"), key("P1")
    alt = {"routes": [r("pi_classifier", "fernwood_dw.raw_crm.contacts")]}
    res = [score(decision(r("catalog_lookup")), k1, 1, names),
           score(decision(r("catalog_lookup"), confidence="medium", alternative=alt), k2, 1, names)]
    s = summarize(res, [k1, k2])
    assert s["accuracy_by_confidence"] == {"high": 1.0, "medium": 0.0}
    assert s["alternative_was_expected"] == 1.0 and s["route_accuracy"] == 0.5 and "P1" in s["failing"]
