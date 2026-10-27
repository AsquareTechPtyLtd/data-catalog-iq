"""Tests for the data category tools: search_data_categories, get_data_category (shared) and get_data_category_evidence (Data Category Recommender)."""

import json

import pytest

from catalog_iq.catalog.db import catalog_path, connect
from catalog_iq.tools.data_category_evidence import data_category_evidence_tools, get_data_category_evidence, lineage_neighbours
from catalog_iq.tools.data_categories import DATA_CATEGORY_TOOLS, fts_query, get_data_category, search_data_categories

pytestmark = pytest.mark.skipif(not catalog_path().exists(),
                                reason="catalog.db missing: run python3 scripts/ingest_catalog.py")

RETURNS = "fernwood_dw.raw_returns.return_requests"
PAGE_VIEWS = "fernwood_dw.raw_web.page_views"
CONTACTS = "fernwood_dw.raw_crm.contacts"


@pytest.fixture(scope="module")
def conn():
    c = connect()
    yield c
    c.close()


def matched(result):
    return [b["data_category_id"] for b in result["data_categories"] if b["matched"]]


def kinds(result):
    return {(s["signal"], tuple(s["suggests"])) for s in result["signals"]}


# --- search_data_categories ----------------------------------------------------------------------


def test_index_has_every_data_category(conn):
    assert conn.execute("SELECT count(*) FROM data_category_search").fetchone()[0] == 13


def test_search_ranks_by_meaning_words(conn):
    assert matched(search_data_categories(conn, "return requests refunds"))[0] == "DC-008"   # "returns" in the definition
    assert matched(search_data_categories(conn, "website page views sessions"))[0] == "DC-005"
    assert matched(search_data_categories(conn, "RMA"))[0] == "DC-008"                       # glossary synonym


def test_small_catalog_lists_every_data_category_matches_first(conn):
    r = search_data_categories(conn, "page views")
    assert r["complete"] and len(r["data_categories"]) == 13
    flags = [b["matched"] for b in r["data_categories"]]
    assert flags == sorted(flags, reverse=True)  # matches, then the rest
    assert [b["rank"] for b in r["data_categories"] if b["matched"]] == list(range(1, r["matched"] + 1))


def test_k_below_catalog_size_returns_matches_only(conn):
    r = search_data_categories(conn, "customer employee order", k=3)
    assert not r["complete"] and len(r["data_categories"]) == 3 and all(b["matched"] for b in r["data_categories"])
    assert "search again" in r["note"]


def test_domain_filter(conn):
    r = search_data_categories(conn, "contact information", domain="people")
    assert {b["data_category_id"] for b in r["data_categories"]} == {"DC-010", "DC-011", "DC-012", "DC-013"}
    assert all(b["domain"] == "people" for b in r["data_categories"])


def test_query_syntax_is_inert(conn):
    assert fts_query('x AND ( "y" NEAR') == '"near"'  # operators quoted or dropped, 1-letter words dropped
    r = search_data_categories(conn, "the of a")  # only stopwords
    assert r["status"] == "no_match" and r["complete"] and len(r["data_categories"]) == 13


def test_data_category_shape_and_citation(conn):
    b = next(b for b in search_data_categories(conn, "pseudonymized hashed")["data_categories"] if b["data_category_id"] == "DC-006")
    assert b["status"] == "Candidate" and b["retention"] == "not defined"
    assert b["sources"] == ["collibra:DC-006"]
    b5 = next(b for b in search_data_categories(conn, "web")["data_categories"] if b["data_category_id"] == "DC-005")
    assert b5["retention"] == {"months": 3, "trigger": "after collection"}


def test_data_category_tools_never_list_tables(conn):
    """Blind evals rely on this: a data category tool must not reveal which tables use a data category."""
    for r in (search_data_categories(conn, "customer order web"), get_data_category(conn, "DC-001")):
        assert "fernwood_dw." not in json.dumps(r)


# --- get_data_category --------------------------------------------------------------------------


def test_get_data_category(conn):
    r = get_data_category(conn, "dc-008")  # ids are case-insensitive
    b = r["data_category"]
    assert r["status"] == "found" and b["data_category_id"] == "DC-008"
    assert b["retention"]["months"] == 84 and "disposal_action" in b["retention"]
    assert {"Data Owner", "Data Steward"} <= set(b["stewards"])
    assert "Return Request" in {t["term"] for t in b["glossary_terms"]}


def test_get_data_category_not_found(conn):
    r = get_data_category(conn, "DC-099")
    assert r["status"] == "not_found" and "search_data_categories" in r["hint"]


# --- get_data_category_evidence -----------------------------------------------------------------


def test_open_table_evidence(conn):
    r = get_data_category_evidence(conn, "raw_returns.return_requests")
    assert r["status"] == "found" and r["table"]["fqn"] == RETURNS and r["table"]["domain"] == "finance"
    assert {c["column"] for c in r["columns"]} >= {"return_id", "contact_email", "refund_amount_usd"}
    assert r["neighbours"] == [{**r["neighbours"][0], "data_category_id": None}]  # stg_ only, no data category either
    assert ("definition", ("DC-008",)) in kinds(r)
    assert {("data_category_domain", ("DC-003",)), ("data_category_domain", ("DC-008",))} <= kinds(r)
    assert all(s["sources"] for s in r["signals"])  # every signal is citable
    assert not any(s["signal"] == "neighbour" for s in r["signals"])


def test_pi_is_context_only(conn):
    r = get_data_category_evidence(conn, "raw_web.page_views")
    pi = {c["column"]: c.get("pi") for c in r["columns"]}
    assert pi["ip_address"] == {"status": "confirmed", "category": "ip_address", "data_subject": "customer"}
    assert pi["page_path"] is None
    assert all(s["signal"] != "pi" for s in r["signals"])


def test_lineage_neighbours_all_hops(conn):
    near = lineage_neighbours(conn, PAGE_VIEWS)
    assert near == [("fernwood_dw.staging.stg_web__page_views", "downstream", 1),
                    ("fernwood_dw.intermediate.int_web_sessions", "downstream", 2),
                    ("fernwood_dw.marts.mart_web_sessions", "downstream", 3)]


def test_neighbour_data_category_is_a_strong_signal(conn, monkeypatch):
    r = get_data_category_evidence(conn, "marts.mart_store_performance")  # dbt-only, no data category; its upstream tables have one
    assert r["status"] == "found"
    strong = [s for s in r["signals"] if s["signal"] == "neighbour"]
    assert strong and all(s["strong"] for s in strong)


def test_assigned_table_needs_no_recommendation(conn):
    r = get_data_category_evidence(conn, CONTACTS)
    assert r["status"] == "assigned" and r["data_category"]["data_category_id"] == "DC-001"
    assert r["data_category"]["sources"] == [f"collibra:{CONTACTS}"]


def test_blind_hides_the_table_and_neighbour_data_categories(conn):
    r = get_data_category_evidence(conn, CONTACTS, blind=True)
    assert r["status"] == "found"
    assert r["neighbours"] and all("data_category_id" not in n for n in r["neighbours"])
    assert not any(s["signal"] == "neighbour" for s in r["signals"])
    near = {n["fqn"] for n in r["neighbours"]}
    shown = {t for d in r["same_domain_data_categories"] for t in d["tables"]}
    assert CONTACTS not in shown and not (near & shown)
    assert shown == {"fernwood_dw.raw_storefront.customers",  # same domain, not in its lineage
                     "fernwood_dw.marts.legacy_customer_export"}


def test_blind_is_not_in_the_tool_schema():
    tools = {t["name"]: t for t in data_category_evidence_tools(blind=True)}
    assert set(tools) == {"get_data_category_evidence", *(t["name"] for t in DATA_CATEGORY_TOOLS)}
    assert set(tools["get_data_category_evidence"]["input_schema"]["properties"]) == {"table"}


def test_stg_and_int_are_not_applicable(conn):
    r = get_data_category_evidence(conn, "stg_web__page_views")
    assert r["status"] == "not_applicable" and r["inherits_from"] == PAGE_VIEWS
    assert get_data_category_evidence(conn, "int_web_sessions")["inherits_from"] == "fernwood_dw.marts.mart_web_sessions"


def test_not_found(conn):
    assert get_data_category_evidence(conn, "raw_web.clicks")["status"] == "not_found"
