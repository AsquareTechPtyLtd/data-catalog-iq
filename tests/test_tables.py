"""Tests for the table lookup tools against the real catalog.db."""

import json
import sqlite3

import pytest

from catalog_iq.catalog.db import catalog_path, connect
from catalog_iq.tools.tables import TOOLS, find_table, get_table

pytestmark = pytest.mark.skipif(not catalog_path().exists(),
                                reason="catalog.db missing: run python3 scripts/ingest_catalog.py")


@pytest.fixture(scope="module")
def conn():
    c = connect()
    yield c
    c.close()


def fqns(result):
    return [m["fqn"] for m in result["matches"]]


def all_cited_ids(result):
    """Citation ids referenced anywhere in a get_table result."""
    found = set()

    def walk(node):
        if isinstance(node, dict):
            found.update(node.get("sources", []))
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)
    walk({k: v for k, v in result.items() if k != "citations"})
    return found


# --- find_table -------------------------------------------------------------------


@pytest.mark.parametrize("query", ["dim_customers", "dim customers", "marts.dim_customers",
                                   "fernwood_dw.marts.dim_customers", "  DIM_Customers "])
def test_find_exact_and_normalised_names(conn, query):
    result = find_table(conn, query)
    assert result["status"] == "found"
    assert fqns(result)[0] == "fernwood_dw.marts.dim_customers"


def test_find_partial_name(conn):
    result = find_table(conn, "loyalty")
    assert result["status"] == "found"
    assert fqns(result)[0] == "fernwood_dw.marts.mart_loyalty_members"


@pytest.mark.parametrize("query, mart, source", [
    ("customers", "fernwood_dw.marts.dim_customers", "fernwood_dw.raw_storefront.customers"),
    ("orders", "fernwood_dw.marts.fct_orders", "fernwood_dw.raw_storefront.orders"),
    ("email", "fernwood_dw.marts.mart_email_engagement", "fernwood_dw.raw_marketing.email_events"),
])
def test_find_prefers_mart_over_source(conn, query, mart, source):
    result = find_table(conn, query)
    assert result["status"] == "found"
    assert fqns(result)[0] == mart
    assert source in fqns(result)  # still listed, so the agent can say "also matched"


def test_find_qualified_source_name_is_not_overridden(conn):
    result = find_table(conn, "raw_storefront.customers")
    assert result["status"] == "found"
    assert fqns(result)[0] == "fernwood_dw.raw_storefront.customers"


@pytest.mark.parametrize("query, exposure", [
    ("the loyalty dashboard", "exposure.fernwood.loyalty_dashboard"),
    ("Loyalty Program Dashboard", "exposure.fernwood.loyalty_dashboard"),
    ("churn model", "exposure.fernwood.churn_model"),
    ("exec_revenue_dashboard", "exposure.fernwood.exec_revenue_dashboard"),
])
def test_find_exposure(conn, query, exposure):
    result = find_table(conn, query, kind="exposure")
    assert result["status"] == "found" and result["kind"] == "exposure"
    assert result["matches"][0]["id"] == exposure
    assert result["matches"][0]["sources"] == [f"dbt:{exposure}"]


def test_find_exposure_ambiguous_and_not_found(conn):
    assert find_table(conn, "dashboard", kind="exposure")["status"] == "ambiguous"
    assert find_table(conn, "kafka clickstream", kind="exposure")["status"] == "not_found"


def test_find_table_default_kind_ignores_exposures(conn):
    result = find_table(conn, "loyalty")
    assert result["kind"] == "table"
    assert all(m["fqn"].startswith("fernwood_dw.") for m in result["matches"])


def test_find_falls_back_to_source_without_a_mart(conn):
    result = find_table(conn, "page views")
    assert result["status"] == "found"
    assert fqns(result)[0] == "fernwood_dw.raw_web.page_views"


def test_find_several_marts_is_ambiguous(conn):
    result = find_table(conn, "customer")
    assert result["status"] == "ambiguous"
    assert {"fernwood_dw.marts.dim_customers", "fernwood_dw.marts.customer_lifetime_value"} <= set(fqns(result))


@pytest.mark.parametrize("query, expected", [
    ("loyalty table", "fernwood_dw.marts.mart_loyalty_members"),
    ("the orders table", "fernwood_dw.marts.fct_orders"),
    ("The customers table", "fernwood_dw.marts.dim_customers"),
])
def test_find_ignores_filler_words(conn, query, expected):
    result = find_table(conn, query)
    assert result["status"] == "found" and result["matches"][0]["fqn"] == expected
    assert result["query"] == query  # reported as typed


def test_find_only_filler_words_kept(conn):
    assert find_table(conn, "tables")["query"] == "tables"  # no crash, nothing stripped to empty


def test_find_not_found(conn):
    assert find_table(conn, "kafka streaming clickstream")["status"] == "not_found"


def test_find_hides_stg_int_from_fuzzy_search(conn):
    result = find_table(conn, "storefront customers")
    assert all(m["search_visible"] for m in result["matches"])
    assert "fernwood_dw.staging.stg_storefront__customers" not in fqns(result)


def test_find_stg_int_on_exact_name(conn):
    result = find_table(conn, "int_web_sessions")
    assert result["status"] == "found"
    assert result["matches"][0]["search_visible"] is False


def test_find_respects_limit(conn):
    assert len(find_table(conn, "customer", limit=2)["matches"]) <= 2


# --- get_table: answer-key cases (table-inventory.md) ------------------------------


def test_certified_mart_owner_data_category_steward(conn):
    r = get_table(conn, "dim_customers")
    assert r["contract"]["status"] == "active"
    assert r["owners"]["status"] == "contract"
    assert [p["name"] for p in r["owners"]["people"] if p["role"] == "Owner"] == ["Maya Okafor"]
    assert r["certification"]["status"] == "certified"
    assert r["data_category"]["data_category_id"] == "DC-001"
    assert r["data_category"]["retention"]["months"] == 36
    assert [s["name"] for s in r["data_category"]["stewards"]["Data Steward"]] == ["Rahul Menon"]
    assert "contract:data/sources/contracts/marts.dim_customers.yaml" in r["owners"]["sources"]


def test_source_lists_all_owners(conn):
    r = get_table(conn, "raw_storefront.customers")
    owners = sorted(p["name"] for p in r["owners"]["people"] if p["role"] == "Owner")
    assert owners == ["Carlos Mendes", "Samuel Adeyemi"]


@pytest.mark.parametrize("table", ["mart_web_sessions", "raw_returns.return_requests"])
def test_missing_contract_escalates_to_data_strategy(conn, table):
    r = get_table(conn, table)
    assert r["contract"]["status"] == "missing"
    assert r["owners"]["status"] == "none"
    assert r["owners"]["people"] == []
    assert r["owners"]["escalation"]["email"] == "data-strategy@fernwood.example"
    assert "contract_missing" in r["certification"]["failed_criteria"]


def test_int_owner_missing_when_built_for_mart_has_no_contract(conn):
    r = get_table(conn, "int_web_sessions")
    assert r["owners"]["status"] == "none"
    assert r["owners"]["inherited_from"] == "fernwood_dw.marts.mart_web_sessions"
    assert r["owners"]["escalation"]["team"] == "Data Strategy"
    assert r["certification"]["status"] == "not_applicable"


def test_stg_owner_inherited_from_source_contract(conn):
    r = get_table(conn, "stg_storefront__customers")
    assert r["owners"]["status"] == "inherited"
    assert r["owners"]["inherited_from"] == "fernwood_dw.raw_storefront.customers"
    assert r["contract"]["status"] == "not_applicable"
    assert "contract:data/sources/contracts/raw_storefront.customers.yaml" in r["owners"]["sources"]


def test_int_owner_inherited_from_built_for_mart(conn):
    r = get_table(conn, "int_customer_identity")
    assert r["owners"]["inherited_from"] == "fernwood_dw.marts.dim_customers"
    assert [p["name"] for p in r["owners"]["people"] if p["role"] == "Owner"] == ["Maya Okafor"]


def test_uncertified_suggested_pi(conn):
    r = get_table(conn, "raw_helpdesk.tickets")
    assert r["certification"]["failed_criteria"] == ["pi_not_reviewed"]
    suggested = {c["column"]: c["confidence"] for c in r["pi_columns"] if c["status"] == "suggested"}
    assert suggested == {"subject": 0.71, "body": 0.62}


def test_deprecated_contract_still_shows_owners(conn):
    r = get_table(conn, "legacy_customer_export")
    assert r["contract"]["status"] == "deprecated"
    assert r["certification"]["failed_criteria"] == ["contract_not_active"]
    assert any(p["name"] == "Maya Okafor" for p in r["owners"]["people"])
    assert "deprecated" in r["owners"]["note"]


def test_candidate_data_category_without_retention(conn):
    r = get_table(conn, "mart_email_engagement")
    assert r["data_category"]["data_category_id"] == "DC-006"
    assert r["data_category"]["status"] == "Candidate"
    assert r["data_category"]["retention"] == "not defined"
    assert r["certification"]["failed_criteria"] == ["data_category_not_approved"]


def test_data_category_assigned_to_the_table_in_collibra(conn):
    r = get_table(conn, "raw_storefront.orders")  # holds PI (shipping_address); the table is about orders
    assert r["data_category"]["data_category_id"] == "DC-008" and r["data_category"]["assigned"] == "collibra"
    assert r["data_category"]["sources"] == ["collibra:DC-008", "collibra:fernwood_dw.raw_storefront.orders"]
    assert "candidates" not in r["data_category"]


def test_data_category_inherited_by_stg_and_int(conn):
    stg = get_table(conn, "stg_crm__contacts")["data_category"]
    assert (stg["data_category_id"], stg["assigned"], stg["inherited_from"]) == (
        "DC-001", "inherited", "fernwood_dw.raw_crm.contacts")
    assert "catalog:fernwood_dw.staging.stg_crm__contacts#data_category_inheritance" in stg["sources"]
    built_for = get_table(conn, "int_customer_order_stats")["data_category"]  # follows built_for, not its order inputs
    assert (built_for["data_category_id"], built_for["inherited_from"]) == ("DC-001", "fernwood_dw.marts.dim_customers")


def test_no_data_category_is_a_gap(conn):
    r = get_table(conn, "raw_web.page_views")
    assert r["data_category"]["data_category_id"] is None and "No data category is assigned" in r["data_category"]["note"]
    assert any(g["type"] == "data_category_missing" for g in r["governance_gaps"])
    assert "data_category_missing" in r["certification"]["failed_criteria"]
    inherited = get_table(conn, "int_web_sessions")["data_category"]
    assert inherited["data_category_id"] is None and inherited["inherited_from"] == "fernwood_dw.marts.mart_web_sessions"


def test_columns_carry_no_data_category(conn):
    assert "candidate_data_category_id" not in [r[1] for r in conn.execute("PRAGMA table_info(columns)")]


def test_hashed_email_inherited_in_staging(conn):
    r = get_table(conn, "stg_marketing__email_events")
    hashed = [c for c in r["pi_columns"] if c["column"] == "recipient_email_sha256"]
    assert hashed[0]["category"] == "hashed_email"
    assert hashed[0]["classification"] == "inherited"


def test_employee_email_has_employee_subject(conn):
    r = get_table(conn, "dim_employees")
    emails = [c for c in r["pi_columns"] if c["category"] == "email"]
    assert emails and all(c["data_subject"] == "employee" for c in emails)


# --- get_table: lookup errors -------------------------------------------------------


def test_get_unknown_table(conn):
    r = get_table(conn, "customer_streaming_events")
    assert r["status"] == "not_found"
    assert "find_table" in r["hint"]


def test_get_requires_exact_name(conn):
    assert get_table(conn, "dim customer")["status"] == "not_found"


# --- contracts every tool result must honour ---------------------------------------

SAMPLE_TABLES = ["dim_customers", "raw_crm.contacts", "mart_web_sessions", "int_web_sessions",
                 "stg_marketing__email_events", "legacy_customer_export", "raw_returns.return_requests",
                 "mart_store_performance"]


@pytest.mark.parametrize("table", SAMPLE_TABLES)
def test_every_cited_id_is_described(conn, table):
    r = get_table(conn, table)
    described = {c["id"] for c in r["citations"]}
    assert all_cited_ids(r) <= described
    assert None not in all_cited_ids(r)


def test_citations_point_at_real_sources(conn):
    paths = {r["ref"] for r in conn.execute("SELECT ref FROM provenance WHERE source = 'contract'")}
    for table in SAMPLE_TABLES:
        for c in get_table(conn, table)["citations"]:
            if c["source"] == "contract":
                assert c["ref"] in paths


@pytest.mark.parametrize("table", SAMPLE_TABLES)
def test_results_are_json_serialisable(conn, table):
    json.dumps(get_table(conn, table))
    json.dumps(find_table(conn, table))


def test_catalog_connection_is_read_only(conn):
    with pytest.raises(sqlite3.OperationalError):
        conn.execute("DELETE FROM tables")


def test_tool_definitions_match_functions():
    for tool in TOOLS:
        assert {"name", "description", "input_schema", "function"} <= tool.keys()
        assert tool["function"].__name__ == tool["name"]
        assert set(tool["input_schema"]["required"]) <= set(tool["input_schema"]["properties"])
