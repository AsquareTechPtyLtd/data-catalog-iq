"""Tests for filter_tables against the real catalog.db."""

import json

import pytest

from catalog_iq.catalog.db import catalog_path, connect
from catalog_iq.core.guardrails import validate_args
from catalog_iq.tools import filter as f
from catalog_iq.tools.filter import FILTER_TOOLS, filter_tables

pytestmark = pytest.mark.skipif(not catalog_path().exists(),
                                reason="catalog.db missing: run python3 scripts/ingest_catalog.py")

HERO = {"certification": "certified", "pi_type": "email", "data_subject": "customer"}
HERO_TABLES = ["fernwood_dw.marts.dim_customers", "fernwood_dw.marts.mart_loyalty_members",
               "fernwood_dw.marts.mart_marketing_audience", "fernwood_dw.raw_crm.contacts",
               "fernwood_dw.raw_marketing.email_events", "fernwood_dw.raw_storefront.customers"]
NO_OWNER = ["fernwood_dw.marts.mart_web_sessions", "fernwood_dw.raw_returns.return_requests"]


@pytest.fixture(scope="module")
def conn():
    c = connect()
    yield c
    c.close()


def fqns(result):
    return [t["fqn"] for t in result["tables"]]


def by_fqn(result):
    return {t["fqn"]: t for t in result["tables"]}


def values(conn, sql):
    return sorted(r[0] for r in conn.execute(sql))


# --- matching ---------------------------------------------------------------------


def test_hero_filter_finds_the_six_tables(conn):
    r = filter_tables(conn, **HERO)
    assert r["status"] == "found" and fqns(r) == HERO_TABLES
    assert r["query"]["count"] == 6 and r["query"]["matched"] == HERO_TABLES
    assert r["query"]["filters"]["pi_status"] == "confirmed"  # PI defaults to confirmed


@pytest.mark.parametrize("trap", [
    "fernwood_dw.marts.mart_email_engagement",   # hashed email only
    "fernwood_dw.raw_hr.employees",              # employee email
    "fernwood_dw.marts.dim_employees",
    "fernwood_dw.marts.dim_products",            # rejected email_template_id flag
    "fernwood_dw.marts.legacy_customer_export",  # uncertified
    "fernwood_dw.raw_helpdesk.tickets",
])
def test_hero_traps_are_excluded(conn, trap):
    assert trap not in fqns(filter_tables(conn, **HERO))


def test_matched_columns_say_why(conn):
    cols = by_fqn(filter_tables(conn, **HERO))["fernwood_dw.raw_marketing.email_events"]["matched_columns"]
    assert [(c["column"], c["category"], c["data_subject"]) for c in cols] == [("recipient_email", "email", "customer")]
    assert cols[0]["sources"] == ["collibra:fernwood_dw.raw_marketing.email_events.recipient_email"]


def test_pi_conditions_hold_on_the_same_column(conn):
    # employee tables have email columns and customer-free PI; email+customer must be one column
    r = filter_tables(conn, pi_type="email", data_subject="employee")
    assert set(fqns(r)) == {"fernwood_dw.marts.dim_employees", "fernwood_dw.raw_hr.employees"}


def test_hashed_email_is_its_own_category(conn):
    assert fqns(filter_tables(conn, pi_type="hashed_email", data_subject="customer")) == [
        "fernwood_dw.marts.mart_email_engagement"]


def test_owner_role_only(conn):
    r = filter_tables(conn, owner="Maya Okafor")
    assert set(fqns(r)) == {"fernwood_dw.marts.customer_lifetime_value", "fernwood_dw.marts.dim_customers",
                            "fernwood_dw.marts.legacy_customer_export", "fernwood_dw.marts.mart_loyalty_members",
                            "fernwood_dw.raw_crm.contacts"}
    assert by_fqn(r)["fernwood_dw.marts.legacy_customer_export"]["status"] == "deprecated"


def test_data_category_and_suggested_pi(conn):
    assert len(fqns(filter_tables(conn, data_category="DC-001"))) == 5
    assert set(fqns(filter_tables(conn, pi_status="suggested"))) == {
        "fernwood_dw.marts.mart_support_ticket_summary", "fernwood_dw.raw_helpdesk.tickets"}


def test_gap_filters(conn):
    r = filter_tables(conn, gap_type="retention_not_defined")
    assert len(fqns(r)) == 5
    assert all(g["type"] == "retention_not_defined" for t in r["tables"] for g in t["matched_gaps"])
    r = filter_tables(conn, gap_type="any")
    assert r["query"]["count"] == 15
    gaps = by_fqn(r)["fernwood_dw.raw_returns.return_requests"]["matched_gaps"]
    assert {"contract_missing", "pi_unclassified"} <= {g["type"] for g in gaps}
    assert gaps[0]["sources"] == ["catalog:fernwood_dw.raw_returns.return_requests#governance_gaps"]


# --- scope (D4) -------------------------------------------------------------------


def test_default_scope_is_sources_and_marts(conn):
    assert fqns(filter_tables(conn, has_contract=False)) == NO_OWNER  # not the 20 stg/int tables
    assert len(fqns(filter_tables(conn, sensitivity="Highly Confidential"))) == 6
    assert len(fqns(filter_tables(conn, sensitivity="Highly Confidential", include_internal=True))) == 10


def test_internal_layers_imply_include_internal(conn):
    r = filter_tables(conn, layer="staging", pi_type="email", data_subject="customer")
    assert r["query"]["scope"] == "all" and len(fqns(r)) == 3
    assert fqns(filter_tables(conn, status="in_development")) == ["fernwood_dw.marts.mart_store_performance"]


def test_sorted_marts_first(conn):
    layers = [t["layer"] for t in filter_tables(conn, certification="certified")["tables"]]
    assert layers == sorted(layers, key=f.LAYER_ORDER.get)


# --- owners, contract, escalation (D5, D12) ---------------------------------------


def test_owner_and_contract_on_each_table(conn):
    t = by_fqn(filter_tables(conn, **HERO))["fernwood_dw.raw_crm.contacts"]
    assert t["owner"]["names"] == ["Maya Okafor", "Samuel Adeyemi"] and t["owner"]["contract"] == "active"
    assert t["owner"]["sources"] == ["contract:data/sources/contracts/raw_crm.contacts.yaml"]
    assert t["catalog"] == {"domain": "customer", "sensitivity": "Highly Confidential", "data_category": "DC-001",
                            "sources": ["catalog:fernwood_dw.raw_crm.contacts"]}
    assert "loaded_from" in t and t["type"] == "table"  # lineage node shape


def test_no_owner_carries_escalation(conn):
    for t in filter_tables(conn, owner_status="none")["tables"]:
        assert t["owner"]["contract"] == "missing" and t["owner"]["escalation"]["team"] == "Data Strategy"


# --- the query block (D7) and limits (D9) -----------------------------------------


def test_no_match_is_citable(conn):
    r = filter_tables(conn, domain="support", sensitivity="Highly Confidential")
    assert r["status"] == "no_match" and r["tables"] == [] and r["query"]["count"] == 0
    assert r["query"]["text"].startswith("No tables match domain=support, sensitivity=Highly Confidential")
    assert r["query"]["sources"] == ["catalog:filter?domain=support&sensitivity=Highly+Confidential"]
    assert r["citations"][0]["url"].endswith("/search?domain=support&sensitivity=Highly+Confidential")


def test_customer_salary_is_no_match(conn):
    assert filter_tables(conn, pi_type="salary", data_subject="customer")["status"] == "no_match"


def test_query_id_is_canonical(conn):
    a = filter_tables(conn, sensitivity="Internal", has_contract=True)["query"]["sources"]
    b = filter_tables(conn, has_contract=True, sensitivity="Internal")["query"]["sources"]
    assert a == b == ["catalog:filter?has_contract=true&sensitivity=Internal"]
    assert filter_tables(conn, sensitivity="Internal", include_internal=True)["query"]["sources"] == [
        "catalog:filter?sensitivity=Internal&scope=all"]


def test_a_condition_is_required(conn):
    for kwargs in ({}, {"include_internal": True, "with_consumers": True, "limit": 5}):
        r = filter_tables(conn, **kwargs)
        assert r["status"] == "error" and "at least one condition" in r["error"]


def test_limit_truncates_explicitly(conn):
    r = filter_tables(conn, status="active", include_internal=True)
    assert r["query"]["count"] == 48 and r["query"]["returned"] == f.DEFAULT_LIMIT == len(r["tables"])
    assert r["truncated"] is True and r["query"]["matched"] == fqns(r)
    assert "truncated" not in filter_tables(conn, certification="certified")


def test_inline_ids_and_linked_citations(conn):
    r = filter_tables(conn, **HERO, with_consumers=True)
    assert all("url" in c for c in r["citations"])
    assert r["query"]["sources"][0] in {c["id"] for c in r["citations"]}


# --- unclassified candidates (D3 ii) ----------------------------------------------


def test_unclassified_candidate_is_reported(conn):
    r = filter_tables(conn, pi_type="email", data_subject="customer")
    assert len(fqns(r)) == 9
    assert [(c["fqn"], c["column"]) for c in r["unclassified_candidates"]] == [
        ("fernwood_dw.raw_returns.return_requests", "contact_email")]
    assert "fernwood_dw.raw_returns.return_requests" not in r["query"]["matched"]


def test_candidates_respect_the_other_conditions(conn):
    assert "unclassified_candidates" not in filter_tables(conn, **HERO)  # return_requests is uncertified


# --- consumers (D5) ---------------------------------------------------------------


def test_consumers_for_the_hero_question(conn):
    r = filter_tables(conn, **HERO, with_consumers=True)
    raw = by_fqn(r)["fernwood_dw.raw_storefront.customers"]["consumers"]
    assert "fernwood_dw.marts.dim_customers" in raw["marts_and_exposures"]
    assert not any(".staging." in n or ".intermediate." in n for n in raw["marts_and_exposures"])
    assert raw["staging_intermediate_count"] > 0
    node_ids = [n.get("fqn") or n["id"] for n in r["nodes"]]
    assert len(node_ids) == len(set(node_ids)) and not set(node_ids) & set(HERO_TABLES)  # each listed once
    assert "exposure.fernwood.churn_model" in node_ids


def test_edges_prove_indirect_links(conn):
    edges = {tuple(e) for e in filter_tables(conn, **HERO, with_consumers=True)["edges"]}
    assert ("fernwood_dw.raw_storefront.customers", "fernwood_dw.staging.stg_storefront__customers") in edges
    assert ("fernwood_dw.marts.mart_loyalty_members", "exposure.fernwood.loyalty_dashboard") in edges


def test_consumers_need_a_narrow_filter(conn):
    r = filter_tables(conn, certification="certified", with_consumers=True)
    assert "nodes" not in r and "10 or fewer" in r["consumers_note"]


def test_consumers_dropped_when_too_large(conn):
    r = filter_tables(conn, certification="certified", owner="Samuel Adeyemi", with_consumers=True)
    assert len(fqns(r)) == 9 and "nodes" not in r and "get_lineage" in r["consumers_note"]
    assert not any(c["id"].startswith("dbt:exposure") for c in r["citations"])  # no stray consumer citations


def test_realistic_results_fit_the_size_target(conn):
    cases = [HERO | {"with_consumers": True}, {"pi_type": "email", "data_subject": "customer",
                                               "with_consumers": True},
             {"certification": "certified"}, {"gap_type": "any"},
             {"certification": "certified", "owner": "Samuel Adeyemi", "with_consumers": True}]
    for kwargs in cases:
        assert len(json.dumps(filter_tables(conn, **kwargs))) <= f.TARGET_CHARS, kwargs


def test_largest_possible_result_is_not_trimmed(conn):
    from catalog_iq.core.guardrails import MAX_RESULT_CHARS
    sizes = [len(json.dumps(filter_tables(conn, status=s, include_internal=True))) for s in f.STATUSES]
    assert max(sizes) < MAX_RESULT_CHARS  # 25 tables at the limit still fit the tool guardrail


# --- definition -------------------------------------------------------------------


@pytest.mark.parametrize("enum, sql", [
    (f.CERTIFICATION, "SELECT DISTINCT certification FROM tables"),
    (f.LAYERS, "SELECT DISTINCT layer FROM tables"),
    (f.STATUSES, "SELECT DISTINCT status FROM tables"),
    (f.DOMAINS, "SELECT DISTINCT domain FROM tables WHERE domain IS NOT NULL"),
    (f.SENSITIVITIES, "SELECT DISTINCT sensitivity FROM tables WHERE sensitivity IS NOT NULL"),
    (f.OWNER_STATUSES, "SELECT DISTINCT owner_status FROM tables"),
    (f.OWNERS, "SELECT DISTINCT full_name FROM ownership WHERE role = 'Owner'"),
    (f.DATA_CATEGORIES, "SELECT data_category_id FROM data_categories"),
    (f.GAP_TYPES, "SELECT DISTINCT gap_type FROM governance_gaps"),
    (f.PI_TYPES, "SELECT DISTINCT pi_type FROM columns WHERE pi_type IS NOT NULL"),
    (f.DATA_SUBJECTS, "SELECT DISTINCT data_subject FROM columns WHERE data_subject IS NOT NULL"),
])
def test_enums_match_the_catalog(conn, enum, sql):
    assert sorted(enum) == values(conn, sql)


def test_tool_definition_matches_function():
    tool = FILTER_TOOLS[0]
    assert tool["function"].__name__ == tool["name"] == "filter_tables"
    params = set(tool["input_schema"]["properties"])
    assert set(f.CONDITIONS) | {"include_internal", "with_consumers", "limit"} == params


def test_args_validated():
    schema = FILTER_TOOLS[0]["input_schema"]
    assert validate_args(HERO | {"with_consumers": True}, schema) == []
    assert validate_args({"owner": "Rahul Menon"}, schema)[0].startswith("'owner' must be one of")
    assert validate_args({"has_contract": "no"}, schema) == ["'has_contract' must be boolean"]
    assert validate_args({"certification": "certified", "limit": 26}, schema) == ["'limit' must be <= 25"]
