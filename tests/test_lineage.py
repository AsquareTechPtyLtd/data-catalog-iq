"""Tests for the lineage tools against the real catalog.db."""

import json

import pytest

from catalog_iq.catalog.db import catalog_path, connect
from catalog_iq.core.guardrails import validate_args
from catalog_iq.tools.lineage import LINEAGE_TOOLS, get_lineage, trace_path

pytestmark = pytest.mark.skipif(not catalog_path().exists(),
                                reason="catalog.db missing: run python3 scripts/ingest_catalog.py")

DIM_CUSTOMERS = "fernwood_dw.marts.dim_customers"
LOYALTY_MART = "fernwood_dw.marts.mart_loyalty_members"
LOYALTY_DASHBOARD = "exposure.fernwood.loyalty_dashboard"


@pytest.fixture(scope="module")
def conn():
    c = connect()
    yield c
    c.close()


def ids(nodes):
    return [n.get("fqn") or n["id"] for n in nodes]


def by_id(nodes):
    return {n.get("fqn") or n["id"]: n for n in nodes}


def cited_ids(result):
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


# --- get_lineage ------------------------------------------------------------------


def test_direct_consumers_of_dim_customers(conn):
    r = get_lineage(conn, "dim_customers", "downstream", depth=1)
    assert r["status"] == "found" and r["root"]["fqn"] == DIM_CUSTOMERS
    assert set(ids(r["downstream"])) == {
        "fernwood_dw.marts.customer_lifetime_value", "fernwood_dw.marts.mart_marketing_audience",
        "fernwood_dw.marts.mart_campaign_performance", LOYALTY_MART,
        "fernwood_dw.marts.mart_support_ticket_summary", "exposure.fernwood.churn_model"}
    assert "upstream" not in r


def test_full_downstream_walk_sorted_by_depth(conn):
    r = get_lineage(conn, "dim_customers", "downstream")
    depths = [(n["depth"], n.get("fqn") or n["id"]) for n in r["downstream"]]
    assert depths == sorted(depths)
    assert by_id(r["downstream"])[LOYALTY_DASHBOARD]["depth"] == 2
    assert "more_beyond" not in r


def test_loaded_from_lists_every_source_of_a_multi_source_table(conn):
    r = get_lineage(conn, "dim_customers", "downstream")
    assert by_id(r["downstream"])[LOYALTY_MART]["loaded_from"] == [
        DIM_CUSTOMERS, "fernwood_dw.staging.stg_crm__contacts", "fernwood_dw.staging.stg_stores__pos_transactions"]


def test_upstream_includes_staging_and_intermediate(conn):
    r = get_lineage(conn, "dim_customers", "upstream")
    layers = {n["fqn"]: n["layer"] for n in r["upstream"]}
    assert layers["fernwood_dw.intermediate.int_customer_identity"] == "intermediate"
    assert layers["fernwood_dw.staging.stg_crm__contacts"] == "staging"
    assert layers["fernwood_dw.raw_crm.contacts"] == "source"
    assert "downstream" not in r


def test_both_directions(conn):
    r = get_lineage(conn, "mart_loyalty_members", "both")
    assert LOYALTY_DASHBOARD in ids(r["downstream"])
    assert DIM_CUSTOMERS in ids(r["upstream"])


def test_depth_limit_flags_incomplete_walk(conn):
    r = get_lineage(conn, "raw_storefront.customers", "downstream", depth=2)
    assert max(n["depth"] for n in r["downstream"]) == 2
    assert r["stopped_at_depth"] == 2 and r["more_beyond"] == ["downstream"]


def test_table_with_no_consumers(conn):
    r = get_lineage(conn, "mart_web_sessions", "downstream")
    assert r["status"] == "found" and r["downstream"] == []
    assert "Nothing downstream" in r["downstream_note"]["text"]
    assert r["downstream_note"]["sources"] == ["dbt:model.fernwood.mart_web_sessions"]


def test_node_details_and_citations(conn):
    node = by_id(get_lineage(conn, "dim_customers", "downstream")["downstream"])[LOYALTY_MART]
    assert node["type"] == "table" and node["layer"] == "mart" and node["status"] == "active"
    assert node["owner"]["names"] == ["Maya Okafor"]
    assert node["owner"]["sources"] == ["contract:data/sources/contracts/marts.mart_loyalty_members.yaml"]
    assert node["certification"] == {"status": "certified",
                                     "sources": [f"catalog:{LOYALTY_MART}#certification"]}
    assert node["sources"] == ["dbt:model.fernwood.mart_loyalty_members"]


def test_inherited_owner(conn):
    node = by_id(get_lineage(conn, "dim_customers", "upstream")["upstream"])[
        "fernwood_dw.intermediate.int_customer_identity"]
    assert node["owner"]["status"] == "inherited" and node["owner"]["names"] == ["Maya Okafor"]
    assert node["owner"]["inherited_from"] == DIM_CUSTOMERS
    assert node["owner"]["sources"] == ["contract:data/sources/contracts/marts.dim_customers.yaml"]


def test_ownerless_node_carries_escalation(conn):
    r = get_lineage(conn, "raw_web.page_views", "downstream")
    owner = by_id(r["downstream"])["fernwood_dw.marts.mart_web_sessions"]["owner"]
    assert owner["status"] == "none" and owner["names"] == []
    assert owner["escalation"]["team"] == "Data Strategy"
    assert owner["sources"] == ["catalog:fernwood_dw.marts.mart_web_sessions#ownership"]


def test_exposure_node(conn):
    node = by_id(get_lineage(conn, "dim_customers", "downstream")["downstream"])["exposure.fernwood.churn_model"]
    assert node["type"] == "exposure" and node["exposure_type"] == "ml"
    assert node["owner"]["names"] == ["Maya Okafor"]
    assert node["sources"] == ["dbt:exposure.fernwood.churn_model"]


def test_exposure_as_root(conn):
    r = get_lineage(conn, "churn_model", "upstream", depth=1)
    assert r["root"]["id"] == "exposure.fernwood.churn_model"
    assert set(ids(r["upstream"])) == {DIM_CUSTOMERS, "fernwood_dw.marts.customer_lifetime_value"}


def test_unknown_table(conn):
    r = get_lineage(conn, "kafka_clickstream", "downstream")
    assert r["status"] == "not_found" and "find_table" in r["hint"]


def test_every_cited_id_is_described(conn):
    for table in ("dim_customers", "raw_web.page_views", "raw_storefront.customers"):
        r = get_lineage(conn, table, "both")
        assert cited_ids(r) <= {c["id"] for c in r["citations"]}


def test_largest_walk_fits_result_limit(conn):
    sizes = [len(json.dumps(get_lineage(conn, fqn, "both")))
             for (fqn,) in conn.execute("SELECT fqn FROM tables")]
    assert max(sizes) < 18_000  # tool guardrail trims at 20k


# --- trace_path -------------------------------------------------------------------


def test_trace_path_finds_both_routes(conn):
    r = trace_path(conn, "raw_crm.contacts", "loyalty_dashboard")
    assert r["status"] == "found" and r["to"] == LOYALTY_DASHBOARD
    assert r["paths"] == [
        ["fernwood_dw.raw_crm.contacts", "fernwood_dw.staging.stg_crm__contacts", LOYALTY_MART, LOYALTY_DASHBOARD],
        ["fernwood_dw.raw_crm.contacts", "fernwood_dw.staging.stg_crm__contacts",
         "fernwood_dw.intermediate.int_customer_identity", DIM_CUSTOMERS, LOYALTY_MART, LOYALTY_DASHBOARD]]
    assert set(ids(r["nodes"])) == {n for p in r["paths"] for n in p}
    assert "paths_truncated" not in r


def test_trace_path_reversed(conn):
    r = trace_path(conn, "loyalty_dashboard", "raw_crm.contacts")
    assert r["status"] == "no_path" and r["reverse_path_exists"] is True


def test_trace_path_unrelated(conn):
    r = trace_path(conn, "raw_hr.employees", "loyalty_dashboard")
    assert r["status"] == "no_path" and r["reverse_path_exists"] is False


def test_trace_path_unknown_node(conn):
    assert trace_path(conn, "raw_crm.contacts", "kafka dashboard")["status"] == "not_found"


def test_trace_path_cap(conn, monkeypatch):
    monkeypatch.setattr("catalog_iq.tools.lineage.MAX_PATHS", 1)
    r = trace_path(conn, "raw_crm.contacts", "loyalty_dashboard")
    assert len(r["paths"]) == 1 and r["paths_truncated"] is True


# --- definitions ------------------------------------------------------------------


def test_tool_definitions_match_functions():
    for tool in LINEAGE_TOOLS:
        assert {"name", "description", "input_schema", "function"} <= tool.keys()
        assert tool["function"].__name__ == tool["name"]
        assert set(tool["input_schema"]["required"]) <= set(tool["input_schema"]["properties"])


def test_direction_and_depth_validated():
    schema = LINEAGE_TOOLS[0]["input_schema"]
    assert validate_args({"table": "x", "direction": "downstream", "depth": 2}, schema) == []
    assert validate_args({"table": "x", "direction": "sideways"}, schema) == [
        "'direction' must be one of upstream, downstream, both"]
    assert validate_args({"table": "x", "direction": "upstream", "depth": 11}, schema) == ["'depth' must be <= 10"]
