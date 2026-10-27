"""Tests for the guardrails, using real tools and hand-written drafts (no LLM)."""

import json

import pytest

from catalog_iq.catalog.db import catalog_path, connect
from catalog_iq.core.faithfulness import Vocabulary, lineage_nodes
from catalog_iq.core.guardrails import MAX_ATTEMPTS, Guard, validate_args
from catalog_iq.core.response import AgentDraft
from catalog_iq.tools.tables import TOOLS

pytestmark = pytest.mark.skipif(not catalog_path().exists(),
                                reason="catalog.db missing: run python3 scripts/ingest_catalog.py")

DIM = "fernwood_dw.marts.dim_customers"
DIM_CONTRACT = "contract:data/sources/contracts/marts.dim_customers.yaml"
WEB = "fernwood_dw.marts.mart_web_sessions"
WEB_OWNERSHIP = f"catalog:{WEB}#ownership"


@pytest.fixture(scope="module")
def conn():
    c = connect()
    yield c
    c.close()


@pytest.fixture(scope="module")
def vocab(conn):
    return Vocabulary(conn)


@pytest.fixture
def guard(conn, vocab):
    return Guard(conn, agent="catalog_lookup", question="test question", vocab=vocab)


def draft(answer, facts=(), tables=(), status="answered", clarification=None):
    return AgentDraft(status=status, answer=answer, tables=list(tables), clarification=clarification,
                      facts=[{"claim": c, "citations": ids} for c, ids in facts])


def checks(events):
    return [e.check for e in events]


# --- tool guardrail -------------------------------------------------------------------


def test_good_call_passes_through(guard):
    result = guard.call("get_table", {"table": "dim_customers"})
    assert result["status"] == "found"
    assert guard.tool_results == [result]
    assert guard.evidence[0].result_status == "found" and guard.events == []


def test_unknown_tool_blocked(guard):
    result = guard.call("drop_table", {"table": "dim_customers"})
    assert result["status"] == "error" and "Available: find_table, get_table" in result["error"]
    assert checks(guard.events) == ["tool_unknown"]
    assert guard.evidence[0].result_status == "blocked" and guard.tool_results == []


@pytest.mark.parametrize("name, args, message", [
    ("get_table", {}, "missing required argument 'table'"),
    ("get_table", {"table": 5}, "'table' must be string"),
    ("get_table", {"table": "  "}, "'table' must not be empty"),
    ("get_table", {"table": "x", "sql": "DROP"}, "unknown argument 'sql'"),
    ("find_table", {"query": "x", "limit": 50}, "'limit' must be <= 10"),
    ("find_table", {"query": "x", "limit": True}, "'limit' must be integer"),
    ("find_table", {"query": "x", "kind": "column"}, "'kind' must be one of table, exposure"),
    ("find_table", "dim_customers", "arguments must be an object"),
])
def test_bad_args_blocked(guard, name, args, message):
    result = guard.call(name, args)
    assert result["status"] == "error" and message in result["error"]
    assert checks(guard.events) == ["bad_args"]


def test_validate_args_accepts_valid():
    schema = next(t for t in TOOLS if t["name"] == "find_table")["input_schema"]
    assert validate_args({"query": "loyalty", "limit": 3}, schema) == []


def test_tool_budget(conn, vocab):
    guard = Guard(conn, "catalog_lookup", "q", vocab=vocab, max_tool_calls=2)
    guard.call("get_table", {"table": "dim_customers"})
    guard.call("nope", {})                       # blocked calls count towards the budget
    result = guard.call("get_table", {"table": "fct_orders"})
    assert result["status"] == "error" and "Answer with the results you already have" in result["error"]
    assert checks(guard.events) == ["tool_unknown", "tool_budget_exceeded"]


def test_large_result_trimmed(conn, vocab):
    guard = Guard(conn, "catalog_lookup", "q", vocab=vocab, max_result_chars=3500)  # full result ~4.8k
    result = guard.call("get_table", {"table": "raw_returns.return_requests"})
    assert len(json.dumps(result)) <= 3500 + 200          # + the truncation note
    assert "truncated" in result and len(result["governance_gaps"]) < 7
    assert {c["id"] for c in result["citations"]}         # citations never trimmed
    assert checks(guard.events) == ["result_truncated"]


# --- output guardrail -----------------------------------------------------------------


def good_d1(guard):
    guard.call("get_table", {"table": "dim_customers"})
    return draft("dim_customers is owned by Maya Okafor.",
                 facts=[("Owner of dim_customers: Maya Okafor", [DIM_CONTRACT])], tables=[DIM])


def test_good_draft_passes(guard):
    assert guard.check_output(good_d1(guard)) == []


@pytest.mark.parametrize("answer, facts, tables, expected", [
    ("dim_customers is owned by Rahul Menon.", [("Owner: Rahul Menon", [DIM_CONTRACT])], [DIM],
     "unsupported_claim"),
    ("dim_customers is owned by Maya Okafor.", [("Owner: Maya Okafor", ["contract:made/up.yaml"])], [DIM],
     "invalid_citation"),
    ("Owned by Maya Okafor; Daniel Reyes is the engineer.", [("Owner: Maya Okafor", [DIM_CONTRACT])], [DIM],
     "answer_entity_not_in_facts"),
    ("dim_customers feeds fct_orders; owned by Maya Okafor.", [("Owner: Maya Okafor", [DIM_CONTRACT])], [DIM],
     "answer_entity_not_in_facts"),  # only the subject table (dim_customers) is exempt
    ("dim_customers is owned by Maya Okafor.", [("Owner: Maya Okafor", [DIM_CONTRACT])],
     [DIM, "fernwood_dw.marts.customer_master"], "table_not_returned"),
])
def test_bad_draft_triggers_event(guard, answer, facts, tables, expected):
    guard.call("get_table", {"table": "dim_customers"})
    events = guard.check_output(draft(answer, facts, tables))
    assert expected in checks(events)
    assert all(e.part == "output" and e.attempt == 1 for e in events)


def test_answered_without_evidence(guard):
    guard.call("find_table", {"query": "kafka clickstream"})
    events = guard.check_output(draft("The Kafka table is owned by Maya Okafor.",
                                      facts=[("Owner: Maya Okafor", [DIM_CONTRACT])]))
    assert "no_evidence" in checks(events)


def test_escalation_must_be_in_answer(guard):
    guard.call("get_table", {"table": "mart_web_sessions"})
    facts = [("mart_web_sessions has no data contract", [WEB_OWNERSHIP])]
    events = guard.check_output(draft("mart_web_sessions has no data contract.", facts, [WEB]))
    assert checks(events) == ["escalation_not_in_answer"]


def test_substitute_owner(guard):
    guard.call("get_table", {"table": "mart_web_sessions"})
    events = guard.check_output(draft(
        "mart_web_sessions has no data contract. Maya Okafor owns it. Ask Data Strategy.",
        facts=[("mart_web_sessions has no data contract", [WEB_OWNERSHIP])], tables=[WEB]))
    assert "substitute_owner" in checks(events)


def test_feedback_lists_problems(guard):
    guard.call("get_table", {"table": "dim_customers"})
    events = guard.check_output(draft("Owned by Rahul Menon.", [("Owner: Rahul Menon", [DIM_CONTRACT])], [DIM]))
    message = guard.feedback(events)
    assert "unsupported_claim: fact 1 'Rahul Menon' not in cited source" in message
    assert "insufficient_context" in message


# --- completion -----------------------------------------------------------------------


def test_complete_high_confidence(guard):
    d = good_d1(guard)
    assert guard.check_output(d) == []
    r = guard.complete(d)
    assert r.confidence == "high" and r.escalation is None
    assert r.citations[0].id == DIM_CONTRACT and r.citations[0].source == "contract"
    assert r.tables[0].url == f"https://catalog.fernwood.example/table/{DIM}"
    assert (r.guardrails.attempts, r.guardrails.outcome, r.guardrails.events) == (1, "passed", [])


def test_single_clear_match_stays_high(guard):
    guard.call("find_table", {"query": "loyalty"})   # other matches score 0.3: not plausible
    guard.call("get_table", {"table": "fernwood_dw.marts.mart_loyalty_members"})
    d = draft("mart_loyalty_members is owned by Maya Okafor.",
              facts=[("Owner: Maya Okafor", ["contract:data/sources/contracts/marts.mart_loyalty_members.yaml"])],
              tables=["fernwood_dw.marts.mart_loyalty_members"])
    assert guard.check_output(d) == []
    assert guard.complete(d).confidence == "high"


def test_picked_over_strong_match_is_medium(guard):
    guard.call("find_table", {"query": "customers"})  # raw_storefront.customers also matches strongly
    d = good_d1(guard)
    assert guard.check_output(d) == []
    assert guard.complete(d).confidence == "medium"


def test_escalation_attached(guard):
    guard.call("get_table", {"table": "mart_web_sessions"})
    d = draft("mart_web_sessions has no data contract, so no owner is on record. Contact Data Strategy.",
              facts=[("mart_web_sessions has no data contract", [WEB_OWNERSHIP])], tables=[WEB])
    assert guard.check_output(d) == []
    r = guard.complete(d)
    assert r.escalation.email == "data-strategy@fernwood.example" and r.confidence == "high"


def test_passed_after_retry_is_medium(guard):
    guard.call("get_table", {"table": "dim_customers"})
    assert guard.check_output(draft("Owned by Rahul Menon.", [("Owner: Rahul Menon", [DIM_CONTRACT])], [DIM]))
    fixed = draft("dim_customers is owned by Maya Okafor.", [("Owner: Maya Okafor", [DIM_CONTRACT])], [DIM])
    assert guard.check_output(fixed) == []
    r = guard.complete(fixed)
    assert r.confidence == "medium"
    assert (r.guardrails.attempts, r.guardrails.outcome) == (2, "passed_after_retry")
    assert [e.attempt for e in r.guardrails.events] == [1]


def test_refuse_after_failed_retry(guard):
    guard.call("get_table", {"table": "dim_customers"})
    bad = draft("Owned by Rahul Menon.", [("Owner: Rahul Menon", [DIM_CONTRACT])], [DIM])
    for _ in range(MAX_ATTEMPTS):
        assert guard.check_output(bad)
    r = guard.refuse()
    assert r.status == "insufficient_context" and r.confidence == "low"
    assert r.facts == [] and "won't guess" in r.answer and "Data Strategy" in r.answer
    assert r.escalation.team == "Data Strategy"
    assert (r.guardrails.attempts, r.guardrails.outcome) == (2, "refused")
    assert [e.attempt for e in r.guardrails.events] == [1, 2]


def test_insufficient_context_is_low(guard):
    guard.call("find_table", {"query": "kafka clickstream"})
    d = draft("No table matching 'Kafka clickstream' is in the catalog.", status="insufficient_context")
    assert guard.check_output(d) == []
    r = guard.complete(d)
    assert r.confidence == "low" and r.escalation is None


@pytest.mark.parametrize("claim, expected", [
    ("dim_customers has 2,000 rows", []),          # source says 2000
    ("dim_customers has 2000 rows", []),
    ("dim_customers has 3,000 rows", ["unsupported_claim"]),
])
def test_thousands_separator(guard, claim, expected):
    guard.call("get_table", {"table": "dim_customers"})
    d = draft(f"{claim}.", facts=[(claim, [f"collibra:{DIM}"])], tables=[DIM])
    assert checks(guard.check_output(d)) == expected


# --- lineage (D8, D9) -----------------------------------------------------------------

LOYALTY = "fernwood_dw.marts.mart_loyalty_members"
CLV = "fernwood_dw.marts.customer_lifetime_value"
DASHBOARD = "exposure.fernwood.loyalty_dashboard"
LOYALTY_DBT = "dbt:model.fernwood.mart_loyalty_members"
CLV_DBT = "dbt:model.fernwood.customer_lifetime_value"
DASHBOARD_DBT = "dbt:exposure.fernwood.loyalty_dashboard"
LOYALTY_CONTRACT = "contract:data/sources/contracts/marts.mart_loyalty_members.yaml"


@pytest.fixture
def lguard(conn, vocab):
    """A guard with the lineage tools, after a downstream walk from dim_customers."""
    from catalog_iq.tools.lineage import LINEAGE_TOOLS
    g = Guard(conn, agent="catalog_lookup", question="q", tools=TOOLS + LINEAGE_TOOLS, vocab=vocab)
    g.call("get_lineage", {"table": "dim_customers", "direction": "downstream"})
    return g


def rel(up, down, kind="direct"):
    return {"upstream": up, "downstream": down, "kind": kind}


def ldraft(answer, facts, tables=(DIM,), status="answered"):
    """facts: (claim, citations, relations)"""
    return AgentDraft(status=status, answer=answer, tables=list(tables),
                      facts=[{"claim": c, "citations": ids, "relations": rels} for c, ids, rels in facts])


def test_lineage_draft_with_relations_passes(lguard):
    claim = "dim_customers is consumed by mart_loyalty_members and customer_lifetime_value"
    d = ldraft(claim + ".", [(claim, [LOYALTY_DBT, CLV_DBT], [rel(DIM, LOYALTY), rel(DIM, CLV)])],
               tables=[DIM, LOYALTY, CLV])
    assert lguard.check_output(d) == []
    r = lguard.complete(d)
    assert {c.id: c.source for c in r.citations} == {LOYALTY_DBT: "dbt", CLV_DBT: "dbt"}  # from compact ids
    assert r.confidence == "high"


def test_l8_trap_direct_link_that_skips_a_hop(lguard):
    claim = "dim_customers feeds the Loyalty Program Dashboard directly"
    d = ldraft(claim + ".", [(claim, [DASHBOARD_DBT, LOYALTY_DBT], [rel(DIM, DASHBOARD)])])
    events = lguard.check_output(d)
    assert checks(events) == ["unsupported_relation"]
    assert "not a direct link" in events[0].detail


def test_indirect_link_passes(lguard):
    claim = "dim_customers feeds the Loyalty Program Dashboard through mart_loyalty_members"
    d = ldraft(claim + ".", [(claim, [DASHBOARD_DBT, LOYALTY_DBT],
                              [rel(DIM, DASHBOARD, "indirect"), rel(DIM, LOYALTY), rel(LOYALTY, DASHBOARD)])],
               tables=[DIM, LOYALTY])
    assert lguard.check_output(d) == []


def test_direct_edge_called_indirect(lguard):
    claim = "dim_customers feeds mart_loyalty_members"
    events = lguard.check_output(ldraft(claim, [(claim, [LOYALTY_DBT], [rel(DIM, LOYALTY, "indirect")])],
                                        tables=[DIM, LOYALTY]))
    assert checks(events) == ["unsupported_relation"] and "direct link, not indirect" in events[0].detail


def test_relation_must_cite_downstream_lineage_id(lguard):
    claim = "dim_customers feeds mart_loyalty_members"
    events = lguard.check_output(ldraft(claim, [(claim, [LOYALTY_CONTRACT], [rel(DIM, LOYALTY)])],
                                        tables=[DIM, LOYALTY]))
    assert "unsupported_relation" in checks(events)
    assert any(e.detail.endswith(f"must cite {LOYALTY_DBT}") for e in events)  # names the exact id


def test_lineage_claim_without_relations(lguard):
    claim = "mart_loyalty_members is built from dim_customers"
    events = lguard.check_output(ldraft(claim, [(claim, [LOYALTY_DBT], [])], tables=[DIM, LOYALTY]))
    assert checks(events) == ["relation_not_structured"]


def test_non_lineage_claim_needs_no_relations(lguard):
    claim = "mart_loyalty_members is owned by Maya Okafor"  # owner cited from the lineage node's owner section
    assert lguard.check_output(ldraft(claim, [(claim, [LOYALTY_CONTRACT], [])], tables=[DIM, LOYALTY])) == []


def test_exposure_in_answer_must_be_in_facts(lguard):
    claim = "dim_customers is consumed by mart_loyalty_members"
    d = ldraft("dim_customers is consumed by mart_loyalty_members, which feeds the Loyalty Program Dashboard.",
               [(claim, [LOYALTY_DBT], [rel(DIM, LOYALTY)])], tables=[DIM, LOYALTY])
    assert [(e.check, e.detail) for e in lguard.check_output(d)] == [("answer_entity_not_in_facts", DASHBOARD)]


def test_lineage_tables_count_as_returned_but_exposures_are_not_tables(lguard):
    claim = "dim_customers is consumed by mart_loyalty_members"
    d = ldraft(claim, [(claim, [LOYALTY_DBT], [rel(DIM, LOYALTY)])], tables=[DIM, LOYALTY, DASHBOARD])
    events = lguard.check_output(d)
    assert checks(events) == ["table_not_returned"] and "is an exposure" in events[0].detail


def test_ownerless_lineage_node_needs_escalation(conn, vocab):
    from catalog_iq.tools.lineage import LINEAGE_TOOLS
    g = Guard(conn, agent="catalog_lookup", question="q", tools=TOOLS + LINEAGE_TOOLS, vocab=vocab)
    g.call("get_lineage", {"table": "raw_web.page_views", "direction": "downstream"})
    claim = "mart_web_sessions has no owner on record"
    facts = [(claim, [WEB_OWNERSHIP], [])]
    tables = ["fernwood_dw.raw_web.page_views", WEB]
    assert checks(g.check_output(ldraft(claim + ".", facts, tables))) == ["escalation_not_in_answer"]
    d = ldraft(claim + "; contact the Data Strategy team.", facts, tables)
    assert g.check_output(d) == []
    assert g.complete(d).escalation.team == "Data Strategy"


def test_ownerless_node_not_in_tables_needs_no_escalation(conn, vocab):
    from catalog_iq.tools.lineage import LINEAGE_TOOLS
    g = Guard(conn, agent="catalog_lookup", question="q", tools=TOOLS + LINEAGE_TOOLS, vocab=vocab)
    g.call("get_lineage", {"table": "raw_web.page_views", "direction": "downstream", "depth": 1})
    claim = "raw_web.page_views is consumed by stg_web__page_views"
    d = ldraft(claim, [(claim, ["dbt:model.fernwood.stg_web__page_views"],
                        [rel("fernwood_dw.raw_web.page_views", "fernwood_dw.staging.stg_web__page_views")])],
               tables=["fernwood_dw.raw_web.page_views", "fernwood_dw.staging.stg_web__page_views"])
    assert g.check_output(d) == [] and g.complete(d).escalation is None


def test_substitute_owner_for_ownerless_lineage_node(conn, vocab):
    from catalog_iq.tools.lineage import LINEAGE_TOOLS
    g = Guard(conn, agent="catalog_lookup", question="q", tools=TOOLS + LINEAGE_TOOLS, vocab=vocab)
    g.call("get_lineage", {"table": "int_web_sessions", "direction": "both"})
    bad = "mart_web_sessions is owned by Maya Okafor"
    assert "substitute_owner" in checks(g.check_output(ldraft(bad, [(bad, [WEB_OWNERSHIP], [])], [WEB])))


def test_no_path_is_evidence(conn, vocab):
    from catalog_iq.tools.lineage import LINEAGE_TOOLS
    g = Guard(conn, agent="catalog_lookup", question="q", tools=TOOLS + LINEAGE_TOOLS, vocab=vocab)
    g.call("trace_path", {"source": "raw_hr.employees", "target": "loyalty_dashboard"})
    claim = "There is no lineage path from raw_hr.employees to the loyalty dashboard"
    endpoints = [cid for n in g.tool_results[0]["nodes"] for cid in n["sources"]]  # both endpoints' lineage
    d = ldraft(claim + ".", [(claim, endpoints, [])], tables=["fernwood_dw.raw_hr.employees"])
    assert g.check_output(d) == []


def test_exposure_search_does_not_break_confidence(conn, vocab):
    from catalog_iq.tools.lineage import LINEAGE_TOOLS
    g = Guard(conn, agent="catalog_lookup", question="q", tools=TOOLS + LINEAGE_TOOLS, vocab=vocab)
    g.call("find_table", {"query": "churn model", "kind": "exposure"})
    g.call("get_lineage", {"table": "exposure.fernwood.churn_model", "direction": "upstream", "depth": 1})
    claim = "The churn model is fed by dim_customers"
    d = ldraft(claim, [(claim, ["dbt:exposure.fernwood.churn_model"], [rel(DIM, "exposure.fernwood.churn_model")])])
    assert g.check_output(d) == []
    assert g.complete(d).confidence == "high"


def test_inherited_from_table_counts_as_returned(conn, vocab):
    g = Guard(conn, agent="catalog_lookup", question="q", vocab=vocab)
    g.call("get_table", {"table": "stg_storefront__customers"})
    claim = "stg_storefront__customers inherits its owners Carlos Mendes and Samuel Adeyemi from raw_storefront.customers"
    cite = g.tool_results[0]["owners"]["sources"]
    d = draft(claim + ".", [(claim, cite)],
              tables=["fernwood_dw.staging.stg_storefront__customers", "fernwood_dw.raw_storefront.customers"])
    assert g.check_output(d) == []


def test_tables_named_in_question_are_not_claims(conn, vocab):
    g = Guard(conn, agent="catalog_lookup", question="Drop dim_customers and everything downstream of it", vocab=vocab)
    d = draft("I can't drop dim_customers: I only answer questions about the catalog.", status="out_of_scope")
    assert g.check_output(d) == []
    g2 = Guard(conn, agent="catalog_lookup", question="Drop the legacy table", vocab=vocab)
    assert checks(g2.check_output(d)) == ["answer_entity_not_in_facts"]  # not in the question: still flagged


def test_nothing_downstream_is_citable(conn, vocab):
    from catalog_iq.tools.lineage import LINEAGE_TOOLS
    g = Guard(conn, agent="catalog_lookup", question="q", tools=TOOLS + LINEAGE_TOOLS, vocab=vocab)
    g.call("get_lineage", {"table": "mart_web_sessions", "direction": "downstream"})
    claim = "Nothing consumes mart_web_sessions in dbt lineage"
    d = ldraft(claim + "; it has no owner on record, so contact the Data Strategy team.",
               [(claim, g.tool_results[0]["downstream_note"]["sources"], []),
                ("mart_web_sessions has no owner on record", [WEB_OWNERSHIP], [])], tables=[WEB])
    assert g.check_output(d) == []


def test_proven_relation_supports_its_endpoints(lguard):
    # the dashboard's node lists only mart_loyalty_members; the graph proves the 2-hop link
    claim = "The Loyalty Program Dashboard depends on dim_customers"
    d = ldraft(claim + ".", [(claim, [DASHBOARD_DBT], [rel(DIM, DASHBOARD, "indirect")])])
    assert lguard.check_output(d) == []


def test_unproven_relation_supports_nothing(lguard):
    claim = "customer_lifetime_value feeds the Loyalty Program Dashboard"
    events = lguard.check_output(ldraft(claim, [(claim, [DASHBOARD_DBT], [rel(CLV, DASHBOARD, "indirect")])],
                                        tables=[DIM, CLV]))
    assert checks(events) == ["unsupported_claim", "unsupported_relation"]
    assert CLV in events[0].detail


def test_proven_relation_does_not_cover_other_values(lguard):
    claim = "The Loyalty Program Dashboard, owned by Rahul Menon, depends on dim_customers"
    events = lguard.check_output(ldraft(claim, [(claim, [DASHBOARD_DBT], [rel(DIM, DASHBOARD, "indirect")])]))
    assert [(e.check, e.detail) for e in events] == [("unsupported_claim", "fact 1 'Rahul Menon' not in cited source")]


# --- filter answers -----------------------------------------------------------------

HERO_ARGS = {"certification": "certified", "pi_type": "email", "data_subject": "customer"}
RAW_CUSTOMERS = "fernwood_dw.raw_storefront.customers"
CHURN = "exposure.fernwood.churn_model"


def fguard(conn, vocab, *calls):
    """A guard with all four tool groups, after the given filter_tables calls."""
    from catalog_iq.tools.filter import FILTER_TOOLS
    from catalog_iq.tools.lineage import LINEAGE_TOOLS
    g = Guard(conn, agent="catalog_lookup", question="q", tools=TOOLS + LINEAGE_TOOLS + FILTER_TOOLS, vocab=vocab)
    for args in calls:
        g.call("filter_tables", args)
    return g


def query_id(result):
    return result["query"]["sources"][0]


def owner_cite(result, fqn):
    return next(t for t in result["tables"] if t["fqn"] == fqn)["owner"]["sources"]


def test_hero_answer_passes(conn, vocab):
    g = fguard(conn, vocab, HERO_ARGS | {"with_consumers": True})
    r = g.tool_results[0]
    tables = r["query"]["matched"]
    names = ", ".join(t.split(".", 1)[1] for t in tables)
    facts = [
        (f"6 certified tables contain customer email: {names}", [query_id(r)], []),
        ("dim_customers is owned by Maya Okafor", owner_cite(r, DIM), []),
        ("The churn model consumes dim_customers", ["dbt:exposure.fernwood.churn_model"], [rel(DIM, CHURN)]),
        ("raw_storefront.customers feeds dim_customers",
         ["dbt:model.fernwood.dim_customers"], [rel(RAW_CUSTOMERS, DIM, "indirect")]),
    ]
    d = ldraft(f"6 certified tables contain customer email: {names}. dim_customers is owned by Maya Okafor; "
               "the churn model consumes it, and raw_storefront.customers feeds it.", facts, tables=tables)
    assert g.check_output(d) == []
    resp = g.complete(d)
    assert {c.id for c in resp.citations} >= {query_id(r), "dbt:model.fernwood.dim_customers"}  # inline-only ids
    assert resp.escalation is None and resp.confidence == "high"


def test_indirect_consumer_called_direct(conn, vocab):
    g = fguard(conn, vocab, HERO_ARGS | {"with_consumers": True})
    claim = "raw_storefront.customers feeds dim_customers"
    events = g.check_output(ldraft(claim, [(claim, ["dbt:model.fernwood.dim_customers"], [rel(RAW_CUSTOMERS, DIM)])],
                                   tables=[RAW_CUSTOMERS, DIM]))
    # the unproven link gives no fix-A exemption, so the name check fails too
    assert checks(events) == ["unsupported_claim", "unsupported_relation"]
    assert "not a direct link" in events[1].detail


def test_list_incomplete_names_missing_tables(conn, vocab):
    g = fguard(conn, vocab, HERO_ARGS)
    r = g.tool_results[0]
    claim = "dim_customers contains customer email and is certified"
    d = ldraft(claim, [(claim, [query_id(r)], [])], tables=[DIM])
    events = g.check_output(d)
    assert checks(events) == ["list_incomplete"]
    assert RAW_CUSTOMERS in events[0].detail and DIM not in events[0].detail.split("matched ")[1]


def test_uncited_exploratory_filter_is_not_required(conn, vocab):
    g = fguard(conn, vocab, {"certification": "certified"}, HERO_ARGS)
    hero = g.tool_results[1]
    claim = f"6 certified tables contain customer email: {', '.join(hero['query']['matched'])}"
    assert g.check_output(ldraft(claim, [(claim, [query_id(hero)], [])], tables=hero["query"]["matched"])) == []


def test_no_match_is_an_answer(conn, vocab):
    g = fguard(conn, vocab, {"domain": "support", "sensitivity": "Highly Confidential"})
    claim = "No support tables hold Highly Confidential data"
    assert g.check_output(ldraft(claim + ".", [(claim, [query_id(g.tool_results[0])], [])], tables=[])) == []


def test_count_is_checked(conn, vocab):
    g = fguard(conn, vocab, {"certification": "uncertified"})
    r = g.tool_results[0]
    good = "7 tables are uncertified"
    assert "unsupported_claim" not in checks(g.check_output(
        ldraft(good, [(good, [query_id(r)], [])], tables=r["query"]["matched"])))
    bad = "8 tables are uncertified"
    assert "unsupported_claim" in checks(g.check_output(
        ldraft(bad, [(bad, [query_id(r)], [])], tables=r["query"]["matched"])))


def test_ownerless_match_needs_escalation(conn, vocab):
    g = fguard(conn, vocab, {"certification": "uncertified"})
    r = g.tool_results[0]
    claim = "7 tables are uncertified"
    tables = r["query"]["matched"]
    assert "escalation_not_in_answer" in checks(g.check_output(ldraft(claim, [(claim, [query_id(r)], [])], tables)))
    answer = (claim + "; mart_web_sessions and raw_returns.return_requests have no contract or owner on record, "
              "so contact the Data Strategy team.")
    facts = [(claim, [query_id(r)], []),
             ("mart_web_sessions has no owner on record", owner_cite(r, WEB), []),
             ("raw_returns.return_requests has no owner on record",
              owner_cite(r, "fernwood_dw.raw_returns.return_requests"), [])]
    d = ldraft(answer, facts, tables)
    assert g.check_output(d) == []
    assert g.complete(d).escalation.team == "Data Strategy"


def test_unclassified_candidate_is_not_a_returned_table(conn, vocab):
    g = fguard(conn, vocab, {"pi_type": "email", "data_subject": "customer"})
    r = g.tool_results[0]
    claim = "raw_returns.return_requests.contact_email is not yet classified and awaits review"
    cand = r["unclassified_candidates"][0]["sources"]
    events = g.check_output(ldraft(claim, [(claim, cand, [])],
                                   tables=r["query"]["matched"] + ["fernwood_dw.raw_returns.return_requests"]))
    assert ("table_not_returned", "fernwood_dw.raw_returns.return_requests") in [(e.check, e.detail) for e in events]


def test_made_up_id_is_still_invalid(conn, vocab):
    g = fguard(conn, vocab, HERO_ARGS)
    claim = "dim_customers contains customer email"
    events = g.check_output(ldraft(claim, [(claim, ["catalog:filter?certification=all"], [])], tables=[DIM]))
    assert "invalid_citation" in checks(events)


def test_loaded_from_tables_count_as_returned(conn, vocab):
    from catalog_iq.tools.lineage import LINEAGE_TOOLS
    g = Guard(conn, agent="catalog_lookup", question="q", tools=TOOLS + LINEAGE_TOOLS, vocab=vocab)
    g.call("get_lineage", {"table": "mart_web_sessions", "direction": "downstream"})
    root = next(n for n in lineage_nodes(g.tool_results) if n.get("fqn") == WEB)
    assert root["loaded_from"]  # the root names its direct upstreams
    d = ldraft("x", [("x", [WEB_OWNERSHIP], [])], tables=[WEB, *root["loaded_from"]])
    assert "table_not_returned" not in checks(g.check_output(d))
