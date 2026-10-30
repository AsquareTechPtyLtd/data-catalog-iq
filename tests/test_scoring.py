"""Tests for the eval answer key and scorer, using real tool results and
hand-written agent responses (no LLM)."""

import json

import pytest

from catalog_iq.catalog.db import ROOT, catalog_path, connect
from catalog_iq.core.response import AgentResponse
from catalog_iq.evals.answer_key import load_questions
from catalog_iq.evals.scoring import AgentRun, Vocabulary, run_eval, score, summarize
from catalog_iq.tools.filter import filter_tables
from catalog_iq.tools.lineage import get_lineage
from catalog_iq.tools.tables import find_table, get_table

pytestmark = pytest.mark.skipif(not catalog_path().exists(),
                                reason="catalog.db missing: run python3 scripts/ingest_catalog.py")

DIM_CONTRACT = "contract:data/sources/contracts/marts.dim_customers.yaml"
DATA_STRATEGY = {"team": "Data Strategy", "email": "data-strategy@fernwood.example", "slack": "#data-strategy"}


@pytest.fixture(scope="module")
def conn():
    c = connect()
    yield c
    c.close()


@pytest.fixture(scope="module")
def vocab(conn):
    return Vocabulary(conn)


@pytest.fixture(scope="module")
def key():
    return {q.id: q for q in load_questions("catalog_lookup")}


def make_run(question, tool_results, status, answer, facts=(), tables=(), escalation=False,
             clarification=None, confidence="high", guardrails=None):
    """Build a run the way code completes a draft: citation details come from tool results."""
    details = {c["id"]: c for r in tool_results for c in r.get("citations", [])}
    cited = {cid for f in facts for cid in f[1]}
    response = AgentResponse(
        agent="catalog_lookup", question=question.question, status=status, answer=answer,
        facts=[{"claim": f[0], "citations": f[1], "relations": f[2] if len(f) > 2 else []} for f in facts],
        tables=[{"fqn": f, "name": f.split(".")[-1], "url": f"https://catalog.fernwood.example/table/{f}"}
                for f in tables],
        escalation=DATA_STRATEGY if escalation else None, clarification=clarification,
        citations=[{"source": cid.split(":")[0], "ref": cid.split(":", 1)[1], **details.get(cid, {"id": cid})}
                   for cid in sorted(cited)],
        confidence=confidence,
        evidence=[{"tool": "get_table", "args": {}, "result_status": r.get("status", "found")}
                  for r in tool_results],
        guardrails=guardrails or {"attempts": 1, "events": [], "outcome": "passed"})
    return AgentRun(response=response, tool_results=tool_results)


def d1_good(conn, key):
    return make_run(key["D1"], [get_table(conn, "dim_customers")], "answered",
                    "dim_customers is owned by Maya Okafor.",
                    facts=[("Owner of dim_customers: Maya Okafor", [DIM_CONTRACT])],
                    tables=["fernwood_dw.marts.dim_customers"])


# --- answer key -----------------------------------------------------------------------


def test_answer_key_loads_with_expected_sets(key):
    sets = [q.set for q in key.values()]
    assert (sets.count("dev"), sets.count("negative"), sets.count("heldout")) == (27, 10, 14)
    assert sum("lineage" in q.slices for q in key.values()) == 15
    assert sum("filter" in q.slices for q in key.values()) == 18


def test_load_filters_by_set():
    assert {q.set for q in load_questions("catalog_lookup", sets=("dev",))} == {"dev"}


def test_expected_tables_exist(conn, key):
    known = {r["fqn"] for r in conn.execute("SELECT fqn FROM tables")}
    for q in key.values():
        assert set(q.expected.tables) <= known, q.id


def test_escalation_slice(key):
    assert sorted(q.id for q in key.values() if q.expected.escalation) == [
        "D4", "D6", "F2", "F3", "H1", "H11", "H6", "N6"]


SEARCHES = {"D3": "loyalty", "D10": "customers"}  # questions answered via find_table first


def test_answer_key_entities_are_reachable_with_tools(conn, key, vocab):
    """Every person, data category and table the key expects is in a tool result for the
    expected table, so a grounded agent can actually say it."""
    for q in key.values():
        results = [get_table(conn, fqn) for fqn in q.expected.tables]
        if "lineage" in q.slices:
            results += [get_lineage(conn, fqn, "both") for fqn in q.expected.tables]
        if q.id in SEARCHES:
            results.append(find_table(conn, SEARCHES[q.id]))
        if q.id in FILTERS:
            results.append(filter_tables(conn, **FILTERS[q.id], with_consumers=True))
        results = json.dumps(results).lower()
        for item in q.expected.mentions:
            for option in (item if isinstance(item, list) else [item]):
                for kind, value in vocab.entities(option):
                    short = value.split(".", 1)[1] if kind == "table" else value
                    assert short.lower() in results, (q.id, value)


# the filter_tables call each filter question needs (also checks the tool against the key)
FILTERS = {
    "F1": {"certification": "certified", "pi_type": "email", "data_subject": "customer"},
    "F2": {"certification": "uncertified"}, "F3": {"has_contract": False},
    "F4": {"sensitivity": "Highly Confidential"}, "F5": {"owner": "Maya Okafor"}, "F6": {"data_category": "DC-001"},
    "F7": {"pi_status": "suggested"}, "F8": {"pi_type": "email", "data_subject": "customer"},
    "F9": {"gap_type": "retention_not_defined"},
    "FN1": {"domain": "support", "sensitivity": "Highly Confidential"},
    "FN2": {"pi_type": "salary", "data_subject": "customer"},
    "FN4": {"pi_type": "email", "data_subject": "customer"},
    "H10": {"certification": "certified", "pi_type": "payment_card", "data_subject": "customer"},
    "H11": {"owner_status": "none"}, "H12": {"layer": "staging", "pi_type": "email", "data_subject": "customer"},
    "H13": {"pi_type": "hashed_email", "data_subject": "customer"},
    "H14": {"certification": "certified", "owner": "Samuel Adeyemi"},
}


def test_filter_questions_match_the_tool(conn, key):
    for qid, kwargs in FILTERS.items():
        exp = key[qid].expected
        got = set(filter_tables(conn, **kwargs)["query"]["matched"])
        if qid != "F8":  # F8 answers with dashboards, not its tables
            assert got == set(exp.tables), qid
        assert not got & set(exp.forbidden_tables), qid
    assert {q.id for q in key.values() if "filter" in q.slices} == set(FILTERS) | {"FN3"}


def _collibra_pi(pi_type: str, subject: str) -> set[str]:
    assets = json.loads((ROOT / "data/sources/collibra_export.json").read_text())["assets"]
    return {a["relations"]["is part of"][0] for a in assets if a["type"] == "Column"
            and a["attributes"].get("PI Status") == "Confirmed" and a["attributes"].get("PI Type") == pi_type
            and a["attributes"].get("Data Subject") == subject}


def _contract_owners() -> dict[str, list[str]]:
    import yaml
    owners = {}
    for p in (ROOT / "data/sources/contracts").glob("*.yaml"):
        c = yaml.safe_load(p.read_text())
        owners[c["schema"][0]["physicalName"]] = [t["name"] for t in c["team"] if t["role"] == "Owner"]
    return owners


def test_filter_expectations_match_source_files(key):
    """PI sets from the Collibra export and owner sets from the contracts, not catalog.db."""
    assert set(key["FN4"].expected.tables) == _collibra_pi("email", "customer")
    assert set(key["F1"].expected.tables) < _collibra_pi("email", "customer")
    assert set(key["H13"].expected.tables) == _collibra_pi("hashed_email", "customer")
    assert set(key["H10"].expected.tables) == _collibra_pi("payment_card", "customer")
    owners = _contract_owners()
    assert set(key["F5"].expected.tables) == {t for t, o in owners.items() if "Maya Okafor" in o}
    assert set(key["F3"].expected.tables) == set(key["H11"].expected.tables)
    assert not set(key["F3"].expected.tables) & set(owners)  # no contract file
    assert set(key["H14"].expected.tables) < {t for t, o in owners.items() if "Samuel Adeyemi" in o}


def _manifest_edges() -> set[tuple[str, str]]:
    m = json.loads((ROOT / "data/sources/dbt_manifest.json").read_text())
    fqn = {u: n["relation_name"] for u, n in {**m["nodes"], **m["sources"]}.items() if "relation_name" in n}
    edges = {(fqn[p], fqn[u]) for u, n in m["nodes"].items() if n["resource_type"] == "model"
             for p in n["depends_on"]["nodes"]}
    return edges | {(fqn[p], u) for u, e in m["exposures"].items() for p in e["depends_on"]["nodes"]}


def test_expected_relations_match_dbt_manifest(key):
    """Expected links are checked against the manifest, not catalog.db."""
    edges = _manifest_edges()

    def reaches(a, b, seen=()):
        return any(d == b or (d not in seen and reaches(d, b, (*seen, d))) for u, d in edges if u == a)

    for q in key.values():
        for r in q.expected.relations:
            link = (r.upstream, r.downstream)
            assert (link in edges) if r.kind == "direct" else (link not in edges and reaches(*link)), (q.id, link)


# --- correctness ----------------------------------------------------------------------


def test_good_answer_passes(conn, key, vocab):
    result = score(d1_good(conn, key), key["D1"], vocab)
    assert result.passed, (result.correctness.reasons, result.faithfulness.reasons)


def test_wrong_status(conn, key, vocab):
    run = make_run(key["D1"], [get_table(conn, "dim_customers")], "insufficient_context", "Not sure.")
    assert any(r.startswith("wrong_status") for r in score(run, key["D1"], vocab).correctness.reasons)


def test_wrong_table_and_missing_fact(conn, key, vocab):
    run = make_run(key["D1"], [get_table(conn, "raw_storefront.customers")], "answered",
                   "customers is owned by Carlos Mendes.",
                   facts=[("Owner: Carlos Mendes", ["contract:data/sources/contracts/raw_storefront.customers.yaml"])],
                   tables=["fernwood_dw.raw_storefront.customers"])
    reasons = score(run, key["D1"], vocab).correctness.reasons
    assert "wrong_table: fernwood_dw.marts.dim_customers missing" in reasons
    assert "missing_fact: Maya Okafor" in reasons


def test_unexpected_table(conn, key, vocab):
    q = key["FN2"]
    r = filter_tables(conn, pi_type="salary", data_subject="customer")
    qid = r["query"]["sources"]
    good = make_run(q, [r], "answered", "No tables contain customer salary data.",
                    facts=[("No tables match pi_type=salary, data_subject=customer", qid)])
    assert score(good, q, vocab).correctness.passed
    bad = make_run(q, [r], "answered", "No tables contain customer salary data.",
                   facts=[("No tables match pi_type=salary, data_subject=customer", qid)],
                   tables=["fernwood_dw.raw_hr.employees"])
    assert "unexpected_table: fernwood_dw.raw_hr.employees" in score(bad, q, vocab).correctness.reasons


def test_any_of_mentions_and_contractions(conn, key, vocab):
    r = get_table(conn, "mart_web_sessions")
    run = make_run(key["D4"], [r], "answered",
                   "mart_web_sessions doesn't have a data contract, so no owner is on record. "
                   "Contact the Data Strategy team.",
                   facts=[("mart_web_sessions has no data contract",
                           ["catalog:fernwood_dw.marts.mart_web_sessions#ownership"])],
                   tables=["fernwood_dw.marts.mart_web_sessions"], escalation=True)
    result = score(run, key["D4"], vocab)
    assert result.passed, (result.correctness.reasons, result.faithfulness.reasons)


def test_escalation_required(conn, key, vocab):
    r = get_table(conn, "mart_web_sessions")
    facts = [("mart_web_sessions has no data contract", ["catalog:fernwood_dw.marts.mart_web_sessions#ownership"])]
    no_esc = make_run(key["D4"], [r], "answered", "mart_web_sessions has no data contract.",
                      facts=facts, tables=["fernwood_dw.marts.mart_web_sessions"])
    assert "no_escalation" in score(no_esc, key["D4"], vocab).correctness.reasons
    silent = make_run(key["D4"], [r], "answered", "mart_web_sessions has no data contract.",
                      facts=facts, tables=["fernwood_dw.marts.mart_web_sessions"], escalation=True)
    assert "escalation_not_in_answer" in score(silent, key["D4"], vocab).correctness.reasons


def test_unexpected_escalation(conn, key, vocab):
    run = d1_good(conn, key)
    run.response.escalation = DATA_STRATEGY
    assert "unexpected_escalation" in score(run, key["D1"], vocab).correctness.reasons


def test_negative_question_abstains(conn, key, vocab):
    run = make_run(key["N1"], [find_table(conn, "Kafka clickstream")], "insufficient_context",
                   "No table matching 'Kafka clickstream' is in the catalog.", confidence="low")
    assert score(run, key["N1"], vocab).passed


# --- faithfulness ---------------------------------------------------------------------


def test_invalid_citation(conn, key, vocab):
    run = make_run(key["D1"], [get_table(conn, "dim_customers")], "answered",
                   "dim_customers is owned by Maya Okafor.",
                   facts=[("Owner: Maya Okafor", ["contract:data/sources/contracts/marts.dim_customer.yaml"])],
                   tables=["fernwood_dw.marts.dim_customers"])
    result = score(run, key["D1"], vocab)
    assert any(r.startswith("invalid_citation") for r in result.faithfulness.reasons)
    assert result.unsupported_claims == 1


def test_valid_citation_but_unsupported_claim(conn, key, vocab):
    # Rahul Menon is in the result (data category steward) but not in the contract the claim cites
    run = make_run(key["D1"], [get_table(conn, "dim_customers")], "answered",
                   "dim_customers is owned by Maya Okafor and Rahul Menon.",
                   facts=[("Owner: Maya Okafor", [DIM_CONTRACT]), ("Owner: Rahul Menon", [DIM_CONTRACT])],
                   tables=["fernwood_dw.marts.dim_customers"])
    result = score(run, key["D1"], vocab)
    assert "unsupported_claim: fact 2 'Rahul Menon' not in cited source" in result.faithfulness.reasons
    assert result.correctness.passed and not result.passed


def test_steward_claim_citing_data_category_is_supported(conn, key, vocab):
    run = d1_good(conn, key)
    run.response.facts.append(type(run.response.facts[0])(
        claim="DC-001 data steward: Rahul Menon", citations=["collibra:DC-001"]))
    run.response.citations.append(type(run.response.citations[0])(
        id="collibra:DC-001", source="collibra", ref="DC-001"))
    result = score(run, key["D1"], vocab)
    assert result.faithfulness.passed, result.faithfulness.reasons


def test_unsupported_number(conn, key, vocab):
    run = make_run(key["D8"], [get_table(conn, "dim_products")], "answered",
                   "dim_products uses DC-009; retention is 36 months.",
                   facts=[("DC-009 retention: 36 months", ["collibra:DC-009"])],
                   tables=["fernwood_dw.marts.dim_products"])
    assert "unsupported_claim: fact 1 '36' not in cited source" in score(run, key["D8"], vocab).faithfulness.reasons


def test_answer_entity_must_be_in_facts(conn, key, vocab):
    run = d1_good(conn, key)
    run.response.answer = "dim_customers is owned by Maya Okafor; Daniel Reyes is the engineer."
    reasons = score(run, key["D1"], vocab).faithfulness.reasons
    assert reasons == ["answer_entity_not_in_facts: Daniel Reyes"]


def test_substitute_owner(conn, key, vocab):
    r = get_table(conn, "mart_web_sessions")
    run = make_run(key["D4"], [r], "answered",
                   "mart_web_sessions has no data contract. Maya Okafor owns it. Ask Data Strategy.",
                   facts=[("mart_web_sessions has no data contract",
                           ["catalog:fernwood_dw.marts.mart_web_sessions#ownership"]),
                          ("Maya Okafor", ["collibra:group/Data Strategy"])],
                   tables=["fernwood_dw.marts.mart_web_sessions"], escalation=True)
    assert "substitute_owner: Maya Okafor" in score(run, key["D4"], vocab).faithfulness.reasons


def test_d10_mart_with_also_matched_source(conn, key, vocab):
    found = find_table(conn, "customers")
    run = make_run(key["D10"], [found, get_table(conn, "dim_customers")], "answered",
                   "dim_customers is the customer mart, owned by Maya Okafor. "
                   "Also matched: raw_storefront.customers (source table).",
                   facts=[("dim_customers owner: Maya Okafor", [DIM_CONTRACT]),
                          ("Also matched: raw_storefront.customers",
                           ["catalog:fernwood_dw.raw_storefront.customers"])],
                   tables=["fernwood_dw.marts.dim_customers"])
    result = score(run, key["D10"], vocab)
    assert result.passed, (result.correctness.reasons, result.faithfulness.reasons)


# --- metrics --------------------------------------------------------------------------


LOYALTY = "fernwood_dw.marts.mart_loyalty_members"
L5_SOURCES = ["fernwood_dw.marts.dim_customers", "fernwood_dw.staging.stg_crm__contacts",
              "fernwood_dw.staging.stg_stores__pos_transactions"]


def l5_run(conn, key, kind="direct", relations=True):
    claim = "mart_loyalty_members is built from dim_customers, stg_crm__contacts and stg_stores__pos_transactions"
    rels = [{"upstream": u, "downstream": LOYALTY, "kind": kind} for u in L5_SOURCES] if relations else []
    return make_run(key["L5"], [get_lineage(conn, "mart_loyalty_members", "upstream")], "answered", claim,
                    facts=[(claim, ["dbt:model.fernwood.mart_loyalty_members"], rels)], tables=[LOYALTY, *L5_SOURCES])


def test_lineage_answer_with_relations_passes(conn, key, vocab):
    assert score(l5_run(conn, key), key["L5"], vocab).correctness.passed


def test_missing_relation(conn, key, vocab):
    reasons = score(l5_run(conn, key, relations=False), key["L5"], vocab).correctness.reasons
    assert reasons == [f"missing_relation: {u} -> {LOYALTY} (direct)" for u in L5_SOURCES]


def test_relation_kind_must_match(conn, key, vocab):
    reasons = score(l5_run(conn, key, kind="indirect"), key["L5"], vocab).correctness.reasons
    assert len(reasons) == 3 and all(r.startswith("missing_relation") for r in reasons)


def test_lineage_slice_in_summary(conn, key, vocab):
    results = [score(l5_run(conn, key), key["L5"], vocab), score(d1_good(conn, key), key["D1"], vocab)]
    assert results[0].slices == ["lineage"] and results[1].slices == []
    assert summarize(results)["slices"]["lineage"] == 1.0


def test_summarize(conn, key, vocab):
    good = d1_good(conn, key)
    bad = make_run(key["D1"], [get_table(conn, "dim_customers")], "answered",
                   "dim_customers is owned by Rahul Menon.",
                   facts=[("Owner: Rahul Menon", [DIM_CONTRACT])],
                   tables=["fernwood_dw.marts.dim_customers"])
    abstain = make_run(key["N1"], [find_table(conn, "Kafka clickstream")], "insufficient_context",
                       "No such table.", confidence="low")
    retried = make_run(key["N1"], [find_table(conn, "Kafka clickstream")], "insufficient_context",
                       "No such table.", confidence="low", guardrails={
                           "attempts": 2, "outcome": "refused",
                           "events": [{"part": "output", "check": "no_evidence", "detail": "x", "attempt": 1},
                                      {"part": "output", "check": "no_evidence", "detail": "x", "attempt": 2}]})
    runs = iter([good, abstain, bad, retried])
    results = run_eval(lambda _q: next(runs), [key["D1"], key["N1"]], conn, runs=2)

    s = summarize(results)
    assert s["question_pass_rate"] == 0.75
    assert s["by_set"] == {"dev": 0.5, "negative": 1.0}
    assert s["slices"]["negative"] == 1.0 and s["slices"]["escalation"] is None
    assert s["derived"]["hallucination_rate"] == 0.5        # 1 of 2 claims unsupported
    assert s["derived"]["agreement_rate"] == 0.5            # D1 differed between runs
    assert s["derived"]["calibration_high"] == 0.5          # 2 high answers, 1 correct
    assert s["reliability"] == {"retry_rate": 0.25, "retry_success_rate": 0.0}
    assert s["outcomes"] == {"answered": 0.5, "abstained": 0.25, "verification_refusal": 0.25,
                             "budget_stop": 0.0, "error": 0.0}
    assert s["refusals"] == {"false_refusal_rate": 0.0, "missed_refusal_rate": 0.0,
                             "abstention_precision": 1.0, "abstention_recall": 1.0}
    assert s["derived"]["confidence_distribution"] == {"high": 0.5, "medium": 0.0, "low": 0.5}
    assert s["derived"]["guardrail_trigger_rate"] == {"output:no_evidence": 0.25}  # counted once per answer
    assert list(s["failures"]) == ["D1"]
