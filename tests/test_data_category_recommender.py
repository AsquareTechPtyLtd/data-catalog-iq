"""Tests for the Data Category Recommender: draft schema, confidence, guardrails, completion and the loop."""

import pytest
from pydantic import ValidationError

from catalog_iq.agents.data_category_recommender import AGENT, recommend_data_category, request
from catalog_iq.agents.data_category_recommender.guard import DataCategoryGuard, confidence
from catalog_iq.agents.data_category_recommender.schema import DataCategoryDraft
from catalog_iq.catalog.db import catalog_path, connect
from catalog_iq.core.providers.base import LLMCall, Reply, ToolCallRequest
from catalog_iq.evals.data_category import load_key, score
from catalog_iq.tools.data_category_evidence import data_category_evidence_tools

pytestmark = pytest.mark.skipif(not catalog_path().exists(),
                                reason="catalog.db missing: run python3 scripts/ingest_catalog.py")

RETURNS = "fernwood_dw.raw_returns.return_requests"
PAGE_VIEWS = "fernwood_dw.raw_web.page_views"


@pytest.fixture(scope="module")
def conn():
    c = connect()
    yield c
    c.close()


def guard_for(conn, table, search="return requests refund order", blind=False):
    tools = data_category_evidence_tools(blind)
    g = DataCategoryGuard(conn, agent=AGENT, question=request(table), tools=tools, max_tool_calls=5)
    g.call("get_data_category_evidence", {"table": table})
    if search:
        g.call("search_data_categories", {"query": search})
    return g


def returns_draft(**changes):
    d = {"table": RETURNS, "outcome": "recommended", "data_category_id": "DC-008", "considered": ["DC-001"],
         "reason": "The table records return requests and refunds, which DC-008 Order Transaction covers.",
         "citations": [f"catalog:{RETURNS}", "collibra:DC-008"],
         "summary": "Recommend DC-008 Order Transaction for the returns table."}
    return DataCategoryDraft(**{**d, **changes})


def checks(events):
    return {e.check for e in events}


# --- draft schema ---------------------------------------------------------------------


def test_recommended_needs_a_data_category_id():
    with pytest.raises(ValidationError, match="needs a data_category_id"):
        returns_draft(data_category_id=None)


def test_other_outcomes_have_no_data_category_id():
    with pytest.raises(ValidationError, match="has no data_category_id"):
        returns_draft(outcome="no_fit")
    with pytest.raises(ValidationError):
        returns_draft(data_category_id="Order Transaction")  # not an id
    with pytest.raises(ValidationError, match="considered"):
        returns_draft(considered=["Order Transaction"])


# --- confidence (N3) ------------------------------------------------------------------


def sig(kind, data_category, strong=False):
    suggests = [data_category] if isinstance(data_category, str) else data_category
    return {"signal": kind, "suggests": suggests, "strong": strong, "detail": ""}


@pytest.mark.parametrize("outcome, data_category, signals, retried, level", [
    ("recommended", "DC-005", [sig("definition", "DC-005"), sig("data_category_domain", ["DC-005"])], False, "high"),
    ("recommended", "DC-005", [sig("definition", "DC-005"), sig("data_category_domain", ["DC-005"])], True, "medium"),
    ("recommended", "DC-008", [sig("definition", "DC-008")], False, "medium"),
    ("recommended", "DC-001", [sig("neighbour", "DC-001", True)], False, "medium"),
    ("recommended", "DC-001", [sig("data_category_domain", ["DC-001", "DC-002"])], False, "low"),
    ("recommended", "DC-001", [sig("definition", "DC-001"), sig("neighbour", "DC-008", True)], False, "low"),
    ("recommended", "DC-001", [sig("definition", "DC-001"), sig("neighbour", "DC-001", True),
                                sig("neighbour", "DC-008", True)], False, "medium"),
    ("no_fit", None, [sig("definition", "DC-004")], False, "medium"),
    ("no_fit", None, [sig("neighbour", "DC-008", True)], False, "low"),
    ("needs_manual_review", None, [sig("definition", "DC-008")], False, "low"),
])
def test_confidence(outcome, data_category, signals, retried, level):
    assert confidence(outcome, data_category, signals, retried) == level


# --- output guardrail -----------------------------------------------------------------


def test_good_draft_passes(conn):
    assert guard_for(conn, RETURNS).check_output(returns_draft()) == []


def test_data_category_must_come_from_a_search(conn):
    g = guard_for(conn, RETURNS, search=None)  # DC-008 is in a signal, but no definitions were compared
    assert checks(g.check_output(returns_draft())) == {"no_search", "data_category_not_retrieved"}  # DC-001: no signal


def test_made_up_data_category_is_not_retrieved(conn):
    g = guard_for(conn, RETURNS)
    found = checks(g.check_output(returns_draft(data_category_id="DC-099", considered=[],
                                                citations=[f"catalog:{RETURNS}", "collibra:DC-099"])))
    assert {"data_category_not_retrieved", "invalid_citation"} <= found


def test_table_and_data_category_must_be_cited(conn):
    found = checks(guard_for(conn, RETURNS).check_output(returns_draft(citations=["collibra:DC-001"])))
    assert {"uncited_table", "uncited_data_category"} <= found


def test_unsupported_data_category_in_reason(conn):
    # names DC-003 in the reason but cites nothing that states it
    found = checks(guard_for(conn, RETURNS).check_output(returns_draft(
        reason="Like DC-003 tables, this records refunds; DC-008 Order Transaction covers returns.")))
    assert "unsupported_claim" in found


def test_wrong_table(conn):
    found = checks(guard_for(conn, RETURNS).check_output(returns_draft(table="fernwood_dw.raw_returns.returns")))
    assert found == {"table_not_returned"}


def test_assigned_and_inherited_tables_get_no_recommendation(conn):
    for table in ("raw_crm.contacts", "stg_web__page_views"):
        g = guard_for(conn, table, search=None)
        fqn = g.tool_results[0]["table"]["fqn"]
        draft = returns_draft(table=fqn, citations=[f"catalog:{fqn}"])
        assert checks(g.check_output(draft)) == {"nothing_to_recommend"}
        ok = DataCategoryDraft(table=fqn, outcome="no_recommendation", reason="It already has one.",
                               citations=[f"catalog:{fqn}"], summary="Nothing to recommend.")
        assert g.check_output(ok) == []


def test_open_table_needs_an_outcome(conn):
    draft = DataCategoryDraft(table=RETURNS, outcome="no_recommendation", reason="r", citations=[f"catalog:{RETURNS}"],
                              summary="s")
    assert "outcome_missing" in checks(guard_for(conn, RETURNS).check_output(draft))


# --- completion -----------------------------------------------------------------------


def test_completion_adds_catalog_facts(conn):
    g = guard_for(conn, RETURNS)
    draft = returns_draft()
    assert g.check_output(draft) == []
    r = g.complete(draft)
    assert r.status == "recommended" and r.review_status == "pending_steward_review"
    rec = r.recommendation
    assert (rec.data_category_id, rec.name, rec.status) == ("DC-008", "Order Transaction", "Approved")
    assert rec.retention["months"] == 84 and rec.stewards and rec.flags == []
    assert r.confidence == "high"  # definition rank 1 + finance governance domain + same-domain tables
    assert {c.id for c in r.citations} == {f"catalog:{RETURNS}", "collibra:DC-008",
                                           "collibra:DC-001"}  # the considered data category, cited by code
    assert r.signals and r.guardrails.outcome == "passed"


def test_candidate_data_category_is_flagged(conn):
    g = guard_for(conn, "mart_email_engagement", search="hashed email engagement pseudonymized", blind=True)
    fqn = "fernwood_dw.marts.mart_email_engagement"
    draft = DataCategoryDraft(table=fqn, outcome="recommended", data_category_id="DC-006",
                              reason="Engagement per hashed recipient matches DC-006 Pseudonymized Marketing Data.",
                              citations=[f"catalog:{fqn}", "collibra:DC-006"], summary="Recommend DC-006.")
    assert g.check_output(draft) == []
    rec = g.complete(draft).recommendation
    assert rec.retention == "not defined" and "not Approved" in rec.flags[0]


def test_low_confidence_becomes_manual_review(conn):
    g = guard_for(conn, RETURNS, search="employee employment record")
    draft = returns_draft(data_category_id="DC-012", considered=[], citations=[f"catalog:{RETURNS}", "collibra:DC-012"],
                          reason="DC-012 Employee Employment Record covers it.", summary="Recommend DC-012.")
    assert g.check_output(draft) == []  # verifiable, but no signal supports it
    r = g.complete(draft)
    assert r.confidence == "low" and r.status == "needs_manual_review" and "categorize manually" in r.note
    assert r.recommendation.data_category_id == "DC-012"  # kept so the steward sees what was suggested


def test_no_fit_refers_to_data_strategy(conn):
    g = guard_for(conn, RETURNS)
    draft = DataCategoryDraft(table=RETURNS, outcome="no_fit", considered=["DC-008"],
                              reason="No data category describes return requests.", citations=[f"catalog:{RETURNS}"],
                              summary="No fit.")
    r = g.complete(draft)
    assert r.status == "no_fit" and r.recommendation is None and "Data Strategy" in r.note


def test_refuse(conn):
    r = guard_for(conn, RETURNS).refuse()
    assert r.status == "needs_manual_review" and r.confidence == "low" and r.guardrails.outcome == "refused"
    assert r.citations[0].id == f"catalog:{RETURNS}"


def test_assigned_response_names_the_current_data_category(conn):
    g = guard_for(conn, "raw_crm.contacts", search=None)
    fqn = "fernwood_dw.raw_crm.contacts"
    draft = DataCategoryDraft(table=fqn, outcome="no_recommendation", reason="It already has DC-001.",
                              citations=[f"catalog:{fqn}"], summary="Already assigned.")
    r = g.complete(draft)
    assert r.status == "assigned" and r.current.data_category_id == "DC-001" and r.recommendation is None


# --- the loop, with a scripted provider ----------------------------------------------


class FakeProvider:
    name, model = "fake", "fake-1"

    def __init__(self, *script):
        self.script = list(script)

    def chat(self, system, messages, tools):
        step = self.script.pop(0)
        calls = [ToolCallRequest(id=f"c{i}", name=n, args=a) for i, (n, a) in enumerate(step)]
        return Reply(text="", tool_calls=calls, usage=LLMCall("fake", "fake-1", 100, 20, 0.1, 0.5))


def test_recommend_data_category_end_to_end(conn):
    bad = returns_draft().model_dump()
    bad["citations"] = ["collibra:DC-008"]
    provider = FakeProvider([("get_data_category_evidence", {"table": "raw_returns.return_requests"})],
                            [("search_data_categories", {"query": "return requests refunds orders"})],
                            [("final_answer", bad)], [("final_answer", returns_draft().model_dump())])
    run = recommend_data_category("raw_returns.return_requests", conn=conn, provider=provider)
    r = run.response
    assert r.status == "recommended" and r.recommendation.data_category_id == "DC-008"
    assert r.guardrails.outcome == "passed_after_retry" and r.guardrails.events[0].check == "uncited_table"
    assert r.confidence == "medium"  # retry caps at medium
    key = next(k for k in load_key(("open",)) if k.table == RETURNS)
    s = score(r.model_dump(), run.tool_results, key)
    assert s.outcome == "correct" and s.truth_rank == 1


def test_blind_run_hides_neighbour_data_categories(conn):
    provider = FakeProvider([("get_data_category_evidence", {"table": "raw_crm.contacts"})],
                            [("final_answer", {"table": "fernwood_dw.raw_crm.contacts", "outcome": "needs_manual_review",
                                               "reason": "Unclear.", "citations": ["catalog:fernwood_dw.raw_crm.contacts"],
                                               "summary": "Needs review."})])
    run = recommend_data_category("raw_crm.contacts", conn=conn, provider=provider, blind=True)
    assert run.tool_results[0]["status"] == "found"
    assert all("data_category_id" not in n for n in run.tool_results[0]["neighbours"])
    assert run.response.status == "needs_manual_review" and run.response.confidence == "low"


def test_cli_recommend_data_category(conn, capsys):
    from catalog_iq.cli import main

    provider = FakeProvider([("get_data_category_evidence", {"table": "raw_returns.return_requests"})],
                            [("search_data_categories", {"query": "return requests refunds orders"})],
                            [("final_answer", returns_draft().model_dump())])
    assert main(["recommend-category", "raw_returns.return_requests", "-v"], provider=provider) == 0
    out = capsys.readouterr().out
    assert "Recommended data category (pending steward review): DC-008 Order Transaction (Approved)" in out
    assert "retention: 84 months" in out and "Signals" in out


def test_chosen_data_category_definition_checked_beyond_the_top_3(conn):
    """Scale test finding: in a big catalog distractors fill the catalog-wide top 3, so the guard checks
    the chosen data category's definition itself (here DC-001 matches returns' contact_email but ranks below 3)."""
    g = guard_for(conn, RETURNS, search="customer contact")
    top3 = {b for s in g.tool_results[0]["signals"] if s["signal"] == "definition" for b in s["suggests"]}
    assert "DC-001" not in top3
    draft = returns_draft(data_category_id="DC-001", considered=["DC-008"],
                          citations=[f"catalog:{RETURNS}", "collibra:DC-001"],
                          reason="DC-001 Customer Contact Information covers the contact email.",
                          summary="Recommend DC-001.")
    r = g.complete(draft)
    assert any(s.signal == "definition" and s.suggests == ["DC-001"] and "chosen" in s.detail for s in r.signals)
    assert r.confidence == "medium" and r.status == "recommended"  # a definition match alone
