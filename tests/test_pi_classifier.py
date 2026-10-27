"""Tests for the PI Classifier: get_columns, confidence, guardrails, the loop and the eval scorer."""

import json

import pytest
from pydantic import ValidationError

from catalog_iq.agents.pi_classifier import recommend_pi
from catalog_iq.agents.pi_classifier.guard import PIGuard, confidence
from catalog_iq.agents.pi_classifier.schema import ColumnDraft, PIDraft
from catalog_iq.catalog.db import catalog_path, connect
from catalog_iq.core.providers.base import LLMCall, Reply, ToolCallRequest
from catalog_iq.evals.pi import load_key, parse_truth, score, summarize
from catalog_iq.tools.columns import SENSITIVITY, column_tools, get_columns
from catalog_iq.tools.filter import PI_TYPES

pytestmark = pytest.mark.skipif(not catalog_path().exists(),
                                reason="catalog.db missing: run python3 scripts/ingest_catalog.py")

RETURNS = "fernwood_dw.raw_returns.return_requests"
TICKETS = "fernwood_dw.raw_helpdesk.tickets"


@pytest.fixture(scope="module")
def conn():
    c = connect()
    yield c
    c.close()


def cols(result):
    return {c["column"]: c for c in result["open_columns"]}


def signal_kinds(col):
    return {(s["signal"], tuple(s["suggests"])) for s in col["signals"]}


# --- get_columns ----------------------------------------------------------------------


def test_open_and_reviewed_split(conn):
    r = get_columns(conn, "raw_helpdesk.tickets")
    assert r["status"] == "found" and list(cols(r)) == ["subject", "body"]
    assert {c["column"] for c in r["reviewed_columns"]} >= {"requester_email", "ticket_id"}
    assert cols(r)["subject"]["recorded"]["status"] == "suggested"


def test_profile_and_signals_for_the_traps(conn):
    t = cols(get_columns(conn, TICKETS))
    assert t["subject"]["profile"]["distinct"] == 8
    assert ("few_distinct", ("not_pi",)) in signal_kinds(t["subject"])
    assert ("scanner", ("free_text",)) in signal_kinds(t["subject"])
    assert ("name", ("free_text",)) in signal_kinds(t["body"])
    r = cols(get_columns(conn, RETURNS))
    assert r["contact_email"]["profile"]["pattern_pct"] == {"email": 100.0}
    assert {("name", ("email",)), ("pattern", ("email", "personal_email"))} <= signal_kinds(r["contact_email"])
    assert signal_kinds(r["customer_id"]) == {("identifier", ("not_pi",))}


def test_description_words_are_weak(conn):
    sends = cols(get_columns(conn, "mart_email_engagement", blind=True))["sends"]
    assert [(s["signal"], s["strong"]) for s in sends["signals"]] == [("description", False)]


def test_no_values_in_the_result(conn):
    text = json.dumps(get_columns(conn, RETURNS, blind=True))
    assert "@" not in text  # no email address, only ratios


def test_blind_hides_recorded_pi(conn):
    normal, blind = get_columns(conn, "dim_customers"), get_columns(conn, "dim_customers", blind=True)
    assert normal["open_columns"] == [] and normal["note"]
    assert len(blind["open_columns"]) == len(normal["reviewed_columns"]) and blind["reviewed_columns"] == []
    for c in blind["open_columns"]:
        assert c["recorded"] is None
        assert all("pi_status" not in n for n in c["neighbours"])
        assert not {s["signal"] for s in c["signals"]} & {"scanner", "neighbour"}
    assert "Maya" not in json.dumps(blind) and "confirmed" not in json.dumps(blind)


def test_blind_is_not_an_llm_argument():
    assert column_tools(blind=True)[0]["input_schema"]["properties"].keys() == {"table"}


def test_staging_is_not_applicable(conn):
    r = get_columns(conn, "stg_helpdesk__tickets")
    assert r["status"] == "not_applicable" and r["upstream_tables"] == [TICKETS]


def test_not_found(conn):
    assert get_columns(conn, "no_such_table")["status"] == "not_found"


def test_sensitivity_lookup_matches_catalog(conn):
    assert set(SENSITIVITY) == set(PI_TYPES)
    rows = conn.execute("SELECT DISTINCT pi_type, sensitivity FROM columns WHERE pi_status = 'confirmed' "
                        "AND classification_source = 'collibra'").fetchall()
    # the catalog records sensitivity per column; a confirmed column is never below its category's level
    order = ["Public", "Internal", "Confidential", "Highly Confidential"]
    assert all(order.index(s) >= order.index(SENSITIVITY[c]) for c, s in rows)


# --- confidence (D8) ------------------------------------------------------------------


def sig(kind, suggests, strong=True):
    return {"signal": kind, "suggests": suggests, "strong": strong, "detail": ""}


@pytest.mark.parametrize("rec, cat, signals, expected", [
    ("pi", "email", [sig("name", ["email"]), sig("pattern", ["email", "personal_email"])], "high"),
    ("pi", "free_text", [sig("name", ["free_text"])], "medium"),
    ("pi", "name", [], "low"),
    ("not_pi", None, [], "high"),                                            # nothing points to PI
    ("not_pi", None, [sig("few_distinct", ["not_pi"]), sig("scanner", ["free_text"])], "medium"),  # overturn
    ("not_pi", None, [sig("name", ["salary"])], "low"),
    ("not_pi", None, [sig("pattern", ["name"], strong=False)], "high"),      # weak shapes do not contradict
    ("pi", "email", [sig("name", ["email"]), sig("identifier", ["not_pi"]), sig("x", ["not_pi"])], "low"),
    ("needs_manual_review", None, [sig("name", ["email"])], "low"),
])
def test_confidence(rec, cat, signals, expected):
    assert confidence(rec, cat, signals) == expected


def test_retry_caps_confidence():
    assert confidence("not_pi", None, [], retried=True) == "medium"


# --- guardrails -----------------------------------------------------------------------


def guard(conn, table=RETURNS, blind=False):
    g = PIGuard(conn, agent="pi_classifier", question=f"Recommend PI for {table}", tools=column_tools(blind))
    g.call("get_columns", {"table": table})
    return g


def rec(table, column, recommendation="not_pi", category=None, subject=None, reason="An identifier.", extra=()):
    return ColumnDraft(column=column, recommendation=recommendation, category=category, data_subject=subject,
                       reason=reason, citations=[f"catalog:{table}#{column}", *extra])


def good_returns():
    return PIDraft(table=RETURNS, summary="Two of seven open columns look like customer PI.", recommendations=[
        rec(RETURNS, "return_id"), rec(RETURNS, "order_id"), rec(RETURNS, "customer_id"),
        rec(RETURNS, "contact_email", "pi", "email", "customer", "100% of values match the email pattern."),
        rec(RETURNS, "reason_text", "pi", "free_text", "customer", "Free-text return reason written by customers."),
        rec(RETURNS, "refund_amount_usd", reason="An amount."), rec(RETURNS, "requested_at", reason="A timestamp.")])


def checks(events):
    return sorted({e.check for e in events})


def test_good_draft_passes_and_completes(conn):
    g = guard(conn)
    assert g.check_output(good_returns()) == []
    r = g.complete(good_returns())
    by = {c.column: c for c in r.recommendations}
    assert r.status == "recommended" and r.review_status == "pending_steward_review"
    assert by["contact_email"].sensitivity == "Internal" and by["contact_email"].confidence == "high"
    assert by["reason_text"].sensitivity == "Confidential" and by["reason_text"].confidence == "medium"
    assert by["customer_id"].confidence == "high" and by["customer_id"].recorded_status == "unclassified"
    assert r.counts == {"not_pi": 5, "pi": 2} and r.confidence == "medium"
    assert r.guardrails.outcome == "passed"


def test_incomplete_not_open_duplicate_uncited(conn):
    g = guard(conn)
    d = good_returns()
    d.recommendations = d.recommendations[1:] + [
        rec(RETURNS, "contact_email"), ColumnDraft(column="order_id", recommendation="not_pi", reason="x",
                                                   citations=[f"catalog:{RETURNS}"]), rec(RETURNS, "ticket_id")]
    # ticket_id is not a column of the table, so its id is not a citation either
    assert checks(g.check_output(d)) == ["duplicate", "incomplete", "invalid_citation", "not_open", "uncited_column"]


def test_reviewed_column_is_not_open(conn):
    g = guard(conn, TICKETS)
    d = PIDraft(table=TICKETS, summary="x", recommendations=[
        rec(TICKETS, "subject", reason="8 distinct values: a template."),
        rec(TICKETS, "body", "pi", "free_text", "customer", "Customer messages."),
        rec(TICKETS, "requester_email", "pi", "email", "customer", "An email.")])
    assert checks(g.check_output(d)) == ["not_open"]


def test_numbers_must_be_in_the_cited_column(conn):
    g = guard(conn)
    d = good_returns()
    d.recommendations[3] = rec(RETURNS, "contact_email", "pi", "email", "customer", "97% match the email pattern.")
    events = g.check_output(d)
    assert checks(events) == ["unsupported_claim"] and "'97'" in events[0].detail


def test_made_up_citation(conn):
    g = guard(conn)
    d = good_returns()
    d.recommendations[0] = rec(RETURNS, "return_id", extra=["collibra:made.up"])
    assert checks(g.check_output(d)) == ["invalid_citation"]


def test_wrong_table(conn):
    g = guard(conn)
    d = good_returns()
    d.table = "raw_returns.return_requests"
    assert checks(g.check_output(d)) == ["table_not_returned"]


def test_not_applicable_has_no_recommendations(conn):
    stg = "fernwood_dw.staging.stg_helpdesk__tickets"
    g = guard(conn, stg)
    assert g.check_output(PIDraft(table=stg, summary="PI is inherited from raw_helpdesk.tickets.")) == []
    assert g.complete(PIDraft(table=stg, summary="x")).status == "not_applicable"
    bad = PIDraft(table=stg, summary="x", recommendations=[rec(stg, "body", "pi", "free_text", "customer", "x")])
    assert checks(g.check_output(bad)) == ["not_applicable"]


def test_schema_enforces_category_rules():
    with pytest.raises(ValidationError):
        ColumnDraft(column="x", recommendation="pi", reason="x", citations=["a"])
    with pytest.raises(ValidationError):
        ColumnDraft(column="x", recommendation="not_pi", category="email", reason="x", citations=["a"])
    with pytest.raises(ValidationError):
        ColumnDraft(column="x", recommendation="pi", category="favourite_colour", data_subject="customer",
                    reason="x", citations=["a"])


def test_low_confidence_becomes_manual_review(conn):
    g = guard(conn)
    d = good_returns()
    d.recommendations[2] = rec(RETURNS, "customer_id", "pi", "name", "customer", "Points to a customer.")
    assert g.check_output(d) == []
    c = next(c for c in g.complete(d).recommendations if c.column == "customer_id")
    assert c.recommendation == "needs_manual_review" and c.confidence == "low" and "review manually" in c.note
    assert c.sensitivity is None


def test_refuse_sends_every_open_column_to_review(conn):
    r = guard(conn).refuse()
    assert r.guardrails.outcome == "refused" and r.status == "recommended"
    assert {c.recommendation for c in r.recommendations} == {"needs_manual_review"}
    assert len(r.recommendations) == 7


# --- loop, with a fake provider -------------------------------------------------------


class FakeProvider:
    name, model = "fake", "fake-1"

    def __init__(self, *script):
        self.script = list(script)

    def chat(self, system, messages, tools):
        step = self.script.pop(0)
        calls = [ToolCallRequest(id=f"c{i}", name=n, args=a) for i, (n, a) in enumerate(step)]
        return Reply(text="", tool_calls=calls, usage=LLMCall("fake", "fake-1", 100, 20, 0.1, 0.5))


def test_recommend_pi_end_to_end(conn):
    bad = good_returns().model_dump()
    bad["recommendations"] = bad["recommendations"][:3]
    provider = FakeProvider([("get_columns", {"table": "raw_returns.return_requests"})],
                            [("final_answer", bad)], [("final_answer", good_returns().model_dump())])
    run = recommend_pi("raw_returns.return_requests", conn=conn, provider=provider)
    r = run.response
    assert r.guardrails.outcome == "passed_after_retry" and r.guardrails.events[0].check == "incomplete"
    assert {c.confidence for c in r.recommendations} <= {"medium", "low"}  # retry caps at medium


def test_blind_run_uses_blind_tool(conn):
    provider = FakeProvider([("get_columns", {"table": "dim_customers"})],
                            [("final_answer", {"table": "fernwood_dw.marts.dim_customers", "summary": "x"})],
                            [("final_answer", {"table": "fernwood_dw.marts.dim_customers", "summary": "x"})])
    run = recommend_pi("dim_customers", conn=conn, provider=provider, blind=True)
    assert len(run.tool_results[0]["open_columns"]) == 9
    assert run.response.guardrails.events[0].check == "incomplete"


# --- eval answer key and scorer -------------------------------------------------------


def test_answer_key_matches_catalog(conn):
    keys = load_key()
    assert {k.set for k in keys} == {"open", "dev", "heldout"}
    blind_categories, all_categories = set(), set()
    for k in keys:
        rows = {r["name"]: r for r in conn.execute("SELECT * FROM columns WHERE table_fqn = ?", (k.table,))}
        assert set(k.columns) <= set(rows), k.table
        for col, truth in k.columns.items():
            category, subject = parse_truth(truth)
            r = rows[col]
            if k.set == "open":
                assert r["pi_status"] in ("suggested", "unclassified"), (k.table, col)
            elif r["pi_status"] in ("confirmed", "rejected"):  # no planted mistakes: reviewed = truth
                assert (r["pi_type"], r["data_subject"]) == ((category, subject) if category else (None, None)) \
                    or r["pi_status"] == "rejected" and category is None, (k.table, col)
            if category:
                all_categories.add(category)
                if k.blind:
                    blind_categories.add(category)
        assert set(k.traps) <= set(k.columns)
    assert all_categories == set(PI_TYPES)
    assert set(PI_TYPES) - blind_categories == {"free_text"}  # only on the open tables
    open_tables = {k.table for k in keys if k.set == "open"}
    open_cols = conn.execute("SELECT DISTINCT c.table_fqn FROM columns c JOIN tables t ON t.fqn = c.table_fqn "
                             "WHERE c.pi_status IN ('suggested', 'unclassified') AND t.layer IN ('source', 'mart')")
    assert open_tables == {r[0] for r in open_cols}


def test_scorer(conn):
    key = next(k for k in load_key(("open",)) if k.table == RETURNS)
    g = guard(conn)
    d = good_returns()
    d.recommendations[3] = rec(RETURNS, "contact_email", "pi", "personal_email", "customer", "An email.")
    d.recommendations[4] = rec(RETURNS, "reason_text", reason="Short codes.")  # name says free text: low → review
    d.recommendations[0] = rec(RETURNS, "return_id", "needs_manual_review", reason="Unsure.")
    result = score(g.complete(d), key)
    outcomes = {c.column: (c.outcome, c.category_ok) for c in result.columns}
    assert outcomes["contact_email"] == ("tp", False)
    assert outcomes["reason_text"] == ("flagged_pi", None)
    assert outcomes["customer_id"] == ("tn", None)
    assert outcomes["return_id"] == ("flagged_not_pi", None)
    assert not result.passed
    s = summarize([result])
    assert s["precision"] == 1.0 and s["recall"] == 1.0 and s["strict_recall"] == 0.5
    assert s["category_accuracy"] == 0.0 and s["manual_review_rate"] == round(2 / 7, 3)
    assert s["traps"] == {f"{RETURNS}.contact_email": "0/1", f"{RETURNS}.reason_text": "0/1"}
