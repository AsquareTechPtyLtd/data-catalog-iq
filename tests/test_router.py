"""Tests for the router: route specs, plan checks, sample agreement and the entry points (no network)."""

import inspect
import threading

import pytest
from pydantic import ValidationError

from catalog_iq.catalog.db import catalog_path, connect
from catalog_iq.core.providers.base import LLMCall, Reply, ToolCallRequest
from catalog_iq.router import (
    RUNNERS, ROUTE_TOOL, UNROUTABLE, UNSURE, check_plan, decide, route, run_routes, system_prompt, user_message,
)
from catalog_iq.router.schema import RoutePlan, Turn, TurnRoute
from catalog_iq.router.specs import AGENTS, load_specs
from catalog_iq.agents.data_category_recommender import recommend_data_category
from catalog_iq.agents.catalog_lookup import ask
from catalog_iq.agents.pi_classifier import recommend_pi

pytestmark = pytest.mark.skipif(not catalog_path().exists(),
                                reason="catalog.db missing: run python3 scripts/ingest_catalog.py")

CONTACTS = "fernwood_dw.raw_crm.contacts"
SESSIONS = "fernwood_dw.marts.mart_web_sessions"
FCT_ORDERS = "fernwood_dw.marts.fct_orders"


class FakeRouter:
    """Returns scripted route plans in order (thread-safe: samples run in parallel)."""

    name, model = "fake", "fake-1"

    def __init__(self, *plans):
        self.plans = list(plans)
        self.sent = []
        self.lock = threading.Lock()

    def chat(self, system, messages, tools):
        with self.lock:
            self.sent.append(list(messages))
            plan = self.plans.pop(0) if self.plans else None
        calls = [] if plan is None else [ToolCallRequest(id="r1", name=ROUTE_TOOL, args=plan)]
        return Reply(text="" if calls else "hmm", tool_calls=calls,
                     usage=LLMCall("fake", "fake-1", input_tokens=500, output_tokens=40, ttft_s=0.1, total_s=0.4))


def item(agent, table=None, request="q", **kw):
    return {"agent": agent, "table": table, "request": request, "reason": "because", **kw}


def plan(*items, clarify=None):
    return {"routes": list(items), "clarify": clarify}


@pytest.fixture(scope="module")
def conn():
    c = connect()
    yield c
    c.close()


@pytest.fixture(scope="module")
def specs():
    return load_specs()


# --- route specs (R8) -----------------------------------------------------------------


def test_every_agent_has_a_valid_spec(specs):
    assert set(specs) == set(AGENTS) == set(RUNNERS)
    entry = {"catalog_lookup": ask, "pi_classifier": recommend_pi, "data_category_recommender": recommend_data_category}
    for agent, spec in specs.items():
        assert spec.examples and spec.handles and spec.label
        first = next(iter(inspect.signature(entry[agent]).parameters))
        assert spec.needs_table == (first == "table"), agent  # `needs` matches the entry point


def test_prompt_is_built_from_the_specs(specs):
    p = system_prompt(specs)
    for spec in specs.values():
        assert f"## {spec.agent} ({spec.label})" in p and spec.examples[0] in p
    assert "{agents}" not in p


def test_history_goes_in_the_user_message():
    h = [Turn(request="What feeds mart_web_sessions?", routes=[TurnRoute(agent="catalog_lookup")],
              summary="int_web_sessions feeds it.")]
    m = user_message("And which data category should it use?", h)
    assert "1. User: What feeds mart_web_sessions?" in m
    assert m.endswith("Request: And which data category should it use?")
    assert user_message("hi", []) == "Request: hi"


# --- plan schema and checks -----------------------------------------------------------


def test_plan_needs_routes_or_clarify():
    with pytest.raises(ValidationError):
        RoutePlan.model_validate(plan())
    with pytest.raises(ValidationError):
        RoutePlan.model_validate(plan(item("catalog_lookup"), clarify="which?"))
    with pytest.raises(ValidationError):
        RoutePlan.model_validate(plan(item("sql_agent")))


def check(conn, specs, p, request="q", history=()):
    return check_plan(RoutePlan.model_validate(p), request, list(history), conn, specs)


def test_recommender_table_is_resolved(conn, specs):
    out, problems = check(conn, specs, plan(item("pi_classifier", "raw_crm.contacts")))
    assert not problems and out.routes[0].table == CONTACTS and out.routes[0].table_as_written == "raw_crm.contacts"
    out, _ = check(conn, specs, plan(item("data_category_recommender", "web sessions mart")))  # name search, one match
    assert out.routes[0].table == SESSIONS


def test_partial_name_matching_several_tables_clarifies(conn, specs):
    out, problems = check(conn, specs, plan(item("pi_classifier", "the customer table")))
    assert not problems and out.clarification and len(out.clarification.options) >= 2
    assert all(o.routes[0].agent == "pi_classifier" and o.routes[0].table for o in out.clarification.options)


def test_recommender_without_a_table_clarifies(conn, specs):
    out, problems = check(conn, specs, plan(item("data_category_recommender")))
    assert not problems and "Which table" in out.clarification.question


def test_unknown_table_is_a_problem(conn, specs):
    _, problems = check(conn, specs, plan(item("pi_classifier", "zzz_quux")))
    assert problems and "not in the catalog" in problems[0]


def test_lone_lookup_keeps_the_question(conn, specs):
    out, _ = check(conn, specs, plan(item("catalog_lookup", request="Owner of fct_orders?")),
                   request="Who owns fct_orders?")
    assert out.routes[0].request == "Who owns fct_orders?" and not out.routes[0].rewritten


def test_rewrite_may_not_invent_a_table(conn, specs):
    h = [Turn(request="What feeds mart_web_sessions?", routes=[TurnRoute(agent="catalog_lookup")])]
    out, problems = check(conn, specs, plan(item("catalog_lookup", request="Who owns mart_web_sessions?")),
                          request="What about its owner?", history=h)
    assert not problems and out.routes[0].rewritten
    _, problems = check(conn, specs, plan(item("catalog_lookup", request="Who owns dim_customers?")),
                        request="What about its owner?", history=h)
    assert problems and "dim_customers" in problems[0]


def test_multi_intent_routes_dedupe_and_cap(conn, specs):
    req = "Who owns fct_orders and does it hold personal data?"
    out, problems = check(conn, specs, plan(item("catalog_lookup", request="Who owns fct_orders?"),
                                            item("pi_classifier", "fct_orders"), item("pi_classifier", FCT_ORDERS)),
                          request=req)
    assert not problems and [(r.agent, r.table) for r in out.routes] == [("catalog_lookup", None),
                                                                          ("pi_classifier", FCT_ORDERS)]
    many = [item("catalog_lookup", request=f"Who owns {t}?") for t in ("fct_orders", "dim_customers",
                                                                        "dim_products", "fct_payments")]
    out, _ = check(conn, specs, plan(*many), request="Who owns fct_orders, dim_customers, dim_products and "
                                                    "fct_payments?")
    assert len(out.routes) == 3 and "first 3" in out.note


# --- agreement (R11) ------------------------------------------------------------------


def outcomes(conn, specs, *plans, request="q"):
    return [check(conn, specs, p, request=request)[0] for p in plans]


PI = plan(item("pi_classifier", "raw_crm.contacts"))
LOOKUP = plan(item("catalog_lookup", request="Is raw_crm.contacts PI?"))
DATA_CATEGORY = plan(item("data_category_recommender", "raw_crm.contacts"))


def test_all_agree_is_high(conn, specs):
    d = decide("q", outcomes(conn, specs, PI, PI, PI))
    assert (d.status, d.confidence, d.agreement, d.alternative) == ("routed", "high", "3/3", None)


def test_majority_is_medium_with_the_alternative_offered(conn, specs):
    d = decide("q", outcomes(conn, specs, PI, LOOKUP, PI))
    assert (d.status, d.confidence, d.agreement) == ("routed", "medium", "2/3")
    assert d.routes[0].agent == "pi_classifier" and d.alternative.routes[0].agent == "catalog_lookup"


def test_no_majority_is_low_and_runs_nothing(conn, specs):
    d = decide("q", outcomes(conn, specs, PI, LOOKUP, DATA_CATEGORY))
    assert (d.status, d.confidence, d.routes) == ("clarify", "low", [])
    assert d.clarification.question == UNSURE and len(d.clarification.options) == 3


def test_majority_clarify_asks(conn, specs):
    ask_ = plan(clarify="Which table?")
    d = decide("q", outcomes(conn, specs, ask_, ask_, PI))
    assert (d.status, d.confidence) == ("clarify", "medium") and d.alternative.routes[0].agent == "pi_classifier"


# --- route() --------------------------------------------------------------------------


def test_route_runs_samples_in_parallel(conn):
    fake = FakeRouter(PI, PI, PI)
    r = route("Does raw_crm.contacts have personal data?", conn=conn, provider=fake)
    assert r.decision.confidence == "high" and r.decision.routes[0].table == CONTACTS
    assert len(fake.sent) == 3 and len(r.llm_calls) == 3


def test_route_retries_once_then_gives_up(conn):
    bad = plan(item("pi_classifier", "zzz_quux"))
    fake = FakeRouter(bad, PI)
    r = route("classify zzz", conn=conn, provider=fake, samples=1)
    assert r.decision.routes[0].table == CONTACTS and r.decision.samples[0].attempts == 2
    assert "rejected" in str(fake.sent[1][-1]["results"][0].result)
    r = route("classify zzz", conn=conn, provider=FakeRouter(bad, bad), samples=1)
    assert r.decision.status == "clarify" and r.decision.clarification.question == UNROUTABLE


def test_route_nudges_when_no_tool_call(conn):
    fake = FakeRouter(None, PI)
    r = route("Does raw_crm.contacts have personal data?", conn=conn, provider=fake, samples=1)
    assert r.decision.status == "routed" and len(r.llm_calls) == 2


def test_run_routes_calls_each_agent_in_order(conn, monkeypatch):
    called = []
    monkeypatch.setitem(RUNNERS, "catalog_lookup", lambda r, c, p: called.append(("lookup", r.request)) or "a")
    monkeypatch.setitem(RUNNERS, "pi_classifier", lambda r, c, p: called.append(("pi", r.table)) or "b")
    d = route("Who owns fct_orders and does it hold personal data?", conn=conn, samples=1,
              provider=FakeRouter(plan(item("catalog_lookup", request="Who owns fct_orders?"),
                                       item("pi_classifier", "fct_orders")))).decision
    assert run_routes(d, conn) == ["a", "b"]
    assert called == [("lookup", "Who owns fct_orders?"), ("pi", FCT_ORDERS)]


def test_cli_route_prints_the_decision(capsys):
    from catalog_iq.cli import main
    fake = FakeRouter(*[plan(item("catalog_lookup", request="Who owns fct_orders?"),
                             item("pi_classifier", "fct_orders"))] * 3)
    assert main(["route", "Who owns fct_orders and does it hold personal data?"], provider=fake) == 0
    out = capsys.readouterr().out
    assert "routing confidence: high (3/3" in out and "→ PI Classifier · fernwood_dw.marts.fct_orders" in out
    assert "asks: Who owns fct_orders?" in out


def test_parallel_samples_share_the_connection_safely(conn):
    """Regression: the samples' table checks used one SQLite connection from several threads at once."""
    multi = plan(item("catalog_lookup", request="Who owns fct_orders?"), item("pi_classifier", "the customer table"))
    for _ in range(20):
        d = route("Who owns fct_orders and classify the customer table", conn=conn, samples=6,
                  provider=FakeRouter(*[multi] * 6)).decision
        assert d.status == "clarify" and d.confidence == "high" and len(d.clarification.options) >= 2
