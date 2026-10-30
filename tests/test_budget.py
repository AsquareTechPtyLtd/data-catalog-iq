"""Run budgets: no LLM call starts once a run's time or token budget is used (no network)."""

import pytest

from catalog_iq.catalog.db import catalog_path, connect
from catalog_iq.core.budget import Budget
from catalog_iq.core.loop import run_agent
from catalog_iq.core.providers.base import LLMCall
from catalog_iq.router import route
from test_api import router_says
from test_loop import FakeProvider, get

CALL = LLMCall("fake", "fake-1", input_tokens=100, output_tokens=20, ttft_s=0.1, total_s=0.5)


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def test_budget_time_and_tokens():
    clock = Clock()
    b = Budget(seconds=10, tokens=200, clock=clock)
    assert b.exceeded() is None
    b.spend(CALL)
    assert b.used_tokens == 120 and b.exceeded() is None
    b.spend(CALL)
    assert "token budget" in b.exceeded()
    b = Budget(seconds=10, tokens=200, clock=clock)
    clock.now = 10
    assert "time budget" in b.exceeded()


@pytest.fixture(scope="module")
def conn():
    if not catalog_path().exists():
        pytest.skip("catalog.db missing: run python3 scripts/ingest_catalog.py")
    c = connect()
    yield c
    c.close()


def budget_event(run):
    return [e for e in run.response.guardrails.events if e.check == "budget"]


def test_agent_refuses_once_tokens_are_used(conn):
    provider = FakeProvider(get(), get(), get())
    run = run_agent("Who owns dim_customers?", conn, provider, "system", "catalog_lookup",
                    budget=Budget(seconds=60, tokens=200))  # each fake call spends 120
    assert len(run.llm_calls) == 2 and len(provider.sent) == 2
    assert run.response.status == "insufficient_context" and run.response.guardrails.outcome == "refused"
    assert "token budget of 200" in budget_event(run)[0].detail
    assert run.budget.stopped_by == budget_event(run)[0].detail


def test_agent_refuses_once_time_is_up(conn):
    clock = Clock()

    class Slow(FakeProvider):
        def chat(self, system, messages, tools):
            clock.now += 40
            return super().chat(system, messages, tools)

    run = run_agent("Who owns dim_customers?", conn, Slow(get(), get(), get()), "system", "catalog_lookup",
                    budget=Budget(seconds=60, tokens=100_000, clock=clock))
    assert len(run.llm_calls) == 2 and "time budget of 60 s" in budget_event(run)[0].detail


def test_router_spends_nothing_once_its_budget_is_used(conn):
    provider = router_says({"agent": "catalog_lookup", "request": "Who owns dim_customers?", "reason": "owner"})
    rr = route("Who owns dim_customers?", conn=conn, provider=provider, budget=Budget(seconds=20, tokens=0))
    assert rr.llm_calls == [] and rr.decision.status == "clarify"
    assert all(p.startswith("budget:") for s in rr.decision.samples for p in s.problems)
    assert rr.budget.stopped_by.startswith("token budget")


def test_budgets_from_env(monkeypatch):
    from catalog_iq.core.budget import agent_budget, router_budget
    for var in ("AGENT_BUDGET_SECONDS", "AGENT_BUDGET_TOKENS", "ROUTER_BUDGET_SECONDS", "ROUTER_BUDGET_TOKENS"):
        monkeypatch.delenv(var, raising=False)
    assert (agent_budget().seconds, agent_budget().tokens) == (60.0, 100_000)
    assert (router_budget().seconds, router_budget().tokens) == (20.0, 30_000)
    monkeypatch.setenv("AGENT_BUDGET_SECONDS", "90")
    monkeypatch.setenv("AGENT_BUDGET_TOKENS", "50000")
    assert (agent_budget().seconds, agent_budget().tokens) == (90.0, 50_000)


@pytest.mark.parametrize("raw, message", [("abc", "must be a whole number"), ("1.5", "must be a whole number"),
                                          ("0", "must be positive"), ("-3", "must be positive")])
def test_bad_settings_fail_clearly(monkeypatch, raw, message):
    from catalog_iq.core.budget import agent_budget
    monkeypatch.setenv("AGENT_BUDGET_TOKENS", raw)
    with pytest.raises(ValueError, match=f"AGENT_BUDGET_TOKENS {message}"):
        agent_budget()


# --- cancellation and what the user is told ---

def test_cancel_stops_every_budget_and_retries():
    import threading

    from catalog_iq.core.budget import BudgetExceeded, cancellable
    from catalog_iq.core.providers.retry import with_retries
    event = threading.Event()
    with cancellable(event):
        b = Budget(seconds=60, tokens=1000)
        assert b.exceeded() is None
        event.set()
        assert b.exceeded() == "client disconnected"
        with pytest.raises(BudgetExceeded, match="client disconnected"):
            with_retries(lambda timeout: "reply", lambda e: None, (1,))
    assert Budget(seconds=60, tokens=1000).exceeded() is None  # outside the request


def test_agent_stops_after_the_client_leaves(conn):
    import threading

    from catalog_iq.core.budget import cancellable
    event = threading.Event()

    class LeavesAfterFirstCall(FakeProvider):
        def chat(self, system, messages, tools):
            event.set()  # the client disconnects while this call runs
            return super().chat(system, messages, tools)

    with cancellable(event):
        run = run_agent("Who owns dim_customers?", conn, LeavesAfterFirstCall(get(), get()), "system", "catalog_lookup")
    assert len(run.llm_calls) == 1 and budget_event(run)[0].detail == "client disconnected"


def test_budget_stop_says_which_limit(conn):
    run = run_agent("Who owns dim_customers?", conn, FakeProvider(get(), get()), "system", "catalog_lookup",
                    budget=Budget(seconds=60, tokens=200))
    assert run.response.answer.startswith(
        "I stopped before finishing: this question reached its limit of 200 tokens. I couldn't verify")
    clock = Clock()
    b = Budget(seconds=60, tokens=1000, clock=clock)
    clock.now = 61
    assert b.user_note() == "I stopped before finishing: this question reached its time limit of 60 seconds."


def test_recommender_budget_stop_says_which_limit(conn, monkeypatch):
    import catalog_iq.core.loop as loop
    from catalog_iq.agents.pi_classifier import recommend_pi
    monkeypatch.setattr(loop, "agent_budget", lambda: Budget(seconds=60, tokens=1))
    columns = [("get_columns", {"table": "raw_crm.contacts"})]
    run = recommend_pi("raw_crm.contacts", conn=conn, provider=FakeProvider(columns, []))
    assert run.response.summary.startswith("I stopped before finishing: this question reached its limit of 1 tokens.")


def test_router_budget_stop_says_so(conn):
    provider = router_says({"agent": "catalog_lookup", "request": "Who owns dim_customers?", "reason": "owner"})
    rr = route("Who owns dim_customers?", conn=conn, provider=provider, budget=Budget(seconds=20, tokens=0))
    assert rr.decision.clarification.question == \
        "I stopped before finishing: this question reached its limit of 0 tokens. Please ask again."
