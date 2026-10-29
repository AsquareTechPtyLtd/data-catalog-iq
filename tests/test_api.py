"""Tests for the REST API: routing, direct agent calls, tables, health and errors (no network)."""

import shutil
import threading

import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel

from catalog_iq.api import create_app
from catalog_iq.catalog.db import catalog_path
from catalog_iq.core.loop import FINAL_ANSWER, AgentRun
from catalog_iq.core.providers.base import LLMCall, Reply, ToolCallRequest
from catalog_iq.router import RUNNERS, ROUTE_TOOL

pytestmark = pytest.mark.skipif(not catalog_path().exists(),
                                reason="catalog.db missing: run python3 scripts/ingest_catalog.py")

DIM = "fernwood_dw.marts.dim_customers"
DIM_CONTRACT = "contract:data/sources/contracts/marts.dim_customers.yaml"
USAGE = LLMCall("fake", "fake-1", input_tokens=100, output_tokens=20, ttft_s=0.1, total_s=0.5)


class Scripted:
    """Replays tool-call steps in order; thread-safe (the router samples in parallel)."""

    name, model = "fake", "fake-1"

    def __init__(self, *steps, repeat=False):
        self.steps, self.repeat, self.lock = list(steps), repeat, threading.Lock()

    def chat(self, system, messages, tools):
        with self.lock:
            step = self.steps[0] if self.repeat else (self.steps.pop(0) if self.steps else [])
        calls = [ToolCallRequest(id=f"c{i}", name=n, args=a) for i, (n, a) in enumerate(step)]
        return Reply(text="", tool_calls=calls, usage=USAGE)


def router_says(*routes, clarify=None):
    return Scripted([(ROUTE_TOOL, {"routes": list(routes), "clarify": clarify})], repeat=True)


def lookup_agent():
    good = {"status": "answered", "answer": "dim_customers is owned by Maya Okafor.", "tables": [DIM],
            "facts": [{"claim": "Owner of dim_customers: Maya Okafor", "citations": [DIM_CONTRACT]}]}
    return Scripted([("get_table", {"table": "dim_customers"})], [(FINAL_ANSWER, good)])


class FakeResponse(BaseModel):
    agent: str
    summary: str


def client(**providers):
    return TestClient(create_app(providers=providers))


def test_ask_routes_and_runs_the_real_lookup_agent():
    c = client(router=router_says({"agent": "catalog_lookup", "request": "Who owns dim_customers?",
                                   "reason": "an owner lookup"}), catalog_lookup=lookup_agent())
    r = c.post("/ask", json={"request": "Who owns dim_customers?"})
    assert r.status_code == 200
    body = r.json()
    assert body["decision"]["confidence"] == "high" and body["decision"]["agreement"] == "3/3"
    [a] = body["answers"]
    assert a["route"]["agent"] == "catalog_lookup" and a["response"]["status"] == "answered"
    assert "Maya Okafor" in a["response"]["answer"] and a["usage"]["llm_calls"] == 2
    assert body["router_usage"]["llm_calls"] == 3


def test_multi_intent_returns_one_answer_per_route(monkeypatch):
    seen = []

    def fake(agent):
        return lambda r, conn, p: seen.append(agent) or AgentRun(FakeResponse(agent=agent, summary=agent), [], [])

    monkeypatch.setitem(RUNNERS, "catalog_lookup", fake("catalog_lookup"))
    monkeypatch.setitem(RUNNERS, "pi_classifier", fake("pi_classifier"))
    c = client(router=router_says({"agent": "catalog_lookup", "request": "Who owns fct_orders?", "reason": "x"},
                                  {"agent": "pi_classifier", "table": "fct_orders", "request": "q", "reason": "y"}))
    body = c.post("/ask", json={"request": "Who owns fct_orders and does it hold personal data?"}).json()
    assert [a["response"]["agent"] for a in body["answers"]] == ["catalog_lookup", "pi_classifier"]  # route order
    assert sorted(seen) == ["catalog_lookup", "pi_classifier"]  # routes run in parallel, in any order
    assert body["answers"][1]["route"]["table"] == "fernwood_dw.marts.fct_orders"


def test_clarify_runs_nothing(monkeypatch):
    monkeypatch.setitem(RUNNERS, "data_category_recommender", lambda *a: pytest.fail("must not run"))
    c = client(router=router_says({"agent": "data_category_recommender", "request": "Recommend a data category",
                                   "reason": "x"}))
    body = c.post("/ask", json={"request": "Recommend a data category"}).json()
    assert body["decision"]["status"] == "clarify" and body["answers"] == []
    assert "Which table" in body["decision"]["clarification"]["question"]


def test_history_is_validated_and_capped():
    c = client(router=router_says({"agent": "catalog_lookup", "request": "q", "reason": "x"}),
               catalog_lookup=lookup_agent())
    turn = {"request": "What feeds dim_customers?", "routes": [{"agent": "catalog_lookup"}], "summary": "..."}
    assert c.post("/ask", json={"request": "Who owns it?", "history": [turn] * 4}).status_code == 422
    assert c.post("/ask", json={"request": "Who owns it?", "history": [{"request": "x", "secret": 1}]}
                  ).status_code == 422


def test_direct_recommender_skips_the_router(monkeypatch):
    monkeypatch.setitem(RUNNERS, "pi_classifier",
                        lambda r, conn, p: AgentRun(FakeResponse(agent="pi", summary=r.table), [], [USAGE]))
    c = client(router=Scripted())  # would return no route if called
    body = c.post("/agents/pi_classifier", json={"table": "page_views"}).json()
    assert body["decision"] is None and body["answers"][0]["response"]["summary"] == "page_views"
    assert body["answers"][0]["usage"]["tokens"] == 120


def test_input_checks():
    c = client(router=Scripted())
    assert c.post("/ask", json={"request": ""}).status_code == 422
    assert c.post("/ask", json={"request": "   "}).status_code == 422
    assert c.post("/ask", json={"request": "x" * 1001}).status_code == 422
    assert c.post("/agents/catalog_lookup", json={"request": "q", "extra": 1}).status_code == 422


def test_agent_failure_is_a_503_without_details(monkeypatch):
    def boom(r, conn, p):
        raise RuntimeError("ANTHROPIC_API_KEY=sk-ant-secret")

    monkeypatch.setitem(RUNNERS, "data_category_recommender", boom)
    r = client().post("/agents/data_category_recommender", json={"table": "page_views"})
    assert r.status_code == 503 and "secret" not in r.text and "RuntimeError" in r.text


def test_tables_and_health_and_page(monkeypatch, tmp_path):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("CATALOG_MAX_AGE_HOURS", "100000")  # fresh; staleness is tested in test_freshness
    db = tmp_path / "catalog.db"
    shutil.copy(catalog_path(), db)  # away from the real ingest log
    c = TestClient(create_app(catalog=db, providers=dict(
        router=Scripted(), catalog_lookup=Scripted(), pi_classifier=Scripted(), data_category_recommender=Scripted())))
    tables = c.get("/tables").json()
    assert {"fqn": "fernwood_dw.raw_web.page_views", "name": "page_views", "layer": "source",
            "domain": "digital", "data_category_id": None} in tables
    assert not any(t["layer"] in ("staging", "intermediate") for t in tables)
    h = c.get("/health").json()
    assert h["status"] == "ok" and h["catalog"] and set(h["agents"]) == {
        "router", "catalog_lookup", "pi_classifier", "data_category_recommender"}
    assert "sk-" not in str(h)
    assert "Data Catalog IQ" in c.get("/").text


def test_full_gate_answers_503_at_once_and_frees_the_slot():
    release, started = threading.Event(), threading.Event()

    class Blocking(Scripted):
        def chat(self, system, messages, tools):
            started.set()
            release.wait(5)
            return super().chat(system, messages, tools)

    blocking = Blocking([("get_table", {"table": "dim_customers"})], lookup_agent().steps[-1])
    c = TestClient(create_app(providers={"catalog_lookup": blocking}, max_in_flight=1))
    first = {}
    t = threading.Thread(target=lambda: first.update(r=c.post("/agents/catalog_lookup",
                                                              json={"request": "Who owns dim_customers?"})))
    t.start()
    assert started.wait(5)
    busy = c.post("/agents/catalog_lookup", json={"request": "Who owns dim_customers?"})
    assert busy.status_code == 503 and busy.headers["retry-after"] == "5"
    assert c.get("/tables").status_code == 200  # only LLM endpoints are gated
    release.set()
    t.join(5)
    assert first["r"].status_code == 200
    c2 = c.post("/agents/catalog_lookup", json={"request": "Who owns dim_customers?"})
    assert c2.status_code != 503  # the slot was freed


def test_rate_limit_answers_429_with_retry_after():
    from catalog_iq.ratelimit import RateLimiter
    from test_budget import Clock
    c = TestClient(create_app(providers={"catalog_lookup": Scripted([], repeat=True)},
                              limiter=RateLimiter(per_minute=10, burst=2, clock=Clock())))
    codes = [c.post("/agents/catalog_lookup", json={"request": "Who owns dim_customers?"}).status_code
             for _ in range(3)]
    assert codes == [200, 200, 429]
    r = c.post("/agents/catalog_lookup", json={"request": "Who owns dim_customers?"})
    assert r.headers["retry-after"] == "6"
    assert r.json()["detail"] == "You've reached the limit of 10 questions a minute. Try again in 6 seconds."
    assert c.get("/tables").status_code == 200 and c.get("/health").status_code == 200


def test_usage_reports_budgets():
    c = client(router=router_says({"agent": "catalog_lookup", "request": "Who owns dim_customers?",
                                   "reason": "an owner lookup"}), catalog_lookup=lookup_agent())
    body = c.post("/ask", json={"request": "Who owns dim_customers?"}).json()
    u, ru = body["answers"][0]["usage"], body["router_usage"]
    assert (u["token_budget"], u["seconds_budget"], u["stopped_by_budget"]) == (100_000, 60.0, False)
    assert (ru["token_budget"], ru["seconds_budget"], ru["stopped_by_budget"]) == (30_000, 20.0, False)
    assert u["tokens"] == 240 and ru["tokens"] == 360


def test_usage_flags_a_run_stopped_by_its_budget():
    from catalog_iq.api import _usage
    from catalog_iq.core.budget import Budget
    b = Budget(seconds=60, tokens=100)
    b.stop("token budget of 100 used (120)")
    u = _usage([USAGE], b)
    assert u.stopped_by_budget and u.token_budget == 100 and u.tokens == 120
