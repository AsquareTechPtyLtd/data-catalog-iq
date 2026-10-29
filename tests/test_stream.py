"""POST /ask/stream: progress, decision, answers and done as newline-delimited JSON (no network)."""

import json
import threading

import pytest
from fastapi.testclient import TestClient

from catalog_iq.api import create_app
from catalog_iq.catalog.db import catalog_path
from catalog_iq.core.loop import FINAL_ANSWER
from catalog_iq.core.progress import describe_tool, progress, retry_reason
from test_api import Scripted, router_says
from test_loop import BAD, GOOD

pytestmark = pytest.mark.skipif(not catalog_path().exists(),
                                reason="catalog.db missing: run python3 scripts/ingest_catalog.py")

OWNER = {"agent": "catalog_lookup", "request": "Who owns dim_customers?", "reason": "an owner lookup"}
GET = [("get_table", {"table": "dim_customers"})]


def stream(c, request="Who owns dim_customers?"):
    r = c.post("/ask/stream", json={"request": request})
    assert r.status_code == 200 and r.headers["content-type"].startswith("application/x-ndjson")
    return [json.loads(line) for line in r.text.splitlines()]


class Lookup:
    """Stateless, so parallel routes can share it: a table lookup first, then the answer."""

    name, model = "fake", "fake-1"

    def chat(self, system, messages, tools):
        return Scripted(GET if len(messages) == 1 else [(FINAL_ANSWER, GOOD)]).chat(system, messages, tools)


def test_events_arrive_in_order_and_tagged_by_route():
    agent = Lookup()
    c = TestClient(create_app(providers={"router": router_says(OWNER, {**OWNER, "request": "And its steward?"}),
                                         "catalog_lookup": agent}))
    events = stream(c, "Who owns dim_customers and who is its steward?")
    kinds = [e["event"] for e in events]
    assert kinds[0] == "progress" and events[0]["step"] == "routing" and "route_index" not in events[0]
    assert kinds.index("decision") < kinds.index("answer") and kinds[-1] == "done"
    assert sorted(e["route_index"] for e in events if e["event"] == "answer") == [0, 1]  # as each finishes
    assert events[kinds.index("decision")]["router_usage"]["token_budget"] == 30_000
    for i in (0, 1):
        steps = [e["text"] for e in events if e["event"] == "progress" and e.get("route_index") == i]
        assert steps == ["Catalog Lookup started", "Reading governance facts for dim_customers",
                         "Checking the answer against its sources"]
    answer = next(e for e in events if e["event"] == "answer")
    assert "Maya Okafor" in answer["response"]["answer"] and answer["usage"]["token_budget"] == 100_000


def test_a_retry_says_why():
    agent = Scripted(GET, [(FINAL_ANSWER, BAD)], [(FINAL_ANSWER, GOOD)])
    c = TestClient(create_app(providers={"router": router_says(OWNER), "catalog_lookup": agent}))
    retries = [e["text"] for e in stream(c) if e.get("step") == "retry"]
    assert retries == ["Trying again: a claim wasn't in its cited source"]


def test_a_failure_is_an_error_event_and_frees_the_slot():
    class Broken:
        name, model = "fake", "fake-1"

        def chat(self, system, messages, tools):
            raise ConnectionError("down")

    c = TestClient(create_app(providers={"router": Broken()}, max_in_flight=1))
    for _ in range(2):  # the second request gets in: the first one released its slot
        events = stream(c)
        assert events[-1] == {"event": "error", "detail": "Router failed: ConnectionError"}


def test_full_gate_is_a_plain_503():
    started, hold = threading.Event(), threading.Event()

    class Waiting:
        name, model = "fake", "fake-1"

        def chat(self, system, messages, tools):
            started.set()
            hold.wait(5)
            raise ConnectionError("stop")

    c = TestClient(create_app(providers={"router": Waiting()}, max_in_flight=1))
    first = threading.Thread(target=lambda: c.post("/ask/stream", json={"request": "Who owns dim_customers?"}))
    first.start()
    assert started.wait(5)
    busy = c.post("/ask/stream", json={"request": "Who owns dim_customers?"})
    hold.set()
    first.join(5)
    assert busy.status_code == 503 and busy.headers["retry-after"] == "5"


def test_tool_phrases():
    assert describe_tool("get_lineage", {"table": "fct_orders", "direction": "downstream"}) == \
        "Walking the lineage of fct_orders downstream"
    assert describe_tool("filter_tables", {"certification": "certified", "pi_type": "email",
                                           "data_subject": "customer", "with_consumers": True}) == \
        "Filtering tables: certified, email, customer"
    assert describe_tool("find_table", {"query": "loyalty", "kind": "exposure"}) == \
        "Searching dashboards and apps for 'loyalty'"
    assert describe_tool("unknown_tool", {}) == "Calling unknown_tool"
    assert retry_reason(["bad_args", "unsupported_relation"]) == "a lineage link wasn't cited"


def test_progress_without_a_client_does_nothing():
    progress("tool", "nobody is listening")  # no sink attached: no error, no output


def test_disconnect_stops_further_llm_calls():
    """The client reads the first event and leaves while the router's call is running."""
    reading, go = threading.Event(), threading.Event()
    agent_calls = []

    class SlowRouter:
        name, model = "fake", "fake-1"

        def __init__(self):
            self.inner = router_says(OWNER)

        def chat(self, system, messages, tools):
            go.wait(5)
            return self.inner.chat(system, messages, tools)

    class Agent(Scripted):
        def chat(self, system, messages, tools):
            agent_calls.append(1)
            return super().chat(system, messages, tools)

    app = create_app(providers={"router": SlowRouter(), "catalog_lookup": Agent(GET, [(FINAL_ANSWER, GOOD)])},
                     max_in_flight=1)
    c = TestClient(app)
    with c.stream("POST", "/ask/stream", json={"request": "Who owns dim_customers?"}) as r:
        first = json.loads(next(r.iter_lines()))
        assert first["step"] == "routing"
    go.set()
    for t in [t for t in threading.enumerate() if t.name == "ask-stream"]:
        t.join(5)
    assert agent_calls == []


def test_routes_run_in_parallel():
    both_started = threading.Barrier(2, timeout=5)

    class Meet(Lookup):
        def chat(self, system, messages, tools):
            if len(messages) == 1:
                both_started.wait()  # only passes if the two routes are in flight at once
            return super().chat(system, messages, tools)

    c = TestClient(create_app(providers={"router": router_says(OWNER, {**OWNER, "request": "And its steward?"}),
                                         "catalog_lookup": Meet()}))
    events = stream(c, "Who owns dim_customers and who is its steward?")
    assert sorted(e["route_index"] for e in events if e["event"] == "answer") == [0, 1]
    assert events[-1]["event"] == "done"
