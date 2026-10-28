"""Tracing: spans for the agent, LLM calls, tools and guardrails (in-memory exporter, no Phoenix)."""

import json

import pytest

pytest.importorskip("opentelemetry.sdk")

from opentelemetry.sdk.trace import TracerProvider  # noqa: E402
from opentelemetry.sdk.trace.export import SimpleSpanProcessor  # noqa: E402
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter  # noqa: E402

from catalog_iq.catalog.db import catalog_path, connect  # noqa: E402
from catalog_iq.core import tracing  # noqa: E402
from catalog_iq.core.loop import run_agent  # noqa: E402
from test_loop import BAD, GOOD, FakeProvider, final, get  # noqa: E402

pytestmark = pytest.mark.skipif(not catalog_path().exists(),
                                reason="catalog.db missing: run python3 scripts/ingest_catalog.py")


@pytest.fixture
def spans(monkeypatch):
    monkeypatch.setenv("TRACE_REDACTION", "off")  # these tests read real values; test_redaction covers scrubbing
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracing.setup_tracing(tracer_provider=provider)
    yield exporter
    tracing.shutdown_tracing()


def run(*script):
    conn = connect()
    try:
        return run_agent("Who owns dim_customers?", conn, FakeProvider(*script), "system", "catalog_lookup")
    finally:
        conn.close()


def test_one_trace_per_question(spans):
    run(get(), final(BAD), final(GOOD))
    got = spans.get_finished_spans()
    kinds = [(s.name, s.attributes["openinference.span.kind"]) for s in got]
    assert kinds == [("llm", "LLM"), ("tool:get_table", "TOOL"), ("llm", "LLM"), ("guardrail:output", "GUARDRAIL"),
                     ("llm", "LLM"), ("guardrail:output", "GUARDRAIL"), ("catalog_lookup", "AGENT")]
    root = got[-1]
    assert all(s.context.trace_id == root.context.trace_id for s in got)
    assert json.loads(root.attributes["metadata"])["outcome"] == "passed_after_retry"


def test_span_attributes(spans):
    run(get(), final(GOOD))
    by_name = {s.name: s for s in spans.get_finished_spans()}
    llm = by_name["llm"].attributes
    assert (llm["llm.model_name"], llm["llm.token_count.prompt"], llm["llm.token_count.completion"]) == ("fake-1", 100, 20)
    tool = by_name["tool:get_table"].attributes
    assert json.loads(tool["input.value"]) == {"table": "dim_customers"}
    assert json.loads(tool["output.value"])["status"] == "found"
    guard = json.loads(by_name["guardrail:output"].attributes["output.value"])
    assert guard == {"passed": True, "events": []}


def test_blocked_tool_is_traced(spans):
    run([("drop_table", {})], get(), final(GOOD))
    blocked = next(s for s in spans.get_finished_spans() if s.name == "tool:drop_table")
    meta = json.loads(blocked.attributes["metadata"])
    assert meta == {"result_status": "blocked", "guardrail_events": ["tool_unknown"]}


def test_off_by_default():
    tracing.shutdown_tracing()
    with tracing.span("x", "CHAIN") as s:
        s.set(a=1)
        s.output({"b": 2})  # no-op, no error


def test_api_ask_is_one_trace_with_the_router_and_the_agent(spans):
    from test_api import client, lookup_agent, router_says
    c = client(router=router_says({"agent": "catalog_lookup", "request": "Who owns dim_customers?",
                                   "reason": "an owner lookup"}), catalog_lookup=lookup_agent())
    assert c.post("/ask", json={"request": "Who owns dim_customers?"}).status_code == 200
    got = spans.get_finished_spans()
    root = got[-1]
    assert root.name == "ask" and root.parent is None
    assert len({s.context.trace_id for s in got}) == 1
    children = {s.name for s in got if s.parent and s.parent.span_id == root.context.span_id}
    assert children == {"router", "catalog_lookup"}
    assert sum(s.name.startswith("router sample") for s in got) == 3
