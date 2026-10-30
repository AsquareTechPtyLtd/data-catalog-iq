"""Redaction vault: names and emails ↔ tokens (no network)."""

import json

import pytest
from fastapi.testclient import TestClient

from catalog_iq.agents.catalog_lookup import ask
from catalog_iq.api import create_app
from catalog_iq.catalog.db import catalog_path, connect, settings
from catalog_iq.core import tracing
from catalog_iq.core.providers import make_provider
from catalog_iq.core.providers.base import Reply, ToolCallRequest
from catalog_iq.core.providers.redacting import RedactingProvider
from catalog_iq.core.redaction import TOKEN, Vault, current_vault, redaction_scope
from test_api import lookup_agent, router_says
from test_loop import GOOD, FakeProvider, final, get

NAMES = ["Maya Okafor", "Tom Brennan", "Anna Maria", "Anna Maria Lopez"]


@pytest.fixture
def vault():
    return Vault(NAMES)


def test_round_trip_nested(vault):
    original = {"owners": [{"name": "Maya Okafor", "email": "maya.okafor@fernwood.example", "role": "Owner"}],
                "note": "Escalate to data-strategy@fernwood.example or Tom Brennan.", "row_count": 42,
                "certified": True, "url": None, "pair": ("Tom Brennan", 1)}
    redacted = vault.tokenize(original)
    text = str(redacted)
    for secret in ("Maya Okafor", "maya.okafor@", "Tom Brennan", "data-strategy@"):
        assert secret not in text
    assert redacted["owners"][0]["role"] == "Owner" and redacted["row_count"] == 42
    assert redacted["certified"] is True and redacted["url"] is None
    assert vault.restore(redacted) == original


def test_same_value_same_token_within_a_vault(vault):
    a, b, c = vault.tokenize(["Maya Okafor", "Maya Okafor", "Tom Brennan"])
    assert a == b != c
    assert TOKEN.fullmatch(a) and a.startswith("PERSON_")


def test_tokens_differ_across_vaults():
    assert Vault(NAMES).tokenize("Maya Okafor") != Vault(NAMES).tokenize("Maya Okafor")


def test_names_match_any_case_and_spacing_and_restore_to_catalog_spelling(vault):
    t = vault.tokenize("what does maya  OKAFOR own?")
    assert t == f"what does {vault.tokenize('Maya Okafor')} own?"
    assert vault.restore(t) == "what does Maya Okafor own?"


def test_whole_words_only(vault):
    for text in ("Mayan Okafor", "Maya Okaforian", "xMaya Okafor"):
        assert vault.tokenize(text) == text


def test_longest_name_wins(vault):
    t = vault.tokenize("Anna Maria Lopez")
    assert t.count("PERSON_") == 1 and vault.restore(t) == "Anna Maria Lopez"


def test_any_email_is_tokenized(vault):
    t = vault.tokenize("mail stranger.person@elsewhere.org")
    assert t.startswith("mail EMAIL_") and vault.restore(t) == "mail stranger.person@elsewhere.org"


def test_unknown_tokens_are_left_alone(vault):
    assert vault.restore("PERSON_00000000 and EMAIL_ffffffff") == "PERSON_00000000 and EMAIL_ffffffff"


def test_text_without_secrets_unchanged(vault):
    text = "mart_finance_daily_revenue has 3 PI columns"
    assert vault.tokenize(text) == text and vault.restore(text) == text


@pytest.mark.skipif(not catalog_path().exists(), reason="catalog.db missing: run python3 scripts/ingest_catalog.py")
def test_every_catalog_person_is_tokenized():
    conn = connect()
    try:
        people = conn.execute("SELECT full_name, email FROM people").fetchall()
        vault = Vault.from_catalog(conn)
    finally:
        conn.close()
    assert people
    for name, email in people:
        assert TOKEN.fullmatch(vault.tokenize(name)) and TOKEN.fullmatch(vault.tokenize(email))


# --- RedactingProvider: the LLM only sees tokens; the loop, tools and guardrails see real values ---

@pytest.fixture(scope="module")
def conn():
    if not catalog_path().exists():
        pytest.skip("catalog.db missing: run python3 scripts/ingest_catalog.py")
    c = connect()
    yield c
    c.close()


@pytest.fixture(autouse=True)
def redaction_on(monkeypatch):
    monkeypatch.delenv("REDACTION", raising=False)
    monkeypatch.delenv("TRACE_REDACTION", raising=False)


class Model:
    """Stands in for the LLM: records what it was sent and, like a real model, answers in the
    tokens it saw (its script is written with real values and tokenized with the request's Vault)."""

    def __init__(self, scripted):
        self.scripted, self.name, self.model = scripted, scripted.name, scripted.model
        self.seen: list[str] = []

    def chat(self, system, messages, tools):
        self.seen.append(json.dumps([system, messages, tools], default=lambda o: getattr(o, "__dict__", str(o))))
        reply, v = self.scripted.chat(system, messages, tools), current_vault()
        return Reply(text=v.tokenize(reply.text), usage=reply.usage,
                     tool_calls=[ToolCallRequest(c.id, c.name, v.tokenize(c.args)) for c in reply.tool_calls])


def leaks_in(conn, text: str) -> list[str]:
    secrets_ = [v for r in conn.execute("SELECT full_name, email FROM people") for v in r]
    return [s for s in [*secrets_, settings(conn)["escalation_email"]] if s in text]


def leaks(conn, *models):
    return leaks_in(conn, "".join(seen for m in models for seen in m.seen))


def lookup(*script):
    return Model(FakeProvider(*script))


def test_answer_comes_back_real_and_nothing_leaks(conn):
    m = lookup(get(), final(GOOD))
    r = ask("Who owns dim_customers? Is maya.okafor@fernwood.example right?", conn=conn,
            provider=RedactingProvider(m))
    assert r.response.guardrails.outcome == "passed"
    assert "Maya Okafor" in r.response.answer and "PERSON_" not in r.response.answer
    assert m.seen and leaks(conn, m) == []


def test_owner_filter_gets_the_real_name(conn):
    m = lookup([("filter_tables", {"owner": "Maya Okafor"})])
    r = ask("What does Maya Okafor own?", conn=conn, provider=RedactingProvider(m))
    assert r.response.evidence[0].tool == "filter_tables"
    assert r.response.evidence[0].result_status not in ("blocked", "error")
    assert leaks(conn, m) == []


def test_router_samples_and_agent_share_one_vault(conn):
    router = Model(router_says({"agent": "catalog_lookup", "request": "Who owns dim_customers? Maya Okafor says she does.",
                                "reason": "an owner lookup"}))
    agent = Model(lookup_agent())
    c = TestClient(create_app(providers={"router": RedactingProvider(router), "catalog_lookup": RedactingProvider(agent)}))
    r = c.post("/ask", json={"request": "Who owns dim_customers? Maya Okafor says she does."})
    assert r.status_code == 200 and "Maya Okafor" in r.json()["answers"][0]["response"]["answer"]
    assert len(router.seen) == 3 and leaks(conn, router, agent) == []
    tokens = {t for m in (router, agent) for seen in m.seen for t in TOKEN.findall(seen) if t.startswith("PERSON_")}
    maya = [t for t in tokens if all(t in seen for seen in router.seen)]
    assert len(maya) == 1 and maya[0] in agent.seen[0]


# --- request scope ---

def test_scope_opens_reuses_and_closes(conn):
    assert current_vault() is None
    with redaction_scope(conn) as outer:
        assert outer is current_vault() is not None
        with redaction_scope(conn) as inner:
            assert inner is outer
    assert current_vault() is None


def test_scope_opens_while_either_switch_is_on(conn, monkeypatch):
    monkeypatch.setenv("REDACTION", "off")
    with redaction_scope(conn) as v:
        assert v is not None  # traces still need it
    monkeypatch.setenv("TRACE_REDACTION", "off")
    with redaction_scope(conn) as v:
        assert v is None


def test_no_scope_no_llm_call():
    fake = FakeProvider("hi")
    with pytest.raises(RuntimeError, match="no redaction_scope"):
        RedactingProvider(fake).chat("system", [{"role": "user", "text": "Maya Okafor"}], [])
    assert fake.sent == []


def test_make_provider_wraps_unless_off(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    cfg = {"provider": "anthropic", "model": "claude-haiku-5-5"}
    assert isinstance(make_provider(cfg), RedactingProvider)
    monkeypatch.setenv("REDACTION", "off")
    assert not isinstance(make_provider(cfg), RedactingProvider)


# --- trace scrubbing ---

@pytest.fixture
def spans():
    pytest.importorskip("opentelemetry.sdk")
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracing.setup_tracing(tracer_provider=provider)
    yield lambda: json.dumps([dict(s.attributes) for s in exporter.get_finished_spans()])
    tracing.shutdown_tracing()


def ask_through_api():
    router = router_says({"agent": "catalog_lookup", "request": "Who owns dim_customers?", "reason": "an owner lookup"})
    c = TestClient(create_app(providers={"router": RedactingProvider(Model(router)),
                                         "catalog_lookup": RedactingProvider(Model(lookup_agent()))}))
    r = c.post("/ask", json={"request": "Who owns dim_customers? Ask maya.okafor@fernwood.example"})
    assert r.status_code == 200 and "Maya Okafor" in r.json()["answers"][0]["response"]["answer"]


def test_traces_hold_tokens_only(conn, spans):
    ask_through_api()
    traced = spans()
    assert "PERSON_" in traced and "EMAIL_" in traced
    assert leaks_in(conn, traced) == []


def test_trace_redaction_off_gives_readable_traces(conn, spans, monkeypatch):
    monkeypatch.setenv("TRACE_REDACTION", "off")
    ask_through_api()
    traced = spans()
    assert "Maya Okafor" in traced and "PERSON_" not in traced


def test_traces_without_a_scope_are_withheld(spans):
    with tracing.span("outside", "CHAIN") as s:
        s.input({"owner": "Maya Okafor"})
    traced = spans()
    assert "Maya Okafor" not in traced and tracing.WITHHELD in traced
