"""CLI tests with the fake provider (no network, no tracing)."""

import json

import pytest

from catalog_iq import cli
from catalog_iq.catalog.db import catalog_path
from test_loop import BAD, GOOD, FakeProvider, final, get

pytestmark = pytest.mark.skipif(not catalog_path().exists(),
                                reason="catalog.db missing: run python3 scripts/ingest_catalog.py")


@pytest.fixture(autouse=True)
def no_tracing(monkeypatch):
    monkeypatch.setenv("PHOENIX_COLLECTOR_ENDPOINT", "")  # .env must not turn tracing on in tests


def test_text_output(capsys):
    assert cli.main(["ask", "Who owns dim_customers?"], provider=FakeProvider(get(), final(GOOD))) == 0
    out = capsys.readouterr().out
    assert out.startswith("dim_customers is owned by Maya Okafor.")
    assert "status: answered · evidence: high" in out and "why: Answered on the first attempt" in out
    assert "1. Owner of dim_customers: Maya Okafor  [1]" in out
    assert "[1] contract: data/sources/contracts/marts.dim_customers.yaml" in out
    assert "2 LLM calls" in out and "Tool calls" not in out


def test_verbose_shows_guardrails(capsys):
    cli.main(["ask", "Who owns dim_customers?", "-v"], provider=FakeProvider(get(), final(BAD), final(GOOD)))
    out = capsys.readouterr().out
    assert "get_table({'table': 'dim_customers'}) → found" in out
    assert "Guardrails: passed_after_retry, 2 attempt(s)" in out
    assert "output:unsupported_claim" in out


def test_json_output(capsys):
    cli.main(["ask", "Who owns dim_customers?", "--json"], provider=FakeProvider(get(), final(GOOD)))
    assert json.loads(capsys.readouterr().out)["status"] == "answered"


@pytest.mark.parametrize("question", ["   ", "x" * (cli.MAX_QUESTION_CHARS + 1)])
def test_bad_question_rejected_before_llm(capsys, question):
    provider = FakeProvider()
    assert cli.main(["ask", question], provider=provider) == 2
    assert provider.sent == [] and capsys.readouterr().err


def test_interactive(monkeypatch, capsys):
    answers = iter(["Who owns dim_customers?", ""])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    assert cli.main([], provider=FakeProvider(get(), final(GOOD))) == 0
    assert "Maya Okafor" in capsys.readouterr().out
