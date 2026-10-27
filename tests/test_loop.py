"""Tests for the agent loop, with a fake provider that replays scripted replies (no network)."""

import json

import pytest

from catalog_iq.agents.catalog_lookup import ask, system_prompt
from catalog_iq.catalog.db import catalog_path, connect
from catalog_iq.core.loop import FINAL_ANSWER, NUDGE, run_agent, tool_definitions
from catalog_iq.core.providers.base import LLMCall, Reply, ToolCallRequest, ToolResult
from catalog_iq.core.providers.gemini import declarations, to_contents

pytestmark = pytest.mark.skipif(not catalog_path().exists(),
                                reason="catalog.db missing: run python3 scripts/ingest_catalog.py")

DIM = "fernwood_dw.marts.dim_customers"
DIM_CONTRACT = "contract:data/sources/contracts/marts.dim_customers.yaml"
GOOD = {"status": "answered", "answer": "dim_customers is owned by Maya Okafor.", "tables": [DIM],
        "facts": [{"claim": "Owner of dim_customers: Maya Okafor", "citations": [DIM_CONTRACT]}]}
BAD = {**GOOD, "answer": "dim_customers is owned by Rahul Menon.",
       "facts": [{"claim": "Owner: Rahul Menon", "citations": [DIM_CONTRACT]}]}


class FakeProvider:
    """Returns scripted replies in order and records what it was sent."""

    name, model = "fake", "fake-1"

    def __init__(self, *script):
        self.script = list(script)
        self.sent: list[list[dict]] = []

    def chat(self, system, messages, tools):
        self.sent.append(list(messages))
        step = self.script.pop(0) if self.script else "I am not sure."
        calls = [ToolCallRequest(id=f"c{i}", name=n, args=a) for i, (n, a) in enumerate(step)] \
            if isinstance(step, list) else []
        return Reply(text=step if isinstance(step, str) else "", tool_calls=calls,
                     usage=LLMCall("fake", "fake-1", input_tokens=100, output_tokens=20, ttft_s=0.1, total_s=0.5))


def get(table="dim_customers"):
    return [("get_table", {"table": table})]


def final(draft):
    return [(FINAL_ANSWER, draft)]


@pytest.fixture(scope="module")
def conn():
    c = connect()
    yield c
    c.close()


def run(conn, *script, **kw):
    provider = FakeProvider(*script)
    return run_agent("Who owns dim_customers?", conn, provider, "system", "catalog_lookup", **kw), provider


def last_tool_results(provider):
    return [m for m in provider.sent[-1] if m["role"] == "tool"][-1]["results"]


def test_answer_on_first_try(conn):
    r, _ = run(conn, get(), final(GOOD))
    assert r.response.status == "answered" and r.response.confidence == "high"
    assert r.response.guardrails.outcome == "passed"
    assert [e.tool for e in r.response.evidence] == ["get_table"]
    assert len(r.tool_results) == 1 and len(r.llm_calls) == 2


def test_retry_gets_feedback_then_passes(conn):
    r, provider = run(conn, get(), final(BAD), final(GOOD))
    feedback = last_tool_results(provider)[0].result
    assert feedback["status"] == "rejected" and "Rahul Menon" in feedback["feedback"]
    assert (r.response.guardrails.outcome, r.response.confidence) == ("passed_after_retry", "medium")


def test_refuses_after_failed_retry(conn):
    r, _ = run(conn, get(), final(BAD), final(BAD), final(GOOD))
    assert r.response.status == "insufficient_context" and r.response.guardrails.outcome == "refused"
    assert len(r.llm_calls) == 3  # stops at the second failed draft


def test_invalid_draft_counts_as_attempt(conn):
    no_facts = {"status": "answered", "answer": "Maya Okafor."}
    r, _ = run(conn, get(), final(no_facts), final(GOOD))
    assert r.response.guardrails.outcome == "passed_after_retry"
    assert r.response.guardrails.events[0].check == "invalid_draft"


def test_blocked_tool_error_goes_back_to_llm(conn):
    r, provider = run(conn, [("drop_table", {"table": "dim_customers"})], get(), final(GOOD))
    assert provider.sent[1][-1]["results"][0].result["status"] == "error"
    assert r.response.guardrails.outcome == "passed"
    assert [e.check for e in r.response.guardrails.events] == ["tool_unknown"]


def test_text_reply_is_nudged(conn):
    r, provider = run(conn, "Let me think.", get(), final(GOOD))
    assert provider.sent[1][-1] == {"role": "user", "text": NUDGE}
    assert r.response.status == "answered"


def test_out_of_turns_refuses(conn):
    r, _ = run(conn, *[get()] * 3, max_turns=3)
    assert r.response.guardrails.outcome == "refused"
    assert r.response.guardrails.events[-1].check == "max_turns"


def test_out_of_scope_without_tools(conn):
    draft = {"status": "out_of_scope", "answer": "I can only answer questions about the data catalog."}
    r, _ = run(conn, final(draft))
    assert r.response.status == "out_of_scope" and r.response.evidence == []


def test_ask_uses_given_provider(conn):
    r = ask("Who owns dim_customers?", conn=conn, provider=FakeProvider(get(), final(GOOD)))
    assert r.response.agent == "catalog_lookup" and r.response.question == "Who owns dim_customers?"


def test_system_prompt_filled_from_settings(conn):
    prompt = system_prompt(conn)
    assert "data-strategy@fernwood.example" in prompt and "{" not in prompt


def test_final_answer_schema_has_no_refs():
    final_tool = tool_definitions()[-1]
    assert final_tool["name"] == FINAL_ANSWER
    assert "$ref" not in json.dumps(final_tool["input_schema"])


def test_gemini_conversion():
    decls = declarations(tool_definitions())
    assert [d.name for d in decls] == ["find_table", "get_table", FINAL_ANSWER]
    call = ToolCallRequest(id="c0", name="get_table", args={"table": "dim_customers"})
    reply = Reply(text="", tool_calls=[call], usage=None)
    contents = to_contents([{"role": "user", "text": "q"}, {"role": "assistant", "reply": reply},
                            {"role": "tool", "results": [ToolResult(call, {"status": "found"})]}])
    assert [c.role for c in contents] == ["user", "model", "user"]
    assert contents[1].parts[0].function_call.name == "get_table"
    assert contents[2].parts[0].function_response.response == {"status": "found"}


@pytest.mark.parametrize("message, seconds", [
    ("Quota exceeded ... Please retry in 4h42m37.448711179s.", 4 * 3600 + 42 * 60 + 37.448711179),
    ("Please retry in 12.5s.", 12.5),
    ("Resource exhausted.", None),
])
def test_retry_after_parsed(message, seconds):
    from catalog_iq.core.providers.gemini import retry_after_s
    assert retry_after_s(message) == pytest.approx(seconds) if seconds else retry_after_s(message) is None


def test_anthropic_conversion():
    from catalog_iq.core.providers.anthropic import to_messages
    ok = ToolCallRequest(id="t1", name="get_table", args={"table": "dim_customers"})
    bad = ToolCallRequest(id="t2", name="drop_table", args={})
    reply = Reply(text="Looking.", tool_calls=[ok, bad], usage=None)
    out = to_messages([{"role": "user", "text": "q"}, {"role": "assistant", "reply": reply},
                       {"role": "tool", "results": [ToolResult(ok, {"status": "found"}),
                                                     ToolResult(bad, {"status": "error", "error": "x"})]}])
    assert [m["role"] for m in out] == ["user", "assistant", "user"]
    assert out[1]["content"][1] == {"type": "tool_use", "id": "t1", "name": "get_table",
                                    "input": {"table": "dim_customers"}}
    results = out[2]["content"]
    assert results[0]["tool_use_id"] == "t1" and json.loads(results[0]["content"]) == {"status": "found"}
    assert results[1]["is_error"] is True and "is_error" not in results[0]


def test_invalid_draft_feedback_names_the_field(conn):
    empty_cite = {"status": "answered", "answer": "Maya Okafor.", "facts": [{"claim": "Maya Okafor", "citations": []}]}
    r, _ = run(conn, get(), final(empty_cite), final(GOOD))
    event = r.response.guardrails.events[0]
    assert event.check == "invalid_draft" and event.detail.startswith("facts.0.citations:")
