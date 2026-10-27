"""Tests for the agent output schema."""

import json

import pytest
from pydantic import ValidationError

from catalog_iq.core.response import SCHEMAS_DIR, AgentDraft, AgentResponse, json_schemas

CONTRACT = "contract:data/sources/contracts/marts.dim_customers.yaml"
DATA_STRATEGY = {"team": "Data Strategy", "email": "data-strategy@fernwood.example", "slack": "#data-strategy"}


def answered_draft(**overrides):
    data = {
        "status": "answered",
        "answer": "dim_customers is owned by Maya Okafor.",
        "facts": [{"claim": "Owner: Maya Okafor", "citations": [CONTRACT]}],
        "tables": ["fernwood_dw.marts.dim_customers"],
    }
    return {**data, **overrides}


def response(**overrides):
    data = {
        "agent": "catalog_lookup",
        "question": "Who owns dim_customers?",
        "status": "answered",
        "answer": "dim_customers is owned by Maya Okafor.",
        "facts": [{"claim": "Owner: Maya Okafor", "citations": [CONTRACT]}],
        "tables": [{"fqn": "fernwood_dw.marts.dim_customers", "name": "dim_customers",
                    "url": "https://catalog.fernwood.example/table/fernwood_dw.marts.dim_customers"}],
        "citations": [{"id": CONTRACT, "source": "contract",
                       "ref": "data/sources/contracts/marts.dim_customers.yaml"}],
        "confidence": "high",
        "evidence": [{"tool": "get_table", "args": {"table": "dim_customers"}, "result_status": "found"}],
        "guardrails": {"attempts": 1, "events": [], "outcome": "passed"},
    }
    return {**data, **overrides}


# --- AgentDraft (LLM output) -----------------------------------------------------


def test_valid_answered_draft():
    draft = AgentDraft.model_validate(answered_draft())
    assert draft.facts[0].citations == [CONTRACT]


def test_answered_needs_a_fact():
    with pytest.raises(ValidationError, match="at least one cited fact"):
        AgentDraft.model_validate(answered_draft(facts=[]))


def test_fact_needs_a_citation():
    with pytest.raises(ValidationError):
        AgentDraft.model_validate(answered_draft(facts=[{"claim": "Owner: Maya Okafor", "citations": []}]))


def test_insufficient_context_without_facts():
    draft = AgentDraft.model_validate({"status": "insufficient_context",
                                       "answer": "No table matching 'Kafka clickstream' was found."})
    assert draft.facts == []


def test_needs_clarification_requires_options():
    with pytest.raises(ValidationError, match="requires a clarification"):
        AgentDraft.model_validate({"status": "needs_clarification", "answer": "Which table?"})
    with pytest.raises(ValidationError):
        AgentDraft.model_validate({"status": "needs_clarification", "answer": "Which table?",
                                   "clarification": {"question": "Which one?", "options": ["only_one"]}})


def test_clarification_only_with_needs_clarification():
    clar = {"question": "Which one?", "options": ["a", "b"]}
    with pytest.raises(ValidationError, match="only allowed"):
        AgentDraft.model_validate(answered_draft(clarification=clar))


@pytest.mark.parametrize("bad", [{"status": "maybe"}, {"answer": ""}, {"confidence": "high"}])
def test_draft_rejects_bad_values_and_unknown_fields(bad):
    # confidence is computed by code, so the LLM may not supply it
    with pytest.raises(ValidationError):
        AgentDraft.model_validate(answered_draft(**bad))


def test_draft_parses_from_llm_json_text():
    text = json.dumps(answered_draft())
    assert AgentDraft.model_validate_json(text).status == "answered"


# --- AgentResponse (final) ----------------------------------------------------------


def test_valid_response_defaults_schema_version():
    r = AgentResponse.model_validate(response())
    assert r.schema_version == "1.1"
    assert r.escalation is None


def test_missing_contract_is_answered_with_escalation():
    r = AgentResponse.model_validate(response(
        question="Who owns mart_web_sessions?",
        answer="No data contract exists for mart_web_sessions, so no owner is on record. "
               "Contact the Data Strategy team.",
        facts=[{"claim": "mart_web_sessions has no data contract",
                "citations": ["catalog:fernwood_dw.marts.mart_web_sessions#certification"]}],
        citations=[{"id": "catalog:fernwood_dw.marts.mart_web_sessions#certification", "source": "catalog",
                    "ref": "fernwood_dw.marts.mart_web_sessions#certification"}],
        escalation=DATA_STRATEGY,
    ))
    assert r.status == "answered"
    assert r.escalation.team == "Data Strategy"


def test_fact_citation_must_have_details():
    with pytest.raises(ValidationError, match="no citation details"):
        AgentResponse.model_validate(response(citations=[]))


def test_citation_ids_unique():
    cite = response()["citations"][0]
    with pytest.raises(ValidationError, match="unique"):
        AgentResponse.model_validate(response(citations=[cite, cite]))


def test_guardrails_block_required_and_validated():
    with pytest.raises(ValidationError):
        AgentResponse.model_validate({k: v for k, v in response().items() if k != "guardrails"})
    with pytest.raises(ValidationError):
        AgentResponse.model_validate(response(guardrails={"attempts": 0, "events": [], "outcome": "passed"}))
    r = AgentResponse.model_validate(response(guardrails={
        "attempts": 2, "outcome": "passed_after_retry",
        "events": [{"part": "output", "check": "unsupported_claim", "detail": "fact 1", "attempt": 1}]}))
    assert r.guardrails.events[0].part == "output"


def test_unknown_citation_source_rejected():
    bad = [{"id": CONTRACT, "source": "slack", "ref": "x"}]
    with pytest.raises(ValidationError):
        AgentResponse.model_validate(response(citations=bad))


def test_response_round_trips_as_json():
    r = AgentResponse.model_validate(response())
    assert AgentResponse.model_validate_json(r.model_dump_json()) == r


# --- generated JSON Schemas ---------------------------------------------------------


@pytest.mark.parametrize("name", list(json_schemas()))
def test_committed_json_schema_is_current(name):
    path = SCHEMAS_DIR / name
    assert path.exists(), f"{path} missing: run python -m catalog_iq.core.response"
    assert json.loads(path.read_text()) == json_schemas()[name], \
        f"{name} is stale: run python -m catalog_iq.core.response"


def test_draft_schema_hides_code_filled_fields():
    props = json_schemas()["agent_draft.schema.json"]["properties"]
    assert set(props) == {"status", "answer", "facts", "tables", "clarification"}
