"""Agent output schema: the structured output every agent returns.

Two models:
- AgentDraft: the part the LLM writes (status, answer text, cited facts,
  table names, clarification). Its JSON Schema is given to the LLM as the
  structured-output format.
- AgentResponse: the final response. Code completes the draft with anything
  that can be looked up instead of generated (citation details, table links,
  escalation contact, tool-call evidence, confidence), so those fields cannot
  be hallucinated, plus the guardrails record. Other agents, the API and the
  UI consume this model.

Generated JSON Schemas live in schemas/; regenerate with
    python -m catalog_iq.core.response
"""

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

SCHEMA_VERSION = "1.1"  # 1.1: Fact.relations (lineage)
SCHEMAS_DIR = Path(__file__).resolve().parents[3] / "schemas"

Status = Literal["answered", "needs_clarification", "insufficient_context", "out_of_scope"]
Confidence = Literal["high", "medium", "low"]
CitationSource = Literal["contract", "collibra", "dbt", "catalog"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Relation(_Strict):
    """A lineage link stated by a fact, so it can be checked against the tool's edges."""

    upstream: str = Field(min_length=1, description="Table FQN or exposure id the data comes from.")
    downstream: str = Field(min_length=1, description="Table FQN or exposure id the data flows into.")
    kind: Literal["direct", "indirect"] = Field(description=(
        "direct: downstream is loaded straight from upstream (one edge). "
        "indirect: connected through other tables."))


class Fact(_Strict):
    claim: str = Field(min_length=1, description="One factual statement used in the answer.")
    citations: list[str] = Field(
        min_length=1,
        description="Citation ids that support the claim, copied exactly from tool results "
                    "(e.g. 'contract:data/sources/contracts/marts.dim_customers.yaml').",
    )
    relations: list[Relation] = Field(
        default_factory=list,
        description="Lineage links the claim states (one per link: a path or a table built from "
                    "several sources has several). Empty for claims that are not about lineage.",
    )


class Clarification(_Strict):
    question: str = Field(min_length=1, description="Question to ask the user.")
    options: list[str] = Field(min_length=2, description="Candidate table FQNs to choose from.")


def _check_status(status: str, facts: list, clarification) -> None:
    """Rules shared by the draft and the final response."""
    if status == "answered" and not facts:
        raise ValueError("an answered response needs at least one cited fact")
    if status == "needs_clarification" and clarification is None:
        raise ValueError("needs_clarification requires a clarification")
    if status != "needs_clarification" and clarification is not None:
        raise ValueError("clarification is only allowed with status needs_clarification")


class AgentDraft(_Strict):
    """What the LLM returns at the end of the agent loop."""

    status: Status = Field(description=(
        "answered: the question is answered from tool results (including 'no owner is on record'). "
        "needs_clarification: several tables match and the user must choose. "
        "insufficient_context: the catalog has no data to answer. "
        "out_of_scope: not a data catalog question."))
    answer: str = Field(min_length=1, description="Concise plain-English answer for the user.")
    facts: list[Fact] = Field(default_factory=list, description="Every factual claim in the answer, with citations.")
    tables: list[str] = Field(default_factory=list, description="FQNs of the tables the answer is about.")
    clarification: Clarification | None = None

    @model_validator(mode="after")
    def _status_rules(self):
        _check_status(self.status, self.facts, self.clarification)
        return self


class Citation(_Strict):
    id: str
    source: CitationSource
    ref: str
    url: str | None = None


class TableRef(_Strict):
    fqn: str
    name: str
    url: str


class Escalation(_Strict):
    team: str
    email: str
    slack: str


class ToolCall(_Strict):
    tool: str
    args: dict
    result_status: str


class GuardrailEvent(_Strict):
    part: Literal["input", "tool", "output"]
    check: str = Field(description="Check that fired, e.g. bad_args, unsupported_claim.")
    detail: str
    attempt: int | None = Field(default=None, description="Draft attempt, for output checks.")


class Guardrails(_Strict):
    attempts: int = Field(ge=1, description="Drafts the LLM produced (2 = retried once).")
    events: list[GuardrailEvent] = Field(default_factory=list)
    outcome: Literal["passed", "passed_after_retry", "refused"]


class AgentResponse(_Strict):
    """Final response returned to users, the API, the UI and other agents."""

    schema_version: Literal["1.1"] = SCHEMA_VERSION
    agent: str = Field(description="Agent that produced the response, e.g. catalog_lookup.")
    question: str
    status: Status
    answer: str = Field(min_length=1)
    facts: list[Fact] = Field(default_factory=list)
    tables: list[TableRef] = Field(default_factory=list)
    escalation: Escalation | None = Field(
        default=None, description="Who to contact when a table has no data contract or no owner.")
    clarification: Clarification | None = None
    citations: list[Citation] = Field(default_factory=list, description="Details for every cited id.")
    confidence: Confidence
    evidence: list[ToolCall] = Field(default_factory=list, description="Tool calls made, in order.")
    guardrails: Guardrails = Field(description="What the guardrails checked, blocked or retried.")

    @model_validator(mode="after")
    def _consistency(self):
        _check_status(self.status, self.facts, self.clarification)
        ids = [c.id for c in self.citations]
        if len(ids) != len(set(ids)):
            raise ValueError("citation ids must be unique")
        missing = {cid for f in self.facts for cid in f.citations} - set(ids)
        if missing:
            raise ValueError(f"facts cite ids with no citation details: {sorted(missing)}")
        return self


def json_schemas() -> dict[str, dict]:
    return {
        "agent_draft.schema.json": AgentDraft.model_json_schema(),
        "agent_response.schema.json": AgentResponse.model_json_schema(),
    }


def export_schemas(directory: Path = SCHEMAS_DIR) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, schema in json_schemas().items():
        path = directory / name
        path.write_text(json.dumps(schema, indent=2) + "\n")
        paths.append(path)
    return paths


if __name__ == "__main__":
    for p in export_schemas():
        print(f"wrote {p.relative_to(SCHEMAS_DIR.parent)}")
