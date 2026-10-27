"""PI Classifier output: the draft the LLM writes and the response code completes.

- PIDraft: per open column, the LLM's recommendation (pi / not_pi /
  needs_manual_review), pi_type and data subject, a short reason and the
  citation ids it rests on.
- PIResponse: code adds what must not be generated: sensitivity (looked up
  from the pi_type, data-model §3.1), confidence (from the signals, D8), the
  signals themselves, the recorded status, citation details, tool calls and
  the guardrails record. Every item is a recommendation pending steward
  review; nothing in the catalog changes.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from catalog_iq.core.response import Citation, Confidence, Guardrails, TableRef, ToolCall
from catalog_iq.tools.filter import DATA_SUBJECTS, PI_TYPES

SCHEMA_VERSION = "1.0"

Recommendation = Literal["pi", "not_pi", "needs_manual_review"]
PIType = Literal[tuple(PI_TYPES)]
Subject = Literal[tuple(DATA_SUBJECTS)]
PIStatus = Literal["recommended", "no_open_columns", "not_applicable", "not_found"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ColumnDraft(_Strict):
    column: str = Field(min_length=1, description="Column name exactly as in open_columns.")
    recommendation: Recommendation = Field(description=(
        "pi: holds personal information. not_pi: does not. needs_manual_review: the evidence is too "
        "weak or conflicting to recommend either."))
    pi_type: PIType | None = Field(default=None, description="PI type; required for pi, empty for not_pi.")
    data_subject: Subject | None = Field(default=None, description="Whose data; required for pi, empty for not_pi.")
    reason: str = Field(min_length=1, description="One sentence: the evidence that decides it.")
    citations: list[str] = Field(min_length=1, description="Citation ids from the tool result, starting with "
                                                           "the column's own id.")

    @model_validator(mode="after")
    def _pi_type_rules(self):
        if self.recommendation == "pi" and (self.pi_type is None or self.data_subject is None):
            raise ValueError("a pi recommendation needs a pi_type and a data_subject")
        if self.recommendation == "not_pi" and (self.pi_type or self.data_subject):
            raise ValueError("a not_pi recommendation has no pi_type or data_subject")
        return self


class PIDraft(_Strict):
    """What the LLM returns at the end of the agent loop."""

    table: str = Field(min_length=1, description="FQN of the table, as get_columns returned it.")
    summary: str = Field(min_length=1, description="One or two sentences for the steward.")
    recommendations: list[ColumnDraft] = Field(
        default_factory=list, description="Exactly one per open column; none for reviewed columns.")


class Signal(_Strict):
    signal: str
    suggests: list[str]
    strong: bool
    detail: str


class ColumnRecommendation(_Strict):
    column: str
    recommendation: Recommendation
    pi_type: PIType | None = None
    data_subject: Subject | None = None
    sensitivity: str | None = Field(default=None, description="Looked up from the pi_type (data-model §3.1).")
    confidence: Confidence
    recorded_status: str | None = Field(default=None, description="pi_status in the catalog today.")
    reason: str
    citations: list[str]
    signals: list[Signal] = Field(default_factory=list, description="Code-computed evidence behind the confidence.")
    note: str | None = None


class PIResponse(_Strict):
    """Final response: recommendations pending steward review."""

    schema_version: Literal["1.0"] = SCHEMA_VERSION
    agent: str = "pi_classifier"
    question: str
    status: PIStatus
    table: TableRef | None = None
    summary: str
    review_status: Literal["pending_steward_review"] = "pending_steward_review"
    recommendations: list[ColumnRecommendation] = Field(default_factory=list)
    counts: dict[str, int] = Field(default_factory=dict)
    citations: list[Citation] = Field(default_factory=list)
    confidence: Confidence = Field(description="Lowest confidence among the recommendations.")
    evidence: list[ToolCall] = Field(default_factory=list)
    guardrails: Guardrails
