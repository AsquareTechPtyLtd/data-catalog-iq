"""Data Category Recommender output: the draft the LLM writes and the response code completes.

- DataCategoryDraft: for one table, the LLM's outcome (recommended / no_fit /
  needs_manual_review, or no_recommendation when the table already has a data category,
  inherits one, or was not found), the data category id when it recommends one, the data categories
  it weighed, a short reason and the citation ids it rests on.
- DataCategoryResponse: code adds what must not be generated: the data category's name, status,
  retention and stewards (from the catalog), confidence (from the signals,
  N3), the signals themselves, flags (a Candidate data category), citation details,
  tool calls and the guardrails record. A recommendation is pending steward
  review; nothing in the catalog changes.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from catalog_iq.core.confidence import MEANING
from catalog_iq.core.response import Citation, Confidence, Guardrails, TableRef, ToolCall

SCHEMA_VERSION = "1.1"  # 1.1: confidence_reasons

Outcome = Literal["recommended", "no_fit", "needs_manual_review", "no_recommendation"]
DataCategoryStatus = Literal["recommended", "no_fit", "needs_manual_review", "assigned", "not_applicable", "not_found"]
DATA_CATEGORY_ID = r"^DC-\d{3,5}$"  # large catalogs (scale test) have longer ids


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DataCategoryDraft(_Strict):
    """What the LLM returns at the end of the agent loop."""

    table: str = Field(min_length=1, description="FQN of the table, as get_data_category_evidence returned it.")
    outcome: Outcome = Field(description=(
        "recommended: one data category describes the table. no_fit: no data category's definition describes it. "
        "needs_manual_review: two fit equally well, or the evidence is too thin. no_recommendation: the "
        "table already has a data category, inherits one (stg_/int_), or was not found."))
    data_category_id: str | None = Field(default=None, pattern=DATA_CATEGORY_ID,
                               description=("The recommended data category, from a search_data_categories "
                                                 "result; only when recommended."))
    considered: list[str] = Field(default_factory=list,
                                  description="Other data category ids weighed, e.g. the runner-up.")
    reason: str = Field(min_length=1, description="One or two sentences: the evidence that decides it.")
    citations: list[str] = Field(min_length=1, description="Citation ids from the tool results: the table's, the "
                                                           "data category's, and any neighbour or column relied on.")
    summary: str = Field(min_length=1, description="One sentence for the steward.")

    @model_validator(mode="after")
    def _data_category_rules(self):
        if self.outcome == "recommended" and self.data_category_id is None:
            raise ValueError("a recommended outcome needs a data_category_id")
        if self.outcome != "recommended" and self.data_category_id is not None:
            raise ValueError(f"a {self.outcome} outcome has no data_category_id; name candidates in considered")
        if any(not isinstance(b, str) or not b.startswith("DC-") for b in self.considered):
            raise ValueError("considered lists data category ids such as DC-008")
        return self


class Signal(_Strict):
    signal: str
    suggests: list[str]
    strong: bool
    detail: str
    sources: list[str] = Field(default_factory=list)


class DataCategoryRecommendation(_Strict):
    data_category_id: str
    name: str
    status: str = Field(description="Approved / Candidate (from the catalog).")
    domain: str | None = None
    definition: str | None = None
    retention: dict | str = Field(description="months, trigger, disposal_action, legal_basis; or 'not defined'.")
    stewards: dict[str, list[dict]] = Field(default_factory=dict, description="role → people, from the catalog.")
    flags: list[str] = Field(default_factory=list,
                             description="e.g. a Candidate data category would not certify the table.")


class CurrentDataCategory(_Strict):
    data_category_id: str | None
    name: str | None = None
    inherits_from: str | None = None


class DataCategoryResponse(_Strict):
    """Final response: one table's data category recommendation, pending steward review."""

    schema_version: Literal["1.1"] = SCHEMA_VERSION
    agent: str = "data_category_recommender"
    question: str
    status: DataCategoryStatus
    table: TableRef | None = None
    summary: str
    review_status: Literal["pending_steward_review"] = "pending_steward_review"
    recommendation: DataCategoryRecommendation | None = None
    considered: list[str] = Field(default_factory=list)
    reason: str | None = None
    note: str | None = Field(default=None, description="Why code changed the outcome (low confidence), or a referral.")
    current: CurrentDataCategory | None = Field(
        default=None, description="assigned / not_applicable: the data category it has today.")
    confidence: Confidence = Field(description=MEANING)
    confidence_reasons: list[str] = Field(default_factory=list, description="The signals for and against.")
    signals: list[Signal] = Field(default_factory=list, description="Code-computed evidence behind the confidence.")
    citations: list[Citation] = Field(default_factory=list)
    evidence: list[ToolCall] = Field(default_factory=list)
    guardrails: Guardrails
