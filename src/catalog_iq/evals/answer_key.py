"""Answer key: the expected result for each eval question.

Loaded from evals/<agent>/questions.yaml. Only the scorer reads it; it is
never given to an agent.
"""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from catalog_iq.catalog.db import ROOT
from catalog_iq.core.response import Relation, Status

EVALS_DIR = ROOT / "evals"

QuestionSet = Literal["dev", "negative", "heldout"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Expected(_Strict):
    status: Status
    tables: list[str] = Field(default_factory=list, description="FQNs the response must include.")
    mentions: list[str | list[str]] = Field(
        default_factory=list,
        description="Each item must appear in the answer or a fact claim; a list means any one of them.")
    escalation: bool = False
    clarification_options: list[str] = Field(default_factory=list)
    relations: list[Relation] = Field(
        default_factory=list, description="Lineage links the facts' `relations` must include (same kind).")
    forbidden_tables: list[str] = Field(
        default_factory=list, description="FQNs the response must not include (traps a filter must exclude).")


class Question(_Strict):
    id: str
    set: QuestionSet
    question: str
    slices: list[str] = Field(default_factory=list, description="Extra groupings for metrics, e.g. lineage.")
    expected: Expected


class AnswerKey(_Strict):
    questions: list[Question]

    @model_validator(mode="after")
    def _unique_ids(self):
        ids = [q.id for q in self.questions]
        if len(ids) != len(set(ids)):
            raise ValueError("question ids must be unique")
        return self


def answer_key_path(agent: str) -> Path:
    return EVALS_DIR / agent / "questions.yaml"


def load_questions(agent: str, sets: tuple[str, ...] | None = None) -> list[Question]:
    key = AnswerKey.model_validate(yaml.safe_load(answer_key_path(agent).read_text()))
    return [q for q in key.questions if sets is None or q.set in sets]
