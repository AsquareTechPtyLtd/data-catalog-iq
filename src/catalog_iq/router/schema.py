"""Router input and output.

- RoutePlan: what the LLM returns through the `route` tool, one per sample.
- RouteDecision: what code returns after checking each sample (tables
  resolved, rewrites checked) and comparing the samples (R11): the routes to
  run, the routing confidence, an alternative (medium) or a clarification
  (low, or a request that cannot be routed safely).
- Turn: one earlier exchange, sent by the client as history (R10).
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from catalog_iq.core.response import Confidence

SCHEMA_VERSION = "1.0"
AgentName = Literal["catalog_lookup", "pi_classifier", "data_category_recommender"]
MAX_ROUTES = 3
MAX_HISTORY = 3


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RouteItem(_Strict):
    agent: AgentName
    table: str | None = Field(default=None, description=(
        "The table the request is about, as the user wrote it, or as resolved from the conversation "
        "('it', 'that table'). Required for pi_classifier and data_category_recommender."))
    request: str = Field(min_length=1, description=(
        "For catalog_lookup: the user's question unchanged; for one part of a multi-part request, "
        "that part as a standalone question; for a follow-up, a standalone rewrite naming the table. "
        "For the other agents: the user's request."))
    rewritten: bool = Field(default=False, description="True when `request` is a follow-up rewrite.")
    reason: str = Field(min_length=1, description="One short sentence: why this agent.")


class RoutePlan(_Strict):
    """The router's answer: 1–3 routes, or a clarifying question."""

    routes: list[RouteItem] = Field(default_factory=list, description=(
        f"One route per intent, at most {MAX_ROUTES}, in the order the user asked."))
    clarify: str | None = Field(default=None, description=(
        "A question for the user instead of routes, when a recommender has no table or the request is "
        "too unclear to route."))

    @model_validator(mode="after")
    def _routes_or_clarify(self):
        if bool(self.routes) == bool(self.clarify):
            raise ValueError("give either routes or clarify, not both and not neither")
        return self


class TurnRoute(_Strict):
    agent: AgentName
    table: str | None = None


class Turn(_Strict):
    """One earlier exchange: what the user asked, where it went, a one-line answer."""

    request: str = Field(min_length=1, max_length=1000)
    routes: list[TurnRoute] = Field(default_factory=list)
    summary: str | None = Field(default=None, max_length=2000)


class Route(_Strict):
    agent: AgentName
    label: str = Field(description="The agent's display name, from its route spec.")
    table: str | None = Field(default=None, description="FQN, resolved by code; None for catalog_lookup.")
    table_as_written: str | None = None
    request: str
    rewritten: bool = False
    reason: str


class RouteOption(_Strict):
    label: str
    routes: list[Route]


class RouteClarification(_Strict):
    question: str
    options: list[RouteOption] = Field(default_factory=list, description="One-click choices; may be empty.")


class Sample(_Strict):
    """One router call's outcome after the code checks, for traces and evals."""

    key: str = Field(description="What was compared: agent·table pairs, or clarify.")
    attempts: int
    problems: list[str] = Field(default_factory=list)


class RouteDecision(_Strict):
    schema_version: Literal["1.0"] = SCHEMA_VERSION
    request: str
    status: Literal["routed", "clarify"]
    routes: list[Route] = Field(default_factory=list)
    confidence: Confidence = Field(description="Agreement of the routing samples (R11): high = all agree, medium = a "
                                               "majority. Not a guarantee the route is right.")
    agreement: str = Field(description="e.g. 3/3")
    alternative: RouteOption | None = Field(default=None, description="medium: the minority plan, offered.")
    clarification: RouteClarification | None = None
    note: str | None = Field(default=None, description="e.g. intents beyond the first three.")
    samples: list[Sample] = Field(default_factory=list)
