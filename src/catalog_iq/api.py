"""REST API (R5) and the web page (R6).

    catalog-iq serve                      # http://127.0.0.1:8000
    POST /ask {request, history?}         → {decision, answers}: routed, one answer per route
    POST /agents/catalog_lookup {request} → {decision: null, answers: [one]}: no router
    POST /agents/pi_classifier {table}
    POST /agents/data_category_recommender {table}
    GET  /tables                          table list for autocomplete
    GET  /health                          catalog loaded, providers configured (never key values)
    GET  /                                the web page

Stateless: the client sends up to 3 earlier turns as `history` (R10).
Synchronous (R7): an answer takes ~5–10 s per route. Each request opens its
own read-only catalog connection. API keys come only from .env.
"""

import math
import os
import sqlite3
import threading
from pathlib import Path
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field

from catalog_iq.catalog.db import catalog_path, connect
from catalog_iq.cli import MAX_QUESTION_CHARS, check_question
from catalog_iq.core.config import agent_config, load_env, number_setting
from catalog_iq.core.loop import AgentRun
from catalog_iq.core.providers import Provider, make_provider
from catalog_iq.core.redaction import redaction_scope
from catalog_iq.core.tracing import span
from catalog_iq.ratelimit import BURST, PER_MINUTE, RateLimiter
from catalog_iq.router import route, run_route
from catalog_iq.router.schema import MAX_HISTORY, Route, RouteDecision, Turn
from catalog_iq.router.specs import load_specs

PAGE = Path(__file__).with_name("web") / "index.html"
KEY_VARS = {"anthropic": "ANTHROPIC_API_KEY", "gemini": "GEMINI_API_KEY"}
MAX_IN_FLIGHT = 4  # LLM requests at once per process; ~12 parallel router calls on one key
RETRY_AFTER_S = 5


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AskRequest(_Strict):
    request: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)
    history: list[Turn] = Field(default_factory=list, max_length=MAX_HISTORY)


class LookupRequest(_Strict):
    request: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)


class TableRequest(_Strict):
    table: str = Field(min_length=1, max_length=300)


class Usage(_Strict):
    llm_calls: int
    tokens: int
    seconds: float
    model: str | None
    token_budget: int | None = Field(default=None, description="Tokens this run may spend (input + output).")
    seconds_budget: float | None = Field(default=None, description="Seconds this run may take.")
    stopped_by_budget: bool = Field(default=False, description="The run was cut short by its budget.")


class Answer(_Strict):
    route: Route
    response: dict = Field(description="The agent's own response JSON, unchanged.")
    usage: Usage


class AskResponse(_Strict):
    decision: RouteDecision | None = Field(description="The router's decision; null for /agents/* calls.")
    answers: list[Answer]
    router_usage: Usage | None = None


def _usage(calls, budget=None) -> Usage:
    return Usage(llm_calls=len(calls), tokens=sum(c.input_tokens + c.output_tokens for c in calls),
                 seconds=round(sum(c.total_s for c in calls), 2),
                 model=f"{calls[0].provider}/{calls[0].model}" if calls else None,
                 **({"token_budget": budget.tokens, "seconds_budget": budget.seconds,
                     "stopped_by_budget": budget.stopped_by is not None} if budget else {}))


def create_app(catalog: Path | None = None, providers: dict[str, Provider] | None = None,
               max_in_flight: int | None = None, limiter: RateLimiter | None = None) -> FastAPI:
    """`providers` (agent or "router" → Provider) replaces the configured ones, e.g. fakes in tests."""
    load_env()
    app = FastAPI(title="Data Catalog IQ", version="0.1.0",
                  description="Ask Fernwood Outfitters' synthetic data catalog; every answer is cited.")
    injected = dict(providers or {})
    made: dict[str, Provider] = {}

    def provider_for(agent: str) -> Provider:
        if agent in injected:
            return injected[agent]
        if agent not in made:
            made[agent] = make_provider(agent_config(agent))
        return made[agent]

    limiter = limiter or RateLimiter(number_setting("RATE_LIMIT_PER_MINUTE", PER_MINUTE),
                                     number_setting("RATE_LIMIT_BURST", BURST, int))
    slots = threading.BoundedSemaphore(max_in_flight or number_setting("MAX_IN_FLIGHT", MAX_IN_FLIGHT, int))

    def rate_limit(request: Request):
        """Keyed on the connection's address; X-Forwarded-For is not trusted (anyone can set it)."""
        wait = limiter.take(request.client.host if request.client else "unknown")
        if wait:
            raise HTTPException(429, "Too many questions from this client. Try again shortly.",
                                headers={"Retry-After": str(math.ceil(wait))})

    def llm_slot():
        """A full gate answers 503 at once: waiting would only spend the request's budget."""
        if not slots.acquire(blocking=False):
            raise HTTPException(503, "Busy: too many questions in progress. Try again in a few seconds.",
                                headers={"Retry-After": str(RETRY_AFTER_S)})
        try:
            yield
        finally:
            slots.release()

    def db():
        conn = connect(catalog)
        try:
            yield conn
        finally:
            conn.close()

    def answer(r: Route, conn: sqlite3.Connection) -> Answer:
        try:
            run: AgentRun = run_route(r, conn, provider_for(r.agent))
        except Exception as e:  # missing key, no credit, network: say so plainly, never a key value
            raise HTTPException(503, f"{r.label} failed: {type(e).__name__}") from e
        return Answer(route=r, response=run.response.model_dump(mode="json"), usage=_usage(run.llm_calls, run.budget))

    def direct(agent: str, table: str | None, request: str) -> Route:
        return Route(agent=agent, label=load_specs()[agent].label, table=table, table_as_written=table,
                     request=request, reason="asked directly (no router)")

    @app.get("/", include_in_schema=False)
    def page():
        return FileResponse(PAGE)

    llm = [Depends(rate_limit), Depends(llm_slot)]  # LLM endpoints: rate limit, then the gate

    @app.post("/ask", response_model=AskResponse, dependencies=llm)
    def ask_(body: AskRequest, conn: sqlite3.Connection = Depends(db)):
        error = check_question(body.request)
        if error:
            raise HTTPException(422, error)
        # one Vault and one trace: the router, then each agent it chose
        with redaction_scope(conn), span("ask", "CHAIN") as s:
            s.input({"request": body.request, "history": [t.model_dump() for t in body.history]})
            try:
                rr = route(body.request, history=body.history, conn=conn, provider=provider_for("router"))
            except Exception as e:
                raise HTTPException(503, f"Router failed: {type(e).__name__}") from e
            answers = [answer(r, conn) for r in rr.decision.routes]  # one after another, never merged (R9)
            s.output({"status": rr.decision.status, "routes": [f"{r.agent}·{r.table or '-'}"
                                                               for r in rr.decision.routes]})
            return AskResponse(decision=rr.decision, answers=answers, router_usage=_usage(rr.llm_calls, rr.budget))

    @app.post("/agents/catalog_lookup", response_model=AskResponse, dependencies=llm)
    def lookup(body: LookupRequest, conn: sqlite3.Connection = Depends(db)):
        error = check_question(body.request)
        if error:
            raise HTTPException(422, error)
        return AskResponse(decision=None, answers=[answer(direct("catalog_lookup", None, body.request), conn)])

    def recommender(agent: Literal["pi_classifier", "data_category_recommender"]):
        def run(body: TableRequest, conn: sqlite3.Connection = Depends(db)):
            r = direct(agent, body.table, f"{load_specs()[agent].label} for {body.table}")
            return AskResponse(decision=None, answers=[answer(r, conn)])
        return run

    for agent in ("pi_classifier", "data_category_recommender"):
        app.post(f"/agents/{agent}", response_model=AskResponse, name=agent, dependencies=llm)(recommender(agent))

    @app.get("/tables")
    def tables(conn: sqlite3.Connection = Depends(db)):
        return [dict(r) for r in conn.execute(
            "SELECT fqn, name, layer, domain, data_category_id FROM tables "
            "WHERE layer IN ('source', 'mart') ORDER BY fqn")]

    @app.get("/health")
    def health():
        path = catalog or catalog_path()
        agents = {}
        for agent in ("router", *load_specs()):
            cfg = agent_config(agent)
            agents[agent] = {"provider": cfg["provider"], "model": cfg["model"],
                             "configured": agent in injected or bool(os.environ.get(KEY_VARS.get(cfg["provider"], "")))}
        ok = path.exists() and all(a["configured"] for a in agents.values())
        return {"status": "ok" if ok else "degraded", "catalog": path.exists(), "agents": agents}

    return app
