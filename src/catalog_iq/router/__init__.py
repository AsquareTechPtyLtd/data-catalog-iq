"""Router: picks the agent (and table) for a free-text request, then runs it.

    decision = route("Who owns fct_orders and does it hold personal data?").decision
    runs = run_routes(decision)          # one AgentRun per route, in order

How it decides (agent-6-router.md):
- R1/R8: one LLM call with a single `route` tool; the system prompt is built
  from the agents' route specs.
- Code checks each plan: a recommender's table must resolve in the catalog
  (exact name, else a name search: one match is used, several → clarify with
  the candidates, none → retry once with feedback; a plain-word name such as
  "orders" that another table also matches strongly clarifies too); a catalog_lookup request
  stays the user's question unless it is one part of a multi-part request or a
  follow-up, and a rewrite may name only tables from the request or the
  history; at most MAX_ROUTES routes (R9), duplicates dropped.
- R11: SAMPLES calls run in parallel and are compared on their agent·table
  pairs. All agree → high; a majority → medium, the minority plan offered as an
  alternative; no majority → low, nothing runs and the distinct plans are
  offered as choices.
- R10: the client sends up to MAX_HISTORY earlier turns; the router resolves
  "it" from them. The API keeps no session.
"""

import contextvars
import re
import sqlite3
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import ValidationError

from catalog_iq.agents.data_category_recommender import recommend_data_category
from catalog_iq.agents.catalog_lookup import ask
from catalog_iq.agents.pi_classifier import recommend_pi
from catalog_iq.catalog.db import connect
from catalog_iq.core.config import agent_config, load_env
from catalog_iq.core.budget import Budget, BudgetExceeded, router_budget
from catalog_iq.core.guardrails import MAX_ATTEMPTS
from catalog_iq.core.loop import AgentRun, _chat, inline_refs
from catalog_iq.core.providers import Provider, make_provider
from catalog_iq.core.progress import progress
from catalog_iq.core.providers.base import LLMCall, ToolResult
from catalog_iq.core.redaction import redaction_scope
from catalog_iq.core.tracing import span
from catalog_iq.router.schema import (
    MAX_HISTORY, MAX_ROUTES, Route, RouteClarification, RouteDecision, RouteItem, RouteOption, RoutePlan,
    Sample, Turn,
)
from catalog_iq.router.specs import RouteSpec, load_specs, specs_prompt
from catalog_iq.tools.tables import _resolve, find_table

AGENT = "router"
PROMPT_PATH = Path(__file__).with_name("prompt.md")
ROUTE_TOOL = "route"
SAMPLES = 3
MAX_LLM_CALLS = 3  # per sample: a route, a retry, a nudge
NUDGE = f"Call the {ROUTE_TOOL} tool."
RIVAL_MIN_SCORE = 0.7  # another table matching a plain-word name this well makes it ambiguous
UNROUTABLE = "I couldn't match that to the catalog. Which table do you mean?"
UNSURE = "I'm not sure which you'd like. Did you mean:"

RUNNERS = {
    "catalog_lookup": lambda r, conn, p: ask(r.request, conn=conn, provider=p),
    "pi_classifier": lambda r, conn, p: recommend_pi(r.table, conn=conn, provider=p),
    "data_category_recommender": lambda r, conn, p: recommend_data_category(r.table, conn=conn, provider=p),
}


@dataclass
class RouteRun:
    decision: RouteDecision
    llm_calls: list[LLMCall] = field(default_factory=list)
    budget: Budget | None = None


@dataclass
class _Outcome:
    """One sample after the code checks: routes, or a clarification."""

    routes: list[Route] = field(default_factory=list)
    clarification: RouteClarification | None = None
    note: str | None = None
    attempts: int = 0
    problems: list[str] = field(default_factory=list)

    @property
    def key(self) -> str:
        if self.clarification:
            return "clarify:" + ",".join(sorted(f"{r.agent}·{r.table}" for o in self.clarification.options
                                                for r in o.routes))
        return " + ".join(sorted(f"{r.agent}·{r.table or '-'}" for r in self.routes))


def route_tool() -> dict:
    return {"name": ROUTE_TOOL, "description": "Route the user's request to one or more agents, or ask to clarify.",
            "input_schema": inline_refs(RoutePlan.model_json_schema())}


def system_prompt(specs: dict[str, RouteSpec]) -> str:
    return PROMPT_PATH.read_text().format(agents=specs_prompt(specs))


def user_message(request: str, history: list[Turn]) -> str:
    if not history:
        return f"Request: {request}"
    lines = ["Conversation so far (oldest first):"]
    for i, t in enumerate(history, 1):
        where = ", ".join(f"{r.agent}" + (f" ({r.table})" if r.table else "") for r in t.routes) or "not routed"
        lines += [f"{i}. User: {t.request}", f"   Routed to: {where}"]
        if t.summary:
            lines += [f"   Answer: {t.summary}"]
    return "\n".join(lines + ["", f"Request: {request}"])


# --- code checks ---------------------------------------------------------------------


def _table_names(conn: sqlite3.Connection) -> dict[str, str]:
    """Every way a table can be written (fqn, schema.name, name) → fqn."""
    out = {}
    for r in conn.execute("SELECT fqn, schema_name, name FROM tables"):
        out.update({r["fqn"]: r["fqn"], f"{r['schema_name']}.{r['name']}": r["fqn"], r["name"]: r["fqn"]})
    return out


def named_tables(text: str, names: dict[str, str]) -> set[str]:
    """FQNs of the tables a text names."""
    return {names[w] for w in re.findall(r"[a-z0-9_]+(?:\.[a-z0-9_]+)*", text.lower()) if w in names}


def _resolve_table(conn: sqlite3.Connection, table: str) -> tuple[str | None, list[str]]:
    """(fqn, []) for one match; (None, candidates) for several; (None, []) for none.

    A plain word ("orders") names a table exactly but is also how people say
    "the orders data", so when another table matches it strongly (fct_orders)
    the user is asked.
    """
    row, error = _resolve(conn, table)
    if row:
        if re.fullmatch(r"[a-z]+", table.strip().lower()):
            rivals = [m["fqn"] for m in find_table(conn, table)["matches"]
                      if m["fqn"] != row["fqn"] and m["score"] >= RIVAL_MIN_SCORE]
            if rivals:
                return None, [row["fqn"], *rivals]
        return row["fqn"], []
    if error["status"] == "ambiguous":
        return None, error["candidates"]
    found = find_table(conn, table)
    if found["status"] == "found":
        return found["matches"][0]["fqn"], []
    return None, [m["fqn"] for m in found["matches"]]


def check_plan(plan: RoutePlan, request: str, history: list[Turn], conn: sqlite3.Connection,
               specs: dict[str, RouteSpec]) -> tuple[_Outcome, list[str]]:
    """Resolve and check one plan. Problems mean the LLM should retry."""
    if plan.clarify:
        return _Outcome(clarification=RouteClarification(question=plan.clarify)), []
    names = _table_names(conn)
    context = " ".join([request] + [t.request + " " + (t.summary or "") + " " +
                                    " ".join(r.table or "" for r in t.routes) for t in history])
    allowed = named_tables(context, names)
    single = len(plan.routes) == 1 and not history
    problems, routes = [], []
    for item in plan.routes:
        spec = specs[item.agent]
        r = Route(agent=item.agent, label=spec.label, table_as_written=item.table, request=item.request,
                  rewritten=item.rewritten, reason=item.reason)
        if spec.needs_table:
            if not item.table:
                return _Outcome(clarification=RouteClarification(
                    question=f"Which table should the {spec.label} look at?")), []
            fqn, candidates = _resolve_table(conn, item.table)
            if fqn is None and candidates:
                return _Outcome(clarification=RouteClarification(
                    question=f"Which table did you mean by '{item.table}'?",
                    options=[RouteOption(label=f"{spec.label} · {c}", routes=[r.model_copy(update={"table": c})])
                             for c in candidates])), []
            if fqn is None:
                problems.append(f"table '{item.table}' is not in the catalog: use a table name from the request "
                                "or the conversation, or clarify")
                continue
            routes.append(r.model_copy(update={"table": fqn, "request": request, "rewritten": False}))
            continue
        if single:  # a lone question goes to catalog_lookup exactly as asked
            r = r.model_copy(update={"request": request, "rewritten": False})
        else:
            new = named_tables(r.request, names) - allowed
            if new:
                problems.append(f"the catalog_lookup request names {', '.join(sorted(new))}, which the request "
                                "and conversation do not mention")
                continue
            r = r.model_copy(update={"rewritten": r.request.strip() != request.strip()})
        routes.append(r)

    seen, unique = set(), []
    for r in routes:
        k = (r.agent, r.table, r.request if r.agent == "catalog_lookup" else None)
        if k not in seen:
            seen.add(k)
            unique.append(r)
    note = None
    if len(unique) > MAX_ROUTES:
        note = f"I handled the first {MAX_ROUTES} parts; ask about the rest separately."
        unique = unique[:MAX_ROUTES]
    return _Outcome(routes=unique, note=note), problems


# --- one sample ----------------------------------------------------------------------


def _sample(request: str, history: list[Turn], conn: sqlite3.Connection, provider: Provider,
            specs: dict[str, RouteSpec], system: str, n: int,
            db_lock: threading.Lock, budget: Budget) -> tuple[_Outcome, list[LLMCall]]:
    with span(f"router sample {n}", "CHAIN") as s:
        messages = [{"role": "user", "text": user_message(request, history)}]
        tools = [route_tool()]
        calls, attempts, problems = [], 0, []
        for _ in range(MAX_LLM_CALLS):
            why = budget.exceeded()
            if why:
                problems = [f"budget: {budget.stop(why)}"]
                break
            try:
                reply = _chat(provider, system, messages, tools)
            except BudgetExceeded as e:
                problems = [f"budget: {budget.stop(str(e))}"]
                break
            budget.spend(reply.usage)
            calls.append(reply.usage)
            messages.append({"role": "assistant", "reply": reply})
            call = next((c for c in reply.tool_calls if c.name == ROUTE_TOOL), None)
            if call is None:
                messages.append({"role": "user", "text": NUDGE})
                continue
            attempts += 1
            try:
                plan = RoutePlan.model_validate(call.args)
            except ValidationError as e:
                problems = ["; ".join(f"{'.'.join(map(str, err['loc']))}: {err['msg']}" if err["loc"]
                                      else err["msg"] for err in e.errors())]
            else:
                with db_lock:  # the samples share one SQLite connection; only the LLM calls run in parallel
                    outcome, problems = check_plan(plan, request, history, conn, specs)
                if not problems:
                    outcome.attempts = attempts
                    s.output({"key": outcome.key})
                    return outcome, calls
            if attempts >= MAX_ATTEMPTS:
                break
            feedback = {"status": "rejected", "feedback": "Your route failed verification:\n" +
                        "\n".join(f"- {p}" for p in problems)}
            messages.append({"role": "tool", "results": [ToolResult(call, feedback)]})
        outcome = _Outcome(clarification=RouteClarification(question=UNROUTABLE), attempts=attempts,
                           problems=problems or ["no route call"])
        s.output({"key": outcome.key, "problems": outcome.problems})
        return outcome, calls


# --- agreement (R11) -----------------------------------------------------------------


def decide(request: str, outcomes: list[_Outcome]) -> RouteDecision:
    counts = Counter(o.key for o in outcomes)
    key, votes = counts.most_common(1)[0]
    if votes > 1 and list(counts.values()).count(votes) > 1:
        votes = 1  # a tie between two plans is no majority
    n = len(outcomes)
    samples = [Sample(key=o.key, attempts=o.attempts, problems=o.problems) for o in outcomes]
    agreement = f"{votes}/{n}"

    if votes * 2 <= n and n > 1:  # no majority: offer each distinct plan
        options, seen = [], set()
        for o in outcomes:
            for opt in ([RouteOption(label=_label(o.routes), routes=o.routes)] if o.routes
                        else o.clarification.options):
                k = _label(opt.routes)
                if k not in seen:
                    seen.add(k)
                    options.append(opt)
        return RouteDecision(request=request, status="clarify", confidence="low", agreement=agreement,
                             clarification=RouteClarification(question=UNSURE, options=options),
                             samples=samples)

    chosen = next(o for o in outcomes if o.key == key)
    confidence = "high" if votes == n else "medium"
    alternative = None
    if confidence == "medium":
        other = next((o for o in outcomes if o.key != key and o.routes), None)
        alternative = RouteOption(label=_label(other.routes), routes=other.routes) if other else None
    if chosen.clarification:
        return RouteDecision(request=request, status="clarify", confidence=confidence, agreement=agreement,
                             clarification=chosen.clarification, alternative=alternative, samples=samples)
    return RouteDecision(request=request, status="routed", routes=chosen.routes, confidence=confidence,
                         agreement=agreement, alternative=alternative, note=chosen.note, samples=samples)


def _label(routes: list[Route]) -> str:
    return " + ".join(f"{r.label}" + (f" · {r.table.split('.')[-1]}" if r.table else f": {r.request}")
                      for r in routes)


# --- entry points --------------------------------------------------------------------


def route(request: str, history: list[Turn] | None = None, conn: sqlite3.Connection | None = None,
          provider: Provider | None = None, samples: int = SAMPLES, budget: Budget | None = None) -> RouteRun:
    history = (history or [])[-MAX_HISTORY:]
    budget = budget or router_budget()  # shared by the parallel samples
    own_conn = conn is None
    conn = conn or connect()
    try:
        if provider is None:
            load_env()
            provider = make_provider(agent_config(AGENT))
        specs = load_specs()
        system = system_prompt(specs)
        progress("routing", "Working out which agent should answer")
        with redaction_scope(conn), span(AGENT, "CHAIN", agent=AGENT) as s:
            s.input({"request": request, "history": [t.model_dump() for t in history]})
            db_lock = threading.Lock()
            with budget.active(), ThreadPoolExecutor(max_workers=samples) as pool:  # threads keep the context
                futures = [pool.submit(contextvars.copy_context().run, _sample, request, history, conn,
                                       provider, specs, system, n, db_lock, budget)
                           for n in range(1, samples + 1)]
                results = [f.result() for f in futures]
            decision = decide(request, [o for o, _ in results])
            if budget.stopped_by and decision.status == "clarify" and decision.clarification:
                decision = decision.model_copy(update={"clarification": decision.clarification.model_copy(
                    update={"question": f"{budget.user_note()} Please ask again."})})
            s.output(decision)
            s.set(metadata={"status": decision.status, "confidence": decision.confidence,
                            "agreement": decision.agreement})
        return RouteRun(decision=decision, llm_calls=[c for _, calls in results for c in calls], budget=budget)
    finally:
        if own_conn:
            conn.close()


def run_route(r: Route, conn: sqlite3.Connection | None = None, provider: Provider | None = None) -> AgentRun:
    return RUNNERS[r.agent](r, conn, provider)


def run_routes(decision: RouteDecision, conn: sqlite3.Connection | None = None,
               providers: dict[str, Provider] | None = None) -> list[AgentRun]:
    """Run the decided routes one after another (R9); answers are never merged."""
    providers = providers or {}
    return [run_route(r, conn, providers.get(r.agent)) for r in decision.routes]


def turn_summary(run: AgentRun) -> str:
    """One line of an answer, for the next request's history."""
    r = run.response
    return getattr(r, "answer", None) or getattr(r, "summary", "")
