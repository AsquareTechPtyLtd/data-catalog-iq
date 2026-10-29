"""The agent loop: LLM ↔ tools until a final answer passes the guardrails.

    for each turn (max MAX_TURNS LLM calls):
        reply = provider.chat(...)
        tool calls      → guard.call() → results back to the LLM
        final_answer    → AgentDraft → guard.check_output()
                          passed            → guard.complete()   (done)
                          failed, can retry → feedback as the final_answer result
                          failed, no retry  → guard.refuse()     (done)
    out of turns, or the run's time/token budget used → guard.refuse()

The LLM finishes by calling the `final_answer` tool, whose input schema is the
guard's draft model (AgentDraft for Catalog Lookup): the same mechanism on
every provider, and its arguments are validated like any other tool call.
"""

import copy
import sqlite3
from dataclasses import dataclass, field

from pydantic import BaseModel, ValidationError

from catalog_iq.core.budget import Budget, BudgetExceeded, agent_budget
from catalog_iq.core.guardrails import MAX_ATTEMPTS, Guard
from catalog_iq.core.progress import describe_tool, progress, retry_reason
from catalog_iq.core.providers.base import LLMCall, Provider, ToolResult
from catalog_iq.core.response import AgentDraft, AgentResponse, GuardrailEvent
from catalog_iq.core.tracing import llm_messages, span
from catalog_iq.tools.tables import TOOLS

MAX_TURNS = 8
FINAL_ANSWER = "final_answer"
NUDGE = f"Finish by calling the {FINAL_ANSWER} tool."


@dataclass
class AgentRun:
    """One answered question: the response, what the tools returned and what the LLM calls cost."""

    response: AgentResponse
    tool_results: list[dict]
    llm_calls: list[LLMCall] = field(default_factory=list)
    budget: Budget | None = None


def inline_refs(schema: dict) -> dict:
    """Replace $ref with the referenced $defs entry; some providers reject $ref."""
    defs = schema.get("$defs", {})

    def resolve(node):
        if isinstance(node, dict):
            if "$ref" in node:
                return resolve(copy.deepcopy(defs[node["$ref"].split("/")[-1]]))
            return {k: resolve(v) for k, v in node.items() if k != "$defs"}
        if isinstance(node, list):
            return [resolve(v) for v in node]
        return node

    return resolve(schema)


def final_answer_tool(draft_model: type[BaseModel] = AgentDraft) -> dict:
    return {
        "name": FINAL_ANSWER,
        "description": ("Give your final answer. Call it once per attempt, after the tools have returned "
                        "what you need (or straight away for out_of_scope questions)."),
        "input_schema": inline_refs(draft_model.model_json_schema()),
    }


FINAL_ANSWER_TOOL = final_answer_tool()


def tool_definitions(tools: list[dict] = TOOLS, draft_model: type[BaseModel] = AgentDraft) -> list[dict]:
    """Neutral definitions sent to the LLM: the catalog tools plus final_answer."""
    return ([{k: t[k] for k in ("name", "description", "input_schema")} for t in tools]
            + [final_answer_tool(draft_model)])


def run_agent(question: str, conn: sqlite3.Connection, provider: Provider, system: str, agent: str,
              max_turns: int = MAX_TURNS, guard: Guard | None = None, tools: list[dict] = TOOLS,
              budget: Budget | None = None) -> AgentRun:
    with span(agent, "AGENT", agent=agent, llm__provider=provider.name, llm__model_name=provider.model) as s:
        s.input({"question": question})
        budget = budget or agent_budget()
        with budget.active():
            run = _run(question, conn, provider, system, agent, max_turns, guard, tools, budget)
        r = run.response
        s.output(r)
        s.set(metadata={"status": r.status, "confidence": r.confidence, "outcome": r.guardrails.outcome,
                        "attempts": r.guardrails.attempts, "llm_calls": len(run.llm_calls)})
        return run


def _run(question, conn, provider, system, agent, max_turns, guard, tools, budget) -> AgentRun:
    guard = guard or Guard(conn, agent=agent, question=question, tools=tools)
    tools = tool_definitions(tools, guard.draft_model)
    messages: list[dict] = [{"role": "user", "text": question}]
    llm_calls: list[LLMCall] = []

    def done(response: AgentResponse) -> AgentRun:
        return AgentRun(response=response, tool_results=guard.tool_results, llm_calls=llm_calls, budget=budget)

    def over_budget(why: str) -> AgentRun:
        progress("budget", f"Stopped: {why}")
        guard.events.append(GuardrailEvent(part="output", check="budget", detail=budget.stop(why)))
        refusal = guard.refuse()
        text = "answer" if "answer" in type(refusal).model_fields else "summary"
        return done(refusal.model_copy(update={text: f"{budget.user_note()} {getattr(refusal, text)}"}))

    for _ in range(max_turns):
        why = budget.exceeded()
        if why:
            return over_budget(why)
        try:
            reply = _chat(provider, system, messages, tools)
        except BudgetExceeded as e:
            return over_budget(str(e))
        budget.spend(reply.usage)
        llm_calls.append(reply.usage)
        messages.append({"role": "assistant", "reply": reply})
        if not reply.tool_calls:
            messages.append({"role": "user", "text": NUDGE})
            continue

        results = []
        for call in reply.tool_calls:
            if call.name != FINAL_ANSWER:
                results.append(ToolResult(call, _tool(guard, call)))
                continue
            progress("check", "Checking the answer against its sources")
            draft, problems = _check(guard, call)
            if not problems:
                return done(guard.complete(draft))
            if guard.attempts >= MAX_ATTEMPTS:
                progress("refused", "The answer failed its checks again, so it won't be shown")
                return done(guard.refuse())
            progress("retry", f"Trying again: {retry_reason([e.check for e in problems])}")
            results.append(ToolResult(call, {"status": "rejected", "feedback": guard.feedback(problems)}))
        messages.append({"role": "tool", "results": results})

    guard.events.append(GuardrailEvent(part="output", check="max_turns",
                                       detail=f"no accepted final answer after {max_turns} LLM calls"))
    return done(guard.refuse())


def _chat(provider: Provider, system: str, messages: list[dict], tools: list[dict]):
    with span("llm", "LLM", llm__provider=provider.name, llm__model_name=provider.model) as s:
        s.input({"system": system, "messages": llm_messages(messages)})
        reply = provider.chat(system, messages, tools)
        u = reply.usage
        s.output({"text": reply.text, "tool_calls": [{"name": c.name, "args": c.args} for c in reply.tool_calls]})
        s.set(llm__token_count__prompt=u.input_tokens, llm__token_count__completion=u.output_tokens,
              llm__token_count__total=u.input_tokens + u.output_tokens,
              llm__token_count__prompt_details__cache_read=u.cached_input_tokens,
              metadata={"ttft_s": u.ttft_s, "total_s": u.total_s})
        return reply


def _tool(guard: Guard, call) -> dict:
    progress("tool", describe_tool(call.name, call.args))
    with span(f"tool:{call.name}", "TOOL", tool__name=call.name) as s:
        s.input(call.args)
        events_before = len(guard.events)
        result = guard.call(call.name, call.args)
        s.output(result)
        s.set(metadata={"result_status": guard.evidence[-1].result_status,
                        "guardrail_events": [e.check for e in guard.events[events_before:]]})
        return result


def _check(guard: Guard, call) -> tuple[BaseModel | None, list[GuardrailEvent]]:
    """Run the output guardrail on a final_answer call. No events = passed."""
    with span("guardrail:output", "GUARDRAIL") as s:
        s.input(call.args)
        draft = None
        try:
            draft = guard.draft_model.model_validate(call.args)
        except ValidationError as e:
            problems = guard.reject_draft("; ".join(
                f"{'.'.join(map(str, err['loc']))}: {err['msg']}" if err["loc"] else err["msg"]
                for err in e.errors()))
        else:
            problems = guard.check_output(draft)
        s.output({"passed": not problems, "events": problems})
        s.set(metadata={"attempt": guard.attempts, "checks": [e.check for e in problems]})
        return draft, problems
