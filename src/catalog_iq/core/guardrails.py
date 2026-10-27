"""Guardrails: deterministic checks on every live request (no LLM).

One Guard per question. The agent loop uses it like this:

    guard = Guard(conn, agent="catalog_lookup", question=q)
    result = guard.call(name, args)          # tool guardrail, every tool call
    problems = guard.check_output(draft)     # output guardrail, every draft
    if not problems:  response = guard.complete(draft)
    elif guard.attempts < MAX_ATTEMPTS:  send guard.feedback(problems) to the LLM, get a new draft
    else:  response = guard.refuse()

Tool guardrail: budget, allow-list, argument validation, result size. A
blocked call returns an error to the LLM (so it can correct itself) instead of
aborting the question.

Output guardrail: faithfulness (core.faithfulness, incl. lineage relations),
tables returned by a tool (get_table, find_table, lineage nodes and their
loaded_from, or filter_tables matches, which have the lineage node shape), every table of a
cited filter result listed (list_incomplete), "answered" backed by tool data
(no_path and no_match count), escalation mentioned when a tool flagged it
(incl. an ownerless lineage node or filter match listed in `tables`).
On failure the loop retries once, then refuses.

Completion: code fills in citation details, table links, escalation and
confidence, so the LLM cannot hallucinate them. Every guardrail event is
recorded in AgentResponse.guardrails so it can be measured offline (eval) and
online (trace).
"""

import copy
import json
import sqlite3

from catalog_iq.catalog.db import settings
from catalog_iq.core.faithfulness import Vocabulary, check_faithfulness, lineage_nodes
from catalog_iq.core.response import (AgentDraft, AgentResponse, Citation, Escalation, GuardrailEvent,
                                      Guardrails, TableRef, ToolCall)
from catalog_iq.tools.tables import TOOLS

MAX_TOOL_CALLS = 6
MAX_RESULT_CHARS = 20_000
MAX_ATTEMPTS = 2       # first draft + one retry
STRONG_MATCH = 0.7     # another find_table match this strong lowers confidence to medium

REFUSAL = ("I couldn't verify an answer to this from the data catalog, so I won't guess. "
           "Please contact the {team} team ({email}, {slack}).")

_JSON_TYPES = {"string": str, "integer": int, "number": (int, float), "boolean": bool, "object": dict}


def validate_args(args, schema: dict) -> list[str]:
    """Check tool arguments against the tool's input_schema (the subset our tools use)."""
    if not isinstance(args, dict):
        return ["arguments must be an object"]
    props = schema.get("properties", {})
    problems = [f"missing required argument '{k}'" for k in schema.get("required", []) if k not in args]
    for key, value in args.items():
        spec = props.get(key)
        if spec is None:
            problems.append(f"unknown argument '{key}'")
            continue
        expected = _JSON_TYPES.get(spec.get("type"))
        if expected and (not isinstance(value, expected) or (isinstance(value, bool) and expected is not bool)):
            problems.append(f"'{key}' must be {spec['type']}")
            continue
        if "minimum" in spec and value < spec["minimum"]:
            problems.append(f"'{key}' must be >= {spec['minimum']}")
        if "maximum" in spec and value > spec["maximum"]:
            problems.append(f"'{key}' must be <= {spec['maximum']}")
        if spec.get("type") == "string" and not value.strip():
            problems.append(f"'{key}' must not be empty")
        if "enum" in spec and value not in spec["enum"]:
            problems.append(f"'{key}' must be one of {', '.join(spec['enum'])}")
    return problems


class Guard:
    draft_model = AgentDraft  # what final_answer must match; agents with another output override it

    def __init__(self, conn: sqlite3.Connection, agent: str, question: str, tools: list[dict] = TOOLS,
                 max_tool_calls: int = MAX_TOOL_CALLS, max_result_chars: int = MAX_RESULT_CHARS,
                 vocab: Vocabulary | None = None):
        self.conn = conn
        self.agent = agent
        self.question = question
        self.tools = {t["name"]: t for t in tools}
        self.max_tool_calls = max_tool_calls
        self.max_result_chars = max_result_chars
        self.vocab = vocab or Vocabulary(conn)
        self.settings = settings(conn)
        self.tool_results: list[dict] = []   # exactly what the LLM saw
        self.evidence: list[ToolCall] = []
        self.events: list[GuardrailEvent] = []
        self.attempts = 0

    # --- tool guardrail ---------------------------------------------------------------

    def call(self, name: str, args) -> dict:
        """Run one tool call through the checks. Returns the result to give the LLM."""
        if len(self.evidence) >= self.max_tool_calls:
            return self._block(name, args, "tool_budget_exceeded",
                               f"Limit of {self.max_tool_calls} tool calls reached. "
                               "Answer with the results you already have.")
        tool = self.tools.get(name)
        if tool is None:
            return self._block(name, args, "tool_unknown",
                               f"Unknown tool '{name}'. Available: {', '.join(self.tools)}.")
        problems = validate_args(args, tool["input_schema"])
        if problems:
            return self._block(name, args, "bad_args", "; ".join(problems))

        result = self._trim(name, tool["function"](self.conn, **args))
        self.tool_results.append(result)
        self.evidence.append(ToolCall(tool=name, args=args, result_status=result.get("status", "")))
        return result

    def _block(self, name: str, args, check: str, detail: str) -> dict:
        self.events.append(GuardrailEvent(part="tool", check=check, detail=f"{name}: {detail}"))
        self.evidence.append(ToolCall(tool=name, args=args if isinstance(args, dict) else {},
                                      result_status="blocked"))
        return {"status": "error", "error": detail}

    def _trim(self, name: str, result: dict) -> dict:
        size = len(json.dumps(result))
        if size <= self.max_result_chars:
            return result
        result = copy.deepcopy(result)
        while len(json.dumps(result)) > self.max_result_chars:
            # halve the longest list; citations stay whole so cited ids remain checkable
            lists = [k for k, v in result.items() if isinstance(v, list) and v and k != "citations"]
            if not lists:
                break
            key = max(lists, key=lambda k: len(json.dumps(result[k])))
            result[key] = result[key][: len(result[key]) // 2]
        result["truncated"] = f"Result trimmed from {size} characters; some list items are omitted."
        self.events.append(GuardrailEvent(part="tool", check="result_truncated",
                                          detail=f"{name}: {size} > {self.max_result_chars} characters"))
        return result

    # --- output guardrail -------------------------------------------------------------

    def check_output(self, draft: AgentDraft) -> list[GuardrailEvent]:
        """Check one draft against the tool results. Empty list = passed."""
        self.attempts += 1
        problems, _ = check_faithfulness(draft.answer, draft.facts, self.tool_results, self.vocab,
                                         subject_tables=draft.tables, question=self.question)

        returned = self._returned_tables()
        problems += [("table_not_returned", f"{fqn} is an exposure; list only tables in tables"
                      if fqn.startswith("exposure.") else fqn) for fqn in draft.tables if fqn not in returned]
        problems += self._list_incomplete(draft)

        if draft.status == "answered" and not any(r.get("status") in ("found", "ambiguous", "no_path", "no_match")
                                                  for r in self.tool_results):
            problems.append(("no_evidence", "status is answered but no tool returned catalog data"))

        if (draft.status == "answered" and self._escalation(draft) is not None
                and self.settings["escalation_team"].lower() not in draft.answer.lower()):
            problems.append(("escalation_not_in_answer",
                             f"a tool flagged no contract/owner; the answer must refer to "
                             f"{self.settings['escalation_team']}"))

        return self._output_events(problems)

    def reject_draft(self, detail: str) -> list[GuardrailEvent]:
        """A final answer that does not fit the AgentDraft schema counts as a failed attempt."""
        self.attempts += 1
        return self._output_events([("invalid_draft", detail)])

    def _output_events(self, problems: list[tuple[str, str]]) -> list[GuardrailEvent]:
        events = [GuardrailEvent(part="output", check=c, detail=d, attempt=self.attempts) for c, d in problems]
        self.events += events
        return events

    def feedback(self, events: list[GuardrailEvent]) -> str:
        """Retry message for the LLM."""
        lines = [f"- {e.check}: {e.detail}" for e in events]
        return ("Your answer failed verification against the tool results:\n" + "\n".join(lines) +
                "\nRevise it. State only values that appear in the tool results, cite each fact with the "
                "ids given in those results, list only tables the tools returned, and name every person, "
                "table or data category in the answer in a cited fact. State each lineage link in the fact's "
                "relations (direct only if the downstream node's loaded_from contains the upstream), "
                "citing the downstream node's lineage id. If the catalog cannot answer, use "
                "status insufficient_context.")

    def _list_incomplete(self, draft: AgentDraft) -> list[tuple[str, str]]:
        """Every table a cited filter result matched must be in `tables` (D8)."""
        cited = {cid for f in draft.facts for cid in f.citations}
        problems = []
        for r in self.tool_results:
            query = r.get("query")
            if not isinstance(query, dict) or not set(query.get("sources", [])) & cited:
                continue
            missing = [fqn for fqn in query.get("matched", []) if fqn not in draft.tables]
            if missing:
                problems.append(("list_incomplete", f"the cited filter result ({query['sources'][0]}) also matched "
                                                    f"{', '.join(missing)}; list every matched table in tables"))
        return problems

    def _returned_tables(self) -> set[str]:
        fqns = set()
        for r in self.tool_results:
            if isinstance(r.get("table"), dict):
                fqns.add(r["table"]["fqn"])
            if isinstance(r.get("owners"), dict) and r["owners"].get("inherited_from"):
                fqns.add(r["owners"]["inherited_from"])  # where a stg_/int_ table's owner comes from
            fqns.update(m["fqn"] for m in r.get("matches", []) if "fqn" in m)
        for n in lineage_nodes(self.tool_results):
            if n["type"] == "table":
                fqns.add(n["fqn"])
            # a node's direct upstreams are tables the tool returned too, by name
            fqns.update(up for up in n["loaded_from"] if not up.startswith("exposure."))
        return fqns

    def _escalation(self, draft: AgentDraft) -> dict | None:
        """Escalation contact from a get_table result or lineage node about a table in the answer."""
        for node in lineage_nodes(self.tool_results):
            if node["type"] == "table" and node["fqn"] in draft.tables and node["owner"].get("escalation"):
                return node["owner"]["escalation"]
        for r in self.tool_results:
            table = r.get("table")
            if not isinstance(table, dict) or (draft.tables and table["fqn"] not in draft.tables):
                continue
            for part in ("contract", "owners"):
                esc = (r.get(part) or {}).get("escalation")
                if esc:
                    return esc
        return None

    # --- completion -------------------------------------------------------------------

    def complete(self, draft: AgentDraft) -> AgentResponse:
        """Turn a draft that passed check_output into the final response."""
        details = {c["id"]: c for r in self.tool_results for c in r.get("citations", [])}
        cited = list(dict.fromkeys(cid for f in draft.facts for cid in f.citations))
        esc = self._escalation(draft) if draft.status == "answered" else None
        return self._response(
            status=draft.status, answer=draft.answer, facts=draft.facts, clarification=draft.clarification,
            tables=[self._table_ref(fqn) for fqn in draft.tables],
            citations=[Citation(**{"source": cid.split(":", 1)[0], "ref": cid.split(":", 1)[1],
                                   **details.get(cid, {"id": cid})})
                       for cid in cited],  # compact citations carry only id (+ url); some ids are only inline
            escalation=Escalation(team=esc["team"], email=esc["email"], slack=esc["slack"]) if esc else None,
            confidence=self._confidence(draft),
            outcome="passed" if self.attempts <= 1 else "passed_after_retry")

    def refuse(self) -> AgentResponse:
        """Safe fallback when the retry also failed."""
        s = self.settings
        return self._response(
            status="insufficient_context",
            answer=REFUSAL.format(team=s["escalation_team"], email=s["escalation_email"],
                                  slack=s["escalation_slack"]),
            escalation=Escalation(team=s["escalation_team"], email=s["escalation_email"],
                                  slack=s["escalation_slack"]),
            confidence="low", outcome="refused")

    def _response(self, outcome: str, **fields) -> AgentResponse:
        return AgentResponse(
            agent=self.agent, question=self.question, evidence=self.evidence,
            guardrails=Guardrails(attempts=max(self.attempts, 1), events=self.events, outcome=outcome),
            **fields)

    def _table_ref(self, fqn: str) -> TableRef:
        return TableRef(fqn=fqn, name=fqn.split(".")[-1], url=f"{self.settings['base_url']}/table/{fqn}")

    def _confidence(self, draft: AgentDraft) -> str:
        if draft.status != "answered":
            return "low"
        if self.attempts > 1:
            return "medium"
        for r in self.tool_results:
            matches = [m for m in r.get("matches", []) if "fqn" in m]  # exposure matches have ids
            if not any(m["fqn"] in draft.tables for m in matches):
                continue
            others = [m for m in matches if m["fqn"] not in draft.tables and m["score"] >= STRONG_MATCH]
            if r.get("status") == "ambiguous" or others:
                return "medium"  # the table was picked over other plausible matches
        return "high"
