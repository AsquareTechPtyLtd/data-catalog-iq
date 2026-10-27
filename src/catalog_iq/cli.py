"""Command line: ask the Catalog Lookup agent a question, or get PI / data category recommendations.

    catalog-iq ask "Who owns dim_customers?"
    catalog-iq ask "Who owns dim_customers?" --json      # full AgentResponse
    catalog-iq ask "Who owns dim_customers?" --verbose   # + tool calls, guardrails, tokens
    catalog-iq                                           # interactive: one question per line
    catalog-iq recommend-pi raw_returns.return_requests  # PI Classifier: recommendations for review
    catalog-iq recommend-category raw_web.page_views          # Data Category Recommender: one data category for review
    catalog-iq route "Who owns fct_orders and is it PI?" # the router's decision only (no agent runs)
    catalog-iq serve                                     # API + web page on http://127.0.0.1:8000

Each question is answered on its own (no chat memory). With
PHOENIX_COLLECTOR_ENDPOINT set, every question is a trace in Phoenix.
"""

import argparse
import sys

from catalog_iq.agents.data_category_recommender import recommend_data_category
from catalog_iq.agents.catalog_lookup import ask
from catalog_iq.agents.pi_classifier import recommend_pi
from catalog_iq.catalog.db import connect
from catalog_iq.core.config import load_env
from catalog_iq.core.loop import AgentRun
from catalog_iq.core.providers import Provider
from catalog_iq.core.tracing import setup_tracing, shutdown_tracing

MAX_QUESTION_CHARS = 1000


def check_question(question: str) -> str | None:
    """Input check before any LLM call. Returns an error message, or None."""
    if not question.strip():
        return "Please ask a question."
    if len(question) > MAX_QUESTION_CHARS:
        return f"Questions are limited to {MAX_QUESTION_CHARS} characters."
    return None


def render(run: AgentRun, verbose: bool = False) -> str:
    r = run.response
    lines = [f"{r.answer}", "", f"status: {r.status} · confidence: {r.confidence}"]
    if r.clarification:
        lines += ["", r.clarification.question] + [f"  • {o}" for o in r.clarification.options]
    if r.tables:
        lines += ["", "Tables"] + [f"  • {t.fqn}  {t.url}" for t in r.tables]
    if r.facts:
        ref = {c.id: i for i, c in enumerate(r.citations, 1)}
        lines += ["", "Facts"]
        lines += [f"  {i}. {f.claim}  " + "".join(f"[{ref[c]}]" for c in f.citations)
                  for i, f in enumerate(r.facts, 1)]
        lines += ["", "Sources"] + [f"  [{i}] {c.source}: {c.ref}" + (f"  {c.url}" if c.url else "")
                                    for i, c in enumerate(r.citations, 1)]
    if r.escalation:
        e = r.escalation
        lines += ["", f"Contact: {e.team} team · {e.email} · {e.slack}"]
    if verbose:
        lines += ["", "Tool calls"] + [f"  {c.tool}({c.args}) → {c.result_status}" for c in r.evidence]
        g = r.guardrails
        lines += ["", f"Guardrails: {g.outcome}, {g.attempts} attempt(s)"]
        lines += [f"  {e.part}:{e.check} {e.detail}" + (f" (attempt {e.attempt})" if e.attempt else "")
                  for e in g.events]
    calls = run.llm_calls
    if calls:
        tokens = sum(c.input_tokens + c.output_tokens for c in calls)
        seconds = sum(c.total_s for c in calls)
        lines += ["", f"— {len(calls)} LLM calls · {tokens / 1000:.1f}k tokens · {seconds:.1f}s · "
                      f"{calls[0].provider}/{calls[0].model}"]
    return "\n".join(lines)


def render_pi(run: AgentRun, verbose: bool = False) -> str:
    r = run.response
    lines = [r.summary, "", f"status: {r.status} · {r.review_status.replace('_', ' ')}"]
    if r.table:
        lines += [f"table: {r.table.fqn}  {r.table.url}"]
    if r.recommendations:
        lines += ["", "Recommendations (pending steward review)"]
        for c in r.recommendations:
            what = f"PI: {c.pi_type} ({c.data_subject}), {c.sensitivity}" if c.recommendation == "pi" else \
                "not PI" if c.recommendation == "not_pi" else "needs manual review"
            lines += [f"  • {c.column}: {what} · {c.confidence} confidence (recorded: {c.recorded_status})",
                      f"      {c.reason}"] + ([f"      note: {c.note}"] if c.note else [])
            if verbose:
                lines += [f"      signal {s.signal} → {', '.join(s.suggests)}: {s.detail}" for s in c.signals]
    if verbose:
        g = r.guardrails
        lines += ["", "Tool calls"] + [f"  {c.tool}({c.args}) → {c.result_status}" for c in r.evidence]
        lines += ["", f"Guardrails: {g.outcome}, {g.attempts} attempt(s)"]
        lines += [f"  {e.part}:{e.check} {e.detail}" for e in g.events]
    calls = run.llm_calls
    if calls:
        tokens = sum(c.input_tokens + c.output_tokens for c in calls)
        lines += ["", f"— {len(calls)} LLM calls · {tokens / 1000:.1f}k tokens · "
                      f"{sum(c.total_s for c in calls):.1f}s · {calls[0].provider}/{calls[0].model}"]
    return "\n".join(lines)


def render_data_category(run: AgentRun, verbose: bool = False) -> str:
    r = run.response
    lines = [r.summary, "", f"status: {r.status} · confidence: {r.confidence} · {r.review_status.replace('_', ' ')}"]
    if r.table:
        lines += [f"table: {r.table.fqn}  {r.table.url}"]
    if r.current:
        lines += [f"current data category: {r.current.data_category_id} {r.current.name or ''}".rstrip()
                  if r.current.data_category_id else f"inherits its data category from {r.current.inherits_from}"]
    b = r.recommendation
    if b:
        retention = (f"{b.retention['months']} months {b.retention['trigger']}" if isinstance(b.retention, dict)
                     else b.retention)
        stewards = "; ".join(f"{role}: {', '.join(p['name'] for p in people)}" for role, people in b.stewards.items())
        lines += ["", f"Recommended data category (pending steward review): {b.data_category_id} {b.name} ({b.status})",
                  f"  retention: {retention}", f"  stewards: {stewards or 'none on record'}"]
        lines += [f"  flag: {f}" for f in b.flags]
    if r.reason:
        lines += ["", f"Reason: {r.reason}"]
    if r.considered:
        lines += [f"Also considered: {', '.join(r.considered)}"]
    if r.note:
        lines += [f"Note: {r.note}"]
    if verbose:
        lines += ["", "Signals"] + [f"  {s.signal} → {', '.join(s.suggests)}: {s.detail}" for s in r.signals]
        g = r.guardrails
        lines += ["", "Tool calls"] + [f"  {c.tool}({c.args}) → {c.result_status}" for c in r.evidence]
        lines += ["", f"Guardrails: {g.outcome}, {g.attempts} attempt(s)"]
        lines += [f"  {e.part}:{e.check} {e.detail}" for e in g.events]
    calls = run.llm_calls
    if calls:
        tokens = sum(c.input_tokens + c.output_tokens for c in calls)
        lines += ["", f"— {len(calls)} LLM calls · {tokens / 1000:.1f}k tokens · "
                      f"{sum(c.total_s for c in calls):.1f}s · {calls[0].provider}/{calls[0].model}"]
    return "\n".join(lines)


def render_route(d) -> str:
    lines = [f"status: {d.status} · routing confidence: {d.confidence} ({d.agreement} samples agree)"]
    for r in d.routes:
        lines += [f"  → {r.label}" + (f" · {r.table}" if r.table else "") + f"  ({r.reason})"]
        if r.agent == "catalog_lookup" and r.request != d.request:
            lines += [f"      asks: {r.request}"]
    if d.alternative:
        lines += [f"  or did you mean: {d.alternative.label}"]
    if d.clarification:
        lines += ["", d.clarification.question] + [f"  • {o.label}" for o in d.clarification.options]
    if d.note:
        lines += [f"note: {d.note}"]
    return "\n".join(lines)


def answer(question: str, as_json: bool, verbose: bool, conn, provider: Provider | None) -> int:
    error = check_question(question)
    if error:
        print(error, file=sys.stderr)
        return 2
    try:
        run = ask(question, conn=conn, provider=provider)
    except Exception as e:  # missing key, no credit, network: say so plainly
        print(f"error: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    print(run.response.model_dump_json(indent=2) if as_json else render(run, verbose))
    return 0


def main(argv: list[str] | None = None, provider: Provider | None = None) -> int:
    parser = argparse.ArgumentParser(prog="catalog-iq", description="Ask the Fernwood data catalog.")
    sub = parser.add_subparsers(dest="command")
    ask_cmd = sub.add_parser("ask", help="ask one question")
    ask_cmd.add_argument("question")
    pi_cmd = sub.add_parser("recommend-pi", help="PI recommendations for one table's open columns")
    pi_cmd.add_argument("table")
    data_category_cmd = sub.add_parser("recommend-category",
                                       help="data category recommendation for one table that has none")
    data_category_cmd.add_argument("table")
    route_cmd = sub.add_parser("route", help="show which agent the router picks (no agent runs)")
    route_cmd.add_argument("request")
    serve_cmd = sub.add_parser("serve", help="run the API and the web page")
    serve_cmd.add_argument("--host", default="127.0.0.1")
    serve_cmd.add_argument("--port", type=int, default=8000)
    for p in (parser, ask_cmd, pi_cmd, data_category_cmd, route_cmd):
        p.add_argument("--json", action="store_true", help="print the full AgentResponse as JSON")
        p.add_argument("--verbose", "-v", action="store_true", help="show tool calls, guardrails and tokens")
    args = parser.parse_args(argv)
    if args.command == "serve":
        import uvicorn  # optional extra: pip install -e ".[api]"

        from catalog_iq.api import create_app
        uvicorn.run(create_app(), host=args.host, port=args.port)
        return 0

    load_env()
    traced = setup_tracing()
    conn = connect()
    try:
        if args.command == "ask":
            return answer(args.question, args.json, args.verbose, conn, provider)
        if args.command == "route":
            from catalog_iq.router import route
            error = check_question(args.request)
            if error:
                print(error, file=sys.stderr)
                return 2
            try:
                d = route(args.request, conn=conn, provider=provider).decision
            except Exception as e:  # missing key, no credit, network: say so plainly
                print(f"error: {type(e).__name__}: {e}", file=sys.stderr)
                return 1
            print(d.model_dump_json(indent=2) if args.json else render_route(d))
            return 0
        if args.command in ("recommend-pi", "recommend-category"):
            agent, render_fn = ((recommend_pi, render_pi) if args.command == "recommend-pi"
                                else (recommend_data_category, render_data_category))
            try:
                run = agent(args.table, conn=conn, provider=provider)
            except Exception as e:  # missing key, no credit, network: say so plainly
                print(f"error: {type(e).__name__}: {e}", file=sys.stderr)
                return 1
            print(run.response.model_dump_json(indent=2) if args.json else render_fn(run, args.verbose))
            return 0
        print("Ask about Fernwood's data catalog. Empty line or Ctrl-D to quit."
              + (" Tracing to Phoenix." if traced else ""))
        while True:
            try:
                question = input("\n> ")
            except EOFError:
                break
            if not question.strip():
                break
            answer(question, args.json, args.verbose, conn, provider)
        return 0
    finally:
        conn.close()
        shutdown_tracing()


if __name__ == "__main__":
    sys.exit(main())
