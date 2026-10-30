"""Load-test the API: how one process behaves at 1, 4, 8 and 16 concurrent clients.

Runs the real app (gate, budgets, router, agent loop, tools, guardrails, streaming) over
real HTTP, with fake models that sleep like the real ones. Sleep times are drawn from a
lognormal fitted to the median and p95 of LLM call times in the evals (claude-haiku-5-5,
2026-10-28): router calls 1.15 s / 2.09 s (3 samples in parallel), agent calls
1.69 s / 8.35 s (a table lookup, then the answer). No API key or network needed.

The rate limit is raised for the run: every client comes from 127.0.0.1, so it would
otherwise measure the rate limit, not capacity. The in-flight gate keeps its setting.

Each client asks --questions questions one after another. On a 503 it waits the
Retry-After and asks again, as a person would; a question's time runs from its first try.

Usage: python3 scripts/load_test.py [--levels 1 4 8 16] [--questions 5] [--seed 7]
"""

import argparse
import json
import math
import random
import statistics
import sys
import threading
import time
from pathlib import Path

import httpx
import uvicorn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from catalog_iq.api import create_app  # noqa: E402
from catalog_iq.core.loop import FINAL_ANSWER  # noqa: E402
from catalog_iq.core.providers.base import LLMCall, Reply, ToolCallRequest  # noqa: E402
from catalog_iq.ratelimit import RateLimiter  # noqa: E402
from catalog_iq.router import ROUTE_TOOL  # noqa: E402

ROUTER_S = (1.15, 2.09)  # (median, p95) seconds per LLM call, from evals
AGENT_S = (1.69, 8.35)
PORT = 8766
QUESTION = "Who owns dim_customers?"
ROUTE = {"agent": "catalog_lookup", "request": QUESTION, "reason": "an owner lookup"}
ANSWER = {"status": "answered", "answer": "dim_customers is owned by Maya Okafor.",
          "tables": ["fernwood_dw.marts.dim_customers"],
          "facts": [{"claim": "Owner of dim_customers: Maya Okafor",
                     "citations": ["contract:data/sources/contracts/marts.dim_customers.yaml"]}]}


class Sleepy:
    """A fake model that takes as long as the real one and gives a fixed, valid reply."""

    name, model = "fake", "sleepy-1"

    def __init__(self, timing, reply_for, rng):
        median, p95 = timing
        self.mu, self.sigma = math.log(median), math.log(p95 / median) / 1.645
        self.reply_for, self.rng, self.lock = reply_for, rng, threading.Lock()

    def chat(self, system, messages, tools):
        with self.lock:
            seconds = self.rng.lognormvariate(self.mu, self.sigma)
        time.sleep(seconds)
        name, args = self.reply_for(messages)
        usage = LLMCall("fake", self.model, input_tokens=9000, output_tokens=300, ttft_s=seconds, total_s=seconds)
        return Reply(text="", usage=usage, tool_calls=[ToolCallRequest("c1", name, args)])


def lookup_reply(messages):
    first = not any(m["role"] == "assistant" for m in messages)
    return ("get_table", {"table": "dim_customers"}) if first else (FINAL_ANSWER, ANSWER)


def ask(client):
    """One question through /ask/stream, retrying on 503.

    Returns (outcome, seconds to decision, seconds to done, seconds each 503 took)."""
    start, busy = time.monotonic(), []
    while True:
        tried = time.monotonic()
        with client.stream("POST", "/ask/stream", json={"request": QUESTION}) as r:
            if r.status_code == 503:
                r.read()
                busy.append(time.monotonic() - tried)
                time.sleep(float(r.headers["retry-after"]))
                continue
            if r.status_code != 200:
                r.read()
                return r.status_code, None, None, busy
            decision = None
            for line in r.iter_lines():
                event = json.loads(line)["event"]
                if event == "decision":
                    decision = time.monotonic() - start
                elif event == "error":
                    return "error", None, None, busy
        return 200, decision, time.monotonic() - start, busy


def run_level(base, clients, questions):
    results, lock = [], threading.Lock()

    def client():
        with httpx.Client(base_url=base, timeout=120) as c:
            for _ in range(questions):
                outcome = ask(c)
                with lock:
                    results.append(outcome)

    threads = [threading.Thread(target=client) for _ in range(clients)]
    start = time.monotonic()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return results, time.monotonic() - start


def pct(xs, p):
    return sorted(xs)[min(len(xs) - 1, int(p * len(xs)))] if xs else float("nan")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--levels", type=int, nargs="+", default=[1, 4, 8, 16])
    parser.add_argument("--questions", type=int, default=5, help="questions per client")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    providers = {"router": Sleepy(ROUTER_S, lambda m: (ROUTE_TOOL, {"routes": [ROUTE], "clarify": None}), rng),
                 "catalog_lookup": Sleepy(AGENT_S, lookup_reply, rng)}
    app = create_app(providers=providers, limiter=RateLimiter(per_minute=1e6, burst=10**6))
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=PORT, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    while not server.started:
        time.sleep(0.05)

    print("| Clients | Questions | Answered | 503s (each took) | Other | Decision p50 | Answer p50 | Answer p95 "
          "| Answers/min |")
    print("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for clients in args.levels:
        results, wall = run_level(f"http://127.0.0.1:{PORT}", clients, args.questions)
        ok = [r for r in results if r[0] == 200]
        busy = [s for r in results for s in r[3]]
        took = f" ({max(busy) * 1000:.0f} ms max)" if busy else ""
        other = len(results) - len(ok)
        decisions, answers = [r[1] for r in ok], [r[2] for r in ok]
        print(f"| {clients} | {len(results)} | {len(ok)} | {len(busy)}{took} | {other} "
              f"| {statistics.median(decisions):.1f} s "
              f"| {statistics.median(answers):.1f} s | {pct(answers, 0.95):.1f} s | {len(ok) / wall * 60:.0f} |",
              flush=True)
    server.should_exit = True


if __name__ == "__main__":
    main()
