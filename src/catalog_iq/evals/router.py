"""Router eval (R4): labelled requests, per-request scoring, metrics and runner.

    python -m catalog_iq.evals.router                    # dev, 3 runs each
    python -m catalog_iq.evals.router --sets heldout     # only once tuning is finished
    python -m catalog_iq.evals.router --ids M1 F2 --runs 1

Key: evals/router/requests.yaml. Only the route is scored; no agent runs.

Per request run:
  route_ok   the routed agents match the expected ones (a multiset: order ignored),
             or, when clarify is expected, the router asked
  table_ok   route_ok and each recommender's resolved table matches
  rewrite_ok a follow-up's rewritten catalog_lookup question names the expected
             table (only when expect.rewrite_names is set)
  passed     all of the above
Metrics: route and table accuracy, pass rate, clarify precision (asked only when
it should) and recall, accuracy by routing confidence (high should be ~100%),
how often a medium decision's offered alternative was the expected route, and
tokens / latency per request (all 3 samples).

Writes evals/router/runs/<timestamp>/{results.jsonl, summary.json}.
"""

import argparse
import json
import time
import traceback
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from catalog_iq.catalog.db import connect
from catalog_iq.core.config import agent_config, load_env
from catalog_iq.core.providers import make_provider
from catalog_iq.core.providers.base import QuotaExhausted
from catalog_iq.core.tracing import setup_tracing, shutdown_tracing, span
from catalog_iq.evals.answer_key import EVALS_DIR
from catalog_iq.evals.run import performance
from catalog_iq.router import AGENT, _table_names, named_tables, route
from catalog_iq.router.schema import RouteDecision, Turn

KEY_PATH = EVALS_DIR / AGENT / "requests.yaml"
TARGETS = {"route_accuracy": 0.90, "high_confidence_accuracy": 0.97}


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Expected(_Strict):
    routes: list[dict] = Field(default_factory=list)
    clarify: bool = False
    rewrite_names: str | None = None


class RequestKey(_Strict):
    id: str
    set: Literal["dev", "heldout"]
    group: str
    request: str
    history: list[Turn] = Field(default_factory=list)
    expect: Expected


class RequestResult(_Strict):
    id: str
    set: str
    group: str
    run: int
    got: str
    expected: str
    confidence: str
    route_ok: bool
    table_ok: bool
    rewrite_ok: bool | None = None
    alternative_ok: bool | None = None

    @property
    def passed(self) -> bool:
        return self.table_ok and self.rewrite_ok is not False


def load_key(sets: tuple[str, ...] = ("dev",), path: Path = KEY_PATH) -> list[RequestKey]:
    return [k for k in (RequestKey(**q) for q in yaml.safe_load(path.read_text())) if k.set in sets]


def _pairs(routes) -> Counter:
    return Counter((r["agent"], r.get("table")) for r in routes)


def _describe(pairs: Counter) -> str:
    return " + ".join(sorted(f"{a}·{(t or '-').split('.')[-1]}" for a, t in pairs.elements())) or "-"


def score(decision: dict, key: RequestKey, run: int, names: dict[str, str]) -> RequestResult:
    expected = "clarify" if key.expect.clarify else _describe(_pairs(key.expect.routes))
    got_pairs = _pairs(decision["routes"])
    got = "clarify" if decision["status"] == "clarify" else _describe(got_pairs)
    if key.expect.clarify:
        route_ok = table_ok = decision["status"] == "clarify"
    else:
        want = _pairs(key.expect.routes)
        route_ok = Counter(a for a, _ in got_pairs.elements()) == Counter(a for a, _ in want.elements())
        table_ok = route_ok and got_pairs == want
    rewrite_ok = None
    if key.expect.rewrite_names:
        lookups = [r for r in decision["routes"] if r["agent"] == "catalog_lookup"]
        rewrite_ok = any(key.expect.rewrite_names in named_tables(r["request"], names) for r in lookups)
    alternative_ok = None
    if decision["confidence"] == "medium" and decision.get("alternative"):
        alternative_ok = (not key.expect.clarify
                          and _pairs(decision["alternative"]["routes"]) == _pairs(key.expect.routes))
    return RequestResult(id=key.id, set=key.set, group=key.group, run=run, got=got, expected=expected,
                         confidence=decision["confidence"], route_ok=route_ok, table_ok=table_ok,
                         rewrite_ok=rewrite_ok, alternative_ok=alternative_ok)


def _rate(xs: list[bool]) -> float | None:
    return round(sum(xs) / len(xs), 3) if xs else None


def summarize(results: list[RequestResult], keys: list[RequestKey]) -> dict:
    by_id = {k.id: k for k in keys}
    asked = [r for r in results if r.got == "clarify"]
    should = [r for r in results if by_id[r.id].expect.clarify]
    by_conf = defaultdict(list)
    for r in results:
        by_conf[r.confidence].append(r.passed)
    by_group = defaultdict(list)
    for r in results:
        by_group[r.group].append(r.passed)
    failing = defaultdict(list)
    for r in results:
        if not r.passed:
            failing[r.id].append(r.got)
    return {
        "route_accuracy": _rate([r.route_ok for r in results]),
        "table_accuracy": _rate([r.table_ok for r in results]),
        "pass_rate": _rate([r.passed for r in results]),
        "clarify_precision": _rate([by_id[r.id].expect.clarify for r in asked]),
        "clarify_recall": _rate([r.got == "clarify" for r in should]),
        "rewrite_accuracy": _rate([r.rewrite_ok for r in results if r.rewrite_ok is not None]),
        "confidence_share": {c: round(len(v) / len(results), 3) for c, v in sorted(by_conf.items())},
        "accuracy_by_confidence": {c: _rate(v) for c, v in sorted(by_conf.items())},
        "alternative_was_expected": _rate([r.alternative_ok for r in results if r.alternative_ok is not None]),
        "by_group": {g: _rate(v) for g, v in sorted(by_group.items())},
        "failing": {i: {"expected": next(r.expected for r in results if r.id == i), "got": Counter(g)}
                    for i, g in sorted(failing.items())},
    }


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sets", nargs="+", default=["dev"], choices=["dev", "heldout"])
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--ids", nargs="+", help="only these request ids")
    args = parser.parse_args(argv)

    load_env()
    cfg = agent_config(AGENT)
    provider = make_provider(cfg)
    keys = [k for k in load_key(tuple(args.sets)) if not args.ids or k.id in args.ids]
    out = EVALS_DIR / AGENT / "runs" / datetime.now().strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True)
    traced = setup_tracing()
    print(f"tracing: {'on (Phoenix)' if traced else 'off'}", flush=True)

    conn = connect()
    names = _table_names(conn)
    results, runs, wall, errors, stopped = [], [], [], [], None
    with (out / "results.jsonl").open("w") as f:
        for n in range(1, args.runs + 1):
            for k in keys:
                start = time.perf_counter()
                with span(f"eval {k.id} run {n}", "CHAIN", session__id=out.name, tag__tags=["eval", k.set, k.id]):
                    try:
                        run = route(k.request, history=k.history, conn=conn, provider=provider)
                    except QuotaExhausted as e:
                        stopped = f"run {n} {k.id}: {e}"
                        print(f"STOPPED {stopped}", flush=True)
                        break
                    except Exception as e:  # an API failure should not lose the rest of the eval
                        errors.append({"id": k.id, "run": n, "error": f"{type(e).__name__}: {e}"})
                        f.write(json.dumps({"id": k.id, "run": n, "error": traceback.format_exc()}) + "\n")
                        print(f"run {n} {k.id}: ERROR {type(e).__name__}", flush=True)
                        continue
                decision: RouteDecision = run.decision
                result = score(decision.model_dump(), k, n, names)
                wall.append(time.perf_counter() - start)
                results.append(result)
                runs.append(run)
                f.write(json.dumps({**result.model_dump(), "passed": result.passed,
                                    "decision": decision.model_dump(),
                                    "llm_calls": [vars(c) for c in run.llm_calls]}) + "\n")
                f.flush()
                print(f"run {n} {k.id}: {'PASS' if result.passed else 'FAIL'} got={result.got} "
                      f"expected={result.expected} conf={result.confidence} ({decision.agreement}) "
                      f"{wall[-1]:.1f}s", flush=True)
            if stopped:
                break
    conn.close()
    shutdown_tracing()

    summary = {"agent": AGENT, "provider": cfg["provider"], "model": cfg["model"], "sets": args.sets,
               "runs": args.runs, "requests": len(keys), "scored": len(results), "stopped": stopped,
               "errors": errors, **summarize(results, keys), "performance": performance(runs, wall),
               "targets": TARGETS}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=dict))
    print(json.dumps({k: v for k, v in summary.items() if k != "failing"}, indent=2, default=dict))
    if summary["failing"]:
        print("failing:", json.dumps(summary["failing"], default=dict))
    print(f"wrote {out}")
    return out


if __name__ == "__main__":
    main()
