"""PI Classifier eval: answer key, per-column scoring, metrics and runner (D4).

    python -m catalog_iq.evals.pi                       # open + dev, 3 runs each
    python -m catalog_iq.evals.pi --sets heldout        # only once tuning is finished
    python -m catalog_iq.evals.pi --tables raw_helpdesk.tickets --runs 1

Answer key: evals/pi_classifier/answer_key.yaml, true PI per column. Only
this module reads it; the agent's tools cannot. `open` tables run as they are
(their open columns are scored); `dev` and `heldout` tables run in blind mode
(every column scored).

Per column, a recommendation is scored against the truth:
  true PI     pi + right category → tp (category_ok) · pi + other category → tp
              needs_manual_review → flagged · not_pi → fn (a miss)
  not PI      not_pi → tn · needs_manual_review → flagged · pi → fp
Metrics: precision = tp / (tp + fp); recall = (tp + flagged true PI) / true PI
(a column sent to a steward is not missed); strict recall = tp / true PI;
category accuracy = right category / tp; manual-review rate; trap columns
right in every run. A table run passes when every column is exactly right
(manual review is not exact) and the guardrails did not refuse.

Writes evals/pi_classifier/runs/<timestamp>/{results.jsonl, summary.json}.
"""

import argparse
import json
import time
import traceback
from collections import Counter, defaultdict
from datetime import datetime
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from catalog_iq.agents.pi_classifier import AGENT, recommend_pi
from catalog_iq.agents.pi_classifier.schema import PIResponse
from catalog_iq.catalog.db import connect
from catalog_iq.core.config import agent_config, load_env
from catalog_iq.core.providers import make_provider
from catalog_iq.core.providers.base import QuotaExhausted
from catalog_iq.core.tracing import setup_tracing, shutdown_tracing, span
from catalog_iq.evals.answer_key import EVALS_DIR
from catalog_iq.evals.run import performance

PISet = Literal["open", "dev", "heldout"]
TARGETS = {"precision": 0.90, "recall": 0.95}


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TableKey(_Strict):
    table: str
    set: PISet
    columns: dict[str, str] = Field(description="column → 'category (subject)' or 'none'")
    traps: dict[str, str] = Field(default_factory=dict)

    @property
    def blind(self) -> bool:
        return self.set != "open"


class PIAnswerKey(_Strict):
    tables: list[TableKey]


def answer_key_path():
    return EVALS_DIR / AGENT / "answer_key.yaml"


def load_key(sets: tuple[str, ...] | None = None) -> list[TableKey]:
    key = PIAnswerKey.model_validate(yaml.safe_load(answer_key_path().read_text()))
    return [t for t in key.tables if sets is None or t.set in sets]


def parse_truth(value: str) -> tuple[str | None, str | None]:
    """'email (customer)' → ('email', 'customer'); 'none' → (None, None)."""
    if value == "none":
        return None, None
    category, subject = value.split(" (")
    return category, subject.rstrip(")")


# --- scoring ------------------------------------------------------------------------


class ColumnScore(_Strict):
    column: str
    truth: str
    got: str            # pi:<category> | not_pi | needs_manual_review | missing
    outcome: Literal["tp", "fp", "tn", "fn", "flagged_pi", "flagged_not_pi", "missing"]
    category_ok: bool | None = None
    subject_ok: bool | None = None
    confidence: str | None = None
    trap: bool = False


class TableResult(_Strict):
    table: str
    set: str
    run: int
    status: str
    outcome: str
    attempts: int
    columns: list[ColumnScore]
    guardrail_checks: list[str]

    @property
    def passed(self) -> bool:
        return self.outcome != "refused" and all(
            c.outcome in ("tn",) or (c.outcome == "tp" and c.category_ok) for c in self.columns)


def score(response: PIResponse, key: TableKey, run_number: int = 1) -> TableResult:
    recs = {r.column: r for r in response.recommendations}
    columns = []
    for col, truth in key.columns.items():
        category, subject = parse_truth(truth)
        r = recs.get(col)
        if r is None:
            columns.append(ColumnScore(column=col, truth=truth, got="missing", outcome="missing", trap=col in key.traps))
            continue
        got = f"pi:{r.category}" if r.recommendation == "pi" else r.recommendation
        if r.recommendation == "pi":
            outcome = "tp" if category else "fp"
        elif r.recommendation == "not_pi":
            outcome = "fn" if category else "tn"
        else:
            outcome = "flagged_pi" if category else "flagged_not_pi"
        columns.append(ColumnScore(
            column=col, truth=truth, got=got, outcome=outcome, confidence=r.confidence, trap=col in key.traps,
            category_ok=(r.category == category) if outcome == "tp" else None,
            subject_ok=(r.data_subject == subject) if outcome == "tp" else None))
    g = response.guardrails
    return TableResult(table=key.table, set=key.set, run=run_number, status=response.status, outcome=g.outcome,
                       attempts=g.attempts, columns=columns,
                       guardrail_checks=sorted({f"{e.part}:{e.check}" for e in g.events}))


def _ratio(a: int, b: int) -> float | None:
    return round(a / b, 3) if b else None


def summarize(results: list[TableResult]) -> dict:
    cols = [c for r in results for c in r.columns]
    n = Counter(c.outcome for c in cols)
    true_pi = sum(1 for c in cols if c.truth != "none")  # a missing true-PI column counts as missed
    tps = [c for c in cols if c.outcome == "tp"]
    traps: dict[str, list[bool]] = defaultdict(list)
    for r in results:
        for c in r.columns:
            if c.trap:
                ok = c.outcome == "tn" or (c.outcome == "tp" and c.category_ok)
                traps[f"{r.table}.{c.column}"].append(ok)
    by_table = defaultdict(list)
    for r in results:
        by_table[r.table].append(r)
    fired = Counter(check for r in results for check in r.guardrail_checks)
    calibration = {lvl: _ratio(sum(1 for c in cols if c.confidence == lvl and c.outcome in ("tp", "tn")
                                   and c.category_ok is not False),
                               sum(1 for c in cols if c.confidence == lvl and c.outcome in ("tp", "tn", "fp", "fn")))
                   for lvl in ("high", "medium")}
    return {
        "table_pass_rate": _ratio(sum(r.passed for r in results), len(results)),
        "columns_scored": len(cols),
        "precision": _ratio(n["tp"], n["tp"] + n["fp"]),
        "recall": _ratio(n["tp"] + n["flagged_pi"], true_pi),
        "strict_recall": _ratio(n["tp"], true_pi),
        "category_accuracy": _ratio(sum(1 for c in tps if c.category_ok), len(tps)),
        "subject_accuracy": _ratio(sum(1 for c in tps if c.subject_ok), len(tps)),
        "manual_review_rate": _ratio(n["flagged_pi"] + n["flagged_not_pi"], len(cols)),
        "outcomes": dict(sorted(n.items())),
        "calibration": calibration,
        "targets_met": {"precision": (_ratio(n["tp"], n["tp"] + n["fp"]) or 0) >= TARGETS["precision"],
                        "recall": (_ratio(n["tp"] + n["flagged_pi"], true_pi) or 0) >= TARGETS["recall"]},
        "traps": {k: f"{sum(v)}/{len(v)}" for k, v in sorted(traps.items())},
        "by_set": {s: _ratio(sum(r.passed for r in results if r.set == s), sum(1 for r in results if r.set == s))
                   for s in sorted({r.set for r in results})},
        "guardrails": {
            "retry_rate": _ratio(sum(r.attempts > 1 for r in results), len(results)),
            "refusal_rate": _ratio(sum(r.outcome == "refused" for r in results), len(results)),
            "trigger_rate": {k: _ratio(v, len(results)) for k, v in sorted(fired.items())},
        },
        "errors_by_table": {t: sorted({f"{c.column}: {c.got} (truth {c.truth})" for r in rs for c in r.columns
                                       if not (c.outcome == "tn" or (c.outcome == "tp" and c.category_ok))})
                            for t, rs in sorted(by_table.items()) if not all(r.passed for r in rs)},
    }


# --- runner -------------------------------------------------------------------------


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sets", nargs="+", default=["open", "dev"], choices=["open", "dev", "heldout"])
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--tables", nargs="+", help="only tables whose fqn ends with one of these")
    args = parser.parse_args(argv)

    load_env()
    cfg = agent_config(AGENT)
    provider = make_provider(cfg)
    keys = [k for k in load_key(tuple(args.sets)) if not args.tables or any(k.table.endswith(t) for t in args.tables)]
    out = EVALS_DIR / AGENT / "runs" / datetime.now().strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True)
    traced = setup_tracing()
    print(f"tracing: {'on (Phoenix)' if traced else 'off'}", flush=True)

    conn = connect()
    results, runs, wall, errors, stopped = [], [], [], [], None
    with (out / "results.jsonl").open("w") as f:
        for n in range(1, args.runs + 1):
            for k in keys:
                start = time.perf_counter()
                with span(f"eval {k.table} run {n}", "CHAIN", session__id=out.name,
                          tag__tags=["eval", k.set, k.table]) as s:
                    s.input({"table": k.table, "blind": k.blind})
                    try:
                        run = recommend_pi(k.table, conn=conn, provider=provider, blind=k.blind)
                    except QuotaExhausted as e:
                        stopped = f"run {n} {k.table}: {e}"
                        print(f"STOPPED {stopped}", flush=True)
                        break
                    except Exception as e:  # an API failure should not lose the rest of the eval
                        errors.append({"table": k.table, "run": n, "error": f"{type(e).__name__}: {e}"})
                        f.write(json.dumps({"table": k.table, "run": n, "error": traceback.format_exc()}) + "\n")
                        print(f"run {n} {k.table}: ERROR {type(e).__name__}", flush=True)
                        continue
                    result = score(run.response, k, n)
                    s.output({"passed": result.passed})
                    s.set(metadata={"table": k.table, "set": k.set, "run": n, "passed": result.passed})
                wall.append(time.perf_counter() - start)
                results.append(result)
                runs.append(run)
                f.write(json.dumps({**result.model_dump(), "passed": result.passed,
                                    "response": run.response.model_dump(), "tool_results": run.tool_results,
                                    "llm_calls": [vars(c) for c in run.llm_calls]}) + "\n")
                f.flush()
                wrong = [c.column for c in result.columns if not (c.outcome == "tn" or
                                                                  (c.outcome == "tp" and c.category_ok))]
                print(f"run {n} {k.table}: {'PASS' if result.passed else 'FAIL'} {result.outcome} "
                      f"attempts={result.attempts} {wall[-1]:.0f}s" + (f" wrong={wrong}" if wrong else ""),
                      flush=True)
            if stopped:
                break
    conn.close()
    shutdown_tracing()

    summary = {"agent": AGENT, "provider": cfg["provider"], "model": cfg["model"], "sets": args.sets,
               "runs": args.runs, "tables": len(keys), "scored": len(results), "stopped": stopped,
               "errors": errors, **summarize(results), "performance": performance(runs, wall)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return out


if __name__ == "__main__":
    main()
