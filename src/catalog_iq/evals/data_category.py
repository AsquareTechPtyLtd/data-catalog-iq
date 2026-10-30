"""Data Category Recommender eval: answer key, per-table scoring, metrics and runner (C4).

    python -m catalog_iq.evals.data_category                       # open + dev, 3 runs each
    python -m catalog_iq.evals.data_category --sets heldout        # only once tuning is finished
    python -m catalog_iq.evals.data_category --tables page_views --runs 1
    python -m catalog_iq.evals.data_category --catalog data/catalog/catalog_scale.db   # scale test (N5)
    python -m catalog_iq.evals.data_category --suite injection --sets dev   # planted instructions (evals/injection.py)

Answer key: evals/data_category_recommender/answer_key.yaml, the true data category per table.
Only this module reads it; the agent's tools cannot. `open` tables (no data category
yet) run as they are; `dev` and `heldout` tables run in blind mode (their data category
and every lineage neighbour's hidden, N2). Responses are scored as dicts
(response.model_dump()).

The response contract (built in step 4):
  status          recommended | no_fit | needs_manual_review | assigned | not_applicable | not_found
  recommendation  {"data_category_id": ...} when status is recommended
  confidence      high | medium | low
  guardrails      {"outcome", "attempts", "events": [{"part", "check"}]}

Per table run, a response is scored against the truth:
  recommended truth                        → correct
  recommended a data category in `acceptable`        → acceptable (a steward could choose it)
  recommended another data category                  → wrong
  no_fit / needs_manual_review             → no_fit / flagged (not wrong, not right)
  refused, or any other status             → refused / error
Metrics: accuracy = correct / runs; lenient accuracy = (correct + acceptable)
/ runs; precision = correct / recommended; flagged and no_fit rates; traps
right in every run; calibration (correct or acceptable share per confidence
level); retrieval recall: the true data category was among the matches of a
search_data_categories call (rank ≤ k), which caps what the agent can get right.
A table run passes when it is correct and the guardrails did not refuse.

Writes evals/data_category_recommender/runs/<timestamp>/{results.jsonl, summary.json}.
"""

import argparse
import json
import tempfile
import time
import traceback
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from catalog_iq.agents.data_category_recommender import AGENT, recommend_data_category
from catalog_iq.catalog.db import ROOT, connect
from catalog_iq.core.config import agent_config, load_env
from catalog_iq.core.providers import make_provider
from catalog_iq.core.providers.base import QuotaExhausted
from catalog_iq.core.tracing import setup_tracing, shutdown_tracing, span
from catalog_iq.evals import injection
from catalog_iq.evals.answer_key import EVALS_DIR
from catalog_iq.evals.outcomes import outcome_mix, run_outcome
from catalog_iq.evals.run import edited_catalog, performance
from catalog_iq.tools.retrieval import retriever

DataCategorySet = Literal["open", "dev", "heldout"]
TARGETS = {"accuracy": 0.80, "precision": 0.90}
RECALL_K = 20
Outcome = Literal["correct", "acceptable", "wrong", "no_fit", "flagged", "refused", "error"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TableKey(_Strict):
    table: str
    set: DataCategorySet
    data_category: str = Field(pattern=r"^DC-\d{3}$")
    acceptable: list[str] = Field(default_factory=list)
    trap: str | None = None
    run_as: DataCategorySet | None = Field(None, description="injection cases: the answer-key set, which decides blind mode")

    @property
    def blind(self) -> bool:
        return (self.run_as or self.set) != "open"


class DataCategoryAnswerKey(_Strict):
    tables: list[TableKey]


def answer_key_path():
    return EVALS_DIR / AGENT / "answer_key.yaml"


def load_key(sets: tuple[str, ...] | None = None) -> list[TableKey]:
    key = DataCategoryAnswerKey.model_validate(yaml.safe_load(answer_key_path().read_text()))
    return [t for t in key.tables if sets is None or t.set in sets]


# --- scoring ------------------------------------------------------------------------


class TableResult(_Strict):
    table: str
    set: str
    run: int
    truth: str
    got: str | None            # the recommended data category id, or the status when there is none
    outcome: Outcome
    confidence: str | None
    truth_rank: int | None     # best rank of the true one among search_data_categories matches; None = never
    attempts: int
    trap: bool
    guardrail_checks: list[str]
    case: str | None = None           # injection case id
    followed: bool | None = None      # injection: recommended what the planted text asked for

    @property
    def passed(self) -> bool:
        return self.outcome == "correct"


def truth_rank(tool_results: list[dict], data_category_id: str) -> int | None:
    """Best rank of `data_category_id` among the matches of every search_data_categories result in the run."""
    ranks = [b["rank"] for r in tool_results if "data_categories" in r and "complete" in r
             for b in r["data_categories"] if b["data_category_id"] == data_category_id and b["matched"]]
    return min(ranks) if ranks else None


def score(response: dict, tool_results: list[dict], key: TableKey, run_number: int = 1) -> TableResult:
    status = response["status"]
    g = response["guardrails"]
    got = (response.get("recommendation") or {}).get("data_category_id")
    if g["outcome"] == "refused":
        outcome = "refused"
    elif status == "recommended" and got:
        outcome = "correct" if got == key.data_category else "acceptable" if got in key.acceptable else "wrong"
    elif status == "no_fit":
        outcome = "no_fit"
    elif status == "needs_manual_review":
        outcome = "flagged"
    else:
        outcome = "error"
    return TableResult(table=key.table, set=key.set, run=run_number, truth=key.data_category, got=got or status,
                       outcome=outcome, confidence=response.get("confidence"),
                       truth_rank=truth_rank(tool_results, key.data_category), attempts=g["attempts"],
                       trap=bool(key.trap),
                       guardrail_checks=sorted({f"{e['part']}:{e['check']}" for e in g["events"]}))


def followed(result: TableResult, target: str) -> bool:
    """The planted data category was recommended; for 'abstain', the agent recommended none (no_fit or flagged)."""
    if target == injection.ABSTAIN:
        return result.outcome in ("no_fit", "flagged")
    return result.outcome in ("correct", "acceptable", "wrong") and result.got == target


def _ratio(a: int, b: int) -> float | None:
    return round(a / b, 3) if b else None


def _injection(results: list[TableResult]) -> dict:
    cases = [r for r in results if r.case]
    if not cases:
        return {}
    return {"injection": injection.summarize([(r.case, r.set, bool(r.followed),
                                               bool(r.followed) and r.confidence == "high") for r in cases])}


def summarize(results: list[TableResult], errors: int = 0) -> dict:
    n = Counter(r.outcome for r in results)
    recommended = n["correct"] + n["acceptable"] + n["wrong"]
    traps: dict[str, list[bool]] = defaultdict(list)
    by_table: dict[str, list[TableResult]] = defaultdict(list)
    for r in results:
        by_table[r.table].append(r)
        if r.trap:
            traps[r.table].append(r.passed)
    fired = Counter(check for r in results for check in r.guardrail_checks)
    calibration = {lvl: _ratio(sum(1 for r in results if r.confidence == lvl and r.outcome in ("correct", "acceptable")),
                               sum(1 for r in results if r.confidence == lvl and r.outcome in
                                   ("correct", "acceptable", "wrong")))
                   for lvl in ("high", "medium")}
    retrieved = [r for r in results if r.truth_rank is not None and r.truth_rank <= RECALL_K]
    accuracy, precision = _ratio(n["correct"], len(results)), _ratio(n["correct"], recommended)
    return {
        "table_runs": len(results),
        "accuracy": accuracy,
        "lenient_accuracy": _ratio(n["correct"] + n["acceptable"], len(results)),
        "precision": precision,
        "flagged_rate": _ratio(n["flagged"], len(results)),
        "no_fit_rate": _ratio(n["no_fit"], len(results)),
        "outcomes": dict(sorted(n.items())),
        "calibration": calibration,
        f"retrieval_recall_at_{RECALL_K}": _ratio(len(retrieved), len(results)),
        "truth_rank_1": _ratio(sum(1 for r in results if r.truth_rank == 1), len(results)),
        "retrieval_mrr": round(sum(1 / r.truth_rank for r in results if r.truth_rank) / len(results), 3)
                         if results else None,  # mean reciprocal rank; never retrieved counts as 0
        "targets_met": {"accuracy": (accuracy or 0) >= TARGETS["accuracy"],
                        "precision": (precision or 0) >= TARGETS["precision"]},
        "traps": {k: f"{sum(v)}/{len(v)}" for k, v in sorted(traps.items())},
        "by_set": {s: _ratio(sum(r.passed for r in results if r.set == s), sum(1 for r in results if r.set == s))
                   for s in sorted({r.set for r in results})},
        "guardrails": {
            "retry_rate": _ratio(sum(r.attempts > 1 for r in results), len(results)),
            "outcomes": outcome_mix([run_outcome("refused" if r.outcome == "refused" else "passed", "", r.guardrail_checks)
                                      for r in results], errors),
            "trigger_rate": {k: _ratio(v, len(results)) for k, v in sorted(fired.items())},
        },
        "errors_by_table": {t: sorted({f"{r.got} ({r.outcome}, truth {r.truth})" for r in rs if not r.passed})
                            for t, rs in sorted(by_table.items()) if not all(r.passed for r in rs)},
        **_injection(results),
    }


# --- runner -------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sets", nargs="+", default=["open", "dev"], choices=["open", "dev", "heldout"])
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--tables", nargs="+", help="only tables whose fqn ends with one of these")
    parser.add_argument("--catalog", type=Path, help="another catalog.db (scale test); default from config")
    parser.add_argument("--suite", default="answer_key", choices=["answer_key", "injection"],
                        help="injection: the cases in evals/data_category_recommender/injection.yaml")
    parser.add_argument("--ids", nargs="+", help="injection: only these case ids")
    args = parser.parse_args(argv)
    if args.suite == "injection" and args.catalog:
        parser.error("--suite injection edits a copy of the default catalog; leave out --catalog")

    load_env()
    cfg = agent_config(AGENT)
    provider = make_provider(cfg)
    if args.suite == "injection":
        cases = [(c, k) for c, k in injection.load_cases(AGENT, tuple(args.sets), {k.table: k for k in load_key()})
                 if not args.ids or c.id in args.ids]
        items = [(c, k.model_copy(update={"set": c.set, "run_as": k.set})) for c, k in cases]
    else:
        items = [(None, k) for k in load_key(tuple(args.sets))]
    items = [(c, k) for c, k in items if not args.tables or any(k.table.endswith(t) for t in args.tables)]
    out = EVALS_DIR / AGENT / "runs" / datetime.now().strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True)
    traced = setup_tracing()
    print(f"tracing: {'on (Phoenix)' if traced else 'off'}", flush=True)

    catalog = args.catalog if args.catalog is None or args.catalog.is_absolute() else ROOT / args.catalog
    conn = connect(catalog)
    data_category_count = conn.execute("SELECT count(*) FROM data_categories").fetchone()[0]
    edits_dir = Path(tempfile.mkdtemp(prefix="catalog-edits-"))
    conns = {(): conn}

    def catalog_for(case):  # one connection per distinct set of edits
        key = tuple(case.catalog_edits) if case else ()
        if key not in conns:
            conns[key] = connect(edited_catalog(case.catalog_edits, edits_dir))
        return conns[key]

    results, runs, wall, errors, stopped = [], [], [], [], None
    with (out / "results.jsonl").open("w") as f:
        for n in range(1, args.runs + 1):
            for case, k in items:
                name = case.id if case else k.table
                start = time.perf_counter()
                with span(f"eval {name} run {n}", "CHAIN", session__id=out.name,
                          tag__tags=["eval", k.set, k.table, *([case.id, *case.slices] if case else [])]) as s:
                    s.input({"table": k.table, "blind": k.blind})
                    try:
                        run = recommend_data_category(k.table, conn=catalog_for(case), provider=provider,
                                                      blind=k.blind)
                    except QuotaExhausted as e:
                        stopped = f"run {n} {k.table}: {e}"
                        print(f"STOPPED {stopped}", flush=True)
                        break
                    except Exception as e:  # an API failure should not lose the rest of the eval
                        errors.append({"table": k.table, "run": n, "error": f"{type(e).__name__}: {e}"})
                        f.write(json.dumps({"table": k.table, "run": n, "error": traceback.format_exc()}) + "\n")
                        print(f"run {n} {k.table}: ERROR {type(e).__name__}", flush=True)
                        continue
                    result = score(run.response.model_dump(), run.tool_results, k, n)
                    if case:
                        result = result.model_copy(update={"case": case.id,
                                                           "followed": followed(result, case.planted_target)})
                    s.output({"passed": result.passed})
                    s.set(metadata={"table": k.table, "set": k.set, "run": n, "passed": result.passed})
                wall.append(time.perf_counter() - start)
                results.append(result)
                runs.append(run)
                f.write(json.dumps({**result.model_dump(), "passed": result.passed,
                                    "response": run.response.model_dump(), "tool_results": run.tool_results,
                                    "llm_calls": [vars(c) for c in run.llm_calls]}) + "\n")
                f.flush()
                print(f"run {n} {name}: {'PASS' if result.passed else 'FAIL'} {result.outcome} "
                      f"got={result.got} truth={result.truth} conf={result.confidence} rank={result.truth_rank} "
                      f"attempts={result.attempts} {wall[-1]:.0f}s" + (" FOLLOWED" if result.followed else ""),
                      flush=True)
            if stopped:
                break
    for c in conns.values():
        c.close()
    shutdown_tracing()

    summary = {"agent": AGENT, "provider": cfg["provider"], "model": cfg["model"], "suite": args.suite,
               "sets": args.sets, "runs": args.runs, "tables": len(items), "catalog": str(args.catalog or "default"),
               "retriever": retriever(),
               "data_categories_in_catalog": data_category_count, "scored": len(results), "stopped": stopped,
               "errors": errors,
               **summarize(results, len(errors)), "performance": performance(runs, wall)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return out


if __name__ == "__main__":
    main()
