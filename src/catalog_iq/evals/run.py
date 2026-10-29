"""Eval runner: ask every question with the real agent, score, summarise.

    python -m catalog_iq.evals.run                 # dev + negative, 3 runs each
    python -m catalog_iq.evals.run --sets heldout  # only once tuning is finished
    python -m catalog_iq.evals.run --slices lineage  # only questions in a slice
    python -m catalog_iq.evals.run --suite grounding --sets dev heldout
        false premises, planted instructions, counterfactual catalog edits

Writes to evals/<suite>/runs/<timestamp>/ (gitignored):
    results.jsonl   one line per question run (score, response, tool results, LLM calls), written as it goes
    summary.json    metrics from scoring.summarize plus performance

With PHOENIX_COLLECTOR_ENDPOINT set, every question run is also a trace in
Phoenix, tagged with its question id, run and pass/fail; one eval = one session.
"""

import argparse
import json
import shutil
import sqlite3
import tempfile
import time
import traceback
from datetime import datetime
from pathlib import Path

from catalog_iq.agents.catalog_lookup import AGENT, ask
from catalog_iq.catalog.db import catalog_path, connect
from catalog_iq.core.config import agent_config, load_env
from catalog_iq.core.faithfulness import Vocabulary
from catalog_iq.core.loop import AgentRun
from catalog_iq.core.providers import make_provider
from catalog_iq.core.providers.base import QuotaExhausted
from catalog_iq.core.tracing import setup_tracing, shutdown_tracing, span
from catalog_iq.evals.answer_key import EVALS_DIR, load_questions
from catalog_iq.evals.scoring import QuestionResult, score, summarize


def performance(runs: list[AgentRun], wall_s: list[float]) -> dict:
    calls = [c for r in runs for c in r.llm_calls]
    n = len(runs) or 1
    return {
        "llm_calls_per_question": round(len(calls) / n, 2),
        "input_tokens_per_question": round(sum(c.input_tokens for c in calls) / n),
        "cached_input_tokens_per_question": round(sum(c.cached_input_tokens for c in calls) / n),
        "output_tokens_per_question": round(sum(c.output_tokens for c in calls) / n),
        "wall_s_per_question": round(sum(wall_s) / n, 2),
        "llm_s_per_question": round(sum(c.total_s for c in calls) / n, 2),
        "mean_tps": round(sum(c.output_tokens for c in calls) / sum(c.total_s for c in calls), 1) if calls else None,
    }


def edited_catalog(edits: list[str], directory: Path) -> Path:
    """A copy of catalog.db with the edits applied; the agent opens it read-only like the real one."""
    path = directory / f"catalog-{abs(hash(tuple(edits)))}.db"
    if not path.exists():
        shutil.copy(catalog_path(), path)
        with sqlite3.connect(path) as conn:
            for sql in edits:
                if conn.execute(sql).rowcount < 1:
                    raise ValueError(f"catalog edit changed nothing: {sql}")
    return path


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sets", nargs="+", default=["dev", "negative"], choices=["dev", "negative", "heldout"])
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--ids", nargs="+", help="only these question ids")
    parser.add_argument("--slices", nargs="+", help="only questions in any of these slices (e.g. lineage)")
    parser.add_argument("--suite", default=AGENT, choices=[AGENT, "grounding"],
                        help="question set: evals/<suite>/questions.yaml")
    args = parser.parse_args(argv)

    load_env()
    cfg = agent_config(AGENT)
    provider = make_provider(cfg)
    questions = [q for q in load_questions(args.suite, tuple(args.sets))
                 if (not args.ids or q.id in args.ids) and (not args.slices or set(q.slices) & set(args.slices))]
    out = EVALS_DIR / args.suite / "runs" / datetime.now().strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True)
    traced = setup_tracing()
    print(f"tracing: {'on (Phoenix)' if traced else 'off'}", flush=True)

    edits_dir = Path(tempfile.mkdtemp(prefix="catalog-edits-"))
    conns: dict[tuple, tuple] = {}

    def catalog_for(q):  # one connection and vocabulary per distinct set of edits
        key = tuple(q.catalog_edits)
        if key not in conns:
            c = connect(edited_catalog(q.catalog_edits, edits_dir) if key else None)
            conns[key] = (c, Vocabulary(c))
        return conns[key]

    results: list[QuestionResult] = []
    runs: list[AgentRun] = []
    wall: list[float] = []
    errors: list[dict] = []
    stopped = None
    with (out / "results.jsonl").open("w") as f:
        for n in range(1, args.runs + 1):
            for q in questions:
                start = time.perf_counter()
                with span(f"eval {q.id} run {n}", "CHAIN", session__id=out.name,
                          tag__tags=["eval", q.set, q.id, *q.slices]) as s:
                    s.input({"question": q.question})
                    conn, vocab = catalog_for(q)
                    try:
                        run = ask(q.question, conn=conn, provider=provider)
                    except QuotaExhausted as e:  # every later call would fail too
                        stopped = f"run {n} {q.id}: {e}"
                        print(f"STOPPED {stopped}", flush=True)
                        break
                    except Exception as e:  # an API failure should not lose the rest of the eval
                        errors.append({"id": q.id, "run": n, "error": f"{type(e).__name__}: {e}"})
                        f.write(json.dumps({"id": q.id, "run": n, "error": traceback.format_exc()}) + "\n")
                        print(f"run {n} {q.id}: ERROR {type(e).__name__}", flush=True)
                        s.set(metadata={"question_id": q.id, "run": n, "error": type(e).__name__})
                        continue
                    result = score(run, q, vocab, n)
                    s.output({"passed": result.passed, "correctness": result.correctness,
                              "faithfulness": result.faithfulness})
                    s.set(metadata={"question_id": q.id, "set": q.set, "run": n, "passed": result.passed,
                                    "reasons": result.correctness.reasons + result.faithfulness.reasons})
                wall.append(time.perf_counter() - start)
                results.append(result)
                runs.append(run)
                f.write(json.dumps({**result.model_dump(), "passed": result.passed,
                                    "response": run.response.model_dump(), "tool_results": run.tool_results,
                                    "llm_calls": [vars(c) for c in run.llm_calls]}) + "\n")
                f.flush()
                print(f"run {n} {q.id}: {'PASS' if result.passed else 'FAIL'} {result.status} "
                      f"{result.confidence} attempts={result.attempts} {wall[-1]:.0f}s", flush=True)
            if stopped:
                break
    for c, _ in conns.values():
        c.close()
    shutil.rmtree(edits_dir, ignore_errors=True)
    shutdown_tracing()

    summary = {"agent": AGENT, "suite": args.suite, "provider": cfg["provider"], "model": cfg["model"], "sets": args.sets,
               "runs": args.runs, "questions": len(questions), "scored": len(results),
               "stopped": stopped, "errors": errors,
               **summarize(results), "performance": performance(runs, wall)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return out


if __name__ == "__main__":
    main()
