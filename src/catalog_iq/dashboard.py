"""The numbers on /dashboard: live user feedback, and the latest saved run of every eval.

For each eval and each set combination (dev, held-out, scale catalog, ...) the newest
complete run is shown. Its summary is computed again from the saved results.jsonl with
today's metric code, so runs saved before a metric existed (outcome mix, refusal
accuracy, MRR, TTFT, TPOT) still show it. Only the aggregates are recomputed: every
row's verdict is the one the scorer saved at run time.

Eval runs are gitignored, so a fresh clone shows none until the evals are run.
"""

import json
import statistics
from pathlib import Path

from catalog_iq.core.providers.base import LLMCall
from catalog_iq.evals import data_category, pi, scoring
from catalog_iq.evals.answer_key import EVALS_DIR, load_questions

# suite directory → (label, agent its summaries name, result model and summarize; None: use summary.json as is)
EVALS = {
    "catalog_lookup": ("Catalog Lookup", "catalog_lookup", scoring.QuestionResult, scoring.summarize),
    "grounding": ("Grounding", "catalog_lookup", scoring.QuestionResult, scoring.summarize),
    "pi_classifier": ("PI Classifier", "pi_classifier", pi.TableResult, pi.summarize),
    "data_category_recommender": ("Data Category Recommender", "data_category_recommender",
                                  data_category.TableResult, data_category.summarize),
    "router": ("Router", "router", None, None),
}
SIZE_KEYS = ("questions", "tables", "requests")
CAVEATS = {  # what a reader must know that the numbers can't say (the README has the details)
    "grounding": "High but wrong is as scored: read by hand, the 5 quote a planted instruction in order to warn "
                 "against it, so none is wrong (SECURITY.md).",
    "router": "Held-out was 91.7% until a fix for plain-word table names; the fix was tested on new dev requests, "
              "but the held-out set is no longer unseen.",
}


def _size(summary: dict) -> int:
    return next((summary[k] for k in SIZE_KEYS if k in summary), 0)


def _variant(summary: dict) -> tuple:
    """What else, beyond its sets, makes runs comparable: catalog, retriever, and the injection suite."""
    return summary.get("catalog", "default"), summary.get("retriever", "bm25"), summary.get("suite") == "injection"


def latest_runs(suite: str, evals_dir: Path = EVALS_DIR) -> list[tuple[Path, dict]]:
    """Per set combination, catalog, retriever and suite, the newest run that finished and covered the whole set.

    Runs of a few --ids or --slices are left out: they have fewer questions than the full run.
    A set combination is also left out once a newer run on the same catalog covers one of its sets
    (an older dev + open run after a newer dev run, say).
    """
    groups: dict[tuple, list[tuple[Path, dict]]] = {}
    for summary_path in sorted((evals_dir / suite / "runs").glob("*/summary.json")):
        s = json.loads(summary_path.read_text())
        if s.get("agent") != EVALS[suite][1] or s.get("stopped") or not s.get("scored"):
            continue
        groups.setdefault((tuple(s.get("sets") or ()), *_variant(s)), []).append((summary_path.parent, s))
    picked = []
    for runs in groups.values():
        full = max(_size(s) for _, s in runs)
        picked.append(max((r for r in runs if _size(r[1]) == full), key=lambda r: r[0].name))
    key = lambda r: (set(r[1].get("sets") or ()), _variant(r[1]))
    return sorted([r for r in picked if not any(o[0].name > r[0].name and key(o)[1] == key(r)[1] and key(o)[0] & key(r)[0]
                                               for o in picked)], key=lambda r: r[0].name, reverse=True)


def _rows(run_dir: Path) -> list[dict]:
    with open(run_dir / "results.jsonl") as f:
        return [row for line in f if line.strip() and "error" not in (row := json.loads(line))]


def _median(values) -> float | None:
    known = [v for v in values if v is not None]
    return round(statistics.median(known), 3) if known else None


def recomputed(suite: str, run_dir: Path, summary: dict) -> dict:
    """The run's summary with today's metrics, from its saved rows; summary.json alone if that fails."""
    _, _, model, summarize = EVALS[suite]
    try:
        rows = _rows(run_dir)
        if model:
            if model is scoring.QuestionResult:  # saved before rows carried the answer key's status
                expected = {q.id: q.expected.status for q in load_questions(suite)}
                rows = [{"expected_status": expected.get(row["id"], "answered"), **row} for row in rows]
            results = [model.model_validate({k: v for k, v in row.items() if k in model.model_fields}) for row in rows]
            summary = {**summary, **summarize(results, len(summary.get("errors") or []))}
        calls = [LLMCall(**c) for row in rows for c in row.get("llm_calls") or []]
        summary["performance"] = {**summary.get("performance", {}),
                                  "median_ttft_s": _median(c.ttft_s for c in calls),
                                  "median_tpot_ms": _median(c.tpot_s * 1000 if c.tpot_s is not None else None
                                                            for c in calls)}
    except (OSError, ValueError, KeyError, TypeError):
        pass  # an older row format: show what summary.json has
    return summary


def _m(label: str, value, kind: str = "pct", note: str | None = None) -> dict:
    return {"label": label, "value": value, "kind": kind, "note": note}


def quality(suite: str, s: dict) -> list[dict]:
    d = s.get("derived", {})
    r = s.get("refusals", {})
    if inj := s.get("injection"):
        exact = _m("Table pass rate", s.get("table_pass_rate")) if suite == "pi_classifier" \
            else _m("Exact match", s.get("accuracy"))
        return [_m("Injection followed", inj.get("followed_rate"), note="did what the planted text asked; target 0"),
                _m("Followed at high", inj.get("followed_high"), "count", "target 0"), exact,
                _m("Calibration (high right)", (s.get("calibration") or {}).get("high"))]
    if suite in ("catalog_lookup", "grounding"):
        return [_m("Pass rate", s.get("question_pass_rate")), _m("Answer correctness", s.get("answer_correctness")),
                _m("Faithfulness", s.get("faithfulness")),
                _m("False refusal rate", r.get("false_refusal_rate"), note="declined an answerable question"),
                _m("Missed refusal rate", r.get("missed_refusal_rate"), note="answered an unanswerable one"),
                _m("Abstention precision", r.get("abstention_precision"), note="declined questions that had no answer"),
                _m("Abstention recall", r.get("abstention_recall"), note="questions with no answer that were declined"),
                _m("Calibration (high right)", d.get("calibration_high")),
                _m("High but wrong", d.get("high_but_wrong"), "count", "target 0")]
    if suite == "pi_classifier":
        return [_m("Table pass rate", s.get("table_pass_rate")), _m("Precision", s.get("precision")),
                _m("Recall", s.get("recall")), _m("PI type accuracy", s.get("pi_type_accuracy")),
                _m("Calibration (high right)", (s.get("calibration") or {}).get("high"))]
    if suite == "data_category_recommender":
        return [_m("Exact match", s.get("accuracy")), _m("Precision", s.get("precision")),
                _m("Recall@20", s.get("retrieval_recall_at_20"), note="true category among the search results"),
                _m("Top-1", s.get("truth_rank_1")), _m("MRR", s.get("retrieval_mrr"), "num"),
                _m("Calibration (high right)", (s.get("calibration") or {}).get("high"))]
    return [_m("Route accuracy", s.get("route_accuracy")), _m("Table accuracy", s.get("table_accuracy")),
            _m("Clarify precision", s.get("clarify_precision")), _m("Clarify recall", s.get("clarify_recall")),
            _m("Accuracy when all 3 samples agree", (s.get("accuracy_by_confidence") or {}).get("high"))]


def reliability(suite: str, s: dict) -> tuple[dict | None, list[dict]]:
    """The outcome mix (None for the router, which never refuses) and the reliability metrics."""
    block = s.get("reliability") or s.get("guardrails") or {}
    outcomes = s.get("outcomes") if suite in ("catalog_lookup", "grounding") else block.get("outcomes")
    p = s.get("performance", {})
    tokens = (p.get("input_tokens_per_question") or 0) + (p.get("output_tokens_per_question") or 0)
    metrics = []
    if outcomes:
        metrics += [_m("Timeout rate", outcomes.get("budget_stop"), note="stopped by its time, token or turn budget"),
                    _m("Error rate", outcomes.get("error"))]
    if "retry_rate" in block:
        metrics.append(_m("Retry rate", block.get("retry_rate")))
    if "retry_success_rate" in block:
        metrics.append(_m("Retry success rate", block.get("retry_success_rate")))
    metrics += [_m("TTFT (median)", p.get("median_ttft_s"), "s"), _m("TPOT (median)", p.get("median_tpot_ms"), "ms"),
                _m("TPS (mean)", p.get("mean_tps"), "num"), _m("Tokens per question", tokens or None, "count"),
                _m("LLM calls per question", p.get("llm_calls_per_question"), "num"),
                _m("Seconds per question", p.get("wall_s_per_question"), "s")]
    return outcomes, metrics


def eval_metrics(evals_dir: Path = EVALS_DIR) -> list[dict]:
    out = []
    for suite, (label, *_rest) in EVALS.items():
        for run_dir, summary in latest_runs(suite, evals_dir):
            s = recomputed(suite, run_dir, summary)
            outcomes, rel = reliability(suite, s)
            catalog = s.get("catalog", "default")
            out.append({"eval": label, "suite": suite, "run": run_dir.name, "sets": s.get("sets") or [],
                        "catalog": None if catalog == "default" else f"{s.get('data_categories_in_catalog')} data categories",
                        "retriever": s.get("retriever") if suite == "data_category_recommender" else None,
                        "injection": s.get("suite") == "injection",
                        "model": s.get("model"), "size": _size(s), "runs_each": s.get("runs"), "scored": s.get("scored"),
                        "quality": quality(suite, s), "outcomes": outcomes, "reliability": rel,
                        "caveat": CAVEATS.get(suite)})
    return out
