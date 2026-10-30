"""/dashboard: which eval runs it shows, and the metrics it computes from their saved rows (no network)."""

import json

import pytest
from fastapi.testclient import TestClient

from catalog_iq.api import create_app
from catalog_iq.catalog.db import catalog_path
from catalog_iq.dashboard import eval_metrics, latest_runs

CALL = {"provider": "fake", "model": "fake-1", "input_tokens": 100, "output_tokens": 11, "ttft_s": 0.5, "total_s": 1.5}


def save(evals, suite, name, sets, size=10, agent="router", rows=(), **extra):
    run = evals / suite / "runs" / name
    run.mkdir(parents=True)
    (run / "summary.json").write_text(json.dumps({"agent": agent, "sets": sets, "requests": size, "scored": 3,
                                                  "stopped": None, "model": "fake-1", "runs": 3, **extra}))
    (run / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_the_newest_full_run_per_set_and_nothing_superseded(tmp_path):
    save(tmp_path, "router", "20260101-000000", ["dev"])
    save(tmp_path, "router", "20260102-000000", ["dev"])                # newer: shown
    save(tmp_path, "router", "20260103-000000", ["dev"], size=1)        # a few --ids: not the full set
    save(tmp_path, "router", "20260104-000000", ["dev"], stopped="quota")
    save(tmp_path, "router", "20260105-000000", ["heldout"])
    save(tmp_path, "router", "20251201-000000", ["open", "dev"])        # older and overlaps dev: superseded
    save(tmp_path, "router", "20260106-000000", ["dev"], agent="old_name")
    assert [r.name for r, _ in latest_runs("router", tmp_path)] == ["20260105-000000", "20260102-000000"]


def test_new_metrics_are_computed_from_saved_rows(tmp_path):
    def row(qid, status, outcome, checks=()):
        return {"id": qid, "set": "dev", "run": 1, "escalation_expected": False, "status": status, "confidence": "high",
                "correctness": {"passed": True, "reasons": []}, "faithfulness": {"passed": True, "reasons": []},
                "claims": 1, "unsupported_claims": 0, "attempts": 1, "outcome": outcome,
                "guardrail_checks": list(checks), "passed": True, "llm_calls": [CALL]}
    rows = [row("D1", "answered", "passed"), row("N1", "insufficient_context", "passed"),
            row("D2", "insufficient_context", "refused", ["output:budget"])]
    save(tmp_path, "catalog_lookup", "20260101-000000", ["dev", "negative"], agent="catalog_lookup", rows=rows,
         questions=3, errors=[])
    [lookup] = eval_metrics(tmp_path)
    assert lookup["outcomes"] == {"answered": 0.333, "abstained": 0.333, "verification_refusal": 0.0,
                                  "budget_stop": 0.333, "error": 0.0}
    quality = {m["label"]: m["value"] for m in lookup["quality"]}
    assert (quality["False refusal rate"], quality["Abstention recall"]) == (0.0, 1.0)  # the budget stop is left out
    reliability = {m["label"]: m["value"] for m in lookup["reliability"]}
    assert (reliability["Timeout rate"], reliability["TTFT (median)"], reliability["TPOT (median)"]) == (0.333, 0.5, 100.0)


def test_no_runs_saved(tmp_path):
    assert eval_metrics(tmp_path) == []


@pytest.mark.skipif(not catalog_path().exists(), reason="catalog.db missing: run python3 scripts/ingest_catalog.py")
def test_the_dashboard_is_served():
    c = TestClient(create_app(providers={}))
    assert c.get("/dashboard").headers["content-type"].startswith("text/html")
    assert isinstance(c.get("/dashboard/evals").json(), list)
