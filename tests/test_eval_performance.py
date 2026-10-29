"""Eval performance summary: tokens, TPS, TTFT and TPOT per question."""

from types import SimpleNamespace

from catalog_iq.core.providers.base import LLMCall
from catalog_iq.evals.run import performance


def call(output_tokens, ttft_s, total_s):
    return LLMCall(provider="fake", model="m", input_tokens=100, output_tokens=output_tokens,
                   ttft_s=ttft_s, total_s=total_s)


def test_performance_reports_ttft_tpot_and_tps():
    runs = [SimpleNamespace(llm_calls=[call(11, 0.5, 1.5), call(21, 1.0, 3.0)]),
            SimpleNamespace(llm_calls=[call(1, None, 0.4)])]  # no TTFT recorded, too short for TPOT
    p = performance(runs, [2.0, 1.0])
    assert p["llm_calls_per_question"] == 1.5
    assert p["mean_tps"] == round(33 / 4.9, 1)
    assert p["median_ttft_s"] == 0.75            # median of 0.5 and 1.0
    assert p["median_tpot_ms"] == 100.0          # (1.5-0.5)/10 and (3.0-1.0)/20 are both 0.1 s


def test_performance_with_no_calls():
    p = performance([], [])
    assert p["mean_tps"] is None and p["median_ttft_s"] is None and p["median_tpot_ms"] is None
