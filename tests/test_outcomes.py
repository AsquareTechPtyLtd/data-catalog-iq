"""Run outcomes and refusal accuracy, shared by every eval."""

from catalog_iq.evals.outcomes import outcome_mix, refusal_accuracy, run_outcome


def test_run_outcome_tells_budget_stops_from_verification_refusals():
    assert run_outcome("passed", "answered", [], {"answered"}) == "answered"
    assert run_outcome("passed_after_retry", "insufficient_context", [], {"answered"}) == "abstained"
    assert run_outcome("refused", "insufficient_context", ["output:no_evidence"]) == "verification_refusal"
    assert run_outcome("refused", "insufficient_context", ["output:no_evidence", "output:budget"]) == "budget_stop"
    assert run_outcome("refused", "insufficient_context", ["output:max_turns"]) == "budget_stop"
    assert run_outcome("passed", "no_fit", []) == "answered"  # no answered set: any accepted run


def test_outcome_mix_adds_up_with_errors():
    mix = outcome_mix(["answered", "answered", "abstained", "budget_stop"], errors=1)
    assert mix == {"answered": 0.4, "abstained": 0.2, "verification_refusal": 0.0, "budget_stop": 0.2, "error": 0.2}
    assert outcome_mix([]) == dict.fromkeys(mix)


def test_refusal_accuracy():
    runs = [(True, "answered"), (True, "answered"), (True, "abstained"),      # one false refusal of three
            (False, "abstained"), (False, "verification_refusal"), (False, "answered"),  # one missed of three
            (True, "budget_stop"), (False, "budget_stop")]                    # left out
    r = refusal_accuracy(runs)
    assert r["false_refusal_rate"] == 0.333 and r["missed_refusal_rate"] == 0.333
    assert r["abstention_precision"] == 0.667   # 2 of 3 declines had no answer
    assert r["abstention_recall"] == 0.667      # 2 of 3 unanswerable questions declined
    assert refusal_accuracy([(True, "answered")])["missed_refusal_rate"] is None
