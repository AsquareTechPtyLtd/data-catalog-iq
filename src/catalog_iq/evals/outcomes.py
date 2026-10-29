"""How each agent run ended, counted the same way in every eval.

Every run ends in exactly one outcome, so the rates add up to 1:

  answered              the guardrails accepted an answer
  abstained             the model chose not to answer (insufficient_context, out_of_scope,
                        needs_clarification); the guardrails accepted that
  verification_refusal  code refused: the draft failed its checks twice
  budget_stop           code refused: the run used its time, tokens or turns
  error                 an exception; the eval records it instead of a result

A budget stop and a verification refusal both end in Guard.refuse(), so they are told
apart by the guardrail event that stopped the run (output:budget, output:max_turns).
Keeping them apart stops a slow provider from looking like a quality problem.

Refusal accuracy compares answering with the answer key: a false refusal declines a
question that has an answer (over-refusal), a missed refusal answers one that has none.
Budget stops and errors are left out of it: they say nothing about whether the model
knew when to answer.
"""

OUTCOMES = ("answered", "abstained", "verification_refusal", "budget_stop", "error")
STOP_CHECKS = {"output:budget", "output:max_turns"}
DECLINED = {"abstained", "verification_refusal"}


def run_outcome(guardrail_outcome: str, status: str, checks: list[str], answered: set[str] | None = None) -> str:
    """One run's outcome. `answered` lists the statuses that count as an answer (None: every accepted one)."""
    if guardrail_outcome == "refused":
        return "budget_stop" if STOP_CHECKS & set(checks) else "verification_refusal"
    return "answered" if answered is None or status in answered else "abstained"


def _ratio(n: int, d: int) -> float | None:
    return round(n / d, 3) if d else None


def outcome_mix(outcomes: list[str], errors: int = 0) -> dict:
    total = len(outcomes) + errors
    return {o: _ratio(errors if o == "error" else outcomes.count(o), total) for o in OUTCOMES}


def refusal_accuracy(runs: list[tuple[bool, str]]) -> dict:
    """`runs` pairs (the answer key has an answer, outcome)."""
    runs = [(answerable, o) for answerable, o in runs if o in DECLINED | {"answered"}]
    answerable = [o for a, o in runs if a]
    unanswerable = [o for a, o in runs if not a]
    declined = [a for a, o in runs if o in DECLINED]
    return {
        "false_refusal_rate": _ratio(sum(o in DECLINED for o in answerable), len(answerable)),
        "missed_refusal_rate": _ratio(sum(o == "answered" for o in unanswerable), len(unanswerable)),
        "abstention_precision": _ratio(sum(not a for a in declined), len(declined)),
        "abstention_recall": _ratio(sum(o in DECLINED for o in unanswerable), len(unanswerable)),
    }
