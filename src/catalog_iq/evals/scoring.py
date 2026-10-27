"""Eval scorer: checks agent runs and rolls them up into metrics.

Two metrics, each fed by deterministic checks (no LLM judge):

- Answer correctness (against the answer key): right status, right tables
  (and none of the forbidden ones), expected facts mentioned, expected
  lineage relations stated, escalation present when expected (and only
  then), clarification options offered.
- Faithfulness (against the tool results): the same checks the live output
  guardrail runs, from core.faithfulness.

A question passes when both metrics pass. `summarize` turns the per-question
results into rates, slices and derived metrics, including what the guardrails
did (retries, refusals, checks that fired).
"""

import re
import sqlite3
from collections import Counter, defaultdict
from collections.abc import Callable

from pydantic import BaseModel

from catalog_iq.core.faithfulness import Vocabulary, check_faithfulness
from catalog_iq.core.loop import AgentRun
from catalog_iq.core.response import AgentResponse
from catalog_iq.evals.answer_key import Question


class CheckResult(BaseModel):
    passed: bool
    reasons: list[str]


class QuestionResult(BaseModel):
    id: str
    set: str
    run: int
    slices: list[str] = []
    escalation_expected: bool
    status: str
    confidence: str
    correctness: CheckResult
    faithfulness: CheckResult
    claims: int
    unsupported_claims: int
    attempts: int
    outcome: str
    guardrail_checks: list[str]  # "<part>:<check>" that fired, once each

    @property
    def passed(self) -> bool:
        return self.correctness.passed and self.faithfulness.passed


def _normalise(text: str) -> str:
    text = text.lower().replace("’", "'").replace("n't", " not")
    return re.sub(r"\s+", " ", text)


# --- checks -------------------------------------------------------------------------


def check_correctness(response: AgentResponse, question: Question) -> CheckResult:
    exp = question.expected
    reasons = []
    if response.status != exp.status:
        reasons.append(f"wrong_status: {response.status} (expected {exp.status})")

    got_tables = {t.fqn for t in response.tables}
    reasons += [f"wrong_table: {fqn} missing" for fqn in exp.tables if fqn not in got_tables]
    reasons += [f"unexpected_table: {fqn}" for fqn in exp.forbidden_tables if fqn in got_tables]

    text = _normalise(" ".join([response.answer, *(f.claim for f in response.facts)]))
    for item in exp.mentions:
        options = item if isinstance(item, list) else [item]
        if not any(_normalise(o) in text for o in options):
            reasons.append(f"missing_fact: {' | '.join(options)}")

    stated = {(r.upstream, r.downstream, r.kind) for f in response.facts for r in f.relations}
    reasons += [f"missing_relation: {r.upstream} -> {r.downstream} ({r.kind})" for r in exp.relations
                if (r.upstream, r.downstream, r.kind) not in stated]

    if exp.escalation:
        if response.escalation is None or response.escalation.team != "Data Strategy":
            reasons.append("no_escalation")
        elif "data strategy" not in response.answer.lower():
            reasons.append("escalation_not_in_answer")
    elif response.escalation is not None:
        reasons.append("unexpected_escalation")

    offered = set(response.clarification.options) if response.clarification else set()
    reasons += [f"missing_option: {o}" for o in exp.clarification_options if o not in offered]
    return CheckResult(passed=not reasons, reasons=reasons)


def check_faithfulness_result(response: AgentResponse, tool_results: list[dict],
                              vocab: Vocabulary) -> tuple[CheckResult, int]:
    problems, unsupported = check_faithfulness(response.answer, response.facts, tool_results, vocab,
                                               subject_tables=[t.fqn for t in response.tables],
                                               question=response.question)
    reasons = [f"{check}: {detail}" for check, detail in problems]
    return CheckResult(passed=not reasons, reasons=reasons), unsupported


def score(run: AgentRun, question: Question, vocab: Vocabulary, run_number: int = 1) -> QuestionResult:
    faithfulness, unsupported = check_faithfulness_result(run.response, run.tool_results, vocab)
    g = run.response.guardrails
    return QuestionResult(
        id=question.id, set=question.set, run=run_number, slices=question.slices,
        escalation_expected=question.expected.escalation,
        status=run.response.status, confidence=run.response.confidence,
        correctness=check_correctness(run.response, question), faithfulness=faithfulness,
        claims=len(run.response.facts), unsupported_claims=unsupported,
        attempts=g.attempts, outcome=g.outcome,
        guardrail_checks=sorted({f"{e.part}:{e.check}" for e in g.events}))


def run_eval(agent: Callable[[str], AgentRun], questions: list[Question], conn: sqlite3.Connection,
             runs: int = 3) -> list[QuestionResult]:
    vocab = Vocabulary(conn)
    return [score(agent(q.question), q, vocab, n) for n in range(1, runs + 1) for q in questions]


# --- metrics ------------------------------------------------------------------------


def _rate(items: list, test: Callable) -> float | None:
    return round(sum(map(test, items)) / len(items), 3) if items else None


def summarize(results: list[QuestionResult]) -> dict:
    by_id: dict[str, list[QuestionResult]] = defaultdict(list)
    for r in results:
        by_id[r.id].append(r)
    claims = sum(r.claims for r in results)
    high = [r for r in results if r.confidence == "high"]
    retried = [r for r in results if r.attempts > 1]
    fired = Counter(check for r in results for check in r.guardrail_checks)
    return {
        "question_pass_rate": _rate(results, lambda r: r.passed),
        "answer_correctness": _rate(results, lambda r: r.correctness.passed),
        "faithfulness": _rate(results, lambda r: r.faithfulness.passed),
        "by_set": {s: _rate([r for r in results if r.set == s], lambda r: r.passed)
                   for s in sorted({r.set for r in results})},
        "slices": {
            "escalation": _rate([r for r in results if r.escalation_expected], lambda r: r.passed),
            "negative": _rate([r for r in results if r.set == "negative"], lambda r: r.passed),
            **{s: _rate([r for r in results if s in r.slices], lambda r: r.passed)
               for s in sorted({s for r in results for s in r.slices})},
        },
        "derived": {
            "hallucination_rate": round(sum(r.unsupported_claims for r in results) / claims, 3) if claims else None,
            "agreement_rate": _rate(list(by_id.values()),
                                    lambda rs: len({(r.passed, r.status) for r in rs}) == 1),
            "calibration_high": _rate(high, lambda r: r.correctness.passed),
            "retry_rate": _rate(results, lambda r: r.attempts > 1),
            "retry_success_rate": _rate(retried, lambda r: r.outcome == "passed_after_retry"),
            "refusal_rate": _rate(results, lambda r: r.outcome == "refused"),
            "confidence_distribution": {c: _rate(results, lambda r, c=c: r.confidence == c)
                                        for c in ("high", "medium", "low")},
            "guardrail_trigger_rate": {check: round(n / len(results), 3) for check, n in sorted(fired.items())},
        },
        "failures": {qid: sorted({reason for r in rs for reason in r.correctness.reasons + r.faithfulness.reasons})
                     for qid, rs in sorted(by_id.items()) if not all(r.passed for r in rs)},
    }
