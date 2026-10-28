"""LLM call retries bounded by the run's time left (no network)."""

import anthropic
import httpx
import pytest

from catalog_iq.core.budget import Budget, BudgetExceeded
from catalog_iq.core.providers.anthropic import AnthropicProvider, retry_wait
from catalog_iq.core.providers.retry import TIMEOUT_S, with_retries
from test_budget import Clock


class Flaky(Exception):
    pass


def attempts(*outcomes):
    """attempt(timeout) that raises or returns each outcome in turn and records the timeouts."""
    seen, queue = [], list(outcomes)

    def attempt(timeout):
        seen.append(timeout)
        out = queue.pop(0)
        if isinstance(out, Exception):
            raise out
        return out
    return attempt, seen


def flaky_wait(wait=0.0):
    return lambda e: wait if isinstance(e, Flaky) else None


def test_retries_then_succeeds_outside_a_run():
    attempt, seen = attempts(Flaky(), Flaky(), "reply")
    slept = []
    assert with_retries(attempt, flaky_wait(), (1, 2, 4), sleep=slept.append) == "reply"
    assert seen == [TIMEOUT_S] * 3 and slept == [1, 2]


def test_gives_up_after_the_last_delay():
    attempt, seen = attempts(Flaky(), Flaky())
    with pytest.raises(Flaky):
        with_retries(attempt, flaky_wait(), (1,), sleep=lambda s: None)
    assert len(seen) == 2


def test_other_errors_are_not_retried():
    attempt, seen = attempts(ValueError("bad request"))
    with pytest.raises(ValueError):
        with_retries(attempt, flaky_wait(), (1, 2), sleep=lambda s: None)
    assert len(seen) == 1


def test_server_asked_wait_is_honoured():
    attempt, _ = attempts(Flaky(), "reply")
    slept = []
    with_retries(attempt, flaky_wait(wait=7), (1,), sleep=slept.append)
    assert slept == [7]


def test_timeout_is_capped_at_time_left():
    clock = Clock()
    budget = Budget(seconds=60, tokens=1000, clock=clock)
    clock.now = 50
    attempt, seen = attempts("reply")
    with budget.active():
        with_retries(attempt, flaky_wait(), (1,), sleep=lambda s: None)
    assert seen == [10]


def test_a_wait_that_does_not_fit_is_a_budget_stop():
    clock = Clock()
    budget = Budget(seconds=60, tokens=1000, clock=clock)
    clock.now = 55
    attempt, seen = attempts(Flaky())
    slept = []
    with budget.active(), pytest.raises(BudgetExceeded, match="retry would wait 30 s"):
        with_retries(attempt, flaky_wait(wait=30), (1,), sleep=slept.append)
    assert len(seen) == 1 and slept == []


def test_no_time_left_means_no_attempt():
    clock = Clock()
    budget = Budget(seconds=60, tokens=1000, clock=clock)
    clock.now = 60
    attempt, seen = attempts("reply")
    with budget.active(), pytest.raises(BudgetExceeded):
        with_retries(attempt, flaky_wait(), (1,), sleep=lambda s: None)
    assert seen == []


def status_error(cls, code, headers=None):
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    return cls("error", response=httpx.Response(code, headers=headers or {}, request=request), body=None)


def test_anthropic_retry_policy():
    assert retry_wait(status_error(anthropic.RateLimitError, 429, {"retry-after": "3"})) == 3.0
    assert retry_wait(status_error(anthropic.InternalServerError, 500)) == 0.0
    assert retry_wait(status_error(anthropic.OverloadedError, 529)) == 0.0
    assert retry_wait(anthropic.APITimeoutError(request=httpx.Request("POST", "https://x"))) == 0.0
    assert retry_wait(status_error(anthropic.BadRequestError, 400)) is None
    assert retry_wait(status_error(anthropic.AuthenticationError, 401)) is None
    assert retry_wait(ValueError()) is None


def test_anthropic_sdk_retries_are_off():
    assert AnthropicProvider("claude-haiku-5-5", api_key="test").client.max_retries == 0


def test_agent_refuses_when_a_call_cannot_fit():
    from catalog_iq.catalog.db import catalog_path, connect
    from catalog_iq.core.loop import run_agent
    if not catalog_path().exists():
        pytest.skip("catalog.db missing")

    class OutOfTime:
        name, model = "fake", "fake-1"

        def chat(self, system, messages, tools):
            raise BudgetExceeded("retry would wait 30 s, 2.0 s left")

    conn = connect()
    try:
        run = run_agent("Who owns dim_customers?", conn, OutOfTime(), "system", "catalog_lookup")
    finally:
        conn.close()
    assert run.response.guardrails.outcome == "refused"
    assert [e.detail for e in run.response.guardrails.events if e.check == "budget"] == \
        ["retry would wait 30 s, 2.0 s left"]
