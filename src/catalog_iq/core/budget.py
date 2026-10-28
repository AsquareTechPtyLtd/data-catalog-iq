"""Run budget: a deadline and a token cap for one agent run or one routing.

Checked before every LLM call. While a run is active (`with budget.active():`) the
provider adapters read `time_left()`: each attempt's timeout is capped at the time left
and a retry only waits if the wait fits (providers/retry.py); otherwise BudgetExceeded.

Defaults are about twice the slowest question seen in evals (2026-10-28: 30.5 s,
46k input + 8k output tokens), so they stop runaways, not answers. .env can change them
(AGENT_BUDGET_SECONDS, AGENT_BUDGET_TOKENS, ROUTER_BUDGET_SECONDS, ROUTER_BUDGET_TOKENS);
evals use the same budgets as the app.
The router's samples run in parallel and share one budget.
"""

import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar

from catalog_iq.core.config import number_setting
from catalog_iq.core.providers.base import LLMCall

AGENT_SECONDS, AGENT_TOKENS = 60.0, 100_000
ROUTER_SECONDS, ROUTER_TOKENS = 20.0, 30_000


def agent_budget() -> "Budget":
    return Budget(number_setting("AGENT_BUDGET_SECONDS", AGENT_SECONDS),
                  number_setting("AGENT_BUDGET_TOKENS", AGENT_TOKENS, int))


def router_budget() -> "Budget":
    return Budget(number_setting("ROUTER_BUDGET_SECONDS", ROUTER_SECONDS),
                  number_setting("ROUTER_BUDGET_TOKENS", ROUTER_TOKENS, int))


class BudgetExceeded(RuntimeError):
    """An LLM call (or its retry) does not fit in the run's time left."""


class Budget:
    def __init__(self, seconds: float, tokens: int, clock=time.monotonic):
        self.seconds, self.tokens, self._clock = seconds, tokens, clock
        self._started = clock()
        self._lock = threading.Lock()
        self.used_tokens = 0
        self.stopped_by: str | None = None  # why the run was stopped, if it was

    def spend(self, call: LLMCall) -> None:
        with self._lock:
            self.used_tokens += call.input_tokens + call.output_tokens

    def elapsed_s(self) -> float:
        return self._clock() - self._started

    def exceeded(self) -> str | None:
        """Why no further LLM call may start, or None."""
        if self.elapsed_s() >= self.seconds:
            return f"time budget of {self.seconds:g} s used ({self.elapsed_s():.1f} s)"
        if self.used_tokens >= self.tokens:
            return f"token budget of {self.tokens:,} used ({self.used_tokens:,})"
        return None

    def stop(self, why: str) -> str:
        self.stopped_by = why
        return why

    def left_s(self) -> float:
        return self.seconds - self.elapsed_s()

    @contextmanager
    def active(self):
        token = _active.set(self)
        try:
            yield self
        finally:
            _active.reset(token)


_active: ContextVar[Budget | None] = ContextVar("run_budget", default=None)


def time_left() -> float | None:
    """Seconds left in the active run's budget; None outside a run (no limit)."""
    b = _active.get()
    return None if b is None else b.left_s()
