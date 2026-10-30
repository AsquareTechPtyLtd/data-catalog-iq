"""Retries for one LLM call, bounded by the run's time left (core.budget).

The SDKs' own retries are off so that every attempt and every wait between attempts
is checked against the budget. Outside a run (no active budget) attempts get the full
per-call timeout, as before.
"""

import time

from catalog_iq.core.budget import DISCONNECTED, BudgetExceeded, cancelled, time_left

TIMEOUT_S = 90  # per attempt; without it a stalled call hangs the run


def with_retries(attempt, retry_wait, delays_s, sleep=time.sleep):
    """attempt(timeout_s) → reply. retry_wait(error) → seconds the server asked to wait (0 if it
    didn't say), or None when the error is not worth retrying."""
    for delay in (*delays_s, None):
        if cancelled():
            raise BudgetExceeded(DISCONNECTED)
        left = time_left()
        if left is not None and left <= 0:
            raise BudgetExceeded("no time left for an LLM call")
        try:
            return attempt(TIMEOUT_S if left is None else min(TIMEOUT_S, left))
        except BudgetExceeded:
            raise
        except Exception as e:
            wait = retry_wait(e)
            if wait is None:
                raise
            pause = max(delay or 0, wait)
            left = time_left()
            if left is not None and pause >= left:
                raise BudgetExceeded(f"retry would wait {pause:g} s, {max(left, 0):.1f} s left") from e
            if delay is None:
                raise
            sleep(pause)
