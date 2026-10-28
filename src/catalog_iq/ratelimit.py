"""Per-client rate limit for the LLM endpoints: a token bucket per key (client IP).

Each key starts with `burst` requests and earns back `per_minute` a minute, up to
`burst`. In memory, per process: behind a proxy or with several processes this
belongs at the gateway (SECURITY.md).
"""

import threading
import time

PER_MINUTE, BURST = 10, 5
MAX_KEYS = 10_000  # past this, buckets that have refilled are dropped


class RateLimiter:
    def __init__(self, per_minute: float = PER_MINUTE, burst: int = BURST, clock=time.monotonic):
        self.rate = per_minute / 60
        self.burst = burst
        self._clock = clock
        self._buckets: dict[str, tuple[float, float]] = {}  # key → (tokens, when counted)
        self._lock = threading.Lock()

    def take(self, key: str) -> float:
        """0 when the request may go ahead (a token is used), else seconds until it may."""
        now = self._clock()
        with self._lock:
            tokens = self._tokens(key, now)
            if tokens >= 1:
                self._buckets[key] = (tokens - 1, now)
                return 0.0
            self._buckets[key] = (tokens, now)
            if len(self._buckets) > MAX_KEYS:
                self._buckets = {k: v for k, v in self._buckets.items() if self._tokens(k, now) < self.burst}
            return (1 - tokens) / self.rate

    def _tokens(self, key: str, now: float) -> float:
        tokens, then = self._buckets.get(key, (self.burst, now))
        return min(self.burst, tokens + (now - then) * self.rate)
