"""Per-client rate limit: token bucket (no network)."""

from catalog_iq.ratelimit import RateLimiter
from test_budget import Clock


def test_burst_then_steady_rate():
    clock = Clock()
    rl = RateLimiter(per_minute=10, burst=5, clock=clock)
    assert [rl.take("a") for _ in range(5)] == [0.0] * 5
    assert rl.take("a") == 6.0  # one token every 6 s at 10 a minute
    clock.now = 3
    assert rl.take("a") == 3.0
    clock.now = 6
    assert rl.take("a") == 0.0 and rl.take("a") > 0


def test_refills_only_up_to_the_burst():
    clock = Clock()
    rl = RateLimiter(per_minute=10, burst=5, clock=clock)
    rl.take("a")
    clock.now = 3600
    assert [rl.take("a") for _ in range(6)].count(0.0) == 5


def test_clients_are_counted_separately():
    rl = RateLimiter(per_minute=10, burst=1, clock=Clock())
    assert rl.take("a") == 0.0 and rl.take("a") > 0 and rl.take("b") == 0.0
